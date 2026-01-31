import collections
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import time
import pandas as pd
from tqdm import tqdm


# === 1. 环境与种子设置 ===
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed(42)


# === 2. 网络定义 (Offset Attention) ===
class OffsetAttention(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(OffsetAttention, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.LBR = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.BatchNorm1d(dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        F_sa, _ = self.self_attn(x, x, x, need_weights=False)
        offset = x - F_sa  # 计算偏移量
        batch_size, seq_len, d_model = offset.size()
        offset_reshaped = offset.reshape(batch_size * seq_len, d_model)
        lbr_output = self.LBR(offset_reshaped).reshape(batch_size, seq_len, d_model)
        output = self.dropout(lbr_output) + x
        return self.layer_norm(output)


class AttentionQNet(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=32, nhead=4, num_layers=1, dropout=0.1):
        super(AttentionQNet, self).__init__()
        self.value_proj = nn.Linear(1, d_model)
        self.field_embedding = nn.Embedding(input_dim, d_model)
        self.encoder_layers = nn.ModuleList([
            OffsetAttention(d_model, nhead, d_model * 4, dropout) for _ in range(num_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(d_model * input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        batch_size, num_features = x.size(0), x.size(1)
        positions = torch.arange(num_features, device=x.device).unsqueeze(0).expand(batch_size, -1)
        tokens = self.value_proj(x.unsqueeze(-1)) + self.field_embedding(positions)
        encoded = tokens
        for layer in self.encoder_layers:
            encoded = layer(encoded)
        return self.head(encoded.reshape(batch_size, -1))


# === 3. 核心指标计算函数 (分类型统计) ===
def calculate_metrics_5class(predictions, labels):
    """
    针对 5 个类别分别计算 Acc, FAR, DR, F1
    计算逻辑严格遵循 One-vs-Rest
    """
    preds = np.array(predictions)
    true = np.array(labels)
    class_names = {0: 'Normal', 1: 'DoS', 2: 'Probe', 3: 'R2L', 4: 'U2R'}
    results = {}

    for i in range(5):
        # 对应你图片中的标准公式
        tp = ((preds == i) & (true == i)).sum()
        tn = ((preds != i) & (true != i)).sum()
        fp = ((preds == i) & (true != i)).sum()
        fn = ((preds != i) & (true == i)).sum()

        # Accuracy = (TP + TN) / (TP + TN + FP + FN)
        acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        # Detection Rate (Recall) = TP / (TP + FN)
        dr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        # False Alarm Rate = FP / (FP + TN)
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        # Precision (用于 F1) = TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * (precision * dr) / (precision + dr) if (precision + dr) > 0 else 0.0

        results[class_names[i]] = {'Acc': acc, 'FAR': far, 'DR': dr, 'F1': f1}

    return pd.DataFrame(results).T.round(4)


# === 4. DQN Agent 与 评估 ===
class DDQN:
    def __init__(self, state_dim, action_dim, lr, gamma, epsilon, target_update, device):
        self.q_net = AttentionQNet(state_dim, action_dim).to(device)
        self.target_q_net = AttentionQNet(state_dim, action_dim).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma, self.epsilon, self.target_update = gamma, epsilon, target_update
        self.count, self.device, self.action_dim = 0, device, action_dim
        self.criterion = nn.SmoothL1Loss()

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        return self.q_net(state_tensor).argmax().item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)

        q_values = self.q_net(states).gather(1, actions)
        with torch.no_grad():
            max_action = self.q_net(next_states).argmax(1).view(-1, 1)
            max_next_q = self.target_q_net(next_states).gather(1, max_action)
        q_targets = rewards + self.gamma * max_next_q

        loss = self.criterion(q_values, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1


def evaluate(agent, X_test, y_test, batch_size=512):
    agent.q_net.eval()
    all_preds = []
    X_tensor = torch.FloatTensor(X_test)
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch = X_tensor[i:i + batch_size].to(agent.device)
            all_preds.extend(agent.q_net(batch).argmax(1).cpu().numpy())

    df_metrics = calculate_metrics_5class(all_preds, y_test)
    global_acc = (np.array(all_preds) == np.array(y_test)).sum() / len(y_test)
    agent.q_net.train()
    return global_acc, df_metrics


# === 5. 主训练流程 ===
if __name__ == '__main__':
    # 加载数据（确保标签是 0-4 的原始 5 分类）
    data = np.load('/root/yzh/ALL-RIGHT/nsl_kdd_preprocessed_5class_smotetomek-5W.npz')
    X_train, y_train = data['X_train'], data['y_train'].astype(int)
    X_test, y_test = data['X_test'], data['y_test'].astype(int)

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    agent = DDQN(X_train.shape[1], 5, 5e-4, 0.01, 0.5, 100, device)

    buffer = collections.deque(maxlen=10000)
    batch_size = 128

    print("开始 5 分类详细训练统计...")
    for ep in range(50):
        start_time = time.time()
        train_preds, train_labels = [], []

        indices = np.arange(len(X_train))
        np.random.shuffle(indices)

        for i in indices:
            state = X_train[i]
            action = agent.take_action(state)

            train_preds.append(action)
            train_labels.append(y_train[i])

            reward = 1 if action == y_train[i] else 0

            next_state = X_train[(i + 1) % len(X_train)]
            buffer.append((state, action, reward, next_state))

            if len(buffer) > 500:
                transitions = random.sample(buffer, batch_size)
                s, a, r, ns = zip(*transitions)
                agent.update({'states': np.array(s), 'actions': a, 'rewards': r, 'next_states': np.array(ns)})

        # === 训练集详细指标 ===
        train_global_acc = (np.array(train_preds) == np.array(train_labels)).sum() / len(train_labels)
        df_train = calculate_metrics_5class(train_preds, train_labels)

        agent.epsilon = max(0.001, agent.epsilon * 0.75)

        print(f"\nEpisode {ep + 1} | 训练集全局 Acc: {train_global_acc:.4f} | Time: {time.time() - start_time:.1f}s")
        print("【训练集各类型详细指标】")
        print(df_train)

        # === 测试集详细指标 ===
        if ep % 2 == 0:
            test_acc, df_test = evaluate(agent, X_test, y_test)
            print(f"\n【测试集评估】 全局 Acc: {test_acc:.4f}")
            print(df_test)
            print("-" * 60)