import collections
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm


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


class Qnet(nn.Module):
    def __init__(self, input_dim, output_dim=2):
        super(Qnet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim),
        )

    def forward(self, x):
        return self.fc(x)


# 定义 Offset Attention 层（基于PCT论文原理）
class OffsetAttention(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(OffsetAttention, self).__init__()
        # 传统自注意力模块
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # LBR模块（Linear-BatchNorm-ReLU）
        self.LBR = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.BatchNorm1d(dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )

        # 残差连接的dropout
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, key_padding_mask=None):
        """
        Offset Attention 前向传播
        核心公式：F_out = LBR(F_in - F_sa) + F_in
        Args:
            x: 输入特征 (batch_size, seq_len, d_model)
            key_padding_mask:  padding掩码 (batch_size, seq_len)
        Returns:
            融合偏移信息后的特征 (batch_size, seq_len, d_model)
        """
        # 1. 传统自注意力输出 F_sa
        F_sa, _ = self.self_attn(
            x, x, x,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )

        # 2. 计算偏移量：F_in - F_sa
        offset = x - F_sa

        # 3. LBR处理偏移量
        batch_size, seq_len, d_model = offset.size()
        offset_reshaped = offset.reshape(batch_size * seq_len, d_model)  # (batch*seq, d_model)
        lbr_output = self.LBR(offset_reshaped)
        lbr_output = lbr_output.reshape(batch_size, seq_len, d_model)  # (batch, seq, d_model)

        # 4. 残差连接：F_out = LBR(offset) + F_in
        output = self.dropout(lbr_output) + x

        # 5. 层归一化
        output = self.layer_norm(output)

        return output


# 基于Offset Attention的Q网络
class AttentionQNet(nn.Module):
    def __init__(self, input_dim, output_dim=2, d_model=32, nhead=4, num_layers=1, dropout=0.1):
        super(AttentionQNet, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead

        # 特征投影：将每个特征维度映射到d_model
        self.value_proj = nn.Linear(1, d_model)

        # 字段嵌入：为每个特征位置添加位置信息
        self.field_embedding = nn.Embedding(input_dim, d_model)

        # Offset Attention编码器（堆叠多层）
        self.encoder_layers = nn.ModuleList([
            OffsetAttention(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        # 最终输出头
        self.head = nn.Sequential(
            nn.Linear(d_model * input_dim, 64),  # 拼接所有特征的输出
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        """
        Args:
            x: 输入状态 (batch_size, input_dim)
        Returns:
            Q值 (batch_size, output_dim)
        """
        batch_size, num_features = x.size(0), x.size(1)

        # 1. 生成位置索引
        positions = torch.arange(num_features, device=x.device).unsqueeze(0).expand(batch_size, -1)  # (batch, seq_len)

        # 2. 特征投影 + 位置嵌入
        tokens = x.unsqueeze(-1)  # (batch, seq_len, 1)
        tokens = self.value_proj(tokens)  # (batch, seq_len, d_model)
        tokens = tokens + self.field_embedding(positions)  # (batch, seq_len, d_model)

        # 3. 经过多层Offset Attention
        encoded = tokens
        for layer in self.encoder_layers:
            encoded = layer(encoded)  # (batch, seq_len, d_model)

        # 4. 特征拼接（将序列维度展平）
        encoded_flat = encoded.reshape(batch_size, -1)  # (batch, seq_len * d_model)

        # 5. 输出Q值
        return self.head(encoded_flat)


class DQN:
    def __init__(self, state_dim, action_dim, learning_rate, gamma, epsilon, target_update, device, use_attention):
        if use_attention:
            self.q_net = AttentionQNet(state_dim, action_dim).to(device)
            self.target_q_net = AttentionQNet(state_dim, action_dim).to(device)
        else:
            self.q_net = Qnet(state_dim, action_dim).to(device)
            self.target_q_net = Qnet(state_dim, action_dim).to(device)
        self.action_dim = action_dim
        self.q_net.train(True)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.device = device
        self.criterion = nn.SmoothL1Loss()

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)

        q_values = self.q_net(states).gather(1, actions)

        with torch.no_grad():
            max_action = self.q_net(next_states).max(1)[1].view(-1, 1)
            max_next_q_values = self.target_q_net(next_states).gather(1, max_action)

        q_targets = rewards + self.gamma * max_next_q_values
        dqn_loss = self.criterion(q_values, q_targets)

        self.optimizer.zero_grad()
        dqn_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state)

    def size(self):
        return len(self.buffer)

def calculate_metrics(predictions, labels):
    """
    计算准确率、误报率、检测率和F1分数
    """
    # === 关键修改：必须先将 list 转换为 numpy array ===
    predictions = np.array(predictions)
    labels = np.array(labels)
    # ===============================================

    # TP: 预测为1，真实为1
    TP = ((predictions == 1) & (labels == 1)).sum()
    # TN: 预测为0，真实为0
    TN = ((predictions == 0) & (labels == 0)).sum()
    # FP: 预测为1，真实为0 (误报)
    FP = ((predictions == 1) & (labels == 0)).sum()
    # FN: 预测为0，真实为1 (漏报)
    FN = ((predictions == 0) & (labels == 1)).sum()

    # 1. 准确率 (Accuracy)
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0

    # 2. 误报率 (False Alarm Rate)
    if (FP + TN) > 0:
        false_alarm_rate = FP / (FP + TN)
    else:
        false_alarm_rate = 0.0

    # 3. 检测率 (Detection Rate) / 召回率 (Recall)
    if (TP + FN) > 0:
        detection_rate = TP / (TP + FN)
    else:
        detection_rate = 0.0

    # 4. 精确率 (Precision) - 仅用于计算F1
    if (TP + FP) > 0:
        precision = TP / (TP + FP)
    else:
        precision = 0.0

    # 5. F1-Score
    if (precision + detection_rate) > 0:
        f1_score = 2 * (precision * detection_rate) / (precision + detection_rate)
    else:
        f1_score = 0.0

    return accuracy, false_alarm_rate, detection_rate, f1_score

def evaluate(agent, X_test, y_test, batch_size=256):
    """
    批量评估测试集，大幅提升速度
    """
    agent.q_net.eval()
    predictions = []

    # 将数据转换为 Tensor
    X_test_tensor = torch.FloatTensor(X_test)

    with torch.no_grad():
        # 分批次处理
        for i in range(0, len(X_test), batch_size):
            # 获取当前批次
            batch_states = X_test_tensor[i: i + batch_size].to(agent.device)

            # 模型预测
            outputs = agent.q_net(batch_states)
            batch_actions = outputs.argmax(dim=1).cpu().numpy()

            predictions.extend(batch_actions)

    predictions = np.array(predictions)

    # 使用之前修改好的计算 4 个指标的函数
    acc, far, dr, f1 = calculate_metrics(predictions, y_test)

    print(f"Test Set Evaluation: "
          f"Acc: {acc:.4f} | FAR: {far:.4f} | DR: {dr:.4f} | F1: {f1:.4f}")

    agent.q_net.train()
    return acc, far, dr, f1

if __name__ == '__main__':
    data = np.load('/root/yzh/ALL-RIGHT/nsl_kdd_preprocessed_5class_smotetomek-5W.npz')
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    # === 新增：将 5 分类转换为 2 分类 ===
    # 假设 0 是 normal，1,2,3,4 都是攻击
    # 将所有大于 0 的标签都改为 1
    y_train = (y_train > 0).astype(int)
    y_test = (y_test > 0).astype(int)
    print("加载数据成功")
    print(f"训练集样本数：{len(X_train)}, 测试集样本数：{len(X_test)}")
    print(f"训练集正常样本数：{(y_train == 0).sum()}, 训练集异常样本数：{(y_train == 1).sum()}")
    print(f"测试集正常样本数：{(y_test == 0).sum()}, 测试集异常样本数：{(y_test == 1).sum()}\n")

    lr = 5e-4
    num_episodes = 50
    gamma = 0.01
    epsilon_start = 0.5
    epsilon_end = 0.001
    epsilon_decay = 0.75
    target_update = 100
    buffer_size = 10000
    minimal_size = 500
    batch_size = 128
    device = torch.device("cuda:2") if torch.cuda.is_available() else torch.device("cpu")

    replay_buffer = ReplayBuffer(buffer_size)
    state_dim = X_train.shape[1]
    action_dim = 2

    # 使用修改后的Offset Attention Q网络
    agent = DQN(state_dim, action_dim, lr, gamma, epsilon_start, target_update, device, use_attention=True)
    # ... (前面的代码保持不变) ...

    # 修改列表定义，用于存储新指标以便画图（如果需要）
    return_list = []  # 准确率
    fa_return_list = []  # 误报率
    dr_return_list = []  # 检测率 (新增)
    f1_return_list = []  # F1 (新增)

    print("start Training")
    for episode in range(num_episodes):
        start_time = time.time()
        total_reward = 0

        # === 关键修正：确保训练预测值和标签对齐 ===
        train_predictions = []
        train_labels_record = []  # 专门用于记录乱序后的真实标签

        indices = np.arange(len(X_train))
        np.random.shuffle(indices)

        for i in indices:
            state = X_train[i]
            action = agent.take_action(state)

            # 记录预测值和对应的真实标签
            train_predictions.append(action)
            train_labels_record.append(y_train[i])

            reward = 1 if action == y_train[i] else 0
            total_reward += reward

            next_i = (i + 1) % len(X_train)
            next_state = X_train[next_i]
            replay_buffer.add(state, action, reward, next_state)

            if replay_buffer.size() > minimal_size:
                b_s, b_a, b_r, b_ns = replay_buffer.sample(batch_size)
                transition_dict = {
                    'states': b_s,
                    'actions': b_a,
                    'next_states': b_ns,
                    'rewards': b_r
                }
                agent.update(transition_dict)

        # 计算训练集指标 (使用记录好的 train_labels_record)
        train_acc, train_fa, train_dr, train_f1 = calculate_metrics(train_predictions, train_labels_record)

        # 更新epsilon
        agent.epsilon = max(epsilon_end, agent.epsilon * epsilon_decay)

        # 打印训练信息
        print(f"Episode {episode + 1}/{num_episodes} | "
              f"Acc: {train_acc:.4f} | "
              f"FAR: {train_fa:.4f} | "
              f"DR: {train_dr:.4f} | "
              f"F1: {train_f1:.4f} | "
              f"Reward: {total_reward:.0f} | "
              f"Epsi: {agent.epsilon:.3f} | "
              f"Time: {time.time() - start_time:.1f}s")

        # 每2个episode评估测试集
        if episode % 2 == 0:
            test_acc, test_fa, test_dr, test_f1 = evaluate(agent, X_test, y_test)
            return_list.append(test_acc)
            fa_return_list.append(test_fa)
            dr_return_list.append(test_dr)
            f1_return_list.append(test_f1)
            print("-" * 80)
            #注，此版本仅改动buffer-size为10000