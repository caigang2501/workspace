import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 五子棋棋盘大小为 15x15
BOARD_SIZE = 15

# 定义五子棋神经网络
class GomokuNet(nn.Module):
    def __init__(self):
        super(GomokuNet, self).__init__()
        # 卷积层处理棋盘数据
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)  # 输入 1 通道，输出 64 个特征图
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256 * BOARD_SIZE * BOARD_SIZE, 512)  # 全连接层
        self.fc2 = nn.Linear(512, BOARD_SIZE * BOARD_SIZE)  # 输出为棋盘大小的概率分布

    def forward(self, x):
        # 卷积层 + ReLU 激活函数
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # 展平处理后输入全连接层
        x = x.view(-1, 256 * BOARD_SIZE * BOARD_SIZE)
        x = F.relu(self.fc1(x))
        
        # 最终输出预测的概率分布，表示每个格子的概率
        x = self.fc2(x)
        x = x.view(-1, BOARD_SIZE, BOARD_SIZE)  # 输出重塑为 15x15 棋盘
        return x


import numpy as np
from torch.utils.data import Dataset, DataLoader

# 创建一个简单的五子棋数据集
class GomokuDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.board_size = BOARD_SIZE

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # 随机生成棋盘状态 (1, 15, 15) 和目标位置 (15, 15)
        board = np.random.randint(0, 3, (1, self.board_size, self.board_size)).astype(np.float32)
        label = np.random.randint(0, 2, (self.board_size, self.board_size)).astype(np.float32)
        return board, label

# 初始化数据集和数据加载器
dataset = GomokuDataset(1000)  # 创建一个1000条数据的假数据集
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)


# 初始化模型、损失函数和优化器
model = GomokuNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# 训练过程
def train(model, data_loader, optimizer, criterion, epochs=10):
    model.train()  # 设置模型为训练模式
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (boards, labels) in enumerate(data_loader):
            boards, labels = torch.tensor(boards), torch.tensor(labels)
            
            # 前向传播
            outputs = model(boards)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()  # 清除梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            
            # 打印损失
            running_loss += loss.item()
            if (i+1) % 10 == 0:  # 每10个 batch 打印一次损失
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(data_loader)}], Loss: {running_loss/10:.4f}')
                running_loss = 0.0

# 训练模型
train(model, data_loader, optimizer, criterion, epochs=10)
