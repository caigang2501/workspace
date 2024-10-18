import numpy as np
import torch.nn as nn
import os,torch
from torch.utils.data import Dataset, DataLoader
from game import GomokuNet,BOARD_SIZE

# 加载对局数据集
class GomokuGameDataset(Dataset):
    def __init__(self, data_folder="games_data"):
        self.data = []
        for file in os.listdir(data_folder):
            file_path = os.path.join(data_folder, file)
            game_data = np.load(file_path, allow_pickle=True)
            self.data.extend(game_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        board, player = self.data[idx]
        board = torch.tensor(board, dtype=torch.float32).unsqueeze(0)
        return board, player

# 训练函数
def train_model(model, data_loader, epochs=5, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        for boards, players in data_loader:
            boards = boards.float()
            players = players.float()

            # 预测
            predictions = model(boards).view(-1, BOARD_SIZE * BOARD_SIZE)

            # 计算损失
            loss = criterion(predictions, players)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# 测试训练代码
if __name__ == "__main__":
    dataset = GomokuGameDataset()
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = GomokuNet()
    train_model(model, data_loader)
