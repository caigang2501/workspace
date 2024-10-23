import numpy as np
import torch.nn as nn
import os,torch
from torch.utils.data import Dataset, DataLoader
from game import GomokuNet,BOARD_SIZE
from utils import *

class StrategyDataset(Dataset):
    def __init__(self,value_model,steps=None):
        self.data = steps
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE))
        self.winned = True if len(self.data)%2==1 else False # 奇数先手赢 偶数后手赢 
        self.value_model = value_model

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx%2==0: 
            self.board[*self.data[idx]] = 1
            target = 1 if self.winned else 0
        else:
            self.board[*self.data[idx]] = -1
            target = 0 if self.winned else 1
        board = torch.tensor(self.board, dtype=torch.float32).unsqueeze(0)
        return board, target
    
class ValueDataset(Dataset):
    def __init__(self,steps=None):
        self.data = steps
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE))
        self.winned = len(self.data)%2 # 1:win 0:lost

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx%2==0:
            self.board[*self.data[idx]] = 1
            target = 1 if self.winned else 0
            board = torch.tensor(self.board, dtype=torch.float32).unsqueeze(0)
        else:
            self.board[*self.data[idx]] = -1
            target = 0 if self.winned else 1
            board = -torch.tensor(self.board, dtype=torch.float32).unsqueeze(0)
        return board, target

def train_model(model, data_loader, epochs=5, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        for boards, players in data_loader:
            boards = boards.float()
            players = players.float()

            # 预测
            predictions = model(boards).view(-1, BOARD_SIZE * BOARD_SIZE)
            loss = criterion(predictions, players)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    dataset = GomokuGameDataset()
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = GomokuNet()
    train_model(model, data_loader)
