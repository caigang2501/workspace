import numpy as np
import torch.nn as nn
import os,torch
from torch.utils.data import Dataset, DataLoader
from constent import BOARD_SIZE
from utils import *
class StrategyDataset(Dataset):
    def __init__(self,steps=None,value_model=None):
        self.data = steps
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE))
        self.winned = True if len(self.data)%2==1 else False # 奇数先手赢 偶数后手赢 
        self.value_model = value_model

    def __len__(self):
        return (len(self.data)+1)//2

    def __getitem__(self, idx):
        idx *= 2

        if self.winned:
            target = self.data[idx][0]*BOARD_SIZE+self.data[idx][1]
            board = torch.tensor(self.board, dtype=torch.float32)
            self.board[*self.data[idx]] = 1
        else:
            self.board[*self.data[idx]] = 1
            target = self.data[idx+1][0]*BOARD_SIZE+self.data[idx+1][1]
            board = -torch.tensor(self.board, dtype=torch.float32)

        board = oneto3_channel(board)
        if idx+1<len(self.data):
            self.board[*self.data[idx+1]] = -1

        return board,torch.tensor(target, dtype=torch.long)
    
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
            board = torch.tensor(self.board, dtype=torch.float32)
            board = oneto3_channel(board)
            target = np.array([1 if self.winned else 0])
        else:
            self.board[*self.data[idx]] = -1
            board = -torch.tensor(self.board, dtype=torch.float32)
            board = oneto3_channel(board)
            target = np.array([0 if self.winned else 1])
        return board, torch.tensor(target,dtype=torch.float32)
        
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
    steps = load_latest_steps()
    # print(steps[::2])
    dataset = StrategyDataset(steps)
    # dataset = ValueDataset(steps)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    for board,labels in data_loader:
        print(board.shape,labels.shape,labels)
        print(board[4,1],labels[4])
    

