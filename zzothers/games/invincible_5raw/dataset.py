import numpy as np
import torch.nn as nn
import os,torch
from torch.utils.data import Dataset, DataLoader
from constent import BOARD_SIZE,DATASET_PATH_TRAIN,VALUE_TARGET_ALLONES
from utils import *


def random_data(batch_size=32, board_size=15):
    board_state = torch.randint(0, 3, (batch_size, 3, board_size, board_size)).float()
    target_label = torch.randint(0, board_size * board_size, (batch_size,))
    
    return board_state, target_label


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
        if VALUE_TARGET_ALLONES:
            target = np.array([1])
        return board, torch.tensor(target,dtype=torch.float32)

def strategy_dataset(path,threechannel=True):
    dataset = []
    labels = []
    for file in os.listdir(path):
        steps_path = os.path.join(path, file)
        steps = np.load(steps_path, allow_pickle=True)
        board = np.zeros((BOARD_SIZE, BOARD_SIZE))
        winned = True if len(steps)%2==1 else False
        for idx in range((len(steps)+1)//2):
            idx *= 2
            if winned:
                target = steps[idx][0]*BOARD_SIZE+steps[idx][1]
                board_state = torch.tensor(board, dtype=torch.float32)
                board[*steps[idx]] = 1
            else:
                board[*steps[idx]] = 1
                target = steps[idx+1][0]*BOARD_SIZE+steps[idx+1][1]
                board_state = -torch.tensor(board, dtype=torch.float32)

            if threechannel:
                board_state = oneto3_channel(board_state)
            if idx+1<len(steps):
                board[*steps[idx+1]] = -1
            dataset.append(board_state)
            labels.append(torch.tensor(target, dtype=torch.long))
    return torch.stack(dataset),torch.stack(labels)

def value_dataset(path,threechannel=True,target2one=True):
    dataset = []
    labels = []
    for file in os.listdir(path):
        steps_path = os.path.join(path, file)
        steps = np.load(steps_path, allow_pickle=True)
        board = np.zeros((BOARD_SIZE, BOARD_SIZE))
        winned = True if len(steps)%2==1 else False
        for idx in range(len(steps)):
            if idx%2==0:
                board[*steps[idx]] = 1
                board_state = torch.tensor(board, dtype=torch.float32)
                target = np.array([1 if winned else 0])
            else:
                board[*steps[idx]] = -1
                board_state = -torch.tensor(board, dtype=torch.float32)
                target = np.array([0 if winned else 1])

            if threechannel:
                board_state = oneto3_channel(board_state)
            if target2one:
                target = np.array([1])
            dataset.append(board_state)
            labels.append(torch.tensor(target, dtype=torch.float32))
    return torch.stack(dataset),torch.stack(labels)



if __name__ == "__main__":
    dataset,labels = value_dataset(DATASET_PATH_TRAIN)
    print(dataset.shape,labels.shape)
    

