import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from constent import BOARD_SIZE,MODEL_PATH


def board_to_tensor(board_state):
    empty = (torch.tensor(board_state) == 0).float()  # 空位通道
    black = (torch.tensor(board_state) == 1).float()  # 黑棋通道
    white = (torch.tensor(board_state) == -1).float() # 白棋通道
    return torch.stack([empty, black, white], dim=0)  # (3, board_size, board_size)

def test_board_to_tensor(board_state):
    board_tensor = board_to_tensor(board_state)
    batch_board = board_tensor.unsqueeze(0)  # 增加 batch 维度 (1, 3, 9, 9)
    print(batch_board.shape)

class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
def test_MLPClassifier():
    input_size = 5 * 5  
    hidden_size = 128   
    num_classes = 10    

    model = MLPClassifier(input_size, hidden_size, num_classes)


class SimplifiedAlphaGoNet(nn.Module):
    def __init__(self, board_size=15):
        super(SimplifiedAlphaGoNet, self).__init__()
        self.board_size = board_size
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(256 * board_size * board_size, 512)
        self.fc2 = nn.Linear(512, board_size * board_size)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        x = self.softmax(x)
        return x.view(-1, self.board_size, self.board_size)
    

class ValueNetwork(nn.Module):
    def __init__(self, board_size=15):
        super(ValueNetwork, self).__init__()
        self.board_size = board_size

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(256 * board_size * board_size, 512)
        self.fc2 = nn.Linear(512, 1)  # 输出一个值，表示当前局面获胜的概率

        # 激活函数
        self.relu = nn.ReLU()
        self.tanh = nn.Sigmoid()

    def forward(self, x):
        # 输入是 (batch_size, 3, board_size, board_size)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))

        # 展平
        x = x.view(x.size(0), -1)  # (batch_size, 256 * board_size * board_size)

        # 全连接层
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        # 使用 tanh 激活函数，将输出限制在 [-1, 1] 范围
        x = self.tanh(x)

        return x

def init_stategy_model():
    model = SimplifiedAlphaGoNet(BOARD_SIZE)
    torch.save(model.state_dict(), MODEL_PATH+'stategy_15.pth')

def init_value_model():
    model = ValueNetwork(BOARD_SIZE)
    torch.save(model.state_dict(), MODEL_PATH+'value_15.pth')

