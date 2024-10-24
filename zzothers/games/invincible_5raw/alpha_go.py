import torch
import torch.nn as nn
import torch.optim as optim
from model import SimplifiedAlphaGoNet,ValueNetwork


def board_to_tensor(board_state):
    empty = (torch.tensor(board_state) == 0).float()  # 空位通道
    black = (torch.tensor(board_state) == 1).float()  # 黑棋通道
    white = (torch.tensor(board_state) == -1).float() # 白棋通道
    return torch.stack([empty, black, white], dim=0)  # (3, board_size, board_size)

def test_board_to_tensor(board_state):
    board_tensor = board_to_tensor(board_state)
    batch_board = board_tensor.unsqueeze(0)  # 增加 batch 维度 (1, 3, 9, 9)
    print(batch_board.shape)


# class SimplifiedAlphaGoNet(nn.Module):
#     def __init__(self, board_size=9):
#         super(SimplifiedAlphaGoNet, self).__init__()
#         self.board_size = board_size

#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#         self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

#         self.fc1 = nn.Linear(256 * board_size * board_size, 512)
#         self.fc2 = nn.Linear(512, board_size * board_size)

#         self.relu = nn.ReLU()
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, x):
#         x = self.relu(self.conv1(x))
#         x = self.relu(self.conv2(x))
#         x = self.relu(self.conv3(x))
#         x = self.relu(self.conv4(x))

#         x = x.view(x.size(0), -1)  # (batch_size, 256 * board_size * board_size)
        
#         print(x.shape)
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)

#         x = self.softmax(x)
#         return x.view(-1, self.board_size, self.board_size)
def train_stategy(model=None):
    board_size = 15
    if not model:
        model = SimplifiedAlphaGoNet(board_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    batch_size = 10
    dummy_board = torch.randn(batch_size, 3, board_size, board_size)  # 假设输入棋盘状态
    dummy_target = torch.randint(0, board_size * board_size, (batch_size,))  # 假设落子目标
    print(dummy_board.shape,dummy_target.shape,dummy_target)
    print(type(dummy_board[0,0,0,0]),type(dummy_target[0]))
    print(dummy_board[0,0,0,0],dummy_target[0])    

    outputs = model(dummy_board)
    print(outputs.shape)
    print(outputs.view(batch_size, -1).shape)
    loss = criterion(outputs.view(batch_size, -1), dummy_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("Loss:", loss.item())


# class ValueNetwork(nn.Module):
#     def __init__(self, board_size=15):
#         super(ValueNetwork, self).__init__()
#         self.board_size = board_size

#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#         self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

#         self.fc1 = nn.Linear(256 * board_size * board_size, 512)
#         self.fc2 = nn.Linear(512, 1)  # 输出一个值，表示当前局面获胜的概率

#         # 激活函数
#         self.relu = nn.ReLU()
#         self.tanh = nn.Sigmoid()  

#     def forward(self, x):
#         # 输入是 (batch_size, 3, board_size, board_size)
#         x = self.relu(self.conv1(x))
#         x = self.relu(self.conv2(x))
#         x = self.relu(self.conv3(x))
#         x = self.relu(self.conv4(x))

#         # 展平
#         x = x.view(x.size(0), -1)  # (batch_size, 256 * board_size * board_size)

#         # 全连接层
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)

#         # 使用 tanh 激活函数，将输出限制在 [-1, 1] 范围
#         x = self.tanh(x)

#         return x

def train_value():
    board_size = 15
    value_net = ValueNetwork(board_size)
    torch.save(value_net.state_dict(), 'models/value_15.pth')

    batch_size = 12
    dummy_board = torch.randn(batch_size, 3, board_size, board_size)  # 假设输入棋盘状态
    dummy_target_value = torch.rand(batch_size, 1)

    print(dummy_board.shape,dummy_target_value.shape)
    print(type(dummy_board[0,0,0,0]),type(dummy_target_value[0,0]))
    print(dummy_board[0,0,0,0],dummy_target_value[0,0])
    criterion = nn.MSELoss()  # 使用均方误差损失函数
    optimizer = optim.Adam(value_net.parameters(), lr=0.001)

    predicted_value = value_net(dummy_board)
    loss = criterion(predicted_value, dummy_target_value)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("Loss:", loss.item())


if __name__=='__main__':
    # train_value()
    train_stategy()
    pass
