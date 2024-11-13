import torch
from torch import nn
from torchvision.models import resnet18,mobilenet_v2,efficientnet_b0
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset

from dataset import *
from constent import BOARD_SIZE,STRATEGY_MODEL_NAME,VALUE_MODEL_NAME





class StrategyLinear(nn.Module):
    def __init__(self,board_size=BOARD_SIZE):
        super(StrategyLinear, self).__init__()
        self.fc1 = nn.Linear(board_size*board_size, 200)
        self.actf1 = nn.Sigmoid()
        self.fc2 = nn.Linear(200, 200)
        self.actf2 = nn.Sigmoid()
        self.fc3 = nn.Linear(200, 200)
        self.actf3 = nn.Sigmoid()
        self.fc4 = nn.Linear(200, board_size*board_size)
        self.actf4 = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.actf1(x)
        x = self.fc2(x)
        x = self.actf2(x)
        x = self.fc3(x)
        x = self.actf3(x)
        x = self.fc4(x)
        x = self.actf4(x)
        return x
    
class ValueLinear(nn.Module):
    def __init__(self,board_size=BOARD_SIZE):
        super(ValueLinear, self).__init__()
        self.fc1 = nn.Linear(board_size*board_size, 200)
        self.actf1 = nn.Sigmoid()
        self.fc2 = nn.Linear(200, 200)
        self.actf2 = nn.Sigmoid()
        self.fc3 = nn.Linear(200, 200)
        self.actf3 = nn.Sigmoid()
        self.fc4 = nn.Linear(200, 1)
        self.actf4 = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.actf1(x)
        x = self.fc2(x)
        x = self.actf2(x)
        x = self.fc3(x)
        x = self.actf3(x)
        x = self.fc4(x)
        x = self.actf4(x)
        return x
    
def test_MLPClassifier():
    input_size = 5 * 5
    hidden_size = 128
    num_classes = 10

    model = StrategyLinear(input_size, hidden_size, num_classes)


class CustomStrategyNet(nn.Module):
    def __init__(self, board_size=BOARD_SIZE):
        super(CustomStrategyNet, self).__init__()
        self.board_size = board_size
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(256 * board_size * board_size, 512)
        self.fc2 = nn.Linear(512, board_size * board_size)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1) # 使输出的所有元素加起来为1

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x.view(-1, self.board_size, self.board_size)     # torch.Size([1, 15, 15]), the first dimension is batch size

class CustomValueNet(nn.Module):
    def __init__(self, board_size=BOARD_SIZE):
        super(CustomValueNet, self).__init__()
        self.board_size = board_size

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(256 * board_size * board_size, 512)
        self.fc2 = nn.Linear(512, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # input: (batch_size, 3, board_size, board_size)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))

        x = x.view(x.size(0), -1)  # (batch_size, 256 * board_size * board_size)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        x = self.sigmoid(x)

        return x

class StrategyResnet18(nn.Module):
    def __init__(self, board_size=BOARD_SIZE):
        super(StrategyResnet18, self).__init__()
        self.board_size = board_size
        self.resnet = resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, board_size * board_size)

    #     self.outputs = {}
    #     for name, layer in self.resnet.named_children():
    #         layer.register_forward_hook(self.save_output_hook(name))

    # def save_output_hook(self, layer_name):
    #     def hook(module, input, output):
    #         self.outputs[layer_name] = output
    #     return hook

    def forward(self, x):
        x = self.resnet(x)
        return x.view(-1, self.board_size, self.board_size)
    
class StrategyMobilenetV2(nn.Module):
    def __init__(self, board_size=BOARD_SIZE):
        super(StrategyMobilenetV2, self).__init__()
        self.board_size = board_size
        self.mobilenet = mobilenet_v2(pretrained=False)
        self.mobilenet.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.mobilenet.classifier = nn.Linear(self.mobilenet.last_channel, board_size * board_size)

    def forward(self, x):
        x = self.mobilenet(x)
        return x.view(-1, self.board_size, self.board_size)


class ValueResnet18(nn.Module):
    def __init__(self, board_size=BOARD_SIZE):
        super(ValueResnet18, self).__init__()
        self.board_size = board_size
        self.resnet = resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)

    def forward(self, x):
        return self.resnet(x)


class ValueEfficientnetB0(nn.Module):
    def __init__(self):
        super(ValueEfficientnetB0, self).__init__()
        self.efficientnet = efficientnet_b0(pretrained=False)
        self.efficientnet.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.efficientnet.classifier = nn.Linear(self.efficientnet.classifier[-1].in_features, 1)

    def forward(self, x):
        return self.efficientnet(x)


def init_stategy_model():
    model = StrategyResnet18(BOARD_SIZE)
    torch.save(model.state_dict(), STRATEGY_MODEL_NAME)

def init_value_model():
    model = ValueEfficientnetB0()
    torch.save(model.state_dict(), VALUE_MODEL_NAME)

if __name__=='__main__':
    current_board_state = torch.randint(0, 3, (3, 15, 15)).float()  # 0, 1, 2 表示不同的棋子
    current_board_state = current_board_state.unsqueeze(0)
    model = ValueResnet18(BOARD_SIZE)
    model.eval()
    print(current_board_state.shape)
    output = model(current_board_state)
    print(output)
    # init_value_model()