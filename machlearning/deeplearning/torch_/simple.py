import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        # nn.Conv2d：二维卷积层。
        # nn.LSTM：长短时记忆网络（LSTM）层等。
        self.fc1 = nn.Linear(10, 5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 创建模型实例
model = SimpleModel()

# 输入数据
input_data = torch.randn(1, 10)

# 进行前向传播
output = model(input_data)
print(output)
