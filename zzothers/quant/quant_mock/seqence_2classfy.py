import sys,os
sys.path.append(os.getcwd())
from zzothers.quant.data.tushare.data import data_ts1

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim

# data = data_ts1()
# data.to_csv('data/stock_data.csv')
data = pd.read_csv('data/quant/stock_data.csv')

data['Price_Change'] = data['close'].pct_change()
data['Target'] = np.where(data['Price_Change'] > 0, 1, 0)
data = data.dropna()

features = ['open', 'high', 'low', 'close', 'vol']
X = data[features]
y = data['Target']
scaler = StandardScaler()
# scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)




X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

class StockPredictionModel(nn.Module):
    def __init__(self):
        super(StockPredictionModel, self).__init__()
        self.fc1 = nn.Linear(5, 64)  # 输入5个特征，输出64个神经元
        self.fc2 = nn.Linear(64, 32) # 隐藏层
        self.fc3 = nn.Linear(32, 2)  # 输出层，二分类问题

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 激活函数 ReLU
        x = torch.relu(self.fc2(x))  # 激活函数 ReLU
        x = self.fc3(x)              # 输出层
        return x

def train():
    model = StockPredictionModel()
    criterion = nn.CrossEntropyLoss()  # 二分类问题用 CrossEntropyLoss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 30
    best_loss = float('inf')
    train_result = []
    for epoch in range(epochs):
        model.train()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch+1) % 2 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

        curr_loss = round(loss.item(),8)
        train_result.append(curr_loss)
        if curr_loss<best_loss:
            torch.save(model.state_dict(), f'data/quant/mlp.pth')
            best_loss = curr_loss
        
        train_result

def predict():
    model = StockPredictionModel()
    model.load_state_dict(torch.load(f'data/quant/mlp.pth'))
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        
        accuracy = accuracy_score(y_test, predicted.numpy())
        print(f'Accuracy: {accuracy:.4f}')

# train()
predict()
