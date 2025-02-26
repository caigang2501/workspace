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

class StockPredictionModel(nn.Module):
    def __init__(self):
        super(StockPredictionModel, self).__init__()
        self.fc1 = nn.Linear(5, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, outputs:torch.Tensor, targets):
        aim = torch.where(targets > 0, 1, 0)
        t1 = torch.where(targets > 0, 1, -1)
        t2 = 2*(outputs[:, 0] < outputs[:, 1]).int()-1
        t = t1*t2                                                           # 正确:1 错误:-1
        # targets = torch.argmax(one_hot_targets, dim=1)                    # one_hot -> 值
        targets_onehot = torch.nn.functional.one_hot(aim, num_classes=2)    # 值 -> one_hot
        adj_param = 2/(1+torch.e**(torch.abs(targets)*t))
        adj_param = adj_param.unsqueeze(1)
        mse = torch.mean((outputs-targets_onehot)**2)
        adj_mse = torch.mean((outputs-targets_onehot)**2*adj_param)

        # print('outputs',outputs)
        # print('targets',targets)
        # print('t',t)
        # print('adj_param',adj_param)
        # print('(outputs-targets_onehot)**2',(outputs-targets_onehot)**2)
        return adj_mse
    

def train_data():
    def day2month(data):
        features = ['trade_date','open', 'high', 'low','close','pct_chg','vol','amount']
        data = data[features]
        data.loc[:, 'trade_date'] = data['trade_date']//100
        data = data.groupby('trade_date').agg({'open':'last','high':'max','low':'min','close': 'first','vol': 'sum','amount': 'sum'})
        data['pct_chg'] = data['close'].pct_change()*100
        data = data.dropna()
        return data

    # df_day = data_ts1('20230201')
    # df_day.to_csv('data/quant/df_day.csv')
    df_day = pd.read_csv('data/quant/df_day.csv')
    df_month = day2month(df_day)
    features = ['open', 'high', 'low', 'close', 'vol']
    df_day['pct_chg'] = df_day['pct_chg'].shift(1)
    data = df_day.dropna()

    X = data[features]
    y = data['pct_chg']
    scaler = StandardScaler()
    # scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)
    # print(X_scaled[:5,:])
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    return X_train_tensor,X_test_tensor,y_train_tensor,y_test_tensor

def train():
    X_train_tensor,X_test_tensor,y_train_tensor,y_test_tensor = train_data()
    # y_train_tensor = torch.where(y_train_tensor > 0, 1, 0)
    # y_test_tensor = torch.where(y_test_tensor > 0, 1, 0)
    # criterion = nn.CrossEntropyLoss()
    criterion = CustomLoss()
    
    model = StockPredictionModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 200
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

    # 测试集准确率
    model = StockPredictionModel()
    model.load_state_dict(torch.load(f'data/quant/mlp.pth'))
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        y_test = torch.where(y_test_tensor > 0, 1, 0)
        print(predicted)
        accuracy = accuracy_score(y_test, predicted.numpy())
        print(f'Accuracy: {accuracy:.4f}')

def predict():
    pass

if __name__=='__main__':
    train()
