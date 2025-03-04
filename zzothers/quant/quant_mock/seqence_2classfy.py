import sys,os
sys.path.append(os.getcwd())
from zzothers.quant.data.tushare.data import data_ts1

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim

class StockPredictionModel(nn.Module):
    def __init__(self):
        super(StockPredictionModel, self).__init__()
        self.fc1 = nn.Linear(25, 64)
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
    
def predict():
    pass

def mock_trading(predicted,pct_chg,double_direction=False):
    base_money = 100
    up_right,up_wrong,down_right,down_wrong = 0,0,0,0
    earn,loss = 0,0
    for out,p in zip(predicted,pct_chg/100):
        if not double_direction:
            if out>0:
                base_money *= 1+p
                if 1+p>1:
                    earn += 1
                elif 1+p<1:
                    loss += 1
        else:
            if out>0:
                base_money *= 1+p
                if 1+p>1:
                    earn += 1
                elif 1+p<1:
                    loss += 1
            else:
                base_money *= 1-p
                if 1-p>1:
                    earn += 1
                elif 1-p<1:
                    loss += 1            
        if p>0:
            if out==1:
                up_right += 1
            else:
                up_wrong += 1
        elif p<0:
            if out==0:
                down_right += 1
            else:
                down_wrong += 1
    if not double_direction:
        print('up_accuracy:',round(up_right/(up_right+up_wrong),2),'down_accuracy:',round(down_right/(down_right+down_wrong),2))

    # print(earn,loss)
    return base_money

def train_data():
    window_size = 5
    def day2month(data):
        features = ['trade_date','open', 'high', 'low','close','pct_chg','vol','amount']
        data = data[features]
        data.loc[:, 'trade_date'] = data['trade_date']//100
        data = data.groupby('trade_date').agg({'open':'last','high':'max','low':'min','close': 'first','vol': 'sum','amount': 'sum'})
        data['pct_chg'] = data['close'].pct_change()*100
        data = data.dropna()
        return data
    def data_nday(data, window_size=10):
        X = []
        for i in range(len(data)-window_size):
            if isinstance(data,pd.DataFrame):
                features = data.iloc[i:i+window_size,:].values.flatten()
            else:
                features = data[i:i+window_size,:].flatten()
            X.append(features)
        X = np.array(X)
        return X
    # df_day = data_ts1('000002.SZ','20230201',save=True)
    df_day = pd.read_csv('data/quant/000002SZ.csv')
    # b = df_day.iloc[-1, df_day.columns.get_loc('close')]
    # print((df_day.loc[0,'close']-b)/b)
    df_month = day2month(df_day)
    features = ['open', 'high', 'low', 'close', 'vol']
    df_day['pct_chg'] = df_day['pct_chg'].shift(1)
    data = df_day.dropna()
    # print(data.head())
    X = data_nday(data[features],window_size)
    y = data['pct_chg'].iloc[:-window_size]
    # print(X[:3,:])
    # print(y[:3])
    scaler = StandardScaler()
    # scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)
    # print(X_scaled[:5,:])
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


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
        if (epoch+1) % 20 == 0:
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
        accuracy = accuracy_score(y_test,predicted.numpy())

    final_money = mock_trading(predicted.numpy(),y_test_tensor)
    final_money_d = mock_trading(predicted.numpy(),y_test_tensor,double_direction=True)
    
    print('pre_up_pct:',sum(predicted)/len(predicted),'real_up_pct:',sum([1 if p>0 else 0 for p in y_test])/len(predicted))
    print(f'Accuracy: {accuracy:.2f}')
    print(f'final_money: {final_money:.2f}')
    print(f'final_money_d: {final_money_d:.2f}')

    with torch.no_grad():
        outputs = model(X_train_tensor)
        _, predicted = torch.max(outputs, 1)
        y_test = torch.where(y_train_tensor > 0, 1, 0)
        accuracy = accuracy_score(y_test,predicted.numpy())

    final_money = mock_trading(predicted.numpy(),y_train_tensor)
    final_money_d = mock_trading(predicted.numpy(),y_train_tensor,double_direction=True)
    print('pre_up_pct:',sum(predicted)/len(predicted),'real_up_pct:',sum([1 if p>0 else 0 for p in y_test])/len(predicted))
    print(f'Accuracy: {accuracy:.2f}')
    print(f'final_money: {final_money:.2f}')
    print(f'final_money_d: {final_money_d:.2f}')

if __name__=='__main__':
    train()
