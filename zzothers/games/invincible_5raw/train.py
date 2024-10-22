import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from model import *
from constent import BOARD_SIZE


def train(model, data_loader, optimizer, criterion, epochs=10):
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (boards, labels) in enumerate(data_loader):
            boards, labels = torch.tensor(boards), torch.tensor(labels)
            
            outputs = model(boards)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()  
            
            # 打印损失
            running_loss += loss.item()
            if (i+1) % 10 == 0:  # 每10个 batch 打印一次损失
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(data_loader)}], Loss: {running_loss/10:.4f}')
                running_loss = 0.0

def test_train(input_size, hidden_size, num_classes):
    dataset = GomokuDataset(1000)  # 创建一个1000条数据的假数据集
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = GomokuNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    # criterion = nn.CrossEntropyLoss()
    train(model, data_loader, optimizer, criterion, epochs=10)


    torch.save(model.state_dict(), 'model.pth')
    model = MLPClassifier(input_size, hidden_size, num_classes)
    model.load_state_dict(torch.load('model.pth'))


def train_stategy(model_path,epochs):
    dataset = GomokuDataset(1000)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = SimplifiedAlphaGoNet(BOARD_SIZE)
    try:
        model.load_state_dict(torch.load(model_path))
    except Exception as e:
        init_stategy_model()
    # criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (boards, labels) in enumerate(data_loader):
            boards, labels = torch.tensor(boards), torch.tensor(labels)
            
            outputs = model(boards)
            loss = criterion(outputs, labels)
            # loss = criterion(outputs.view(batch_size, -1), dummy_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(data_loader)}], Loss: {running_loss/10:.4f}')
                running_loss = 0.0

def train_value(model_path):
    board_size = BOARD_SIZE
    model = ValueNetwork(BOARD_SIZE)
    try:
        model.load_state_dict(torch.load(model_path))
    except Exception as e:
        init_stategy_model()

    batch_size = 16
    dummy_board = torch.randn(batch_size, 3, board_size, board_size)  # 假设输入棋盘状态
    dummy_target_value = torch.rand(batch_size, 1) * 2 - 1

    criterion = nn.MSELoss()  # 使用均方误差损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    predicted_value = model(dummy_board)
    loss = criterion(predicted_value, dummy_target_value)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("Loss:", loss.item())




