import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from model import *
from dataset import *
from constent import BOARD_SIZE,STEPS_PATH


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
            
            running_loss += loss.item()
            if (i+1) % 10 == 0:  
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(data_loader)}], Loss: {running_loss/10:.4f}')
                running_loss = 0.0

def test_train(input_size, hidden_size, num_classes):
    dataset = StrategyDataset(1000)  
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = ValueDataset()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    # criterion = nn.CrossEntropyLoss()
    train(model, data_loader, optimizer, criterion, epochs=10)


    torch.save(model.state_dict(), 'model.pth')
    model = MLPClassifier(input_size, hidden_size, num_classes)
    model.load_state_dict(torch.load('model.pth'))


def train_stategy(model_path,epochs):
    model = SimplifiedAlphaGoNet(BOARD_SIZE)
    try:
        model.load_state_dict(torch.load(model_path))
    except Exception as e:
        init_stategy_model()
    # criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for file in os.listdir(STEPS_PATH):
        steps_path = os.path.join(STEPS_PATH, file)
        steps = np.load(steps_path, allow_pickle=True)
        dataset = StrategyDataset(steps)
        data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for board, labels in data_loader:
                outputs = model(board)
                loss = criterion(outputs.view(outputs.shape[0], -1), labels)
                # loss = criterion(outputs.view(batch_size, -1), dummy_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() 
                print(f'Epoch [{epoch+1}/{epochs}], Step [{1}/{len(data_loader)}], Loss: {running_loss/10:.4f}')
    
    # torch.save(model.state_dict(), model_path)

def train_value(model_path,epochs):
    model = ValueNetwork(BOARD_SIZE)
    try:
        model.load_state_dict(torch.load(model_path))
    except Exception as e:
        init_value_model()
    criterion = nn.MSELoss()
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for file in os.listdir(STEPS_PATH):
        steps_path = os.path.join(STEPS_PATH, file)
        steps = np.load(steps_path, allow_pickle=True)
        dataset = ValueDataset(steps)
        data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for board, labels in data_loader:

                outputs = model(board)
                loss = criterion(outputs, labels)
                # loss = criterion(outputs.view(batch_size, -1), dummy_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                print(f'Epoch [{epoch+1}/{epochs}], Step [1/{len(data_loader)}], Loss: {running_loss/10:.4f}')
    
    torch.save(model.state_dict(), model_path)


if __name__=='__main__':
    train_stategy('models/strategy_15.pth',epochs=10)
    # train_value('models/value_15.pth',epochs=10)

