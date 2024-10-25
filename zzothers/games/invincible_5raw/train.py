import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from model import *
from dataset import *
from constent import BOARD_SIZE,STEPS_PATH,BATCH_SIZE
from tqdm import tqdm


def train_strategy(model_path,epochs):
    model = SimplifiedAlphaGoNet(BOARD_SIZE)
    try:
        model.load_state_dict(torch.load(model_path))
    except Exception as e:
        init_stategy_model()
    # criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in tqdm(range(epochs)):
        log = []
        for file in os.listdir(STEPS_PATH):
            steps_path = os.path.join(STEPS_PATH, file)
            steps = np.load(steps_path, allow_pickle=True)
            dataset = StrategyDataset(steps)
            data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
            model.train()
            running_loss = 0.0
            for board, labels in data_loader:
                outputs = model(board)
                loss = criterion(outputs.view(outputs.shape[0], -1), labels)
                # loss = criterion(outputs.view(batch_size, -1), dummy_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                log.append(round(running_loss/10,4)) 
                # print(f'Epoch [{epoch+1}/{epochs}], Step [{1}/{len(data_loader)}], Loss: {running_loss/10:.4f}')
        
        print(log)
    torch.save(model.state_dict(), model_path)

def train_value(model_path,epochs):
    model = ValueNetwork(BOARD_SIZE)
    try:
        model.load_state_dict(torch.load(model_path))
    except Exception as e:
        init_value_model()
    criterion = nn.MSELoss()
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in tqdm(range(epochs)):
        log = []
        for file in os.listdir(STEPS_PATH):
            steps_path = os.path.join(STEPS_PATH, file)
            steps = np.load(steps_path, allow_pickle=True)
            dataset = ValueDataset(steps)
            data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
            model.train()
            running_loss = 0.0
            for board, labels in data_loader:

                outputs = model(board)
                loss = criterion(outputs, labels)
                # loss = criterion(outputs.view(batch_size, -1), dummy_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
                log.append(round(running_loss/10,4))
                # print(f'Epoch [{epoch+1}/{epochs}], Step [1/{len(data_loader)}], Loss: {running_loss/10:.4f}')

        print(log)
    torch.save(model.state_dict(), model_path)


if __name__=='__main__':
    train_strategy('models/strategy_15.pth',epochs=10)
    train_value('models/value_15.pth',epochs=10)

