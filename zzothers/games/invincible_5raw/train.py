import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
from model import *
from dataset import *
from constent import *
from tqdm import tqdm


def train_strategy(model_path,epochs):
    model = StrategyResnet18(BOARD_SIZE)
    try:
        model.load_state_dict(torch.load(model_path))
    except Exception as e:
        pass
    # criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()   # CrossEntropyLoss 会自动对输出进行 softmax
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    min_loss = float('inf')
    for epoch in tqdm(range(epochs)):
        # dataset = StrategyDataset(steps)
        train_data,train_labels = strategy_dataset(DATASET_PATH_TRAIN)
        dataset = TensorDataset(train_data, train_labels)

        data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        model.train()
        running_loss = 0.0
        for board, labels in data_loader:
            outputs = model(board)
            # print(outputs.shape)
            # topk_values, topk_indices = torch.topk(outputs, k=3)
            # print(topk_values, topk_indices)
            loss = criterion(outputs.view(outputs.shape[0], -1), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            # print(f'Epoch [{epoch+1}/{epochs}], Step [{1}/{len(data_loader)}], Loss: {running_loss/10:.4f}')
            
        print(f' epoch: {epoch}/{epochs}    loss: {running_loss}')
        if running_loss<min_loss:
            min_loss = running_loss
            torch.save(model.state_dict(), MODEL_NEW_TRAINED_PATH+f'strategy15_{epoch}_{epochs}_{round(running_loss,5)}.pth')

def train_value(model_path,epochs):
    model = ValueEfficientnetB0()
    try:
        model.load_state_dict(torch.load(model_path))
    except Exception as e:
        pass
    criterion = nn.MSELoss()
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    min_loss = float('inf')
    for epoch in tqdm(range(epochs)):
        value_data,value_labels = value_dataset(DATASET_PATH_TRAIN)
        dataset = TensorDataset(value_data, value_labels)
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

            # print(f'Epoch [{epoch+1}/{epochs}], Step [1/{len(data_loader)}], Loss: {running_loss/10:.4f}')

        print(f' epoch: {epoch}/{epochs}    loss: {running_loss}')
        if running_loss<min_loss:
            min_loss = running_loss
            torch.save(model.state_dict(), MODEL_NEW_TRAINED_PATH+f'value15_{epoch}_{epochs}_{round(running_loss,5)}.pth')


if __name__=='__main__':
    train_strategy('models/strategy15.pth',epochs=STRATEGY_EPOCHS)
    train_value('models/value15.pth',epochs=VALUE_EPOCHS)

