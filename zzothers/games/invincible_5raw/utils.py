import os,datetime
import numpy as np
import torch
from constent import STEPS_PATH


def game_id():
    now = datetime.datetime.now()
    return str(now.day)+str(now.hour)+str(now.minute)+str(now.second)

def save_steps(game_data,folder=STEPS_PATH):
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, game_id()+".npy")
    with open(file_path, 'wb') as f:
        np.save(f, game_data)

def load_steps(data_folder=STEPS_PATH):
    steps = []
    for file in os.listdir(data_folder):
        file_path = os.path.join(data_folder, file)
        game_data = np.load(file_path, allow_pickle=True)
        steps.append(game_data)
    return steps

def load_latest_steps(folder=STEPS_PATH):
    file_path = os.path.join(folder, os.listdir(folder)[-1])
    game_data = np.load(file_path, allow_pickle=True)

    return game_data

def oneto3_channel(board_state):
    empty = (board_state == 0).float()
    black = (board_state == 1).float()
    white = (board_state == -1).float()
    return torch.stack([empty, black, white], dim=0)  # (3, board_size, board_size)

def test_load_data():
    # save_steps([[1,2],[3,4],[5,6]])
    r = load_latest_steps()[::2]
    print(r)

def test_to3channel():
    list = [
        [0, 0,0,0,0],
        [0,-1,0,1,0],
        [0, 0,0,1,0],
        [0, 0,0,0,0],
    ]
    data = torch.tensor(list)
    data3 = oneto3_channel(data)
    print(data3)

if __name__=='__main__':
    test_to3channel()

