import os,datetime
import numpy as np
import torch


def game_id():
    now = datetime.datetime.now()
    return str(now.day)+str(now.hour)+str(now.minute)

def save_steps(game_data,folder="games_data"):
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, game_id()+".npy")
    with open(file_path, 'wb') as f:
        np.save(f, game_data)
        f.flush()

def load_steps(data_folder="games_data"):
    steps = []
    for file in os.listdir(data_folder):
        file_path = os.path.join(data_folder, file)
        game_data = np.load(file_path, allow_pickle=True)
        steps.append(game_data)
    return steps

def load_latext_steps(folder="games_data"):
    file_path = os.path.join(folder, os.listdir(folder)[-1])
    game_data = np.load(file_path, allow_pickle=True)

    return game_data

def oneto3_channel(board_state):
    empty = (board_state == 0).float()  # 空位通道
    black = (board_state == 1).float()  # 黑棋通道
    white = (board_state == -1).float() # 白棋通道
    return torch.stack([empty, black, white], dim=0)  # (3, board_size, board_size)

if __name__=='__main__':
    # save_steps([[1,2],[3,4],[5,6]])
    r = load_latext_steps()
    print(type(r),r)

