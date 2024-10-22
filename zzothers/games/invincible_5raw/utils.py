import os,datetime
import numpy as np


def game_id():
    now = datetime.datetime.now()
    return str(now.day)+str(now.hour)+str(now.minute)

def save_txt(game_data,folder="games_data"):
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, game_id()+".txt")
    with open(file_path, 'w') as f:
        f.write(str(game_data))

def read_txt(folder="games_data"):
    for file in os.listdir(folder):
        file_path = os.path.join(folder, f"{file}.txt")
        with open(file_path, 'r') as f:
            text = f.read()
    return text

def save_nr(game_data,folder="games_data"):
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, game_id()+".npy")
    with open(file_path, 'wb') as f:
        np.save(f, game_data)
        f.flush()  


def load_nr(data_folder="games_data"):
    for file in os.listdir(data_folder):
        file_path = os.path.join(data_folder, file)
        game_data = np.load(file_path, allow_pickle=True)

    return game_data



if __name__=='__main__':
    save_txt([[1,2],[3,4]])


