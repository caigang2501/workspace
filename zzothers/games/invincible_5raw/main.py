import pygame
import sys
import numpy as np
import torch
from torch import nn
# 初始化 Pygame
pygame.init()

# 常量定义
BOARD_SIZE = 15  # 棋盘大小：15x15
CELL_SIZE = 40  # 每个格子的大小
BOARD_WIDTH = BOARD_SIZE * CELL_SIZE  # 棋盘宽度
BOARD_HEIGHT = BOARD_SIZE * CELL_SIZE  # 棋盘高度
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

screen = pygame.display.set_mode((BOARD_WIDTH, BOARD_HEIGHT))
pygame.display.set_caption("五子棋")
board = np.zeros((BOARD_SIZE, BOARD_SIZE))  # 0 表示空，1 表示黑棋，2 表示白棋

class GomokuNet(nn.Module):
    def __init__(self):
        super(GomokuNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256 * BOARD_SIZE * BOARD_SIZE, 512)
        self.fc2 = nn.Linear(512, BOARD_SIZE * BOARD_SIZE)

    def forward(self, x):
        print(x.shape)
        x = torch.relu(self.conv1(x))
        print(x.shape)
        x = torch.relu(self.conv2(x))
        print(x.shape)
        x = torch.relu(self.conv3(x))
        print(x.shape)
        x = x.view(-1, 256 * BOARD_SIZE * BOARD_SIZE)
        x = torch.relu(self.fc1(x))
        return self.fc2(x).view(-1, BOARD_SIZE, BOARD_SIZE)

def draw_board():
    screen.fill(WHITE)
    for row in range(BOARD_SIZE):
        pygame.draw.line(screen, BLACK, (CELL_SIZE//2, row * CELL_SIZE + CELL_SIZE//2), 
                         (BOARD_WIDTH - CELL_SIZE//2, row * CELL_SIZE + CELL_SIZE//2))
    for col in range(BOARD_SIZE):
        pygame.draw.line(screen, BLACK, (col * CELL_SIZE + CELL_SIZE//2, CELL_SIZE//2), 
                         (col * CELL_SIZE + CELL_SIZE//2, BOARD_HEIGHT - CELL_SIZE//2))

# 在棋盘上绘制棋子
def draw_pieces():
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row, col] == 1:  # 黑棋
                pygame.draw.circle(screen, BLACK, 
                                   (col * CELL_SIZE + CELL_SIZE//2, row * CELL_SIZE + CELL_SIZE//2), CELL_SIZE//2 - 2)
            elif board[row, col] == -1:  # 白棋
                pygame.draw.circle(screen, WHITE, 
                                   (col * CELL_SIZE + CELL_SIZE//2, row * CELL_SIZE + CELL_SIZE//2), CELL_SIZE//2 - 2)
                pygame.draw.circle(screen, BLACK, 
                                   (col * CELL_SIZE + CELL_SIZE//2, row * CELL_SIZE + CELL_SIZE//2), CELL_SIZE//2 - 2, 1)

# 判断胜利条件
def check_win(player):
    # 横、竖、斜方向分别判断
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if col <= BOARD_SIZE - 5 and np.all(board[row, col:col+5] == player):  # 横向
                return True
            if row <= BOARD_SIZE - 5 and np.all(board[row:row+5, col] == player):  # 纵向
                return True
            if row <= BOARD_SIZE - 5 and col <= BOARD_SIZE - 5 and np.all([board[row+i, col+i] == player for i in range(5)]):  # 右下对角线
                return True
            if row >= 4 and col <= BOARD_SIZE - 5 and np.all([board[row-i, col+i] == player for i in range(5)]):  # 右上对角线
                return True
    return False

xx,yy = 0,0
def ai_move(board):
    global xx
    xx += 1
    board[xx, yy] = 2
    if check_win(2):
        print(f"AI 胜利！")
        return True
    return False

def ai_move(model):
    board_tensor = torch.tensor(board, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        prediction = model(board_tensor).squeeze(0)
    while True:
        max_idx = torch.argmax(prediction).item()
        x, y = divmod(max_idx, BOARD_SIZE)
        if board[x][y] == 0:
            board[x, y] = -1
            break
        else:
            prediction[x][y] = -float('inf')  

    board[x, y] = -1
    # game_data.append((board.copy(), current_player))
    if check_win(-1):
        return True
    return False

def play_game(model):
    player = 1  # 玩家 1（黑棋）先手
    game_over = False
    
    while not game_over:
        draw_board()
        draw_pieces()
        pygame.display.flip()
        if player==1:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = pygame.mouse.get_pos()
                    col = x // CELL_SIZE
                    row = y // CELL_SIZE
                    if board[row, col] == 0:  # 如果该位置为空，则可以下棋
                        board[row, col] = player
                        if check_win(player):
                            game_over = True
                            print(f"玩家 {player} 胜利！")
                    player = -1
        else:
            game_over = ai_move(model)
            player = 1
    
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    model = GomokuNet()
    # model.load_state_dict(torch.load("trained_model.pth"))  
    play_game(model)
