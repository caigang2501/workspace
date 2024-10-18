import pygame
import numpy as np
import torch
import os
import random
from torch import nn

# Pygame 初始化
pygame.init()

# 棋盘和窗口设置
BOARD_SIZE = 15
CELL_SIZE = 40  # 每个格子的像素大小
MARGIN = 20     # 棋盘边缘留白
WINDOW_SIZE = CELL_SIZE * BOARD_SIZE + 2 * MARGIN
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
FPS = 30

# 初始化窗口
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("五子棋 - 人机对弈")

# AI 模型
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

# 绘制棋盘
def draw_board():
    screen.fill(WHITE)
    for i in range(BOARD_SIZE):
        pygame.draw.line(screen, BLACK, (MARGIN + i * CELL_SIZE, MARGIN),
                         (MARGIN + i * CELL_SIZE, MARGIN + (BOARD_SIZE - 1) * CELL_SIZE))
        pygame.draw.line(screen, BLACK, (MARGIN, MARGIN + i * CELL_SIZE),
                         (MARGIN + (BOARD_SIZE - 1) * CELL_SIZE, MARGIN + i * CELL_SIZE))

# 绘制棋子
def draw_pieces(board):
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            if board[x, y] == 1:
                pygame.draw.circle(screen, BLACK, (MARGIN + y * CELL_SIZE, MARGIN + x * CELL_SIZE), CELL_SIZE // 2 - 5)
            elif board[x, y] == -1:
                pygame.draw.circle(screen, RED, (MARGIN + y * CELL_SIZE, MARGIN + x * CELL_SIZE), CELL_SIZE // 2 - 5)

# 检查是否有玩家胜利
def check_win(board, player):
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            if board[x][y] == player:
                if check_direction(board, x, y, player, 1, 0) or \
                   check_direction(board, x, y, player, 0, 1) or \
                   check_direction(board, x, y, player, 1, 1) or \
                   check_direction(board, x, y, player, 1, -1):
                    return True
    return False

def check_direction(board, x, y, player, dx, dy):
    count = 0
    for i in range(5):
        nx, ny = x + i * dx, y + i * dy
        if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and board[nx][ny] == player:
            count += 1
        else:
            break
    return count == 5

# AI 落子
def ai_move(model, board):
    board_tensor = torch.tensor(board, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        prediction = model(board_tensor).squeeze(0)
    while True:
        max_idx = torch.argmax(prediction).item()
        x, y = divmod(max_idx, BOARD_SIZE)
        if board[x][y] == 0:
            return x, y
        else:
            prediction[x][y] = -float('inf')  # 如果已经被占，继续找下一个位置

# 保存对局数据
def save_game_data(game_data, game_id, folder="games_data"):
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, f"game_{game_id}.npy")
    try:
        with open(file_path, 'wb') as f:
            np.save(f, game_data)
            f.flush()  # 确保数据写入磁盘
        print(f"对局数据保存至: {file_path}")
    except Exception as e:
        print(f"保存文件时出错: {e}")



# 人机对弈
def play_game(model):
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int32)
    game_data = []
    current_player = 1  # 1为玩家, -1为AI
    running = True
    clock = pygame.time.Clock()

    while running:
        draw_board()
        draw_pieces(board)
        pygame.display.flip()

        if current_player == 1:  # 玩家回合
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    row = (y - MARGIN) // CELL_SIZE
                    col = (x - MARGIN) // CELL_SIZE
                    if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE and board[row, col] == 0:
                        board[row, col] = current_player
                        game_data.append((board.copy(), current_player))  # 记录棋盘状态
                        if check_win(board, current_player):
                            print("玩家获胜!")
                            running = False
                        current_player *= -1  # 切换到 AI

        else:  # AI 回合
            x, y = ai_move(model, board)
            board[x, y] = current_player
            game_data.append((board.copy(), current_player))
            if check_win(board, current_player):
                print("AI 获胜!")
                running = False
            current_player *= -1  # 切换到玩家

        clock.tick(FPS)

    game_id = random.randint(1000, 9999)
    save_game_data(game_data, game_id)

# 测试代码
if __name__ == "__main__":
    model = GomokuNet()
    # model.load_state_dict(torch.load("trained_model.pth"))  
    play_game(model)
    pygame.quit()

