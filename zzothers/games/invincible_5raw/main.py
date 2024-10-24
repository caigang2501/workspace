import sys,pygame,torch
import numpy as np
from torch import nn
from utils import *
from model import *
from constent import *


pygame.init()
screen = pygame.display.set_mode((BOARD_WIDTH, BOARD_HEIGHT))
pygame.display.set_caption("五子棋")
board = np.zeros((BOARD_SIZE, BOARD_SIZE))
steps = []

def draw_board():
    screen.fill(WHITE)
    for row in range(BOARD_SIZE):
        pygame.draw.line(screen, BLACK, (CELL_SIZE//2, row * CELL_SIZE + CELL_SIZE//2),
                         (BOARD_WIDTH - CELL_SIZE//2, row * CELL_SIZE + CELL_SIZE//2))
    for col in range(BOARD_SIZE):
        pygame.draw.line(screen, BLACK, (col * CELL_SIZE + CELL_SIZE//2, CELL_SIZE//2),
                         (col * CELL_SIZE + CELL_SIZE//2, BOARD_HEIGHT - CELL_SIZE//2))

def draw_pieces():
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row, col] == 1:
                pygame.draw.circle(screen, BLACK, 
                                   (col * CELL_SIZE + CELL_SIZE//2, row * CELL_SIZE + CELL_SIZE//2), CELL_SIZE//2 - 2)
            elif board[row, col] == -1:
                pygame.draw.circle(screen, WHITE, 
                                   (col * CELL_SIZE + CELL_SIZE//2, row * CELL_SIZE + CELL_SIZE//2), CELL_SIZE//2 - 2)
                pygame.draw.circle(screen, BLACK, 
                                   (col * CELL_SIZE + CELL_SIZE//2, row * CELL_SIZE + CELL_SIZE//2), CELL_SIZE//2 - 2, 1)

def check_win(player):
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if col <= BOARD_SIZE - 5 and np.all(board[row, col:col+5] == player):  
                return True
            if row <= BOARD_SIZE - 5 and np.all(board[row:row+5, col] == player):  
                return True
            if row <= BOARD_SIZE - 5 and col <= BOARD_SIZE - 5 and np.all([board[row+i, col+i] == player for i in range(5)]):
                return True
            if row >= 4 and col <= BOARD_SIZE - 5 and np.all([board[row-i, col+i] == player for i in range(5)]):
                return True
    return False


def ai_move(model):
    # unsqueeze:加维度  squeeze:删维度
    board_tensor = board_to_tensor(board).unsqueeze(0)
    with torch.no_grad():
        prediction = model(board_tensor).squeeze(0)     # torch.Size([15, 15])
    while True:
        max_idx = torch.argmax(prediction).item()
        x, y = divmod(max_idx, BOARD_SIZE)
        if board[x,y] == 0:
            board[x,y] = -1
            break
        else:
            prediction[x][y] = -float('inf')
    board[x,y] = -1
    steps.append([x,y])
    if check_win(-1):
        return True
    return False

def play_game(model):
    player = 1  
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
                    steps.append([row,col])
                    if board[row, col] == 0:
                        board[row, col] = player
                        if check_win(player):
                            game_over = True
                            print(f"玩家 {player} 胜利！")
                    player = -1
        else:
            game_over = ai_move(model)
            player = 1
    print(len(steps))
    save_steps(steps)
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    model = SimplifiedAlphaGoNet(15)
    # model.load_state_dict(torch.load("trained_model.pth"))  
    play_game(model)
