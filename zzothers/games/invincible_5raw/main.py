import sys,pygame,torch,time
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


def play_game(player1,player2):
    def generate_value(player,step):
        tmp_board = torch.tensor(board,dtype=torch.float32)
        x, y = divmod(int(step), BOARD_SIZE)
        tmp_board[x,y] = player
        if player==1:
            board_state = torch.tensor(board, dtype=torch.float32)
        else:
            board_state = -torch.tensor(board, dtype=torch.float32)
        board_state = oneto3_channel(board_state).unsqueeze(0)
        value = value_model(board_state)
        return value
    def ai_move(player):
        # time.sleep(1)
        # unsqueeze:加维度  squeeze:降维度
        board_tensor = board_to_tensor(board).unsqueeze(0)
        with torch.no_grad():
            prediction = strategy_model(board_tensor).squeeze(0)     # torch.Size([15, 15])
        while True:
            # max_idx_ = torch.argmax(prediction).item()
            topk_values, topk_indices = torch.topk(prediction.view(-1), k=BRANCH_COUND)
            max_idx_ = max([(generate_value(player,idx),idx) for v,idx in zip(topk_values,topk_indices)])[1]
            x, y = divmod(int(max_idx_), BOARD_SIZE)
            if board[x,y] == 0:
                board[x,y] = player
                break
            else:
                prediction[x][y] = -float('inf')
        board[x,y] = player
        steps.append([x,y])
        if check_win(player):
            print('AI胜利')
            return True,-player
        return False,-player
    
    def player_move(game_over,quit_game,player):
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
                player = -player
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    quit_game = True
            if event.type == pygame.QUIT:  
                quit_game = True        
        return game_over,quit_game,player
    
    first = 1
    player = first  
    game_over = False
    quit_game = False
    global board

    while not quit_game:
        draw_board()
        draw_pieces()
        pygame.display.flip()
        if player==1:
            if player1==PERSON:
                game_over,quit_game,player = player_move(game_over,quit_game,player)
            else:
                game_over,player = ai_move(player)
        else:
            if player2==PERSON:
                game_over,quit_game,player = player_move(game_over,quit_game,player)
            else:
                game_over,player = ai_move(player)
            
        if game_over:
            if SAVE_BOARD:
                save_steps(steps,folder=DATASET_PATH_SAVE)
            board = board*0
            game_over = False
            player = first

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    strategy_model = StrategyResnet18(15).eval()
    value_model = ValueEfficientnetB0().eval()
    strategy_model.load_state_dict(torch.load(STRATEGY_MODEL_NAME))
    value_model.load_state_dict(torch.load(VALUE_MODEL_NAME))
    play_game(player1=PERSON,player2=MACHINE)
