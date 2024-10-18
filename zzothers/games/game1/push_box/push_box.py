import pygame
import sys
import maps

# 初始化 Pygame
pygame.init()

# 定义颜色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)

# 设置窗口大小
WIDTH, HEIGHT = 640, 480
TILE_SIZE = 40

# 初始化屏幕
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Sokoban - 推箱子")

# 定义游戏地图
game_maps = maps.map

# 当前关卡
current_level = 0
game_map = game_maps[current_level]

# 定义角色的初始位置
player_x = 3
player_y = 3

# 定义地图对象的字典
tile_mapping = {
    "#": BLACK,  # 墙
    " ": WHITE,  # 空地
    "$": BLUE,   # 箱子
    ".": RED,    # 目标点
    "@": GREEN,  # 玩家
}

# 绘制游戏地图
def draw_map():
    for row in range(len(game_map)):
        for col in range(len(game_map[row])):
            tile = game_map[row][col]
            color = tile_mapping.get(tile, WHITE)
            pygame.draw.rect(screen, color, pygame.Rect(col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE))

# 检查是否所有箱子都在目标点上
def check_win():
    for row in game_map:
        if '.' in row:
            return False
    return True

# 更新玩家和箱子的位置
def move_player(dx, dy):
    global player_x, player_y

    new_x = player_x + dx
    new_y = player_y + dy

    next_tile = game_map[new_y][new_x]

    # 如果玩家前面是墙，不能移动
    if next_tile == "#":
        return

    # 如果玩家前面是箱子，检查箱子后面是否为空地或目标点
    if next_tile == "$":
        box_new_x = new_x + dx
        box_new_y = new_y + dy
        box_next_tile = game_map[box_new_y][box_new_x]

        # 如果箱子可以被推动到的地方是空地或者目标点
        if box_next_tile == " " or box_next_tile == ".":
            # 移动箱子
            game_map[new_y] = game_map[new_y][:new_x] + " " + game_map[new_y][new_x+1:]
            game_map[box_new_y] = game_map[box_new_y][:box_new_x] + "$" + game_map[box_new_y][box_new_x+1:]
            player_x = new_x
            player_y = new_y
    # 如果是空地或目标点，玩家可以移动
    elif next_tile == " " or next_tile == ".":
        player_x = new_x
        player_y = new_y

# 显示“闯关成功”消息和按钮
def show_success_screen():
    font = pygame.font.SysFont(None, 55)
    text = font.render("succeed", True, YELLOW)
    screen.blit(text, (200, 150))

    button_rect = pygame.Rect(240, 250, 160, 50)
    pygame.draw.rect(screen, GREEN, button_rect)
    button_font = pygame.font.SysFont(None, 40)
    button_text = button_font.render("next level", True, BLACK)
    screen.blit(button_text, (255, 260))

    return button_rect

# 加载下一关
def load_next_level():
    global current_level, game_map, player_x, player_y
    current_level += 1
    if current_level < len(game_maps):
        game_map = game_maps[current_level]
        player_x, player_y = 1, 3  # 玩家新关卡的起始位置
    else:
        print("恭喜你，通关了！")
        pygame.quit()
        sys.exit()

# 游戏主循环
game_won = False
while True:
    # 处理事件
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        # 检查按键移动玩家
        if not game_won and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                move_player(-1, 0)
            elif event.key == pygame.K_RIGHT:
                move_player(1, 0)
            elif event.key == pygame.K_UP:
                move_player(0, -1)
            elif event.key == pygame.K_DOWN:
                move_player(0, 1)

            # 检查是否游戏胜利
            if check_win():
                game_won = True

        # 如果胜利，检查按钮点击事件
        if game_won and event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = event.pos
            if next_button.collidepoint(mouse_x, mouse_y):
                game_won = False
                load_next_level()

    # 填充背景
    screen.fill(WHITE)

    # 绘制游戏地图
    draw_map()

    # 绘制玩家
    pygame.draw.rect(screen, GREEN, pygame.Rect(player_x * TILE_SIZE, player_y * TILE_SIZE, TILE_SIZE, TILE_SIZE))

    # 如果游戏胜利，显示成功界面
    if game_won:
        next_button = show_success_screen()

    # 更新屏幕
    pygame.display.flip()

    # 控制游戏帧率
    pygame.time.Clock().tick(30)


