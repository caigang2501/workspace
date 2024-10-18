import pygame
import random
import sys

# 初始化 Pygame
pygame.init()

# 定义颜色
GRAY = (128, 128, 128)  # 灰色背景
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# 设置窗口大小
WIDTH, HEIGHT = 600, 600  # 窗口高度调整为600
PLAYER_WIDTH = 60
PLAYER_HEIGHT = 20
BLOCK_WIDTH = 50  # 石块的宽度
BLOCK_HEIGHT = 50  # 石块的高度
FPS = 2  # 设置帧率为2帧/秒

# 初始化屏幕
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("躲避石块游戏 - 一帧一帧移动")

# 定义玩家
player_x = WIDTH // 2 - PLAYER_WIDTH // 2
player_y = HEIGHT - PLAYER_HEIGHT - 10
player_speed = BLOCK_WIDTH  # 玩家每次移动的距离与石块的宽度相同
last_player_input = 0  # 记录玩家最后一次输入（-1 表示向左, 1 表示向右, 0 表示不动）

# 定义石块
block_speed = BLOCK_WIDTH  # 石块每次移动一个 BLOCK_WIDTH 的距离
blocks = []

# 生成新的石块
def create_block():
    x = random.randint(0, WIDTH - BLOCK_WIDTH)
    y = -BLOCK_HEIGHT
    return pygame.Rect(x, y, BLOCK_WIDTH, BLOCK_HEIGHT)

# 碰撞检测
def check_collision(player, blocks):
    for block in blocks:
        if player.colliderect(block):
            return True
    return False

# 显示游戏结束画面
def show_game_over():
    font = pygame.font.SysFont(None, 75)
    text = font.render("游戏结束", True, RED)
    screen.blit(text, (WIDTH // 2 - 150, HEIGHT // 2 - 50))

    font = pygame.font.SysFont(None, 55)
    restart_text = font.render("按 R 重新开始", True, GREEN)
    screen.blit(restart_text, (WIDTH // 2 - 150, HEIGHT // 2 + 50))

# 重置游戏
def reset_game():
    global player_x, blocks, block_speed, last_player_input
    player_x = WIDTH // 2 - PLAYER_WIDTH // 2
    blocks = []
    block_speed = BLOCK_WIDTH  # 石块速度恢复为初始值
    last_player_input = 0

# 游戏主循环
game_over = False
clock = pygame.time.Clock()
frame_count = 0  # 用于控制奇偶帧

while True:
    # 事件处理
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    keys = pygame.key.get_pressed()

    if not game_over:
        # 奇数帧：更新石块下落，玩家不能移动
        if frame_count % 2 == 1:
            # 创建新的石块
            if random.randint(1, 20) == 1:
                blocks.append(create_block())

            # 更新石块位置，每次移动一个 BLOCK_WIDTH
            for block in blocks:
                block.y += block_speed

            # 移除离开屏幕的石块
            blocks = [block for block in blocks if block.y < HEIGHT]

        # 偶数帧：更新玩家移动，但石块不更新
        else:
            # 处理玩家的最后输入
            if keys[pygame.K_LEFT]:
                last_player_input = -1
            elif keys[pygame.K_RIGHT]:
                last_player_input = 1
            else:
                last_player_input = 0

            # 根据最后的输入更新玩家的位置
            player_x += last_player_input * player_speed

            # 边界检查
            if player_x < 0:
                player_x = 0
            elif player_x > WIDTH - PLAYER_WIDTH:
                player_x = WIDTH - PLAYER_WIDTH

        # 碰撞检测
        player_rect = pygame.Rect(player_x, player_y, PLAYER_WIDTH, PLAYER_HEIGHT)
        if check_collision(player_rect, blocks):
            game_over = True

    # 清空屏幕，使用灰色背景
    screen.fill(GRAY)

    # 绘制玩家
    player_rect = pygame.Rect(player_x, player_y, PLAYER_WIDTH, PLAYER_HEIGHT)
    pygame.draw.rect(screen, GREEN, player_rect)

    # 绘制石块
    for block in blocks:
        pygame.draw.rect(screen, BLACK, block)

    # 显示游戏结束
    if game_over:
        show_game_over()
        if keys[pygame.K_r]:
            game_over = False
            reset_game()

    # 刷新屏幕
    pygame.display.flip()

    # 控制游戏帧率
    clock.tick(FPS)

    # 更新帧计数
    frame_count += 1
