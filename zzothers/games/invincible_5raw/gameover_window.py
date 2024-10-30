
import sys
import pygame
from constent import *

class Button:
    def __init__(self, text, x, y, width, height, color, hover_color, action=None):
        self.text = text
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.hover_color = hover_color
        self.action = action

    def draw(self, screen):
        mouse_pos = pygame.mouse.get_pos()
        if self.rect.collidepoint(mouse_pos):
            pygame.draw.rect(screen, self.hover_color, self.rect)
        else:
            pygame.draw.rect(screen, self.color, self.rect)

        font = pygame.font.SysFont(None, 50)
        text_surface = font.render(self.text, True, BLACK)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

    def check_click(self):
        mouse_pos = pygame.mouse.get_pos()
        if self.rect.collidepoint(mouse_pos):
            if pygame.mouse.get_pressed()[0]:  # 检查是否按下鼠标左键
                if self.action:
                    self.action()

# def game_over_screen():
#     continue_button = Button("continue", 300, 200, 200, 50, GREEN, (0, 200, 0), action=lambda: play_game())
#     quit_button = Button("exit", 300, 300, 200, 50, RED, (200, 0, 0), quit_game)

#     while True:
#         screen.fill(WHITE)
#         continue_button.draw(screen)
#         quit_button.draw(screen)
#         continue_button.check_click()
#         quit_button.check_click()
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 pygame.quit()
#                 sys.exit()
#         pygame.display.update()

def quit_game():
    pygame.quit()
    sys.exit()

