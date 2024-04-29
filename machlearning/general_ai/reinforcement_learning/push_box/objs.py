
import pygame
import constants as c


class Block(pygame.sprite.Sprite):

    # Constructor. Pass in the color of the block,
    # and its x and y position
    def __init__(self, x, y, p):
        # Call the parent class (Sprite) constructor
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface(p['shape'])
        self.image.fill(p['color'])
        self.rect = self.image.get_rect()
        self.rect.topleft = (x,y) 

        # self.rect.x,self.rect.x = self.x,self.y

    def update(self, x, y):
        self.rect.x, self.rect.y = x,y

class Ball(pygame.sprite.Sprite):
    def __init__(self, x, y, p):
        super().__init__()
        self.image = pygame.Surface(p['shape'])
        self.image.fill(p['color'])  
        self.rect = self.image.get_rect()
        self.rect.topleft = (x, y)  
        self.mask = pygame.mask.from_surface(self.image) 