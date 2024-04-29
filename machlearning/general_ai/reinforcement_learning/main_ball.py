import numpy as np
import pygame
from push_box import constants as c
from push_box import objs

def init_obj():
    player = objs.Block(c.GREEN,50,50)

def pooling_min(screenshot):
    pooling_size = (20, 20)  
    stride = (20, 20)  

    output_height = (screenshot.shape[0] - pooling_size[0]) // stride[0] + 1
    output_width = (screenshot.shape[1] - pooling_size[1]) // stride[1] + 1

    output = np.zeros((output_height, output_width, screenshot.shape[2]), dtype=np.uint8)

    for i in range(output_height):
        for j in range(output_width):
            start_h = i * stride[0]
            start_w = j * stride[1]
            end_h = start_h + pooling_size[0]
            end_w = start_w + pooling_size[1]
            output[i, j] = np.min(screenshot[start_h:end_h, start_w:end_w], axis=(0, 1))

    return output


def main():
    # pygame setup
    pygame.init()
    screen = pygame.display.set_mode((400, 300))
    clock = pygame.time.Clock()
    running = True
    dt = 0

    player = objs.Block(100,200,c.PLAYER_PROPERTYS)
    # enemys = [objs.Block(x*100,100,c.FOOD_PROPERTYS) for x in range(3)]
    all_sprites = pygame.sprite.Group()
    all_sprites.add(player,*[objs.Block(x*100,100,c.FOOD_PROPERTYS) for x in range(3)])

    x,y = 100,200
    while running:
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # fill the screen with a color to wipe away anything from last frame
        screen.fill("white")

        speed = 100*(player.speed/10)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            y = y-speed * dt if y>0 else 0
        if keys[pygame.K_s]:
            y = y+speed * dt if y<screen.get_height()-c.PLAYER_HEIGHT else screen.get_height()-c.PLAYER_HEIGHT 
        if keys[pygame.K_a]:
            x = x-speed * dt if x>0 else 0
        if keys[pygame.K_d]:
            x = x+speed * dt if x<screen.get_width()-c.PLAYER_WIDTH else screen.get_width()-c.PLAYER_WIDTH 

        player.update(x,y)
        player.energy -= 1
        if player.energy%200==0:
            print(player.energy)

        for e in all_sprites:
            if e!=player and pygame.sprite.collide_rect(player, e):
                player.speed += 3
                e.kill()
                player.energy += 1000
        all_sprites.draw(screen)
        pygame.display.flip()
        screenshot = pygame.surfarray.array3d(pygame.display.get_surface())
        # limits FPS to 60
        # dt is delta time in seconds since last frame, used for framerate-
        # independent physics.
        dt = clock.tick(60) / 1000

    pygame.quit()


if __name__=='__main__':
    main()


