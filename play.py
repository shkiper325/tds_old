import pygame
from Player import Player


def process_keys(keys, player):
    if keys[pygame.K_w]:
        player.movementVector[1] -= 1
    if keys[pygame.K_a]:
        player.movementVector[0] -= 1
    if keys[pygame.K_s]:
        player.movementVector[1] += 1
    if keys[pygame.K_d]:
        player.movementVector[0] += 1


if __name__ == "__main__":
    pygame.init()
    size = (800, 600)
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Two Player Test")

    clock = pygame.time.Clock()

    player1 = Player(size)
    player2 = Player(size)

    # Increase sprite size for both players
    bigger = 16
    for p in (player1, player2):
        p.image = pygame.Surface([bigger, bigger])
        p.image.fill((255, 0, 0))
        p.rect = p.image.get_rect(topleft=p.pos)
        # Slow down movement speed
        p.movementSpeed *= 0.5

    player2.pos = [size[0] // 4, size[1] // 2]
    player2.rect.topleft = player2.pos

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        mouse = pygame.mouse.get_pressed()

        process_keys(keys, player1)
        if mouse[0]:
            player1.shoot(pygame.mouse.get_pos())

        dt = clock.get_time() / 17
        player1.move(size, dt)
        player2.move(size, dt)  # stays still because movementVector is zero

        for proj in list(Player.projectiles):
            proj.move(size, dt)
            if proj.rect.colliderect(player2.rect):
                proj.kill()

        screen.fill((255, 255, 255))
        player1.render(screen)
        player2.render(screen)
        for proj in Player.projectiles:
            proj.render(screen)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
