"""Game projectile object."""

import pygame

class Projectile(pygame.sprite.Sprite):
    """Bullet or similar entity that travels across the screen."""
    def __init__(self, source, target, speed, lifetime, color):
        """Create a projectile starting at source and moving towards target."""
        super().__init__()
        self.image = pygame.Surface([4, 4])
        self.image.set_colorkey(pygame.Color('black'))
        self.rect = self.image.get_rect(x=source[0], y=source[1])
        pygame.draw.circle(self.image, color,
                           (self.rect.width // 2, self.rect.height // 2),
                           self.rect.width // 2)
        
        self.pos = [source[0], source[1]]
        self.movementVector = [target[0], target[1]]
        self.speed = speed
        self.lifetime = lifetime
        self.createdAt = pygame.time.get_ticks()
        self.damagedAt = 0

    def __str__(self):
        """Return string representation used for debugging and dumps."""
        return "Projectile [pos ({0}, {1}); movementVector ({2}, {3}); createdAt {4}; damagedAt {5}] \n".format(
            self.pos[0],
            self.pos[1],
            self.movementVector[0],
            self.movementVector[1],
            self.createdAt,
            self.damagedAt,
        )
        
    def move(self, surfaceSize, tDelta):
        """Update projectile position and destroy it if needed."""
        if pygame.time.get_ticks() > self.createdAt + self.lifetime:
            self.kill()
        self.pos[0] += self.movementVector[0] * self.speed * tDelta
        self.pos[1] += self.movementVector[1] * self.speed * tDelta
        self.rect.topleft = self.pos
        if self.pos[0] > surfaceSize[0] or self.pos[0] < 0  or \
           self.pos[1] > surfaceSize[1] or self.pos[1] < 0:
            self.kill()
    def render(self, surface):
        """Draw projectile on provided surface."""
        surface.blit(self.image, self.pos)
