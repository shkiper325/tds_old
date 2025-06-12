"""Enemy entity that chases the player and shoots."""

import pygame
import math
from Projectile import Projectile

from Vars import SPEED

def normalize_vector(vector):
    """Normalize a vector returning a list with length 1."""
    if vector == [0, 0]:
        return [0, 0]
    pythagoras = math.sqrt(vector[0]*vector[0] + vector[1]*vector[1])
    return (vector[0] / pythagoras, vector[1] / pythagoras)

class Enemy(pygame.sprite.Sprite):
    """AI controlled opponent."""

    # Projectiles fired by all enemies
    projectiles = pygame.sprite.Group()

    def __init__(self, pos):
        """Spawn enemy at a given position."""
        super().__init__()
        self.image = pygame.Surface([8, 8])
        self.image.fill(pygame.Color('black'))
        self.rect = self.image.get_rect(x=pos[0], y=pos[1])
        self.radius = self.rect.width / 2
        
        self.pos = list(pos)
        self.movementVector = [0, 0]
        self.movementVector2 = [0, 0]
        self.movementSpeed = 3 * SPEED
        self.lastShot = pygame.time.get_ticks()
        self.weaponCooldown = 250 / SPEED
        self.timestamp_start=pygame.time.get_ticks()
        self.timestamp_end=0
    def move(self, enemies, playerPos, tDelta):
        """Move towards the player and avoid overlapping with other enemies."""
        self.movementVector = (playerPos[0] - self.pos[0],
                               playerPos[1] - self.pos[1])
        self.movementVector = normalize_vector(self.movementVector)
        self.pos[0] += self.movementVector[0] * self.movementSpeed * tDelta
        self.pos[1] += self.movementVector[1] * self.movementSpeed * tDelta
        self.movementVector2 = self.movementVector
        
        # Collision test with other enemies
        self.movementVector = [0, 0]
        for sprite in enemies:
            if sprite is self:
                continue
            if pygame.sprite.collide_circle(self, sprite):
                self.movementVector[0] += self.pos[0] - sprite.pos[0]
                self.movementVector[1] += self.pos[1] - sprite.pos[1]

        self.movementVector = normalize_vector(self.movementVector)
        self.pos[0] += self.movementVector[0] * 0.5  # The constant is how far the sprite will be
        self.pos[1] += self.movementVector[1] * 0.5  # dragged from the sprite it collided with
        
        self.rect.topleft = self.pos
        self.timestamp_end=pygame.time.get_ticks()

    def shoot(self, playerPos):
        """Fire a projectile towards the player if cooldown allows."""
        currentTime = pygame.time.get_ticks()
        if currentTime - self.lastShot > self.weaponCooldown:
            direction = (playerPos[0] - self.pos[0], playerPos[1] - self.pos[1])
            if (direction[0]**2+direction[1]**2) < 62500:
                self.lastShot = currentTime
                self.projectiles.add(Projectile(self.pos,
                                            normalize_vector(direction),
                                            5 * SPEED, 2000, (255, 0, 0)))
    def __str__(self):
        """Return printable representation used in logs."""
        return (
            "Enemy[pos ({0}, {1}); movementVector ({2}, {3}); lastShot {4}; timestamp_start {5}; timestamp_end {6}]\n".format(
                self.pos[0],
                self.pos[1],
                self.movementVector2[0],
                self.movementVector2[1],
                self.lastShot,
                self.timestamp_start,
                self.timestamp_end,
            )
        )

    def get_pos(self):
        """Return current position of the enemy."""
        return self.pos

    def render(self, surface):
        """Draw enemy on the given surface."""
        surface.blit(self.image, self.pos)
