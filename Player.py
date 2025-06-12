"""Player entity definition and movement logic."""

import pygame
import math
import Weapon

from Vars import SPEED

PLAYERCOLOR = (255,   0,   0)

def normalize_vector(vector):
    """Return a normalized copy of a 2D vector."""
    pythagoras = math.sqrt(vector[0]*vector[0] + vector[1]*vector[1])
    if pythagoras < 1e-5:
        return [0, 0]
    return (vector[0] / pythagoras, vector[1] / pythagoras)

class Player(pygame.sprite.Sprite):
    """Main controllable character."""

    # Group used to store projectiles spawned by all players
    projectiles = pygame.sprite.Group()

    def __init__(self, screenSize):
        """Create a player centered on the screen."""
        super().__init__()
        self.image = pygame.Surface([8, 8])
        self.image.fill(PLAYERCOLOR)
        self.rect = self.image.get_rect(x=screenSize[0]//2,
                                        y=screenSize[1]//2)
        
        self.pos = [screenSize[0] // 2, screenSize[1] // 2]
        self.health = 3
        self.alive = True
        self.movementVector = [0, 0]
        self.movementVector2 = [0, 0]
        self.movementSpeed = 3 * SPEED
        self.availableWeapons = [Weapon.Pistol(),
                                 Weapon.Shotgun(),
                                 Weapon.MachineGun()]
        self.equippedWeapon = self.availableWeapons[0]

    def move(self, screenSize, tDelta):
        """Move the player according to current movement vector."""
        self.movementVector = normalize_vector(self.movementVector)
        newPos = (self.pos[0] + self.movementVector[0]*self.movementSpeed*tDelta,
                  self.pos[1] + self.movementVector[1]*self.movementSpeed*tDelta)
        if newPos[0] < 0:
            self.pos[0] = 0
        elif newPos[0] > screenSize[0] - self.rect.width:
            self.pos[0] = screenSize[0] - self.rect.width
        else:
            self.pos[0] = newPos[0]

        if newPos[1] < 0:
            self.pos[1] = 0
        elif newPos[1] > screenSize[1]-self.rect.height:
            self.pos[1] = screenSize[1]-self.rect.width
        else:
            self.pos[1] = newPos[1]
        
        self.rect.topleft = self.pos
        self.movementVector2 = self.movementVector
        self.movementVector = [0, 0]
        
    def shoot(self, mousePos):
        """Fire currently equipped weapon towards the given position."""
        self.equippedWeapon.shoot(self, mousePos)

    def __str__(self):
        """Return a string representation useful for logging."""
        return "Player [pos ({0}, {1}); movementVector ({2}, {3}); health {4}; alive {5}]\n".format(
            self.pos[0],
            self.pos[1],
            self.movementVector2[0],
            self.movementVector2[1],
            self.health,
            self.alive,
        )

    def get_pos(self):
        """Return current player position."""
        return self.pos
        
    def render(self, surface):
        """Draw player sprite on the provided surface."""
        surface.blit(self.image, self.pos)
