"""Collection of weapon classes used by the player and enemies."""

import pygame
import math
import random
from Projectile import Projectile

from Vars import SPEED

class Weapon():
    """Base class for weapons."""
    def __init__(self):
        """Initialize cooldown tracking."""
        self.lastShot = 0
    
    def shoot():
        """Placeholder method for subclasses to override."""
        pass
    
    @staticmethod
    def normalize_vector(vector):
        """Return a normalized 2D vector."""
        pythagoras = math.sqrt(vector[0]*vector[0] + vector[1]*vector[1])
        if pythagoras < 1e-5:
            return [0, 0]
        return (vector[0] / pythagoras, vector[1] / pythagoras)
    
    @staticmethod
    def rotate_vector(vector, theta):
        """Rotate a vector by the provided angle in radians."""
        resultVector = (vector[0] * math.cos(theta)
                        - vector[1] * math.sin(theta),
                        vector[0] * math.sin(theta)
                        + vector[1] * math.cos(theta))
        return resultVector

class Pistol(Weapon):
    """Single shot weapon with moderate cooldown."""
    def __init__(self):
        super().__init__()
        self.weaponCooldown = 250 / SPEED
    
    def shoot(self, user, mousePos):
        """Spawn a single projectile towards the cursor."""
        currentTime = pygame.time.get_ticks()
        if currentTime - self.lastShot > self.weaponCooldown:
            direction = (mousePos[0] - user.pos[0], mousePos[1] - user.pos[1]) \
                if mousePos != user.pos else (1, 1)
            self.lastShot = currentTime
            user.projectiles.add(Projectile(user.pos,
                                            super().normalize_vector(direction),
                                            5 * SPEED, 2000, (0, 0, 255)))
    def __str__(self):
        return 'Weapon [lastShot {0}]\n'.format(self.lastShot)
            
class Shotgun(Weapon):
    """Weapon firing a spread of projectiles."""
    def __init__(self):
        super().__init__()
        self.weaponCooldown = 750
        self.spreadArc = 90
        self.projectilesCount = 7
        
    def shoot(self, user, mousePos):
        """Fire multiple projectiles in a cone towards the cursor."""
        currentTime = pygame.time.get_ticks()
        if currentTime - self.lastShot > self.weaponCooldown:
            direction = (mousePos[0] - user.pos[0], mousePos[1] - user.pos[1]) \
                if mousePos != user.pos else (1, 1)
            self.lastShot = currentTime
            arcDifference = self.spreadArc / (self.projectilesCount - 1)
            for proj in range(self.projectilesCount):
                theta = math.radians(arcDifference*proj - self.spreadArc/2)
                projDir = super().rotate_vector(direction, theta)
                user.projectiles.add(Projectile(user.pos,
                                                super().normalize_vector(projDir),
                                                7, 500, (232, 144, 42)))
                
class MachineGun(Weapon):
    """Fast firing weapon with small spread."""
    def __init__(self):
        super().__init__()
        self.weaponCooldown = 100
        self.spreadArc = 25
        
    def shoot(self, user, mousePos):
        """Rapidly spawn projectiles with slight random spread."""
        currentTime = pygame.time.get_ticks()
        if currentTime - self.lastShot > self.weaponCooldown:
            direction = (mousePos[0] - user.pos[0], mousePos[1] - user.pos[1]) \
                if mousePos != user.pos else (1, 1)
            self.lastShot = currentTime
            theta = math.radians(random.random()*self.spreadArc - self.spreadArc/2)
            projDir = super().rotate_vector(direction, theta)   
            user.projectiles.add(Projectile(user.pos,
                                            super().normalize_vector(projDir),
                                            6, 1000, (194, 54, 16)))
