"""Core game entities with optimized implementation."""

import pygame
import numpy as np
import math
import random
from utils import normalize_vector, distance, rotate_vector, SPEED

class Projectile(pygame.sprite.Sprite):
    """Optimized projectile class."""
    
    def __init__(self, pos, direction, speed, lifetime, color, owner=None):
        super().__init__()
        self.image = pygame.Surface([4, 4])
        self.image.set_colorkey(pygame.Color('black'))
        self.rect = self.image.get_rect(center=pos)
        pygame.draw.circle(self.image, color, (2, 2), 2)
        
        self.pos = np.array(pos, dtype=float)
        self.velocity = normalize_vector(direction) * speed
        self.lifetime = lifetime
        self.created_at = pygame.time.get_ticks()
        self.owner = owner
    
    def update(self, dt, screen_size):
        """Update projectile position and check bounds."""
        if pygame.time.get_ticks() - self.created_at > self.lifetime:
            self.kill()
            return
        
        self.pos += self.velocity * dt
        self.rect.center = self.pos
        
        # Kill if out of bounds
        if (self.pos[0] < 0 or self.pos[0] > screen_size[0] or 
            self.pos[1] < 0 or self.pos[1] > screen_size[1]):
            self.kill()
    
    def render(self, surface):
        surface.blit(self.image, self.rect)

class Weapon:
    """Base weapon class with common functionality."""
    
    def __init__(self, cooldown, projectile_speed=300, projectile_lifetime=2000, color=(0, 0, 255)):
        self.cooldown = cooldown / SPEED  # Faster cooldown with higher SPEED
        self.last_shot = 0
        self.projectile_speed = projectile_speed * SPEED  # Faster projectiles with higher SPEED
        self.projectile_lifetime = projectile_lifetime / SPEED  # Shorter lifetime to keep range similar
        self.color = color
    
    def can_shoot(self):
        return pygame.time.get_ticks() - self.last_shot >= self.cooldown
    
    def shoot(self, owner, target_pos, projectiles_group):
        """Base shooting method - override in subclasses."""
        if not self.can_shoot():
            return False
        
        self.last_shot = pygame.time.get_ticks()
        direction = np.array(target_pos) - np.array(owner.pos)
        
        projectile = Projectile(
            owner.pos.copy(), 
            direction, 
            self.projectile_speed, 
            self.projectile_lifetime, 
            self.color,
            owner
        )
        projectiles_group.add(projectile)
        return True

class Pistol(Weapon):
    def __init__(self):
        super().__init__(cooldown=250, color=(0, 0, 255))

class Shotgun(Weapon):
    def __init__(self):
        super().__init__(cooldown=750, color=(232, 144, 42))
    
    def shoot(self, owner, target_pos, projectiles_group):
        if not self.can_shoot():
            return False
        
        self.last_shot = pygame.time.get_ticks()
        direction = normalize_vector(np.array(target_pos) - np.array(owner.pos))
        
        # Fire 7 projectiles in a spread
        spread_angles = np.linspace(-math.pi/6, math.pi/6, 7)
        for angle in spread_angles:
            proj_dir = rotate_vector(direction, angle)
            projectile = Projectile(
                owner.pos.copy(), 
                proj_dir, 
                self.projectile_speed, 
                500, 
                self.color,
                owner
            )
            projectiles_group.add(projectile)
        return True

class MachineGun(Weapon):
    def __init__(self):
        super().__init__(cooldown=100, color=(194, 54, 16))
    
    def shoot(self, owner, target_pos, projectiles_group):
        if not self.can_shoot():
            return False
        
        self.last_shot = pygame.time.get_ticks()
        direction = normalize_vector(np.array(target_pos) - np.array(owner.pos))
        
        # Add random spread
        spread_angle = (random.random() - 0.5) * math.pi/6
        proj_dir = rotate_vector(direction, spread_angle)
        
        projectile = Projectile(
            owner.pos.copy(), 
            proj_dir, 
            self.projectile_speed, 
            1000, 
            self.color,
            owner
        )
        projectiles_group.add(projectile)
        return True

class Player(pygame.sprite.Sprite):
    """Optimized player class."""
    
    def __init__(self, screen_size, color=(255, 0, 0), size=100):
        super().__init__()
        self.image = pygame.Surface([size, size])
        self.image.fill(color)
        self.rect = self.image.get_rect(center=(screen_size[0]//2, screen_size[1]//2))
        
        self.pos = np.array([screen_size[0] // 2, screen_size[1] // 2], dtype=float)
        self.velocity = np.array([0.0, 0.0])
        self.max_speed = 150.0 * SPEED  # pixels per second, scaled by SPEED
        self.health = 3
        self.max_health = 3
        self.alive = True
        
        # Weapons
        self.weapons = [Pistol(), Shotgun(), MachineGun()]
        self.current_weapon = 0
        
        self.screen_size = screen_size
    
    def update(self, dt, move_action, projectiles_group):
        """Update player position and handle movement."""
        # Apply movement
        self.velocity = normalize_vector(move_action) * self.max_speed
        new_pos = self.pos + self.velocity * dt
        
        # Boundary checking
        half_size = self.rect.width // 2
        new_pos[0] = np.clip(new_pos[0], half_size, self.screen_size[0] - half_size)
        new_pos[1] = np.clip(new_pos[1], half_size, self.screen_size[1] - half_size)
        
        self.pos = new_pos
        self.rect.center = self.pos
    
    def shoot(self, target_pos, projectiles_group):
        """Shoot current weapon."""
        return self.weapons[self.current_weapon].shoot(self, target_pos, projectiles_group)
    
    def take_damage(self, amount=1):
        """Take damage and check if dead."""
        self.health -= amount
        if self.health <= 0:
            self.alive = False
    
    def heal(self, amount=1):
        """Heal the player."""
        self.health = min(self.max_health, self.health + amount)
    
    def render(self, surface):
        surface.blit(self.image, self.rect)
