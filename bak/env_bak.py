from hashlib import new
import os

import numpy as np

from core import *
from Vars import *
from utils import *

import gym

def i_to_dir(i):
    if i == 0:
        return np.array([1, 0])
    elif i == 1:
        return np.array([1, 1]) / np.sqrt(2)
    elif i == 2:
        return np.array([0, 1])
    elif i == 3:
        return np.array([-1, 1]) / np.sqrt(2)
    elif i == 4:
        return np.array([-1, 0])
    elif i == 5:
        return np.array([-1, -1]) / np.sqrt(2)
    elif i == 6:
        return np.array([0, -1])
    elif i == 7:
        return np.array([1, -1]) / np.sqrt(2)
    else:
        print('Error')
        quit(1)

class Env(gym.Env):
    def __init__(self, _):
        self.observation_space = gym.spaces.Box(
            low = -np.ones((IN_DIM,), dtype = np.float32) - 0.01,
            high = np.ones((IN_DIM,), dtype = np.float32) + 0.01,
            shape = (IN_DIM,),
            dtype = np.float32,
        )
        self.action_space = gym.spaces.Discrete(128)

        self.reset()

    def get_state(self):
        movement_vector = api.data_Player.movementVector
        player_pos = api.data_Player.pos
        health = api.data_Player.health
        shootCooldown = max(0, 250 - (pygame.time.get_ticks() - api.data_Shoot.lastShot))
    
        enemies_1 = api.data_Enemies.enemies
        enemies_poses = [[enemy.pos[0], enemy.pos[1]] for enemy in enemies_1]
        enemies_poses = enemies_poses + [[0, 0]] * (2 - len(enemies_poses))
        
        enemy_pjs = api.data_Enemy_Projectiles.projectiles
        enemy_pjs_dists = [dist(player_pos, pj.pos) for pj in enemy_pjs]
        enemy_pjs_poses = np.array(enemy_pjs[np.argmin(enemy_pjs_dists)].pos) if len(enemy_pjs) != 0 else np.array([0, 0])

        def foo(x):
            return (np.array(x) - 0.5) * 2

        movement_vector = normalize([movement_vector])[0]
        player_pos = foo(np.array(player_pos) / np.array((600, 800)))
        health = foo(health / 3)
        shootCooldown = foo(shootCooldown / 250)

        enemies_poses = foo(np.array(enemies_poses) / np.array((600, 800)))
        enemy_pjs_poses = foo(np.array(enemy_pjs_poses) / np.array((600, 800)))

        features = flatten_them_all([movement_vector, player_pos, health, shootCooldown, enemies_poses, enemy_pjs_poses])
        
        return features


    def reset(self):
        self.hero = pygame.sprite.GroupSingle(Player(screen.get_size()))
        self.enemies = pygame.sprite.Group()
        self.lastEnemy = pygame.time.get_ticks()
        self.cum_reward = 0

        return np.ones(shape=(IN_DIM,), dtype=np.float32) #self.get_state()
        
    def step(self, action):

        action = dirac_delta(int(action), 128)
        action = np.reshape(action, newshape=(2, 8, 8))
        chosen = np.unravel_index(np.argmax(action, axis=None), action.shape)

        api.input.shoot = True if chosen[0] == 1 else False
        api.input.move = i_to_dir(chosen[1])
        api.input.shoot_dir = i_to_dir(chosen[2])

        api.on_update1(self.hero.sprite) 
        process_api(api, self.hero)
        api.on_update2(hero.sprite)

        if self.lastEnemy < pygame.time.get_ticks() - 10 and len(self.enemies) < 2:
            spawnSide = random.random()
            if spawnSide < 0.25:
                self.enemies.add(Enemy((0, random.randint(0, size[1]))))
            elif spawnSide < 0.5:
                self.enemies.add(Enemy((size[0], random.randint(0, size[1]))))
            elif spawnSide < 0.75:
                self.enemies.add(Enemy((random.randint(0, size[0]), 0)))
            else:
                self.enemies.add(Enemy((random.randint(0, size[0]), size[1])))
            self.lastEnemy = pygame.time.get_ticks()

        self.done = self.hero.sprite.alive
        reward = -10 if not self.done else api.data_Player.health / 3

        self.cum_reward += reward

        clock.tick(120 * SPEED)

        return  self.get_state(), \
                reward, \
                self.done, \
                dict()

    def render(self):
        print(self.cum_reward)