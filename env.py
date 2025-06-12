import numpy as np

from core import *
from Vars import *
from utils import *

import gym

def i_to_dir_8(i):
    if i == 0:
        return (1, 0)
    elif i == 1:
        return (1, 1)
    elif i == 2:
        return (0, 1)
    elif i == 3:
        return (-1, 1)
    elif i == 4:
        return (-1, 0)
    elif i == 5:
        return (-1, -1)
    elif i == 6:
        return (0, -1)
    elif i == 7:
        return (1, -1)
    else:
        print('Error')
        quit(1)

def i_to_dir_4(i):
    if i == 0:
        return (1, 0)
    elif i == 1:
        return (0, 1)
    elif i == 2:
        return (-1, 0)
    elif i == 3:
        return (0, -1)
    else:
        print('Error')
        quit(1)

class Env(gym.Env):
    def __init__(self, abc=None):
        self.observation_space = gym.spaces.Box(
            low = -np.ones((IN_DIM,), dtype = np.float32) - 0.01,
            high = np.ones((IN_DIM,), dtype = np.float32) + 0.01,
            shape = (IN_DIM,),
            dtype = np.float32,
        )
        self.action_space = gym.spaces.Discrete(32)

        self.reset()

    def get_state(self):
        global api

        H = 600
        W = 800

        def foo(x):
            return (np.array(x) - 0.5) * 2

        movement_vector = api.data_Player.movementVector
        movement_vector = normalize([movement_vector])[0]

        player_x, player_y = api.data_Player.pos
        player_x /= W
        player_y /= H
        player_x, player_y = foo(player_x), foo(player_y)
        player_pos = [player_x, player_y]

        health = foo(api.data_Player.health / 3)

        shootCooldown = max(0, 250 - (pygame.time.get_ticks() - api.data_Shoot.lastShot)) #TODO: fix values
        shootCooldown = foo(shootCooldown / 250)
    
        enemies = api.data_Enemies.enemies
        enemies_poses = [[enemy.pos[0], enemy.pos[1]] for enemy in enemies]
        enemies_dists = [dist(enemy_pos, player_pos) for enemy_pos in enemies_poses] + [-1000] * (2 - len(enemies))
        enemies_angles = [angle(player_pos, enemy_pos) for enemy_pos in enemies_poses] + [-2 * PI] * (2 - len(enemies))
        enemies_dists = np.array(enemies_dists, dtype=np.float32) / 1000
        enemies_angles = np.array(enemies_angles) / 2 / PI
        
        enemy_pjs = api.data_Enemy_Projectiles.projectiles
        enemy_pjs = sorted(enemy_pjs, key = lambda pj: dist(pj.pos, player_pos))
        if len(enemy_pjs) > 5:
            enemy_pjs = enemy_pjs[:5]

        enemy_pjs_dists = [dist(pj.pos, player_pos) for pj in enemy_pjs] + [-1000] * (5 - len(enemy_pjs))
        enemy_pjs_angles = [angle(player_pos, pj.pos) for pj in enemy_pjs] + [-2 * PI] * (5 - len(enemy_pjs))
        enemy_pjs_dists = np.array(enemy_pjs_dists) / 1000
        enemy_pjs_angles = np.array(enemy_pjs_angles) / 2 / PI
        enemy_pjs_dirs = np.array(normalize([pj.movementVector for pj in enemy_pjs]) + [np.array([0, 0])] * (5 - len(enemy_pjs)))

        features = flatten_them_all([
            movement_vector,
            player_pos,
            health,
            shootCooldown,
            enemies_dists,
            enemies_angles,
            enemy_pjs_dists,
            enemy_pjs_angles,
            enemy_pjs_dirs])
        
        return features

    def reset(self):
        global hero, enemies

        hero = pygame.sprite.GroupSingle(Player(screen.get_size()))
        enemies = pygame.sprite.Group()
        api.on_restart()

        self.lastEnemy = pygame.time.get_ticks()
        self.cum_reward = 0
        self.steps = 0
        self.frags = 0

        return self.get_state() #self.get_state()
        
    def step(self, action):
        global hero
        global done
        global enemies

        start_health = api.data_Player.health

        keys = pygame.key.get_pressed()
        mouse = pygame.mouse.get_pressed()

        action = dirac_delta(int(action), 32)
        action = np.reshape(action, newshape=(2, 4, 4))
        chosen = np.unravel_index(np.argmax(action, axis=None), action.shape)

        api.input.shoot = True if chosen[0] == 1 else False
        api.input.move = i_to_dir_4(chosen[1])
        api.input.shoot_dir = i_to_dir_4(chosen[2])

        process_keys(keys, hero)
        process_mouse(mouse, hero)

        if self.lastEnemy < pygame.time.get_ticks() - 10 and len(enemies) < 2:
            spawnSide = random.random()
            if spawnSide < 0.25:
                enemies.add(Enemy((0, random.randint(0, size[1]))))
            elif spawnSide < 0.5:
                enemies.add(Enemy((size[0], random.randint(0, size[1]))))
            elif spawnSide < 0.75:
                enemies.add(Enemy((random.randint(0, size[0]), 0)))
            else:
                enemies.add(Enemy((random.randint(0, size[0]), size[1])))
            self.lastEnemy = pygame.time.get_ticks()

        move_entities(hero, enemies, clock.get_time()/17)

        api.on_update1(hero.sprite) 
        process_api(api, hero)

        clock.tick(120 * SPEED)

        state = self.get_state()
        api.on_update2(hero.sprite)
        
        killed = False
        if len(bot_killed_projectiles) > 0:
            killed = abs(bot_killed_projectiles[-1].damagedAt - pygame.time.get_ticks()) <= 5

        self.steps += 1
        
        done_1 = done or (not hero.sprite.alive) or (self.steps > 2000)
        if hero.sprite.alive:
            reward = 1
        else:
            reward = -10

        if killed:
            reward += 10
            self.frags += 1

        if api.data_Player.health != start_health:
            reward = -10

        self.cum_reward += reward

        return  state, \
                reward, \
                done_1, \
                dict()

    def render(self, mode=None):
        pass
        #print('Frags:', self.frags, '|', 'Score:', self.cum_reward)