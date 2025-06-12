import os

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

import torch
import torch.optim as optim

from core import *
from model_gpu import Model
from Vars import *
from utils import *

MAX_ITER_COUNT = 10000
LR = 0.001
ROLLOUT_PER_EPOCH = 50
EPOCH_COUNT = 2000

def rollout_for_score(model):
    hero = pygame.sprite.GroupSingle(Player(screen.get_size()))
    enemies = pygame.sprite.Group()
    lastEnemy = pygame.time.get_ticks()
    score = 0
    pt_ret = 0

    start_time = pygame.time.get_ticks()
    cold_start = True
    while hero.sprite.alive:
        currentTime = pygame.time.get_ticks()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
        screen.fill(BGCOLOR)

        #=========================================================

        shoot = False
        move = 0
        shoot_dir = 0

        #=========================================================
        api.input.shoot = True if shoot == 1 else False
        api.input.move = i_to_dir(move)
        api.input.shoot_dir = i_to_dir(shoot_dir)
        if lastEnemy < currentTime - 10 and len(enemies) < 2:
            spawnSide = random.random()
            if spawnSide < 0.25:
                enemies.add(Enemy((0, random.randint(0, size[1]))))
            elif spawnSide < 0.5:
                enemies.add(Enemy((size[0], random.randint(0, size[1]))))
            elif spawnSide < 0.75:
                enemies.add(Enemy((random.randint(0, size[0]), 0)))
            else:
                enemies.add(Enemy((random.randint(0, size[0]), size[1])))
            lastEnemy = currentTime

        api.on_update1(hero.sprite) 
        process_api(api, hero)

        #=========================================================

        # Enemy spawning process
        
        
        score += move_entities(hero, enemies, clock.get_time()/17)

        if not cold_start:
            movement_vector = api.data_Player.movementVector
            player_pos = api.data_Player.pos
            health = api.data_Player.health
            shootCooldown = max(0, 250 - (pygame.time.get_ticks() - api.data_Shoot.lastShot))
        
            enemies_1 = api.data_Enemies.enemies
            enemies_poses = [[enemy.pos[0], enemy.pos[1]] for enemy in enemies_1]
            enemies_poses = enemies_poses + [[-100, -100]] * (2 - len(enemies_poses))
            
            enemy_pjs = api.data_Enemy_Projectiles.projectiles
            enemy_pjs_dists = [dist(player_pos, pj.pos) for pj in enemy_pjs]
            enemy_pjs_poses = np.array(enemy_pjs[np.argmin(enemy_pjs_dists)].pos) if len(enemy_pjs) != 0 else np.array([-100, -100])

            #=========================================================

            def foo(x):
                return (np.array(x) - 0.5) * 2

            movement_vector = normalize([movement_vector])[0]
            player_pos = foo(np.array(player_pos) / np.array((600, 800)))
            health = foo(health / 3)
            shootCooldown = foo(shootCooldown / 250)

            enemies_poses = foo(np.array(enemies_poses) / np.array((600, 800)))
            enemy_pjs_poses = foo(np.array(enemy_pjs_poses) / np.array((600, 800)))

            #=========================================================

            features = flatten_them_all([movement_vector, player_pos, health, shootCooldown, enemies_poses, enemy_pjs_poses])
            #2 + 2 + 1 + 1 + 4 + 2 = 12

            #=========================================================

            action_torch = model(FloatTensor(np.expand_dims(features, 0)))
            action = action_torch.detach().cpu().numpy()[0]

            shoot = np.random.choice(2, p=action[:2])
            move = np.random.choice(8, p=action[2:10])
            shoot_dir = np.random.choice(8, p=action[10:])
            
            pt_ret = pt_ret + torch.log(action_torch[0, shoot] + 1e-5) + torch.log(action_torch[0, move + 2] + 1e-5) + torch.log(action_torch[0, shoot_dir + 10] + 1e-5)
        else:
            cold_start = False

        #=========================================================


        render_entities(hero, enemies)
        
        # Health and score render
        for hp in range(hero.sprite.health):
            screen.blit(healthRender, (15 + hp*35, 0))
        scoreRender = scoreFont.render(str(score), True, pygame.Color('black'))
        scoreRect = scoreRender.get_rect()
        scoreRect.right = size[0] - 20
        scoreRect.top = 20
        screen.blit(scoreRender, scoreRect)
        api.on_update2(hero.sprite)
        pygame.display.flip()
        clock.tick(120 * SPEED)

    return pt_ret, abs(pygame.time.get_ticks() - start_time) / 100

def main():
    pi = Model()
    optimizer = optim.Adam(pi.parameters(), lr=LR)

    last_epoch = get_last_epoch()
    if last_epoch >= 0:
        d = load(os.path.join('models', str(last_epoch)))
        pi.load_state_dict(d['pi'])
        optimizer.load_state_dict(d['opt'])
        print('Checkpiont loaded')
    else:
        print('Cold start')
        pi.apply(init_weights)

    rollout_for_score(pi)

if __name__ == '__main__':
    main()