import os

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from cmaes import CMA

from core import *

from model_cmaes import Model
from Vars import SPEED

ROLLOUT_SIZE = 100
INIT_STD = 0.02
MAX_ITER_COUNT = 10000

def dist(x, y):
    return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)

def norm(x):
    return dist(x, [0, 0])

def normalize(l):
    ret = []

    for x in l:
        if norm(x) < 1e-5:
            ret.append([0, 0])
        else:
            n = norm(x)
            ret.append([x[0] / n, x[1] / n])

    return ret

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

def flatten_them_all(l):
    ret = []
    for x in l:
        ret.append(np.array(x).flatten())
    ret = np.concatenate(ret)

    return ret

def rollout_for_score(model, vec):
    done = False
    hero = pygame.sprite.GroupSingle(Player(screen.get_size()))
    enemies = pygame.sprite.Group()
    lastEnemy = pygame.time.get_ticks()
    score = 0

    start_time = pygame.time.get_ticks()
    cold_start = True
    while hero.sprite.alive and not done:
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
            
            # print('#############################################################')
            # print((str(api.data_Enemies.enemies)))
            enemies_1 = api.data_Enemies.enemies
            # print(enemies_1)
            enemies_poses = [[enemy.pos[0], enemy.pos[1]] for enemy in enemies_1]
            # print(enemies_poses)
            enemies_poses = enemies_poses + [[-100, -100]] * (2 - len(enemies_poses))
            # print(enemies_poses)
            
            enemy_pjs = api.data_Enemy_Projectiles.projectiles
            enemy_pjs_dists = [dist(player_pos, pj.pos) for pj in enemy_pjs]
            enemy_pjs_poses = np.array(enemy_pjs[np.argmin(enemy_pjs_dists)].pos) if len(enemy_pjs) != 0 else np.array([-100, -100])

            #=========================================================

            def foo(x):
                return (np.array(x) - 0.5) * 2

            # def div_by_size(a, b):
            #     return np.array([[x[0] / b[0], x[1] / b[1]]  for x in a])

            movement_vector = normalize([movement_vector])[0]
            player_pos = foo(np.array(player_pos) / np.array((600, 800)))
            health = foo(health / 3)
            shootCooldown = foo(shootCooldown / 250)

            enemies_poses = foo(np.array(enemies_poses) / np.array((600, 800)))
            enemy_pjs_poses = foo(np.array(enemy_pjs_poses) / np.array((600, 800)))

            #=========================================================

            features = flatten_them_all([movement_vector, player_pos, health, shootCooldown, enemies_poses, enemy_pjs_poses])
            #2 + 2 + 1 + 1 + 4 + 2 = 12
            # print('Features:', features)
            # print('Enemies:', api.data_Enemies.enemies)
            # print('Enemy pj:', api.data_Enemy_Projectiles.projectiles)

            #=========================================================

            action_encoded = model(features, vec)

            shoot = np.random.choice(2, p=action_encoded[:2])
            move = np.random.choice(8, p=action_encoded[2:10])
            shoot_dir = np.random.choice(8, p=action_encoded[10:])
        else:
            cold_start = False

        #=========================================================


        # render_entities(hero, enemies)
        
        # # Health and score render
        # for hp in range(hero.sprite.health):
        #     screen.blit(healthRender, (15 + hp*35, 0))
        # scoreRender = scoreFont.render(str(score), True, pygame.Color('black'))
        # scoreRect = scoreRender.get_rect()
        # scoreRect.right = size[0] - 20
        # scoreRect.top = 20
        # screen.blit(scoreRender, scoreRect)
        api.on_update2(hero.sprite)
        # pygame.display.flip()
        clock.tick(120 * SPEED)
    
    # a = abs(pygame.time.get_ticks() - start_time)
    # print(a)
    return abs(pygame.time.get_ticks() - start_time) / 100

def save(vec, fn):
    if fn is None:
        fn = 'checkpoint'

    np.save(fn, vec)

def load(fn):
    if fn is None:
        fn = 'checkpoint'

    return np.load(fn)

def get_last_epoch():
    fns = os.listdir('models')

    if len(fns) == 0:
        return -1

    epochs = [int(x.split('.')[0]) for x in fns]
    return max(epochs)

def slicing_mean(n, data):
    window = np.ones(shape=(n,)) / n

    ret = []
    for i in range(len(data) - n):
        ret.append(np.sum(window * data[i:i+n]))

    return ret

def rollout(vec):
    model = Model()
    ret = 0
    for i in range(ROLLOUT_SIZE):
        ret += rollout_for_score(model, vec)
    return ret / ROLLOUT_SIZE


def train():
    weights_size = 12 * 12 + 12 * 12 + 12 * 18

    last_epoch = get_last_epoch()
    if last_epoch >= 0:
        vec = load(os.path.join('models', str(last_epoch)))
        print('Checkpiont loaded')
    else:
        vec = np.random.randn(weights_size) * INIT_STD
        print('Cold start')

    optimizer = CMA(mean=vec, sigma=1.3, population_size=200)

    for generation in tqdm(range(50)):
        solutions = []
        for _ in tqdm(range(optimizer.population_size)):
            x = optimizer.ask()
            value = rollout(x)
            solutions.append((x, value))
        optimizer.tell(solutions)

        best = sorted(solutions, key=lambda z: z[1])[-1][0]
        save(best, os.path.join('models', str(generation)))

        print(sorted(solutions, key=lambda z: z[1])[-1][1])

def test():
    last_epoch = get_last_epoch()
    vec = load(os.path.join('models', str(last_epoch) + '_checkpoint'))
    print(rollout(vec))

if __name__ == '__main__':
    train()