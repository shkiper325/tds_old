# -*- coding: utf-8 -*-

import pygame
import random
import sys, time
from Player import Player
from Enemy import Enemy
from Projectile import Projectile
from Api import Api, create_json, bot_killed_projectiles
SPEED = 50.

pygame.init()
param1 = ''#sys.argv[1]
param2 = ''#sys.argv[1]
param3 = ''#sys.argv[1]
param4 = ''#sys.argv[1]
create_json(param1, param2, param3, param4)
size    = (800, 600)
BGCOLOR = (255, 255, 255)
screen = pygame.display.set_mode(size, pygame.HIDDEN)
scoreFont = pygame.font.Font("fonts/UpheavalPro.ttf", 30)
healthFont = pygame.font.Font("fonts/OmnicSans.ttf", 50)
healthRender = healthFont.render('z', True, pygame.Color('red'))
pygame.display.set_caption("Top Down")

done = False
hero = pygame.sprite.GroupSingle(Player(screen.get_size()))
enemies = pygame.sprite.Group()
global api
api = Api()
lastEnemy = 0
score = 0
clock = pygame.time.Clock()

def move_entities(hero, enemies, timeDelta):
    score = 0
    hero.sprite.move(screen.get_size(), timeDelta)
    for enemy in enemies:
        enemy.move(enemies, ((hero.sprite.rect.topleft[0] + random.randint(0, 2000)-1000), (hero.sprite.rect.topleft[1] + random.randint(0, 2000)-1000)), timeDelta)
        enemy.shoot(hero.sprite.rect.topleft)
        if (enemy.get_pos()[0]-hero.sprite.get_pos()[0])**2+(enemy.get_pos()[1]-hero.sprite.get_pos()[1])**2 < 62500:
            api.data_Enemies.add_enemy(api.data_Enemies, enemy)
    for proj in Enemy.projectiles:
        proj.move(screen.get_size(), timeDelta)
        if (proj.pos[0]-hero.sprite.get_pos()[0])**2+(proj.pos[1]-hero.sprite.get_pos()[1])**2 < 62500:
            api.data_Enemy_Projectiles.add_enemy_proj(api.data_Enemy_Projectiles, proj)
        if pygame.sprite.spritecollide(proj, hero, False):
            proj.damagedAt = pygame.time.get_ticks()
            api.on_player_damage(proj)
            proj.kill()
            hero.sprite.health -= 1
            if hero.sprite.health <= 0:
                hero.sprite.alive = False
                api.on_restart()
    for proj in Player.projectiles:
        proj.move(screen.get_size(), timeDelta)
        if (proj.pos[0]-hero.sprite.get_pos()[0])**2+(proj.pos[1]-hero.sprite.get_pos()[1])**2 < 62500:
            api.data_Player_Projectiles.add_player_proj(api.data_Player_Projectiles, proj)
        enemiesHit = pygame.sprite.spritecollide(proj, enemies, True)
        if enemiesHit:
            proj.damagedAt = pygame.time.get_ticks()
            api.on_bot_kill(proj)
            proj.kill()
            score += len(enemiesHit)
    return score

def render_entities(hero, enemies):
    hero.sprite.render(screen)
    for proj in Player.projectiles:
        proj.render(screen)
    for proj in Enemy.projectiles:
        proj.render(screen)
    for enemy in enemies:
        enemy.render(screen)
    
def process_keys(keys, hero):
    if keys[pygame.K_w]:
        hero.sprite.movementVector[1] -= 1
    if keys[pygame.K_a]:
        hero.sprite.movementVector[0] -= 1
    if keys[pygame.K_s]:
        hero.sprite.movementVector[1] += 1
    if keys[pygame.K_d]:
        hero.sprite.movementVector[0] += 1
    if keys[pygame.K_1]:
        hero.sprite.equippedWeapon = hero.sprite.availableWeapons[0]
    if keys[pygame.K_2]:
        hero.sprite.equippedWeapon = hero.sprite.availableWeapons[1]
    if keys[pygame.K_3]:
        hero.sprite.equippedWeapon = hero.sprite.availableWeapons[2]
        
def process_mouse(mouse, hero):
    if mouse[0]:
        hero.sprite.shoot(pygame.mouse.get_pos())
        api.data_Shoot.update_data_Shoot(api.data_Shoot, hero.sprite.equippedWeapon)


def process_api(api, hero):
    hero.sprite.movementVector[0] = api.input.move[0] #-1, 1 double
    hero.sprite.movementVector[1] = api.input.move[1] #-1, 1 double
    if api.input.shoot:
        pygame.mouse.set_pos(api.input.shoot_dir[0], api.input.shoot_dir[1])
        hero.sprite.shoot(pygame.mouse.get_pos())
        api.data_Shoot.update_data_Shoot(api.data_Shoot, hero.sprite.equippedWeapon)

def game_loop():
    done = False
    hero = pygame.sprite.GroupSingle(Player(screen.get_size()))
    enemies = pygame.sprite.Group()
    lastEnemy = pygame.time.get_ticks()
    score = 0
    
    while hero.sprite.alive and not done:
        keys = pygame.key.get_pressed()
        mouse = pygame.mouse.get_pressed()
        currentTime = pygame.time.get_ticks()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
        screen.fill(BGCOLOR)
        
        process_keys(keys, hero)
        process_mouse(mouse, hero)
        
        # Enemy spawning process
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
        
        score += move_entities(hero, enemies, clock.get_time()/17)
        #render_entities(hero, enemies)
        
        # Health and score render
        #for hp in range(hero.sprite.health):
        #    screen.blit(healthRender, (15 + hp*35, 0))
        #scoreRender = scoreFont.render(str(score), True, pygame.Color('black'))
        #scoreRect = scoreRender.get_rect()
        #scoreRect.right = size[0] - 20
        #scoreRect.top = 20
        #screen.blit(scoreRender, scoreRect)
        api.on_update(hero.sprite)
        #pygame.display.flip()
        clock.tick(120)

if __name__ == '__main__':
    done = game_loop()
    while not done:
        keys = pygame.key.get_pressed()
        mouse = pygame.mouse.get_pressed()
        currentTime = pygame.time.get_ticks()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        if keys[pygame.K_r] or api.input.restart:
            create_json(param1, param2, param3, param4)
            for proj in Enemy.projectiles:
                proj.kill()
            for proj in Player.projectiles:
                proj.kill()
            for enemy in enemies:
                enemy.kill()
            done = game_loop()

    pygame.quit()
