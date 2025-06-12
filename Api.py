import pygame
import math
import json
import time
import os
screenSize    = (800, 600)
from Enemy import Enemy
bot_killed_projectiles = []
player_damaged_projectiles = []
AUTO_RESTART = True
i = 0 
B = ""

from Vars import SPEED

class Api():

	def __init__(self):
		self.input()
		self.data_Player()
		self.data_Shoot()
		self.data_Enemy_Projectiles()
		self.data_Player_Projectiles()
		self.data_Enemies()
		
	class input():
		shoot = False
		restart = False
		move = (1, 0)
		shoot_dir = (1, 0)

		def __init__(self):
			self.shoot = False
			self.restart = False
			self.move = (1, 0)
			self.shoot_dir = (1, 0)

	class data_Player(): 
		pos = [0.5, 0.5]
		health = 0
		alive = True
		movementVector = [0, 0]

		def __init__(self):
			self.pos = [screenSize[0] // 2, screenSize[1] // 2]
			self.health = 3
			self.alive = True
			self.movementVector = [0, 0]
			self.movementSpeed = 3 * SPEED

		def update_data_Player(self, player):
			self.pos = player.pos
			self.health = player.health
			self.alive = player.alive
			self.movementVector = player.movementVector2

		def __str__(self):
			return "Player [pos ({0}, {1}); movementVector ({2}, {3}); health {4}; alive {5}]\n".format(self.pos[0],self.pos[1],self.movementVector[0], self.movementVector[0], self.health, self.alive)

	class data_Shoot():
		lastShot = 0 
		weaponCooldown = 250

		def __init__(self):
			self.lastShot = 0
			self.weaponCooldown = 250

		def update_data_Shoot(self, shoot):
			self.lastShot = shoot.lastShot

		def __str__(self):
			return 'Weapon [lastShot {0}]\n'.format(self.lastShot)

	class data_Enemy_Projectiles():
		projectiles = []

		def __init__(self):
			self.projectiles = []

		def add_enemy_proj(self, proj):
			if pygame.time.get_ticks() <= proj.createdAt + proj.lifetime and proj.pos[0] <= screenSize[0] and proj.pos[0] >= 0  and \
           proj.pos[1] <= screenSize[1] and proj.pos[1] >= 0 and proj.damagedAt == 0: 
				self.projectiles.append(proj)

		def del_enemy_proj(self):
			self.projectiles = []

		def __str__(self):
			A = "Enemy_Projectiles \n[\n"
			for proj in self.projectiles:
				A += "Projectile [pos ({0}, {1}); movementVector ({2}, {3}); createdAt {4}; damagedAt {5}] \n".format(proj.pos[0],proj.pos[1],proj.movementVector[0], proj.movementVector[0], proj.createdAt, proj.damagedAt)
			A+= "]\n"
			return A

	class data_Player_Projectiles():
		projectiles = []

		def __init__(self):
			self.projectiles = []

		def add_player_proj(self, proj):
			if pygame.time.get_ticks() <= proj.createdAt + proj.lifetime and proj.pos[0] <= screenSize[0] and proj.pos[0] >= 0  and \
           proj.pos[1] <= screenSize[1] and proj.pos[1] >= 0 and proj.damagedAt == 0:
				self.projectiles.append(proj)

		def del_player_proj(self):
			self.projectiles = []

		def __str__(self):
			A = "Player_Projectiles \n[\n"
			for proj in self.projectiles:
				A += "Projectile [pos ({0}, {1}); movementVector ({2}, {3}); createdAt {4}; damagedAt {5}] \n".format(proj.pos[0],proj.pos[1],proj.movementVector[0], proj.movementVector[0], proj.createdAt, proj.damagedAt)
			A+= "]\n"
			return A

	class data_Enemies():
		enemies = []

		def __init__(self):
			self.enemies = []

		def add_enemy(self, enemy):
			self.enemies.append(enemy)

		def del_enemy(self):
			self.enemies = []

		def __str__(self):
			A = "Enemies \n[\n"
			for enemy in self.enemies:
				A += "Enemy[pos ({0}, {1}); movementVector ({2}, {3}); lastShot {4}; timestamp_start {5}; timestamp_end {6}]\n".format(enemy.pos[0],enemy.pos[1],enemy.movementVector2[0], enemy.movementVector2[0], enemy.lastShot, enemy.timestamp_start, enemy.timestamp_end)
			A += "]\n"
			return A

	def on_bot_kill(self, projectile):
		bot_killed_projectiles.append(projectile)

	def on_player_damage(self, projectile):
		player_damaged_projectiles.append(projectile)

	def on_update1(self, player):
		global B
		self.data_Player.update_data_Player(self.data_Player, player)
		B += ("CURRENT_TIME:{0}".format(pygame.time.get_ticks())+ '\n[\n')
		B += (self.data_Player.__str__(self.data_Player))
		B += (self.data_Shoot.__str__(self.data_Shoot))
		B += (self.data_Enemy_Projectiles.__str__(self.data_Enemy_Projectiles))
		B += (self.data_Player_Projectiles.__str__(self.data_Player_Projectiles))
		B += (self.data_Enemies.__str__(self.data_Enemies))
		B += ("]\n")
	def on_update2(self, player):
		self.data_Enemies.del_enemy(self.data_Enemies)
		self.data_Player_Projectiles.del_player_proj(self.data_Player_Projectiles)
		self.data_Enemy_Projectiles.del_enemy_proj(self.data_Enemy_Projectiles)


	def on_restart(self):
		global i
		global B
		global bot_killed_projectiles, player_damaged_projectiles
		with open(os.path.join('dumps', 'dump_'+str(i)+'.txt'), 'a') as f:
			f.write(B+'\n')
			f.write("bot_killed_projectiles:"+ '\n')
			for k in bot_killed_projectiles:
				f.write(k.__str__())
			f.write("player_damaged_projectiles:"+'\n')
			for k in player_damaged_projectiles:
				f.write(k.__str__())
			B = ""
			player_damaged_projectiles = []
			bot_killed_projectiles = []
		if AUTO_RESTART:
			i+=1
			self.input.restart = True

def create_json(param1, param2, param3, param4):
	global bot_killed_projectiles, player_damaged_projectiles
	bot_killed_projectiles = []
	player_damaged_projectiles = []
	with open('dump_'+str(i)+'.txt', 'w') as f:
		f.write(str(time.time())+'\n')
		f.write(str(param1)+ '\n')
		f.write(str(param2)+ '\n')
		f.write(str(param3)+ '\n')
		f.write(str(param4)+ '\n')
	pass