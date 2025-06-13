import numpy as np
import pygame
import gym
from gym import spaces

from Player import Player
from Vars import SPEED
from utils import normalize, dirac_delta, flatten_them_all, dist, angle, PI


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
        raise ValueError("Invalid direction index")


class PvPEnv(gym.Env):
    """Simple two-player battle environment compatible with Stable Baselines3."""

    metadata = {"render.modes": ["human"]}

    OBS_DIM = 14

    def __init__(self):
        pygame.init()
        self.size = (800, 600)
        self.screen = pygame.display.set_mode(self.size, pygame.HIDDEN)
        self.clock = pygame.time.Clock()

        self.observation_space = spaces.Box(
            low=-np.ones((self.OBS_DIM,), dtype=np.float32) - 0.01,
            high=np.ones((self.OBS_DIM,), dtype=np.float32) + 0.01,
            shape=(self.OBS_DIM,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(32)

        self.reset()

    def _shoot(self, player, direction):
        target = [player.pos[0] + direction[0] * 10, player.pos[1] + direction[1] * 10]
        before = set(Player.projectiles)
        player.shoot(target)
        for pj in set(Player.projectiles) - before:
            pj.owner = player

    def _ai_step(self):
        p1 = self.player1.sprite
        p2 = self.player2.sprite
        move = [0, 0]
        if p2.pos[0] < p1.pos[0] - 5:
            move[0] = 1
        elif p2.pos[0] > p1.pos[0] + 5:
            move[0] = -1
        if p2.pos[1] < p1.pos[1] - 5:
            move[1] = 1
        elif p2.pos[1] > p1.pos[1] + 5:
            move[1] = -1
        p2.movementVector = move
        self._shoot(p2, normalize([p1.pos])[0])

    def _normalize_pos(self, pos):
        x = (pos[0] / self.size[0] - 0.5) * 2
        y = (pos[1] / self.size[1] - 0.5) * 2
        return [x, y]

    def get_state(self):
        p1 = self.player1.sprite
        p2 = self.player2.sprite

        def foo(x):
            return (np.array(x) - 0.5) * 2

        mv1 = normalize([p1.movementVector2])[0]
        mv2 = normalize([p2.movementVector2])[0]
        pos1 = self._normalize_pos(p1.pos)
        pos2 = self._normalize_pos(p2.pos)
        health1 = foo(p1.health / 3)
        health2 = foo(p2.health / 3)
        cd1 = max(0, p1.equippedWeapon.weaponCooldown - (pygame.time.get_ticks() - p1.equippedWeapon.lastShot))
        cd1 = foo(cd1 / p1.equippedWeapon.weaponCooldown)
        cd2 = max(0, p2.equippedWeapon.weaponCooldown - (pygame.time.get_ticks() - p2.equippedWeapon.lastShot))
        cd2 = foo(cd2 / p2.equippedWeapon.weaponCooldown)
        d = dist(p1.pos, p2.pos) / 1000
        ang = angle(p1.pos, p2.pos) / (2 * PI)
        features = flatten_them_all([mv1, pos1, [health1], [cd1], mv2, pos2, [health2], [cd2], [d], [ang]])
        return features.astype(np.float32)

    def reset(self):
        for pj in list(Player.projectiles):
            pj.kill()
        self.player1 = pygame.sprite.GroupSingle(Player(self.size))
        self.player2 = pygame.sprite.GroupSingle(Player(self.size))
        self.player1.sprite.pos = [self.size[0] * 0.25, self.size[1] / 2]
        self.player1.sprite.rect.topleft = self.player1.sprite.pos
        self.player2.sprite.pos = [self.size[0] * 0.75, self.size[1] / 2]
        self.player2.sprite.rect.topleft = self.player2.sprite.pos
        self.player1.sprite.health = 3
        self.player2.sprite.health = 3
        self.steps = 0
        return self.get_state()

    def step(self, action):
        p1 = self.player1.sprite
        p2 = self.player2.sprite
        start_h1 = p1.health
        start_h2 = p2.health

        action = dirac_delta(int(action), 32)
        action = np.reshape(action, (2, 4, 4))
        chosen = np.unravel_index(np.argmax(action), action.shape)
        shoot = chosen[0] == 1
        move_dir = i_to_dir_4(chosen[1])
        shoot_dir = i_to_dir_4(chosen[2])

        p1.movementVector = list(move_dir)
        if shoot:
            self._shoot(p1, shoot_dir)

        self._ai_step()

        t_delta = self.clock.get_time() / 17
        p1.move(self.size, t_delta)
        p2.move(self.size, t_delta)

        for proj in list(Player.projectiles):
            proj.move(self.size, t_delta)
            if pygame.sprite.collide_rect(proj, p1) and getattr(proj, "owner", None) is not p1:
                proj.kill()
                p1.health -= 1
            elif pygame.sprite.collide_rect(proj, p2) and getattr(proj, "owner", None) is not p2:
                proj.kill()
                p2.health -= 1

        self.clock.tick(120 * SPEED)

        state = self.get_state()
        reward = 0
        reward += (start_h2 - p2.health) * 5
        reward -= (start_h1 - p1.health) * 5

        done = False
        if p2.health <= 0:
            reward += 50
            done = True
        if p1.health <= 0:
            reward -= 50
            done = True

        self.steps += 1
        if self.steps > 2000:
            done = True

        return state, reward, done, {}

    def render(self, mode="human"):
        pass
