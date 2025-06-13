import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import pygame

from pvp_env import PvPEnv
from Player import Player
from Vars import SPEED
from pvp_env import i_to_dir_4
from utils import dirac_delta, normalize, flatten_them_all, dist, angle, PI


class PvPEnvTwoAgents(PvPEnv):
    """Variant of PvPEnv that allows both players to be controlled by agents."""

    def _ai_step(self):
        # Disable built-in AI behaviour
        pass

    def get_obs(self):
        """Return observations for player1 and player2 separately."""
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

        obs1 = flatten_them_all([mv1, pos1, [health1], [cd1], mv2, pos2, [health2], [cd2], [d], [ang]]).astype(np.float32)
        obs2 = flatten_them_all([mv2, pos2, [health2], [cd2], mv1, pos1, [health1], [cd1], [d], [(ang + 1) % 1]]).astype(np.float32)
        return obs1, obs2

    def step(self, actions):
        """Perform a step given actions for both players."""
        action1, action2 = actions
        p1 = self.player1.sprite
        p2 = self.player2.sprite
        start_h1 = p1.health
        start_h2 = p2.health

        a1 = dirac_delta(int(action1), 32).reshape(2, 4, 4)
        c1 = np.unravel_index(np.argmax(a1), a1.shape)
        if c1[0] == 1:
            self._shoot(p1, i_to_dir_4(c1[2]))
        p1.movementVector = list(i_to_dir_4(c1[1]))

        a2 = dirac_delta(int(action2), 32).reshape(2, 4, 4)
        c2 = np.unravel_index(np.argmax(a2), a2.shape)
        if c2[0] == 1:
            self._shoot(p2, i_to_dir_4(c2[2]))
        p2.movementVector = list(i_to_dir_4(c2[1]))

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

        obs1, obs2 = self.get_obs()
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

        reward = np.tanh(reward)

        return (obs1, obs2), (reward, -reward), done, {}


class Policy(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim, 1280),
            nn.ReLU(),
            nn.Linear(1280, 1280),
            nn.ReLU(),
            nn.Linear(1280, action_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Agent:
    def __init__(self, obs_dim, action_dim, lr=1e-3):
        self.policy = Policy(obs_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def act(self, obs):
        obs_t = torch.tensor(obs, dtype=torch.float32)
        probs = self.policy(obs_t)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(self, log_probs, rewards, gamma=0.99):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        loss = -(torch.stack(log_probs) * returns).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def train(num_episodes=1000, verbose=False):
    env = PvPEnvTwoAgents()
    if verbose:
        env.screen = pygame.display.set_mode(env.size)
        pygame.display.set_caption("Training PvP Agents")
    obs1, obs2 = env.get_obs()
    obs_dim = len(obs1)
    action_dim = 32

    agent1 = Agent(obs_dim, action_dim)
    agent2 = Agent(obs_dim, action_dim)

    for episode in range(num_episodes):
        env.reset()
        obs1, obs2 = env.get_obs()
        log_p1, rew1 = [], []
        log_p2, rew2 = [], []
        done = False
        while not done:
            a1, lp1 = agent1.act(obs1)
            a2, lp2 = agent2.act(obs2)
            (obs1, obs2), (r1, r2), done, _ = env.step((a1, a2))
            if verbose:
                env.screen.fill((255, 255, 255))
                env.player1.sprite.render(env.screen)
                env.player2.sprite.render(env.screen)
                for proj in list(Player.projectiles):
                    proj.render(env.screen)
                pygame.display.flip()
                pygame.event.pump()
            log_p1.append(lp1)
            rew1.append(r1)
            log_p2.append(lp2)
            rew2.append(r2)
        agent1.update(log_p1, rew1)
        agent2.update(log_p2, rew2)
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1} complete")

    torch.save(agent1.policy.state_dict(), "agent1.pth")
    torch.save(agent2.policy.state_dict(), "agent2.pth")
    if verbose:
        pygame.quit()


if __name__ == "__main__":
    train(verbose=True)
