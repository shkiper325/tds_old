"""Optimized PvP environment for training two RL agents."""

import pygame
import numpy as np
import gym
from gym import spaces
from entities import Player, Projectile
from utils import normalize_vector, distance, angle_between, flatten_features, get_4_directions, PI

class PvPEnvironment(gym.Env):
    """Two-agent PvP environment with continuous action space."""
    
    def __init__(self, screen_size=(350, 350), max_steps=500, render_mode=None,
                 player_speed=450.0, player_health=3, reward_params=None, weapon_params=None):
        super().__init__()
        
        self.screen_size = screen_size
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.player_speed = player_speed
        self.player_health = player_health
        
        # Reward parameters
        self.reward_params = reward_params or {
            'damage': 1.0, 'hit': -1.0, 'kill': 20.0, 'tick': -0.01,
            'wall': -0.5, 'distance': 0.3, 'optimal_distance': 200.0, 'distance_tolerance': 150.0
        }
        
        # Weapon parameters  
        self.weapon_params = weapon_params or {
            'pistol_cooldown': 250, 'shotgun_cooldown': 750, 'machinegun_cooldown': 100,
            'projectile_speed': 300.0
        }
        
        self.screen_size = screen_size
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        # Initialize pygame
        pygame.init()
        if render_mode == "human":
            self.screen = pygame.display.set_mode(screen_size)
            pygame.display.set_caption("PvP Training")
        else:
            self.screen = pygame.Surface(screen_size)
        
        self.clock = pygame.time.Clock()
        
        # Action space: [move_x, move_y, shoot_x, shoot_y] for each agent
        # move_x, move_y in [-1, 1], shoot_x, shoot_y in [-1, 1] (relative direction)
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1, -1]), 
            high=np.array([1, 1, 1, 1]), 
            dtype=np.float32
        )
        
        # Observation space: normalized features for both players
        # [own_pos(2), own_vel(2), own_health(1), own_weapon_cd(1), 
        #  enemy_pos(2), enemy_vel(2), enemy_health(1), enemy_weapon_cd(1),
        #  relative_distance(1), relative_angle(1), projectiles(10)]
        obs_dim = 24  # Total observation dimension
        self.observation_space = spaces.Box(
            low=-np.ones(obs_dim, dtype=np.float32),
            high=np.ones(obs_dim, dtype=np.float32),
            dtype=np.float32
        )
        
        # Game objects
        self.player1 = None
        self.player2 = None
        self.projectiles = pygame.sprite.Group()
        
        # Game state
        self.steps = 0
        self.episode_rewards = [0, 0]
        
        self.reset()
    
    def _normalize_position(self, pos):
        """Normalize position to [-1, 1] range."""
        return np.array([
            (pos[0] / self.screen_size[0] - 0.5) * 2,
            (pos[1] / self.screen_size[1] - 0.5) * 2
        ])
    
    def _normalize_health(self, health, max_health=3):
        """Normalize health to [-1, 1] range."""
        return (health / max_health - 0.5) * 2
    
    def _normalize_cooldown(self, weapon):
        """Normalize weapon cooldown to [-1, 1] range."""
        if weapon.cooldown == 0:
            return -1.0
        
        time_since_shot = pygame.time.get_ticks() - weapon.last_shot
        cooldown_ratio = min(1.0, time_since_shot / weapon.cooldown)
        return (cooldown_ratio - 0.5) * 2
    
    def _get_observation(self, player_id):
        """Get observation for a specific player."""
        if player_id == 1:
            own_player = self.player1
            enemy_player = self.player2
        else:
            own_player = self.player2
            enemy_player = self.player1
        
        # Own player features
        own_pos = self._normalize_position(own_player.pos)
        own_vel = normalize_vector(own_player.velocity) if np.linalg.norm(own_player.velocity) > 0 else np.array([0, 0])
        own_health = self._normalize_health(own_player.health)
        own_weapon_cd = self._normalize_cooldown(own_player.weapons[own_player.current_weapon])
        
        # Enemy player features
        enemy_pos = self._normalize_position(enemy_player.pos)
        enemy_vel = normalize_vector(enemy_player.velocity) if np.linalg.norm(enemy_player.velocity) > 0 else np.array([0, 0])
        enemy_health = self._normalize_health(enemy_player.health)
        enemy_weapon_cd = self._normalize_cooldown(enemy_player.weapons[enemy_player.current_weapon])
        
        # Relative features
        rel_distance = distance(own_player.pos, enemy_player.pos) / 1000.0  # Normalized by max possible distance
        rel_angle = angle_between(own_player.pos, enemy_player.pos) / (2 * PI)
        
        # Projectile features (closest 5 projectiles)
        projectile_features = []
        enemy_projectiles = [p for p in self.projectiles if p.owner == enemy_player]
        
        # Sort by distance to own player
        enemy_projectiles.sort(key=lambda p: distance(p.pos, own_player.pos))
        
        for i in range(5):  # Track 5 closest enemy projectiles
            if i < len(enemy_projectiles):
                proj = enemy_projectiles[i]
                proj_dist = distance(proj.pos, own_player.pos) / 1000.0
                proj_angle = angle_between(own_player.pos, proj.pos) / (2 * PI)
                projectile_features.extend([proj_dist, proj_angle])
            else:
                projectile_features.extend([-1.0, -1.0])  # No projectile
        
        # Combine all features
        observation = flatten_features([
            own_pos, own_vel, [own_health], [own_weapon_cd],
            enemy_pos, enemy_vel, [enemy_health], [enemy_weapon_cd],
            [rel_distance], [rel_angle],
            projectile_features
        ])
        
        return observation.astype(np.float32)
    
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Clear projectiles
        self.projectiles.empty()
        
        # Create players at opposite sides
        self.player1 = Player(self.screen_size, color=(255, 0, 0), 
                             max_speed=self.player_speed, max_health=self.player_health,
                             weapon_params=self.weapon_params)
        self.player2 = Player(self.screen_size, color=(0, 0, 255),
                             max_speed=self.player_speed, max_health=self.player_health, 
                             weapon_params=self.weapon_params)
        
        # Position players
        self.player1.pos = np.array([self.screen_size[0] * 0.25, self.screen_size[1] * 0.5])
        self.player1.rect.center = self.player1.pos
        
        self.player2.pos = np.array([self.screen_size[0] * 0.75, self.screen_size[1] * 0.5])
        self.player2.rect.center = self.player2.pos
        
        # Reset game state
        self.steps = 0
        self.episode_rewards = [0, 0]
        
        # Return initial observations for both players
        obs1 = self._get_observation(1)
        obs2 = self._get_observation(2)
        
        info = {"player1": {}, "player2": {}}
        
        return (obs1, obs2), info
    
    def step(self, actions):
        """Execute one environment step with actions from both agents."""
        action1, action2 = actions
        
        # Store initial health for reward calculation
        initial_health1 = self.player1.health
        initial_health2 = self.player2.health
        
        dt = self.clock.get_time() / 1000.0  # Scale simulation time
        if dt > 0.05:
            dt = 0.05
        
        # Process player 1 action
        move_action1 = action1[:2]
        shoot_action1 = action1[2:]
        
        self.player1.update(dt, move_action1, self.projectiles)
        
        # Shoot if shoot action is significant
        if np.linalg.norm(shoot_action1) > 0.1:
            shoot_target1 = self.player1.pos + normalize_vector(shoot_action1) * 100
            self.player1.shoot(shoot_target1, self.projectiles)
        
        # Process player 2 action
        move_action2 = action2[:2]
        shoot_action2 = action2[2:]
        
        self.player2.update(dt, move_action2, self.projectiles)
        
        # Shoot if shoot action is significant
        if np.linalg.norm(shoot_action2) > 0.1:
            shoot_target2 = self.player2.pos + normalize_vector(shoot_action2) * 100
            self.player2.shoot(shoot_target2, self.projectiles)
        
        # Update projectiles
        for projectile in list(self.projectiles):
            projectile.update(dt, self.screen_size)
        
        # Handle collisions
        self._handle_collisions()
        
        # Calculate rewards
        reward1, reward2 = self._calculate_rewards(initial_health1, initial_health2)
        self.episode_rewards[0] += reward1
        self.episode_rewards[1] += reward2
        
        # Check if episode is done
        done = (not self.player1.alive or not self.player2.alive or 
                self.steps >= self.max_steps)
        
        # Get new observations
        obs1 = self._get_observation(1)
        obs2 = self._get_observation(2)
        
        self.steps += 1
        # Keep a stable frame rate while allowing accelerated simulation
        self.clock.tick(60)
        
        info = {
            "player1": {"health": self.player1.health, "alive": self.player1.alive},
            "player2": {"health": self.player2.health, "alive": self.player2.alive},
            "episode_rewards": self.episode_rewards.copy()
        }
        
        return (obs1, obs2), (reward1, reward2), done, info
    
    def _handle_collisions(self):
        """Handle projectile-player collisions."""
        for projectile in list(self.projectiles):
            # Check collision with player1
            if (projectile.owner != self.player1 and 
                pygame.sprite.collide_rect(projectile, self.player1)):
                projectile.kill()
                self.player1.take_damage()
            
            # Check collision with player2
            elif (projectile.owner != self.player2 and 
                  pygame.sprite.collide_rect(projectile, self.player2)):
                projectile.kill()
                self.player2.take_damage()
    
    def _calculate_rewards(self, initial_health1, initial_health2):
        """Shaped reward for a 1-vs-1 top-down shooter."""
        # Get reward parameters
        W_DMG = self.reward_params['damage']
        W_HIT = self.reward_params['hit']
        W_KILL = self.reward_params['kill']
        W_TICK = self.reward_params['tick']
        W_WALL = self.reward_params['wall']
        W_DIST = self.reward_params['distance']
        OPT_DIST = self.reward_params['optimal_distance']
        DIST_TOL = self.reward_params['distance_tolerance']
        
        # ---------- базовый сигнал ----------
        dmg_to_p2 = initial_health2 - self.player2.health
        dmg_to_p1 = initial_health1 - self.player1.health
        
        reward1  = W_DMG * dmg_to_p2 + W_HIT * dmg_to_p1
        reward2  = W_DMG * dmg_to_p1 + W_HIT * dmg_to_p2
        
        # ---------- победа / поражение ----------
        if not self.player2.alive and self.player1.alive:
            reward1 += W_KILL
            reward2 -= W_KILL
        elif not self.player1.alive and self.player2.alive:
            reward1 -= W_KILL
            reward2 += W_KILL

        # ---------- шаг времени ----------
        reward1 += W_TICK
        reward2 += W_TICK
        
        # ---------- штраф за «зону стен» ----------
        def near_wall(pos):
            x, y = pos
            return (
                x < 140 or x > self.screen_size[0] - 140 or
                y < 140 or y > self.screen_size[1] - 140
            )
        if near_wall(self.player1.pos):
            reward1 += W_WALL
        if near_wall(self.player2.pos):
            reward2 += W_WALL
        
        # ---------- плавный bonus за «правильную» дистанцию ----------
        # Гауссова кривая: максимум = 0 (идеально), всё что дальше — отрицательно.
        dist = distance(self.player1.pos, self.player2.pos)
        dist_reward = -((dist - OPT_DIST) / DIST_TOL) ** 2
        reward1 += W_DIST * dist_reward
        reward2 += W_DIST * dist_reward
        
        return reward1, reward2


    def render(self, mode="human"):
        """Render the environment."""
        if mode == "human" and self.render_mode == "human":
            self.screen.fill((255, 255, 255))
            
            # Draw players
            self.player1.render(self.screen)
            self.player2.render(self.screen)
            
            # Draw projectiles
            for projectile in self.projectiles:
                projectile.render(self.screen)
            
            # Draw health bars
            self._draw_health_bar(self.player1, (10, 10))
            self._draw_health_bar(self.player2, (10, 40))
            
            pygame.display.flip()
            
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
    
    def _draw_health_bar(self, player, pos):
        """Draw health bar for a player."""
        bar_width = 100
        bar_height = 20
        
        # Background
        pygame.draw.rect(self.screen, (100, 100, 100), 
                        (*pos, bar_width, bar_height))
        
        # Health bar
        health_ratio = player.health / player.max_health
        health_width = int(bar_width * health_ratio)
        color = (0, 255, 0) if health_ratio > 0.5 else (255, 255, 0) if health_ratio > 0.25 else (255, 0, 0)
        
        pygame.draw.rect(self.screen, color, 
                        (*pos, health_width, bar_height))
    
    def close(self):
        """Close the environment."""
        if hasattr(self, 'screen'):
            pygame.quit()
