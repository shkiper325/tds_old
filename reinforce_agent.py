"""Optimized REINFORCE agent for PvP training."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import pygame
from collections import deque
import matplotlib.pyplot as plt
import os
from torch.utils.tensorboard import SummaryWriter

class PolicyNetwork(nn.Module):
    """Policy network for continuous action space."""
    
    def __init__(self, obs_dim, action_dim, hidden_dims=[128, 128]):
        super().__init__()
        
        layers = []
        prev_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        
        # Separate heads for mean and log_std
        self.mean_head = nn.Linear(prev_dim, action_dim)
        self.log_std_head = nn.Linear(prev_dim, action_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.backbone(x)
        
        mean = torch.tanh(self.mean_head(features))  # Actions in [-1, 1]
        log_std = torch.clamp(self.log_std_head(features), -20, 2)
        std = torch.exp(log_std)
        
        return mean, std

class REINFORCEAgent:
    """REINFORCE agent with baseline and continuous actions."""
    
    def __init__(self, obs_dim, action_dim, lr=3e-4, gamma=0.99, entropy_coef=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.policy_net = PolicyNetwork(obs_dim, action_dim).to(self.device)
        self.value_net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(self.device)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        
        # Hyperparameters
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        
        # Episode storage
        self.reset_episode()
    
    def reset_episode(self):
        """Reset episode storage."""
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.entropies = []
    
    def act(self, obs):
        """Select action using current policy."""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        # Get policy output (with gradients for training)
        mean, std = self.policy_net(obs_tensor)
        
        # Get value (with gradients for training)  
        value = self.value_net(obs_tensor)
        
        # Create distribution and sample action
        dist = Normal(mean, std)
        action = dist.sample()
        
        # Store for learning (these need gradients)
        self.log_probs.append(dist.log_prob(action).sum(dim=-1))
        self.values.append(value.squeeze())
        self.entropies.append(dist.entropy().sum(dim=-1))
        
        return action.squeeze().detach().cpu().numpy()
    
    def store_reward(self, reward):
        """Store reward for current step."""
        self.rewards.append(reward)
    
    def update(self):
        """Update policy and value networks using REINFORCE with baseline."""
        if len(self.rewards) == 0:
            return 0.0, 0.0
        
        # Calculate discounted returns
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Convert stored tensors
        log_probs = torch.stack(self.log_probs).to(self.device)
        values = torch.stack(self.values).to(self.device)
        entropies = torch.stack(self.entropies).to(self.device)
        
        # Calculate advantages
        advantages = returns - values.detach()
        
        # Policy loss (REINFORCE with baseline)
        policy_loss = -(log_probs * advantages).mean()
        
        # Add entropy bonus
        entropy_loss = -entropies.mean()
        total_policy_loss = policy_loss + self.entropy_coef * entropy_loss
        
        # Value loss
        value_loss = nn.MSELoss()(values, returns)
        
        # Update policy network
        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        self.policy_optimizer.step()
        
        # Update value network
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
        self.value_optimizer.step()
        
        self.reset_episode()
        
        return policy_loss.item(), value_loss.item()
    
    def save(self, filepath):
        """Save agent networks."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath):
        """Load agent networks."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.value_net.load_state_dict(checkpoint['value_net'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])

class TrainingManager:
    """Manages the training process for two agents."""
    
    def __init__(self, env, agent1, agent2, max_episodes=5000, tensorboard_dir="tb"):
        self.env = env
        self.agent1 = agent1
        self.agent2 = agent2
        self.max_episodes = max_episodes
        
        # Create TensorBoard writer
        os.makedirs(tensorboard_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=tensorboard_dir)
        
        # Tracking
        self.episode_rewards = deque(maxlen=100)
        self.win_rates = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        
        # Extended tracking for more detailed metrics
        self.agent1_rewards = deque(maxlen=100)
        self.agent2_rewards = deque(maxlen=100)
        self.health_differences = deque(maxlen=100)
        self.damage_dealt = deque(maxlen=100)
        
        # Logging
        self.policy_losses = []
        self.value_losses = []
        
    def train(self, save_interval=100, render_interval=None, verbose=False):
        """Train both agents."""
        print(f"Starting training for {self.max_episodes} episodes...")
        print(f"Device: {self.agent1.device}")
        print(f"TensorBoard logs will be saved to: {self.writer.log_dir}")
        
        if verbose:
            print("Verbose mode enabled - showing episode rendering")
        
        for episode in range(self.max_episodes):
            should_render = (render_interval and episode % render_interval == 0) or verbose
            episode_reward1, episode_reward2, episode_length, winner, episode_info = self._run_episode(
                render=should_render,
                verbose=verbose
            )
            
            # Update agents
            policy_loss1, value_loss1 = self.agent1.update()
            policy_loss2, value_loss2 = self.agent2.update()
            
            # Track metrics
            self.episode_rewards.append((episode_reward1, episode_reward2))
            self.episode_lengths.append(episode_length)
            self.win_rates.append(winner)
            self.agent1_rewards.append(episode_reward1)
            self.agent2_rewards.append(episode_reward2)
            
            # Track additional metrics
            health_diff = episode_info.get('final_health_diff', 0)
            damage_dealt_info = episode_info.get('damage_dealt', (0, 0))
            self.health_differences.append(health_diff)
            self.damage_dealt.append(damage_dealt_info)
            
            if policy_loss1 > 0:  # Only track if there was an update
                self.policy_losses.append((policy_loss1, policy_loss2))
                self.value_losses.append((value_loss1, value_loss2))
            
            # TensorBoard logging every episode
            self._log_to_tensorboard(episode, episode_reward1, episode_reward2, 
                                   episode_length, winner, policy_loss1, 
                                   policy_loss2, value_loss1, value_loss2, episode_info)
            
            # Console logging
            if (episode + 1) % 10 == 0:
                avg_reward1 = np.mean([r[0] for r in list(self.episode_rewards)[-10:]])
                avg_reward2 = np.mean([r[1] for r in list(self.episode_rewards)[-10:]])
                avg_length = np.mean(list(self.episode_lengths)[-10:])
                
                win_rate1 = np.mean([1 for w in list(self.win_rates)[-10:] if w == 1])
                win_rate2 = np.mean([1 for w in list(self.win_rates)[-10:] if w == 2])
                draw_rate = np.mean([1 for w in list(self.win_rates)[-10:] if w == 0])
                
                # Average reward for last 100 episodes (key metric)
                avg_reward_100_ep = np.mean([r[0] + r[1] for r in list(self.episode_rewards)]) / 2
                
                print(f"Episode {episode + 1:4d} | "
                      f"Avg Reward: {avg_reward1:6.2f}/{avg_reward2:6.2f} | "
                      f"Win Rate: {win_rate1:.2f}/{win_rate2:.2f}/{draw_rate:.2f} | "
                      f"Length: {avg_length:.1f} | "
                      f"Avg100: {avg_reward_100_ep:.2f}")
            
            # Save checkpoints
            if (episode + 1) % save_interval == 0:
                self.save_checkpoint(f"checkpoint_episode_{episode + 1}")
                self.plot_training_progress()
        
        print("Training completed!")
        self.save_checkpoint("final_model")
        self.plot_training_progress()
        self.writer.close()
    
    def _log_to_tensorboard(self, episode, reward1, reward2, episode_length, winner, 
                           policy_loss1, policy_loss2, value_loss1, value_loss2, episode_info):
        """Log metrics to TensorBoard."""
        
        # Basic episode metrics
        self.writer.add_scalar('Episode/Reward_Agent1', reward1, episode)
        self.writer.add_scalar('Episode/Reward_Agent2', reward2, episode)
        self.writer.add_scalar('Episode/Combined_Reward', reward1 + reward2, episode)
        self.writer.add_scalar('Episode/Length', episode_length, episode)
        
        # Winner tracking
        self.writer.add_scalar('Episode/Winner', winner, episode)
        
        # Losses (if available)
        if policy_loss1 > 0:
            self.writer.add_scalar('Training/Policy_Loss_Agent1', policy_loss1, episode)
            self.writer.add_scalar('Training/Policy_Loss_Agent2', policy_loss2, episode)
        if value_loss1 > 0:
            self.writer.add_scalar('Training/Value_Loss_Agent1', value_loss1, episode)
            self.writer.add_scalar('Training/Value_Loss_Agent2', value_loss2, episode)
        
        # Rolling averages - key metrics
        if len(self.episode_rewards) >= 10:
            # Last 10 episodes
            avg_reward1_10 = np.mean(list(self.agent1_rewards)[-10:])
            avg_reward2_10 = np.mean(list(self.agent2_rewards)[-10:])
            avg_combined_10 = (avg_reward1_10 + avg_reward2_10) / 2
            
            self.writer.add_scalar('Averages/Reward_Agent1_10ep', avg_reward1_10, episode)
            self.writer.add_scalar('Averages/Reward_Agent2_10ep', avg_reward2_10, episode)
            self.writer.add_scalar('Averages/Combined_Reward_10ep', avg_combined_10, episode)
            
            # Win rates (last 10)
            recent_wins = list(self.win_rates)[-10:]
            win_rate1_10 = sum(1 for w in recent_wins if w == 1) / len(recent_wins)
            win_rate2_10 = sum(1 for w in recent_wins if w == 2) / len(recent_wins)
            draw_rate_10 = sum(1 for w in recent_wins if w == 0) / len(recent_wins)
            
            self.writer.add_scalar('WinRates/Agent1_10ep', win_rate1_10, episode)
            self.writer.add_scalar('WinRates/Agent2_10ep', win_rate2_10, episode)
            self.writer.add_scalar('WinRates/Draw_10ep', draw_rate_10, episode)
        
        # КЛЮЧЕВАЯ МЕТРИКА: Средний reward за последние 100 роллаутов
        if len(self.episode_rewards) >= 50:
            # Last 50 episodes (or all if less than 100)
            recent_rewards = list(self.episode_rewards)[-50:]
            avg_reward1_50 = np.mean([r[0] for r in recent_rewards])
            avg_reward2_50 = np.mean([r[1] for r in recent_rewards])
            avg_combined_50 = (avg_reward1_50 + avg_reward2_50) / 2
            
            self.writer.add_scalar('Averages/Reward_Agent1_50ep', avg_reward1_50, episode)
            self.writer.add_scalar('Averages/Reward_Agent2_50ep', avg_reward2_50, episode)
            self.writer.add_scalar('Averages/Combined_Reward_50ep', avg_combined_50, episode)
        
        # Полные 100 эпизодов
        if len(self.episode_rewards) >= 100:
            # ГЛАВНАЯ МЕТРИКА: Last 100 episodes
            recent_100_rewards = list(self.episode_rewards)[-100:]
            avg_reward1_100 = np.mean([r[0] for r in recent_100_rewards])
            avg_reward2_100 = np.mean([r[1] for r in recent_100_rewards])
            avg_combined_100 = (avg_reward1_100 + avg_reward2_100) / 2
            
            self.writer.add_scalar('KEY_METRICS/Reward_Agent1_100ep', avg_reward1_100, episode)
            self.writer.add_scalar('KEY_METRICS/Reward_Agent2_100ep', avg_reward2_100, episode)
            self.writer.add_scalar('KEY_METRICS/Combined_Reward_100ep', avg_combined_100, episode)
            
            # Win rates (last 100)
            recent_wins_100 = list(self.win_rates)[-100:]
            win_rate1_100 = sum(1 for w in recent_wins_100 if w == 1) / 100
            win_rate2_100 = sum(1 for w in recent_wins_100 if w == 2) / 100
            draw_rate_100 = sum(1 for w in recent_wins_100 if w == 0) / 100
            
            self.writer.add_scalar('KEY_METRICS/WinRate_Agent1_100ep', win_rate1_100, episode)
            self.writer.add_scalar('KEY_METRICS/WinRate_Agent2_100ep', win_rate2_100, episode)
            self.writer.add_scalar('KEY_METRICS/DrawRate_100ep', draw_rate_100, episode)
            
            # Episode length average
            avg_length_100 = np.mean(list(self.episode_lengths)[-100:])
            self.writer.add_scalar('KEY_METRICS/Avg_Episode_Length_100ep', avg_length_100, episode)
        
        # Additional detailed metrics if available
        if episode_info:
            # Health and damage metrics
            if 'final_health_diff' in episode_info:
                self.writer.add_scalar('Battle/Health_Difference', episode_info['final_health_diff'], episode)
            
            if 'damage_dealt' in episode_info:
                damage1, damage2 = episode_info['damage_dealt']
                self.writer.add_scalar('Battle/Damage_Agent1', damage1, episode)
                self.writer.add_scalar('Battle/Damage_Agent2', damage2, episode)
                self.writer.add_scalar('Battle/Total_Damage', damage1 + damage2, episode)
            
            # Player positions and movement metrics
            if 'total_distance_moved' in episode_info:
                dist1, dist2 = episode_info['total_distance_moved']
                self.writer.add_scalar('Movement/Distance_Agent1', dist1, episode)
                self.writer.add_scalar('Movement/Distance_Agent2', dist2, episode)
            
            # Shooting metrics
            if 'shots_fired' in episode_info:
                shots1, shots2 = episode_info['shots_fired']
                self.writer.add_scalar('Combat/Shots_Agent1', shots1, episode)
                self.writer.add_scalar('Combat/Shots_Agent2', shots2, episode)
                
                # Accuracy if both damage and shots are available
                if 'damage_dealt' in episode_info and shots1 > 0:
                    accuracy1 = episode_info['damage_dealt'][0] / shots1 if shots1 > 0 else 0
                    accuracy2 = episode_info['damage_dealt'][1] / shots2 if shots2 > 0 else 0
                    self.writer.add_scalar('Combat/Accuracy_Agent1', accuracy1, episode)
                    self.writer.add_scalar('Combat/Accuracy_Agent2', accuracy2, episode)
        self.writer.close()
    
    def _run_episode(self, render=False, verbose=False):
        """Run a single episode."""
        obs, _ = self.env.reset()
        obs1, obs2 = obs
        
        self.agent1.reset_episode()
        self.agent2.reset_episode()
        
        episode_reward1 = 0
        episode_reward2 = 0
        step = 0
        
        # Additional tracking for detailed metrics
        damage_dealt1 = 0
        damage_dealt2 = 0
        shots_fired1 = 0
        shots_fired2 = 0
        distance_moved1 = 0
        distance_moved2 = 0
        prev_pos1 = None
        prev_pos2 = None
        
        if verbose:
            print(f"Starting episode... (render={render})")
        
        done = False
        while not done:
            # Get actions
            action1 = self.agent1.act(obs1)
            action2 = self.agent2.act(obs2)
            
            # Step environment
            (obs1, obs2), (reward1, reward2), done, info = self.env.step([action1, action2])
            
            # Track additional metrics if info is available
            if info and 'player1' in info and 'player2' in info:
                # Track damage dealt (estimate from health changes)
                if hasattr(self.env, 'prev_health1') and hasattr(self.env, 'prev_health2'):
                    health_loss1 = getattr(self.env, 'prev_health1', 100) - info['player1'].get('health', 100)
                    health_loss2 = getattr(self.env, 'prev_health2', 100) - info['player2'].get('health', 100)
                    if health_loss1 > 0:
                        damage_dealt2 += health_loss1
                    if health_loss2 > 0:
                        damage_dealt1 += health_loss2
                
                # Track movement (if position info available)
                if 'x' in info['player1'] and 'y' in info['player1']:
                    pos1 = (info['player1']['x'], info['player1']['y'])
                    pos2 = (info['player2']['x'], info['player2']['y'])
                    
                    if prev_pos1 is not None:
                        distance_moved1 += np.sqrt((pos1[0] - prev_pos1[0])**2 + (pos1[1] - prev_pos1[1])**2)
                        distance_moved2 += np.sqrt((pos2[0] - prev_pos2[0])**2 + (pos2[1] - prev_pos2[1])**2)
                    
                    prev_pos1 = pos1
                    prev_pos2 = pos2
                
                # Track shooting (if shooting action was taken - action[2] is typically shoot)
                if len(action1) > 2 and action1[2] > 0.5:  # Assuming shoot action > 0.5 threshold
                    shots_fired1 += 1
                if len(action2) > 2 and action2[2] > 0.5:
                    shots_fired2 += 1
            
            # Store rewards
            self.agent1.store_reward(reward1)
            self.agent2.store_reward(reward2)
            
            episode_reward1 += reward1
            episode_reward2 += reward2
            step += 1
            
            if render:
                self.env.render()
                if verbose:
                    # Show detailed info every 100 steps
                    if step % 100 == 0:
                        print(f"  Step {step:3d}: P1_HP={info['player1']['health']}, P2_HP={info['player2']['health']}, "
                              f"Rewards=({reward1:6.2f}, {reward2:6.2f})")
                pygame.time.wait(16 if verbose else 50)  # Faster in verbose mode
        
        if verbose:
            winner_name = "Player 1" if info["player1"]["alive"] and not info["player2"]["alive"] else \
                         "Player 2" if info["player2"]["alive"] and not info["player1"]["alive"] else "Draw"
            print(f"Episode finished: Winner={winner_name}, Steps={step}, "
                  f"Final rewards=({episode_reward1:.2f}, {episode_reward2:.2f})")
        
        # Determine winner
        if info["player1"]["alive"] and not info["player2"]["alive"]:
            winner = 1
        elif info["player2"]["alive"] and not info["player1"]["alive"]:
            winner = 2
        else:
            winner = 0  # Draw
        
        # Prepare episode info for logging
        episode_info = {
            'final_health_diff': info['player1'].get('health', 0) - info['player2'].get('health', 0),
            'damage_dealt': (damage_dealt1, damage_dealt2),
            'shots_fired': (shots_fired1, shots_fired2),
            'total_distance_moved': (distance_moved1, distance_moved2)
        }
        
        return episode_reward1, episode_reward2, step, winner, episode_info
    
    def save_checkpoint(self, name):
        """Save training checkpoint."""
        self.agent1.save(f"{name}_agent1.pth")
        self.agent2.save(f"{name}_agent2.pth")
        
        # Save training metrics
        np.save(f"{name}_metrics.npy", {
            'episode_rewards': list(self.episode_rewards),
            'win_rates': list(self.win_rates),
            'episode_lengths': list(self.episode_lengths),
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses
        })
    
    def plot_training_progress(self):
        """Plot training progress."""
        if len(self.episode_rewards) < 10:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Rewards
        rewards1 = [r[0] for r in self.episode_rewards]
        rewards2 = [r[1] for r in self.episode_rewards]
        
        axes[0, 0].plot(rewards1, label='Agent 1', alpha=0.7)
        axes[0, 0].plot(rewards2, label='Agent 2', alpha=0.7)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Win rates (rolling average)
        if len(self.win_rates) >= 20:
            win_rates_1 = []
            win_rates_2 = []
            draw_rates = []
            
            for i in range(20, len(self.win_rates)):
                recent_games = list(self.win_rates)[i-20:i]
                win_rates_1.append(sum(1 for w in recent_games if w == 1) / 20)
                win_rates_2.append(sum(1 for w in recent_games if w == 2) / 20)
                draw_rates.append(sum(1 for w in recent_games if w == 0) / 20)
            
            x = range(20, len(self.win_rates))
            axes[0, 1].plot(x, win_rates_1, label='Agent 1 Win Rate')
            axes[0, 1].plot(x, win_rates_2, label='Agent 2 Win Rate')
            axes[0, 1].plot(x, draw_rates, label='Draw Rate')
            axes[0, 1].set_title('Win Rates (20-episode rolling average)')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Win Rate')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Episode lengths
        axes[1, 0].plot(self.episode_lengths)
        axes[1, 0].set_title('Episode Lengths')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Steps')
        axes[1, 0].grid(True)
        
        # Losses
        if self.policy_losses:
            policy_losses_1 = [l[0] for l in self.policy_losses]
            policy_losses_2 = [l[1] for l in self.policy_losses]
            
            axes[1, 1].plot(policy_losses_1, label='Agent 1 Policy Loss')
            axes[1, 1].plot(policy_losses_2, label='Agent 2 Policy Loss')
            axes[1, 1].set_title('Policy Losses')
            axes[1, 1].set_xlabel('Update')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("Training progress plot saved as 'training_progress.png'")
