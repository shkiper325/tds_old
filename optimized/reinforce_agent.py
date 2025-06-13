"""Optimized REINFORCE agent for PvP training."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import pygame
from collections import deque
import matplotlib.pyplot as plt

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
                nn.Dropout(0.1)
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
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
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
    
    def __init__(self, env, agent1, agent2, max_episodes=5000):
        self.env = env
        self.agent1 = agent1
        self.agent2 = agent2
        self.max_episodes = max_episodes
        
        # Tracking
        self.episode_rewards = deque(maxlen=100)
        self.win_rates = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        
        # Logging
        self.policy_losses = []
        self.value_losses = []
        
    def train(self, save_interval=100, render_interval=None, verbose=False):
        """Train both agents."""
        print(f"Starting training for {self.max_episodes} episodes...")
        print(f"Device: {self.agent1.device}")
        
        if verbose:
            print("Verbose mode enabled - showing episode rendering")
        
        for episode in range(self.max_episodes):
            should_render = (render_interval and episode % render_interval == 0) or verbose
            episode_reward1, episode_reward2, episode_length, winner = self._run_episode(
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
            
            if policy_loss1 > 0:  # Only track if there was an update
                self.policy_losses.append((policy_loss1, policy_loss2))
                self.value_losses.append((value_loss1, value_loss2))
            
            # Logging
            if (episode + 1) % 10 == 0:
                avg_reward1 = np.mean([r[0] for r in list(self.episode_rewards)[-10:]])
                avg_reward2 = np.mean([r[1] for r in list(self.episode_rewards)[-10:]])
                avg_length = np.mean(list(self.episode_lengths)[-10:])
                
                win_rate1 = np.mean([1 for w in list(self.win_rates)[-10:] if w == 1])
                win_rate2 = np.mean([1 for w in list(self.win_rates)[-10:] if w == 2])
                draw_rate = np.mean([1 for w in list(self.win_rates)[-10:] if w == 0])
                
                print(f"Episode {episode + 1:4d} | "
                      f"Avg Reward: {avg_reward1:6.2f}/{avg_reward2:6.2f} | "
                      f"Win Rate: {win_rate1:.2f}/{win_rate2:.2f}/{draw_rate:.2f} | "
                      f"Length: {avg_length:.1f}")
            
            # Save checkpoints
            if (episode + 1) % save_interval == 0:
                self.save_checkpoint(f"checkpoint_episode_{episode + 1}")
                self.plot_training_progress()
        
        print("Training completed!")
        self.save_checkpoint("final_model")
        self.plot_training_progress()
    
    def _run_episode(self, render=False, verbose=False):
        """Run a single episode."""
        obs, _ = self.env.reset()
        obs1, obs2 = obs
        
        self.agent1.reset_episode()
        self.agent2.reset_episode()
        
        episode_reward1 = 0
        episode_reward2 = 0
        step = 0
        
        if verbose:
            print(f"Starting episode... (render={render})")
        
        done = False
        while not done:
            # Get actions
            action1 = self.agent1.act(obs1)
            action2 = self.agent2.act(obs2)
            
            # Step environment
            (obs1, obs2), (reward1, reward2), done, info = self.env.step([action1, action2])
            
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
        
        return episode_reward1, episode_reward2, step, winner
    
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
