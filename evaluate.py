"""Script to evaluate trained agents."""

import pygame
import numpy as np
import argparse
from pvp_environment import PvPEnvironment
from reinforce_agent import REINFORCEAgent

def evaluate_agents(agent1_path, agent2_path, num_episodes=10, render=True, hidden_dims=[128, 128]):
    """Evaluate two trained agents."""
    
    # Create environment
    env = PvPEnvironment(render_mode="human" if render else None)
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Load agents with specified architecture
    agent1 = REINFORCEAgent(obs_dim, action_dim, hidden_dims=hidden_dims)
    agent2 = REINFORCEAgent(obs_dim, action_dim, hidden_dims=hidden_dims)
    
    agent1.load(agent1_path)
    agent2.load(agent2_path)
    
    # Set to evaluation mode (no exploration)
    agent1.policy_net.eval()
    agent2.policy_net.eval()
    
    results = []
    
    print(f"Evaluating agents for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        obs1, obs2 = obs
        
        episode_reward1 = 0
        episode_reward2 = 0
        step = 0
        
        done = False
        while not done:
            # Get deterministic actions (use mean of policy)
            with torch.no_grad():
                action1_mean, _ = agent1.policy_net(torch.FloatTensor(obs1).unsqueeze(0).to(agent1.device))
                action2_mean, _ = agent2.policy_net(torch.FloatTensor(obs2).unsqueeze(0).to(agent2.device))
                
                action1 = action1_mean.squeeze().cpu().numpy()
                action2 = action2_mean.squeeze().cpu().numpy()
            
            # Step environment
            (obs1, obs2), (reward1, reward2), done, info = env.step([action1, action2])
            
            episode_reward1 += reward1
            episode_reward2 += reward2
            step += 1
            
            if render:
                env.render()
                pygame.time.wait(16)  # ~60 FPS
        
        # Determine winner
        if info["player1"]["alive"] and not info["player2"]["alive"]:
            winner = "Agent 1"
            winner_num = 1
        elif info["player2"]["alive"] and not info["player1"]["alive"]:
            winner = "Agent 2"
            winner_num = 2
        else:
            winner = "Draw"
            winner_num = 0
        
        results.append({
            'episode': episode + 1,
            'winner': winner_num,
            'reward1': episode_reward1,
            'reward2': episode_reward2,
            'steps': step
        })
        
        print(f"Episode {episode + 1:2d}: {winner:8s} | "
              f"Rewards: {episode_reward1:7.2f}/{episode_reward2:7.2f} | "
              f"Steps: {step:3d}")
    
    # Summary statistics
    agent1_wins = sum(1 for r in results if r['winner'] == 1)
    agent2_wins = sum(1 for r in results if r['winner'] == 2)
    draws = sum(1 for r in results if r['winner'] == 0)
    
    avg_reward1 = np.mean([r['reward1'] for r in results])
    avg_reward2 = np.mean([r['reward2'] for r in results])
    avg_steps = np.mean([r['steps'] for r in results])
    
    print(f"\nEvaluation Summary:")
    print(f"  Agent 1 wins: {agent1_wins:2d} ({agent1_wins/num_episodes*100:5.1f}%)")
    print(f"  Agent 2 wins: {agent2_wins:2d} ({agent2_wins/num_episodes*100:5.1f}%)")
    print(f"  Draws:        {draws:2d} ({draws/num_episodes*100:5.1f}%)")
    print(f"  Avg rewards:  {avg_reward1:7.2f} / {avg_reward2:7.2f}")
    print(f"  Avg steps:    {avg_steps:7.1f}")
    
    env.close()
    return results

def human_vs_agent(agent_path, human_player=1, hidden_dims=[128, 128]):
    """Let a human play against a trained agent."""
    import torch
    
    env = PvPEnvironment(render_mode="human")
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Load agent with specified architecture
    agent = REINFORCEAgent(obs_dim, action_dim, hidden_dims=hidden_dims)
    agent.load(agent_path)
    agent.policy_net.eval()
    
    print("Human vs Agent mode!")
    print("Controls:")
    print("  WASD - Move")
    print("  Arrow keys - Shoot direction")
    print("  ESC - Quit")
    
    running = True
    clock = pygame.time.Clock()
    
    while running:
        obs, _ = env.reset()
        obs1, obs2 = obs
        
        episode_reward1 = 0
        episode_reward2 = 0
        
        done = False
        while not done and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False
                    break
            
            if not running:
                break
            
            # Get human input
            keys = pygame.key.get_pressed()
            
            human_action = np.array([0.0, 0.0, 0.0, 0.0])
            
            # Movement
            if keys[pygame.K_a]:
                human_action[0] = -1
            if keys[pygame.K_d]:
                human_action[0] = 1
            if keys[pygame.K_w]:
                human_action[1] = -1
            if keys[pygame.K_s]:
                human_action[1] = 1
            
            # Shooting
            if keys[pygame.K_LEFT]:
                human_action[2] = -1
            if keys[pygame.K_RIGHT]:
                human_action[2] = 1
            if keys[pygame.K_UP]:
                human_action[3] = -1
            if keys[pygame.K_DOWN]:
                human_action[3] = 1
            
            # Get agent action
            with torch.no_grad():
                if human_player == 1:
                    agent_obs = obs2
                else:
                    agent_obs = obs1
                
                agent_action_mean, _ = agent.policy_net(torch.FloatTensor(agent_obs).unsqueeze(0).to(agent.device))
                agent_action = agent_action_mean.squeeze().cpu().numpy()
            
            # Combine actions
            if human_player == 1:
                actions = [human_action, agent_action]
            else:
                actions = [agent_action, human_action]
            
            # Step environment
            (obs1, obs2), (reward1, reward2), done, info = env.step(actions)
            
            episode_reward1 += reward1
            episode_reward2 += reward2
            
            env.render()
            clock.tick(60)
        
        if running:
            # Show results
            if info["player1"]["alive"] and not info["player2"]["alive"]:
                winner = "Player 1"
            elif info["player2"]["alive"] and not info["player1"]["alive"]:
                winner = "Player 2"
            else:
                winner = "Draw"
            
            print(f"Game over! Winner: {winner}")
            print(f"Rewards: {episode_reward1:.2f} / {episode_reward2:.2f}")
            print("Press any key to play again or ESC to quit...")
            
            waiting = True
            while waiting and running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        running = False
                        waiting = False
                    elif event.type == pygame.KEYDOWN:
                        waiting = False
    
    env.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained agents')
    parser.add_argument('--agent1', type=str, required=True, help='Path to agent 1 model')
    parser.add_argument('--agent2', type=str, default=None, help='Path to agent 2 model (default: same as agent1)')
    parser.add_argument('--episodes', type=int, default=10, help='Number of evaluation episodes')
    parser.add_argument('--no-render', action='store_true', help='Disable rendering')
    parser.add_argument('--human', action='store_true', help='Human vs agent mode')
    parser.add_argument('--human-player', type=int, default=1, choices=[1, 2], help='Which player is human (1 or 2)')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[128, 128], help='Hidden layer dimensions')
    
    args = parser.parse_args()
    
    if args.agent2 is None:
        args.agent2 = args.agent1
    
    if args.human:
        human_vs_agent(args.agent1 if args.human_player == 2 else args.agent2, args.human_player, args.hidden_dims)
    else:
        evaluate_agents(args.agent1, args.agent2, args.episodes, not args.no_render, args.hidden_dims)

if __name__ == "__main__":
    import torch
    main()
