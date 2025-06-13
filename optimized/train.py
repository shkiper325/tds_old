"""Main training script for two-agent PvP environment."""

import os
import argparse
from pvp_environment import PvPEnvironment
from reinforce_agent import REINFORCEAgent, TrainingManager

def main():
    parser = argparse.ArgumentParser(description='Train two agents in PvP environment')
    parser.add_argument('--episodes', type=int, default=5000, help='Number of training episodes')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--entropy-coef', type=float, default=0.01, help='Entropy coefficient')
    parser.add_argument('--render-interval', type=int, default=None, help='Render every N episodes')
    parser.add_argument('--save-interval', type=int, default=100, help='Save checkpoint every N episodes')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint prefix')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose mode with episode rendering')

    args = parser.parse_args()
    
    # Create environment
    render_mode = "human" if args.render_interval or args.verbose else None
    env = PvPEnvironment(render_mode=render_mode)
    
    # Get observation and action dimensions
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"Environment created:")
    print(f"  Observation dimension: {obs_dim}")
    print(f"  Action dimension: {action_dim}")
    print(f"  Screen size: {env.screen_size}")
    
    # Create agents
    agent1 = REINFORCEAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        lr=args.lr,
        gamma=args.gamma,
        entropy_coef=args.entropy_coef
    )
    
    agent2 = REINFORCEAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        lr=args.lr,
        gamma=args.gamma,
        entropy_coef=args.entropy_coef
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        if os.path.exists(f"{args.resume}_agent1.pth"):
            print(f"Resuming from checkpoint: {args.resume}")
            agent1.load(f"{args.resume}_agent1.pth")
            agent2.load(f"{args.resume}_agent2.pth")
        else:
            print(f"Checkpoint {args.resume} not found, starting fresh")
    
    # Create training manager
    trainer = TrainingManager(
        env=env,
        agent1=agent1,
        agent2=agent2,
        max_episodes=args.episodes
    )
    
    # Start training
    try:
        trainer.train(
            save_interval=args.save_interval,
            render_interval=args.render_interval,
            verbose=args.verbose
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer.save_checkpoint("interrupted_model")
    finally:
        env.close()

if __name__ == "__main__":
    main()
