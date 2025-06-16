"""Main training script for two-agent PvP environment."""

import os
import argparse
from pvp_environment import PvPEnvironment
from reinforce_agent import REINFORCEAgent, TrainingManager

def main():
    parser = argparse.ArgumentParser(
        description='Train two agents in PvP environment',
        epilog='''
Examples:
  Basic training:
    python train.py --episodes 1000 --lr 0.001 --save-interval 100

  Aggressive setup:
    python train.py --episodes 1000 --lr 0.001 --save-interval 100 --reward-kill 50.0 --player-speed 600

  Tactical setup:
    python train.py --episodes 1000 --lr 0.001 --save-interval 100 --reward-distance 1.0 --optimal-distance 150

  Large network:
    python train.py --episodes 1000 --lr 0.001 --save-interval 100 --hidden-dims 256 256 128

See hyperparameter_examples.md for more detailed examples.
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--episodes', type=int, required=True, help='Number of training episodes')
    parser.add_argument('--lr', type=float, required=True, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--entropy-coef', type=float, default=0.01, help='Entropy coefficient')
    parser.add_argument('--render-interval', type=int, default=None, help='Render every N episodes')
    parser.add_argument('--save-interval', type=int, required=True, help='Save checkpoint every N episodes')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint prefix')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose mode with episode rendering')
    parser.add_argument('--tensorboard-dir', type=str, default='tb', help='TensorBoard log directory')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    
    # Neural network architecture
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[128, 128], help='Hidden layer dimensions')
    parser.add_argument('--grad-clip', type=float, default=0.5, help='Gradient clipping value')
    
    # Environment parameters
    parser.add_argument('--screen-size', type=int, nargs=2, default=[350, 350], help='Screen size [width, height]')
    parser.add_argument('--max-steps', type=int, default=500, help='Maximum steps per episode')
    parser.add_argument('--player-speed', type=float, default=450.0, help='Player movement speed')
    parser.add_argument('--player-health', type=int, default=3, help='Player maximum health')
    
    # Reward system parameters
    parser.add_argument('--reward-damage', type=float, default=1.0, help='Reward for dealing damage')
    parser.add_argument('--reward-hit', type=float, default=-1.0, help='Penalty for taking damage')
    parser.add_argument('--reward-kill', type=float, default=20.0, help='Reward for killing enemy')
    parser.add_argument('--reward-tick', type=float, default=-0.01, help='Time penalty per step')
    parser.add_argument('--reward-wall', type=float, default=-0.5, help='Penalty for being near walls')
    parser.add_argument('--reward-distance', type=float, default=0.3, help='Distance reward weight')
    parser.add_argument('--optimal-distance', type=float, default=200.0, help='Optimal distance between players')
    parser.add_argument('--distance-tolerance', type=float, default=150.0, help='Distance tolerance for rewards')
    
    # Weapon parameters
    parser.add_argument('--pistol-cooldown', type=int, default=250, help='Pistol cooldown in ms')
    parser.add_argument('--shotgun-cooldown', type=int, default=750, help='Shotgun cooldown in ms')
    parser.add_argument('--machinegun-cooldown', type=int, default=100, help='Machine gun cooldown in ms')
    parser.add_argument('--projectile-speed', type=float, default=300.0, help='Projectile speed')

    args = parser.parse_args()
    
    # Create environment
    render_mode = "human" if args.render_interval or args.verbose else None
    env = PvPEnvironment(
        render_mode=render_mode,
        screen_size=tuple(args.screen_size),
        max_steps=args.max_steps,
        player_speed=args.player_speed,
        player_health=args.player_health,
        reward_params={
            'damage': args.reward_damage,
            'hit': args.reward_hit,
            'kill': args.reward_kill,
            'tick': args.reward_tick,
            'wall': args.reward_wall,
            'distance': args.reward_distance,
            'optimal_distance': args.optimal_distance,
            'distance_tolerance': args.distance_tolerance
        },
        weapon_params={
            'pistol_cooldown': args.pistol_cooldown,
            'shotgun_cooldown': args.shotgun_cooldown,
            'machinegun_cooldown': args.machinegun_cooldown,
            'projectile_speed': args.projectile_speed
        }
    )
    
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
        entropy_coef=args.entropy_coef,
        hidden_dims=args.hidden_dims,
        grad_clip=args.grad_clip
    )
    
    agent2 = REINFORCEAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        lr=args.lr,
        gamma=args.gamma,
        entropy_coef=args.entropy_coef,
        hidden_dims=args.hidden_dims,
        grad_clip=args.grad_clip
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint_path1 = os.path.join(args.checkpoint_dir, f"{args.resume}_agent1.pth")
        checkpoint_path2 = os.path.join(args.checkpoint_dir, f"{args.resume}_agent2.pth")
        if os.path.exists(checkpoint_path1):
            print(f"Resuming from checkpoint: {args.resume}")
            agent1.load(checkpoint_path1)
            agent2.load(checkpoint_path2)
        else:
            print(f"Checkpoint {args.resume} not found, starting fresh")
    
    # Create training manager
    trainer = TrainingManager(
        env=env,
        agent1=agent1,
        agent2=agent2,
        max_episodes=args.episodes,
        tensorboard_dir=args.tensorboard_dir,
        checkpoint_dir=args.checkpoint_dir
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
