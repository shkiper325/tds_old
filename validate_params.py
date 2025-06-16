#!/usr/bin/env python3
"""Validate hyperparameters before training."""

import argparse
import sys

def validate_hyperparameters(args):
    """Validate that hyperparameters make sense."""
    errors = []
    warnings = []
    
    # Learning rate validation
    if args.lr <= 0 or args.lr > 1:
        errors.append(f"Learning rate {args.lr} should be between 0 and 1")
    if args.lr > 0.01:
        warnings.append(f"Learning rate {args.lr} is quite high, consider < 0.01")
    
    # Gamma validation
    if args.gamma <= 0 or args.gamma > 1:
        errors.append(f"Gamma {args.gamma} should be between 0 and 1")
    
    # Entropy coefficient validation  
    if args.entropy_coef < 0:
        errors.append(f"Entropy coefficient {args.entropy_coef} should be >= 0")
    if args.entropy_coef > 0.5:
        warnings.append(f"Entropy coefficient {args.entropy_coef} is quite high")
    
    # Architecture validation
    if len(args.hidden_dims) == 0:
        errors.append("At least one hidden layer is required")
    if any(dim <= 0 for dim in args.hidden_dims):
        errors.append("All hidden dimensions must be positive")
    if any(dim > 2048 for dim in args.hidden_dims):
        warnings.append("Very large hidden dimensions may be slow to train")
    
    # Environment validation
    if args.screen_size[0] <= 0 or args.screen_size[1] <= 0:
        errors.append("Screen size must be positive")
    if args.screen_size[0] < 200 or args.screen_size[1] < 200:
        warnings.append("Small screen size may make the game too cramped")
    if args.screen_size[0] > 1000 or args.screen_size[1] > 1000:
        warnings.append("Large screen size may be slow to render")
    
    if args.max_steps <= 0:
        errors.append("Max steps must be positive")
    if args.max_steps < 100:
        warnings.append("Very short episodes may not allow for learning")
    
    if args.player_speed <= 0:
        errors.append("Player speed must be positive")
    if args.player_health <= 0:
        errors.append("Player health must be positive")
    
    # Reward validation
    if args.reward_kill <= 0:
        warnings.append("Zero or negative kill reward may not encourage winning")
    if abs(args.reward_tick) > 1.0:
        warnings.append("Large time penalty may dominate other rewards")
    
    # Weapon validation
    cooldowns = [args.pistol_cooldown, args.shotgun_cooldown, args.machinegun_cooldown]
    if any(cd <= 0 for cd in cooldowns):
        errors.append("All weapon cooldowns must be positive")
    if args.projectile_speed <= 0:
        errors.append("Projectile speed must be positive")
    
    # Distance validation
    if args.optimal_distance <= 0 or args.distance_tolerance <= 0:
        errors.append("Distance parameters must be positive")
    
    # Print results
    if errors:
        print("❌ ERRORS found in hyperparameters:")
        for error in errors:
            print(f"  • {error}")
        return False
    
    if warnings:
        print("⚠️  WARNINGS about hyperparameters:")
        for warning in warnings:
            print(f"  • {warning}")
        print()
    
    print("✅ Hyperparameters validation passed!")
    
    # Print summary
    print(f"\n📋 Training Configuration Summary:")
    print(f"  Episodes: {args.episodes}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Network: {args.hidden_dims}")
    print(f"  Environment: {args.screen_size[0]}x{args.screen_size[1]}, {args.max_steps} steps")
    print(f"  Rewards: Kill={args.reward_kill}, Damage={args.reward_damage}, Hit={args.reward_hit}")
    print(f"  Weapons: Pistol={args.pistol_cooldown}ms, Shotgun={args.shotgun_cooldown}ms, MG={args.machinegun_cooldown}ms")
    
    return True

if __name__ == "__main__":
    # Import the same argument parser from train.py
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    
    from train import main
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Validate hyperparameters')
    # Add all the same arguments as train.py
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--entropy-coef', type=float, default=0.01)
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[128, 128])
    parser.add_argument('--grad-clip', type=float, default=0.5)
    parser.add_argument('--screen-size', type=int, nargs=2, default=[350, 350])
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--player-speed', type=float, default=450.0)
    parser.add_argument('--player-health', type=int, default=3)
    parser.add_argument('--reward-damage', type=float, default=1.0)
    parser.add_argument('--reward-hit', type=float, default=-1.0)
    parser.add_argument('--reward-kill', type=float, default=20.0)
    parser.add_argument('--reward-tick', type=float, default=-0.01)
    parser.add_argument('--reward-wall', type=float, default=-0.5)
    parser.add_argument('--reward-distance', type=float, default=0.3)
    parser.add_argument('--optimal-distance', type=float, default=200.0)
    parser.add_argument('--distance-tolerance', type=float, default=150.0)
    parser.add_argument('--pistol-cooldown', type=int, default=250)
    parser.add_argument('--shotgun-cooldown', type=int, default=750)
    parser.add_argument('--machinegun-cooldown', type=int, default=100)
    parser.add_argument('--projectile-speed', type=float, default=300.0)
    
    args = parser.parse_args()
    
    if not validate_hyperparameters(args):
        sys.exit(1)
