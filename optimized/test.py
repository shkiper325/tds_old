"""Quick test to verify the optimized implementation works correctly."""

import numpy as np
import pygame
from pvp_environment import PvPEnvironment
from reinforce_agent import REINFORCEAgent

def test_environment():
    """Test basic environment functionality."""
    print("Testing environment...")
    
    # Set headless mode for testing
    import os
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    
    # Create environment
    env = PvPEnvironment()
    
    # Test reset
    obs, info = env.reset()
    obs1, obs2 = obs
    
    print(f"✓ Environment created")
    print(f"  Observation shape: {obs1.shape}")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space}")
    
    # Test random actions
    total_reward1 = 0
    total_reward2 = 0
    steps = 0
    
    done = False
    while not done and steps < 100:  # Limit steps for testing
        # Random actions
        action1 = env.action_space.sample()
        action2 = env.action_space.sample()
        
        # Step
        obs, rewards, done, info = env.step([action1, action2])
        obs1, obs2 = obs
        reward1, reward2 = rewards
        
        total_reward1 += reward1
        total_reward2 += reward2
        steps += 1
        
        # Verify observation shape
        assert obs1.shape == (22,), f"Expected obs shape (22,), got {obs1.shape}"
        assert obs2.shape == (22,), f"Expected obs shape (22,), got {obs2.shape}"
    
    print(f"✓ Random episode completed")
    print(f"  Steps: {steps}")
    print(f"  Total rewards: {total_reward1:.2f}, {total_reward2:.2f}")
    print(f"  Player 1 alive: {info['player1']['alive']}")
    print(f"  Player 2 alive: {info['player2']['alive']}")
    
    env.close()
    print("✓ Environment test passed")

def test_agent():
    """Test basic agent functionality."""
    print("\nTesting agent...")
    
    # Create agent
    obs_dim = 22
    action_dim = 4
    agent = REINFORCEAgent(obs_dim, action_dim)
    
    print(f"✓ Agent created")
    print(f"  Device: {agent.device}")
    
    # Test action selection
    obs = np.random.randn(obs_dim).astype(np.float32)
    action = agent.act(obs)
    
    print(f"✓ Action selection works")
    print(f"  Action shape: {action.shape}")
    print(f"  Action range: [{action.min():.3f}, {action.max():.3f}]")
    
    # Test reward storage and update
    agent.store_reward(1.0)
    agent.store_reward(-0.5)
    agent.store_reward(0.5)
    
    policy_loss, value_loss = agent.update()
    
    print(f"✓ Update works")
    print(f"  Policy loss: {policy_loss:.4f}")
    print(f"  Value loss: {value_loss:.4f}")
    
    print("✓ Agent test passed")

def test_integration():
    """Test environment-agent integration."""
    print("\nTesting integration...")
    
    # Create environment and agents
    env = PvPEnvironment()
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent1 = REINFORCEAgent(obs_dim, action_dim)
    agent2 = REINFORCEAgent(obs_dim, action_dim)
    
    # Run a short episode
    obs, _ = env.reset()
    obs1, obs2 = obs
    
    episode_rewards = [0, 0]
    
    for step in range(50):  # Short episode
        # Get actions
        action1 = agent1.act(obs1)
        action2 = agent2.act(obs2)
        
        # Step environment
        (obs1, obs2), (reward1, reward2), done, info = env.step([action1, action2])
        
        # Store rewards
        agent1.store_reward(reward1)
        agent2.store_reward(reward2)
        
        episode_rewards[0] += reward1
        episode_rewards[1] += reward2
        
        if done:
            break
    
    # Update agents
    loss1 = agent1.update()
    loss2 = agent2.update()
    
    print(f"✓ Integration test passed")
    print(f"  Episode length: {step + 1}")
    print(f"  Episode rewards: {episode_rewards[0]:.2f}, {episode_rewards[1]:.2f}")
    print(f"  Final losses: {loss1[0]:.4f}, {loss2[0]:.4f}")
    
    env.close()

if __name__ == "__main__":
    print("Running optimized implementation tests...\n")
    
    try:
        test_environment()
        test_agent()
        test_integration()
        
        print("\n🎉 All tests passed! The optimized implementation is working correctly.")
        print("\nYou can now run:")
        print("  python train.py                    # Start training")
        print("  python train.py --render-interval 10  # Training with visualization")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
