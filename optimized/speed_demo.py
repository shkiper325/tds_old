"""Example showing how to use SPEED variable for fast rollouts."""

import time
from utils import set_simulation_speed, get_simulation_speed
from pvp_environment import PvPEnvironment
from reinforce_agent import REINFORCEAgent

def benchmark_speeds():
    """Compare different simulation speeds."""
    
    speeds = [1.0, 2.0, 5.0, 10.0]
    results = []
    
    for speed in speeds:
        print(f"\nTesting speed {speed}x...")
        set_simulation_speed(speed)
        
        # Create environment and agents
        env = PvPEnvironment()
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        agent1 = REINFORCEAgent(obs_dim, action_dim)
        agent2 = REINFORCEAgent(obs_dim, action_dim)
        
        # Time a single episode
        start_time = time.time()
        
        obs, _ = env.reset()
        obs1, obs2 = obs
        
        steps = 0
        done = False
        while not done and steps < 1000:  # Limit steps for consistent comparison
            action1 = agent1.act(obs1)
            action2 = agent2.act(obs2)
            
            (obs1, obs2), (reward1, reward2), done, info = env.step([action1, action2])
            steps += 1
        
        episode_time = time.time() - start_time
        results.append((speed, episode_time, steps))
        
        print(f"  Speed: {speed}x, Time: {episode_time:.3f}s, Steps: {steps}")
        
        env.close()
    
    print(f"\nResults:")
    print(f"{'Speed':<8} {'Time':<8} {'Steps':<8} {'Steps/sec':<12}")
    print("-" * 40)
    for speed, time_taken, steps in results:
        steps_per_sec = steps / time_taken if time_taken > 0 else 0
        print(f"{speed:<8.1f} {time_taken:<8.3f} {steps:<8d} {steps_per_sec:<12.1f}")

def fast_evaluation_example():
    """Example of using high speed for fast model evaluation."""
    
    print("\nFast evaluation example:")
    
    # Set high speed for fast rollouts
    set_simulation_speed(10.0)
    print(f"Simulation speed set to {get_simulation_speed()}x")
    
    env = PvPEnvironment()
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent1 = REINFORCEAgent(obs_dim, action_dim)
    agent2 = REINFORCEAgent(obs_dim, action_dim)
    
    # Quick evaluation of 5 episodes
    total_rewards = [0, 0]
    wins = [0, 0, 0]  # [agent1, agent2, draws]
    
    start_time = time.time()
    
    for episode in range(5):
        obs, _ = env.reset()
        obs1, obs2 = obs
        
        episode_rewards = [0, 0]
        done = False
        
        while not done:
            action1 = agent1.act(obs1)
            action2 = agent2.act(obs2)
            
            (obs1, obs2), (reward1, reward2), done, info = env.step([action1, action2])
            
            episode_rewards[0] += reward1
            episode_rewards[1] += reward2
        
        # Track results
        total_rewards[0] += episode_rewards[0]
        total_rewards[1] += episode_rewards[1]
        
        if info["player1"]["alive"] and not info["player2"]["alive"]:
            wins[0] += 1
        elif info["player2"]["alive"] and not info["player1"]["alive"]:
            wins[1] += 1
        else:
            wins[2] += 1
    
    total_time = time.time() - start_time
    
    print(f"Evaluation completed in {total_time:.2f} seconds")
    print(f"Average rewards: Agent1={total_rewards[0]/5:.1f}, Agent2={total_rewards[1]/5:.1f}")
    print(f"Win rates: Agent1={wins[0]/5:.1f}, Agent2={wins[1]/5:.1f}, Draws={wins[2]/5:.1f}")
    
    env.close()

if __name__ == "__main__":
    print("SPEED Variable Demo")
    print("=" * 50)
    
    # Show current speed
    print(f"Current simulation speed: {get_simulation_speed()}x")
    
    # Benchmark different speeds
    benchmark_speeds()
    
    # Example of fast evaluation
    fast_evaluation_example()
    
    print(f"\nUsage examples:")
    print(f"  python train.py --speed 5.0    # 5x faster training")
    print(f"  python train.py --speed 0.5    # 2x slower (for debugging)")
    print(f"  python evaluate.py --agent1 model.pth --speed 10.0  # Fast evaluation")
