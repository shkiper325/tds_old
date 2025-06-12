from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import torch.nn.functional as F

import gym

from env import Env

def main():
    env = Env(0)
    env = make_vec_env(Env, n_envs=32)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1000000)
    model.save("ppo")

    del model

    model = PPO.load("ppo")

    for _ in range(5):
        obs = env.reset()
        done = False
        while not done:
            action, _state = model.predict(obs)
            obs, reward, done, info = env.step(action)
            env.render()

if __name__ == '__main__':
    main()