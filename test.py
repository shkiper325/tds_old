import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from stable_baselines import DQN

from tqdm import tqdm
import numpy as np

from env import Env

ROLLOUTS = 500

def main():
    env = Env(0)

    ticks_1 = 0
    for _ in tqdm(range(ROLLOUTS)):
        obs = env.reset()
        done = False
        while not done:
            ticks_1 += 1
            action = np.random.randint(32)
            obs, reward, done, info = env.step(action)

    print('Random:', ticks_1 / ROLLOUTS)

    model = DQN.load("normal2.zip")

    ticks_2 = 0
    for _ in tqdm(range(ROLLOUTS)):
        obs = env.reset()
        done = False
        while not done
            ticks_2 += 1
            action, _state = model.predict(obs)
            obs, reward, done, info = env.step(action)

    print('Normal:',ticks_2 / ROLLOUTS)

    model = DQN.load("weird2.zip")

    ticks_3 = 0
    for _ in tqdm(range(ROLLOUTS)):
        obs = env.reset()
        done = False
        while not done:
            ticks_3 += 1
            action, _state = model.predict(obs)
            obs, reward, done, info = env.step(action)

    print('Weird:',ticks_3 / ROLLOUTS)

if __name__ == '__main__':
    main()