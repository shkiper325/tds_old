import os

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from cmaes import CMA

from model_cmaes import Model
from env import Env

def save(vec, fn):
    if fn is None:
        fn = 'checkpoint'

    np.save(fn, vec)

def load(fn):
    if fn is None:
        fn = 'checkpoint'

    return np.load(fn)

def get_last_epoch():
    fns = os.listdir('models')

    if len(fns) == 0:
        return -1

    epochs = [int(x.split('.')[0]) for x in fns]
    return max(epochs)

def slicing_mean(n, data):
    window = np.ones(shape=(n,)) / n

    ret = []
    for i in range(len(data) - n):
        ret.append(np.sum(window * data[i:i+n]))

    return ret

POPULATION_SIZE = 200
INIT_STD = 0.02
MAX_DEPTH = 2000

def rollout(vec):
    env = Env()
    model = Model()

    cum_reward = 0

    obs = env.reset()
    for i in range(MAX_DEPTH):
        action = np.argmax(model(obs, vec))
        obs, reward, done, info = env.step(action)
        
        cum_reward += reward
        if done:
            break

    return cum_reward

def train():
    weights_size = 1860

    last_epoch = get_last_epoch()
    if last_epoch >= 0:
        vec = load(os.path.join('models', str(last_epoch)))
        print('Checkpiont loaded')
    else:
        vec = np.random.randn(weights_size) * INIT_STD
        print('Cold start')

    optimizer = CMA(mean=vec, sigma=1.3, population_size=POPULATION_SIZE)

    for generation in tqdm(range(50)):
        solutions = []
        for _ in tqdm(range(optimizer.population_size)):
            x = optimizer.ask()
            value = -rollout(x)
            solutions.append((x, value))
        optimizer.tell(solutions)

        best = sorted(solutions, key=lambda z: z[1])[-1][0]
        save(best, os.path.join('models', str(generation)))

        print(sorted(solutions, key=lambda z: z[1])[-1][1])

def test(epoch=None):
    last_epoch = get_last_epoch() if epoch is None else 0
    vec = load(os.path.join('models', str(last_epoch) + '.npy'))
    return rollout(vec)

if __name__ == '__main__':
    train()

    result = 0
    for i in tqdm(range(100)):
        result += test(0)
    print('0:', result / 100)

    result = 0
    for i in tqdm(range(100)):
        result += test(49)
    print('49:', result / 100)
