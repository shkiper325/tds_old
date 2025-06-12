import os

import numpy as np

# import torch
# import torch.nn as nn

from Vars import *

# def save(data, fn):
#     if fn is None:
#         fn = 'checkpoint'

#     torch.save(data, fn)

# def load(fn):
#     if fn is None:
#         fn = 'checkpoint'

#     return torch.load(fn)

PI = 3.1415

def dist(x, y):
    return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)

def dirac_delta(i, n, dtype=int):
    ret =  np.zeros((n,), dtype=dtype)
    ret[i] = 1

    return ret

def angle(v_start, v_end):
    diff = np.array(v_end) - np.array(v_start)
    x, y = diff

    if x < 1e-5 and y < 1e-5:
        return 0
    else:
        angle = np.arccos(x / dist(diff, [0, 0]))
        if y < 0:
            angle = 2 * PI - angle

        return angle

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

def norm(x):
    return dist(x, [0, 0])

def normalize(l):
    ret = []

    for x in l:
        if norm(x) < 1e-5:
            ret.append([0, 0])
        else:
            n = norm(x)
            ret.append([x[0] / n, x[1] / n])

    return ret

def i_to_dir(i):
    if i == 0:
        return np.array([1, 0])
    elif i == 1:
        return np.array([1, 1]) / np.sqrt(2)
    elif i == 2:
        return np.array([0, 1])
    elif i == 3:
        return np.array([-1, 1]) / np.sqrt(2)
    elif i == 4:
        return np.array([-1, 0])
    elif i == 5:
        return np.array([-1, -1]) / np.sqrt(2)
    elif i == 6:
        return np.array([0, -1])
    elif i == 7:
        return np.array([1, -1]) / np.sqrt(2)
    else:
        print('Error')
        quit(1)

def flatten_them_all(l):
    ret = []
    for x in l:
        ret.append(np.array(x).flatten())
    ret = np.concatenate(ret)

    return ret

# def init_weights(m):
#     if type(m) in [nn.Conv2d, nn.Linear, nn.ConvTranspose2d]:
#         torch.nn.init.normal_(m.weight.data, mean=0, std=INIT_STD)
#         if m.bias is not None:
#             torch.nn.init.normal_(m.bias.data, mean=0, std=INIT_STD)
#     elif type(m) in [nn.BatchNorm2d, nn.LeakyReLU, nn.ReLU, nn.Sequential, nn.Tanh]:
#         return
#     else:
#         print('Couldn\'t init wieghts of layer with type:', type(m))