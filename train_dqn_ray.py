from pprint import pprint

from ray.rllib.agents.dqn import DQNTrainer, DEFAULT_CONFIG
import ray
from ray.rllib.models import ModelCatalog

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor

from env import Env

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()

        self.lin_1 = nn.Linear(in_features=12, out_features=162)
        self.lin_2 = nn.Linear(in_features=162, out_features=162)
        self.lin_3 = nn.Linear(in_features=162, out_features=162)
        
    def forward(self, x):
        x = F.relu(self.lin_1(x))
        x = F.relu(self.lin_2(x))
        x = self.lin_3(x)

        print('###########################################################')
        print('###########################################################')
        print('In model:')
        print(x)
        print('###########################################################')
        print('###########################################################')

        return x

def main():
    ray.init()

    ModelCatalog.register_custom_model("simple_model", SimpleModel)

    config = DEFAULT_CONFIG.copy()

    config["num_workers"] = 1
    config["num_gpus"] = 1
    config["framework"] = "torch"
    config["horizon"] = 1000

    config["n_step"] = 2
    config["noisy"] = True 
    config["num_atoms"] = 20
    config["v_min"] = -10.0
    config["v_max"] = 10.0

    config["env"] = Env
    config["env_config"] = dict()

    config["model"] =  {
        "custom_model": "simple_model", \
        "custom_model_config": {},
    }

    trainer = DQNTrainer(config)

    for i in range(5):
        results = trainer.train()

    pprint(results)

if __name__ == '__main__':
    main()