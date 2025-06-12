import torch
import torch.nn as nn
import torch.nn.functional as F

from Vars import USE_CUDA, FloatTensor

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        d = 12

        self.lin_1 = nn.Linear(in_features=d, out_features=d)
        self.lin_2 = nn.Linear(in_features=d, out_features=d)
        self.lin_3 = nn.Linear(in_features=d, out_features=18)

        if USE_CUDA:
            self.cuda()
        
    def forward(self, x):
        x = F.relu(self.lin_1(x))
        x = F.relu(self.lin_2(x))
        x = self.lin_3(x)

        x_prob = torch.softmax(x[:, :2], dim=1)
        x_dir_1 = torch.softmax(x[:, 2:10], dim=1)
        x_dir_2 = torch.softmax(x[:, 10:], dim=1)

        x = torch.cat([x_prob, x_dir_1, x_dir_2], dim=1)

        return x
