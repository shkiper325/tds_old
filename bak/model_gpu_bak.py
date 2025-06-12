import torch
import torch.nn as nn
import torch.nn.functional as F

from Vars import USE_CUDA

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

class MyRNN(nn.Module):
    def __init__(self, l):
        super(MyRNN, self).__init__()

        self.lin_1 = nn.Linear(in_features=l, out_features=l)
        self.lin_2 = nn.Linear(in_features=l, out_features=l)

        if USE_CUDA:
            self.cuda()

    def forward(self, x):
        out = torch.zeros(x[0].size())
        if USE_CUDA:
            out = out.cuda()

        for i in range(x.size()[0]):
            out = F.relu(self.lin_1(x[i] + out))
            out = self.lin_2(x[i])

        return out

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        d = 6 + 2 * 5

        self.lin_1 = nn.Linear(in_features=d, out_features=d)
        self.lin_2 = nn.Linear(in_features=d, out_features=d)
        self.lin_3 = nn.Linear(in_features=d, out_features=10)
        
        self.rnns = nn.ModuleList([MyRNN(2) for i in range(5)])

        if USE_CUDA:
            self.cuda()
        
    def forward(self, X):
        rnns_out = []
        for i in range(1, 6):
            rnns_out.append(self.rnns[i - 1](X[i]))

        x = torch.cat([X[0]] + rnns_out, dim=1)

        x = F.relu(self.lin_1(x))
        x = F.relu(self.lin_2(x))
        x = self.lin_3(x)

        x_prob = torch.softmax(x[:, 0:2], dim=1)
        x_dir_1 = torch.softmax(x[:, 2:6], dim=1)
        x_dir_2 = torch.softmax(x[:, 6:10], dim=1)

        x = torch.cat([x_prob, x_dir_1, x_dir_2], dim=1)

        return x
