import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import math
from torch.nn import init

import rnn


class NoisyLinear(nn.Module):
    """Factorised Gaussian NoisyNet"""
    def __init__(self, in_features, out_features, sigma0=1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.noisy_weight = nn.Parameter(
            torch.Tensor(out_features, in_features))
        self.noisy_bias = nn.Parameter(torch.Tensor(out_features))
        self.noise_std = sigma0 / math.sqrt(self.in_features)

        self.reset_parameters()
        self.register_noise()

    def register_noise(self):
        in_noise = torch.FloatTensor(self.in_features)
        out_noise = torch.FloatTensor(self.out_features)
        noise = torch.FloatTensor(self.out_features, self.in_features)
        self.register_buffer('in_noise', in_noise)
        self.register_buffer('out_noise', out_noise)
        self.register_buffer('noise', noise)

    def sample_noise(self):
        self.in_noise.normal_(0, self.noise_std)
        self.out_noise.normal_(0, self.noise_std)
        self.noise = torch.mm(self.out_noise.view(-1, 1),
                              self.in_noise.view(1, -1))

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.noisy_weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            self.noisy_bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """
        Note: noise will be updated if x is not volatile
        """
        normal_y = nn.functional.linear(x, self.weight, self.bias)
        if self.training:
            # update the noise once per update
            self.sample_noise()

        noisy_weight = self.noisy_weight * self.noise
        noisy_bias = self.noisy_bias * self.out_noise
        noisy_y = nn.functional.linear(x, noisy_weight, noisy_bias)
        return noisy_y + normal_y


class BaseConv(nn.Module):
    def __init__(self, in_size, noise_linear=False):
        super(BaseConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_size, 32, 3, stride=2),
                                  nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                                  nn.Conv2d(32, 64, 3, stride=2),
                                  nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                  nn.Conv2d(64, 128, 3, stride=2),
                                  nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                                  nn.Conv2d(128, 128, 3, stride=2),
                                  nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                                  nn.Conv2d(128, 128, 3, stride=1),
                                  nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                                  nn.Conv2d(128, 128, 3, stride=1),
                                  nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                                  nn.Conv2d(128, 128, 3, stride=1),
                                  nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                                  nn.Conv2d(128, 128, 2, stride=1),
                                  nn.BatchNorm2d(128), nn.ReLU(inplace=True))

    def forward(self, x):
        if len(x.shape) == 5:
            step = x.shape[0]
            batch = x.shape[1]
            x = x.view(-1, *x.shape[2:])
            x = self.conv(x)
            x = x.view(step, batch, -1)

        elif len(x.shape) == 4:
            x = self.conv(x)
            x = x.view(x.shape[0], -1)
        return x


class RNNActorCriticNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, use_noisy_net):
        super(RNNActorCriticNetwork, self).__init__()
        if use_noisy_net:
            print('Use NoisyNet')
            linear = NoisyLinear
        else:
            linear = nn.Linear

        self.conv = BaseConv(num_inputs)
        self.rnn = rnn.GRU(512, 1024)
        self.critic_linear = nn.Sequential(linear(1024, 256),
                                           nn.ReLU(inplace=True),
                                           linear(256, 1))
        self.actor = nn.Sequential(linear(1024, 256), nn.ReLU(inplace=True),
                                   linear(256, num_actions), nn.Softmax(dim=2))

    def forward(self, x, hidden=None, masks=None):
        x = self.conv(x)
        x, hidden = self.rnn(x, hidden, masks)
        p = self.actor(x)
        v = self.critic_linear(x)
        return p, v, hidden
