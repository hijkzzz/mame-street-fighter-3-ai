import torch
import torch.nn as nn
from torch.nn import init

import numpy as np

def init_weight(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        init.orthogonal_(m.weight, np.sqrt(2))
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()