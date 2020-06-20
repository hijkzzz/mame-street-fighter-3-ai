import numpy as np
import os
import random

import torch
from torch.utils.tensorboard import SummaryWriter

from model import RNNActorCriticNetwork
from env import create_train_env
from config import get_args


def main():
    args = get_args()
    device = torch.device('cuda' if args.cuda else 'cpu')
    
    env = create_train_env(1, args.difficulty, args.macro, 'env1.mp4')
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n

    model = RNNActorCriticNetwork(input_size, output_size,
                                  args.noise_linear).to(device)
    model.eval()
    
    dummy_input = torch.rand(1, 1, *env.observation_space.shape).to(device=device)
    writer = SummaryWriter(log_dir=args.log_dir)
    writer.add_graph(model, (dummy_input, ))

if __name__ == '__main__':
    main()