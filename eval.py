import numpy as np
import os
import random

import torch
from torch.distributions.categorical import Categorical

from model import RNNActorCriticNetwork
from env import create_train_env
from config import get_args


def get_action(model, device, state, hidden):
    with torch.no_grad():
        state = torch.Tensor(state).to(device).float()
        action_probs, value, hidden = model(state, hidden)
        action_dist = Categorical(action_probs)
        action = action_dist.sample()

    return action.cpu().numpy().squeeze(), value.cpu().numpy().squeeze(
    ), action_probs.cpu().numpy().squeeze(), hidden


def main():
    args = get_args()

    device = torch.device('cuda' if args.cuda else 'cpu')
    env = create_train_env(1, args.difficulty, args.macro, 'env1.mp4')

    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n

    model_path = os.path.join(args.save_dir, 'policy.cpt')
    model = RNNActorCriticNetwork(input_size, output_size,
                                  args.noise_linear).to(device)
    if args.cuda:
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))

    model.eval()
    print('Testing...')

    # looping
    obs = env.reset()
    hidden = None

    sample_rall = 0
    sample_step = 0
    sample_max_stage = 0
    done = False

    while not done:
        action, _, action_probs, hidden = get_action(model, device,
                                                     obs[None,
                                                         None, :], hidden)
        obs, rew, done, info = env.step(int(action))

        sample_rall += rew
        sample_max_stage = max(sample_max_stage, info['stage'])
        sample_step += 1

    print('Max Stage: %d | Reward: %f | Total Steps: %d' \
            % (sample_max_stage, sample_rall, sample_step))


if __name__ == '__main__':
    main()
