import numpy as np
import os
import random
import collections

import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.multiprocessing import Pipe
from torch.utils.tensorboard import SummaryWriter

from model import RNNActorCriticNetwork
from env import create_train_env
from venv import SubprocVecEnv
from config import get_args
from utils import init_weight


def get_action(model, device, state, hidden):
    with torch.no_grad():
        state = torch.Tensor(state).to(device).float()
        action_probs, values, hidden = model(state, hidden)
        action_dist = Categorical(action_probs)
        action = action_dist.sample()
        log_probs = action_dist.log_prob(action)

    return action.cpu().numpy().squeeze(), values.cpu().numpy().squeeze(
    ), log_probs.cpu().numpy().squeeze(), hidden


def make_train_data(reward, done, value, gamma, gae_lambda, num_step,
                    num_worker, use_gae):
    returns = np.zeros((num_step, num_worker))

    # Discounted Return
    if use_gae:
        advantage = np.zeros((num_step + 1, num_worker))
        for t in range(num_step - 1, -1, -1):
            delta = reward[t, :] + gamma * value[t + 1, :] * (
                1 - done[t, :]) - value[t, :]
            advantage[t, :] = delta + advantage[t + 1] * gamma * gae_lambda * (
                1 - done[t, :])

            returns[t, :] = advantage[t, :] + value[t, :]
        advantage = advantage[:-1, :]
    else:
        running_add = value[:, -1]
        for t in range(num_step - 1, -1, -1):
            running_add = reward[t, :] + gamma * running_add * (1 - done[t, :])
            returns[t, :] = running_add

        advantage = returns - value[:-1, :]

    return returns, advantage


def train_model(args, device, output_size, model, model_optimizer, total_obs,
                total_target, total_action, total_adv, total_log_prob, init_hidden,
                total_max_stage, total_done):
    if args.rew_norm:
        total_adv = (total_adv - total_adv.mean()) / (total_adv.std() + 1e-8)

    actor_loss_ = []
    critic_loss_ = []
    entropy_ = []
    
    smooth_l1_loss = nn.SmoothL1Loss(reduction='none')

    for i in range(args.epoch):
        for j in range(args.num_batch):
            slic = sorted(
                random.sample(range(args.num_worker),
                              args.num_worker // args.num_batch))
            
            # copy to device
            obs = torch.FloatTensor(total_obs)[:, slic].to(device)
            target = torch.FloatTensor(total_target)[:, slic].to(device)
            action = torch.LongTensor(total_action)[:, slic].to(device)
            adv = torch.FloatTensor(total_adv)[:, slic].to(device)
            log_prob_old = torch.FloatTensor(total_log_prob)[:, slic].to(device)
            if args.use_grad_ratio:
                grad_ratio = torch.FloatTensor(total_max_stage)[:, slic].to(device) / 20 + 1
            mask = 1 - torch.FloatTensor(total_done)[:, slic].to(device)

            # PPO
            hidden = init_hidden[:, slic] if not init_hidden is None else None

            action_probs, values, _ = model(obs, hidden, mask)
            m = Categorical(action_probs)
            log_prob = m.log_prob(action)

            ratio = torch.exp(log_prob - log_prob_old)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - args.eps,
                                1.0 + args.eps) * adv

            coef = grad_ratio if args.use_grad_ratio else 1
            actor_loss = (-torch.min(surr1, surr2) * coef).mean()

            critic_loss = (smooth_l1_loss(values.squeeze(), target)).mean()
            entropy = m.entropy().mean()

            model_optimizer.zero_grad()
            ppo_loss = actor_loss + critic_loss - args.entropy_coef * entropy
            ppo_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           args.max_grad_norm)
            model_optimizer.step()

            # log
            actor_loss_.append(actor_loss.item())
            critic_loss_.append(critic_loss.item())
            entropy_.append(entropy.item())

    return np.mean(actor_loss_), np.mean(critic_loss_), np.mean(entropy_)


def main():
    args = get_args()
    device = torch.device('cuda' if args.cuda else 'cpu')

    def make_env(rank):
        def _thunk():
            return create_train_env(rank, args.difficulty, args.macro)

        return _thunk

    envs = SubprocVecEnv([make_env(i) for i in range(args.num_worker)])

    input_size = envs.observation_space.shape[0]
    feature_shape = envs.observation_space.shape
    output_size = envs.action_space.n

    # save path
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    model_path = os.path.join(args.save_dir, 'policy.cpt')
    optimizer_path = os.path.join(args.save_dir, 'optimizer.cpt')

    # model
    model = RNNActorCriticNetwork(input_size, output_size,
                                  args.noise_linear).to(device)
    model.apply(init_weight)
    model_optimizer = optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    if args.load_model:
        print('Loading pretrained models...')
        model.load_state_dict(torch.load(model_path, map_location=args.device))
        model_optimizer.load_state_dict(
            torch.load(optimizer_path, map_location=args.device))

    model.train()
    print('Training...')
    # logger
    writer = SummaryWriter(log_dir=args.log_dir)

    global_update = 0
    global_step = 0

    sample_episode = 0
    sample_step = np.zeros(args.num_worker)
    sample_rall = np.zeros(args.num_worker)
    sample_max_stage = np.ones(args.num_worker)

    latest_dones = collections.deque(maxlen=args.num_worker)
    best_mean_stage = 1

    # data
    total_obs = np.zeros((args.num_step + 1, args.num_worker) + feature_shape)
    total_reward = np.zeros((args.num_step, args.num_worker))
    total_done = np.zeros((args.num_step, args.num_worker))
    total_action = np.zeros((args.num_step, args.num_worker))
    total_log_prob = np.zeros((args.num_step, args.num_worker))
    total_max_stage = np.zeros((args.num_step, args.num_worker))

    # looping
    obs = envs.reset()
    hidden = None

    while True:
        global_step += args.num_worker * args.num_step
        global_update += 1

        # Step 1. n-step rollout
        init_hidden = hidden
        for t in range(args.num_step):
            actions, _, log_prob, hidden = get_action(model, device,
                                                      obs[None, :], hidden)
            next_obs, rews, dones, infos = envs.step(actions)

            # save transitions
            total_obs[t, :] = obs
            total_action[t, :] = actions
            total_reward[t, :] = rews
            total_log_prob[t, :] = log_prob
            total_max_stage[t, :] = sample_max_stage
            if args.mask:
                total_done[t, :] = dones

            obs = next_obs

            # log
            sample_episode += np.sum(dones)
            sample_rall += rews
            sample_step += 1
            for i in range(args.num_worker):
                sample_max_stage[i] = max(sample_max_stage[i],
                                          infos[i]['stage'])
                if sample_max_stage[i] == 10 and infos[i]['stage'] < 10:
                    print('Pass all!!!')

            # done
            for i, done in enumerate(dones):
                if done:
                    if args.mask:
                        hidden[:, i] = 0

                    latest_dones.append(
                        (sample_step[i], sample_rall[i], sample_max_stage[i]))
                    sample_step[i] = 0
                    sample_rall[i] = 0
                    sample_max_stage[i] = 1

                    if sample_episode >= args.num_worker:
                        sample_steps, sample_ralls, sample_max_stages = list(
                            zip(*list(latest_dones)))

                        writer.add_scalar('data/reward_per_epi',
                                          np.mean(sample_ralls), global_update)
                        writer.add_scalar('data/mean_stage_per_epi',
                                          np.mean(sample_max_stages),
                                          global_update)
                        writer.add_scalar('data/max_stage_per_epi',
                                          np.max(sample_max_stages),
                                          global_update)
                        writer.add_scalar('data/sample_episode',
                                          sample_episode, global_update)

                        if np.mean(sample_max_stages) >= best_mean_stage:
                            print('Saved models... {}'.format(global_update))
                            best_mean_stage = np.mean(sample_max_stages)
                            
                            torch.save(model.state_dict(), model_path)
                            torch.save(model_optimizer.state_dict(), optimizer_path)
                            
        # last step obs
        total_obs[-1, :] = obs

        # Step 2. make target and advantage
        _, total_value, _, _ = get_action(model, device, total_obs,
                                          init_hidden)

        # reward calculate
        target_value, adv = make_train_data(total_reward, total_done,
                                            total_value, args.ext_gamma,
                                            args.gae_lambda, args.num_step,
                                            args.num_worker, args.use_gae)

        # Step 3. Training!
        actor_loss, critic_loss, entropy = train_model(
            args, device, output_size, model, model_optimizer, total_obs[:-1],
            target_value, total_action, adv, total_log_prob, init_hidden,
            total_max_stage, total_done)

        # log
        writer.add_scalar('train/actor_loss', actor_loss, global_update)
        writer.add_scalar('train/critic_loss', critic_loss, global_update)
        writer.add_scalar('train/entropy', entropy, global_update)


if __name__ == '__main__':
    main()