# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import copy
import dataclasses
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

import buffer
import models
import utils


@dataclasses.dataclass
class Hyperparameters:
    # Generic
    batch_size: int = 256
    buffer_size: int = 1e6
    discount: float = 0.99
    target_update_freq: int = 250

    # Exploration
    buffer_size_before_training: int = 10e3
    exploration_noise: float = 0.2

    # TD3
    target_policy_noise: float = 0.2
    noise_clip: float = 0.3

    # Encoder Loss
    dyn_weight: float = 1
    reward_weight: float = 0.1
    done_weight: float = 0.1

    # Replay Buffer (LAP)
    prioritized: bool = True
    alpha: float = 0.4
    min_priority: float = 1
    enc_horizon: int = 5
    Q_horizon: int = 3

    # Encoder Model
    zs_dim: int = 512
    zsa_dim: int = 512
    za_dim: int = 256
    enc_hdim: int = 512
    enc_activ: str = 'elu'
    enc_lr: float = 1e-4
    enc_wd: float = 1e-4
    pixel_augs: bool = True
    
    # Value Model
    value_hdim: int = 512
    value_activ: str = 'elu'
    value_lr: float = 3e-4
    value_wd: float = 1e-4
    value_grad_clip: float = 20

    # Policy Model
    policy_hdim: int = 512
    policy_activ: str = 'relu'
    policy_lr: float = 3e-4
    policy_wd: float = 1e-4
    gumbel_tau: float = 10
    pre_activ_weight: float = 1e-5

    # Reward model
    num_bins: int = 65
    lower: float = -10
    upper: float = 10

    def __post_init__(self): utils.enforce_dataclass_type(self)


class Agent:
    def __init__(self, obs_shape: tuple, action_dim: int, max_action: float, pixel_obs: bool, discrete: bool, 
        device: torch.device, history: int=1, hp: Dict=dict()):
        self.name = 'MR.Q'    

        self.hp = Hyperparameters(**hp)
        utils.set_instance_vars(self.hp, self)
        self.device = device

        if discrete: # Scale action noise since discrete actions are [0,1] and continuous actions are [-1,1].
            self.exploration_noise *= 0.5
            self.noise_clip *= 0.5
            self.target_policy_noise *= 0.5

        self.replay_buffer = buffer.ReplayBuffer(
            obs_shape, action_dim, max_action, pixel_obs, self.device,
            history, max(self.enc_horizon, self.Q_horizon), self.buffer_size, self.batch_size,
            self.prioritized, initial_priority=self.min_priority)

        self.encoder = models.Encoder(obs_shape[0] * history, action_dim, pixel_obs, 
            self.num_bins, self.zs_dim, self.za_dim, self.zsa_dim,
            self.enc_hdim, self.enc_activ).to(self.device)
        self.encoder_optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=self.enc_lr, weight_decay=self.enc_wd)
        self.encoder_target = copy.deepcopy(self.encoder)

        self.policy = models.Policy(action_dim, discrete, self.gumbel_tau, self.zs_dim,
            self.policy_hdim, self.policy_activ).to(self.device)
        self.policy_optimizer = torch.optim.AdamW(self.policy.parameters(), lr=self.policy_lr, weight_decay=self.policy_wd)
        self.policy_target = copy.deepcopy(self.policy)

        self.value = models.Value(self.zsa_dim, self.value_hdim, self.value_activ).to(self.device)
        self.value_optimizer = torch.optim.AdamW(self.value.parameters(), lr=self.value_lr, weight_decay=self.value_wd)
        self.value_target = copy.deepcopy(self.value)

        # Used by reward prediction
        self.two_hot = TwoHot(self.device, self.lower, self.upper, self.num_bins)
        self.gammas = torch.zeros(1, self.Q_horizon, 1, device=self.device)
        discount = 1
        for t in range(self.Q_horizon):
            self.gammas[:,t] = discount
            discount *= self.discount
        self.discount = discount

        # Environment properties
        self.pixel_obs = pixel_obs
        self.state_shape = self.replay_buffer.state_shape # This includes history, horizon, channels, etc. 
        self.discrete = discrete
        self.action_dim = action_dim
        self.max_action = max_action

        # Tracked values
        self.reward_scale, self.target_reward_scale = 1, 0
        self.training_steps = 0


    def select_action(self, state: np.array, use_exploration: bool=True):
        if self.replay_buffer.size < self.buffer_size_before_training and use_exploration:
            return None # Sample random action from environment instead.

        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float, device=self.device).reshape(-1, *self.state_shape)
            zs = self.encoder.zs(state)
            action = self.policy.act(zs)
            if use_exploration: action += torch.randn_like(action) * self.exploration_noise
            return int(action.argmax()) if self.discrete else action.clamp(-1,1).cpu().data.numpy().flatten() * self.max_action


    def train(self):
        if self.replay_buffer.size <= self.buffer_size_before_training: return

        self.training_steps += 1

        if (self.training_steps-1) % self.target_update_freq == 0:
            self.policy_target.load_state_dict(self.policy.state_dict())
            self.value_target.load_state_dict(self.value.state_dict())
            self.encoder_target.load_state_dict(self.encoder.state_dict())
            self.target_reward_scale = self.reward_scale
            self.reward_scale = self.replay_buffer.reward_scale()

            for _ in range(self.target_update_freq):
                state, action, next_state, reward, not_done = self.replay_buffer.sample()
                state, next_state = maybe_augment_state(state, next_state, self.pixel_obs, self.pixel_augs)
                self.train_encoder(state, action, next_state, reward, not_done, self.replay_buffer.env_terminates)

        state, action, next_state, reward, not_done = self.replay_buffer.sample(horizon=self.Q_horizon)
        state, next_state = maybe_augment_state(state, next_state, self.pixel_obs, self.pixel_augs)
        reward = multi_step_reward(reward, self.gammas)

        Q, Q_target = self.train_RL(state, action, next_state, reward, not_done, 
            self.reward_scale, self.target_reward_scale)

        if self.prioritized:
            priority = (Q - Q_target.expand(-1,2)).abs().max(1).values
            priority = priority.clamp(min=self.min_priority).pow(self.alpha)
            self.replay_buffer.update_priority(priority)


    def train_encoder(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor,
        reward: torch.Tensor, not_done: torch.Tensor, env_terminates: bool):
        with torch.no_grad():
            encoder_target = self.encoder_target.zs(next_state.reshape(-1, *self.state_shape)).reshape(state.shape[0], -1, self.zs_dim)

        pred_zs = self.encoder.zs(state[:,0])
        prev_not_done = 1 # In subtrajectories with termination, mask out losses after termination.
        encoder_loss = 0 # Loss is accumluated over enc_horizon.

        for i in range(self.enc_horizon):
            pred_d, pred_zs, pred_r = self.encoder.model_all(pred_zs, action[:,i])
                        
            dyn_loss = masked_mse(pred_zs, encoder_target[:,i], prev_not_done)
            reward_loss = (self.two_hot.cross_entropy_loss(pred_r, reward[:,i]) * prev_not_done).mean()
            done_loss = masked_mse(pred_d, 1. - not_done[:,i].reshape(-1,1), prev_not_done) if env_terminates else 0

            encoder_loss = encoder_loss + self.dyn_weight * dyn_loss + self.reward_weight * reward_loss + self.done_weight * done_loss
            prev_not_done = not_done[:,i].reshape(-1,1) * prev_not_done # Adjust mask

        self.encoder_optimizer.zero_grad(set_to_none=True)
        encoder_loss.backward()
        self.encoder_optimizer.step()


    def train_RL(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor,
        reward: torch.Tensor, not_done: torch.Tensor, reward_scale: float, target_reward_scale: float):
        with torch.no_grad():
            next_zs = self.encoder_target.zs(next_state)

            noise = (torch.randn_like(action) * self.target_policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = realign(self.policy_target.act(next_zs) + noise, self.discrete) # Clips to (-1,1) OR one_hot of argmax.

            next_zsa = self.encoder_target(next_zs, next_action)
            Q_target = self.value_target(next_zsa).min(1,keepdim=True).values
            Q_target = (reward + not_done * self.discount * Q_target * target_reward_scale)/reward_scale

            zs = self.encoder.zs(state)
            zsa = self.encoder(zs, action)

        Q = self.value(zsa)
        value_loss = F.smooth_l1_loss(Q, Q_target.expand(-1,2))

        self.value_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.value_grad_clip)
        self.value_optimizer.step()

        policy_action, pre_activ = self.policy(zs)
        zsa = self.encoder(zs, policy_action)
        Q = self.value(zsa)
        policy_loss = -Q.mean() + self.pre_activ_weight * pre_activ.pow(2).mean()

        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()
        
        return Q, Q_target


    def save(self, save_folder: str):
        # Save models/optimizers
        models = [
            'encoder', 'encoder_target', 'encoder_optimizer',
            'policy', 'policy_target', 'policy_optimizer',
            'value', 'value_target', 'value_optimizer'
        ]
        for k in models: torch.save(self.__dict__[k].state_dict(), f'{save_folder}/{k}.pt')

        # Save variables
        vars = ['hp', 'reward_scale', 'target_reward_scale', 'training_steps']
        var_dict = {k: self.__dict__[k] for k in vars}
        np.save(f'{save_folder}/agent_var.npy', var_dict)

        self.replay_buffer.save(save_folder)


    def load(self, save_folder: str):
        # Load models/optimizers
        models = [
            'encoder', 'encoder_target', 'encoder_optimizer',
            'policy', 'policy_target', 'policy_optimizer',
            'value', 'value_target', 'value_optimizer'
        ]
        for k in models: self.__dict__[k].load_state_dict(torch.load(f'{save_folder}/{k}.pt', weights_only=True))

        # Load variables
        var_dict = np.load(f'{save_folder}/agent_var.npy', allow_pickle=True).item()
        for k, v in var_dict.items(): self.__dict__[k] = v

        self.replay_buffer.load(save_folder)


class TwoHot:
    def __init__(self, device: torch.device, lower: float=-10, upper: float=10, num_bins: int=101):
        self.bins = torch.linspace(lower, upper, num_bins, device=device)
        self.bins = self.bins.sign() * (self.bins.abs().exp() - 1) # symexp
        self.num_bins = num_bins


    def transform(self, x: torch.Tensor):
        diff = x - self.bins.reshape(1,-1)
        diff = diff - 1e8 * (torch.sign(diff) - 1)
        ind = torch.argmin(diff, 1, keepdim=True)

        lower = self.bins[ind]
        upper = self.bins[(ind+1).clamp(0, self.num_bins-1)]
        weight = (x - lower)/(upper - lower)

        two_hot = torch.zeros(x.shape[0], self.num_bins, device=x.device)
        two_hot.scatter_(1, ind, 1 - weight)
        two_hot.scatter_(1, (ind+1).clamp(0, self.num_bins), weight)
        return two_hot


    def inverse(self, x: torch.Tensor):
        return (F.softmax(x, dim=-1) * self.bins).sum(-1, keepdim=True)


    def cross_entropy_loss(self, pred: torch.Tensor, target: torch.Tensor):
        pred = F.log_softmax(pred, dim=-1)
        target = self.transform(target)
        return -(target * pred).sum(-1, keepdim=True)


def realign(x, discrete: bool):
    return F.one_hot(x.argmax(1), x.shape[1]).float() if discrete else x.clamp(-1,1)


def masked_mse(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
    return (F.mse_loss(x, y, reduction='none') * mask).mean()


def multi_step_reward(reward: torch.Tensor, gammas: torch.Tensor):
    return (reward * gammas).sum(1)


def maybe_augment_state(state: torch.Tensor, next_state: torch.Tensor, pixel_obs: bool, use_augs: bool):
    if pixel_obs and use_augs:
        if len(state.shape) != 5: state = state.unsqueeze(1)
        batch_size, horizon, history, height, width = state.shape

        # Group states before augmenting.
        both_state = torch.concatenate([state.reshape(-1, history, height, width), next_state.reshape(-1, history, height, width)], 0)
        both_state = shift_aug(both_state)

        state, next_state = torch.chunk(both_state, 2, 0)
        state = state.reshape(batch_size, horizon, history, height, width)
        next_state = next_state.reshape(batch_size, horizon, history, height, width)

        if horizon == 1:
            state = state.squeeze(1)
            next_state = next_state.squeeze(1)
    return state, next_state


# Random shift
def shift_aug(image: torch.Tensor, pad: int=4):
    batch_size, _, height, width = image.size()
    image = F.pad(image, (pad, pad, pad, pad), 'replicate')
    eps = 1.0 / (height + 2 * pad)

    arange = torch.linspace(-1.0 + eps, 1.0 - eps, height + 2 * pad, device=image.device, dtype=torch.float)[:height]
    arange = arange.unsqueeze(0).repeat(height, 1).unsqueeze(2)

    base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
    base_grid = base_grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    shift = torch.randint(0, 2 * pad + 1, size=(batch_size, 1, 1, 2), device=image.device, dtype=torch.float)
    shift *= 2.0 / (height + 2 * pad)
    return F.grid_sample(image, base_grid + shift, padding_mode='zeros', align_corners=False)