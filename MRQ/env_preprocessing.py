# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from collections import deque
import dataclasses
from functools import partial
from typing import Dict

import gymnasium as gym
import numpy as np

import utils

# Used by Atari
# import ale_py
# import cv2

# Used by Dmc
# from dm_control import suite
# from dm_control.suite.wrappers import action_scale


# 1. Makes environment, sets seeds, applies wrappers.
# 2. Unifies some basic attributes like action_dim, obs_shape.
# 3. Tracks some basic information like episode timesteps and reward.
class Env:
    def __init__(self, env_name: str, seed: int=0, eval: bool=False, remove_info: bool=True):
        env_type = env_name.split('-',1)[0]
        self.env = globals()[f'{env_type}Preprocessing'](env_name, seed, eval) # Calls the corresponding preprocessing class.
        
        # Copy instance variables
        for k in ['offline', 'pixel_obs', 'obs_shape', 'history', 'max_ep_timesteps', 'action_space']:
            self.__dict__[k] = self.env.__dict__[k]

        # Only used for printing
        self.env_name = env_name
        self.seed = seed

        self.action_space.seed(seed)
        self.discrete = self.action_space.__class__.__name__ == 'Discrete'
        self.action_dim = self.action_space.n if self.discrete else self.action_space.shape[0]
        self.max_action = 1 if self.discrete else float(self.action_space.high[0]) 

        self.remove_info = remove_info
        self.ep_total_reward = 0 
        self.ep_timesteps = 0 
        self.ep_num = 1


    def reset(self):
        self.ep_total_reward = 0
        self.ep_timesteps = 0
        self.ep_num += 1

        state, info = self.env.reset()
        return state if self.remove_info else (state, info)


    def step(self, action: int | float):
        next_state, reward, terminated, truncated, info = self.env.step(action)

        self.ep_total_reward += reward
        self.ep_timesteps += 1

        return (next_state, reward, terminated, truncated) if self.remove_info else (next_state, reward, terminated, truncated, info)


class GymPreprocessing:
    def __init__(self, env_name: str, seed: int=0, eval: bool=False):
        self.env = gym.make(env_name.replace('Gym-', ''))
        self.env.reset(seed=seed)

        self.offline = False
        self.pixel_obs = False
        self.obs_shape = self.env.observation_space.shape
        self.history = 1
        self.max_ep_timesteps = self.env.spec.max_episode_steps
        self.action_space = self.env.action_space


    def step(self, action: int | float): 
        return self.env.step(action)


    def reset(self):
        return self.env.reset()


@dataclasses.dataclass
class DmcHyperparameters:
    action_repeat: int = 2
    # Proprioceptive tasks only
    history: int = 1
    # Visual tasks only
    visual_history: int = 3 # Overrides history.
    image_size: int = 84

    def __post_init__(self): utils.enforce_dataclass_type(self)


class DmcPreprocessing:
    def __init__(self, env_name: str, seed: int=0, eval: bool=False, hp: Dict=dict()):
        from dm_control import suite
        from dm_control.suite.wrappers import action_scale
        self.hp = DmcHyperparameters(**hp)
        utils.set_instance_vars(self.hp, self)

        self.pixel_obs = '-visual' in env_name
        self.domain, task = env_name.replace('Dmc-', '').replace('visual-', '').split('-', 1)
        self.env = suite.load(self.domain, task, task_kwargs={'random': seed}, visualize_reward=False)
        self.env = action_scale.Wrapper(self.env, minimum=-1., maximum=1.)
        self.offline = False

        if self.pixel_obs:
            self.obs_shape = (3, self.image_size, self.image_size) # The first dim (3) is color channels (RGB).
            self.history = self.visual_history
        else:
            self.obs_shape = 0
            for v in self.env.observation_spec().values():
                self.obs_shape += np.prod(v.shape)
            self.obs_shape = (int(self.obs_shape),)
            self.history = self.history

        self.max_ep_timesteps = 1000 // self.action_repeat

        self.action_space = gym.spaces.Box(
            low=-np.ones(self.env.action_spec().shape),
            high=np.ones(self.env.action_spec().shape),
            dtype=self.env.action_spec().dtype)

        self.history_queue = deque(maxlen=self.history)


    def get_obs(self, time_step: object):
        if self.pixel_obs: return self.render(self.image_size)
        return np.concatenate([v.flatten() for v in time_step.observation.values()])


    def reset(self):
        self.t = 0
        time_step = self.env.reset()

        obs = self.get_obs(time_step)
        for _ in range(self.history):
            self.history_queue.append(obs)

        return np.concatenate(self.history_queue), {}


    def step(self, action: float): 
        self.t += 1
        action = action.astype(np.float32) # This shouldn't matter but it can.

        reward = 0.0
        for _ in range(self.action_repeat):
            time_step = self.env.step(action)
            reward += time_step.reward

        obs = self.get_obs(time_step)
        self.history_queue.append(obs)
        return np.concatenate(self.history_queue), reward, False, self.t == self.max_ep_timesteps, {}


    def render(self, size: int=84, camera_id: int=0):
        camera_id = dict(quadruped=2).get(self.domain, camera_id)
        return self.env.physics.render(size, size, camera_id).transpose(2, 0, 1)


@dataclasses.dataclass
class AtariHyperparameters:
    history: int = 4
    training_reward_clipping: bool = False # Only applied during training / not on eval environment.
    max_ep_frames: int = 108e3 
    max_noops: int = 0
    action_repeat: int = 4
    terminal_lives: bool = False
    image_size: int = 84
    pool_frames: bool = True
    grayscale: bool = True
    sticky_actions: bool = True
    eval_eps: float = 1e-3

    def __post_init__(self): utils.enforce_dataclass_type(self)


class AtariPreprocessing:
    def __init__(self, env_name: str, seed: int=0, eval: bool=False, hp: Dict=dict()):
        self.hp = AtariHyperparameters(**hp)
        utils.set_instance_vars(self.hp, self)

        # Only needed for Gymnasium >= 1.0.0
        import ale_py
        gym.register_envs(ale_py)

        import cv2
        self.resize = partial(cv2.resize, interpolation=cv2.INTER_AREA)
        
        self.env = gym.make(
            env_name.replace('Atari-','ALE/'),
            frameskip=1,
            obs_type='grayscale' if self.grayscale else 'rgb',
            repeat_action_probability=0.25 if self.sticky_actions else 0
        )
        self.env.reset(seed=seed)
        self.offline = False

        self.pixel_obs = True
        self.obs_shape = (1 if self.grayscale else 3, self.image_size, self.image_size)
        self.history = self.history
        self.max_ep_timesteps = self.max_ep_frames // self.action_repeat
        self.action_space = self.env.action_space

        self.pool_queue = deque(maxlen=2)
        self.history_queue = deque(maxlen=self.history)
        self.eval = eval


    def get_obs(self):
        if self.action_repeat > 1 and self.pool_frames:
            pool = np.maximum(self.pool_queue[0], self.pool_queue[1])
        else:
            pool = self.pool_queue[1]

        obs = self.resize(pool, (self.image_size, self.image_size))
        return np.asarray(obs, dtype=np.uint8).reshape(self.obs_shape)


    def reset(self):
        self.frames = 0
        obs, info = self.env.reset()

        if self.max_noops > 0:
            for _ in range(np.random.randint(self.max_noops)):
                obs, _, terminal, truncated, info = self.env.step(0)
                if terminal or truncated: obs, info = self.env.reset()

        self.lives = self.env.unwrapped.ale.lives()
        for _ in range(2):
            self.pool_queue.append(obs)

        obs = self.get_obs()
        for _ in range(self.history):
            self.history_queue.append(obs)

        return np.concatenate(self.history_queue), info


    def step(self, action: int): 
        # If evaluation env: somtimes sample actions randomly.
        if self.eval and np.random.uniform(0,1) < self.eval_eps:
            action = self.action_space.sample()

        reward = 0.0
        for t in range(self.action_repeat):
            self.frames += 1
            obs, frame_reward, terminal, truncated, info = self.env.step(action)
            reward += frame_reward
            self.pool_queue.append(obs)

            if self.terminal_lives and self.env.unwrapped.ale.lives() < self.lives:
                terminal = True

            if terminal or truncated: 
                break

        if self.training_reward_clipping and not self.eval:
            reward = np.clip(reward, -1, 1)

        obs = self.get_obs()
        self.history_queue.append(obs)
        return np.concatenate(self.history_queue), reward, terminal, self.frames == self.max_ep_frames, info