# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import dataclasses
import os
import pickle
import time

import numpy as np
import torch

import env_preprocessing
import MRQ
import utils


class OnlineExperiment:
    def __init__(self, agent: object, env: object, eval_env: object, logger: object, evals: list,
            t: int, total_timesteps: int, time_passed: float, 
            eval_freq: int, eval_eps: int, eval_folder: str, project_name: str, 
            save_experiment: bool=False, save_freq: int=1e5, save_folder: str=''):
        self.agent = agent
        self.env = env
        self.eval_env = eval_env
        self.evals = evals

        self.logger = logger

        self.t = t
        self.time_passed = time_passed
        self.start_time = time.time()

        self.total_timesteps = total_timesteps
        self.eval_freq = eval_freq
        self.eval_eps = eval_eps

        self.eval_folder = eval_folder
        self.project_name = project_name
        self.save_experiment = save_experiment
        self.save_freq = save_freq
        self.save_folder = save_folder
        
        self.init_timestep = True


    def run(self):
        state = self.env.reset()
        while self.t <= self.total_timesteps:
            self.maybe_evaluate()
            if self.save_experiment and self.t % self.save_freq == 0 and not self.init_timestep:
                save_experiment(self)

            action = self.agent.select_action(np.array(state))
            if action is None: action = self.env.action_space.sample()

            next_state, reward, terminated, truncated = self.env.step(action)
            self.agent.replay_buffer.add(state, action, next_state, reward, terminated, truncated)
            state = next_state

            self.agent.train()

            if terminated or truncated: 
                self.logger.log_print(
                    f'Total T: {self.t + 1}, ' 
                    f'Episode Num: {self.env.ep_num}, ' 
                    f'Episode T: {self.env.ep_timesteps}, ' 
                    f'Reward: {self.env.ep_total_reward:.3f}')
                state = self.env.reset()

            self.t += 1
            self.init_timestep = False


    def maybe_evaluate(self):
        if self.t % self.eval_freq != 0:
            return

        # We save after evaluating, this avoids re-evaluating immediately after loading an experiment.
        if self.t != 0 and self.init_timestep:
            return

        total_reward = np.zeros(self.eval_eps)
        for ep in range(self.eval_eps):
            state, terminated, truncated = self.eval_env.reset(), False, False
            while not (terminated or truncated):
                action = self.agent.select_action(np.array(state), use_exploration=False)
                state, _, terminated, truncated = self.eval_env.step(action)
            total_reward[ep] = self.eval_env.ep_total_reward

        self.evals.append(total_reward.mean())

        self.logger.title(
            f'Evaluation at {self.t} time steps\n'
            f'Average total reward over {self.eval_eps} episodes: {total_reward.mean():.3f}\n'
            f'Total time passed: {round((time.time() - self.start_time + self.time_passed)/60., 2)} minutes')

        np.savetxt(f'{self.eval_folder}/{self.project_name}.txt', self.evals, fmt='%.14f')


def save_experiment(exp: OnlineExperiment):
    # Save experiment settings
    exp.time_passed += time.time() - exp.start_time 
    var_dict = {k: exp.__dict__[k] for k in ['t', 'eval_freq', 'eval_eps']}
    var_dict['time_passed'] = exp.time_passed + time.time() - exp.start_time 
    var_dict['np_seed'] = np.random.get_state()
    var_dict['torch_seed'] = torch.get_rng_state()
    np.save(f'{exp.save_folder}/{exp.project_name}/exp_var.npy', var_dict)
    # Save eval 
    np.savetxt(f'{exp.save_folder}/{exp.project_name}.txt', exp.evals, fmt='%.14f')
    # Save envs
    pickle.dump(exp.env, file = open(f'{exp.save_folder}/{exp.project_name}/env.pickle', 'wb'))
    pickle.dump(exp.eval_env, file = open(f'{exp.save_folder}/{exp.project_name}/eval_env.pickle', 'wb'))
    # Save agent
    exp.agent.save(f'{exp.save_folder}/{exp.project_name}')

    exp.logger.title('Saved experiment')


def load_experiment(save_folder: str, project_name: str, args: object):
    # Load experiment settings
    exp_dict = np.load(f'{save_folder}/{project_name}/exp_var.npy', allow_pickle=True).item()
    # This is not sufficient to guarantee the experiment will run exactly the same,
    # however, it does mean the original seed is not reused.
    np.random.set_state(exp_dict['np_seed'])
    torch.set_rng_state(exp_dict['torch_seed'])
    # Load eval
    evals = np.loadtxt(f'{save_folder}/{project_name}.txt').tolist()
    # Load envs
    env = pickle.load(open(f'{save_folder}/{project_name}/env.pickle', 'rb'))
    eval_env = pickle.load(open(f'{save_folder}/{project_name}/eval_env.pickle', 'rb'))
    # Load agent
    agent_dict = np.load(f'{save_folder}/{project_name}/agent_var.npy', allow_pickle=True).item()
    agent = MRQ.Agent(env.obs_shape, env.action_dim, env.max_action, 
        env.pixel_obs, env.discrete, device, env.history, dataclasses.asdict(agent_dict['hp']))
    agent.load(f'{save_folder}/{project_name}')

    logger = utils.Logger(f'{args.log_folder}/{args.project_name}.txt')
    logger.title(
        'Loaded experiment\n'
        f'Starting from: {exp_dict["t"]} time steps.')

    return OnlineExperiment(agent, env, eval_env, logger, evals,    
        exp_dict['t'], args.total_timesteps, exp_dict['time_passed'],
        exp_dict['eval_freq'], exp_dict['eval_eps'], args.eval_folder, args.project_name,
        args.save_experiment, args.save_freq, args.save_folder)


@dataclasses.dataclass
class DefaultExperimentArguments:
    Atari_total_timesteps: int = 25e5
    Atari_eval_freq: int = 1e5

    Dmc_total_timesteps: int = 5e5
    Dmc_eval_freq: int = 5e3

    Gym_total_timesteps: int = 1e6
    Gym_eval_freq: int = 5e3

    def __post_init__(self): utils.enforce_dataclass_type(self)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument('--env', default='Gym-HalfCheetah-v4', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--total_timesteps', default=-1, type=int) # Uses default, input to override.
    parser.add_argument('--device', default='cuda', type=str)
    # Evaluation
    parser.add_argument('--eval_freq', default=-1, type=int) # Uses default, input to override.
    parser.add_argument('--eval_eps', default=10, type=int)
    # File name and locations
    parser.add_argument('--project_name', default='', type=str) # Uses default, input to override.
    parser.add_argument('--eval_folder', default='./evals', type=str)
    parser.add_argument('--log_folder', default='./logs', type=str)
    parser.add_argument('--save_folder', default='./checkpoint', type=str)
    # Experiment checkpointing
    parser.add_argument('--save_experiment', default=False, action=argparse.BooleanOptionalAction, type=bool)
    parser.add_argument('--save_freq', default=1e5, type=int)
    parser.add_argument('--load_experiment', default=False, action=argparse.BooleanOptionalAction, type=bool)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and args.device=='cuda' else 'cpu')

    default_arguments = DefaultExperimentArguments()
    env_type = args.env.split('-',1)[0]
    if args.total_timesteps == -1: args.total_timesteps = default_arguments.__dict__[f'{env_type}_total_timesteps']
    if args.eval_freq == -1: args.eval_freq = default_arguments.__dict__[f'{env_type}_eval_freq']

    # File name and make folders
    if args.project_name == '': args.project_name = f'MRQ+{args.env}+{args.seed}'
    if not os.path.exists(args.eval_folder): os.makedirs(args.eval_folder)
    if not os.path.exists(args.log_folder): os.makedirs(args.log_folder)
    if args.save_experiment and not os.path.exists(f'{args.save_folder}/{args.project_name}'):
        os.makedirs(f'{args.save_folder}/{args.project_name}')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.load_experiment:
        exp = load_experiment(args.save_folder, args.project_name, args)
    else:
        env = env_preprocessing.Env(args.env, args.seed, eval=False)
        eval_env = env_preprocessing.Env(args.env, args.seed+100, eval=True) # +100 to make sure the seed is different.

        agent = MRQ.Agent(env.obs_shape, env.action_dim, env.max_action, 
            env.pixel_obs, env.discrete, device, env.history)

        logger = utils.Logger(f'{args.log_folder}/{args.project_name}.txt')

        exp = OnlineExperiment(agent, env, eval_env, logger, [],
            0, args.total_timesteps, 0, 
            args.eval_freq, args.eval_eps, args.eval_folder, args.project_name, 
            args.save_experiment, args.save_freq, args.save_folder)

    exp.logger.title('Experiment')
    exp.logger.log_print(f'Algorithm:\t{exp.agent.name}')
    exp.logger.log_print(f'Env:\t\t{exp.env.env_name}')
    exp.logger.log_print(f'Seed:\t\t{exp.env.seed}')

    exp.logger.title('Environment hyperparameters')
    if hasattr(exp.env.env, 'hp'): exp.logger.log_print(exp.env.env.hp)
    exp.logger.log_print(f'Obs shape:\t\t{exp.env.obs_shape}')
    exp.logger.log_print(f'Action dim:\t\t{exp.env.action_dim}')
    exp.logger.log_print(f'Discrete actions:\t{exp.env.discrete}')
    exp.logger.log_print(f'Pixel observations:\t{exp.env.pixel_obs}')

    exp.logger.title('Agent hyperparameters')
    exp.logger.log_print(exp.agent.hp)
    exp.logger.log_print('-'*40)

    exp.run()