# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import dataclasses
import pprint

import numpy as np


def enforce_dataclass_type(dataclass: dataclasses.dataclass):
    for field in dataclasses.fields(dataclass):
        setattr(dataclass, field.name, field.type(getattr(dataclass, field.name)))


def set_instance_vars(hp: dataclasses.dataclass, c: object):
    for field in dataclasses.fields(hp): 
        c.__dict__[field.name] = getattr(hp, field.name)


class Logger:
    def __init__(self, log_file: str):
        self.log_file = log_file


    def log_print(self, x: str | object):
        with open(self.log_file, 'a') as f: 
            if isinstance(x, str):
                print(x)
                f.write(x+'\n')
            else:
                pprint.pprint(x)
                pprint.pprint(x, f)


    def title(self, text: str):
        self.log_print('-'*40)
        self.log_print(text)
        self.log_print('-'*40)


# Takes the formatted results and returns a dictionary of env -> (timesteps, seed).
def results_to_numpy(file: str='../results/gym_results.txt'):
    results = {}

    for line in open(file):
        if '----' in line:
            continue
        if 'Timestep' in line:
            continue
        if 'Env:' in line:
            env = line.split(' ')[1][:-1]
            results[env] = []
        else:
            timestep = []
            for seed in line.split('\t')[1:]:
                if seed != '':
                    seed = seed.replace('\n', '')
                    timestep.append(float(seed))
            results[env].append(timestep)

    for k in results:
        results[k] = np.array(results[k])
        print(k, results[k].shape)

    return results


gym = [
    'Gym-HalfCheetah-v4',
    'Gym-Hopper-v4',
    'Gym-Walker2d-v4',
    'Gym-Ant-v4',
    'Gym-Humanoid-v4',
]


dmc = [
    'Dmc-acrobot-swingup',
    'Dmc-ball_in_cup-catch',
    'Dmc-cartpole-balance',
    'Dmc-cartpole-balance_sparse',
    'Dmc-cartpole-swingup',
    'Dmc-cartpole-swingup_sparse',
    'Dmc-cheetah-run',
    'Dmc-dog-stand',
    'Dmc-dog-walk',
    'Dmc-dog-trot',
    'Dmc-dog-run',
    'Dmc-finger-spin',
    'Dmc-finger-turn_easy',
    'Dmc-finger-turn_hard',
    'Dmc-fish-swim',
    'Dmc-hopper-stand',
    'Dmc-hopper-hop',
    'Dmc-humanoid-stand',
    'Dmc-humanoid-walk',
    'Dmc-humanoid-run',
    'Dmc-pendulum-swingup',
    'Dmc-quadruped-walk',
    'Dmc-quadruped-run',
    'Dmc-reacher-easy',
    'Dmc-reacher-hard',
    'Dmc-walker-stand',
    'Dmc-walker-walk',
    'Dmc-walker-run'
]


dmc_visual = [
    'Dmc-visual-acrobot-swingup',
    'Dmc-visual-ball_in_cup-catch',
    'Dmc-visual-cartpole-balance',
    'Dmc-visual-cartpole-balance_sparse',
    'Dmc-visual-cartpole-swingup',
    'Dmc-visual-cartpole-swingup_sparse',
    'Dmc-visual-cheetah-run',
    'Dmc-visual-dog-stand',
    'Dmc-visual-dog-walk',
    'Dmc-visual-dog-trot',
    'Dmc-visual-dog-run',
    'Dmc-visual-finger-spin',
    'Dmc-visual-finger-turn_easy',
    'Dmc-visual-finger-turn_hard',
    'Dmc-visual-fish-swim',
    'Dmc-visual-hopper-stand',
    'Dmc-visual-hopper-hop',
    'Dmc-visual-humanoid-stand',
    'Dmc-visual-humanoid-walk',
    'Dmc-visual-humanoid-run',
    'Dmc-visual-pendulum-swingup',
    'Dmc-visual-quadruped-walk',
    'Dmc-visual-quadruped-run',
    'Dmc-visual-reacher-easy',
    'Dmc-visual-reacher-hard',
    'Dmc-visual-walker-stand',
    'Dmc-visual-walker-walk',
    'Dmc-visual-walker-run'
]


atari = [
    'Atari-Alien-v5',
    'Atari-Amidar-v5',
    'Atari-Assault-v5',
    'Atari-Asterix-v5',
    'Atari-Asteroids-v5',
    'Atari-Atlantis-v5',
    'Atari-BankHeist-v5',
    'Atari-BattleZone-v5',
    'Atari-BeamRider-v5',
    'Atari-Berzerk-v5',
    'Atari-Bowling-v5',
    'Atari-Boxing-v5',
    'Atari-Breakout-v5',
    'Atari-Centipede-v5',
    'Atari-ChopperCommand-v5',
    'Atari-CrazyClimber-v5',
    'Atari-DemonAttack-v5',
    'Atari-DoubleDunk-v5',
    'Atari-Enduro-v5',
    'Atari-FishingDerby-v5',
    'Atari-Freeway-v5',
    'Atari-Frostbite-v5',
    'Atari-Gopher-v5',
    'Atari-Gravitar-v5',
    'Atari-Hero-v5',
    'Atari-IceHockey-v5',
    'Atari-Jamesbond-v5',
    'Atari-Kangaroo-v5',
    'Atari-Krull-v5',
    'Atari-KungFuMaster-v5',
    'Atari-MontezumaRevenge-v5',
    'Atari-MsPacman-v5',
    'Atari-NameThisGame-v5',
    'Atari-Phoenix-v5',
    'Atari-Pitfall-v5',
    'Atari-Pong-v5',
    'Atari-PrivateEye-v5',
    'Atari-Qbert-v5',
    'Atari-Riverraid-v5',
    'Atari-RoadRunner-v5',
    'Atari-Robotank-v5',
    'Atari-Seaquest-v5',
    'Atari-Skiing-v5',
    'Atari-Solaris-v5',
    'Atari-SpaceInvaders-v5',
    'Atari-StarGunner-v5',
    'Atari-Tennis-v5',
    'Atari-TimePilot-v5',
    'Atari-Tutankham-v5',
    'Atari-UpNDown-v5',
    'Atari-Venture-v5',
    'Atari-VideoPinball-v5',
    'Atari-WizardOfWor-v5',
    'Atari-YarsRevenge-v5',
    'Atari-Zaxxon-v5',
]