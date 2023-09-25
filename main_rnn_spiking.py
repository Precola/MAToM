# Adapted from this code at https://github.com/starry-sky6688/MARL-Algorithms

from common_sr.arguments import get_common_args, get_coma_args, get_mixer_args
from utils.stag_env_wrapper.env_wrappers import SubprocVecEnv
from utils.mpe_env_wrapper.env_wrappers import SubprocVecEnv, DummyVecEnv
from runner import Runner
from gym_stag_hunt.envs.gym.escalation import EscalationEnv
from gym_stag_hunt.envs.gym.harvest import HarvestEnv
from gym_stag_hunt.envs.gym.hunt import HuntEnv
from gym_stag_hunt.envs.gym.simple import SimpleEnv
import argparse
import numpy as np
parser = argparse.ArgumentParser()
import json
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

if __name__ == '__main__':
    ENVS = {
        "CLASSIC": SimpleEnv,
        "HUNT": HuntEnv,
        "HARVEST": HarvestEnv,
        "ESCALATION": EscalationEnv,
    }

    args = get_common_args()
    args = get_mixer_args(args)

    if args.ENV == 'HUNT':
        args.n_actions = 5  # [5] # up, down, left, right or stand
        args.n_agents = 2  # [2]
        args.obs_shape = 6 + args.forage_quantity * 2

    elif args.ENV == 'ESCALATION':
        args.n_actions = 5  # [5] # up, down, left, right or stand
        args.n_agents = 2  # [2]
        args.obs_shape = 6

    elif args.ENV == 'HARVEST':
        args.n_actions = 5  # [5] # up, down, left, right or stand
        args.n_agents = 2  # [2]
        args.obs_shape = 6 + args.forage_quantity * 5

    if args.env_name == "MPE" and args.num_good_agents is None and args.num_adversaries is None:
        parser.error("--num_agents must be specified when env is MPE'.")

    args.episode_limit = 50
    args.train_steps = 100

    save_path = args.log_dir + '/' + args.alg + args.exp_dir
    print(os.path.exists(save_path))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # save args
    argsDict = args.__dict__
    with open(save_path + '/args_{}'.format(args.num_run), 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')

    def make_env():
        def _thunk():
            if args.ENV == 'HUNT':
                env = ENVS[args.ENV](obs_type="coords", enable_multiagent=True, opponent_policy="random", \
                                     forage_quantity=args.forage_quantity, run_away_after_maul=True)
            elif args.ENV == 'ESCALATION':
                env = ENVS[args.ENV](obs_type="coords", enable_multiagent=True)

            elif args.ENV == 'HARVEST':
                env = ENVS[args.ENV](obs_type="coords", enable_multiagent=True)

            else:   #MPE
                env = make_env(args.ENV, num_good_agents=args.num_good_agents, num_adversaries=args.num_adversaries,
                               discrete_action=args.discrete_action)
            env.seed(args.seed)
            np.random.seed(args.seed)

            return env


        if args.process == 1:
            return DummyVecEnv(_thunk)
        else:
            return SubprocVecEnv([_thunk for i in range(args.process)])
        # return _thunk

    # for i in range(args.num_run):

    envs = [make_env() for i in range(args.process)]
    envs = SubprocVecEnv(envs)

    runner = Runner(envs, args)
    if not args.evaluate:
        runner.run(args.num_run)
    else:
        win_rate, _ = runner.evaluate()
        print('The win rate of {} is  {}'.format(args.alg, win_rate))
    envs.close()


