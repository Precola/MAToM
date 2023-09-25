import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse


A = 2000

def get_common_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=str, default='siql', help='the algorithm to train the agent')

    parser.add_argument('--model_dir', type=str, default='./model', help='model directory of the policy_base')
    parser.add_argument('--result_dir', type=str, default='./result', help='result directory of the policy_base')#./result#/home/zhaozhuoya/exp2/ToM2_test/result
    parser.add_argument('--log_dir', type=str, default='./log', help='args directory')
    parser.add_argument('--plot_dir', type=str, default='./plot', help='args directory')

    parser.add_argument('--exp_dir', type=str, default='/exp1', help='result directory of the policy_base')
    args = parser.parse_args()

    return args

def plt_reward_mean(num_run):
    path_test, path, path_scovdn, path_scoiql, path_scovdn_weight, path_svdn, path_scovdn_3\
        = [], [], [], [], [], [], []
    episode_rewards_scoiql, episode_rewards, episode_rewards_test, episode_rewards_scovdn, \
    episode_rewards_scovdn_3, episode_scovdn_weight, episode_rewards_svdn \
        = [], [], [], [], [], [], []

    path_test.append('/home/username/MAToM/result/svdn_selfHUNT')
    path.append('/home/username/MAToM/result/iql/hunt1')  # aseline
    path_scoiql.append(args.result_dir + '/' + 'scoiql' + '/exp_scoiql')
    path_scovdn.append(args.result_dir + '/' + 'scovdn' + '/exp_scovdn')
    # path_scovdn_3.append(args.result_dir + '/' + 'stomvdn' + '/exp_stomvdn')   #same as svdn
    path_scovdn_weight.append(args.result_dir + '/' + 'svdn_selfHUNT')  # same as svdn
    # path_scovdn_weight.append(args.result_dir + '/' + 'scovdn_weight' + '/exp_scovdn_weight2')
    path_svdn.append(args.result_dir + '/' + 'svdn' + '/exp_svdn')


    for j in range(num_run):
        episode_rewards.append(np.load(path[0] + '/episode_rewards_{}.npy'.format(j)))  #se

    for j in [1]:
        episode_rewards_test.append(np.load(path_test[0] + '/episode_rewards_{}.npy'.format(j)))

    # for j in [2, 5, 8]:
    #     episode_rewards_scoiql.append(np.load(path_scoiql[0] + '/episode_rewards_{}.npy'.format(j)))
    #
    for j in [1, 2]:
        episode_scovdn_weight.append(np.load(path_scovdn_weight[0] + '/episode_rewards_{}.npy'.format(j)))
    #
    # for j in [2]:
    #     episode_rewards_scovdn.append(np.load(path_scovdn[0] + '/episode_rewards_{}.npy'.format(j)))

    # for j in [1]:
    #     episode_rewards_scovdn_3.append(np.load(path_scovdn_3[0] + '/episode_rewards_{}.npy'.format(j)))

    # for j in [4, 5]:
    #     episode_rewards_svdn.append(np.load(path_svdn[0] + '/episode_rewards_{}.npy'.format(j)))

    episode_rewards = np.array(episode_rewards).mean(axis=0)
    episode_rewards_test = np.array(episode_rewards_test).mean(axis=0)
    # episode_rewards_scoiql = np.array(episode_rewards_scoiql).mean(axis=0)
    # episode_rewards_scovdn = np.array(episode_rewards_scovdn).mean(axis=0)
    episode_scovdn_weight = np.array(episode_scovdn_weight).mean(axis=0)
    # episode_rewards_svdn = np.array(episode_rewards_svdn).mean(axis=0)
    # episode_rewards_scovdn_3 = np.array(episode_rewards_scovdn_3).mean(axis=0)



    plt.figure()
    plt.plot(range(A), episode_rewards_test[:A, 0], c='y', label='SIQL')
    # plt.plot(range(A), episode_rewards[:A, 0], c='g', label='IQL')#len(episode_rewards_test)
    # plt.plot(range(A), episode_rewards_scoiql[:A, 0], c='b', label='SCOIQL')#len(episode_rewards_svdn)
    # plt.plot(range(A), episode_rewards_scovdn[:A, 0], c='r', label='SCOVDN')  # len(episode_rewards_svdn)
    plt.plot(range(A-1000), episode_scovdn_weight[:A-1000, 0], label='SCOVDN_WEIGHT')  # len(episode_rewards_svdn)
    # plt.plot(range(A), episode_rewards_svdn[:A, 0], label='SVDN')  # len(episode_rewards_svdn)
    # plt.plot(range(A), episode_rewards_scovdn_3[:A, 0], label='SCOVDN_W')  # len(episode_rewards_svdn)

    LABEL = ['IQL', 'SIQL', 'SCOIQL', 'SCOVDN', 'SCOVDN_WEIGHT', 'SVDN']


    plt.legend()
    plt.xlabel('episodes * 100')
    plt.ylabel('win_rate')
    # plt.savefig('../plot/overview_{}.png')
    plt.show()


if __name__ == '__main__':
    args = get_common_args()

    # plt_win_rate_mean()
    plt_reward_mean(10)#args.num_run