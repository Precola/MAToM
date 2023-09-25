# -*- encoding: utf-8 -*-
'''
@File    :   line_demo.py
@Time    :   2022/06/11 16:41:18
@Author  :   HMX
@Version :   1.0
@Contact :   kzdhb8023@163.com
'''

# here put the import lib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time

def cm2inch(x,y):
    return x/2.54,y/2.54


t1 = time.time()

# 构造数据
# np.random.seed(1503)
# # 构造单一年份数据
# x = np.arange(1,13)
# y = np.sin(x)
# # 构造十年数据以计算均值和置信区间
# year = 10
# ys = []
# # 利用随机数添加一些误差
# for i in range(year):
#     ys.append(y+np.random.rand(len(y))+np.random.randint(1,5,size = (len(y),)))
# ys = np.asarray(ys).reshape(-1,)
# xs = x.tolist()*year

def plt_reward_mean():
    path_test = []
    episode_rewards_test = []

    path_test.append('/home/zhaozhuoya/stag-exp/ToM/ToM2_test/result/svdn_self/HAR_rnn')
    for j in [1]:
        episode_rewards_test.append(np.load(path_test[0] + '/episode_rewards_{}.npy'.format(j))[:,0])
    x = np.arange(1, len(episode_rewards_test[0])+1)
    ys = np.asarray(episode_rewards_test).reshape(-1, )
    xs = x.tolist() * len([1])
    # fig,ax1 = plt.subplots(1, 1)#,figsize=cm2inch(8,6)
    sns.lineplot(xs,ys)
    # plt.xlim(0,14)
    # plt.xticks(np.arange(0,15,2))
    # plt.ylim(0,6)
    # plt.xlabel('xlabel')
    # plt.ylabel('ylabel')
    # plt.tight_layout()
    # plt.savefig(r'D:\公众号\N19\line_demo.png',dpi = 600)
    plt.show()

if __name__ == '__main__':

    plt_reward_mean()#args.num_run