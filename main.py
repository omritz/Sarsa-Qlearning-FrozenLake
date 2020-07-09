# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from World import World
import numpy as np
import seaborn as sns
import pandas as pd


V_optimal = [0.0, 0.28589929, 0.076748, 0.00825989, 0.74736637, 0.57640414,
             0.0, -0.08592467,  0.92809814,  0.58410703,  0.18855706,
             0.08029465, 0.0, 0.0, 0.0, -0.08592108]


def plot_learning(episodes, scores, decay, alpha):
    sns.set(style="darkgrid")
    # Create the data
    x = [i+1 for i in range(episodes)]
    running_avg = np.empty(episodes)
    for t in range(episodes):
        running_avg[t] = np.mean(scores[max(0, t - 20):(t + 1)])
    data = pd.DataFrame(data={'Game': x, 'Score': running_avg})
    plot = sns.regplot(x="Game", y="Score",
                       data=data, scatter_kws={"color": "blue", 's': 1}, line_kws={"color": "red"})
    plt.title('Score over episodes with alpha %s and decay rate %s' % (alpha, decay))
    plt.savefig('alpha_%s_decay_%s.jpg' % (alpha, decay))
    plt.clf()


def calculate_mse(v):
    mse = np.square(np.array(V_optimal) - v)
    mse = np.sum(mse)
    return mse/len(V_optimal)


def find_hyper_param(algorithm='sarsa'):
    epsilon = 1
    gamma = 0.9
    episodes = 100000
    MSE = []
    params = []
    alphas = [0.2, 0.02, 0.002, 0.0002, 0.00002]
    rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    env = World()
    for i in range(len(rates)):
        epsilon_decay = rates[i]
        for j in range(len(alphas)):
            alpha = alphas[j]
            if algorithm == 'sarsa':
                Q, scores = env.sarsa(epsilon, gamma, alpha, episodes, decay_rate=epsilon_decay)
            else:
                Q, scores = env.q_learning(epsilon, gamma, alpha, episodes, decay_rate=epsilon_decay)
            MSE.append(calculate_mse(np.array(np.max(Q, axis=1))))
            params.append([alpha, epsilon_decay])
            plot_learning(episodes, scores, epsilon_decay, alpha)
    best_run = np.argmin(MSE)
    print(MSE)
    print('The best run is: %s, MSE: %s, alpha: %s, decay rate: %s' % (best_run, MSE[best_run],
                                                                       params[best_run][0],
                                                                       params[best_run][1]))


if __name__ == "__main__":
    find_hyper_param()
    decay_rate = 0.00001
    episodes = 500000
    epsilon = 1
    gamma = 0.9
    alpha = 0.0002
    env = World()
    Q, scores = env.sarsa(epsilon, gamma, alpha, episodes, decay_rate)
    print('MSE: %s' % calculate_mse(np.array(np.max(Q, axis=1))))
    print('Average Reward: %s' % np.mean(scores))
    plot_learning(episodes, scores, decay_rate, alpha)
    env.plot_actionValues(Q)
    env.plot_policy(np.array(np.argmax(Q, axis=1))+1)
    Q = env.q_learning(epsilon, gamma, alpha, episodes, decay_rate)
    env.plot_actionValues(Q)
    env.plot_policy(np.array(np.argmax(Q, axis=1)) + 1)
    print('MSE: %s' % calculate_mse(np.array(np.max(Q, axis=1))))
    print('Average Reward: %s' % np.mean(scores))
    plot_learning(episodes, scores, decay_rate, alpha)
