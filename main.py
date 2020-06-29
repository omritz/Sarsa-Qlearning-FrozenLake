# -*- coding: utf-8 -*-

from World import World
import numpy as np

if __name__ == "__main__":
    epsilon = 1
    gamma = 0.9
    alpha = 0.02
    env = World()
    Q = env.sarsa(epsilon, gamma, alpha, 10000)
    env.plot_action_values(Q)
    env.plot_policy(np.array(np.argmax(Q, axis=1))+1)
    Q = env.q_learning(epsilon, gamma, alpha, 10000)
    env.plot_action_values(Q)
    env.plot_policy(np.array(np.argmax(Q, axis=1)) + 1)
    env.show()
