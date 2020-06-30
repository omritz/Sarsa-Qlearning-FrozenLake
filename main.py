# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from World import World
import numpy as np
from itertools import product

if __name__ == "__main__":
    epsilon = 1
    gamma = 0.9
    alpha = 0.02
    env = World()
    Q = env.sarsa(epsilon, gamma, alpha, 100000)
    env.plot_actionValues(Q)
    env.plot_policy(np.array(np.argmax(Q, axis=1))+1)
    Q = env.q_learning(epsilon, gamma, alpha, 100000)
    env.plot_actionValues(Q)
    env.plot_policy(np.array(np.argmax(Q, axis=1)) + 1)
