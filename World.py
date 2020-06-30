import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, draw, show
from numpy.random import choice
import pandas as pd
import time

class World:

    def __init__(self):

        self.nRows = 4
        self.nCols = 4
        self.stateHoles = [1, 7, 14, 15]
        self.stateGoal = [13]
        self.nStates = 16
        self.States = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        self.nActions = 4
        self.rewards = np.array([-1] + [-0.04] * 5 + [-1] + [-0.04] * 5 + [1, -1, -1] + [-0.04])
        self.stateInitial = [4]
        self.observation = []

    def _plot_world(self):

        nRows = self.nRows
        nCols = self.nCols
        stateHoles = self.stateHoles
        stateGoal = self.stateGoal
        coord = [[0, 0], [nCols, 0], [nCols, nRows], [0, nRows], [0, 0]]
        xs, ys = zip(*coord)
        plt.plot(xs, ys, "black")
        for i in stateHoles:
            (I, J) = np.unravel_index(i, shape=(nRows, nCols), order='F')
            coord = [[J, nRows - I],
                     [J + 1, nRows - I],
                     [J + 1, nRows - I + 1],
                     [J, nRows - I + 1],
                     [J, nRows - I]]
            xs, ys = zip(*coord)
            plt.fill(xs, ys, "0.5")
            plt.plot(xs, ys, "black")
        for ind, i in enumerate([stateGoal]):
            (I, J) = np.unravel_index(i, shape=(nRows, nCols), order='F')
            coord = [[J, nRows - I],
                     [J + 1, nRows - I],
                     [J + 1, nRows - I + 1],
                     [J, nRows - I + 1],
                     [J, nRows - I]]
            xs, ys = zip(*coord)
            plt.fill(xs, ys, "0.8")
            plt.plot(xs, ys, "black")
        plt.plot(xs, ys, "black")
        X, Y = np.meshgrid(range(nCols + 1), range(nRows + 1))
        plt.plot(X, Y, 'k-')
        plt.plot(X.transpose(), Y.transpose(), 'k-')

    @staticmethod
    def _truncate(n, decimals=0):
        multiplier = 10 ** decimals
        return int(n * multiplier) / multiplier

    def plot(self):
        """
        plot function
        :return: None
        """
        nStates = self.nStates
        nRows = self.nRows
        nCols = self.nCols

        self._plot_world()
        states = range(1, nStates + 1)
        k = 0
        for i in range(nCols):
            for j in range(nRows, 0, -1):
                plt.text(i + 0.5, j - 0.5, str(states[k]), fontsize=26, horizontalalignment='center',
                         verticalalignment='center')
                k += 1
        plt.title('MDP gridworld', size=16)
        plt.axis("equal")
        plt.axis("off")
        # plt.show(block=False)
        plt.show()

    def plot_value(self, valueFunction):

        nRows = self.nRows
        nCols = self.nCols
        stateHoles = self.stateHoles
        stateGoal = self.stateGoal

        fig = plt.plot(1)
        self._plot_world()
        k = 0
        for i in range(nCols):
            for j in range(nRows, 0, -1):
                if k + 1 not in stateHoles + stateGoal:
                    plt.text(i + 0.5, j - 0.5, str(self._truncate(valueFunction[k], 3)), fontsize=12, horizontalalignment='center', verticalalignment='center')
                k += 1
        plt.title('MDP gridworld', size=16)
        plt.axis("equal")
        plt.axis("off")
        plt.show()

    def plot_policy(self, policy):

        nStates = self.nStates
        nActions = self.nActions
        nRows = self.nRows
        nCols = self.nCols
        stateHoles = self.stateHoles
        stateGoal = self.stateGoal
        policy = policy.reshape(nRows, nCols, order="F").reshape(-1, 1)
        X, Y = np.meshgrid(range(nCols + 1), range(nRows + 1))
        X1 = X[:-1, :-1]
        Y1 = Y[:-1, :-1]
        X2 = X1.reshape(-1, 1) + 0.5
        Y2 = np.flip(Y1.reshape(-1, 1)) + 0.5
        X2 = np.kron(np.ones((1, nActions)), X2)
        Y2 = np.kron(np.ones((1, nActions)), Y2)
        mat = np.cumsum(np.ones((nStates, nActions)), axis=1).astype("int64")
        if policy.shape[1] == 1:
            policy = (np.kron(np.ones((1, nActions)), policy) == mat)
        index_no_policy = stateHoles + stateGoal
        index_policy = [item - 1 for item in range(1, nStates + 1) if item not in index_no_policy]
        mask = policy.astype("int64") * mat
        mask = mask.reshape(nRows, nCols, nCols)
        X3 = X2.reshape(nRows, nCols, nActions)
        Y3 = Y2.reshape(nRows, nCols, nActions)
        alpha = np.pi - np.pi / 2 * mask
        self._plot_world()
        for ii in index_policy:
            ax = plt.gca()
            j = int(ii / nRows)
            i = (ii + 1 - j * nRows) % nCols - 1
            index = np.where(mask[i, j] > 0)[0]
            h = ax.quiver(X3[i, j, index], Y3[i, j, index], np.cos(alpha[i, j, index]), np.sin(alpha[i, j, index]), 0.3)
        states = range(1, nStates + 1)
        k = 0
        for i in range(nCols):
            for j in range(nRows, 0, -1):
                plt.text(i + 0.25, j - 0.25, str(states[k]), fontsize=16, horizontalalignment='right', verticalalignment='bottom')
                k += 1
        plt.axis("equal")
        plt.axis("off")
        plt.show()

    def get_nrows(self):

        return self.nRows

    def get_ncols(self):

        return self.nCols

    def get_stateHoles(self):

        return self.stateHoles

    def get_stateGoal(self):

        return self.stateGoal

    def get_nstates(self):

        return self.nStates

    def get_nactions(self):

        return self.nActions


    def get_transition_model(self, p=0.8):
        nstates = self.nStates
        nrows = self.nRows
        holes_index = self.stateHoles
        goal_index = self.stateGoal
        terminal_index = holes_index + goal_index
        #actions = ["1", "2", "3", "4"]
        actions = [1, 2, 3, 4]     #I changed str to int
        transition_models = {}
        for action in actions:
            transition_model = np.zeros((nstates, nstates))
            for i in range(1, nstates + 1):
                if i not in terminal_index:
                    if action == 1:
                        if i + nrows <= nstates:
                            transition_model[i - 1][i + nrows - 1] += (1 - p) / 2
                        else:
                            transition_model[i - 1][i - 1] += (1 - p) / 2
                        if 0 < i - nrows <= nstates:
                            transition_model[i - 1][i - nrows - 1] += (1 - p) / 2
                        else:
                            transition_model[i - 1][i - 1] += (1 - p) / 2
                        if (i - 1) % nrows > 0:
                            transition_model[i - 1][i - 1 - 1] += p
                        else:
                            transition_model[i - 1][i - 1] += p
                    if action == 2:
                        if i + nrows <= nstates:
                            transition_model[i - 1][i + nrows - 1] += p
                        else:
                            transition_model[i - 1][i - 1] += p
                        if 0 < i % nrows and (i + 1) <= nstates:
                            transition_model[i - 1][i + 1 - 1] += (1 - p) / 2
                        else:
                            transition_model[i - 1][i - 1] += (1 - p) / 2
                        if (i - 1) % nrows > 0:
                            transition_model[i - 1][i - 1 - 1] += (1 - p) / 2
                        else:
                            transition_model[i - 1][i - 1] += (1 - p) / 2
                    if action == 3:
                        if i + nrows <= nstates:
                            transition_model[i - 1][i + nrows - 1] += (1 - p) / 2
                        else:
                            transition_model[i - 1][i - 1] += (1 - p) / 2
                        if 0 < i - nrows <= nstates:
                            transition_model[i - 1][i - nrows - 1] += (1 - p) / 2
                        else:
                            transition_model[i - 1][i - 1] += (1 - p) / 2
                        if 0 < i % nrows and (i + 1):
                            transition_model[i - 1][i + 1 - 1] += p
                        else:
                            transition_model[i - 1][i - 1] += p
                    if action == 4:
                        if 0 < i - nrows <= nstates:
                            transition_model[i - 1][i - nrows - 1] += p
                        else:
                            transition_model[i - 1][i - 1] += p
                        if 0 < i % nrows and (i + 1) <= nstates:
                            transition_model[i - 1][i + 1 - 1] += (1 - p) / 2
                        else:
                            transition_model[i - 1][i - 1] += (1 - p) / 2
                        if (i - 1) % nrows > 0:
                            transition_model[i - 1][i - 1 - 1] += (1 - p) / 2
                        else:
                            transition_model[i - 1][i - 1] += (1 - p) / 2
                elif i in terminal_index:
                    transition_model[i - 1][i - 1] = 1

            transition_models[action] = pd.DataFrame(transition_model, index=range(1, nstates + 1),
                                                     columns=range(1, nstates + 1))
        return transition_models

    # def step(self, action):
    #     observation = self.observation
    #     state = observation[0]
    #     prob = {}
    #     done = False
    #     transition_models = self.get_transition_model(0.8)
    #     # print('inside')
    #     # print(state)
    #     # print(action)
    #     prob = transition_models[action].loc[state, :]
    #     print(transition_models[action].loc[state, :])
    #     s = choice(self.States, 1, p=prob)
    #     next_state = s[0]
    #     reward = self.rewards[next_state - 1]
    #
    #     if next_state in self.stateGoal + self.stateHoles:
    #         done = True
    #     self.observation = [next_state]
    #     return next_state, reward, done

    def step(self, action):
        state = self.observation[0]
        done = 0
        prob = self.transition_model(action-1)[state-1]
        s = choice(self.States, 1, p=prob)
        next_state = s[0]
        reward = self.rewards[next_state - 1]
        if next_state in self.stateGoal + self.stateHoles:
            done = True
        self.observation = [next_state]
        return next_state, reward, done

    def _step(self, state, action, possible_state, reward):
        p_actions = [[0.8, 0.1, 0, 0.1],
                     [0.1, 0.8, 0.1, 0],
                     [0, 0.1, 0.8, 0.1],
                     [0.1, 0, 0.1, 0.8]]
        done = 0
        state += 1
        if state in self.get_stateGoal():
            return 0, state, 1, 1
        elif state not in self.get_stateHoles():
            if possible_state == 0:
                state_ = state - 1 if state % 4 != 1 else state
            elif possible_state == 1:
                state_ = state + 4 if state < 13 else state
            elif possible_state == 2:
                state_ = state + 1 if state % 4 != 0 else state
            else:
                state_ = state - 4 if state > 4 else state
        else:
            return 0, state, -1, 1
        # Check if finish or fail
        if state_ in self.get_stateGoal() or state_ in self.get_stateHoles():
            done = 1
        if state_ not in self.get_stateGoal():
            reward = -1 if state_ in self.get_stateHoles() else reward
        else:
            reward = 1
        return p_actions[action][possible_state], state_, reward, done

    def transition_model(self, action):
        transition_matrix = np.zeros([16, 16])
        for s in range(self.nStates):
            for a in range(self.nActions):
                prob, next_state, reward, done = self._step(s, action, a, 1)
                transition_matrix[s][next_state - 1] += prob
        # print(transition_matrix)
        return transition_matrix

    def reset(self, *args):
        if not args:
            observation = self.stateInitial
        else:
            observation = []
            while not observation:
                observation = np.setdiff1d(choice(self.States), self.stateHoles + self.stateGoal)
        self.observation = observation
        # return observation

    def render(self):

        nStates = self.nStates
        nActions = self.nActions
        nRows = self.nRows
        nCols = self.nCols
        stateHoles = self.stateHoles
        stateGoal = self.stateGoal

        observation = self.observation # observation
        state = observation[0]

        # state = 3

        J = nRows - (state-1) % nRows - 1
        I = int((state-1)/nCols)
        circle = plt.Circle((I+0.5, J+0.5), 0.28, color='black')
        fig = plt.gcf()
        ax = fig.gca()
        ax.add_artist(circle)

        self.plot()

        # plt.ion()
        # plt.show()
        # plt.draw()
        # plt.pause(0.5)
        # plt.ion()
        # plt.show(block=False)
        # time.sleep(1)
        # nRows = self.nRows
        # nCols = self.nCols
        # stateHoles = self.stateHoles
        # stateGoal = self.stateGoal


        # print(state)

        # circle = plt.Circle((0.5, 0.5), 0.1, color='black')
        # fig, ax = plt.subplots()
        # ax.add_artist(circle)

        # k = 0
        # for i in range(nCols):
        #     for j in range(nRows, 0, -1):
        #         if k + 1 not in stateHoles + stateGoal:
        #             plt.text(i + 0.5, j - 0.5, str(self._truncate(valueFunction[k], 3)), fontsize=12,
        #                      horizontalalignment='center', verticalalignment='center')
        #         k += 1

    def choose_action(self, state, Q, epsilon):
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.randint(self.nActions)
        else:
            action = np.argmax(Q[state - 1, :])
        return action + 1

    def learn(self, state, state2, reward, action, action2, Q, gamma, lr_rate, sarsa):
        predict = Q[state - 1, action - 1]
        if sarsa:
            target = reward + gamma * Q[state2 - 1, action2 - 1]
        else:
            target = reward + gamma * np.max(Q[state2 - 1, :])
        Q[state - 1, action - 1] = predict + lr_rate * (target - predict)

    def optimal_Q_values(self, epsilon, gamma, alpha, num_of_episodes, sarsa):
        min_epsilon = 0.01
        decay_rate = 0.0001
        Q = np.zeros([self.nStates, self.nActions])
        for i in range(num_of_episodes):
            print('Episode: %s,\nepsilon: %s' % (i, epsilon))
            np.random.seed(i)
            done = False
            self.reset()
            epsilon = epsilon - decay_rate if epsilon > min_epsilon else min_epsilon
            state = self.observation[0]
            action = self.choose_action(state, Q, epsilon)
            while not done:
                state2, reward, done = self.step(action)
                action2 = self.choose_action(state2, Q, epsilon)
                # print('state: %s, state2: %s, reward: %s, action: %s, action2: %s, Q: %s' %
                #       (state, state2, reward, action, action2, Q))
                self.learn(state, state2, reward, action, action2, Q, gamma, alpha, sarsa)
                if sarsa:
                    action = action2
                else:
                    action = self.choose_action(state, Q, epsilon)
                state = self.observation[0]
        print('Q Values:\n', Q)
        return Q

    def sarsa(self, epsilon, gamma, alpha, num_of_episodes):
        sarsa = True
        return self.optimal_Q_values(epsilon, gamma, alpha, num_of_episodes, sarsa)

    def q_learning(self, epsilon, gamma, alpha, num_of_episodes):
        sarsa = False
        return self.optimal_Q_values(epsilon, gamma, alpha, num_of_episodes, sarsa)

    def plot_actionValues(self, Q):
        '''
        :param Q: state-action values
        :return: plots the state-action values on the gridworld
        '''
        nRows = self.nRows
        nCols = self.nCols
        stateHoles = self.stateHoles
        stateGoal = self.stateGoal

        fig = plt.plot(1)
        self._plot_world()
        k = 0
        for i in range(nCols):
            for j in range(nRows, 0, -1):
                if k + 1 not in stateHoles + stateGoal:
                    plt.text(i + 0.5, j - 0.02, str(self._truncate(Q[k, 0], 2)), fontsize=8,
                             horizontalalignment='center', verticalalignment='top')
                    plt.text(i + 1, j - 0.5, str(self._truncate(Q[k, 1], 2)), fontsize=8,
                             horizontalalignment='right', verticalalignment='center')
                    plt.text(i + 0.5, j - 1, str(self._truncate(Q[k, 2], 2)), fontsize=8,
                             horizontalalignment='center', verticalalignment='bottom')
                    plt.text(i, j - 0.5, str(self._truncate(Q[k, 3], 2)), fontsize=8,
                             horizontalalignment='left', verticalalignment='center')
                    # plot([0, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0])
                    # plot([0, 1, 2, 3, 4, 5])
                    plot([i, i + 1], [j - 1, j], '-g', lw=0.2)
                    plot([i + 1, i], [j - 1, j], '-b', lw=0.2)
                k += 1

        plt.plot([0, 0], [nCols, nRows], 'b', lw=2)
        plt.title('MDP gridworld', size=16)
        plt.axis("equal")
        plt.axis("off")
        plt.show()

    def plot_action_values(self, Q):
        Q = np.array(np.max(Q, axis=1))
        self.plot_value(Q)

    def close(self):
        plt.pause(0.5)
        plt.close()

    def show(self):
        plt.ion()
        plt.show()