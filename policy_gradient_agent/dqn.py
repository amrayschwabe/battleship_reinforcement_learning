# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
import coding_challenge
import matplotlib.pyplot as plt
import matplotlib.animation
import time as t
from keras import losses


import tensorflow as tf

EPISODES = 50000

def toAction(a):
    y = a % 10 / 10
    x = (a // 10) / 10
    return np.array([x, y])


def fromAction(action):
    x = int(action[0] * 10)
    y = int(action[0] * 10)
    return 10 * x + y

def animate(history):
    frames = len(history)
    print("Rendering {} frames...".format(frames))

    M = np.reshape(history[0], [10, 10])

    def render_frame(i):
        M = np.reshape(history[i], [10, 10])
        # Render grid
        matrice.set_array(M)

    fig, ax = plt.subplots()
    matrice = ax.matshow(M, vmin=0, vmax=1)
    plt.colorbar(matrice)
    anim = matplotlib.animation.FuncAnimation(
        fig, render_frame, frames=frames, interval=100, repeat=True
    )

    plt.show()
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=500)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99
        self.learning_rate = 0.005
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(100, input_dim=self.state_size, activation='tanh'))
        model.add(Dense(100, activation='tanh'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss=losses.mean_squared_error,
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):

        current_board = state.reshape(-1)
        if np.random.rand() <= self.epsilon:
            choices = list()
            count = 0
            for i in current_board:
                if i == 0:
                    choices.append(count)
                count += 1
            choice = random.choice(choices)
            return choice
        act_values = self.model.predict(state)[0]
        act_values_possible = [p if (current_board[index] == 0) else -np.infty for index, p in enumerate(act_values)]
        return np.argmax(act_values_possible)  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward + 10
            else:
                # a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
                # target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    BOARD_SIZE = 100
    env = gym.make('Battleship-v0')
    state_size = 100
    action_size = 100
    agent = DQNAgent(state_size, action_size)
    done = False
    batch_size = 32
    game_lengths = []
    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            current_board = state.reshape(1, -1)
            action = agent.act(state)
            next_state, reward, done, _ = env.step(toAction(action))
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                agent.update_target_model()
                game_lengths.append(time)
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        # plot progress every 500 games
        if e > 0 and e % 100 == 0:
            window_size = 100
            running_average_length = [np.mean(game_lengths[i:i + window_size]) for i in
                                      range(len(game_lengths) - window_size)]
            plt.plot(running_average_length)
            plt.show()
