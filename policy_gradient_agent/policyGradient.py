#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import gym
import matplotlib.pyplot as plt
import os
import coding_challenge
import matplotlib.animation

TRAINING = False
LOAD = True
BOARD_SIZE = 100
ALPHA = 0.003  # step size
nr_episodes = 50001  # number of training episodes

#create the model
hidden_units = BOARD_SIZE
output_units = BOARD_SIZE
input_positions = tf.placeholder(tf.float32, shape=(1, BOARD_SIZE))
labels = tf.placeholder(tf.int64)
learning_rate = tf.placeholder(tf.float32, shape=[])
# Generate hidden layer
W1 = tf.Variable(tf.truncated_normal([BOARD_SIZE, hidden_units],
                                     stddev=0.1 / np.sqrt(float(BOARD_SIZE))))
b1 = tf.Variable(tf.zeros([1, hidden_units]))
h1 = tf.tanh(tf.matmul(input_positions, W1) + b1)
# Second layer -- linear classifier for action logits
W2 = tf.Variable(tf.truncated_normal([hidden_units, output_units],
                                     stddev=0.1 / np.sqrt(float(hidden_units))))
b2 = tf.Variable(tf.zeros([1, output_units]))
logits = tf.matmul(h1, W2) + b2
probabilities = tf.nn.softmax(logits)

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits, labels=labels, name='xentropy')
train_step = tf.train.GradientDescentOptimizer(
    learning_rate=learning_rate).minimize(cross_entropy)
init = tf.global_variables_initializer()
# Start TF session
sess = tf.Session()
sess.run(init)

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


def play_game(training=TRAINING):
    # Select random location for ship
    state = env.reset()

    # Initialize logs for game
    board_position_log = []
    action_log = []
    reward_log = []

    # Play through game
    terminal = False
    while not terminal:
        current_board = state.reshape(1, -1)
        board_position_log.append(np.copy(current_board))

        #get probability for each action and exclude the ones that have already been explored
        probs = sess.run([probabilities], feed_dict={input_positions: current_board})[0][0]
        probs = [p * (current_board[0][index] == 0) for index, p in enumerate(probs)]
        probs = [p / sum(probs) for p in probs]

        if training == True:
            bomb_index = np.random.choice(BOARD_SIZE, p=probs)
        else:
            bomb_index = np.argmax(probs)
        # update board, logs

        action_log.append(bomb_index)
        new_state, reward, terminal, info = env.step(toAction(bomb_index))
        reward_log.append(reward)
        state = new_state

    return board_position_log, action_log, reward_log

def load_and_evaluate(source):
    # load existing model
    saver = tf.train.Saver()
    saver.restore(sess,
                  source)

    # evaluate how good our model does in 100 games
    li2 = list()
    for i in range(100):
        board_position_log, action_log, reward_log = play_game(False)
        li2.append(len(board_position_log))

    print("Average argmax: {}".format(sum(li2) / len(li2)))

    matplotlib.use("TkAgg")

    board_position_log, action_log, reward_log = play_game(False)
    animate(board_position_log)

def load_and_train(source):
    # load existing model
    saver = tf.train.Saver()
    saver.restore(sess,
                  source)

    train()


def train():
    # Training loop
    game_lengths = []

    for game in range(nr_episodes):

        if game % 10 == 0:
            print("Episode {} of {}".format(game, nr_episodes))

        #play game once
        board_position_log, action_log, reward_log = play_game(training=TRAINING)
        game_lengths.append(len(action_log))

        #learn from played game if currently training
        for reward, current_board, action in zip(reward_log, board_position_log, action_log):
            sess.run([train_step],
                    feed_dict={input_positions: current_board, labels: [action], learning_rate: ALPHA * reward})

        #save every 5000 game
        if game != 0 and game % 5000 == 0:
            saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=2)
            saver.save(sess, r'/Users/amrayschwabe/Documents/ETH/Master/2. Semester/Deep Reinforcement Leraning Seminar/Coding_Challenge/models/', global_step=game)

        #plot progress every 500 games
        if game > 500 and game % 500 == 0:
            window_size = 500
            running_average_length = [np.mean(game_lengths[i:i + window_size]) for i in
                                      range(len(game_lengths) - window_size)]
            plt.plot(running_average_length)
            plt.show()

if __name__ == "__main__":
    env = gym.make('Battleship-v0')
    if LOAD and not TRAINING:
        load_and_evaluate(r'/Users/amrayschwabe/Documents/ETH/Master/2. Semester/Deep Reinforcement Leraning Seminar/Coding_Challenge/models/-50000')
    elif LOAD and TRAINING:
        load_and_train(
            r'/Users/amrayschwabe/Documents/ETH/Master/2. Semester/Deep Reinforcement Leraning Seminar/Coding_Challenge/models/-50000')
    else:
        train()
