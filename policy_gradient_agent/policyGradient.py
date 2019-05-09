#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import gym
import matplotlib.pyplot as plt
import os
import coding_challenge
import matplotlib.animation


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


env = gym.make('Battleship-v0')

# In[4]:


# 1.2 Define the nn variable network.
# Input is array of BOARD_SIZE values.
# ---------------------------------------
#  -1 value -> Not yet checked
#   0 value -> Checked, no ship
#   1 value -> Checked, is ship location.
# ---------------------------------------
BOARD_SIZE = 100

SMALL_NETWORK = True

if SMALL_NETWORK:
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

    # 1.3 Define the operations we will use

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='xentropy')
    train_step = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(cross_entropy)
    init = tf.global_variables_initializer()
    # Start TF session
    sess = tf.Session()
    sess.run(init)
else:
    hidden_units = 3 * BOARD_SIZE
    output_units = BOARD_SIZE

    input_positions = tf.placeholder(tf.float32, shape=(1, BOARD_SIZE))
    labels = tf.placeholder(tf.int64)
    learning_rate = tf.placeholder(tf.float32, shape=[])
    # Generate hidden layer
    W1 = tf.Variable(tf.truncated_normal([BOARD_SIZE, hidden_units],
                                         stddev=0.1 / np.sqrt(float(BOARD_SIZE))))
    b1 = tf.Variable(tf.zeros([1, hidden_units]))
    h1 = tf.nn.relu(tf.matmul(input_positions, W1) + b1)

    W2 = tf.Variable(tf.truncated_normal([hidden_units, BOARD_SIZE],
                                         stddev=0.1 / np.sqrt(float(hidden_units))))
    b2 = tf.Variable(tf.zeros([1, BOARD_SIZE]))
    h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)

    W3 = tf.Variable(tf.truncated_normal([BOARD_SIZE, BOARD_SIZE],
                                         stddev=0.1 / np.sqrt(float(BOARD_SIZE))))
    b3 = tf.Variable(tf.zeros([1, BOARD_SIZE]))
    h3 = tf.nn.relu(tf.matmul(h2, W3) + b3)

    # Second layer -- linear classifier for action logits
    W5 = tf.Variable(tf.truncated_normal([BOARD_SIZE, output_units],
                                         stddev=0.1 / np.sqrt(float(BOARD_SIZE))))
    b5 = tf.Variable(tf.zeros([1, output_units]))
    logits = tf.matmul(h3, W5) + b5
    probabilities = tf.nn.softmax(logits)

    # 1.3 Define the operations we will use
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='xentropy')
    train_step = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(cross_entropy)
    init = tf.global_variables_initializer()
    # Start TF session
    sess = tf.Session()
    sess.run(init)

# 1.4 Game play definition.
TRAINING = True


def play_game(training=TRAINING):
    """ Play game of battleship using network."""
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

        probs = sess.run([probabilities], feed_dict={input_positions: current_board})[0][0]
        probs = [p if (current_board[index] == 0) else -10 for index, p in enumerate(probs)]
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


# Example:

LOAD = False
if LOAD:
    saver = tf.train.Saver()
    saver.restore(sess, r'/Users/amrayschwabe/Documents/ETH/Master/2. Semester/Deep Reinforcement Leraning Seminar/Coding_Challenge/models')

    li = list()
    for i in range(100):
        board_position_log, action_log, reward_log = play_game(True)
        li.append(len(board_position_log))
    print("Average sample: {}".format(sum(li) / len(li)))

    li2 = list()
    for i in range(100):
        board_position_log, action_log, reward_log = play_game(False)
        li2.append(len(board_position_log))
    print("Average argmax: {}".format(sum(li2) / len(li2)))

    matplotlib.use("TkAgg")

    board_position_log, action_log, reward_log = play_game(False)
    animate(board_position_log)
else:
    # 1.6 Training loop: Play and learn
    game_lengths = []
    TRAINING = True  # Boolean specifies training mode
    ALPHA = 0.0005  # step size

    nr_episodes = 50001
    for game in range(nr_episodes):

        if game % 10 == 0:
            print("Episode {} of {}".format(game, nr_episodes))
        board_position_log, action_log, reward_log = play_game(training=TRAINING)
        game_lengths.append(len(action_log))

        for reward, current_board, action in zip(reward_log, board_position_log, action_log):
            # Take step along gradient
            if TRAINING:
                sess.run([train_step],
                         feed_dict={input_positions: current_board, labels: [action], learning_rate: ALPHA * reward})

        if game != 0 and game % 5000 == 0:
            saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=2)
            saver.save(sess, r'/Users/amrayschwabe/Documents/ETH/Master/2. Semester/Deep Reinforcement Leraning Seminar/Coding_Challenge/models', global_step=game)

        if game > 500 and game % 500 == 0:
            window_size = 500
            running_average_length = [np.mean(game_lengths[i:i + window_size]) for i in
                                      range(len(game_lengths) - window_size)]
            plt.plot(running_average_length)
            plt.show()

    # 1.7 Plot running average game lengths
    window_size = 500
    running_average_length = [np.mean(game_lengths[i:i + window_size]) for i in range(len(game_lengths) - window_size)]
    plt.plot(running_average_length)
    plt.show()
