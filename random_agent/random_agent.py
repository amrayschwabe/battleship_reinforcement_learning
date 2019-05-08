import gym
import coding_challenge
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation

"""
Creates an environment of the Battleship game and plays one episode in it by randomly choosing the actions.
The actions handed to the Battleship environment should be 2 dimensional (x and y position of the bomb you want to
drop) and should be scaled to the range [0,1]. The actions are internally discretized to the 10 by 10 playing field.
"""
env = gym.make('Battleship-v0')
matplotlib.use("TkAgg")

def toAction(a):
    y = a % 10 / 10
    x = (a // 10) / 10
    return np.array([x, y])

def fromAction(action):
    x = int(action[0] * 10)
    y = int(action[0] * 10)
    return 10*x + y

def animate(history):

    frames = len(history)
    print("Rendering {} frames...".format(frames))

    M = np.array(history[0])

    def render_frame(i):
        #print("animate {}".format(i))
        M = np.array(history[i])
        # Render grid
        matrice.set_array(M)


    fig, ax = plt.subplots()
    matrice = ax.matshow(M, vmin=0, vmax=1)
    plt.colorbar(matrice)
    anim = matplotlib.animation.FuncAnimation(
        fig, render_frame, frames=frames, interval=100, repeat=True
    )

    plt.show()

    '''
    plt.rcParams['animation.ffmpeg_path'] = 'C:/Valentin/ffmpeg/bin/ffmpeg.exe'
    Writer = matplotlib.animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    anim.save('im.mp4', writer=writer)
    plt.close()
    '''

nr_plays = 1
sumCount = list()
for i in range(nr_plays):
    state = env.reset()
    terminal = False
    counter = 0
    history = list()
    while not terminal:

        choices = list()
        count = 0
        for i in state.reshape(-1):
            if i == 0:
                choices.append(count)
            count += 1

        action = toAction(random.choice(choices))

        state, reward, terminal, info = env.step(action)
        history.append(np.copy(state.reshape([10,10])))

        #print(state.reshape([10, 10]))

        print(info['game_message'])
        counter += 1
    sumCount.append(counter)
    print(counter)
    animate(history)

print(sum(sumCount) / len(sumCount))