"""
Reinforcement learning maze example.
Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].
This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import logging
import tensorflow as tf
import time
import sys

sys.path.append("game/")

# from maze_env import Maze, MAZE_H, MAZE_W

import wrapped_flappy_bird as Game
from RL_brain import QLearningTable
from ada_irl import AdaModel

formatter = logging.Formatter(
    '%(asctime)s [line:%(lineno)d] %(levelname)5s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
logger=logging.getLogger()
logger.setLevel(logging.DEBUG)

fileHandler=logging.FileHandler("log/ada_model"+str(time.time())+".log")
fileHandler.setLevel(logging.DEBUG)
fileHandler.setFormatter(formatter)

scrHandler=logging.StreamHandler()
scrHandler.setLevel(logging.DEBUG)
scrHandler.setFormatter(formatter)

logger.addHandler(fileHandler)
logger.addHandler(scrHandler)

def update():
    model=AdaModel(env, logger, RL)
    model.update(env, RL)
    print("done")
    env.destory()

if __name__ == "__main__":
    with tf.device('/gpu:0'):
        env = Game.GameState()
        RL = QLearningTable(actions=list([0, 1]), learning_rate=0.1, e_greedy=0.9)
        update()
