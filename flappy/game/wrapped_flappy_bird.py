import os
import random
import math

import pandas as pd
import pickle as pkl
import numpy as np
import pygame

import flappy.game.flappy_bird_utils as fbu
import pygame.surfarray as surfarray
from pygame.locals import *
from itertools import cycle

FPS = 30
SCREENWIDTH  = 288
SCREENHEIGHT = 512

pygame.init()
FPSCLOCK = pygame.time.Clock()
SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
pygame.display.set_caption('Flappy Bird')

IMAGES, SOUNDS, HITMASKS = fbu.load()
PIPEGAPSIZE = 100 # gap between upper and lower part of pipe
BASEY = SCREENHEIGHT * 0.79

PLAYER_WIDTH = IMAGES['player'][0].get_width()
PLAYER_HEIGHT = IMAGES['player'][0].get_height()
PIPE_WIDTH = IMAGES['pipe'][0].get_width()
PIPE_HEIGHT = IMAGES['pipe'][0].get_height()
BACKGROUND_WIDTH = IMAGES['background'].get_width()

PLAYER_INDEX_GEN = cycle([0, 1, 2, 1])


class OWObject(object):
    """
    Object in objectworld.
    """

    def __init__(self, inner_colour, outer_colour):
        """
        inner_colour: Inner colour of object. int.
        outer_colour: Outer colour of object. int.
        -> OWObject
        """

        self.inner_colour = inner_colour
        self.outer_colour = outer_colour

    def __str__(self):
        """
        A string representation of this object.

        -> __str__
        """

        return "<OWObject (In: {}) (Out: {})>".format(self.inner_colour,
                                                      self.outer_colour)

class GameState:
    def __init__(self, n_objects=0, n_colours=0, prepare_tp=True):
        self.score = self.playerIndex = self.loopIter = 0
        self.playerx = int(SCREENWIDTH * 0.2)
        self.playery = int((SCREENHEIGHT - PLAYER_HEIGHT) / 2)
        self.basex = 0
        self.baseShift = IMAGES['base'].get_width() - BACKGROUND_WIDTH

        newPipe1 = getRandomPipe(0)
        newPipe2 = getRandomPipe(1)
        self.upperPipes = [
            {'x': SCREENWIDTH, 'y': newPipe1[0]['y']},
            {'x': SCREENWIDTH + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
        ]
        self.lowerPipes = [
            {'x': SCREENWIDTH, 'y': newPipe1[1]['y']},
            {'x': SCREENWIDTH + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
        ]

        # player velocity, max velocity, downward accleration, accleration on flap
        self.pipeVelX = -4
        self.playerVelY    =  0    # player's velocity along Y, default same as playerFlapped
        self.playerMaxVelY =  10   # max vel along Y, max descend speed
        self.playerMinVelY =  -8   # min vel along Y, max ascend speed
        self.playerAccY    =   1   # players downward accleration
        self.playerFlapAcc =  -9   # players speed on flapping
        self.playerFlapped = False # True when player flaps
        self.last_step=1
        self.time=0

        self.do_jump = 0
        self.do_nothing = 1

        self.n_objects = n_objects
        self.n_colours = n_colours
        self.n_first_states=15 # 300
        self.n_sec_states=760
        self.n_states=self.n_first_states*self.n_sec_states
        self.n_actions=2

        self.starting_state=(  1, 390)
        # self.ending_state=  (276, 485)
        self.ending_state=(15, None)

        self.init_expert_action_frame()

        # Generate objects.
        self.objects = {}
        for _ in range(self.n_objects):
            obj = OWObject(random.randint(0, self.n_colours),
                           random.randint(0, self.n_colours))

            while True:
                x = random.randint(0, self.n_first_states)
                y = random.randint(0, self.n_sec_states)

                if (x, y) not in self.objects:
                    break

            self.objects[x, y] = obj

        self.prepare_tp=prepare_tp
        if prepare_tp:
            if not os.path.exists("flappy_transition_probability.pkl"):
                print("prepare for transition_probability")
                self.transition_probability = np.array(
                    [[[self._transition_probability(i, j, k)
                       for k in range(self.n_states)]
                      for j in range(self.n_actions)]
                     for i in range(self.n_states)])
                pkl.dump(self.transition_probability, open("flappy_transition_probability.pkl", 'wb'))
            else:
                self.transition_probability=pkl.load(open("flappy_transition_probability.pkl", 'rb'))


#####
    # expert state: ([1, 276], [350, 523])
    # state: ([0, 300-1], [0, 760-1])
    # v2 state: ([1, 15], [0, 760-1])
    # TODO: v3 state: ([1, 30], [0, 380-1])
    def _transition_probability(self, state, action, result_state):
        state = self._get_state_by_state_code(state)
        result_state = self._get_state_by_state_code(result_state)

        if state[0]+1!=result_state[0]:
            return 0

        if action==self.do_jump and result_state[1]==state[1]+9:
            return 1
        if action==self.do_nothing:
            # hard to get real probability, due to the simple states
            return 1/self.n_sec_states
        return 0

    def init_expert_action_frame(self):
        # self.expert_action = [self.do_nothing] + [int(x) for x in pkl.load(open("flappy_ops.pkl", "rb"))]
        self.expert_action = [int(x) for x in pkl.load(open("flappy_ops.pkl", "rb"))][1:]
        self.init_expert(load=True)
        self.expert_action_frame = pd.DataFrame(columns=["action"])
        # self.origin_expert_states = [self.starting_state] + self.expert_states[:-1]
        # self.origin_expert_actions = [self.do_nothing] + self.expert_action
        self.origin_expert_states = self.expert_states[:-1]
        self.origin_expert_actions = self.expert_action
        pkl.dump(self.origin_expert_states, open("flappy_origin_expert_states.pkl", 'wb'))
        pkl.dump(self.origin_expert_actions, open("flappy_origin_expert_actions.pkl", 'wb'))

        for state, action in zip(self.origin_expert_states, self.origin_expert_actions):
            if state[0]==self.n_first_states:
                self.ending_state=state
                break
            state_code = self._get_state_code_by_state(state)
            # real_action = self._get_real_action_by_action(action)
            # print("state_code", state_code,
            #     "state", state,
            #     "action", action,
            #     "real_action", real_action)
            self.expert_action_frame.loc[state_code] = action
        pkl.dump(self.expert_action_frame, open("flappy_expert_action_frame.pkl", 'wb'))

    def check_state_code_in_expert_path(self, state_code):
        return state_code in self.expert_action_frame.index

    def _get_real_action_by_action(self, action):
        return [action, 1-action]

    def _get_state_by_state_code(self, state_code):
        return (state_code//self.n_sec_states,
                state_code%self.n_sec_states)

    def _get_state_code_by_state(self, state):
        return state[0]*self.n_sec_states+state[1]

    def _get_ob(self, s):
        return (int(s[0]), int(s[1] + 380))

    def init_expert(self, load=True):
        if load and os.path.exists("flappy_expert_states.pkl")\
                and os.path.exists("flappy_expert_imgs.pkl"):
            self.expert_states = pkl.load(open("flappy_expert_states.pkl", 'rb'))
            self.expert_imgs = pkl.load(open("flappy_expert_imgs.pkl", 'rb'))
            return

        print("init expert data")
        # self.__init__()
        img, observation, termial=self.frame_step(
            self._get_real_action_by_action(self.do_nothing))
        self.expert_states = [self._get_ob(observation)]
        self.expert_imgs=[img]
        print(self._get_ob(observation))

        for _i, action in enumerate(self.expert_action):
            real_action = self._get_real_action_by_action(action)
            img, observation_, termial=self.frame_step(real_action)
            # ob=self._get_ob_str(observation, _i)
            ob_=self._get_ob(observation_)
            print(ob_)

            self.expert_states.append(ob_)
            self.expert_imgs.append(img)

            if termial==True:
                break
        pkl.dump(self.expert_states, open("flappy_expert_states.pkl", 'wb'))
        pkl.dump(self.expert_imgs, open("flappy_expert_imgs.pkl", 'wb'))
        print("init expert data done")

    def optimal_policy_deterministic(self, state_code):
        # in ..._frame.index
        if state_code in self.expert_action_frame.index:
            return self.expert_action_frame.loc[state_code].to_numpy()[0]
        print("not exsits", state_code, self._get_state_by_state_code(state_code),
              '- real state:', self._get_state_by_state_code(state_code))
        return random.randint(0, self.n_actions - 1)

    def get_policy(self):
        return [self.optimal_policy_deterministic(s) for s in range(self.n_states)]

    def reward(self, state_code):
        if state_code == self._get_state_code_by_state(self.ending_state):
            return 1
        return 0

    def generate_trajectories(self, n_trajectories, trajectory_length, policy):
        trajectories = []
        for _ in range(n_trajectories):
            self.__init__() # ?
            self.frame_step(self._get_real_action_by_action(self.do_nothing))
            sx, sy = self.starting_state

            trajectory = []
            for _ in range(trajectory_length):
                # Follow the given policy.
                action = policy(self._get_state_code_by_state((sx, sy)))
                real_action = self._get_real_action_by_action(action)

                img, next_state, terminal = self.frame_step(real_action)
                next_state=(next_state[0], next_state[1]+380)

                if terminal or sx==self.ending_state[0]:
                    break

                print(next_state, terminal)
                state_int = self._get_state_code_by_state((sx, sy))
                action_int = action
                next_state_int = self._get_state_code_by_state(next_state)
                reward = self.reward(next_state_int)
                trajectory.append((state_int, action_int, reward))

                sx = next_state[0]
                sy = next_state[1]

            # print(self.expert_states)
            trajectories.append(np.array(trajectory))
        return np.array(trajectories)

    def feature_vector(self, i, feature_map="ident"):
        # Assume identity map.
        f = np.zeros(self.n_states)
        f[i] = 1
        return f

    def feature_matrix(self, feature_map="ident"):
        features = []
        for n in range(self.n_states):
            f = self.feature_vector(n, feature_map)
            features.append(f)
        return np.array(features)

    def frame_step(self, input_actions):
        self.last_step+=1
        pygame.event.pump()

        reward = 0.1
        terminal = False

        if sum(input_actions) != 1:
            raise ValueError('Multiple input actions!')

        # input_actions[0] == 1: do nothing
        # input_actions[1] == 1: flap the bird
        if input_actions[1] == 1:
            if self.playery > -2 * PLAYER_HEIGHT:
                self.playerVelY = self.playerFlapAcc
                self.playerFlapped = True
                #SOUNDS['wing'].play()

        # check for score
        playerMidPos = self.playerx + PLAYER_WIDTH / 2
        for pipe in self.upperPipes:
            pipeMidPos = pipe['x'] + PIPE_WIDTH / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                self.score += 1
                #SOUNDS['point'].play()
                reward = 1

        # playerIndex basex change
        if (self.loopIter + 1) % 3 == 0:
            self.playerIndex = next(PLAYER_INDEX_GEN)
        self.loopIter = (self.loopIter + 1) % 30
        self.basex = -((-self.basex + 100) % self.baseShift)

        # player's movement
        if self.playerVelY < self.playerMaxVelY and not self.playerFlapped:
            self.playerVelY += self.playerAccY
        if self.playerFlapped:
            self.playerFlapped = False
        self.playery += min(self.playerVelY, BASEY - self.playery - PLAYER_HEIGHT)
        if self.playery < 0:
            self.playery = 0

        # move pipes to left
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            uPipe['x'] += self.pipeVelX
            lPipe['x'] += self.pipeVelX

        # add new pipe when first pipe is about to touch left of screen
        if 0 < self.upperPipes[0]['x'] < 5:
            newPipe = getRandomPipe(self.last_step)
            self.upperPipes.append(newPipe[0])
            self.lowerPipes.append(newPipe[1])

        # remove first pipe if its out of the screen
        if self.upperPipes[0]['x'] < -PIPE_WIDTH:
            self.upperPipes.pop(0)
            self.lowerPipes.pop(0)

        # check if crash here
        isCrash= checkCrash({'x': self.playerx, 'y': self.playery,
                             'index': self.playerIndex},
                            self.upperPipes, self.lowerPipes)
        if isCrash:
            #SOUNDS['hit'].play()
            #SOUNDS['die'].play()
            terminal = True
            # self.__init__()
            reward = -1

        # draw sprites
        SCREEN.blit(IMAGES['background'], (0,0))

        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        SCREEN.blit(IMAGES['base'], (self.basex, BASEY))
        # print score so player overlaps the score
        # showScore(self.score)
        SCREEN.blit(IMAGES['player'][self.playerIndex],
                    (self.playerx, self.playery))

        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()
        FPSCLOCK.tick(FPS)
        #print self.upperPipes[0]['y'] + PIPE_HEIGHT - int(BASEY * 0.2)
        self.time+=1
        dx=self.lowerPipes[-1]['x']-self.playerx
        dy=self.lowerPipes[-1]['y']-self.playery
        # return image_data, [dx, dy], terminal
        return image_data, [self.time, dy], terminal

def getRandomPipe(step):
    """returns a randomly generated pipe"""
    # y of gap between upper and lower pipe
    gapYs = [50, 75, 90, 25, 10]
    # index = random.randint(0, len(gapYs)-1)
    index = step%len(gapYs)
    gapY = gapYs[index]

    gapY += int(BASEY * 0.2)
    pipeX = SCREENWIDTH + 10

    return [
        {'x': pipeX, 'y': gapY - PIPE_HEIGHT},  # upper pipe
        {'x': pipeX, 'y': gapY + PIPEGAPSIZE},  # lower pipe
    ]


def showScore(score):
    """displays score in center of screen"""
    scoreDigits = [int(x) for x in list(str(score))]
    totalWidth = 0 # total width of all numbers to be printed

    for digit in scoreDigits:
        totalWidth += IMAGES['numbers'][digit].get_width()

    Xoffset = (SCREENWIDTH - totalWidth) / 2

    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.1))
        Xoffset += IMAGES['numbers'][digit].get_width()


def checkCrash(player, upperPipes, lowerPipes):
    """returns True if player collders with base or pipes."""
    pi = player['index']
    player['w'] = IMAGES['player'][0].get_width()
    player['h'] = IMAGES['player'][0].get_height()

    # if player crashes into ground
    if player['y'] + player['h'] >= BASEY - 1:
        return True
    else:

        playerRect = pygame.Rect(player['x'], player['y'],
                      player['w'], player['h'])

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            # upper and lower pipe rects
            uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)
            lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)

            # player and upper/lower pipe hitmasks
            pHitMask = HITMASKS['player'][pi]
            uHitmask = HITMASKS['pipe'][0]
            lHitmask = HITMASKS['pipe'][1]

            # if bird collided with upipe or lpipe
            uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
            lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

            if uCollide or lCollide:
                return True

    return False

def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide and not just their rects"""
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in range(rect.width):
        for y in range(rect.height):
            if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                return True
    return False
