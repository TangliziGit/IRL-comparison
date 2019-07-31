import sys
sys.path.append('')

import numpy as np
import pickle as pkl
import tensorflow as tf

from irl.LargeGradientIRL_torch import LargeGradientIRL
import flappy.game.wrapped_flappy_bird as Game

import irl.value_iteration as vi
from irl.validate import validate

def train(discount, n_trajectories, epochs, learning_rate):
    """
    Run maximum entropy inverse reinforcement learning on the gridworld MDP.

    Plots the reward function.

    grid_size: Grid size. int.
    discount: MDP discount factor. float.
    n_trajectories: Number of sampled trajectories. int.
    epochs: Gradient descent iterations. int.
    learning_rate: Gradient descent learning rate. float.
    """

    trajectory_length = 276

    # env = Game.GameState()
    # exit(0)
    # trajectories = env.generate_trajectories(n_trajectories,  trajectory_length, env.optimal_policy_deterministic)

    do_jump = 0
    do_nothing = 1

    n_first_states=15
    n_sec_states=760
    n_states=n_first_states*n_sec_states
    n_actions=2
    trajectories = pkl.load(open("flappy_trajectories.pkl", 'rb'))
    trajectories = [trajectories[0][:-1] for i in range(n_trajectories)]
    # origin_expert_states = pkl.load(open("flappy_origin_expert_states.pkl", 'rb'))
    # origin_expert_actions = pkl.load(open("flappy_origin_expert_actions.pkl", 'rb'))

    # print(trajectories)

    def feature_function(state):
        feature = np.zeros(n_states)
        feature[state]=1
        return feature

    # def _get_state_code_by_state(state):
    #     return state[0]*n_sec_states+state[1]

    # r_origin_expert_states = [-1] * n_states
    # for i, x in enumerate(origin_expert_states[:14]):
    #     code = _get_state_code_by_state(x)
    #     r_origin_expert_states[code] = i

    def _get_state_by_state_code(state_code):
        return (state_code//n_sec_states,
                state_code%n_sec_states)

    # def find_idx_in_expert_states_by_state_code(state_code):
    #     return r_origin_expert_states[state_code]

    def _transition_probability(state_code, action, result_state_code):
        state = _get_state_by_state_code(state_code)
        result_state = _get_state_by_state_code(result_state_code)

        if state[0]+1!=result_state[0]:
            return 0

        if action==do_jump and result_state[1]==state[1]+9:
            return 1
        if action==do_nothing:
            # hard to get real probability, due to the simple states
            return 1/n_sec_states                 # v1
            # if abs(result_state[1]-state[1])<=50: # v2
            #     return 0.01
            # idx = find_idx_in_expert_states_by_state_code(state_code)
            # nidx = find_idx_in_expert_states_by_state_code(result_state_code)
            # if nidx==idx+1:
            #     return 1
        return 0

    def transitionProbability(state_code, action):
        res = {}
        for i in range(n_states):
            res[state_code]=_transition_probability(state_code, action, i)
        return res

    irl = LargeGradientIRL(n_actions, n_states, transitionProbability,
                           feature_function, discount, learning_rate, trajectories, epochs)
    result = irl.gradientIterationIRL()

    pkl.dump(result, open("lg_result.pkl", 'wb'))

    return None

if __name__ == '__main__':
    train(0.01, 1, 300, 0.01)
    # train(0.9, 60, 400, 0.1) # for test

    # rewards = pkl.load(open("flappy_maxent_reward.pkl", 'rb'))
    #
    # env = Game.GameState(prepare_tp=True)
    #
    # value = vi.value(env.get_policy(), env.n_states, env.transition_probability, rewards, 0.3)
    # opt_value = vi.optimal_value(env.n_states, env.n_actions, env.transition_probability, rewards, 0.3)
    # pkl.dump(value, open("flappy_maxent_value.pkl", 'wb'))
    # pkl.dump(opt_value, open("flappy_maxent_opt_value.pkl", 'wb'))
    #
    #
    # value=pkl.load(open("flappy_maxent_value.pkl", 'rb'))
    # opt_value=pkl.load(open("flappy_maxent_opt_value.pkl", 'rb'))
    #
    # status = validate(value)
    # print(status)
    # pkl.dump(status, open("flappy_maxent_status.pkl", 'wb'))
    # status = validate(opt_value)
    # print(status)
    # pkl.dump(status, open("flappy_maxent_opt_status.pkl", 'wb'))
    # status = validate(rewards)
    # print(status)
    # pkl.dump(status, open("flappy_maxent_rewards_status.pkl", 'wb'))