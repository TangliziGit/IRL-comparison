import sys
sys.path.append("/home/zhang/Liuying/result/Inverse-Reinforcement-Learning")

import numpy as np
import pickle as pkl

from irl.LargeGradientIRL_torch import LargeGradientIRL
from airplane import Env
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

    trajectory_length = 268
    n_sec_time_hashing=400
    n_states=4*n_sec_time_hashing
    n_actions=3
    dir = [-1, 1, 0]

    # env = Env()
    # trajectories = env.generate_trajectories(n_trajectories,
    #                                         trajectory_length,
    #                                         env.optimal_policy_deterministic)
    trajectories = pkl.load(open("airplane_trajectories.pkl", 'rb'))

    def _get_real_state_by_state_code(state_code):
        return (state_code//n_sec_time_hashing,
                state_code%n_sec_time_hashing)

    def _transition_probability(state, action, result_state):
        state = _get_real_state_by_state_code(state)
        result_state = _get_real_state_by_state_code(result_state)

        if state[1]+1!=result_state[1]:
            return 0

        next_pos=state[0]+dir[action]
        if next_pos<0: next_pos=0
        if next_pos>3: next_pos=3

        if next_pos == result_state[0]:
            return 1
        return 0

    def feature_function(state):
        feature = np.zeros(n_states)
        feature[state]=1
        return feature

    def transitionProbability(state_code, action):
        res = {}
        for i in range(n_states):
            res[state_code]=_transition_probability(state_code, action, i)
        return res


    irl = LargeGradientIRL(n_actions, n_states, transitionProbability,
                           feature_function, discount, learning_rate, trajectories, epochs)
    result = irl.gradientIterationIRL(save_file_name='airplane_lg_results_%d.pkl')

    pkl.dump(result, open("lg_result.pkl", 'wb'))

    return None

if __name__ == '__main__':
    train(0.01, 1, 400, 0.01)
    # rewards = pkl.load(open("maxent_reward.pkl", 'rb'))
    #
    # env = Env(prepare_tp=True)
    #
    # value = vi.value(env.get_policy(), env.n_states, env.transition_probability, rewards, 0.3)
    # opt_value = vi.optimal_value(env.n_states, env.n_actions, env.transition_probability, rewards, 0.3)
    # pkl.dump(value, open("maxent_value.pkl", 'wb'))
    # pkl.dump(opt_value, open("maxent_opt_value.pkl", 'wb'))
    #
    #
    # value=pkl.load(open("maxent_value.pkl", 'rb'))
    # opt_value=pkl.load(open("maxent_opt_value.pkl", 'rb'))
    #
    # status = validate(value)
    # print(status)
    # pkl.dump(status, open("maxent_status.pkl", 'wb'))
    # status = validate(opt_value)
    # print(status)
    # pkl.dump(status, open("maxent_opt_status.pkl", 'wb'))
    # status = validate(rewards)
    # print(status)
    # pkl.dump(status, open("maxent_rewards_status.pkl", 'wb'))
