import sys
sys.path.append("/home/zhang/Liuying/result/Inverse-Reinforcement-Learning")

import numpy as np
import pickle as pkl

import irl.maxent as maxent
# from airplane import Env
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

    # wind = 0.3
    trajectory_length = 268
    n_actions=3

    # gw = gridworld.Gridworld(grid_size, wind, discount)
    # env = Env()
    # trajectories = env.generate_trajectories(n_trajectories,
    #                                         trajectory_length,
    #                                         env.optimal_policy_deterministic)
    # pkl.dump(trajectories, open('airplane_trajectories.pkl', 'wb'))
    # pkl.dump(env.feature_matrix(), open('airplane_feature_matrix.pkl', 'wb'))
    # pkl.dump(env.transition_probability, open('airplane_transition_probability.pkl', 'wb'))
    #
    # exit(0)
    trajectories = pkl.load(open('airplane_trajectories.pkl', 'rb'))
    feature_matrix = pkl.load(open('airplane_feature_matrix.pkl', 'rb'))
    transition_probability=pkl.load(open('airplane_transition_probability.pkl', 'rb'))

    r = maxent.irl(feature_matrix, n_actions, discount,
        transition_probability, trajectories, epochs, learning_rate,
        "alpha_%d.pkl", "alpha_209.pkl", 209)

    pkl.dump(r, open("maxent_reward.pkl", 'wb'))

    return r

if __name__ == '__main__':
    train(0.01, 1, 400, 0.01)
    rewards = pkl.load(open("maxent_reward.pkl", 'rb'))

    env = Env(prepare_tp=True)

    value = vi.value(env.get_policy(), env.n_states, env.transition_probability, rewards, 0.3)
    opt_value = vi.optimal_value(env.n_states, env.n_actions, env.transition_probability, rewards, 0.3)
    pkl.dump(value, open("maxent_value.pkl", 'wb'))
    pkl.dump(opt_value, open("maxent_opt_value.pkl", 'wb'))


    value=pkl.load(open("maxent_value.pkl", 'rb'))
    opt_value=pkl.load(open("maxent_opt_value.pkl", 'rb'))

    status = validate(value)
    print(status)
    pkl.dump(status, open("maxent_status.pkl", 'wb'))
    status = validate(opt_value)
    print(status)
    pkl.dump(status, open("maxent_opt_status.pkl", 'wb'))
    status = validate(rewards)
    print(status)
    pkl.dump(status, open("maxent_rewards_status.pkl", 'wb'))
