import numpy as np
import pickle as pkl

import irl.maxent as maxent
# import flappy.game.wrapped_flappy_bird as Game

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
    n_actions = 2


    # env = Game.GameState()
    # trajectories = env.generate_trajectories(n_trajectories,
    #                                         trajectory_length,
    #                                         env.optimal_policy_deterministic)
    trajectories = pkl.load(open('lg_trajectories.pkl', 'rb'))
    # feature_matrix = env.feature_matrix()
    feature_matrix = pkl.load(open('flappy_feature_matrix.pkl', 'rb'))
    transition_probability=pkl.load(open('flappy_transition_probability.pkl', 'rb'))

    r = maxent.irl(feature_matrix, n_actions, discount,
        transition_probability, trajectories, epochs, learning_rate,
        "flappy_alpha_%d.pkl", "flappy_alpha_99.pkl", 99)

    pkl.dump(r, open("flappy_maxent_reward.pkl", 'wb'))

    return r

if __name__ == '__main__':
    train(0.01, 1, 400, 0.01)
    rewards = pkl.load(open("flappy_maxent_reward.pkl", 'rb'))

    env = Game.GameState(prepare_tp=True)

    value = vi.value(env.get_policy(), env.n_states, env.transition_probability, rewards, 0.3)
    opt_value = vi.optimal_value(env.n_states, env.n_actions, env.transition_probability, rewards, 0.3)
    pkl.dump(value, open("flappy_maxent_value.pkl", 'wb'))
    pkl.dump(opt_value, open("flappy_maxent_opt_value.pkl", 'wb'))


    value=pkl.load(open("flappy_maxent_value.pkl", 'rb'))
    opt_value=pkl.load(open("flappy_maxent_opt_value.pkl", 'rb'))

    status = validate(value)
    print(status)
    pkl.dump(status, open("flappy_maxent_status.pkl", 'wb'))
    status = validate(opt_value)
    print(status)
    pkl.dump(status, open("flappy_maxent_opt_status.pkl", 'wb'))
    status = validate(rewards)
    print(status)
    pkl.dump(status, open("flappy_maxent_rewards_status.pkl", 'wb'))
