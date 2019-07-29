import numpy as np
import pickle as pkl
import tensorflow as tf

from irl.LargeGradientIRL import LargeGradientIRL
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

    env = Game.GameState()
    trajectories = env.generate_trajectories(n_trajectories,
                                            trajectory_length,
                                            env.optimal_policy_deterministic)

    def feature_function(state):
        feature = np.zeros(env.n_states)
        feature[state]=1
        return feature

    def transitionProbability(state_code, action):
        res = {}
        for i in range(env.n_states):
            res[state_code]=env._transition_probability(state_code, action, i)
        return res

    irl = LargeGradientIRL(env.n_actions, env.n_states, transitionProbability,
                           feature_function, discount, learning_rate, trajectories, epochs)
    result = irl.gradientIterationIRL()

    reward=result[-1][0].reshape(env.n_states, )
    pkl.dump(result, open("lg_result.pkl", 'wb'))
    pkl.dump(reward, open("lg_reward.pkl", 'wb'))

    return reward

if __name__ == '__main__':
    with tf.device('/cpu:0'):
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