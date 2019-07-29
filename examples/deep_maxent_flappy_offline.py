import irl.deep_maxent as deep_maxent
import pickle as pkl

import flappy.game.wrapped_flappy_bird as Game

from irl.validate import draw_airplane_reward

def main(discount, n_objects, n_colours, n_trajectories, epochs, learning_rate, structure):
    # n_objects, n_colours 随便给的
    """
    Run deep maximum entropy inverse reinforcement learning on the objectworld
    MDP.

    Plots the reward function.

    grid_size: Grid size. int.
    discount: MDP discount factor. float.
    n_objects: Number of objects. int.
    n_colours: Number of colours. int.
    n_trajectories: Number of sampled trajectories. int.
    epochs: Gradient descent iterations. int.
    learning_rate: Gradient descent learning rate. float.
    structure: Neural network structure. Tuple of hidden layer dimensions, e.g.,
        () is no neural network (linear maximum entropy) and (3, 4) is two
        hidden layers with dimensions 3 and 4.
    """

    trajectory_length = 268
    l1 = l2 = 0

    do_jump = 0
    do_nothing = 1

    n_first_states=15
    n_sec_states=760
    n_states=n_first_states*n_sec_states
    n_actions=2

    # env = Env(n_objects, n_colours)
    # env=Game.GameState()

    # ground_r = np.array([env.reward_deep_maxent(s) for s in range(env.n_states)])
    # policy = find_policy(env.n_states, env.n_actions, env.transition_probability,
    #                      ground_r, discount, stochastic=False)
    # trajectories = env.generate_trajectories(n_trajectories, trajectory_length,
    # trajectories = env.generate_trajectories(n_trajectories,
    #                                         trajectory_length,
    #                                         env.optimal_policy_deterministic)
    trajectories = pkl.load(open("deep_maxent_trajectories.pkl", 'rb'))
    # feature_matrix = env.feature_matrix_deep_maxent(discrete=False)

    feature_matrix = pkl.load(open("flappy_feature_matrix.pkl", 'rb')) #env.feature_matrix()

    transition_probability = pkl.load(open("flappy_transition_probability.pkl", 'rb'))

    r = deep_maxent.irl((feature_matrix.shape[1],) + structure, feature_matrix,
        n_actions, discount, transition_probability, trajectories, epochs,
        learning_rate, l1=l1, l2=l2)

    pkl.dump(r, open('flappy_deep_maxent_reward.pkl', 'wb'))

if __name__ == '__main__':
    main(0.9, 20, 2, 20, 400, 0.01, (3, 3))
    # reward = pkl.load(open('deep_maxent_reward_epoch_49.pkl', 'rb'))

    # draw_airplane_reward("deep maxent reward for airplane", reward, False)