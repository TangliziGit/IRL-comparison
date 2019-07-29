import sys

import numpy as np
import pickle as pkl

import irl.linear_irl as linear_irl
import irl.value_iteration as vi

import flappy.game.wrapped_flappy_bird as Game

from irl.validate import validate, draw_airplane_reward

def train(discount):
    """
    Run linear programming inverse reinforcement learning on the gridworld MDP.

    Plots the reward function.

    grid_size: Grid size. int.
    discount: MDP discount factor. float.
    """

    env = Game.GameState()

    r = linear_irl.irl(env.n_states, env.n_actions, env.transition_probability,
            env.get_policy(), discount, 1, 5)

    pkl.dump(r, open("flappy_lp_reward.pkl", 'wb'))


if __name__ == '__main__':
    sys.path.append("flappy/game/")

    train(0.2)
    rewards=pkl.load(open("flappy_lp_reward.pkl", 'rb'))

    env = Game.GameState(prepare_tp=True)
    value = vi.value(env.get_policy(), env.n_states, env.transition_probability,
                     rewards, 0.3)
    opt_value = vi.optimal_value(env.n_states, env.n_actions, env.transition_probability,
                     rewards, 0.3)
    pkl.dump(value, open("flappy_lp_value.pkl", 'wb'))
    pkl.dump(opt_value, open("flappy_lp_opt_value.pkl", 'wb'))

    # value = pkl.load(open("lp_value.pkl", 'rb'))
    # opt_value = pkl.load(open("lp_opt_value.pkl", 'rb'))

    status = validate(value)
    print(status)
    pkl.dump(status, open("flappy_lp_status.pkl", 'wb'))
    status = validate(value)
    print(status)
    pkl.dump(status, open("flappy_lp_opt_status.pkl", 'wb'))

    expert_path=np.array([1 if env.check_state_code_in_expert_path(s) else 0 for s in range(env.n_states)])

    draw_airplane_reward("reward", rewards)# , env)
    draw_airplane_reward('value', value)# , env)
    draw_airplane_reward('opt_value', opt_value)#, env)
    draw_airplane_reward('expert_path', expert_path)#, env)