import numpy as np
import pickle as pkl

p = ''

class FlappyUtils:
    do_jump = 0
    do_nothing = 1

    n_first_states = 15
    n_sec_states = 760
    n_states = n_first_states * n_sec_states
    n_actions = 2
    trajectories = pkl.load(open(p + 'flappy_trajectories.pkl', 'rb'))
    feature_matrix = pkl.load(open(p + 'flappy_feature_matrix.pkl', 'rb'))

    def _get_state_by_state_code(state_code):
        return (state_code // FlappyUtils.n_sec_states,
                state_code % FlappyUtils.n_sec_states)

    def _transition_probability(state_code, action, result_state_code):
        if result_state_code == -1:
            return np.array([FlappyUtils._transition_probability(state_code, action, s) \
                             for s in range(FlappyUtils.n_states)])
        state = FlappyUtils._get_state_by_state_code(state_code)
        result_state = FlappyUtils._get_state_by_state_code(result_state_code)

        if state[0] + 1 != result_state[0]:
            return 0

        if action == FlappyUtils.do_jump and result_state[1] == state[1] + 9:
            return 1
        if action == FlappyUtils.do_nothing:
            # hard to get real probability, due to the simple states
            return 1 / FlappyUtils.n_sec_states  # v1
            # if abs(result_state[1]-state[1])<=50: # v2
            #     return 0.01
            # idx = find_idx_in_expert_states_by_state_code(state_code)
            # nidx = find_idx_in_expert_states_by_state_code(result_state_code)
            # if nidx==idx+1:
            #     return 1
        return 0


class AirUtils:
    n_sec_time_hashing = 400
    n_states = 4 * n_sec_time_hashing
    n_actions = 3
    _dir = [-1, 1, 0]
    trajectories = pkl.load(open(p + "airplane_trajectories.pkl", 'rb'))

    def _get_real_state_by_state_code(state_code):
        return (state_code // AirUtils.n_sec_time_hashing,
                state_code % AirUtils.n_sec_time_hashing)

    def _transition_probability(state_code, action, result_state_code):
        if result_state_code == -1:
            return np.array([AirUtils._transition_probability(state_code, action, s) \
                             for s in range(AirUtils.n_states)])
        state = AirUtils._get_real_state_by_state_code(state_code)
        result_state = AirUtils._get_real_state_by_state_code(result_state_code)

        if state[1] + 1 != result_state[1]:
            return 0

        next_pos = state[0] + AirUtils._dir[action]
        if next_pos < 0: next_pos = 0
        if next_pos > 3: next_pos = 3

        if next_pos == result_state[0]:
            return 1
        return 0

    def feature_function(state):
        feature = np.zeros(AirUtils.n_states)
        feature[state] = 1
        return feature

    def transitionProbability(state_code, action):
        res = {}
        for i in range(AirUtils.n_states):
            res[state_code] = AirUtils._transition_probability(state_code, action, i)
        return res


from tqdm import tqdm

def value(policy, n_states, transition_probabilities, reward, discount,
                    threshold=1e-1):
    v = np.zeros(n_states)

    diff = float("inf")
    while diff > threshold:
        diff = 0
        for s in range(n_states):
            vs = v[s]
            a = policy[s]
            v[s] = sum(transition_probabilities(s, a, k) *
                       (reward[k] + discount * v[k])
                       for k in range(n_states))
            diff = max(diff, abs(vs - v[s]))

    return v

def optimal_value(n_states, n_actions, transition_probabilities, reward,
                  discount, threshold=1e-1):
    v = np.zeros(n_states)

    diff = float("inf")
    while diff > threshold:
        diff = 0
        for s in range(n_states):
            max_v = float("-inf")
            for a in range(n_actions):
                tp = transition_probabilities(s, a, -1)
                max_v = max(max_v, np.dot(tp, reward + discount*v))

            new_diff = abs(v[s] - max_v)
            if new_diff > diff:
                diff = new_diff
            v[s] = max_v
        print("diff:", diff, "threshold:", threshold)

    return v

def find_policy(n_states, n_actions, transition_probabilities, reward, discount,
                threshold=1e-2, v=None, stochastic=True):
    if v is None:
        v = optimal_value(n_states, n_actions, transition_probabilities, reward,
                          discount, threshold)

    if stochastic:
        # Get Q using equation 9.2 from Ziebart's thesis.
        Q = np.zeros((n_states, n_actions))
        for i in range(n_states):
            for j in range(n_actions):
                p = transition_probabilities(i, j, -1)
                Q[i, j] = p.dot(reward + discount*v)
        Q -= Q.max(axis=1).reshape((n_states, 1))  # For numerical stability.
        Q = np.exp(Q)/np.exp(Q).sum(axis=1).reshape((n_states, 1))
        return Q

    def _policy(s):
        return max(range(n_actions),
                   key=lambda a: sum(transition_probabilities(s, a, k) *
                                     (reward[k] + discount * v[k])
                                     for k in range(n_states)))
    policy = np.array([_policy(s) for s in tqdm(range(n_states))])
    return policy


import os

os.environ["SDL_VIDEODRIVER"] = "dummy"

from airplane import Env
import flappy.game.wrapped_flappy_bird as Game

expert_trajectory = {'airplane': None, 'flappy': None, }
n_states = {'airplane': 4 * 400, 'flappy': 15 * 760}
n_actions = {'airplane': 3, 'flappy': 2}
n_tra = {'airplane': 270, 'flappy': 15}
transition_probability = {
    'airplane': AirUtils._transition_probability,
    'flappy': FlappyUtils._transition_probability,
}

env_class = {
    'airplane': Env,
    'flappy': Game.GameState,
}

n_iters = {
    'deep_maxent_airplane': 300,
    'deep_maxent_flappy': 80,
    'maxent_airplane': 250,     # waiting
    'maxent_flappy': 132,
    'lg_airplane': 1,
    'lg_flappy': 1,
}

feature_matrix = {
    'airplane': pkl.load(open(p + 'feature_matrix.pkl', 'rb')),
    'flappy': pkl.load(open(p + 'flappy_feature_matrix.pkl', 'rb')),
}

get_reward = {
    'deep_maxent_airplane':
        lambda i: pkl.load(open('deep_maxent_reward_epoch_%d.pkl' % i, 'rb')),
    'deep_maxent_flappy':
        lambda i: pkl.load(open('flappy_deep_maxent_reward_epoch_%d.pkl' % i, 'rb')),
    'maxent_airplane':
        lambda i: feature_matrix['airplane'].dot(pkl.load(open('alpha_%d.pkl' % i, 'rb'))),
    'maxent_flappy':
        lambda i: feature_matrix['flappy'].dot(pkl.load(open('flappy_alpha_%d.pkl' % i, 'rb'))),
    'lg_airplane':
        lambda i: pkl.load(open("airplane_lg_reward.pkl", 'rb')),
    'lg_flappy':
        lambda i: pkl.load(open("flappy_lg_reward.pkl", 'rb')),
}

result = {}

def validate(env_name='airplane', algo_env_name='lg_airplane', result_filename="airplane_lg_result_%d.pkl"):
    env = env_class[env_name]()
    if expert_trajectory[env_name] is None:
        expert_trajectory[env_name] = env.generate_trajectories(1, n_tra[env_name], env.optimal_policy_deterministic)[0]

    result[algo_env_name] = []
    for i in range(n_iters[algo_env_name]):
        reward = get_reward[algo_env_name](i)
        policy = find_policy(n_states[env_name], n_actions[env_name],
                             transition_probability[env_name], reward, 0.3, stochastic=False)
        trajectory = env.generate_trajectories(1, n_tra[env], lambda s: policy[s])[0]

        error_time = 0
        live_time = len(trajectory)
        for idx, (etra, tra) in enumerate(zip(expert_trajectory[env_name], trajectory)):
            if etra[0] != tra[0]: error_time += 1

        error_rate = error_time / len(expert_trajectory[env_name])
        live_rate = live_time / len(expert_trajectory[env_name])

        print(f'iter:{i}, error_rate: {error_rate}, live_rate: {live_rate}')
        pkl.dump((error_rate, live_rate), open(result_filename % i, 'wb'))
        result[algo_env_name].append((error_rate, live_rate))


def get_env_name(algo_env_name):
    if 'airplane' in algo_env_name:
        return 'airplane'
    elif 'flappy' in algo_env_name:
        return 'flappy'
    return 'WRAN'


if __name__=='__main__':
    import sys

    algo_env_name_list=['deep_maxent_airplane', 'deep_maxent_flappy',
                        'maxent_airplane', 'maxent_flappy',
                        'lg_airplane', 'lg_flappy',]
    for algo_env_name in [algo_env_name_list[int(sys.argv[1])]]:
        print(algo_env_name)
        env_name=get_env_name(algo_env_name)
        validate(env_name, algo_env_name, result_filename=algo_env_name+"_result_%d_v.pkl")