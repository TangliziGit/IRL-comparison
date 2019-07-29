import numpy as np
import matplotlib.pyplot as plt

from airplane import Env

def validate(reward):
    env = Env(prepare_tp=False)

    def get_next_state_code_by_state(state):
        states=[]
        next_time_code=env._next_time_hash_code(state[1])
        for i in range(env.n_actions):
            states.append((env._next_position(state[0], i), next_time_code))
        print(state, "next_state", states)
        return states

    def chose_action(ob):
        states_code=[env._get_state_code(s) for s in get_next_state_code_by_state(ob)]
        rewards=np.array([reward[sc] for sc in states_code])
        max_actions=np.where(rewards == np.max(rewards))

        return np.random.choice(max_actions[0])

    def run(env):
        env.reset()

        print("start running")
        i=0
        img, observation, termial=env.frame_step(env.do_nothing)
        states=[env._get_ob_str(observation, 0)]

        while not termial:
            i+=1
            ob=env._get_ob_str(observation, i)
            action = chose_action(ob)

            img, observation_, termial = env.frame_step(action)

            observation = observation_
            states.append(env._get_ob_str(observation, i+1))

            if i>=len(env.expert_states):
                print("Done")
                break
            if termial:
                print("Terminal, your airplane dead")
                break
        return states

    length = len(env.expert_states)
    states = run(env)

    E_time = 0
    for rob, exp in zip(states, env.expert_states):
        if not (rob[0] == exp[0] and rob[1] == exp[1]):
            E_time += 1
    print("E_time", len(env.expert_states), len(states))
    E_time+=len(env.expert_states)-len(states)

    # self.status["KLnorm"].append(KL_norm)
    # self.status["KLAbsSum"].append(KL_abs_sum)
    # self.status["Error"].append(E)
    status={
        "ErrorRate": E_time / length,
        "ErrorTime": E_time,
        "ResultLength": len(states),
        "ExpertLength": len(env.expert_states),
    }
    return status

def draw_airplane_reward(desc, reward, if_save): # , env):
    # ground_r = np.array([env.reward(s) for s in range(env.n_states)])
    # ground_r = ground_r.reshape((4, 400))
    reward = reward.reshape((4, 400))

    # plt.subplot(1, 2, 1)
    # plt.pcolor(ground_r)
    # plt.colorbar()
    # plt.title("Groundtruth reward")

    # plt.subplot(1, 1, 1)
    plt.pcolor(reward)
    plt.colorbar()
    plt.title(desc)

    if if_save:
        plt.savefig(desc+".png")
    else:
        plt.show()