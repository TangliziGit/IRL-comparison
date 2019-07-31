import pickle as pkl
import numpy as np
from tqdm import tqdm

import torch
from torch import nn

class LinearModel(nn.Module):
    def __init__(self, feature_shape):
        super(LinearModel, self).__init__()
        size=feature_shape[0]
        # self.linear=nn.Sequential(
        #     nn.Linear(size, 5 * size),
        #     nn.Tanh(),
        #     nn.Linear(5 * size, 5 * size),
        #     nn.Tanh(),
        #     nn.Linear(5 * size, 5 * size),
        #     nn.Tanh(),
        #     nn.Linear(5 * size, 5 * size),
        #     nn.Tanh(),
        #     nn.Linear(5 * size, 5 * size),
        #     nn.Tanh(),
        #     nn.Linear(5 * size, 2 * size),
        #     nn.Tanh(),
        #     nn.Linear(2 * size, 1),
        #     nn.Tanh(),
        # )
        self.linear = nn.Sequential(
            nn.Linear(size, 1),
            nn.ReLU(),
        )

    def forward(self, input):
        output=self.linear(input)
        return output

class NegativeLogLikelihoodLossFunction(nn.Module):
    def __init__(self, model, n_actions, n_states, transitionProbability,
                 featureFunction,discount):
        super(NegativeLogLikelihoodLossFunction, self).__init__()
        self.n_actions=n_actions
        self.n_states=n_states
        self.discount=discount
        self.transitionProbability=transitionProbability
        self.featureFunction=featureFunction

        self.model=model
        self.Q = {}
        self.approxValue = {}
        self.R = {}
        self.V = {}

    def approx(self, i):
        if i in self.approxValue:
            return self.approxValue[i]
        feature=self.featureFunction(i)
        features = torch.FloatTensor([
            feature.reshape(1, feature.shape[0])
        ])
        self.approxValue[i] = self.model(features)
        return self.approxValue[i]

    def Value(self, state):
        if state in self.V:
           return self.V[state]
        QValue_list=[]
        for j in range(self.n_actions):
            QValue_list.append(self.QValue(state, j))
        self.V[state]=torch.max( \
            torch.FloatTensor(QValue_list))
        return self.V[state]

    def QValue(self, state, action):
        if state in self.Q:
            return self.Q[state][action]
        QValue_list = []
        for j in range(self.n_actions):
            transition = self.transitionProbability(state, j)
            QValue = 0
            for nextstate in list(transition.keys()):
                QValue += transition[nextstate] * self.approx(nextstate)
            QValue_list.append(QValue)
        self.Q[state] = QValue_list
        return self.Q[state][action]

    def Reward(self, state):
        if state in self.R:
            return self.R[state]
        self.R[state] = self.approx(state) - self.discount * self.Value(state)
        return self.R[state]

    def get_reward_list(self):
        reward = [self.Reward(i) for i in tqdm(range(self.n_states))]
        return np.array(reward)

    def forward(self, trajectories):
        self.Q = {}
        self.approxValue = {}
        self.R = {}
        negativeloglikelihood = 0
        for trajectory in trajectories:
            for s, a, r in trajectory:
                QValue_list = []
                for j in range(self.n_actions):
                    QValue_list.append(self.QValue(s, j))
                QValue_list = [x - QValue_list[a] for x in QValue_list]
                QValue_list = torch.stack(QValue_list)
                Exp_QValue_list = torch.exp(QValue_list)
                Sum_QValue_list = torch.sum(Exp_QValue_list)
                negativeloglikelihood += torch.log(Sum_QValue_list)
        return negativeloglikelihood

class LargeGradientIRL(object):
    def __init__(self, n_actions, n_states, transitionProbability,
                 featureFunction,discount,learning_rate,trajectories,epochs):
        self.trajectories=trajectories
        self.learning_rate=learning_rate
        self.epochs=epochs

        self.model=LinearModel(featureFunction(0).shape)
        self.loss_func=NegativeLogLikelihoodLossFunction(
            self.model, n_actions, n_states, transitionProbability,
            featureFunction, discount
        )
        print(self.model)

    def gradientIterationIRL(self, ground_r=None, save_file_name=None):
        def show_grads(model):
            print("")
            for idx, layer in enumerate(model.linear):
                if hasattr(layer, 'weight'):
                    print("[%d] layer weight grad:"%idx, layer.weight.grad.sum())
                    print("[%d] layer bias grad:"%idx, layer.bias.grad.sum())

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        results=[]
        print("start learning")
        for epoch in range(self.epochs):
            loss=self.loss_func(self.trajectories)
            rewards=self.loss_func.get_reward_list()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            show_grads(self.model)
            if ground_r is not None:
                pearson=np.corrcoef(ground_r, rewards)[0,1]
                print(epoch, "%.12f"%loss.data.numpy(), pearson)
                results.append([rewards, pearson])
            else:
                print(epoch, "%.12f"%loss.data.numpy())
                results.append([loss, rewards])
            pkl.dump(results, open(save_file_name%(epoch), 'wb'))

            if loss < 1:
                break

        return results

if __name__=='__main__':
    # all in (x, y)
    world_size=(5, 5)
    n_actions=4
    n_states=world_size[0]*world_size[1]
    starting_state = (0, 4)
    ending_state = (4, 0)

    gr=np.zeros(world_size)
    gr = np.vstack([gr[:-1, :], np.ones((1, world_size[1]))])
    gr = np.hstack([gr[:, :-1], np.ones((world_size[0], 1))])
    gr=gr.reshape(n_states, )
    print(gr)

    expert_actions=[3,3,3,3 ,0,0,0]

    # (dx, dy)
    dir = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    def _transition_probability(state, action, next_state):
        state = int_to_state(state)
        next_state = int_to_state(next_state)
        real_next_state=get_next_state(state, action)

        if real_next_state[0]!=next_state[0] or real_next_state[1]!=next_state[1]:
            return 0
        return 1

    def feature_function(state):
        feature = np.zeros(n_states)
        feature[state]=1
        return np.array(feature).reshape((25, 1))

    def int_to_state(state_int):
        return (state_int // world_size[0], state_int%world_size[0])

    def state_to_int(state):
        return state[0]*world_size[0]+state[1]

    def get_next_state(state, action):
        dx, dy=dir[action]
        next_state = [state[0]+dx, state[1]+dy]
        next_state[0] = max(next_state[0], 0)
        next_state[0] = min(next_state[0], world_size[0] - 1)
        next_state[1] = max(next_state[1], 0)
        next_state[1] = min(next_state[1], world_size[1] - 1)
        return (next_state[0], next_state[1])

    def get_reward(state):
        if state[0]!=ending_state[0] or state[1]!=ending_state[1]:
            return 0
        return 1

    def gen_trajectories(n_trajectories):
        trajectories = []
        while len(trajectories) < n_trajectories:
            trajectory = []
            state = starting_state

            for action in expert_actions:
                next_state = get_next_state(state, action)
                reward = get_reward(next_state)
                state_int = state_to_int(state)

                trajectory.append((state_int, action, reward))
                state = next_state
            trajectories.append(trajectory)
        return trajectories

    def transitionProbability(state_int, action):
        res = {}
        state = int_to_state(state_int)
        for i in range(n_actions):
            next_state = get_next_state(state, i)
            next_state_int = state_to_int(next_state)
            if action == i:
                res[next_state_int] = 1
            else:
                res[next_state_int] = 0
        return res

    trajectories=gen_trajectories(2)
    # print(trajectories[0])
    # exit(0)

    irl=LargeGradientIRL(n_actions, n_states, transitionProbability,
                 feature_function, 0.3, 0.01, trajectories, 8000)
    result=irl.gradientIterationIRL()# gr)

    import matplotlib.pyplot as plt
    reward=result[-1][0].reshape(world_size)
    print(reward)
    plt.pcolor(reward)
    plt.colorbar()
    plt.show()