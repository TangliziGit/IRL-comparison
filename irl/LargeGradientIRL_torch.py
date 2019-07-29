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
            nn.Linear(size, 1 * size),
            nn.Tanh(),
            nn.Linear(1 * size, 1),
            nn.Tanh(),
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
        self.V[state]=torch.max(torch.FloatTensor(QValue_list))
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
                # QValue_list = QValue_list - QValue_list[a]
                # QValue_list = tf.stack(QValue_list)
                # Exp_QValue_list = tf.exp(QValue_list)
                # Sum_QValue_list = tf.reduce_sum(Exp_QValue_list)
                # negativeloglikelihood += tf.log(Sum_QValue_list)
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

    def gradientIterationIRL(self, ground_r=None):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        results=[]
        print("start learning")
        for epoch in range(self.epochs):
            loss=self.loss_func(self.trajectories)
            rewards=None # self.loss_func.get_reward_list()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if ground_r is not None:
                pearson=np.corrcoef(ground_r, rewards)[0,1]
                print(epoch, loss.data.numpy(), pearson)
                results.append([rewards, pearson])
            else:
                print(epoch, loss.data.numpy())
                results.append([loss, rewards])

            if loss < 1:
                break

        return results
