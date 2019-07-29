from PIL import Image
import torch
import copy
import pickle
import numpy as np
# from encoder import Encoder
from flappy.encoder_net import AutoEncoder

crop_points=(0, 0, 406, 288)

class AdaModel:
    def __init__(self, env, logger, RL):
        print("logger", logger)
        self.logger=logger
        
        if_load=False

        # row=380*2;col=380
        if if_load:
            RL.q_table=pickle.load(open("table_checkpoint_v4.pkl", "rb"))
            self.W=pickle.load(open("w_checkpoint_v4.pkl","rb"))
        self.row=380*2 # 512
        self.col=380 # 288
        self.do_nothing=[1, 0]
        self.do_flap=[0, 1]
        #self.origin=np.array([0, 0])*40+20
        self.logger.info("-"*20)
        self.logger.info("a new process!")
        self.eps=1e-6
        self.D=np.ones((self.row, self.col))/(self.row*self.col)

        self.expert_action=[int(x) for x in pickle.load(open("ops2_shorter.pkl", "rb"))]
        self.expert_action=self.expert_action[1:]
        self.expert_action_str=", ".join(pickle.load(open("ops2_shorter.pkl", "rb")))
        # self.expert_status=pickle.load(open("sts2_shorter.pkl", "rb"))

        # self.expert_step=[]
        # for ob in self.expert_status:
        #     self.expert_step.append(self.get_ob_str(ob, 0))

        # self.W=[np.zeros((self.row, self.col)) for i in range(len(self.expert_step))] # 0
        self.W=np.zeros((self.row, self.col)) 
        # self.expert_step_len=len(self.expert_step)
        self.irl_round_iter=20
        self.rl_learn_iter=1
        self.decay_time=0
        self.gamma=0.9
        self.encoder=AutoEncoder()
        self.encoder.load_state_dict(torch.load('encoder2.pkl'))
        self.init_expert(env, RL)
         
        self.status={
            "KLnorm": [],
            "KLAbsSum": [],
            "Error": [],
            "ErrorRate": [],
            "ErrorTime": [],
            "StateLength": [],
            "ExpertLength": []
        }
        if if_load:
            pass
            # self.status=pickle.load(open("status_checkpoint_v3.pkl","rb"))
        
        # print(len(self.expert_step))
        print(len(self.expert_imgs))
        # print(len(self.expert_status))
        print(len(self.expert_action))

    def env_init(self, env):
        env.__init__()

    def init_expert(self, env, RL):
        self.env_init(env)
        img, observation, termial=env.frame_step(self.do_nothing)
        self.expert_states=[self.get_ob_str(observation, 0)]
        self.expert_imgs=[img]

        for _i, action in enumerate(self.expert_action):
            action=[action, 1-action]
            # print(_i)
            img, observation_, termial=env.frame_step(action)
            ob=self.get_ob_str(observation, _i)
            ob_=self.get_ob_str(observation_, _i+1)

            # print(ob, "action:", action)

            observation = observation_
            self.expert_states.append(ob_)
            self.expert_imgs.append(img)

            if termial==True:
                break
        print(len(self.expert_imgs))
        print(len(self.expert_states))
        self.logger.info("expert_states:")
        self.logger.info(self.expert_states)

    def run(self, env, RL):
        self.env_init(env)
        i=0
        img, observation, termial=env.frame_step(self.do_nothing)
        states=[self.get_ob_str(observation, 0)]
        imgs=[img]

        while not termial:
            i+=1
        # for i in range(self.expert_step_len-1):
            ob=self.get_ob_str(observation, i)
            print(ob)
            action = RL.choose_action(str(ob))
            print("action:", action)

            action = [action, 1-action]
            img, observation_, termial = env.frame_step(action)

            observation = observation_
            states.append(self.get_ob_str(observation, i+1))
            imgs.append(img)

        self.logger.info("states:")
        self.logger.info(states)
        return states, imgs

    def getKL(self, s, es):
        s=Image.fromarray(s).crop(crop_points)
        es=Image.fromarray(es).crop(crop_points)
        return self.encoder.getKL(s, es)

    def get_ob_str(self, observation, time):
        ob=self.get_logical_state(observation)
        return np.array([ob[0], ob[1]])

    def learn(self, env, RL):
        step=0
        for episode in range(self.rl_learn_iter):
            self.env_init(env)
            self.logger.info("#%03d RL iter"%(episode))
            img, observation, termial = env.frame_step(self.do_nothing)
            actions_int64=[np.int64(1)]
            actions=[]
            states=[self.get_ob_str(observation, 0)]
            imgs=[img]
            reward=0

            for i, eimg in enumerate(self.expert_states[1:]):
                ob=self.get_ob_str(observation, i)
                _action = RL.choose_action(str(ob))
                # print(type(_action), _action)

                actions.append(str(_action))
                actions_int64.append(_action)
                action = [_action, 1-_action]
                print("action: ", ", ".join(actions))
                print("expert: ", self.expert_action_str)
                img, observation_, termial = env.frame_step(action)
                ob_=self.get_ob_str(observation_, i+1)

                states.append(ob_)
                imgs.append(img)
                #reward=self.get_reward(states, imgs, i+1, reward)
                reward=self.get_reward(states, imgs, len(states)-1, reward)

                # RL.learn(str(ob), _action, reward, str(ob_))
                # RL.learn(str(states[-2]), actions_int64[-2], reward, str(ob)])
                if termial:
                    RL.learn(str(states[-2]), actions_int64[-1], reward, 'terminal')
                    break
                RL.learn(str(states[-2]), actions_int64[-1], reward, str(states[-1]))
                print("states:",states )

                observation = observation_
                step+=1
                # reward*=self.decay_time
                reward=0
            self.logger.info("states in RL:")
            self.logger.info(states)
            self.logger.info("actions in RL:")
            self.logger.info("(1 for nothing, 0 for jump)")
            self.logger.info(", ".join(actions))
            self.logger.info("actions(expert) in RL:")
            self.logger.info(self.expert_action_str)

    def get_reward(self, states, imgs, time, prev_reward):
        # KL_vec=self.get_KL_vec(imgs, self.expert_imgs)
        # print(len(imgs), len(self.expert_imgs), time)
        print("-"*30)
        print("time", time)
        print("prev_reward", prev_reward)
        # KL=abs(states[time][1]-self.expert_states[time][1])
        KL=self.getKL(imgs[time], self.expert_imgs[time])
        print("KL", KL)
        rob=states[time]
        exp=self.expert_states[time]
        print("rob", rob)
        print("exp", exp)

        reward=prev_reward
        if not(rob[0]==exp[0] and rob[1]==exp[1]):
            # reward-=self.W[time][exp[1], exp[0]]*KL
            reward-=self.W[exp[1], exp[0]]*KL
            self.logger.debug("time%d y%d x%d, w%f kl%f"%(time, exp[1], exp[0], self.W[exp[1], exp[0]], KL))

        self.logger.info("reward %f"%(reward))
        return reward

    def get_logical_state(self, s):
        #return [int((s[0]+15-20)/40), int((s[1]+15-20)/40)]
        return [int(s[0]), int(s[1]+380)]

    def get_KL_vec(self, imgs, eimgs):
        res=[]
        for img, eimg in zip(imgs, eimgs):
            res.append(self.getKL(img, eimg))
        return np.array(res)

    def update(self, env, RL):
        i=0
        length=len(self.expert_states)
        
        while True:
            i+=1
            # if i==50:
            #     RL.epsilon=0.95
            # if i==100:
            #     RL.epsilon=0.98
            # if i==150:
            #     RL.epsilon=0.99

            # train with limited iteration
            # run with best policy as new states
            self.logger.info("#"*30)
            self.logger.info("NEW UPDATE")
            self.logger.info("#"*30)
            RL.rand_enable=False
            states, imgs=self.run(env, RL)

            # KL_vec=self.get_KL_vec(states, self.expert_states)
            KL_vec=self.get_KL_vec(imgs, self.expert_imgs)
            # KL_norm=np.linalg.norm(KL_vec)
            KL_abs_sum=np.sum(np.absolute(KL_vec))

            E=0
            E_time=0
            for rob, exp in zip(states, self.expert_states):
                # print("rob, exp:", rob, exp)
                if not(rob[0]==exp[0] and rob[1]==exp[1]):
                    E+=self.D[rob[1], rob[0]]+self.D[exp[1], exp[0]]
                    E_time+=1
            E_time+=len(self.expert_states)-len(states)
            self.logger.info("E:")
            self.logger.info(E)

            # self.status["KLnorm"].append(KL_norm)
            self.status["KLAbsSum"].append(KL_abs_sum)
            self.status["Error"].append(E)
            self.status["ErrorRate"].append(E_time/length)
            self.status["ErrorTime"].append(E_time)
            self.status["StateLength"].append(len(states))
            self.status["ExpertLength"].append(len(self.expert_states))
            self.logger.info("@%03d"%(i))
            self.logger.info(self.status)
            # print(RL.q_table)
            pickle.dump(RL.q_table, open("table3.pkl", "wb"))
            pickle.dump(self.W, open("w3.pkl", "wb"))
            pickle.dump(self.status, open("status3.pkl", "wb"))
            #pickle.load(open("table.pkl"),"wb")
            #pickle.load(open("w.pkl","wb"))
            if KL_abs_sum<self.eps:
                break
            if i>1500:
                break

            newD=self.D*E/(1-E)
            for rob, exp in zip(states, self.expert_states):# self.expert_step):
                if not(rob[0]==exp[0] and rob[1]==exp[1]):
                    newD[rob[1], rob[0]]=self.D[rob[1], rob[0]]
                    newD[exp[1], exp[0]]=self.D[exp[1], exp[0]]
            newD=newD/np.sum(newD)
            # self.logger.info("newD:")
            self.D=newD
            # self.logger.info(self.D)

            print(states)
            print(self.expert_states)
            print(len(states), len(imgs))
            print(len(self.expert_states), len(self.expert_imgs))
            
            newW=self.W.copy()
            error_time=0
            for _i, (rob, exp) in enumerate(zip(states, self.expert_states)):
                if not(rob[0]==exp[0] and rob[1]==exp[1]):
                    print("rob", rob)
                    print("exp", exp)
                    print("rW:", newW[rob[1], rob[0]])
                    print("eW:", newW[exp[1], exp[0]])
                    newW[rob[1], rob[0]]+=newD[rob[1], rob[0]]*KL_vec[_i]*pow(self.gamma, error_time)# _i)
                    newW[exp[1], exp[0]]+=newD[exp[1], exp[0]]*KL_vec[_i]*pow(self.gamma, error_time)# _i)

                    print("_i:", _i)
                    print("KL:", KL_vec[_i])
                    print("rD:", newD[rob[1], rob[0]])
                    print("eD:", newD[exp[1], exp[0]])
                    print("rW:", newW[rob[1], rob[0]])
                    print("eW:", newW[exp[1], exp[0]])
                    error_time+=1
            # for _i, (rob, exp) in enumerate(zip(states, self.expert_states)):
            #     if _i!=0:
            #         self.W[_i]=copy.deepcopy(self.W[_i-1])# self.W[_i-1].copy()
            #     if not(rob[0]==exp[0] and rob[1]==exp[1]):

            #         print("rob", rob)
            #         print("exp", exp)
            #         print("rW:", self.W[_i][rob[1], rob[0]])
            #         print("eW:", self.W[_i][exp[1], exp[0]])
            #         self.W[_i][rob[1], rob[0]]+=newD[rob[1], rob[0]]*KL_vec[_i]*pow(self.gamma, _i)
            #         self.W[_i][exp[1], exp[0]]+=newD[exp[1], exp[0]]*KL_vec[_i]*pow(self.gamma, _i)

            #         print("_i:", _i)
            #         print("KL:", KL_vec[_i])
            #         print("rD:", newD[rob[1], rob[0]])
            #         print("eD:", newD[exp[1], exp[0]])
            #         print("rW:", self.W[_i][rob[1], rob[0]])
            #         print("eW:", self.W[_i][exp[1], exp[0]])
            # self.logger.info("newW:")
            # self.logger.info(self.W)
            self.W=newW

            self.logger.info("#"*30)
            self.logger.info("NEW RUNNING")
            self.logger.info("#"*30)

            self.learn(env, RL)
