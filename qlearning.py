

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 19:15:46 2021

@author: Robonuts Team, Paria Esmatloo (p.esmatloo@utexas.edu)
This code was a modification of the code for the Qlearning tutorial for 
simplified Frozen Lake problem by "The Computer Scientist" developed in 2018 
wich is linked below
https://github.com/the-computer-scientist/OpenAIGym/blob/master/QLearningIntro.ipynb

"""

import gym
import random
import numpy as np
import time
from gym.envs.registration import register
from IPython.display import clear_output

import csv

# from gym.envs.registration import registry, register, make, 


try:
    
    register(
        id='FrozenLakeNoSlip-v0',
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name' : '4x4', 'is_slippery':False},  # to add is_slippery to environment making
        max_episode_steps=100,
        reward_threshold=0.78, # optimum = .8196
    )
except:
    pass
        


# env_name = "CartPole-v1"
# env_name = "MountainCar-v0"
# env_name = "MountainCarContinuous-v0"
# env_name = "Acrobot-v1"
# env_name = "Pendulum-v0"
env_name = "FrozenLake-v0"
env_name = "FrozenLakeNoSlip-v0"




env= gym.make(env_name)
print("Observation spae:",env.observation_space)
print("Abservation spae:",env.action_space)

print(type(env.action_space))

# env.action_space.low, 
class Agent():
    def __init__(self,env):
        self.is_discrete = \
            type(env.action_space) == gym.spaces.discrete.Discrete
            
        if self.is_discrete:
            self.action_size = env.action_space.n
            print("Action size:",self.action_size)
        else:
            self.action_low = env.action_space.low
            self.action_high = env.action_space.high
            self.action_shape = env.action_space.shape
            print("Action range:" ,self.action_low, self.action_high)
        
    def get_action(self, state):
        if self.is_discrete:
            action = random.choice(range(self.action_size))
        # # pole_angle = state[2]
        # action =0 if pole_angle<0 else 1   # pushes the cart left is angle <0 and right otherwise
        # # print(action)
        else:
            l,h,sh=self.action_low , self.action_high, self.action_shape
            # h=self.action_high
            # sh=self.action_shape
            # print(type(l))
            action = np.random.uniform(l,h,sh)
        return action

class QAgent(Agent):
    def __init__(self,env, discount_rate = 0.97, learning_rate = 0.01):
        super().__init__(env)
        self.state_size = env.observation_space.n
        print("State size:", self.state_size)
        
        self.eps = 1
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        
        self.build_model()
        
    def build_model(self):
        self.q_table = 1e-4*np.random.random([self.state_size, self.action_size])
        
    def get_action(self, state):
        q_state = self.q_table[state]
        action_greedy = np.argmax(q_state)
        action_random = super().get_action(state)
        return action_random if random.random() < self.eps else action_greedy
    
    def train(self,experience):
        state, action, next_state, reward, done = experience
        
        q_next = self.q_table[next_state]
        q_next = np.zeros([self.action_size]) if done else q_next
        q_target = reward +self.discount_rate * np.max(q_next)
        
        q_update = q_target - self.q_table[state,action]
        
        self.q_table[state,action] +=self.learning_rate * q_update
        
        if done:
            self.eps = self.eps * 0.99
        
agent = QAgent(env)

total_reward =0

file1 = open ("MyReward.csv")
reward_state = list()

    
with open ("MyReward.csv", mode='w+') as reward_file:
    csv_writer = csv.writer(reward_file, delimiter=',')
    
    
    for ep in range(200):
    
        state = env.reset()
        done= False
    
        while not done:
            # action = env.action_space.sample()
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            agent.train((state,action,next_state,reward,done))
            state = next_state
            total_reward += reward
            reward_state.append(reward)
            
            
            print ('s:',state,'a:',action)
            print('Episode: {}, Total reward: {}, eps: {}'.format(ep,total_reward,agent.eps))
            env.render()
            print(agent.q_table)
            time.sleep(0.05)
            clear_output(wait=True)
        
        csv_writer.writerow([total_reward])