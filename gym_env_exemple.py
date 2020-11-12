import gym
import time 
from gym import wrappers
env = gym.make('CartPole-v1')
# env = gym.make('MountainCar-v0') 
print(env.x_threshold )
print(env.theta_threshold_radians )
print(env.action_space)
# print(env.observation_space)
# print(env.observation_space.high)
# print(env.observation_space.low)

for _ in range(20):
    env.reset()
    starttime = time.time()
    
    for _ in range(300): #刷新率大概是每秒选取60次动作
        
        env.render()  
        action = env.action_space.sample()
        observations, reward, done, _ = env.step(action)
        # print(observations, 'reward',reward, 'done',done)
        
    endtime = time.time()
    print('time: ',endtime-starttime)

env.close()