import numpy as np
import tensorflow as tf
import gym
import os
import datetime
from statistics import mean
from gym import wrappers
import pickle

# keras.model的自定义子类
class MyModel(tf.keras.Model):
    def __init__(self, num_states, hidden_units, num_actions):
        super(MyModel, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_states,))
        self.hidden_layers = []
        for i in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(
                i, activation='tanh', kernel_initializer='RandomNormal'))
        self.output_layer = tf.keras.layers.Dense(
            num_actions, activation='linear', kernel_initializer='RandomNormal')
        
    
    #
    @tf.function
    def call(self, inputs):
        z = self.input_layer(inputs)
        for layer in self.hidden_layers:
            z = layer(z)
        output = self.output_layer(z)
        return output

    def predict(self, inputs):
        return self(np.atleast_2d(inputs.astype('float32')))

class DQN:
    def __init__(self, num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences,
                 batch_size, lr,e_greedy, replace_target_iter = 50,e_greedy_increment=None):
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.optimizer = tf.optimizers.Adam(lr)
        self.gamma = gamma
        self.model = MyModel(num_states, hidden_units, num_actions)
        self.target = MyModel(num_states, hidden_units, num_actions)
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []} #字典形式的记忆库
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences
        self.learn_step_counter = 0
        self.replace_target_iter = replace_target_iter

        self.epsilon_increment = e_greedy_increment
        self.epsilon_max = e_greedy
        self.epsilon = 0.1 if self.epsilon_increment is not None else self.epsilon_max


    # 以Target net作为参考训练
    def train(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.learn_step_counter = 0
            self.copy_weights()

        if len(self.experience['s']) < self.min_experiences:
            return 0 # 0 代表啥也没学到


        ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
        # {'s': array([-0.02874029, -0.58186904,  0.00809103,  0.89257765]), 
        # 'a': 1, 'r': 1.0, 's2': array([-0.04037767, -0.38685777,  0.02594258,  0.60244905]), 'done': False}
        # {'s': array([-0.13791506, -0.78732364,  0.19086538,  1.4356326 ]),
        #  'a': 0, 'r': -200, 's2': array([-0.15366153, -0.98421693,  0.21957803,  1.78138363]), 'done': True}
        #  state = x, x_dot, theta, theta_dot
        states = np.asarray([self.experience['s'][i] for i in ids])
        actions = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])
        states_next = np.asarray([self.experience['s2'][i] for i in ids])
        dones = np.asarray([self.experience['done'][i] for i in ids])

        value_next = np.max(self.target.predict(states_next), axis=1)
        # np.where(condition, x, y)
        # 满足条件(condition)，输出x，不满足输出y。
        actual_values = np.where(dones, rewards, rewards+self.gamma*value_next)

        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(
                self.model.predict(states) * tf.one_hot(actions, self.num_actions), axis=1)
            loss = tf.math.reduce_mean(tf.square(actual_values - selected_action_values))
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))


        self.learn_step_counter += 1   
        return loss 

    # 选取行动
    def get_action(self, states):
        if np.random.random() > self.epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.model.predict(np.atleast_2d(states))[0])

    # 记忆库
    def add_experience(self, exp):
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0) # 满了就删除一个
        for key, value in exp.items():
            self.experience[key].append(value)

    # 更新fronzen/target net
    def copy_weights(self):
        variables1 = self.target.trainable_variables
        variables2 = self.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())

    def epsilon_incre(self):
        self.epsilon = self.epsilon * self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max

    def saveModel(self):
        self.model.save_weights(r'D:\桌面\RL\DQN\eval_net.h5')
        self.target.save_weights(r'D:\桌面\RL\DQN\target_net.h5')
        print('Weights Saved .')

    def loadModel(self):
        # 模型没有被调用过则没有权重，调用一下生成初始权重
        self.model.predict(np.array([1,1,1,1]))
        self.target.predict(np.array([1,1,1,1]))
        self.model.load_weights(r'D:\桌面\RL\DQN\eval_net.h5')
        self.target.load_weights(r'D:\桌面\RL\DQN\target_net.h5')
        print('Weights loaded')

    def saveMemory(self):
        f = open(r'D:\桌面\RL\Memory.data', 'wb')
        # 将变量存储到目标文件中区
        pickle.dump(self.experience, f)
        # 关闭文件
        f.close()
        print('Memory saved .')

    def loadMemory(self):
        fr = open(r'D:\桌面\RL\Memory.data','rb')
        self.experience  = pickle.load(fr)
        print('Memory loaded.')

def play_game(env, DQN):
    rewards = 0
    done = False
    observations = env.reset()
    losses = list()
    DQN.epsilon_incre()
    while not done:
        # env.render() #注释掉则不显示GUI，训练加速

        action = DQN.get_action(observations)
        prev_observations = observations
        observations, reward, done, _ = env.step(action)
        reward = My_reward(env,observations)
        rewards += reward
        if done:
            reward = -10 #适用于cart_pole环境
            env.reset()

        exp = {'s': prev_observations, 'a': action, 'r': reward, 's2': observations, 'done': done}
        DQN.add_experience(exp)

        loss = DQN.train()
        if isinstance(loss, int): #当记忆库数量不足时loss=0
            losses.append(loss)
        else:
            losses.append(loss.numpy()) #类型转换

    return rewards, mean(losses)

def make_video(env, DQN):
    env = wrappers.Monitor(env, r"D:\桌面\RL\videos", force=True)
    rewards = 0
    steps = 0
    done = False
    observation = env.reset()
    DQN.epsilon = 1
    while not done:
        env.render()
        action = DQN.get_action(observation)
        observation, reward, done, _ = env.step(action)
        reward = My_reward(env,observation)
        steps += 1
        rewards += reward
    print("Testing steps: {} rewards {}: ".format(steps, rewards))

def My_reward(env,observations):

    x, _, theta, _ = observations
    rp = (env.x_threshold - abs(x))/env.x_threshold - 0.8  # reward position
    ra = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5 # reward angle
    reward = rp + ra 
    return reward

def My_reward_Car(env,observations): #适用于Mountaincar环境
    position, _ = observations
    # the higher the better
    reward = abs(position - (-0.5))     # r in [0, 1]
    return reward 

