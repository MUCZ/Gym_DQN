from cart_pole import MyModel
from cart_pole import DQN
from cart_pole import play_game
from cart_pole import make_video

import numpy as np
import tensorflow as tf
import gym
import os
import datetime
from statistics import mean
from gym import wrappers


def main():
    ####################初始化#####################

    env = gym.make('CartPole-v0')
    gamma = 0.9
    num_states = len(env.observation_space.sample())
    num_actions = env.action_space.n
    hidden_units = [30, 30]
    max_experiences = 3000
    min_experiences = 1000
    batch_size = 32
    lr = 0.01
    e_greedy = 0.9
    e_greedy_increment = 1.02
    replace_target_iter = 50

    DQN_ = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, 
            min_experiences, batch_size, lr,e_greedy, replace_target_iter,e_greedy_increment)

    # DQN_.loadModel()
    # DQN_.loadMemory()
    
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/dqn/' + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)

    N = 200 # 总训练轮次
    Dispaly_interval = 30
    total_rewards = np.empty(N)
 
    ####################主循环#####################
    for n in range(N):
        total_reward, losses = play_game(env, DQN_)
        total_rewards[n] = total_reward
        avg_rewards = total_rewards[max(0, n - Dispaly_interval):(n + 1)].mean()
        with summary_writer.as_default():
            tf.summary.scalar('episode reward', total_reward, step=n)
            tf.summary.scalar('running avg reward(100)', avg_rewards, step=n)
            tf.summary.scalar('average loss)', losses, step=n)
        if n % Dispaly_interval == 0:
            print("episode:", n, "eps:", DQN_.epsilon, "avg reward (last ",Dispaly_interval,"):", avg_rewards,
                  "episode loss: ", losses)


    ####################结束#####################

    make_video(env, DQN_)
    # DQN_.saveModel()
    # DQN_.saveMemory()
    env.close()


if __name__ == '__main__':
    main()