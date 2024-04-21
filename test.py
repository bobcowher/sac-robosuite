import time
import os
import gym
import pybullet_envs
import numpy as np
from buffer import ReplayBuffer
import datetime
from agent import SAC
from torch.utils.tensorboard import SummaryWriter
import robosuite as suite
# from robosuite_environment import RoboSuiteWrapper
from robosuite.wrappers import GymWrapper


if __name__ == '__main__':

    env_name = "Stack"
    replay_buffer_size = 10000000
    episodes = 3
    warmup = 100
    batch_size = 256
    updates_per_step = 1
    gamma = 0.99
    tau = 0.005
    alpha = 0.2
    policy = "Gaussian"
    target_update_interval = 1
    automatic_entropy_tuning = False
    hidden_size = 756
    learning_rate = 0.0003
    horizon=500 # max episode steps

    env = suite.make(
        env_name,  # Environment
        robots=["Panda"],  # Use two Panda robots
        controller_configs=suite.load_controller_config(default_controller="JOINT_VELOCITY"),  # Controller
        # controller_configs=suite.load_controller_config(default_controller="OSC_POSE"),
        has_renderer=True,  # Enable rendering
        use_camera_obs=False,
        horizon=horizon,
        render_camera="sideview",           # Camera view
        has_offscreen_renderer=True,        # No offscreen rendering
        reward_shaping=True,
        control_freq=20,  # Control frequency
    )
    env = GymWrapper(env)

    # Agent
    agent = SAC(env.observation_space.shape[0], env.action_space, gamma=gamma, tau=tau, alpha=alpha, policy=policy,
                target_update_interval=target_update_interval, automatic_entropy_tuning=automatic_entropy_tuning,
                hidden_size=hidden_size, learning_rate=learning_rate)



    for _ in range(episodes):
        agent.load_checkpoint(evaluate=True)
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = agent.select_action(state, evaluate=True)
            next_state, reward, done, _ = env.step(action)
            env.render()
            episode_reward += reward

            state = next_state

        print(f"Finished with a reward of {episode_reward}")


    env.close()