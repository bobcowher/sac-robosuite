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
    episodes = 10000
    warmup = 20
    batch_size = 64
    updates_per_step = 1
    gamma = 0.99
    tau = 0.005
    alpha = 0.15 # Temperature parameter.
    policy = "Gaussian"
    target_update_interval = 1
    automatic_entropy_tuning = False
    hidden_size = 756
    learning_rate = 0.0001
    horizon=500 # max episode steps

    env = suite.make(
        env_name,  # Environment
        robots=["Panda"],  # Use two Panda robots
        controller_configs=suite.load_controller_config(default_controller="JOINT_VELOCITY"),  # Controller
        # controller_configs=suite.load_controller_config(default_controller="OSC_POSE"),
        has_renderer=False,  # Enable rendering
        use_camera_obs=False,
        horizon=horizon,
        # render_camera="sideview",           # Camera view
        # has_offscreen_renderer=True,        # No offscreen rendering
        reward_shaping=True,
        control_freq=20,  # Control frequency
    )
    env = GymWrapper(env)

    # Agent
    agent = SAC(env.observation_space.shape[0], env.action_space, gamma=gamma, tau=tau, alpha=alpha, policy=policy,
                target_update_interval=target_update_interval, automatic_entropy_tuning=automatic_entropy_tuning,
                hidden_size=hidden_size, learning_rate=learning_rate)

    agent.load_checkpoint()

    # Tesnorboard
    writer = SummaryWriter(f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_SAC')

    # Memory
    memory = ReplayBuffer(replay_buffer_size, input_shape=env.observation_space.shape, n_actions=env.action_space.shape[0])

    # Training Loop
    total_numsteps = 0
    updates = 0

    for i_episode in range(episodes):
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()

        while not done:
            if warmup > i_episode:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(state)  # Sample action from policy

            if memory.can_sample(batch_size=batch_size):
                # Number of updates per step in environment
                for i in range(updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory,
                                                                                                         batch_size,
                                                                                                         updates)

                    writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    writer.add_scalar('loss/policy', policy_loss, updates)
                    writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                    updates += 1

            next_state, reward, done, _ = env.step(action)  # Step
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == horizon else float(not done)

            memory.store_transition(state, action, reward, next_state, mask)  # Append transition to memory

            state = next_state

        writer.add_scalar('reward/train', episode_reward, i_episode)
        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps,
                                                                                      episode_steps,
                                                                                      round(episode_reward, 2)))
        if i_episode % 10 == 0:
            agent.save_checkpoint(env_name=env_name)


    env.close()