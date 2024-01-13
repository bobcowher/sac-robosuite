import time

import gym
import pybullet_envs
import numpy as np
from sac_torch import Agent
from torch.utils.tensorboard import SummaryWriter
import robosuite as suite
from robosuite_environment import RoboSuiteWrapper


if __name__ == '__main__':

    env_name = "Lift"

    env = suite.make(
        env_name,  # Environment
        robots=["Panda"],  # Use two Panda robots
        controller_configs=suite.load_controller_config(default_controller="JOINT_VELOCITY"),  # Controller
        # controller_configs=suite.load_controller_config(default_controller="OSC_POSE"),
        has_renderer=False,  # Enable rendering
        use_camera_obs=False,
        # render_camera="sideview",           # Camera view
        # has_offscreen_renderer=True,        # No offscreen rendering
        reward_shaping=True,
        control_freq=20,  # Control frequency
    )
    env = RoboSuiteWrapper(env)

    # agent = Agent(input_dims=env.observation_space.shape, env=env, n_actions=env.action_space.shape[0])
    agent = Agent(input_dims=env.input_dims, env=env, n_actions=env.action_dim)
    writer = SummaryWriter('logs')
    n_games = 250
    best_score = 0
    score_history = []
    load_checkpoint = False
    episode_identifier = 0

    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            observation = observation_
        score_history.append(score)
        writer.add_scalar(f"Score - {episode_identifier}", score, global_step=i)

        if(len(score_history)>100):
            avg_score = np.mean(score_history[-100])
        else:
            avg_score = np.mean(score_history)



        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)

