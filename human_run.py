"""
A script to collect a batch of human demonstrations.

The demonstrations can be played back using the `playback_demonstrations_from_hdf5.py` script.
"""

import argparse
import datetime
import json
import os
import shutil
import time
from glob import glob

import h5py
import numpy as np

import robosuite as suite
import robosuite.macros as macros
from robosuite import load_controller_config
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
from robosuite.wrappers import GymWrapper
from robosuite.devices import Keyboard


def collect_human_trajectory(env, device, arm, env_configuration):
    env.reset()

    # ID = 2 always corresponds to agentview
    env.render()

    is_first = True

    task_completion_hold_count = -1  # counter to collect 10 timesteps after reaching goal
    device.start_control()

    # Loop until we get a reset from the input or the task completes
    while True:
        # Set active robot
        active_robot = env.robots[0] if env_configuration == "bimanual" else env.robots[arm == "left"]

        # Get the newest action
        action, grasp = input2action(
            device=device, robot=active_robot, active_arm=arm, env_configuration=env_configuration
        )

        # If action is none, then this a reset so we should break
        if action is None:
            break

        # Run environment step
        env.step(action)
        env.render()

        # Also break if we complete the task
        if task_completion_hold_count == 0:
            break

        # state machine to check for having a success for 10 consecutive timesteps
        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1  # latched state, decrement count
            else:
                task_completion_hold_count = 10  # reset count on first success timestep
        else:
            task_completion_hold_count = -1  # null the counter if there's no success

    # cleanup for end of data collection episodes
    env.close()

if __name__ == "__main__":

    # Get controller config
    controller_config = load_controller_config(default_controller="OSC_POSE")

    # Create argument configuration
    config = {
        "env_name": "Lift",
        "robots": "Panda",
        "controller_configs": controller_config,
    }

    pos_sensitivity = 1.0
    rot_sensitivity = 1.0

    env_name = "Stack"

    horizon = 3000
    arm = "right"
    arm_config = "single-arm-opposed"

    # Create environmentw
    env = suite.make(
        env_name,  # Environment
        robots=["Panda"],  # Use two Panda robotss
        controller_configs=suite.load_controller_config(default_controller="OSC_POSE"),  # Controller
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

    # Wrap this with visualization wrapper
    env = VisualizationWrapper(env)

    # # Grab reference to controller config and convert it to json-encoded string
    # env_info = json.dumps(config)

    # wrap the environment with data collection wrapper
    tmp_directory = "/tmp/{}".format(str(time.time()).replace(".", "_"))
    env = DataCollectionWrapper(env, tmp_directory)

    # initialize device

    device = Keyboard(pos_sensitivity=pos_sensitivity, rot_sensitivity=rot_sensitivity)


    # make a new timestamped directory
    t1, t2 = str(time.time()).split(".")

    # collect demonstrations
    while True:
        collect_human_trajectory(env, device, arm, arm_config)
