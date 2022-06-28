import pytest
import sys
import os
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)) + "/..")
import gym
import socnavenv
from env_checker import check_env


def check(env):
    check_env(env)

def test_env():
    env = gym.make("SocNavEnv-v1")
    env.configure(os.path.dirname(os.path.abspath(__file__)) + "/../configs/env.yaml")
    env.set_padded_observations(False)
    check(env)
    env.set_padded_observations(True)
    check(env)