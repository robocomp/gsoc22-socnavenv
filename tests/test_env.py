import pytest
import sys
import os
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)) + "/..")
import gym
import socnavenv
from socnavenv.wrappers import WorldFrameObservations, PartialObservations, NoisyObservations, DiscreteActions
from env_checker import check_env
import numpy as np


def check(env):
    check_env(env)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_env():
    
    for i in range(10):
        env = gym.make("SocNavEnv-v1", config=os.path.dirname(os.path.abspath(__file__)) + "/../configs/test_env.yaml")
        env_world = WorldFrameObservations(env)
        env_noise = NoisyObservations(env, np.random.random(), np.random.random()+1e-5)
        env_partial = PartialObservations(env, np.random.random() * np.pi * 2)
        env_discrete = DiscreteActions(env)

        check(env)
        check(env_world)
        check(env_noise)
        check(env_partial)
        check(env_discrete)

        env.set_padded_observations(True)
        env_world.set_padded_observations(True)
        env_noise.set_padded_observations(True)
        env_partial.set_padded_observations(True)
        env_discrete.set_padded_observations(True)

        check(env)
        check(env_world)
        check(env_noise)
        check(env_partial)
        check(env_discrete)