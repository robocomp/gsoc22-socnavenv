import pytest
import sys
import os
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)) + "/..")
import gym
import socnavenv
import yaml


def test_shape():
    env = gym.make("SocNavEnv-v1")
    env.configure(os.path.dirname(os.path.abspath(__file__)) + "/../configs/env.yaml")
    env.set_padded_observations(True)
    obs = env.reset()

    config_file = os.path.dirname(os.path.abspath(__file__)) + "/../configs/env.yaml"
    with open(config_file, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)
    
    total_max_humans = (
        config["env"]["max_humans"]+ 
        config["env"]["max_h_h_dynamic_interactions"]*config["env"]["max_human_in_h_h_interactions"] + 
        config["env"]["max_h_h_static_interactions"]*config["env"]["max_human_in_h_h_interactions"]+ 
        config["env"]["max_h_l_interactions"]
    )
    assert(obs["humans"].shape == (total_max_humans * 13, ))
    assert(obs["plants"].shape == (config["env"]["max_plants"]*13, ))
    assert(obs["tables"].shape == (config["env"]["max_tables"]*13, ))
    assert(obs["laptops"].shape == ((config["env"]["max_laptops"] + config["env"]["max_h_l_interactions"])*13, ))

    env.set_padded_observations(False)
    obs = env.reset()
    assert(obs["humans"].shape == (env.total_humans * 13, ))
    assert(obs["plants"].shape == (env.NUMBER_OF_PLANTS*13, ))
    assert(obs["tables"].shape == (env.NUMBER_OF_TABLES*13, ))
    assert(obs["laptops"].shape == ((env.NUMBER_OF_LAPTOPS + env.NUMBER_OF_H_L_INTERACTIONS)*13, ))