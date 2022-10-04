import torch
import numpy as np
import gym
import socnavenv
import argparse

from socnavenv.envs.socnavenv_v1 import SocNavEnv_v1

if __name__ == "__main__":
    env:SocNavEnv_v1 = gym.make("SocNavEnv-v1")
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--env_config", required=True, help="path to env config")
    ap.add_argument("-a", "--agent", required=True, help="name of agent (dqn/ duelingdqn/ a2c/ ppo)")
    ap.add_argument("-t", "--type", required=False, help="type of network (mlp/transformer)")
    ap.add_argument("-c", "--config", required=True, help="path to config file for the agent")
    ap.add_argument('-n', '--num_episodes', required=True, help="number of episodes to train the agent")
    ap.add_argument('-w', "--weights", required=True, help="path to weight file")
    args = vars(ap.parse_args())

    env.configure(args["env_config"])

    if args["agent"].lower() == "duelingdqn":
        if args["type"].lower() == "transformer":
            from agents.duelingDQN_transformer import DuelingDQN_Transformer_Agent
            agent = DuelingDQN_Transformer_Agent(env, args["config"])
            agent.eval(int(args["num_episodes"]), args["weights"])
        
        else:
            raise NotImplementedError
    
    elif args["agent"].lower() == "ppo":
        if args["type"].lower() == "transformer":
            from agents.ppo_transformer import PPO_Transformer_Agent
            agent = PPO_Transformer_Agent(env, args["config"])
            agent.eval(int(args["num_episodes"]), actor_path=args["weights"])
        
        else:
            raise NotImplementedError
    else:
        """
        The eval method in the other agents have not been updated with the latest metrics
        """
        raise NotImplementedError
