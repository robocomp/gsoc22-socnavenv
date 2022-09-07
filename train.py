import gym
import torch
import argparse
import socnavenv
from socnavenv.wrappers import WorldFrameObservations

class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            if value[0] == '[':
                try:
                    value = list(map(int,value[1:-1].split(',')))
                except:
                    raise NameError(f"invalid argument for {key}")
            
            elif value == "True":
                value = True
            
            elif value == "False":
                value = False

            else:
                try:
                    value = float(value)
                except:
                    value = str(value)
            getattr(namespace, self.dest)[key] = value

if __name__ == "__main__":
    env = gym.make("SocNavEnv-v1")

    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--env_config", required=True, help="path to env config")
    ap.add_argument("-a", "--agent", required=True, help="name of agent (dqn/ duelingdqn/ a2c/ ppo)")
    ap.add_argument("-t", "--type", required=False, help="type of network (mlp/transformer)")
    ap.add_argument("-c", "--config", required=True, help="path to config file for the agent")
    ap.add_argument('-k', '--kwargs', nargs='*', action=ParseKwargs)
    args = vars(ap.parse_args())

    env.configure(args["env_config"])

    if args["agent"].lower() == "dqn":
        if args["type"].lower() == "mlp":
            from agents.DQN import DQNAgent
            env.set_padded_observations(True)
            if args["kwargs"] is not None:
                agent = DQNAgent(env, args["config"], **args["kwargs"])
            else:
                agent = DQNAgent(env, args["config"])
            agent.train()
        
        elif args["type"].lower() == "transformer":
            from agents.DQN_transformer import DQN_Transformer_Agent
            if args["kwargs"] is not None:
                agent = DQN_Transformer_Agent(env, args["config"], **args["kwargs"])
            else:
                agent = DQN_Transformer_Agent(env, args["config"])
            agent.train()
        
        else: raise NotImplementedError()

    elif args["agent"].lower() == "duelingdqn":
        if args["type"].lower() == "mlp":
            from agents.duelingDQN import DuelingDQNAgent
            env.set_padded_observations(True)
            if args["kwargs"] is not None:
                agent = DuelingDQNAgent(env, args["config"], **args["kwargs"])
            else:
                agent = DuelingDQNAgent(env, args["config"])
            agent.train()
        
        elif args["type"].lower() == "transformer":
            from agents.duelingDQN_transformer import DuelingDQN_Transformer_Agent
            if args["kwargs"] is not None:
                agent = DuelingDQN_Transformer_Agent(env, args["config"], **args["kwargs"])
            else:
                agent = DuelingDQN_Transformer_Agent(env, args["config"])
            agent.train()
        
        else: raise NotImplementedError()
    
    elif args["agent"].lower() == "a2c":
        if args["type"].lower() == "mlp":
            from agents.a2c import A2CAgent
            env.set_padded_observations(True)
            if args["kwargs"] is not None:
                agent = A2CAgent(env, args["config"], **args["kwargs"])
            else:
                agent = A2CAgent(env, args["config"])
            agent.train()
        
        elif args["type"].lower() == "transformer":
            from agents.a2c_transformer import A2C_Transformer_Agent
            if args["kwargs"] is not None:
                agent = A2C_Transformer_Agent(env, args["config"], **args["kwargs"])
            else:
                agent = A2C_Transformer_Agent(env, args["config"])
            agent.train()
        
        else: raise NotImplementedError()

    elif args["agent"].lower() == "ppo":
        if args["type"].lower() == "mlp":
            from agents.ppo import PPOAgent
            env.set_padded_observations(True)
            if args["kwargs"] is not None:
                agent = PPOAgent(env, args["config"], **args["kwargs"])
            else:
                agent = PPOAgent(env, args["config"])
            agent.train()
        
        elif args["type"].lower() == "transformer":
            from agents.ppo_transformer import PPO_Transformer_Agent
            if args["kwargs"] is not None:
                agent = PPO_Transformer_Agent(env, args["config"], **args["kwargs"])
            else:
                agent = PPO_Transformer_Agent(env, args["config"])
            agent.train()
        else: raise NotImplementedError()

    elif args["agent"].lower() == "ddpg":
        if args["type"].lower() == "mlp":
            from agents.ddpg import DDPGAgent
            env.set_padded_observations(True)
            if args["kwargs"] is not None:
                agent = DDPGAgent(env, args["config"], **args["kwargs"])
            else:
                agent = DDPGAgent(env, args["config"])
            agent.train()
        
        elif args["type"].lower() == "transformer":
            from agents.ddpg_transformer import DDPG_Transformer_Agent
            if args["kwargs"] is not None:
                agent = DDPG_Transformer_Agent(env, args["config"], **args["kwargs"])
            else:
                agent = DDPG_Transformer_Agent(env, args["config"])
            agent.train()
        else: raise NotImplementedError()

    elif args["agent"].lower() == "sac":
        if args["type"].lower() == "mlp":
            from agents.sac import SACAgent
            env.set_padded_observations(True)
            if args["kwargs"] is not None:
                agent = SACAgent(env, args["config"], **args["kwargs"])
            else:
                agent = SACAgent(env, args["config"])
            agent.train()
        
        elif args["type"].lower() == "transformer":
            from agents.sac_transformer import SAC_Transformer_Agent
            if args["kwargs"] is not None:
                agent = SAC_Transformer_Agent(env, args["config"], **args["kwargs"])
            else:
                agent = SAC_Transformer_Agent(env, args["config"])
            agent.train()
        else: raise NotImplementedError()

    elif args["agent"].lower() == "crowdnav":
        env = WorldFrameObservations(env)
        from agents.crowdnav import CrowdNavAgent
        if args["kwargs"] is not None:
            agent = CrowdNavAgent(env, args["config"], **args["kwargs"])
        else:
            agent = CrowdNavAgent(env, args["config"])
        agent.train()