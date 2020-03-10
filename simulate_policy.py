import argparse
import torch
import time
import gym  # open ai gym
import numpy as np
import parkour_learning
import pybulletgym  # register PyBullet enviroments with open ai gym
import gym_parkour
from typing import Dict
from rlpyt.agents.qpg.sac_agent import SacAgent
# from mcp_sac_agent import MCPSacAgent
from rlpyt.envs.gym import GymEnvWrapper
from rlpyt.utils.buffer import torchify_buffer, buffer_from_example, numpify_buffer
from mcp_model import PiMCPModel, QofMCPModel, PPOMcpModel
from mcp_vision_model import PiMcpVisionModel, QofMcpVisionModel, PpoMcpVisionModel
from rlpyt.agents.pg.mujoco import MujocoLstmAgent, MujocoFfAgent
# from mcp_vision_model import PiMcpVisionModel, QofMcpVisionModel
from vision_models import PiVisionModel, QofMuVisionModel

def simulate_policy(env, agent):
    # snapshot = torch.load(path_to_params, map_location=torch.device('cpu'))
    # agent_state_dict = snapshot['agent_state_dict']
    # env = GymEnvWrapper(gym.make(env_id, render=True))
    # env = gym.make('HopperPyBulletEnv-v0')
    # env.render(mode='human')
    # env = GymEnvWrapper(env)
    # agent_kwargs = dict(ModelCls=PiMcpVisionModel, QModelCls=QofMcpVisionModel)
    # agent = SacAgent(**agent_kwargs)
    # agent = SacAgent(model_kwargs=dict(hidden_sizes=[512,256, 256]), q_model_kwargs=dict(hidden_sizes=[512, 256, 256]))
    # agent = MujocoFfAgent(ModelCls=PPOMcpModel)
    # agent.initialize(env_spaces=env.spaces)
    # agent.load_state_dict(agent_state_dict)
    # agent.eval_mode(0)
    obs = env.reset()
    observation = buffer_from_example(obs, 1)
    loop_time = 0.04
    while True:
        observation[0] = env.reset()
        action = buffer_from_example(env.action_space.null_value(), 1)
        reward = np.zeros(1, dtype="float32")
        obs_pyt, act_pyt, rew_pyt = torchify_buffer((observation, action, reward))
        done = False
        step = 0
        reward_sum = 0
        env.render()
        # time.sleep(5)
        while not done:
            loop_start = time.time()
            step += 1
            act_pyt, agent_info = agent.step(obs_pyt, act_pyt, rew_pyt)
            action = numpify_buffer(act_pyt)
            obs, reward, done, info = env.step(action[0])
            reward_sum += reward
            observation[0] = obs
            rew_pyt[0] = reward
            sleep_time = loop_time - (time.time() - loop_start)
            sleep_time = 0 if (sleep_time < 0) else sleep_time
            time.sleep(sleep_time)
            env.render(mode='human')
        print('return: ' + str(reward_sum) + '  num_steps: ' + str(step))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', help='path to params.pkl',
                        default='/home/alex/parkour-learning/data/params.pkl')
    parser.add_argument('--env', default='HumanoidPrimitivePretraining-v0',
                        choices=['HumanoidPrimitivePretraining-v0', 'TrackEnv-v0'])
    parser.add_argument('--algo', default='ppo', choices=['sac', 'ppo'])
    args = parser.parse_args()

    snapshot = torch.load(args.path, map_location=torch.device('cpu'))
    agent_state_dict = snapshot['agent_state_dict']
    env = GymEnvWrapper(gym.make(args.env, render=True))
    if args.algo == 'ppo':
        if args.env == 'TrackEnv-v0':
            agent = MujocoFfAgent(ModelCls=PpoMcpVisionModel)
        else:
            agent = MujocoFfAgent(ModelCls=PPOMcpModel)
    else:
        if args.env == 'TrackEnv-v0':
            agent = SacAgent(ModelCls=PiVisionModel, QModelCls=QofMuVisionModel)
        else:
            agent = SacAgent(ModelCls=PiMCPModel, QModelCls=QofMCPModel)

    agent.initialize(env_spaces=env.spaces)
    agent.load_state_dict(agent_state_dict)
    agent.eval_mode(0)
    simulate_policy(env, agent)
