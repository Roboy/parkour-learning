import argparse
import torch
import time
import gym  # open ai gym
import numpy as np
import parkour_learning
import pybulletgym  # register PyBullet enviroments with open ai gym
import gym_parkour
from rlpyt.agents.qpg.sac_agent import SacAgent
from rlpyt.envs.gym import GymEnvWrapper
from rlpyt.utils.buffer import torchify_buffer, buffer_from_example, numpify_buffer
from mcp_model import PiMCPModel, QofMCPModel


def simulate_policy(path_to_params, env_id: str):
    snapshot = torch.load(path_to_params, map_location=torch.device('cpu'))
    agent_state_dict = snapshot['agent_state_dict']
    env = GymEnvWrapper(gym.make(env_id, render=False))
    agent_kwargs = dict(ModelCls=PiMCPModel, QModelCls=QofMCPModel)
    agent = SacAgent(**agent_kwargs)
    agent.initialize(env_spaces=env.spaces)
    agent.load_state_dict(agent_state_dict)
    agent.eval_mode(0)
    obs = env.reset()
    observation = buffer_from_example(obs, 1)

    while True:
        observation[0] = env.reset()
        action = buffer_from_example(env.action_space.null_value(), 1)
        reward = np.zeros(1, dtype="float32")
        obs_pyt, act_pyt, rew_pyt = torchify_buffer((observation, action, reward))
        done = False
        step = 0
        reward_sum = 0
        while not done:
            step += 1
            act_pyt, agent_info = agent.step(obs_pyt, act_pyt, rew_pyt)
            action = numpify_buffer(act_pyt)
            obs, reward, done, info = env.step(action[0])
            reward_sum += reward
            observation[0] = obs
            # action[0] = action
            rew_pyt[0] = reward
            time.sleep(0.02)
            # env.render(mode='human')
        print('return: ' + str(reward_sum) + '  num_steps: ' + str(step))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', help='path to params.pkl',
                        default='/home/alex/parkour-learning/data/params.pkl')
    parser.add_argument('--pretraining', default=False, action='store_true')
    args = parser.parse_args()
    env_id = 'HumanoidPrimitivePretraining-v0' if args.pretraining else 'TrackEnv-v0'
    simulate_policy(args.path, env_id)
