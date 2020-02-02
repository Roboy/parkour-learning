import argparse
import torch
import time
import gym  # open ai gym
import numpy as np
import parkour_learning
import pybulletgym  # register PyBullet enviroments with open ai gym
import gym_parkour
from rlpyt.agents.qpg.sac_agent import SacAgent
from rlpyt.agents.pg.mujoco import MujocoLstmAgent, MujocoFfAgent
from rlpyt.envs.gym import GymEnvWrapper
from vision_models import PiVisionModel, QofMuVisionModel
from rlpyt.utils.buffer import torchify_buffer, buffer_from_example, numpify_buffer
from rlpyt.agents.base import AgentInputs


def simulate_policy(path_to_params, vision=False):
    snapshot = torch.load(path_to_params, map_location=torch.device('cpu'))
    agent_state_dict = snapshot['agent_state_dict']
    optimizer_state_dict = snapshot['optimizer_state_dict']

    agent = SacAgent()
    # agent = SacAgent(ModelCls=PiVisionModel, QModelCls=QofMuVisionModel)
    # agent = MujocoLstmAgent()
    # agent = MujocoFfAgent()
    # env = GymEnvWrapper(gym.make('ParkourChallenge-v0', render=True, vision=vision))
    env = GymEnvWrapper(gym.make('HumanoidDeepMimicBulletEnv-v1', render=True))
    # env = gym.make('HopperPyBulletEnv-v0')
    # env.render(mode='human')
    # wrapped_env = GymEnvWrapper(env)
    agent_kwargs = dict(model_kwargs=dict(hidden_sizes=[1024, 512]), q_model_kwargs=dict(hidden_sizes=[1024, 512]))
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
        prev_action = env.action_space.sample() * 0
        prev_reward = 0
        step = 0
        reward_sum = 0
        while not done:
            # action = env.action_space.sample()
            # if type(obs) == dict:
            #     for key, value in obs.items():
            #         obs[key] = torch.from_numpy(value)
            # else:
            #     obs = torch.from_numpy(obs)
            # agent_inputs = torchify_buffer(AgentInputs(obs, prev_action, prev_reward))
            # action = agent.pi(obs, None, None)[0].detach().numpy()
            # action = agent.step(*agent_inputs).action.numpy()
            step += 1
            act_pyt, agent_info = agent.step(obs_pyt, act_pyt, rew_pyt)
            action = numpify_buffer(act_pyt)
            start = time.time()
            obs, reward, done, info = env.step(action[0])
            reward_sum += reward
            # print(info)
            observation[0] = obs
            time.sleep(0.08)
            # env.render(mode='human')
        print('return: ' + str(reward_sum) + '  num_steps: ' + str(step))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', help='path to params.pkl',
                        default='/home/alex/parkour-learning/data/params.pkl')
    parser.add_argument('--vision', dest='vision', action='store_true',
                        help='if vision, observations will contain camera images')
    parser.add_argument('--no_vision', dest='vision', action='store_false',
                        help='if no_vision, observations will only contain joint info')
    args = parser.parse_args()
    simulate_policy(args.path, args.vision)
