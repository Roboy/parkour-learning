import argparse
import torch
import time
import gym  # open ai gym
import numpy as np
import parkour_learning
import pybulletgym  # register PyBullet enviroments with open ai gym
import gym_parkour
from rlpyt.agents.qpg.sac_agent import SacAgent
# from mcp_sac_agent import MCPSacAgent
from rlpyt.envs.gym import GymEnvWrapper
from rlpyt.utils.buffer import torchify_buffer, buffer_from_example, numpify_buffer
# from mcp_model import PiMCPModel, QofMCPModel, PPOMcpModel
from mcp_vision_model import PPOMcpModel
from rlpyt.agents.pg.mujoco import MujocoLstmAgent, MujocoFfAgent

# from mcp_vision_model import PiMCPModel, QofMCPModel

def simulate_policy(path_to_params, env_id: str):
    snapshot = torch.load(path_to_params, map_location=torch.device('cpu'))
    agent_state_dict = snapshot['agent_state_dict']
    env = GymEnvWrapper(gym.make(env_id, render=True))
    # env = gym.make('HopperPyBulletEnv-v0')
    # env.render(mode='human')
    # env = GymEnvWrapper(env)
    # agent_kwargs = dict(ModelCls=PiMCPModel, QModelCls=QofMCPModel)
    # agent = SacAgent(**agent_kwargs)
    # agent = SacAgent(model_kwargs=dict(hidden_sizes=[512,256, 256]), q_model_kwargs=dict(hidden_sizes=[512, 256, 256]))
    agent = MujocoFfAgent(ModelCls=PPOMcpModel)
    agent.initialize(env_spaces=env.spaces)
    agent.load_state_dict(agent_state_dict)
    agent.eval_mode(0)
    obs = env.reset()
    observation = buffer_from_example(obs, 1)
    loop_time = 0.03
    while True:
        observation[0] = env.reset()
        action = buffer_from_example(env.action_space.null_value(), 1)
        reward = np.zeros(1, dtype="float32")
        obs_pyt, act_pyt, rew_pyt = torchify_buffer((observation, action, reward))
        act_pyt, agent_info = agent.step(obs_pyt, act_pyt, rew_pyt)
        action = numpify_buffer(act_pyt)
        start = time.time()
        obs, reward, done, info = env.step(action[0])
        done = False
        step = 0
        reward_sum = 0
        env.render()
        # time.sleep(5)
        while not done:
            loop_start = time.time()
            step += 1
            print(step)
            act_pyt, agent_info = agent.step(obs_pyt, act_pyt, rew_pyt)
            action = numpify_buffer(act_pyt)
            start = time.time()
            obs, reward, done, info = env.step(action[0])
            reward_sum += reward
            observation[0] = obs
            # action[0] = action
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
    parser.add_argument('--pretraining', default=False, action='store_true')
    args = parser.parse_args()
    env_id = 'HumanoidPrimitivePretraining-v0' if args.pretraining else 'TrackEnv-v0'
    simulate_policy(args.path, env_id)
