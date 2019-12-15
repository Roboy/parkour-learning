import argparse
import torch
import time
import gym  # open ai gym
import pybulletgym  # register PyBullet enviroments with open ai gym
import gym_parkour
from rlpyt.agents.qpg.sac_agent import SacAgent
from rlpyt.envs.gym import GymEnvWrapper


def simulate_policy(path_to_params):
    snapshot = torch.load(path_to_params, map_location=torch.device('cpu'))
    agent_state_dict = snapshot['agent_state_dict']
    optimizer_state_dict = snapshot['optimizer_state_dict']

    agent = SacAgent()
    env = gym.make('ParkourChallenge-v0', render=True,)
    # env = gym.make('HopperPyBulletEnv-v0')
    env.render(mode='human')
    wrapped_env = GymEnvWrapper(env)
    agent.initialize(env_spaces=wrapped_env.spaces)
    agent.load_state_dict(agent_state_dict)

    while True:
        obs = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            obs = torch.from_numpy(obs)
            action = agent.pi(obs, None, None)[0].detach().numpy()
            obs, reward, done, info = env.step(action)
            time.sleep(0.01)
            env.render(mode='human')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', help='path to params.pkl',
                        default='/home/alex/parkour-learning/data/params.pkl')
    args = parser.parse_args()
    simulate_policy(args.path)
