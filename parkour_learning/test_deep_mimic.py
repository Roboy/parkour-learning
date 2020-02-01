import parkour_learning
import gym
import time
import sys
mocap_file_path = './motions/humanoid3d_jump.txt'
env = gym.make('HumanoidDeepMimicBulletEnv-v1', render=True)
while True:
    obs = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()
        time.sleep(0.01)
        obs, reward, done, info = env.step(action)
        # env.render()
