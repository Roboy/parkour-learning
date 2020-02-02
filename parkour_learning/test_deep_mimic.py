import parkour_learning
import gym
import time
import sys

mocap_file_path = './motions/humanoid3d_jump.txt'
env = gym.make('HumanoidDeepMimicBulletEnv-v1', render=True)
while True:
    obs = env.reset()
    env.reset()
    done = False
    time.sleep(0.5)
    step =0

    while not done:
        step += 1
        action = env.action_space.sample()
        action *=0
        time.sleep(0.03)
        obs, reward, done, info = env.step(action)
        print('reward: ' +str(reward))
        # env.render()
    print('num steps:' + str(step))
