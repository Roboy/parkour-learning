import parkour_learning
import time
import gym
import matplotlib.pyplot as plt

# env = gym.make('HumanoidPrimitivePretraining-v0', render=True)
env = gym.make('TrackEnv-v0', render=True)


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
        time.sleep(0.015)
        start = time.time()
        obs, reward, done, info = env.step(action)
        print('complete step: ' + str(time.time() - start))
        # print(len(obs['goal']))
        # print('reward: ' +str(reward))
        # print('step: ' + str(step))
    print('num steps:' + str(step))
