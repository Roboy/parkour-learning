import parkour_learning
import time
import gym

env = gym.make('HumanoidPrimitivePretraining-v0', render=True)


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
    print('num steps:' + str(step))
