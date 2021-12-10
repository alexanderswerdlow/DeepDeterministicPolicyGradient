import gym
import rocket
from stable_baselines3 import TD3

import numpy as np
env = gym.make('RocketEnv-v2')
model = TD3.load("ddpg")
obs = env.reset()

while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()