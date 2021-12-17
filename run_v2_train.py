import gym
from ddpg import DDPG
import rocket

env = gym.make("RocketEnv-v2")

ddpg = DDPG(env)
ddpg.load_models()
ddpg.train()