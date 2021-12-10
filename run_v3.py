import gym
from ddpg import DDPG

env = gym.make("RocketEnv-v2")

ddpg = DDPG(env)
ddpg.load_models()
ddpg.train()
# 

# obs = env.reset()
# while True:
#     action = ddpg.policy_inference(obs)
#     obs, rewards, dones, info = env.step(action.cpu().data.numpy())
#     env.render()