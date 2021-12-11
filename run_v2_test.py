import gym
from ddpg import DDPG
import rocket
env = gym.make("RocketEnv-v2")

ddpg = DDPG(env)
ddpg.load_models()
obs = env.reset()

while True:
    action = ddpg.policy_inference(obs)
    obs, rewards, dones, info = env.step(action.cpu().data.numpy())
    env.render()