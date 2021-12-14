import gym
from ddpg import DDPG
import rocket
env = gym.make("RocketEnv-v2")

ddpg = DDPG(env)
ddpg.load_models()


total_reward = 0
episodes = 10
for i in range(episodes):
    state = env.reset()
    while True:
        action = ddpg.policy_inference(state)
        state, reward, done, _ = env.step(action.cpu().data.numpy())
        total_reward += reward
        env.render()
        if done:
            break
        

print(f'Avg: {total_reward / episodes}')