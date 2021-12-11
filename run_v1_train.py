import gym
import rocket
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3 import TD3

import numpy as np
env = gym.make('RocketEnv-v2')

# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))

import matplotlib.pyplot as plt

from stable_baselines3.common.callbacks import BaseCallback

class SummaryWriterCallback(BaseCallback):

    def _on_training_start(self):
        self._log_freq = 1  # log every 1000 calls
        self.episode_rewards = []

    def _on_step(self) -> bool:
        try:
            self.episode_rewards.append(self.locals['episode_reward'][0])
        except:
            pass
        if self.n_calls % 500 == 0:
            try:
                self.episode_rewards.append(self.locals['episode_reward'][0])
                plt.plot(self.episode_rewards)
                plt.xlabel('Timestep')
                plt.ylabel('Reward')
                plt.savefig('reward.png')
                plt.clf()
            except:
                pass

model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1, tensorboard_log="./ddpg_log")
model.learn(total_timesteps=10000, log_interval=10, callback=SummaryWriterCallback())
model.save("ddpg")