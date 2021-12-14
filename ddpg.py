import torch.nn as nn
import torch
from torch.optim import Adam
from collections import deque
import random
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

# Taken from PyTorch Docs
# Decreases exploration over time
class OUActionNoise:
    def __init__(self, mean, std_deviation=0.2, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

# Actor approximates some policy Pi(state)
class Actor(nn.Module):
    def __init__(self, n_states, n_actions, high_action_bounds):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(n_states, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, n_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.high_action_bounds = high_action_bounds

    def forward(self, state):
        output = self.fc1(state)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.fc3(output)
        output = self.tanh(output)
        return self.high_action_bounds * output


# Critic approximates some Q-function Q(state, action)
class Critic(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(n_states, 400)
        self.fc2 = nn.Linear(400 + n_actions, 300)
        self.fc3 = nn.Linear(300, 1)
        self.relu = nn.ReLU()

    def forward(self, state, action):
        output = self.fc1(state)
        output = self.relu(output)
        output = self.fc2(torch.cat([output, action], 1))
        output = self.relu(output)
        output = self.fc3(output)
        return output


class DDPG(object):
    def __init__(self, env):
        self.env = env
        self.n_states = self.env.observation_space.shape[-1]
        self.n_actions = self.env.action_space.shape[-1]
        self.low_action_bounds = torch.from_numpy(self.env.action_space.low).cuda()
        self.high_action_bounds = torch.from_numpy(self.env.action_space.high).cuda()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)

        self.actor = Actor(self.n_states, self.n_actions, self.high_action_bounds).to(self.device)
        self.critic = Critic(self.n_states, self.n_actions).to(self.device)

        self.target_actor = deepcopy(self.actor).to(self.device)
        self.target_critic = deepcopy(self.critic).to(self.device)

        self.actor_optim = Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optim = Adam(self.critic.parameters(), lr=3e-4)
        self.critic_loss = nn.MSELoss()

        self.random_noise = OUActionNoise(mean=np.zeros(self.n_actions), std_deviation=0.5)
        
        self.tau = 0.995
        self.gamma = 0.99
        self.replay_buffer = deque(maxlen=1000000)
        self.uniform_action_transitions = 10000
        self.bypass_network_update_transitions = 1000
        self.max_steps = 2e6
        self.cur_steps = 0
        self.episode_rewards = []

    def save_models(self):
        torch.save(self.critic.state_dict(), 'checkpoints/critic')
        torch.save(self.target_critic.state_dict(), 'checkpoints/target_critic')
        torch.save(self.actor.state_dict(), 'checkpoints/actor')
        torch.save(self.target_actor.state_dict(), 'checkpoints/target_actor')
        torch.save((self.replay_buffer, self.cur_steps, self.random_noise, self.episode_rewards), 'checkpoints/replay_buffer')

    def load_models(self):
        try:
            self.critic.load_state_dict(torch.load('checkpoints/critic'), self.device)
            self.target_critic.load_state_dict(torch.load('checkpoints/target_critic'), self.device)
            self.actor.load_state_dict(torch.load('checkpoints/actor'), self.device)
            self.target_actor.load_state_dict(torch.load('checkpoints/target_actor'), self.device)
            self.replay_buffer, self.cur_steps, self.random_noise, self.episode_rewards = torch.load('checkpoints/replay_buffer')
        except:
            pass

    def policy_inference(self, state):
        return self.actor(torch.from_numpy(state).float().to(self.device))

    def sample_batch(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.replay_buffer, min(batch_size, len(self.replay_buffer))))
        return torch.from_numpy(np.stack(states)).to(self.device).float(), torch.from_numpy(np.stack(actions)).to(self.device).float(), torch.from_numpy(np.stack(rewards).astype(np.float32)).to(self.device), torch.from_numpy(np.stack(next_states)).to(self.device), torch.from_numpy(np.stack(dones)).to(self.device)

    def sample_action(self, state):
        if self.cur_steps >= self.uniform_action_transitions:
            state = self.actor(torch.from_numpy(state).float().to(self.device)) + torch.from_numpy(np.array(self.random_noise())).to(self.device)
        else:
            state = torch.distributions.uniform.Uniform(self.low_action_bounds, self.high_action_bounds).sample().to(self.device)

        return torch.clamp(state, min=self.low_action_bounds, max=self.high_action_bounds).cpu()

    def execute_action(self, state, action):
        next_state, reward, done, _ = self.env.step(action.data)
        self.replay_buffer.append((state, action.data.numpy(), reward, next_state.astype(np.float32), done))
        return next_state, reward, done

    def update_target(self, target, current):
        for t, c in zip(target.parameters(), current.parameters()):
            t.data.copy_(t.data * (self.tau) + c.data * (1 - self.tau))

    def train(self):
        episode_number = 0
        while self.cur_steps < self.max_steps:

            # Start new episode
            state = self.env.reset().astype(np.float32)
            episode_reward = 0
            episode_steps = 0

            while True:
                # Draw Action from Actor and add noise
                action = self.sample_action(state)

                # Execute Action, Observe Next State, Reward, and Done, and store in replay buffer
                state, reward, done = self.execute_action(state, action)
                episode_reward += reward
                episode_steps += 1
                self.cur_steps += 1

                if episode_steps > 500:
                    done = True

                if self.cur_steps >= self.bypass_network_update_transitions:
                    # Sample N transitions (minibatch) from replay buffer
                    states, actions, rewards, next_states, dones = self.sample_batch(256)

                    # Compute Q_target, Q_predicted
                    Q_target = rewards + self.gamma * dones * torch.squeeze(self.target_critic(next_states, self.target_actor(next_states)))
                    Q_predicted = torch.squeeze(self.critic(states, actions), 1)

                    # Update Critic by SGD w/MSE
                    self.critic_optim.zero_grad()
                    critic_loss = self.critic_loss(Q_target, Q_predicted)
                    critic_loss.backward()
                    self.critic_optim.step()

                    self.actor_optim.zero_grad()
                    actor_loss = -torch.mean(self.critic(states, self.actor(states)))
                    actor_loss.backward()
                    self.actor_optim.step()

                    # Update Target Actor/Critic
                    self.update_target(self.target_actor, self.actor)
                    self.update_target(self.target_critic, self.critic)

                if done:
                    self.episode_rewards.append(episode_reward)
                    print(f'Episode {episode_number} had reward: {episode_reward} at {self.cur_steps} steps with replay len: {len(self.replay_buffer)}')

                    if episode_number % 10 == 0:
                        self.save_models()
                        plt.plot(self.episode_rewards)
                        plt.xlabel('Episode')
                        plt.ylabel('Episode Reward')
                        plt.savefig('reward.png')
                        plt.clf()

                    episode_number += 1
                    break
