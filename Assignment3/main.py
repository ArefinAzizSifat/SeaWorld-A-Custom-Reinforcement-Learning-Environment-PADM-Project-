import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from env import ContinuousMazeEnv
from DQN_model import DQN
from utils import ReplayBuffer, plot_rewards
import matplotlib.pyplot as plt
import os


def plot_bar_chart(rewards, window_size=10):
    episodes = list(range(1, len(rewards) + 1))
    avg_rewards = [
        np.mean(rewards[max(0, i - window_size + 1):i + 1])
        for i in range(len(rewards))
    ]

    plt.figure(figsize=(12, 6))
    plt.bar(episodes, avg_rewards, width=1.0, color='skyblue')
    plt.title(f"Average Reward per Episode (Window Size = {window_size})")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.tight_layout()
    plt.savefig("reward_bar_chart.png")
    plt.close()


# Hyperparameters
EPISODES = 1000
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
BUFFER_SIZE = 50000
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 0.985  # multiplicative
TARGET_UPDATE_FREQ = 10

SAVE_PATH = "dqn_agent.pth"

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Environment setup
env = ContinuousMazeEnv()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Models
policy_net = DQN(state_dim, action_dim).to(device)
target_net = DQN(state_dim, action_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
buffer = ReplayBuffer(BUFFER_SIZE)

# Epsilon setup
epsilon = EPS_START

# Track success streak
consecutive_success = 0
success_required = 100
rewards_list = []

for episode in range(EPISODES):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        # env.render()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = policy_net(state_tensor)
                action = torch.argmax(q_values).item()

        next_state, reward, done, truncated, _ = env.step(action)
        buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        # Learn
        if len(buffer) >= BATCH_SIZE:
            batch = buffer.sample(BATCH_SIZE)
            states = torch.FloatTensor(batch[0]).to(device)
            actions = torch.LongTensor(batch[1]).unsqueeze(1).to(device)
            rewards = torch.FloatTensor(batch[2]).unsqueeze(1).to(device)
            next_states = torch.FloatTensor(batch[3]).to(device)
            dones = torch.FloatTensor(batch[4]).unsqueeze(1).to(device)

            # Compute target
            with torch.no_grad():
                next_q = target_net(next_states).max(1, keepdim=True)[0]
                target_q = rewards + (1 - dones) * GAMMA * next_q

            # Compute current
            current_q = policy_net(states).gather(1, actions)

            loss = nn.MSELoss()(current_q, target_q)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Decay epsilon
    if epsilon > EPS_END:
        epsilon *= EPS_DECAY
        epsilon = max(epsilon, EPS_END)

    rewards_list.append(total_reward)

    # Success check

    if reward > 0 and np.linalg.norm(env.agent_pos - env.goal_pos) <= env.goal_radius:
        consecutive_success += 1
    else:
        consecutive_success = 0

    print(f"Episode {episode+1}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}, Success Streak: {consecutive_success}")

    # Update target network
    if episode % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # Save if trained
    if consecutive_success >= success_required:
        print("Agent trained successfully!")
        torch.save(policy_net.state_dict(), SAVE_PATH)
        plot_rewards(rewards_list)
        plot_bar_chart(rewards_list)
        break

env.close()
