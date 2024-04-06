import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Define the neural network model
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=-1)

class Plotting:
    def __init__(self, episode_numbers, episode_rewards):
        self.episode_numbers = episode_numbers
        self.episode_rewards = episode_rewards

    def plot_training(self):
        plt.plot(self.episode_numbers, self.episode_rewards)
        plt.xlabel('Episode Number')
        plt.ylabel('Total Reward')
        plt.title('Training Progress')
        plt.grid(True)
        plt.show()

class Training:
    def __init__(self, policy_network, optimizer, env, gamma, max_steps, num_episodes):
        self.policy_network = policy_network
        self.optimizer = optimizer
        self.env = env
        self.gamma = gamma
        self.max_steps = max_steps
        self.num_episodes = num_episodes

    # Function to select an action based on the policy network output
    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy_network(state)
        action = np.random.choice(self.env.action_space.n, p=probs.detach().numpy().flatten())
        return action

    # Function to compute the discounted rewards
    def compute_discounted_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards
    
    # Main loop
    def train(self):
        for episode in range(self.num_episodes):
            state = self.env.reset()
            state = np.array(state[0])
            episode_rewards = []
            episode_states = []
            episode_actions = []
            
            for step in range(self.max_steps):
                action = self.select_action(state)
                next_state, reward, done, _, _= self.env.step(action)
                
                episode_rewards.append(reward)
                episode_states.append(state)
                episode_actions.append(action)
                
                state = next_state
                
                if done:
                    break
            
            discounted_rewards = self.compute_discounted_rewards(episode_rewards)
            discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-9)
            
            # Update policy network
            self.optimizer.zero_grad()
            for i in range(len(episode_rewards)):
                state = episode_states[i]
                action = episode_actions[i]
                reward = discounted_rewards[i]
                
                state_tensor = torch.from_numpy(state).float().unsqueeze(0)
                action_tensor = torch.tensor([action])
                reward_tensor = torch.tensor([reward])
                
                probs = self.policy_network(state_tensor)
                action_probs = probs.gather(1, action_tensor.unsqueeze(1))
                loss = -torch.log(action_probs) * reward_tensor
                loss.backward()
            
            self.optimizer.step()
            
            # store episode rewards and episode number
            episode_rewards_list.append(sum(episode_rewards))
            episode_numbers_list.append(episode + 1)
            
            # Print episode information
            total_reward = sum(episode_rewards)
            print(f"Episode {episode + 1}, Total Reward: {total_reward}")

# Hyperparameters
learning_rate = 0.01
gamma = 0.99
hidden_size = 128
num_episodes = 1000
max_steps = 1000

# Initialize environment and policy network
env = gym.make('CartPole-v1')
input_size = env.observation_space.shape[0]
output_size = env.action_space.n
policy_network = PolicyNetwork(input_size, hidden_size, output_size)
optimizer = optim.Adam(policy_network.parameters(), lr=learning_rate)

episode_rewards_list = []
episode_numbers_list = []

training = Training(policy_network, optimizer, env, gamma, max_steps, num_episodes)
training.train()

# Close environment
env.close()

plotting = Plotting(episode_numbers_list, episode_rewards_list)
plotting.plot_training()