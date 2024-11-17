import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from AI.tetris_env import TetrisEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

class DQN(nn.Module):
    """
    Deep Q-Network (DQN) model.
    This class defines a neural network model for a Deep Q-Network (DQN) used in reinforcement learning.
    The network consists of three fully connected layers with ReLU activations.
    Attributes:
        fc1 (nn.Linear): The first fully connected layer.
        fc2 (nn.Linear): The second fully connected layer.
        fc3 (nn.Linear): The third fully connected layer.
    Methods:
        forward(x):
            Defines the forward pass of the network.
            Args:
                x (torch.Tensor): The input tensor.
            Returns:
                torch.Tensor: The output tensor after passing through the network.
    """
    def __init__(self, input_dim, output_dim):
        """
        Initializes the DQN (Deep Q-Network) model.
        Args:
            input_dim (int): The dimension of the input features.
            output_dim (int): The dimension of the output actions.
        Attributes:
            fc1 (nn.Linear): The first fully connected layer.
            fc2 (nn.Linear): The second fully connected layer.
            fc3 (nn.Linear): The third fully connected layer.
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        """
        Perform a forward pass through the neural network.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        """
        Initialize the DQNAgent.
        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
        Attributes:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            memory (deque): Replay memory to store experiences.
            gamma (float): Discount factor for future rewards.
            epsilon (float): Exploration rate for epsilon-greedy policy.
            epsilon_decay (float): Decay rate for exploration probability.
            epsilon_min (float): Minimum exploration probability.
            learning_rate (float): Learning rate for the optimizer.
            batch_size (int): Batch size for training.
            model (DQN): The Q-network model.
            target_model (DQN): The target Q-network model.
            optimizer (torch.optim.Optimizer): Optimizer for the Q-network.
            loss_fn (torch.nn.modules.loss._Loss): Loss function for training the Q-network.
        """
        self.state_dim = state_dim          # Dimension of the state space
        self.action_dim = action_dim        # Dimension of the action space
        self.memory = deque(maxlen=10000)   # Replay memory
        self.gamma = 0.99                   # Discount factor, used in the calculation of the target Q-value
        self.epsilon = 1.0                  # Exploration rate for epsilon-greedy policy
        self.epsilon_decay = 0.995          # Decay rate for epsilon
        self.epsilon_min = 0.01             # Minimum exploration probability
        self.learning_rate = 0.001          # Learning rate for the optimizer
        self.batch_size = 64                # Batch size for training

        # Initialize models and move them to GPU/CPU
        self.model = DQN(state_dim, action_dim).to(device)                          # Q-network model
        self.target_model = DQN(state_dim, action_dim).to(device)                   # Target Q-network model
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate) # Optimizer
        self.loss_fn = nn.MSELoss()                                                 # Loss function
        
        self.last_action = None             # Track the last action taken

    def remember(self, state, action, reward, next_state, done):
        """
        Store the experience in the agent's memory.

        Args:
            state (array-like): The current state of the environment.
            action (int): The action taken by the agent.
            reward (float): The reward received after taking the action.
            next_state (array-like): The state of the environment after taking the action.
            done (bool): A flag indicating whether the episode has ended.
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Selects an action based on the current state using an epsilon-greedy policy.
        Ensures that the 'Hold' action (action == 4) is not chosen consecutively.

        Parameters:
            state (array-like): The current state of the environment.

        Returns:
            int: The selected action.
        """
        # If a random number is less than epsilon, choose a random action
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_dim)
            # Ensure the chosen action is not 'Hold' (4) if the last action was also 'Hold'
            while action == 4 and self.last_action == 4:
                action = random.randrange(self.action_dim)
        else:
            # Otherwise, choose the action with the highest Q-value
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():  # Disable gradient computation for improved performance
                q_values = self.model(state)  # Get the Q-values from the model
            action = np.argmax(q_values.cpu().numpy())  # Move back to CPU for numpy operations

            # Ensure the chosen action is not 'Hold' (4) if the last action was also 'Hold'
            if action == 4 and self.last_action == 4:
                # Choose the next best action (excluding 'Hold')
                q_values[0, 4] = -float('inf')  # Temporarily set 'Hold' Q-value to -inf
                action = np.argmax(q_values.cpu().numpy())  # Recompute the best action

        self.last_action = action  # Update the last action taken
        return action

    def replay(self):
        """
        Perform experience replay to train the DQN agent.
        This method samples a batch of experiences from the agent's memory, computes the target Q-values,
        and updates the model's weights by minimizing the loss between the predicted Q-values and the target Q-values.
        Returns:
            None
        """
        # If the memory is less than the batch size, do not perform replay
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)         # Sample a batch of experiences from the memory
        states, actions, rewards, next_states, dones = zip(*batch)  # Unzip the batch of experiences

        # Combine the lists of NumPy arrays into single NumPy arrays before converting to tensors
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(dones).to(device)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)    # Get the Q-values for the current states
        next_q_values = self.target_model(next_states).max(1)[0]                    # Get the Q-values for the next states
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))      # Calculate the target Q-values

        loss = self.loss_fn(q_values, target_q_values)  # Calculate the loss between the predicted Q-values and the target Q-values
        self.optimizer.zero_grad()                      # Reset the gradients
        loss.backward()                                 # Backpropagate the loss
        self.optimizer.step()                           # Update the weights

        # Decay the exploration rate after each replay, if it is above the minimum
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        """
        Updates the target model's weights to match the current model's weights.

        This method copies the weights from the current model to the target model.
        It is typically used in deep Q-learning to periodically update the target
        network to stabilize training.
        """
        # Copy the weights from the current model to the target model
        self.target_model.load_state_dict(self.model.state_dict())

def train_dqn(episodes):
    """
    Train a Deep Q-Network (DQN) agent to play Tetris.
    Args:
        episodes (int): Number of episodes to train the DQN agent.
    The function initializes the Tetris environment and the DQN agent, then
    runs the training loop for the specified number of episodes. In each
    episode, the agent interacts with the environment, collects experiences,
    and learns from them. The target model is updated at the end of each
    episode, and the total reward for the episode is printed.
    The environment is closed after training is complete.
    """
    env = TetrisEnv()                                                               # Initialize the Tetris environment
    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]     # Get the state dimension
    action_dim = env.action_space.n                                                 # Get the action dimension
    agent = DQNAgent(state_dim, action_dim)                                         # Initialize the DQN agent

    # For each episode
    for episode in range(episodes):
        state = env.reset()         # Reset the environment
        state = state.flatten()     # Flatten the state
        total_reward = 0            # Initialize the total reward
        done = False                

        # While the episode is not done
        while not done:
            action = agent.act(state)                               # Select an action, given the state
            next_state, reward, done, _ = env.step(action)          # Take a step in the environment          
            next_state = next_state.flatten()                       # Flatten the next state
            agent.remember(state, action, reward, next_state, done) # Store the experience in the agent's memory
            state = next_state                                      # Update the current state
            total_reward += reward                                  # Update the total reward
            
            agent.replay()                                          # Perform experience replay, training the agent
            
            # env.render()

        # At the end of the episode, update the target model and print the total reward
        agent.update_target_model()                                 
        print(f"\nEpisode {episode + 1}/{episodes}, Total Reward: {total_reward}")
        
        # Render the environment after each episode
        env.render()

    env.close()

if __name__ == "__main__":
    # Train the DQN agent for 1000 episodes
    train_dqn(1000)