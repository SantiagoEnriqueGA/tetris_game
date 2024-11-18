from bot import TetrisBot

import math
import random
import time
import multiprocessing

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

random.seed(252)

# Define constants for the game
GAME_WIDTH = 15     # Width of the game board
GAME_HEIGHT = 20     # Height of the game board
BLINK_TIMES = 3
BLINK_DELAY = 0
BOT_SPEED = 0

# Define color pairs
COLOR_CYAN = 2
COLOR_BLUE = 3
COLOR_MAGENTA = 4
COLOR_YELLOW = 5
COLOR_GREEN = 6
COLOR_RED = 7
COLOR_WHITE = 8

class TetrisBoard:
    def __init__(self, width, height):
        """
        Initializes the game board with the given width and height.
        Args:
            width (int): The width of the game board.
            height (int): The height of the game board.
        Attributes:
            width (int): The width of the game board.
            height (int): The height of the game board.
            score (int): The current score of the game.
            board (list of list of int): The game board represented as a 2D list, where the walls are marked with 2.
            cell_type (list of list of str): The game board represented as a 2D list, where the walls are marked with 'X'.
        """
        self.width = width      # Width of the board
        self.height = height    # Height of the board
        self.score = 0          # Score of the game
        
        # Create the board and the cell type
        self.board = [[2 if x == 0 or x == self.width - 1 else 0 for x in range(self.width)] for y in range(self.height)]
        self.cell_type = [['X' if x == 0 or x == self.width - 1 else 0 for x in range(self.width)] for y in range(self.height)]

    def is_valid_position(self, block, x, y):
        """Check if the block can be placed at the given position"""
        for i in range(len(block.shape)):           # For each row
            for j in range(len(block.shape[i])):    # For each column
                if block.shape[i][j] == 1:              # If the block has a cell, check if the cell is within the board and the cell is empty
                    if i + y >= self.height or j + x < 0 or j + x >= self.width or self.board[i + y][j + x] >= 1:
                        return False
        return True

    def add_block(self, block, x, y):
        """Add the block to the board at the given position"""
        for i in range(len(block.shape)):                   # For each row
            for j in range(len(block.shape[i])):            # For each column
                if block.shape[i][j] == 1:                      # If the block has a cell, add the cell
                    self.board[i + y][j + x] = 1                # Add the cell to the board
                    self.cell_type[i + y][j + x] = block.color  # Add the cell type to the board

    def remove_full_rows(self):
        """Remove the full rows from the board and update the score"""
        full_rows = []
        for i in range(self.height):    # Check each row
            if all(self.board[i]):      # If the row is full
                full_rows.append(i)     # Add the row to the full rows list
        
        if not full_rows: return        # If no full rows, return
        
        for i in full_rows:
            # Remove the full row
            self.board.pop(i)
            self.board.insert(0, [2 if x == 0 or x == self.width - 1 else 0 for x in range(self.width)])
            
            self.cell_type.pop(i)
            self.cell_type.insert(0, ['X' if x == 0 or x == self.width - 1 else 0 for x in range(self.width)])
            
        # Update the score
        # The score is the square of the number of full rows removed
        # This is to encourage the player to remove more rows at once
        self.score += int(math.pow(len(full_rows), 2))
            
    def is_game_over(self):
        """Check if the game is over, i.e. the top row has any blocks"""
        return any(self.board[0][1:-1])

    def get_score(self):
        """Return the score of the game"""
        return self.score

    def get_board(self):
        """Return the board"""
        return self.board
    
    def get_board_type(self):
        """Return the board"""
        return self.cell_type
    
    
class TetrisBlock:
    def __init__(self, shape, color):
        """A class to represent a Tetris block."""
        self.shape = shape
        self.color = color

class Blocks:
    def __init__(self):               
        self.blocks = {
            'I': TetrisBlock([[1, 1, 1, 1]], COLOR_CYAN),
            'J': TetrisBlock([[1, 0, 0], [1, 1, 1]], COLOR_BLUE),
            'L': TetrisBlock([[0, 0, 1], [1, 1, 1]], COLOR_MAGENTA),
            'O': TetrisBlock([[1, 1], [1, 1]], COLOR_YELLOW),
            'S': TetrisBlock([[0, 1, 1], [1, 1, 0]], COLOR_GREEN),
            'T': TetrisBlock([[0, 1, 0], [1, 1, 1]], COLOR_RED),
            'Z': TetrisBlock([[1, 1, 0], [0, 1, 1]], COLOR_WHITE)
        }

    def get_block(self, block_type):
        """Return the block of the given type: I, J, L, O, S, T, Z"""
        return self.blocks.get(block_type)



class TetrisStateEvaluator(nn.Module):
    def __init__(self):
        super(TetrisStateEvaluator, self).__init__()
        
        # Input features: current board state, current piece, held piece, next piece
        self.feature_network = nn.Sequential(
            nn.Linear(300 + 7 + 7 + 7, 256),  # 300 for board (15x20), 7 for each piece type
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 8)  # Output the 8 weights we need
        )
        
        # Initialize weights using Xavier initialization
        for layer in self.feature_network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, board_state, current_piece, held_piece, next_piece):
        # Flatten and concatenate all inputs
        board_flat = board_state.view(-1)
        x = torch.cat([board_flat, current_piece, held_piece, next_piece], dim=0)
        return self.feature_network(x)

class ReplayMemory:
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class TetrisNNOptimizer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.evaluator = TetrisStateEvaluator().to(self.device)
        self.optimizer = optim.Adam(self.evaluator.parameters(), lr=0.001)
        self.memory = ReplayMemory()
        self.batch_size = 64
        self.gamma = 0.99  # Discount factor
        
    def encode_piece(self, piece_type):
        # One-hot encode the piece type
        piece_types = ['I', 'J', 'L', 'O', 'S', 'T', 'Z']
        encoding = torch.zeros(7)
        if piece_type is not None:
            encoding[piece_types.index(piece_type)] = 1
        return encoding
    
    def get_state_tensor(self, board, current_piece, held_piece, next_piece):
        board_tensor = torch.tensor(board, dtype=torch.float32)
        current_piece_tensor = self.encode_piece(current_piece)
        held_piece_tensor = self.encode_piece(held_piece)
        next_piece_tensor = self.encode_piece(next_piece)
        
        return (board_tensor, current_piece_tensor, held_piece_tensor, next_piece_tensor)
    
    def get_weights(self, state):
        board_tensor, current_piece_tensor, held_piece_tensor, next_piece_tensor = state
        with torch.no_grad():
            weights = self.evaluator(
                board_tensor.to(self.device),
                current_piece_tensor.to(self.device),
                held_piece_tensor.to(self.device),
                next_piece_tensor.to(self.device)
            )
        return {
            'completed_lines': float(weights[0]),
            'holes': float(weights[1]),
            'bumpiness': float(weights[2]),
            'height': float(weights[3]),
            'deep_wells': float(weights[4]),
            'clear_path': float(weights[5]),
            'edge_touch': float(weights[6]),
            'blockade': float(weights[7])
        }
    
    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Sample from memory
        transitions = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))
        
        # Compute the loss
        state_batch = batch[0]
        reward_batch = torch.tensor(batch[2], dtype=torch.float32).to(self.device)
        next_state_batch = batch[3]
        
        # Get current weights
        current_weights = []
        for state in state_batch:
            board_t, curr_p_t, held_p_t, next_p_t = state
            weights = self.evaluator(
                board_t.to(self.device),
                curr_p_t.to(self.device),
                held_p_t.to(self.device),
                next_p_t.to(self.device)
            )
            current_weights.append(weights)
        current_weights = torch.stack(current_weights)
        
        # Get next state weights
        next_weights = []
        for state in next_state_batch:
            if state is None:  # Terminal state
                weights = torch.zeros(8).to(self.device)
            else:
                board_t, curr_p_t, held_p_t, next_p_t = state
                weights = self.evaluator(
                    board_t.to(self.device),
                    curr_p_t.to(self.device),
                    held_p_t.to(self.device),
                    next_p_t.to(self.device)
                )
            next_weights.append(weights)
        next_weights = torch.stack(next_weights)
        
        # Compute expected Q values
        expected_weights = reward_batch.unsqueeze(1) + self.gamma * next_weights
        
        # Compute loss
        loss = nn.MSELoss()(current_weights, expected_weights)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

def optimize_tetris_bot():
    optimizer = TetrisNNOptimizer()
    num_episodes = 1000
    
    for episode in range(num_episodes):
        board = TetrisBoard(GAME_WIDTH, GAME_HEIGHT)
        blocks = Blocks()
        
        current_piece = random.choice(list(blocks.blocks.keys()))
        held_piece = None
        next_piece = random.choice(list(blocks.blocks.keys()))
        
        state = optimizer.get_state_tensor(
            board.get_board(),
            current_piece,
            held_piece,
            next_piece
        )
        
        # Make move and get new state
        x = board.width // 2 - len(blocks.get_block(current_piece).shape[0]) // 2
        y = 0
        
        while not board.is_game_over():     
            episode_reward = 0
            
            # Get weights from neural network
            weights = optimizer.get_weights(state)
            
            # Use weights to make a move
            TetrisBot.WEIGHTS = weights
            tetris_bot = TetrisBot()
            
            # Make move and get new state
            x = board.width // 2 - len(blocks.get_block(current_piece).shape[0]) // 2
            y = 0
            
            x, y, continue_game, current_piece, held_piece = tetris_bot.get_next_move(
                board, blocks.get_block(current_piece), x, y, 
                blocks.get_block(held_piece) if held_piece else None
            )
            
            # Move the block down
            while board.is_valid_position(current_piece, x, y + 1):
                y += 1
            
            # Add block and get reward
            old_score = board.get_score()
            board.add_block(current_piece, x, y)
            board.remove_full_rows()
            
            score, metrics = tetris_bot._evaluate_position(board, current_piece, x, y)
            
            metric_weights = {
                'completed_lines': 100.0,    # Reward for completing lines
                'holes': -20.0,              # Penalty for creating holes
                'bumpiness': -2.0,          # Penalty for uneven surface
                'height': -1.5,             # Penalty for high stacks
                'deep_wells': -5.0,         # Penalty for deep wells
                'clear_path': 10.0,         # Reward for keeping paths to holes
                'edge_touch': 2.0,          # Reward for touching edges
                'blockade': -15.0,          # Penalty for blocking holes
            }
            reward = sum(metric_weights[metric] * value for metric, value in metrics.items())

            
            # Get next state
            future_piece = random.choice(list(blocks.blocks.keys()))
            next_state = None if board.is_game_over() else optimizer.get_state_tensor(
                board.get_board(),
                next_piece,
                held_piece,
                random.choice(list(blocks.blocks.keys()))
            )
            
            # Store transition in memory
            optimizer.memory.push(state, weights, reward, next_state)
            
            # Train the network
            loss = optimizer.train_step()
            
            # Update state
            state = next_state
            current_piece = next_piece
            episode_reward += reward
            
        print(f"Episode {episode + 1}/{num_episodes}, Score: {board.get_score()}, Episode Reward: {episode_reward}")
        
    return optimizer

if __name__ == '__main__':
    optimizer = optimize_tetris_bot()
    
    # Save the trained model
    torch.save(optimizer.evaluator.state_dict(), 'tetris_evaluator.pth')