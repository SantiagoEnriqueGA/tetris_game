import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

from game import TetrisBoard, TetrisBlock, Blocks, print_board

GAME_WIDTH = 15     # Width of the game board
GAME_HEIGHT = 20     # Height of the game board

# Define color pairs
COLOR_CYAN = 2
COLOR_BLUE = 3
COLOR_MAGENTA = 4
COLOR_YELLOW = 5
COLOR_GREEN = 6
COLOR_RED = 7
COLOR_WHITE = 8

class TetrisEnv(gym.Env):
    def __init__(self):
        """
        Initializes the Tetris environment.
        This method sets up the initial state of the Tetris game, including the game board, 
        the current block, the next block, and the held block. It also defines the action 
        space and observation space for the environment.
        Attributes:
            board (TetrisBoard): The game board for Tetris.
            blocks (Blocks): The collection of possible Tetris blocks.
            current_block (Block): The current block in play.
            next_block (Block): The next block to be played.
            held_block (Block or None): The block that is currently held, if any.
            x (int): The x-coordinate of the current block's position.
            y (int): The y-coordinate of the current block's position.
            action_space (spaces.Discrete): The discrete action space for the environment.
            observation_space (spaces.Box): The observation space representing the game board.
        """
        # Initialize the game board, blocks, and current block
        super(TetrisEnv, self).__init__()
        self.board = TetrisBoard(GAME_WIDTH, GAME_HEIGHT)   
        self.blocks = Blocks(curse=False)
        
        # Set the current block, next block, and held block
        self.current_block = self.blocks.get_block(random.choice(list(self.blocks.blocks.keys())))
        self.next_block = self.blocks.get_block(random.choice(list(self.blocks.blocks.keys())))
        self.held_block = None
        
        # Set the starting position for the current block
        self.x = self.board.width // 2 - len(self.current_block.shape[0]) // 2
        self.y = 0

        # Define the action and observation spaces
        self.action_space = spaces.Discrete(6)  # 0: left, 1: right, 2: down, 3: rotate, 4: hold
        self.observation_space = spaces.Box(low=0, high=1, shape=(GAME_HEIGHT, GAME_WIDTH), dtype=np.float32)

    def reset(self):
        """
        Resets the Tetris game environment to its initial state.

        This method initializes a new Tetris board, selects the current and next blocks randomly,
        resets the held block, and sets the starting position for the current block.
        
        Returns:
            observation (any): The initial observation of the game state after reset.
        """
        # Initialize the game board, blocks, and current block
        self.board = TetrisBoard(GAME_WIDTH, GAME_HEIGHT)
        self.current_block = self.blocks.get_block(random.choice(list(self.blocks.blocks.keys())))
        self.next_block = self.blocks.get_block(random.choice(list(self.blocks.blocks.keys())))
        self.held_block = None
        
        # Set the starting position for the current block
        self.x = self.board.width // 2 - len(self.current_block.shape[0]) // 2
        self.y = 0
        
        # Return the initial observation as a NumPy array
        return self._get_observation()

    def step(self, action):
        """
        Execute one step in the Tetris environment based on the given action.
        Enhanced reward function accounts for cleared rows, holes, and board height.
        """
        reward = 0
        done = False

        # Execute the action
        if action == 0:  # Move left
            if self.board.is_valid_position(self.current_block, self.x - 1, self.y):
                self.x -= 1
        elif action == 1:  # Move right
            if self.board.is_valid_position(self.current_block, self.x + 1, self.y):
                self.x += 1
        elif action == 2:  # Move down
            if self.board.is_valid_position(self.current_block, self.x, self.y + 1):
                self.y += 1
        elif action == 3:  # Rotate
            new_shape = list(zip(*self.current_block.shape[::-1]))
            if self.board.is_valid_position(TetrisBlock(new_shape, self.current_block.color), self.x, self.y):
                self.current_block.shape = new_shape
        elif action == 4:  # Drop
            while self.board.is_valid_position(self.current_block, self.x, self.y + 1):
                self.y += 1
        elif action == 5:  # Hold
            if self.held_block is None:
                self.held_block = self.current_block
                self.current_block = self.next_block
                self.next_block = self.blocks.get_block(random.choice(list(self.blocks.blocks.keys())))
            else:
                if self.board.is_valid_position(self.held_block, self.x, self.y):
                    self.held_block, self.current_block = self.current_block, self.held_block

        # If block cannot move down further, place it on the board
        if not self.board.is_valid_position(self.current_block, self.x, self.y + 1):
            # Calculate pre-placement metrics
            pre_height = self.board.get_aggregate_height()
            pre_holes = self.board.get_holes()

            # Add the block to the board and clear rows
            self.board.add_block(self.current_block, self.x, self.y)
            rows_cleared = self.board.remove_full_rows()

            # Calculate post-placement metrics
            post_height = self.board.get_aggregate_height()
            post_holes = self.board.get_holes()

            # Update rewards
            if rows_cleared:
                if rows_cleared > 0:
                    reward += 1000 * rows_cleared  # Reward for clearing rows
            # reward += pre_height - post_height  # Reward for reducing height
            reward -= (post_holes - pre_holes) * 10  # Penalize creating holes

            # Update block and check game over  
            self.current_block = self.next_block
            self.next_block = self.blocks.get_block(random.choice(list(self.blocks.blocks.keys())))
            self.x = self.board.width // 2 - len(self.current_block.shape[0]) // 2
            self.y = 0

            if not self.board.is_valid_position(self.current_block, self.x, self.y):
                done = True
                reward -= 100  # Game over penalty

        observation = self._get_observation()
        return observation, reward, done, {}

    def _get_observation(self):
        """
        Returns the current state of the game board as a NumPy array.

        The game board is retrieved from the `get_board` method of the `board` object
        and converted to a NumPy array with a data type of float32.

        Returns:
            np.ndarray: The current state of the game board.
        """
        # Return the game board as a NumPy array
        return np.array(self.board.get_board(), dtype=np.float32)

    def render(self, mode='human'):
        """
        Renders the current state of the Tetris game board.
        Args:
            mode (str): The mode in which to render the board. Default is 'human'.
                        Other modes can be implemented as needed.
        Returns:
            None
        """
        self._print_board()

    def close(self):
        pass
    
    def _print_board(self):
        """Print the board and the score on the screen"""
        
        for line in self.board.get_board():
            print(' '.join([str(cell) if cell != 0 else ' ' for cell in line]))
        
        # Print the score
        print(f"Score: {self.board.get_score()}")
        
        print('-' * self.board.width*2)       

