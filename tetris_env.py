import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

from game import TetrisBoard, TetrisBlock, Blocks, print_board

GAME_WIDTH = 15     # Width of the game board
GAME_HEIGHT = 20     # Height of the game board

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
        self.blocks = Blocks()
        
        # Set the current block, next block, and held block
        self.current_block = self.blocks.get_block(random.choice(list(self.blocks.blocks.keys())))
        self.next_block = self.blocks.get_block(random.choice(list(self.blocks.blocks.keys())))
        self.held_block = None
        
        # Set the starting position for the current block
        self.x = self.board.width // 2 - len(self.current_block.shape[0]) // 2
        self.y = 0

        # Define the action and observation spaces
        self.action_space = spaces.Discrete(6)  # 0: left, 1: right, 2: down, 3: rotate, 4: drop, 5: hold
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
        Parameters:
        action (int): The action to be performed. The possible actions are:
            0 - Move left
            1 - Move right
            2 - Move down
            3 - Rotate
            4 - Drop
            5 - Hold
        Returns:
        tuple: A tuple containing:
            - observation (any): The current state of the environment.
            - reward (int): The reward obtained from performing the action.
            - done (bool): Whether the game is over.
            - info (dict): Additional information (empty dictionary in this case).
        """
        # For each action, update the game state accordingly
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
            # If no block is held, hold the current block and get the next block
            if self.held_block is None:
                self.held_block = self.current_block
                self.current_block = self.next_block
                self.next_block = self.blocks.get_block(random.choice(list(self.blocks.blocks.keys())))
            # Else, swap the held block with the current block
            else:
                if self.board.is_valid_position(self.held_block, self.x, self.y):
                    self.held_block, self.current_block = self.current_block, self.held_block

        # Check if the game is over or if a row is filled
        if not self.board.is_valid_position(self.current_block, self.x, self.y + 1):
            # Add the current block to the board and remove any full rows
            self.board.add_block(self.current_block, self.x, self.y)
            self.board.remove_full_rows()
            
            # Update the current block and next block
            self.current_block = self.next_block
            self.next_block = self.blocks.get_block(random.choice(list(self.blocks.blocks.keys())))
            
            # Set the starting position for the new current block
            self.x = self.board.width // 2 - len(self.current_block.shape[0]) // 2
            self.y = 0
            
            # Check if the new block can be placed on the board, else the game is over
            if not self.board.is_valid_position(self.current_block, self.x, self.y):
                done = True
                reward = -1
            else:
                done = False
                reward = 1
        # If the game is not over, continue playing
        else:
            done = False
            reward = 0

        return self._get_observation(), reward, done, {}

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
        print_board(self.board)

    def close(self):
        pass