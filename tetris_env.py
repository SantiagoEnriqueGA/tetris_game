import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

from game import TetrisBoard, TetrisBlock, Blocks, print_board

GAME_WIDTH = 15     # Width of the game board
GAME_HEIGHT = 20     # Height of the game board

class TetrisEnv(gym.Env):
    def __init__(self):
        super(TetrisEnv, self).__init__()
        self.board = TetrisBoard(GAME_WIDTH, GAME_HEIGHT)
        self.blocks = Blocks()
        self.current_block = self.blocks.get_block(random.choice(list(self.blocks.blocks.keys())))
        self.next_block = self.blocks.get_block(random.choice(list(self.blocks.blocks.keys())))
        self.held_block = None
        self.x = self.board.width // 2 - len(self.current_block.shape[0]) // 2
        self.y = 0

        self.action_space = spaces.Discrete(6)  # 0: left, 1: right, 2: down, 3: rotate, 4: drop, 5: hold
        self.observation_space = spaces.Box(low=0, high=1, shape=(GAME_HEIGHT, GAME_WIDTH), dtype=np.float32)

    def reset(self):
        self.board = TetrisBoard(GAME_WIDTH, GAME_HEIGHT)
        self.current_block = self.blocks.get_block(random.choice(list(self.blocks.blocks.keys())))
        self.next_block = self.blocks.get_block(random.choice(list(self.blocks.blocks.keys())))
        self.held_block = None
        self.x = self.board.width // 2 - len(self.current_block.shape[0]) // 2
        self.y = 0
        return self._get_observation()

    def step(self, action):
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

        if not self.board.is_valid_position(self.current_block, self.x, self.y + 1):
            self.board.add_block(self.current_block, self.x, self.y)
            self.board.remove_full_rows()
            self.current_block = self.next_block
            self.next_block = self.blocks.get_block(random.choice(list(self.blocks.blocks.keys())))
            self.x = self.board.width // 2 - len(self.current_block.shape[0]) // 2
            self.y = 0
            if not self.board.is_valid_position(self.current_block, self.x, self.y):
                done = True
                reward = -1
            else:
                done = False
                reward = 1
        else:
            done = False
            reward = 0

        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        return np.array(self.board.get_board(), dtype=np.float32)

    def render(self, mode='human'):
        print_board(self.board)

    def close(self):
        pass