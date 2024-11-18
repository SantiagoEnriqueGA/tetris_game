from bot import TetrisBot

import sqlite3
import tqdm

import math
import random
import time
import multiprocessing

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


def main():
    bot = True
    tetris_bot = TetrisBot()
    
    global GAME_WIDTH, GAME_HEIGHT
    
    board = TetrisBoard(GAME_WIDTH, GAME_HEIGHT)                # Initialize the board
    blocks = Blocks()                                           # Initialize the blocks
    
    # Main game loop
    next_block = None
    held_block = None
    last_move_down_time = time.time()
    move_down_interval = 0.75  # Interval in seconds for moving the block down
    
    while not board.is_game_over():     
        # For first pass, get a random block
        if next_block is None:
            next_block = blocks.get_block(random.choice(list(blocks.blocks.keys())))     # Get a random block        
        
        # Set the current block to the next block
        block = next_block
        next_block = blocks.get_block(random.choice(list(blocks.blocks.keys())))         # Get a random block

        x = board.width // 2 - len(block.shape[0]) // 2     # Initial x position, center the block
        y = 0                                               # Initial y position
        
        # Get user input, column index to move the block
        while board.is_valid_position(block, x, y):          
            
            # If bot mode is enabled
            if bot:
                # If no block is held, hold the current block
                if held_block is None:
                    held_block = block
                    block = next_block
                    pass
                
                # Get the bot input
                x, y, continue_game, block, held_block = tetris_bot.get_next_move(board, block, x, y, held_block)
                
                # Move the block down
                while board.is_valid_position(block, x, y + 1):
                    y += 1
                
                # Sleep for the bot speed
                time.sleep(BOT_SPEED)
            
            # Move the block down based on the timer
            if time.time() - last_move_down_time >= move_down_interval:
                if board.is_valid_position(block, x, y + 1):
                    y += 1
                else:
                    break
                last_move_down_time = time.time()
            
            if not continue_game:
                break
        
        board.add_block(block, x, y)    # Add the block to the board
        board.remove_full_rows()  # Remove the full rows from the board
    
    return board.get_score()
    
def run_game(weights):
    """Run a single game with the given weights."""
    TetrisBot.WEIGHTS = weights
    score = main()
    return weights, score

def evolve_weights(best_weights, mutation_rate=0.1):
    """Mutate the best weights to explore nearby configurations."""
    return {
        key: value + random.uniform(-mutation_rate, mutation_rate) * value
        for key, value in best_weights.items()
    }

def random_weights():
    """Generate a random set of weights."""
    return {
        'completed_lines': random.uniform(50, 200),     # Positive, encourages clearing lines
        'holes': random.uniform(-50, -5),               # Negative, penalizes creating holes
        'bumpiness': random.uniform(-10, -1),           # Negative, penalizes uneven surfaces
        'height': random.uniform(-5, -0.5),             # Negative, penalizes high stacks
        'deep_wells': random.uniform(-10, -1),          # Negative, penalizes deep wells
        'clear_path': random.uniform(1, 20),            # Positive, rewards access to holes
        'edge_touch': random.uniform(0, 5),             # Positive, rewards touching edges
        'blockade': random.uniform(-20, -5),            # Negative, penalizes blocking holes
    }

if __name__ == '__main__':
    best_score = 0
    best_weights = random_weights()
    num_generations = 100
    games_per_generation = 25

    print(f"Optimizing weights over {num_generations} generations...")

    for generation in range(num_generations):
        print(f"Generation {generation + 1}/{num_generations}")
        results = []
        
        with multiprocessing.Pool() as pool:
            # Run games with slight variations of the best weights
            weight_variants = [evolve_weights(best_weights) for _ in range(games_per_generation)]
            results = pool.map(run_game, weight_variants)
        
        # Find the best performing weights
        for weights, score in results:
            if score > best_score:
                best_score = score
                best_weights = weights
        
        print(f"Best score: {best_score} with weights: {best_weights}")
    
    print("Final optimal weights:", best_weights)
    