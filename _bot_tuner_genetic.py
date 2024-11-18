from bot import TetrisBot

from tqdm import tqdm

import math
import random
import time
import multiprocessing

random.seed(252)

# Define constants for the game
GAME_WIDTH = 15     # Width of the game board
GAME_HEIGHT = 15     # Height of the game board
BLINK_TIMES = 0
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
    tetris_bot = TetrisBot()
    
    global GAME_WIDTH, GAME_HEIGHT
    
    board = TetrisBoard(GAME_WIDTH, GAME_HEIGHT)                # Initialize the board
    blocks = Blocks()                                           # Initialize the blocks
    
    # Main game loop
    next_block = None
    held_block = None
    
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
            
            if not continue_game:
                break
        
        board.add_block(block, x, y)    # Add the block to the board
        board.remove_full_rows()  # Remove the full rows from the board
    
    return board.get_score()
    
def run_game(weights, num_runs=5):
    """
    Run multiple games with the given weights and return the average score.
    Args:
        weights (dict): The weights to use for the game bot.
        num_runs (int): The number of games to run for averaging.

    Returns:
        tuple: The weights and the average score over all runs.
    """
    TetrisBot.WEIGHTS = weights
    total_score = 0
    
    for _ in range(num_runs):
        total_score += main()
    
    avg_score = total_score / num_runs
    return weights, avg_score


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
    # Genetic algorithm parameters
    population_size = 10
    num_generations = 100
    mutation_rate = 0.1
    crossover_rate = 0.7

    # Initialize population with random weights
    def initialize_population(size):
        return [random_weights() for _ in range(size)]

    def select_parents(population, scores):
        """Select two parents using a weighted probability based on scores."""
        total_score = sum(scores)
        probabilities = [score / total_score for score in scores]
        return random.choices(population, weights=probabilities, k=2)

    def crossover(parent1, parent2):
        """Perform single-point crossover between two parents."""
        child1, child2 = {}, {}
        for key in parent1:
            if random.random() < crossover_rate:
                child1[key] = parent1[key]
                child2[key] = parent2[key]
            else:
                child1[key] = parent2[key]
                child2[key] = parent1[key]
        return child1, child2

    def mutate(weights, rate):
        """Apply random mutations to the weights."""
        return {
            key: value + random.uniform(-rate, rate) * value
            if random.random() < rate else value
            for key, value in weights.items()
        }

    # Evaluate the population
    def evaluate_population(population):
        with multiprocessing.Pool() as pool:
            results = list(tqdm(pool.imap(run_game, population), total=len(population), desc="Evaluating population"))
        return results

    # Genetic algorithm main loop
    print(f"Running genetic algorithm for {num_generations} generations...")
    population = initialize_population(population_size)
    best_score = 0
    best_weights = None

    for generation in range(num_generations):
        print(f"Generation {generation + 1}/{num_generations}")

        # Evaluate the current population
        results = evaluate_population(population)
        scores = [score for _, score in results]

        # Find the best weights in this generation
        max_score = max(scores)
        if max_score > best_score:
            best_score = max_score
            best_weights = population[scores.index(max_score)]
        
        print(f"Best score in generation {generation + 1}: {max_score}")
        print(f"Best weights in generation {generation + 1}: {best_weights}")


        # Create a new generation
        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = select_parents(population, scores)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1, mutation_rate))
            if len(new_population) < population_size:
                new_population.append(mutate(child2, mutation_rate))

        population = new_population

    print(f"Final best score: {best_score}")
    print(f"Final optimal weights: {best_weights}")
