import math
import random
import time
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from game import calculate_score, calculate_lowest_position, calculate_holes

# Random seed for reproducibility
random.seed(0)

# Define constants for the game
GAME_WIDTH = 15     # Width of the game board
GAME_HEIGHT = 20     # Height of the game board
BLINK_TIMES = 0
BLINK_DELAY = 0
BOT_SPEED = 0

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
            'I': TetrisBlock([[1, 1, 1, 1]], None),
            'J': TetrisBlock([[1, 0, 0], [1, 1, 1]], None),
            'L': TetrisBlock([[0, 0, 1], [1, 1, 1]], None),
            'O': TetrisBlock([[1, 1], [1, 1]], None),
            'S': TetrisBlock([[0, 1, 1], [1, 1, 0]], None),
            'T': TetrisBlock([[0, 1, 0], [1, 1, 1]], None),
            'Z': TetrisBlock([[1, 1, 0], [0, 1, 1]], None)
        }

    def get_block(self, block_type):
        """Return the block of the given type: I, J, L, O, S, T, Z"""
        return self.blocks.get(block_type)

def swap_blocks(board, block, held_block, x, y):
    """Swap the held block with the current block."""
    if board.is_valid_position(held_block, x, y):
        held_block, block = block, held_block
        return held_block, block
    else:
        return held_block, block

def run_game(agent_weights):
    global GAME_WIDTH, GAME_HEIGHT
    
    board = TetrisBoard(GAME_WIDTH, GAME_HEIGHT)                         # Initialize the board
    blocks = Blocks()                                           # Initialize the blocks
    
    # Main game loop
    next_block = None
    held_block = None
    
    while not board.is_game_over():  
        # Print board
        # for line in board.get_board():
        #     print(' '.join([str(cell) for cell in line]))
        # print('-' * board.width*2)       
        
        # For first pass, get a random block
        if next_block is None:
            next_block = blocks.get_block(random.choice(list(blocks.blocks.keys())))     # Get a random block        
        
        # Set the current block to the next block
        block = next_block
        next_block = blocks.get_block(random.choice(list(blocks.blocks.keys())))         # Get a random block

        x = board.width // 2 - len(block.shape[0]) // 2     # Initial x position, center the block
        y = 0                                               # Initial y position
        
        # Print block
        # for line in block.shape:
        #     print(' '.join([str(cell) for cell in line]))
        
        # Get user input, column index to move the block
        while board.is_valid_position(block, x, y):          
            # Use agent weights to decide the move
            x, block = decide_index(agent_weights, board, block, x, y)        
            
            # Move the block down
            if board.is_valid_position(block, x, y + 1):
                while board.is_valid_position(block, x, y + 1):
                    y += 1
            else:
                break        
        
        # Move the block down if needed
        while board.is_valid_position(block, x, y + 1):
            y += 1
                                            
        if board.is_valid_position(block, x, y):
            board.add_block(block, x, y)    # Add the block to the board
        else:
            break       
        board.remove_full_rows()  # Remove the full rows from the board
        
    return board.get_score()    # Get the final score

def decide_index(agent_weights, board, block, x, y):
    """Get the best move for the current block based on the highest score, lowest position, and fewest holes * Agent Weights."""
    # Dict with scores for each possible rotation (0, 90, 180, 270) and position (x axis index)
    index_scores = {
                    '90': [0 for _ in range(board.width)],
                    '180': [0 for _ in range(board.width)],
                    '270': [0 for _ in range(board.width)],
                    '0': [0 for _ in range(board.width)]
                    }

    # For each possible rotation and position of the block
    for pos in index_scores:
        new_shape = list(zip(*block.shape[::-1]))
        block.shape = new_shape
        
        # For each column in the board
        for j in range(board.width):
            y = 0
            
            # Get the lowest position for the current rotation and column
            while board.is_valid_position(TetrisBlock(new_shape, block.color), j, y):
                y += 1
            y -= 1
            
            if board.is_valid_position(block, j, y):
                # Calculate the score for the current move
                score  = agent_weights['score_weights'][j] * calculate_score(board, block, j, y)
                lowest = agent_weights['lowest_weights'][j] * calculate_lowest_position(board, block, j, y)
                holes  = agent_weights['holes_weights'][j] * calculate_holes(board, block, j, y)
                
                # Calculate the index for the current move          
                index_scores[pos][j] = score + lowest + holes
        
        
    # Get the best pos and index from the index_scores dict
    best_pos = max(index_scores, key=lambda x: max(index_scores[x]))
    best_index = index_scores[best_pos].index(max(index_scores[best_pos]))
    
    # Rotate the block to the best position
    if best_pos == '90':
        block.shape = rotate_block(block.shape)
    elif best_pos == '180':
        block.shape = rotate_block(rotate_block(block.shape))
    elif best_pos == '270':
        block.shape = rotate_block(rotate_block(rotate_block(block.shape)))
    
    return best_index, block
    

def rotate_block(shape):
    """Rotate the block shape 90 degrees clockwise."""
    return [list(row) for row in zip(*shape[::-1])]

def evaluate_agent_parallel(agent):
    """Evaluate a single agent by running the game with the agent's weights."""
    return run_game(agent)

class ParallelGeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, crossover_rate, num_processes=None):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.num_processes = num_processes or mp.cpu_count()
        self.population = self.initialize_population()

    def initialize_population(self):
        """Initialize the population with random agents."""
        return [self.create_random_agent() for _ in range(self.population_size)]

    def create_random_agent(self):
        """Create a random agent with random weights."""
        return {
            'score_weights': [random.uniform(-1, 1) for _ in range(GAME_WIDTH)],
            'lowest_weights': [random.uniform(-1, 1) for _ in range(GAME_WIDTH)],
            'holes_weights': [random.uniform(-1, 1) for _ in range(GAME_WIDTH)]
        }

    def evaluate_population(self):
        """Evaluate the population in parallel and return their scores."""
        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            scores = list(executor.map(evaluate_agent_parallel, self.population))
        return scores

    def select_parents(self, scores, top_percentage=0.2):
        """Select parents based on truncation selection."""
        num_parents = int(self.population_size * top_percentage)
        sorted_population = [agent for _, agent in sorted(zip(scores, self.population), key=lambda x: x[0], reverse=True)]
        parents = random.choices(sorted_population[:num_parents], k=2)
        return parents

    def crossover(self, parent1, parent2):
        """Perform crossover between two parents to create two children."""
        if random.random() < self.crossover_rate:   # With probability crossover_rate
            child1 = {} 
            child2 = {}
            for key in parent1:
                if random.random() < 0.5:           # With probability 0.5, this key is from parent1, else from parent2
                    child1[key] = parent1[key]
                    child2[key] = parent2[key]
                else:
                    child1[key] = parent2[key]
                    child2[key] = parent1[key]
            return child1, child2
        else:
            return parent1, parent2

    def mutate(self, agent):
        """Mutate an agent's weights."""
        for key in agent:
            for i in range(len(agent[key])):
                if random.random() < self.mutation_rate:
                    agent[key][i] += random.uniform(-0.1, 0.1)
        return agent

    def create_new_population(self, scores):
        """Create a new population using selection, crossover, and mutation."""
        new_population = []
        
        # Keep the best performing agent (elitism)
        best_agent_idx = scores.index(max(scores))
        new_population.append(self.population[best_agent_idx])
        
        while len(new_population) < self.population_size:
            parent1, parent2 = self.select_parents(scores)
            child1, child2 = self.crossover(parent1, parent2)
            new_population.append(self.mutate(child1))
            if len(new_population) < self.population_size:
                new_population.append(self.mutate(child2))
                
        self.population = new_population[:self.population_size]

    def save_agent(self, agent, filename):
        """Save the agent's weights to a file."""
        with open(filename, 'w') as f:
            json.dump(agent, f)

    def load_checkpoint(self, filename):
        """Load the checkpoint from a file."""
        try:
            with open(filename, 'r') as f:
                checkpoint = json.load(f)
            self.population = checkpoint['population']  # Load the population
            start_generation = checkpoint['generation'] # Load the generation
            return start_generation
        except FileNotFoundError:
            return 0

    def save_checkpoint(self, filename, generation):
        """Save the checkpoint to a file."""
        checkpoint = {
            'population': self.population,
            'generation': generation
        }
        with open(filename, 'w') as f:
            json.dump(checkpoint, f, indent=4)

    def run(self, generations):
        """Run the genetic algorithm for a given number of generations."""
        start_time = time.time()
        best_scores = []
        checkpoint_file = 'checkpoint.json'
        
        # Load checkpoint if it exists
        start_generation = self.load_checkpoint(checkpoint_file)
        
        for generation in range(start_generation, generations):
            gen_start_time = time.time()
            
            # Evaluate population in parallel
            scores = self.evaluate_population()
            
            # Track best score and agent
            best_score = max(scores)
            best_agent = self.population[scores.index(best_score)]
            best_scores.append(best_score)
            
            # Save checkpoint
            self.save_checkpoint(checkpoint_file, generation + 1)
            
            # Create new population
            self.create_new_population(scores)
            
            # Calculate and print statistics
            avg_score = sum(scores) / len(scores)
            gen_time = time.time() - gen_start_time
            
            print(f"Generation {generation + 1}/{generations}")
            print(f"Best Score: {best_score}")
            print(f"Average Score: {avg_score:.2f}")
            print(f"Generation Time: {gen_time:.2f}s")
            print("-" * 40)
            
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f}s")
        print(f"Best score achieved: {max(best_scores)}")
        
        return best_scores

if __name__ == '__main__':
    # Configuration
    POPULATION_SIZE = 1000
    MUTATION_RATE = 0.2
    CROSSOVER_RATE = 0.7
    GENERATIONS = 10000
    NUM_PROCESSES = mp.cpu_count()  # Use all available CPU cores
    
    print(f"Running with {NUM_PROCESSES} processes")
    
    # Initialize and run the parallel genetic algorithm
    ga = ParallelGeneticAlgorithm(
        population_size=POPULATION_SIZE,
        mutation_rate=MUTATION_RATE,
        crossover_rate=CROSSOVER_RATE,
        num_processes=NUM_PROCESSES
    )
    
    best_scores = ga.run(generations=GENERATIONS)
