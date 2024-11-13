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

# def run_game(agent_weights):
#     global GAME_WIDTH, GAME_HEIGHT
    
#     board = TetrisBoard(GAME_WIDTH, GAME_HEIGHT)                         # Initialize the board
#     blocks = Blocks()                                           # Initialize the blocks
    
#     # Main game loop
#     next_block = None
#     held_block = None
    
#     while not board.is_game_over():  
#         # Print board
#         # for line in board.get_board():
#         #     print(' '.join([str(cell) for cell in line]))
#         # print('-' * board.width*2)       
        
#         # For first pass, get a random block
#         if next_block is None:
#             next_block = blocks.get_block(random.choice(list(blocks.blocks.keys())))     # Get a random block        
        
#         # Set the current block to the next block
#         block = next_block
#         next_block = blocks.get_block(random.choice(list(blocks.blocks.keys())))         # Get a random block

#         x = board.width // 2 - len(block.shape[0]) // 2     # Initial x position, center the block
#         y = 0                                               # Initial y position
        
#         # Print block
#         # for line in block.shape:
#         #     print(' '.join([str(cell) for cell in line]))
        
#         move_count = 0  # Initialize move counter
#         # Get user input, column index to move the block
#         while board.is_valid_position(block, x, y):          
#             # Use agent weights to decide the move
#             move = decide_move(agent_weights, board, block, held_block, next_block, x, y)
#             move_count += 1  # Increment move counter

#             if move == 'left': 
#                 # If can move left, move left
#                 if board.is_valid_position(block, x - 1, y):
#                     x -= 1
#                 # Else, try to move down, if can't move down, break
#                 elif board.is_valid_position(block, x, y + 1):
#                     y += 1
#                 else:
#                     break              
#             if move == 'right': 
#                 # If can move right, move right
#                 if board.is_valid_position(block, x + 1, y):
#                     x += 1
#                 # Else, try to move down, if can't move down, break
#                 elif board.is_valid_position(block, x, y + 1):
#                     y += 1
#                 else:
#                     break
#             if move == 'rotate': 
#                 # If can rotate, rotate
#                 new_shape = list(zip(*block.shape[::-1]))
#                 if board.is_valid_position(TetrisBlock(new_shape, block.color), x, y):
#                     block.shape = rotate_block(block.shape)
#                 # Else, try to move down, if can't move down, break
#                 elif board.is_valid_position(block, x, y + 1):
#                     y += 1
#                 else:
#                     break
#             if move == 'hold':
#                 # If no held block, swap the current block with the held block
#                 if held_block is None:
#                     held_block = block
#                     block = next_block
#                     next_block = blocks.get_block(random.choice(list(blocks.blocks.keys())))
#                 # Else, swap the current block with the held block, if possible
#                 elif board.is_valid_position(held_block, x, y):
#                     held_block, block = block, held_block
#                 # Else, try to move down, if can't move down, break
#                 elif board.is_valid_position(block, x, y + 1):
#                     y += 1
#                 else:
#                     break
#             if move == 'down' or move_count % 3 == 0:
#                 # If can move down, move down
#                 if board.is_valid_position(block, x, y + 1):
#                     y += 1
#                 # Else, break
#                 else:
#                     break
        
#         # Move the block down if needed
#         while board.is_valid_position(block, x, y + 1):
#             y += 1
                                            
#         if board.is_valid_position(block, x, y):
#             board.add_block(block, x, y)    # Add the block to the board
#         else:
#             break       
#         board.remove_full_rows()  # Remove the full rows from the board
        
#     return board.get_score()    # Get the final score

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
            index = decide_index(agent_weights, board, block, x, y)
            
            # Rotate the block
            rotation = index // 3
            for r in range(rotation):
                new_shape = rotate_block(block.shape)
                if board.is_valid_position(TetrisBlock(new_shape, block.color), x, y):
                    block.shape = new_shape
            
            # Move the block to the left or right
            if board.is_valid_position(block, index // 3, y):
                x = index // 3          
            
            # Move the block down
            if board.is_valid_position(block, x, y + 1):
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

def decide_move(agent_weights, board, block, held_block, next_block, x, y):
    """Decide the move based on agent weights and the current game state."""
    moves = ['left', 'right', 'down', 'rotate', 'hold']
    move_scores = [0] * len(moves)
    
    score = calculate_score(board, block, x, y)
    lowest = calculate_lowest_position(board, block, x, y)
    holes = calculate_holes(board, block, x, y)
    
    # Evaluate each move
    for i, move in enumerate(moves):
        if move == 'left' and board.is_valid_position(block, x - 1, y):
            move_scores[i] = agent_weights[0]
        elif move == 'right' and board.is_valid_position(block, x + 1, y):
            move_scores[i] = agent_weights[1]
        elif move == 'down' and board.is_valid_position(block, x, y + 1):
            move_scores[i] = agent_weights[2]
        elif move == 'rotate':
            new_shape = rotate_block(block.shape)
            if board.is_valid_position(TetrisBlock(new_shape, block.color), x, y):
                move_scores[i] = agent_weights[3]
        elif move == 'hold':
            if held_block is None or board.is_valid_position(held_block, x, y):
                move_scores[i] = agent_weights[4]
    
    # Choose the move with the highest score
    best_move_index = move_scores.index(max(move_scores))
    return moves[best_move_index]

def decide_index(agent_weights, board, block, x, y):
    """Get the best move for the current block based on the highest score, lowest position, and fewest holes * Agent Weights."""
    index_scores = [0] * board.width * 3
    
    # For each possible rotation and position of the block
    for i in range(4):
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
                score = calculate_score(board, block, j, y)
                lowest = calculate_lowest_position(board, block, j, y)
                holes = calculate_holes(board, block, j, y)
                
                # Calculate the index for the current move
                index = j * 3
                index_scores[index] = score * agent_weights[0]
                index_scores[index + 1] = lowest * agent_weights[1]
                index_scores[index + 2] = holes * agent_weights[2]
        
        # Rotate the block for the next iteration
        block.shape = new_shape
    
    return index_scores.index(max(index_scores))

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
        return [random.uniform(-1, 1) for _ in range(GAME_WIDTH * 3)]

    def evaluate_population(self):
        """Evaluate the population in parallel and return their scores."""
        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            scores = list(executor.map(evaluate_agent_parallel, self.population))
        return scores

    def select_parents(self, scores):
        """Select parents based on their scores using roulette wheel selection."""
        total_score = sum(scores)
        if total_score == 0:
            selection_probs = [1 / len(scores) for _ in scores]
        else:
            selection_probs = [score / total_score for score in scores]
        parents = random.choices(self.population, weights=selection_probs, k=2)
        return parents

    def crossover(self, parent1, parent2):
        """Perform crossover between two parents to create two children."""
        if random.random() < self.crossover_rate:
            crossover_point = random.randint(1, len(parent1) - 1)
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
            return child1, child2
        else:
            return parent1, parent2

    def mutate(self, agent):
        """Mutate an agent's weights."""
        for i in range(len(agent)):
            if random.random() < self.mutation_rate:
                agent[i] += random.uniform(-0.1, 0.1)
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

    def run(self, generations):
        """Run the genetic algorithm for a given number of generations."""
        start_time = time.time()
        best_scores = []
        
        for generation in range(generations):
            gen_start_time = time.time()
            
            # Evaluate population in parallel
            scores = self.evaluate_population()
            
            # Track best score and agent
            best_score = max(scores)
            best_agent = self.population[scores.index(best_score)]
            best_scores.append(best_score)
            
            # Optional: Save best agent periodically
            if (generation + 1) % 10 == 0:
                self.save_agent(best_agent, f'best_agent_gen_{generation}.json')
            
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
    POPULATION_SIZE = 100
    MUTATION_RATE = 0.1
    CROSSOVER_RATE = 0.7
    GENERATIONS = 100
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
