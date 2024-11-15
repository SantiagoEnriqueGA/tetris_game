# Implements a simple Tetris game using the curses library for terminal-based graphics.
# Classes:
#     TetrisBoard: Represents the game board and handles game logic such as adding blocks, checking for valid positions, and removing full rows.
#     TetrisBlock: Represents a Tetris block with a specific shape and color.
#     Blocks: Initializes and provides predefined Tetris blocks.
# Functions:
#     print_board(stdscr, board, x_offset=2): Prints the game board and score on the screen.
#     print_block(stdscr, block, x, y, offset=2): Prints a Tetris block on the screen.
#     print_shadow(stdscr, board, block, x, y, offset=2): Prints the shadow of a Tetris block on the screen.
#     main(stdscr): The main function that runs the Tetris game loop.

import sqlite3
import tqdm

import math
import random
import time
import multiprocessing


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

def bot_input(board, block, x, y, held_block):
    """
    Bot that plays the game automatically.
    Logic:
        1. Get the best move for the current block.
        2. Get the best move for the held block.
        3. If the held block has a higher score, swap the blocks.
        4. If the held block has the same score but a lower position, swap the blocks.
        5. If the held block has the same score and position but fewer holes, swap the blocks.
        6. If same score, position, and holes, choose random block.
        7. Perform the best move.
    """    
    # Get score for the current block
    best_move, best_shape, best_score, lowest_position, best_holes = get_best_move(board, block, x, y)
    
    # Get the best move for the held block
    held_move, held_shape, held_score, held_lowest, held_holes = get_best_move(board, held_block, x, y)
    
    # If the held block has a higher score, swap the blocks
    if held_score > best_score:
        best_move = held_move
        best_shape = held_shape
        best_score = held_score
        lowest_position = held_lowest
        best_holes = held_holes
        held_block, block = swap_blocks(board, block, held_block, x, y)
    
    # If the held block has the same score but a lower position, swap the blocks
    elif held_score == best_score and held_lowest > lowest_position:
        best_move = held_move
        best_shape = held_shape
        best_score = held_score
        lowest_position = held_lowest
        best_holes = held_holes
        held_block, block = swap_blocks(board, block, held_block, x, y)
        
    # If the held block has the same score and position but fewer holes, swap the blocks
    elif held_score == best_score and held_lowest == lowest_position and held_holes < best_holes:
        best_move = held_move
        best_shape = held_shape
        best_score = held_score
        lowest_position = held_lowest
        best_holes = held_holes
        held_block, block = swap_blocks(board, block, held_block, x, y)
    
    # If same score, position, and holes, choose random block
    elif held_score == best_score and held_lowest == lowest_position and held_holes == best_holes:
        if random.choice([True, False]):
            best_move = held_move
            best_shape = held_shape
            best_score = held_score
            lowest_position = held_lowest
            best_holes = held_holes
            held_block, block = swap_blocks(board, block, held_block, x, y)
    
    # Perform the best move
    x, block = best_move
    block.shape = best_shape
    
    # Return the new position of the block
    return x, y, True, block, held_block

def swap_blocks(board, block, held_block, x, y):
    """Swap the held block with the current block."""
    if board.is_valid_position(held_block, x, y):
        held_block, block = block, held_block
        return held_block, block
    else:
        return held_block, block

def get_best_move(board, block, x, y):
    """Get the best move for the current block based on the highest score, lowest position, and fewest holes."""
    best_score = 0
    lowest_position = float('inf')
    best_holes = float('inf')
    best_move = None
    best_shape = block.shape
    
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
                
                # Update the best score and move
                if best_move is None:
                    best_move = (j, block)
                    best_score = score
                    lowest_position = lowest
                    best_holes = holes
                    best_shape = block.shape
                # If the score is higher, choose the move with the highest score
                if score > best_score:
                    best_score = score
                    best_move = (j, block)
                    best_shape = block.shape
                    lowest_position = lowest
                    best_holes = holes
                # If the scores are equal, choose the move with the lowest position
                elif score == best_score and lowest > lowest_position:
                    lowest_position = lowest
                    best_move = (j, block)
                    best_shape = block.shape
                    best_holes = holes
                # If the scores and positions are equal, choose the move with the fewest holes
                elif score == best_score and lowest == lowest_position and holes < best_holes:
                    best_holes = holes
                    best_move = (j, block)
                    best_shape = block.shape
        
        # Rotate the block for the next iteration
        block.shape = new_shape
    
    return best_move, best_shape, best_score, lowest_position, best_holes

def calculate_score(board, block, x, y):
    """Calculate the score for the given block position."""
    # Get the board and the board types
    board_types = board.get_board_type()
    board = board.get_board()
    
    # Copy the board and the board types
    board_copy = [row.copy() for row in board]
    board_types_copy = [row.copy() for row in board_types]
    
    # Add the block to the board copy
    for i in range(len(block.shape)):
        for j in range(len(block.shape[i])):
            if block.shape[i][j] == 1:
                board_copy[i + y][j + x] = 1
                board_types_copy[i + y][j + x] = block.color
    
    # Remove the full rows from the board copy
    full_rows = []
    for i in range(len(board_copy)):
        if all(board_copy[i]):
            full_rows.append(i)
    
    for i in full_rows:
        board_copy.pop(i)
        board_copy.insert(0, [2 if x == 0 or x == len(board_copy[0]) - 1 else 0 for x in range(len(board_copy[0]))])
        
        board_types_copy.pop(i)
        board_types_copy.insert(0, ['X' if x == 0 or x == len(board_types_copy[0]) - 1 else 0 for x in range(len(board_types_copy[0]))])
    
    # Calculate the score based on the number of full rows removed
    score = int(math.pow(len(full_rows), 2))
    
    return score

def calculate_lowest_position(board, block, x, y):
    """Calculate the lowest position for the given block."""
    while board.is_valid_position(block, x, y):
        y += 1
    return y - 1

def calculate_holes(board, block, x, y):
    """Calculate the number of holes in the board. For each empty cell under a block cell, increment the holes count."""
    holes = 0
    for i in range(len(block.shape)):
        for j in range(len(block.shape[i])):
            if block.shape[i][j] == 1:
                for k in range(y + i + 1, board.height):
                    if board.board[k][x + j] == 0:
                        holes += 1
    return holes

def read_high_score():
    """Read the high score from the database"""
    conn = sqlite3.connect('high_score.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS scores (high_score INTEGER)''')
    cursor.execute('''SELECT MAX(high_score) FROM scores''')
    result = cursor.fetchone()
    conn.close()
    return result[0] if result[0] is not None else 0

def read_high_scores(count=5):
    """Read the high scores from the database"""
    conn = sqlite3.connect('high_score.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS scores (high_score INTEGER)''')
    cursor.execute('''SELECT high_score FROM scores ORDER BY high_score DESC LIMIT ?''', (count,))
    result = cursor.fetchall()
    conn.close()
    return [row[0] for row in result]
    
def write_high_score(score):
    """Write the high score to the database"""
    conn = sqlite3.connect('high_score.db')
    cursor = conn.cursor()
    cursor.execute('''INSERT INTO scores (high_score) VALUES (?)''', (score,))
    conn.commit()
    conn.close()
    

def main():
    bot = True
    
    global GAME_WIDTH, GAME_HEIGHT
    high_scores = read_high_scores(5)
    
    board = TetrisBoard(GAME_WIDTH, GAME_HEIGHT)                         # Initialize the board
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
                x, y, continue_game, block, held_block = bot_input(board, block, x, y, held_block)
                
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
    
def run_game(seed):
    random.seed(seed)
    score = main()
    return seed, score

if __name__ == '__main__':
    best = 0
    best_seed = 0
    num_games = 1000

    print(f"Finding best seed from {num_games} games...")
    with multiprocessing.Pool() as pool:
        results = list(tqdm.tqdm(pool.imap(run_game, range(num_games)), total=num_games))

    for seed, score in results:
        if score > best:
            best = score
            best_seed = seed

    print(f"Best score: {best}, Seed: {best_seed}")
    
