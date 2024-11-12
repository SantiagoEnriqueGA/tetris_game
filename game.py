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

import curses
from curses import wrapper
import sqlite3

import math
import random
import time

# Random seed for reproducibility
random.seed(0)

# Define global variables for the game board dimensions
game_x = 15     # Width of the game board
game_y = 20     # Height of the game board

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

    def remove_full_rows(self, stdscr):
        """Remove the full rows from the board and update the score"""
        full_rows = []
        for i in range(self.height):    # Check each row
            if all(self.board[i]):      # If the row is full
                full_rows.append(i)     # Add the row to the full rows list
        
        if not full_rows: return        # If no full rows, return
        
        orignal_board = self.board.copy()               # Copy the original board
        original_cell_type = self.cell_type.copy()      # Copy the original cell type
        
        for i in full_rows:             # For each full row               
            # Animate the row removal, 
            # Change the color of the row to red
            # Blink 3 times and remove the row
            for _ in range(3):
                for j in range(1, self.width - 1):
                    self.board[i][j] = 8    
                    self.cell_type[i][j] = 'X'
                print_board(stdscr, self)
                time.sleep(0.1)
                for j in range(1, self.width - 1):
                    self.board[i][j] = 0
                    self.cell_type[i][j] = 'X'
                print_board(stdscr, self)
                time.sleep(0.1)
            
        # Reset the board to the original state
        self.board = orignal_board.copy()
        self.cell_type = original_cell_type.copy()
        
        for i in full_rows:
            # Remove the full row
            self.board.pop(i)
            self.board.insert(0, [2 if x == 0 or x == self.width - 1 else 0 for x in range(self.width)])
            
            self.cell_type.pop(i)
            self.cell_type.insert(0, ['X' if x == 0 or x == self.width - 1 else 0 for x in range(self.width)])
        
        print_board(stdscr, self)
        
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
        """Initializes the Blocks class with predefined colors and block shapes."""
        # Init colors for each block
        curses.init_pair(2, curses.COLOR_CYAN, curses.COLOR_CYAN)
        curses.init_pair(3, curses.COLOR_BLUE, curses.COLOR_BLUE)
        curses.init_pair(4, curses.COLOR_MAGENTA, curses.COLOR_MAGENTA)
        curses.init_pair(5, curses.COLOR_YELLOW, curses.COLOR_YELLOW)
        curses.init_pair(6, curses.COLOR_GREEN, curses.COLOR_GREEN)
        curses.init_pair(7, curses.COLOR_RED, curses.COLOR_RED)
        curses.init_pair(8, curses.COLOR_WHITE, curses.COLOR_WHITE)
               
        # Define the blocks, each block has a shape and a color
        self.blocks = {
            'I': TetrisBlock([[1, 1, 1, 1]], 2),
            'J': TetrisBlock([[1, 0, 0], [1, 1, 1]], 3),
            'L': TetrisBlock([[0, 0, 1], [1, 1, 1]], 4),
            'O': TetrisBlock([[1, 1], [1, 1]], 5),
            'S': TetrisBlock([[0, 1, 1], [1, 1, 0]], 6),
            'T': TetrisBlock([[0, 1, 0], [1, 1, 1]], 7),
            'Z': TetrisBlock([[1, 1, 0], [0, 1, 1]], 8)
        }

    def get_block(self, block_type):
        """Return the block of the given type: I, J, L, O, S, T, Z"""
        return self.blocks.get(block_type)

def print_board(stdscr, board, x_offset=2):
    """Print the board and the score on the screen"""
    
    # Get the board, score, and the board types
    board_types = board.get_board_type()    
    score = board.get_score()
    board = board.get_board()
    
    stdscr.clear()
    stdscr.addstr(0, 0, 'Score: {}'.format(score), curses.COLOR_RED)
    
    # Print the board, 0 is empty cell, 1 is block, 2 is wall
    y_offset = 2
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] != 0:
                type = board_types[i][j]
                
                if type == 0: pass  # Empty cell
                elif type == 'X':   # Wall
                    stdscr.addstr(i + y_offset, j * x_offset, '  ', curses.color_pair(8))
                else:               # Block
                    stdscr.addstr(i + y_offset, j * x_offset, '  ', curses.color_pair(type))

    # Print bottom wall
    for i in range(len(board[0])):
        stdscr.addstr(len(board) + y_offset, i * x_offset, '  ', curses.color_pair(8))   
        
    # Print controls
    stdscr.addstr(len(board) + y_offset + 2, 0, 'Controls:')
    stdscr.addstr(len(board) + y_offset + 3, 4, 'Right:  Right Arrow Key or D')
    stdscr.addstr(len(board) + y_offset + 4, 4, 'Left:   Left Arrow Key or A')
    stdscr.addstr(len(board) + y_offset + 5, 4, 'Down:   Down Arrow Key or S')
    stdscr.addstr(len(board) + y_offset + 6, 4, 'Rotate: Up Arrow Key or W')
    stdscr.addstr(len(board) + y_offset + 7, 4, 'Drop:   Space')
    stdscr.addstr(len(board) + y_offset + 9, 4, 'Hold:   H')
    stdscr.addstr(len(board) + y_offset + 8, 4, 'Pause:  P')
       
    stdscr.refresh()
    
def print_block(stdscr, block, x, y, offset=2):
    """Print the block on the screen"""
    # For each cell in the block, if the cell is 1, print the block
    for i in range(len(block.shape)):
        for j in range(len(block.shape[i])):
            if block.shape[i][j] == 1:
                stdscr.addstr(i + y + 2, (j + x) * offset, ' '*2, curses.color_pair(block.color))
    stdscr.refresh()

def print_shadow(stdscr, board, block, x, y, offset=2):
    """Print the shadow of the block on the screen, the shadow is the block at the lowest position"""
    # While the block can be moved down, move the block down
    while board.is_valid_position(block, x, y):
        y += 1
    
    # Print the shadow of the block with Xs
    for i in range(len(block.shape)):
        for j in range(len(block.shape[i])):
            if block.shape[i][j] == 1:
                stdscr.addstr(i + y - 1 + 2, (j + x) * offset, 'X'*2)
    stdscr.refresh()
    
def print_leaderboard(stdscr, x, y, high_scores):
    """Print the leaderboard on the screen"""
    stdscr.addstr(y, x, 'Leaderboard', curses.color_pair(7))
    
    for i in range(len(high_scores)):
        stdscr.addstr(y + 1 + i, x, '{}. {}'.format(i + 1, high_scores[i]))
    
    stdscr.refresh()
    
def handle_user_input(stdscr, board, block, x, y, held_block, next_block, blocks):
    """Handle user input for moving and rotating the block."""
    # Get the user input
    c = stdscr.getch()
    
    if c == curses.KEY_LEFT or c == ord('a'):
        # If the block can be moved left, move the block left
        if board.is_valid_position(block, x - 1, y):
            x -= 1
    elif c == curses.KEY_RIGHT or c == ord('d'):
        # If the block can be moved right, move the block right
        if board.is_valid_position(block, x + 1, y):
            x += 1
    elif c == curses.KEY_DOWN or c == ord('s'):
        # If the block can be moved down, move the block down
        if board.is_valid_position(block, x, y + 1):
            y += 1
    elif c == curses.KEY_UP or c == ord('w'):
        # Rotate the block, if the block can be rotated
        new_shape = list(zip(*block.shape[::-1]))
        if board.is_valid_position(TetrisBlock(new_shape, block.color), x, y):
            block.shape = new_shape
    elif c == ord(' '):
        # Drop the block to the lowest possible position
        while board.is_valid_position(block, x, y + 1):
            y += 1
        return x, y, False, block, held_block
    elif c == ord('p'):
        # Pause the game
        while True:
            stdscr.addstr(game_y // 2, game_x * 2 + 4, 'Paused', curses.color_pair(7))
            stdscr.refresh()
            c = stdscr.getch()
            if c == ord('p'):
                break
    elif c == ord('h'):
        if held_block is None:
            held_block = block
            block = next_block
            next_block = blocks.get_block(random.choice(list(blocks.blocks.keys())))
        else:
            if board.is_valid_position(held_block, x, y):
                held_block, block = block, held_block
        return x, y, True, block, held_block
    
    # Return the new position of the block
    return x, y, True, block, held_block

def bot_input(board, block, x, y, held_block):
    """Bot that plays the game automatically."""
    # Get the best move for the current block
    
    # Get score for the current block
    best_move, best_shape, best_score, lowest_position = get_best_move(board, block, x, y)
    
    # Get the best move for the held block

    held_move, held_shape, held_score, held_lowest = get_best_move(board, held_block, x, y)
    
    # If the held block has a higher score, swap the blocks
    if held_score > best_score:
        best_move = held_move
        best_shape = held_shape
        best_score = held_score
        lowest_position = held_lowest
    
    # If the held block has the same score but a lower position, swap the blocks
    elif held_score == best_score and held_lowest > lowest_position:
        best_move = held_move
        best_shape = held_shape
        best_score = held_score
        lowest_position = held_lowest
        
    # If same score and position, choose random block
    elif held_score == best_score and held_lowest == lowest_position:
        if random.choice([True, False]):
            best_move = held_move
            best_shape = held_shape
            best_score = held_score
            lowest_position = held_lowest
    
    # Perform the best move
    x, block = best_move
    block.shape = best_shape
    
    # Return the new position of the block
    return x, y, True, block, held_block

def get_best_move(board, block, x, y):
    """Get the best move for the current block based on the highest score and lowest position."""
    best_score = 0
    lowest_position = float('inf')
    best_move = None
    best_shape = block.shape
    
    # For each possible rotation and position of the block
    for i in range(4):
        new_shape = list(zip(*block.shape[::-1]))
        block.shape = new_shape
        
        # For each column in the board
        for j in range(board.width):
            y=0
            
            # Get the lowest position for the current rotation and column
            while board.is_valid_position(TetrisBlock(new_shape, block.color), j, y):
                y += 1
            y -= 1
            
            if board.is_valid_position(block, j, y):
                # Calculate the score for the current move
                score = calculate_score(board, block, j, y)
                lowest = calculate_lowest_position(board, block, j, y)
                
                # Update the best score and move
                if best_move is None:
                    best_move = (j, block)
                    best_score = score
                    lowest_position = lowest
                    best_shape = block.shape
                if score > best_score:
                    best_score = score
                    best_move = (j, block)
                    best_shape = block.shape
                # If the scores are equal, choose the move with the lowest position
                elif score == best_score and lowest > lowest_position:
                    lowest_position = lowest
                    best_move = (j, block)
                    best_shape = block.shape
                # If the scores are equal and the positions are equal, choose the move with the lowest x position
                elif score <= 0 and best_score <= 0 and lowest > lowest_position:
                    lowest_position = lowest
                    best_move = (j, block)
                    best_shape = block.shape
        
        # Rotate the block for the next iteration
        block.shape = new_shape
    
    return best_move, best_shape, best_score, lowest_position

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
    

def main(stdscr, bot=True):
    global game_x, game_y
    high_scores = read_high_scores(5)
    
    curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK) # Define the color pair
    board = TetrisBoard(game_x, game_y)                         # Initialize the board
    blocks = Blocks()                                           # Initialize the blocks
    print_board(stdscr, board)                                  # Print the initial board
    print_leaderboard(stdscr, game_x * 3 + 5, 1, high_scores) # Print the leaderboard
    
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
            print_board(stdscr, board)                      # Print the board
            print_leaderboard(stdscr, game_x * 3 + 5, 1, high_scores) # Print the leaderboard
            print_shadow(stdscr, board, block, x, y)        # Print the shadow of the block
            print_block(stdscr, block, x, y)                # Print the block
            
            stdscr.addstr(1, game_x*2+2, 'Next block:')     # Print the next block text
            print_block(stdscr, next_block, game_x+2, 0)    # Print the next block
            
            if held_block:
                stdscr.addstr(5, game_x*2+2, 'Held block:')
                print_block(stdscr, held_block, game_x+2, 4)
            
            # Wait for and get user input
            stdscr.timeout(100)  # Check for user input every 100ms
            
            # If bot is enabled, use bot input
            if not bot:
                x, y, continue_game, block, held_block = handle_user_input(stdscr, board, block, x, y, held_block, next_block, blocks)
            else:
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
            
            # If user, move the block down based on the timer
            if not bot and time.time() - last_move_down_time >= move_down_interval:
                if board.is_valid_position(block, x, y + 1):
                    y += 1
                else:
                    break
                last_move_down_time = time.time()
            # If bot, sleep
            else:
                time.sleep(0.1)
                break
            
            if not continue_game:
                break
        
        board.add_block(block, x, y)    # Add the block to the board
        board.remove_full_rows(stdscr)  # Remove the full rows from the board
    
    
    # Game over middle screen message
    stdscr.addstr(game_y // 2, game_x * 2 + 4, 'Game Over', curses.color_pair(7))
    
    if board.get_score() > read_high_score():
        stdscr.addstr(game_y // 2 + 2, game_x * 2 + 4, 'New High Score!', curses.color_pair(7))
        
    
    # Wait for user input to close the game
    stdscr.refresh()
    while True:
        c = stdscr.getch()
        if c != -1:
            break
        
    # Update the high score
    board.score = board.get_score()    # Get the final score
    write_high_score(board.score)      # Write the high score to the database
    
    
if __name__ == '__main__':
    # wrapper function to initialize the curses application
    wrapper(main)

