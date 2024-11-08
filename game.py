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

import math
import random
import time

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
    stdscr.addstr(len(board) + y_offset + 3, 4, 'Left:   Left Arrow Key')
    stdscr.addstr(len(board) + y_offset + 4, 4, 'Right:  Right Arrow Key')
    stdscr.addstr(len(board) + y_offset + 5, 4, 'Down:   Down Arrow Key')
    stdscr.addstr(len(board) + y_offset + 6, 4, 'Rotate: Up Arrow Key')
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

def handle_user_input(stdscr, board, block, x, y, held_block, next_block, blocks):
    """Handle user input for moving and rotating the block."""
    # Get the user input
    c = stdscr.getch()
    
    if c == curses.KEY_LEFT:
        # If the block can be moved left, move the block left
        if board.is_valid_position(block, x - 1, y):
            x -= 1
    elif c == curses.KEY_RIGHT:
        # If the block can be moved right, move the block right
        if board.is_valid_position(block, x + 1, y):
            x += 1
    elif c == curses.KEY_DOWN:
        # If the block can be moved down, move the block down
        if board.is_valid_position(block, x, y + 1):
            y += 1
    elif c == curses.KEY_UP:
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

def main(stdscr):
    global game_x, game_y
    
    curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK) # Define the color pair
    board = TetrisBoard(game_x, game_y)                         # Initialize the board
    blocks = Blocks()                                           # Initialize the blocks
    print_board(stdscr, board)                                  # Print the initial board
    
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
            print_shadow(stdscr, board, block, x, y)        # Print the shadow of the block
            print_block(stdscr, block, x, y)                # Print the block
            
            stdscr.addstr(1, game_x*2+2, 'Next block:')     # Print the next block text
            print_block(stdscr, next_block, game_x+2, 0)    # Print the next block
            
            if held_block:
                stdscr.addstr(5, game_x*2+2, 'Held block:')
                print_block(stdscr, held_block, game_x+2, 4)
            
            # Wait for and get user input
            stdscr.timeout(100)  # Check for user input every 100ms
            x, y, continue_game, block, held_block = handle_user_input(stdscr, board, block, x, y, held_block, next_block, blocks)
            
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
        board.remove_full_rows(stdscr)  # Remove the full rows from the board
    
    # Game over middle screen message
    stdscr.addstr(game_y // 2, game_x * 2 + 4, 'Game Over', curses.color_pair(7))
    
    # Wait for user input to close the game
    stdscr.refresh()
    while True:
        c = stdscr.getch()
        if c != -1:
            break
    
if __name__ == '__main__':
    # wrapper function to initialize the curses application
    wrapper(main)

