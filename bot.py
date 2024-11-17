import random
import math

class TetrisBlock:
    def __init__(self, shape, color):
        """A class to represent a Tetris block."""
        self.shape = shape
        self.color = color

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