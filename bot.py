import random
from dataclasses import dataclass
from typing import List, Tuple, Optional
from copy import deepcopy

@dataclass
class TetrisBlock:
    """Represents a Tetris block with its shape and color."""
    shape: List[List[int]]
    color: str

    def rotate(self) -> List[List[int]]:
        """Rotate the block's shape 90 degrees clockwise."""
        return list(zip(*self.shape[::-1]))

class TetrisBot:
    """AI bot that plays Tetris automatically."""
    
    # Weights for different evaluation metrics
    # WEIGHTS = {
    #     'completed_lines': 100.0,    # Reward for completing lines
    #     'holes': -20.0,              # Penalty for creating holes
    #     'bumpiness': -2.0,          # Penalty for uneven surface
    #     'height': -1.5,             # Penalty for high stacks
    #     'deep_wells': -5.0,         # Penalty for deep wells
    #     'clear_path': 10.0,         # Reward for keeping paths to holes
    #     'edge_touch': 2.0,          # Reward for touching edges
    #     'blockade': -15.0,          # Penalty for blocking holes
    # }
    
    # Best weights found by genetic algorithm: _bot_tuner_genetic.py
    WEIGHTS = {'completed_lines': 81.3544563297176, 'holes': -8.807398270155772, 'bumpiness': -8.931406062230426, 'height': -1.2377908767285875, 'deep_wells': -3.7772296733434807, 'clear_path': 6.685700940004363, 'edge_touch': 3.381064696930336, 'blockade': -18.47437323378275}
    
    @dataclass
    class Move:
        """Represents a potential move with its characteristics."""
        x: int
        shape: List[List[int]]
        score: float  # Changed to float for more precise scoring
        lowest_position: int
        holes: int
        block: TetrisBlock
        metrics: dict  # Store all evaluation metrics

    def get_next_move(self, board, current_block: TetrisBlock, x: int, y: int, 
                      held_block: Optional[TetrisBlock]) -> Tuple[int, int, bool, TetrisBlock, TetrisBlock]:
        """
        Determine the best move for the current game state.
        Returns: (x, y, should_move, current_block, held_block)
        """
        current_move = self._find_best_move(board, current_block, x, y)
        
        if held_block:
            held_move = self._find_best_move(board, held_block, x, y)
            if self._should_swap_blocks(current_move, held_move):
                current_move = held_move
                held_block, current_block = self._swap_blocks(board, current_block, held_block, x, y)

        current_block.shape = current_move.shape
        return current_move.x, y, True, current_block, held_block
    
    def _evaluate_position(self, board, block: TetrisBlock, x: int, y: int) -> Tuple[float, dict]:
        """Evaluate a potential move using multiple metrics."""
        board_with_block = self._get_board_with_block(board, block, x, y)
        
        # Calculate all metrics
        metrics = {
            'completed_lines': self._count_full_rows(board_with_block),
            'holes': self._calculate_holes(board, block, x, y),
            'bumpiness': self._calculate_bumpiness(board_with_block),
            'height': self._calculate_aggregate_height(board_with_block),
            'deep_wells': self._calculate_deep_wells(board_with_block),
            'clear_path': self._evaluate_clear_paths(board_with_block),
            'edge_touch': self._calculate_edge_touching(board_with_block, x, block),
            'blockade': self._calculate_blockades(board_with_block),
        }
        
        # Calculate weighted score
        score = sum(self.WEIGHTS[metric] * value for metric, value in metrics.items())
        
        return score, metrics

    def _should_swap_blocks(self, current_move: Move, held_move: Move) -> bool:
        """Determine if we should swap to the held block based on move characteristics."""
        if held_move.score > current_move.score:
            return True
        if held_move.score == current_move.score:
            if held_move.lowest_position > current_move.lowest_position:
                return True
            if (held_move.lowest_position == current_move.lowest_position and 
                held_move.holes < current_move.holes):
                return True
            if (held_move.lowest_position == current_move.lowest_position and 
                held_move.holes == current_move.holes):
                return random.choice([True, False])
        return False

    def _find_best_move(self, board, block: TetrisBlock, x: int, y: int) -> Move:
        """Find the best possible move for a given block."""
        best_move = None
        
        # Try all possible rotations and positions
        for _ in range(4):
            new_shape = list(zip(*block.shape[::-1]))
            test_block = TetrisBlock(new_shape, block.color)
            
            for col in range(board.width):
                lowest_y = self._find_lowest_position(board, test_block, col)
                
                if board.is_valid_position(test_block, col, lowest_y):
                    score, metrics = self._evaluate_position(board, test_block, col, lowest_y)
                    
                    current_move = self.Move(
                        x=col,
                        shape=new_shape,
                        score=score,
                        lowest_position=lowest_y,
                        holes=metrics['holes'],
                        block=test_block,
                        metrics=metrics
                    )
                    
                    if self._is_better_move(current_move, best_move):
                        best_move = current_move
            
            block.shape = new_shape
        
        return best_move

    def _is_better_move(self, current: Move, best: Optional[Move]) -> bool:
        """Determine if the current move is better than the best move found so far."""
        if not best:
            return True
        if current.score > best.score:
            return True
        if current.score == best.score:
            if current.lowest_position > best.lowest_position:
                return True
            if (current.lowest_position == best.lowest_position and 
                current.holes < best.holes):
                return True
        return False

    def _swap_blocks(self, board, current: TetrisBlock, held: TetrisBlock, 
                    x: int, y: int) -> Tuple[TetrisBlock, TetrisBlock]:
        """Swap current and held blocks if the position is valid."""
        if board.is_valid_position(held, x, y):
            return current, held
        return held, current

    def _find_lowest_position(self, board, block: TetrisBlock, x: int) -> int:
        """Find the lowest valid position for a block in a given column."""
        y = 0
        while board.is_valid_position(block, x, y + 1):
            y += 1
        return y

    def _calculate_score(self, board, block: TetrisBlock, x: int, y: int) -> int:
        """Calculate score based on completed lines."""
        board_copy = self._get_board_with_block(board, block, x, y)
        full_rows = self._count_full_rows(board_copy)
        return full_rows * full_rows  # Square of completed lines

    def _calculate_holes(self, board, block: TetrisBlock, x: int, y: int) -> int:
        """Count holes (empty cells) created below placed blocks."""
        holes = 0
        for i, row in enumerate(block.shape):
            for j, cell in enumerate(row):
                if cell == 1:
                    for k in range(y + i + 1, board.height):
                        if board.board[k][x + j] == 0:
                            holes += 1
        return holes

    def _get_board_with_block(self, board, block: TetrisBlock, x: int, y: int) -> List[List[int]]:
        """Get a copy of the board with the block placed at the specified position."""
        board_copy = deepcopy(board.get_board())
        
        for i, row in enumerate(block.shape):
            for j, cell in enumerate(row):
                if cell == 1:
                    board_copy[i + y][j + x] = 1
        
        return board_copy

    def _count_full_rows(self, board: List[List[int]]) -> int:
        """Count the number of complete rows in the board."""
        return sum(1 for row in board if all(row))
    
    def _calculate_bumpiness(self, board: List[List[int]]) -> float:
        """Calculate how bumpy (uneven) the surface is."""
        heights = self._get_column_heights(board)
        bumpiness = sum(abs(heights[i] - heights[i+1]) for i in range(len(heights)-1))
        return bumpiness

    def _calculate_aggregate_height(self, board: List[List[int]]) -> int:
        """Calculate the sum of all column heights."""
        return sum(self._get_column_heights(board))

    def _get_column_heights(self, board: List[List[int]]) -> List[int]:
        """Get the height of each column."""
        heights = []
        for col in range(1, len(board[0])-1):  # Skip border columns
            for row in range(len(board)):
                if board[row][col] == 1:
                    heights.append(len(board) - row)
                    break
            else:
                heights.append(0)
        return heights

    def _calculate_deep_wells(self, board: List[List[int]]) -> int:
        """Calculate the depth of wells (empty columns with higher adjacent columns)."""
        heights = self._get_column_heights(board)
        wells = 0
        for i in range(len(heights)):
            if i == 0:
                wells += max(0, heights[i+1] - heights[i] - 1)
            elif i == len(heights)-1:
                wells += max(0, heights[i-1] - heights[i] - 1)
            else:
                wells += max(0, min(heights[i-1], heights[i+1]) - heights[i] - 1)
        return wells

    def _evaluate_clear_paths(self, board: List[List[int]]) -> float:
        """Evaluate the availability of clear paths to holes."""
        paths = 0
        for col in range(1, len(board[0])-1):
            blocked = False
            for row in range(len(board)):
                if board[row][col] == 1:
                    blocked = True
                elif not blocked and board[row][col] == 0:
                    paths += 1
        return paths

    def _calculate_edge_touching(self, board: List[List[int]], x: int, block: TetrisBlock) -> int:
        """Calculate how many block cells are touching the edges."""
        edge_touches = 0
        for i, row in enumerate(block.shape):
            for j, cell in enumerate(row):
                if cell == 1:
                    if x + j == 1 or x + j == len(board[0])-2:  # Skip border columns
                        edge_touches += 1
        return edge_touches

    def _calculate_blockades(self, board: List[List[int]]) -> int:
        """Calculate the number of solid blocks above holes."""
        blockades = 0
        for col in range(1, len(board[0])-1):
            hole_found = False
            for row in range(len(board)-1, -1, -1):
                if board[row][col] == 0:
                    hole_found = True
                elif hole_found and board[row][col] == 1:
                    blockades += 1
        return blockades

    def _is_better_move(self, current: Move, best: Optional[Move]) -> bool:
        """Determine if the current move is better than the best move found so far."""
        if not best:
            return True
        return current.score > best.score