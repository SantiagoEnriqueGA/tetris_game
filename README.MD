# Terminal Tetris Game

This is a simple terminal-based Tetris game built with the `curses` library in Python. It features classic Tetris gameplay, including rotating and moving blocks, clearing rows, and tracking high scores in a SQLite database (`high_score.db`).

## Features

- **Graphical Display**: Displays blocks, game board, and scores in the terminal using `curses`.
- **Block Types**: Supports standard Tetris blocks (I, J, L, O, S, T, Z) with different colors.
- **Game Controls**: Move, rotate, and drop blocks, with additional options for holding blocks and pausing.
- **Scoring**: Earn points by clearing rows. Higher scores are achieved by clearing multiple rows simultaneously.
- **High Score Storage**: Scores are stored in `high_score.db`.

## Classes and Functions

- **Classes**:
  - **`TetrisBoard`**: Handles game board logic, including adding blocks, checking for valid positions, removing full rows, and scoring.
  - **`TetrisBlock`**: Represents individual blocks with specific shapes and colors.
  - **`Blocks`**: Initializes and provides all predefined Tetris blocks.
- **Functions**:
  - **`print_board(stdscr, board, x_offset=2)`**: Displays the game board and score in the terminal.
  - **`print_block(stdscr, block, x, y, offset=2)`**: Displays a block at a specified position.
  - **`print_shadow(stdscr, board, block, x, y, offset=2)`**: Displays the shadow of a block to indicate where it will land.
  - **`main(stdscr)`**: Main game loop function.

## Controls

| Key                   | Action                        |
|-----------------------|-------------------------------|
| Left Arrow / A        | Move block left               |
| Right Arrow / D       | Move block right              |
| Down Arrow / S        | Move block down              |
| Up Arrow / W          | Rotate block                  |
| Space                 | Hard drop                     |
| H                     | Hold current block            |
| P                     | Pause game                    |

## Installation

To play the game, ensure Python and the `windows-curses` library are installed.

```bash
pip install windows-curses
```

## How to Play

1. Run the game:
    ```bash
    python game.py
    ```
2. Control the falling blocks using the arrow keys or WASD to move or rotate them.
3. Aim to complete full rows to clear them and earn points.
4. Your highest scores will be saved in `high_score.db`.

## High Scores

The game saves high scores to a SQLite database file `high_score.db`. This file is automatically created and updated each time a new high score is achieved.
