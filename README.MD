# Terminal Tetris Game

This is a simple terminal-based Tetris game built with the `curses` library in Python. It features classic Tetris gameplay, including rotating and moving blocks, clearing rows, and tracking high scores in a SQLite database (`high_score.db`). Additionally, it includes an AI bot (`TetrisBot`) that can play the game automatically, with its performance optimized using various tuning methods such as genetic algorithms, neural networks, and manual tuning.

## Screenshot
![alt text](https://raw.githubusercontent.com/SantiagoEnriqueGA/tetris_game/refs/heads/main/game.PNG)

## Features

- **Graphical Display**: Displays blocks, game board, and scores in the terminal using `curses`.
- **Block Types**: Supports standard Tetris blocks (I, J, L, O, S, T, Z) with different colors.
- **Game Controls**: Move, rotate, and drop blocks, with additional options for holding blocks and pausing.
- **Scoring****: Earn points by clearing rows. Higher scores are achieved by clearing multiple rows simultaneously.
- **High Score Storage**: Scores are stored in `high_score.db`.
- **AI Bot**: An AI bot (`TetrisBot`) that can play the game automatically using a set of evaluation metrics.

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

## AI Bot

The game includes an AI bot (`TetrisBot`) that can play the game automatically. The bot uses a set of evaluation metrics to determine the best moves. These metrics include completed lines, holes, bumpiness, height, deep wells, clear paths, edge touching, and blockades. The bot's weights for these metrics have been optimized using a genetic algorithm.

### Bot Tuning

The AI bot (`TetrisBot`) was tuned using different methods to optimize its performance:

- **Genetic Algorithm**: The bot's weights were optimized using a genetic algorithm implemented in `_bot_tuner_genetic.py`. This method involves evolving a population of weight sets over multiple generations to find the best-performing weights.
- **Neural Network**: The bot's weights were optimized using a neural network implemented in `_bot_tuner_nn.py`. This method involves training a neural network to predict the best weights based on the current game state.
- **Manual Tuning**: The bot's weights were manually tuned using `_bot_tuner.py`. This method involves running multiple games with slight variations of the weights to find the best-performing set.

Each method has its own advantages and can be used to further improve the bot's performance.

## File Descriptions

- **`game.py`**: The main file to run the Tetris game. It initializes the game board, handles user inputs, and manages the game loop.
- **`bot.py`**: Contains the implementation of the AI bot (`TetrisBot`) that plays the game automatically. It includes the evaluation metrics and logic for determining the best moves.
- **`_bot_tuner.py`**: A script for manually tuning the bot's weights by running multiple games with slight variations of the weights to find the best-performing set.
- **`_bot_tuner_genetic.py`**: A script that uses a genetic algorithm to optimize the bot's weights. It evolves a population of weight sets over multiple generations to find the best-performing weights.
- **`_bot_tuner_nn.py`**: A script that uses a neural network to optimize the bot's weights. It trains a neural network to predict the best weights based on the current game state.

