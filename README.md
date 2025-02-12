## About

<h1 align="center">Snake_NE</h1><br>
<p align="center">
  <img alt="Snake_NE" title="Snake_NE" src="https://github.com/jasperpieterse/Snake-NE/blob/82b9fea2ae5cf569763c973549716951d57bab29/SnakeGIF.gif?raw=true" width="450"><br>
</p>

<h4 align="center"> A snake game that learns autonomously using the NEAT algorithm</h4>

This repository a **snake game** where the snake learns to navigate its environment **autonomously** using the **NEAT (NeuroEvolution of Augmenting Topologies)** algorithm. It was developed by Jasper Pieterse and Daria Mihalia as the final project for the course **Natural Computing (NWI-IMC042)**.

The goal is to train a virtual snake to **collect apples while avoiding obstacles**. The snake is controlled by a **neural network** that evolves over multiple generations through neuroevolution. This allows users to explore how different **inputs** and **outputs** affect the learning process of the game-playing agent.

A detailed explanation is provided in the attached report.

## Installation 

1. Clone the Repository
  ```bash
  git clone https://github.com/jasperpieterse/SerpentineSynapses.git
  ```
2. Set Up Python Virtual Environment (Optional but Recommended)
  ```bash
  python3 -m venv env
  source env/bin/activate  # On Windows, use `env\Scripts\activate`
  ```
  Or using conda:
  ```bash
  conda env create --name snake_neat -f environment.yml
  conda activate snake_neat
  ```

3. Install dependencies
  ```bash
  pip3 install -r requirements.txt
  ```
This will install the necessary libraries, including:

* **[Pygame](https://github.com/pygame/)** for the game environment
* **[neat-python](https://github.com/CodeReclaimers/neat-python)**: for the NEAT algorithm

## Usage Instructions

The main script can be run via a Jupyter notebook and is configured through the config.py file. The script will run the NEAT algorithm to train a snake to play the game autonomously. The game will be displayed in a Pygame window, showing the snake's progress in real-time.

### NEAT Settings

Modify the following parameters in config.py to adjust the learning process:

- N_RUNS: Number of runs of the NEAT algorithm.
- N_GENERATIONS: Number of generations for each run.
- FITNESS_ITERS: Number of iterations to compute a genome's fitness with in each generation.

Further customization can be done in neat-config.py. Refer to the [NEAT documentation](https://neat-python.readthedocs.io/en/latest/config_file.html) for details.

### Snake Settings

- INPUT_FEATURES: List of features influencing the snake's decision-making process (e.g., 'wall', 'relative_body', 'relative_apple', 'relative_obstacle').
- FRAME_OF_REFERENCE: Choose between 'NSEW' (cardinal directions) or 'SNAKE' (relative to snake's orientation).
- USE_OBSTACLES: Boolean flag indicating whether obstacles should be included in the game environment.
- MAX_TIME_TO_EAT_APPLE: Maximum amount of timesteps the snake has to eat the apple before dying.
- USE_DUMMY_INPUTS: Boolean flag indicating whether to use dummy inputs for the neural network.
- HISTORY_LENGTH: Number of previous moves to remember.

### Input Features

Based on the frame of reference, each feature is in either four directions (North, South, West, East) or three directions (Front, Left, Right).

- Relative Wall: Inverted distances to walls in each direction to fall within the range [0,1].
- Relative Body: Inverted distances of the closest segment of the snake's body in each direction.
- Relative Apple: Inverted distances of the apple's position relative to the snake's head in each direction.
- Relative Obstacle: Inverted distances to obstacle in each direction
- Binary Wall: The presence of walls directly next to the snake's head in each direction (1 for wall, 0 for no wall).
- Binary Body: Binary indicators of segments of the snake's body immediately next to the snake's head in each direction (1 for body, 0 for no body)
- Binary Apple: Binary indicators of the apple's position of apple being present along the axis relative to the snake's head in each direction (1 for apple, 0 for no apple).
- Binary Obstacle: Binary indicators of obstacle immediately next to the snake's head in each direction (1 for obstacle, 0 for no obstacle)
- History: Set amount of nodes representing the previous N moves of the snake using the encoding $1/(N+1)$ for $N = {up:0, down:1, left:2, right:3}$ for the NSEW frame and $N = {forward:0, left:1, right:2}$ for the Snake Frame of reference

## Credits

This project extends the original implementation from [this repository](https://github.com/danielchang2002/5038W_Final) with the following improvements:

-Refactored the code into a class structure, allowing separate instances of the Snake_NE agent to run and compare. Previously, the code used global variables, requiring a full restart for each run and external storing of results for comparisons.
- Flexible and Custom Input Features for the Neural Network (detailed below)
- Flexible frame of reference for the snake orientation (either cardinal directions or relative to the snake's orientation)
- Integration of obstacles in the game environment