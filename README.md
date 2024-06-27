## About

<h1 align="center">Brainy Boa</h1><br>
<p align="center">
  <img alt="Brainy Boa" title="Brainy Boa" src="https://github.com/jasperpieterse/Brainy-Boa/blob/82b9fea2ae5cf569763c973549716951d57bab29/SnakeGIF.gif?raw=true" width="450"><br>
</p>

<h4 align="center"> A snake game that learns autonomously using the NEAT algorithm</h4>

Developed by Jasper Pieterse and Daria Mihalia for the final project of the course Natural Computing (NWI-IMC042).

This repository is allows one to train a virtual snake to collect apples while avoiding obstacles. The snake is controlled by a neural network, trained using the NEAT (NeuroEvolution of Augmenting Topologies) algorithm. This code allows the user to explore how the input and outputs the neural network receives, influence the learning of the game-playing agent. For more information, refer to the report attached.

This project is an extension of the original found in [this repository](https://github.com/danielchang2002/5038W_Final), with additional features and enhancements:

-Refactored the code into a class structure, allowing separate instances of the Brainy Boa agent to run and compare. Previously, the code used global variables, requiring a full restart for each run and external storing of results for comparisons.
- Flexible and Custom Input Features for the Neural Network (detailed below)
- Flexible frame of reference for the snake orientation (either cardinal directions or relative to the snake's orientation)
- Integration of obstacles in the game environment

## How to use:

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

3. Install requirements
  ```bash
  pip3 install -r requirements.txt
  ```

## Dependencies

* **[Pygame](https://github.com/pygame/)**: Used in game development and user interaction
* **[neat-python](https://github.com/CodeReclaimers/neat-python)**: Python implementation of the NEAT neuroevolution algorithm

## Usage Instructions

The main code can be interacted with using the Jupyter notebook and the config.py file. 

### Neat - Settings

Adjust the following parameters in config.py to influence the NEAT algorithm's behavior:

- N_RUNS: Number of runs of the NEAT algorithm.
- N_GENERATIONS: Number of generations for each run.
- FITNESS_ITERS: Number of iterations to compute a genome's fitness with in each generation.

Additional tweaks to the NEAT algorithm can be done in the `neat-config.py` file. For this, refer to [NEAT documentation](https://neat-python.readthedocs.io/en/latest/config_file.html)
  
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
- Binary Body: Binary indicators of segments of the snake's body immediate next to snake head in each direction (1 for body, 0 for no body)
- Binary Apple: Binary indicators of the apple's position of apple being present along the axis relative to the snake's head in each direction (1 for apple, 0 for no apple).
- Binary Obstacle: Binary indicators of obstacle immediately next to snake head in each direction (1 for obstacle, 0 for no obstacle)
- History: Set amount of nodes representing the previous N moves of the snake using the encoding $1/(N+1)$ for $N = {up:0, down:1, left:2, right:3}$ for the NSEW frame and $N = {forward:0, left:1, right:2}$ for the Snake Frame of reference
