## About

<h1 align="center">Snake-NEAT</h1><br>
<p align="center">
  <img alt="Brainy Boa" title="Brainy Boa" src="results/SnakeGIF.gif" width="450"><br>
</p>

<h4 align="center">A snake game that learns autonomously using the NEAT algorithm</h4>

Developed by Jasper Pieterse and Daria Mihalia for the final project of the course Natural Computing (NWI-IMC042). 

This repository is allows one to train a virtual snake to collect apples while avoiding obstacles. The snake is controlled by a neural network, trained using the NEAT (NeuroEvolution of Augmenting Topologies) algorithm. This code allows you to explore how the input and outputs the neural network receives, influence the learning using NEAT. For more informationm, refer to the report attached.

This project is an extension of the original found in [this repository](https://github.com/danielchang2002/5038W_Final), with additional features and enhancements:

- Flexible and Custom Input Features [Listed Below]
- Introducing a frame of reference for the snake, enhancing learning efficiency
- Integration of obstacles in the game environment
- Framework for comparing multiple instances of the snake's behavior

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
- MIN_TIME_TO_EAT_APPLE: Minimum time required to eat an apple.
- FITNESS_ITERS: Number of iterations to compute a genome's fitness
Additional tweaks to the NEAT algorithm can be done in the `neat-config.py` file. For this, refer to [NEAT documentation](https://neat-python.readthedocs.io/en/latest/config_file.html)
  
### Snake Settings
- INPUT_FEATURES: List of features influencing the snake's decision-making process (e.g., 'wall', 'relative_body', 'relative_apple', 'relative_obstacle').
- FRAME_OF_REFERENCE: Choose between 'nswe' (cardinal directions) or 'snake' (relative to snake's orientation).
- HISTORY_LENGTH: Number of previous moves to remember.
- USE_OBSTACLES: Boolean flag indicating whether obstacles should be included in the game environment.

### Input Features
Based on the frame of reference, each feature is in either four directions (North, South, West, East) or three directions (Front, Left, Right).
- Wall: Inverted distances to walls in each direction.
- Relative Body: Inverted distances of nearby segments of the snake's body in each direction.
- Binary Body: Binary indicators of nearby segments of the snake's body in each direction.
- Relative Apple: Inverted distances of the apple's position relative to the snake's head in each direction.
- Binary Apple: Binary indicators of the apple's position relative to the snake's head in each direction.
- Relative Obstacle: Inverted distances to obstacles in each direction
- Binary Obstacle: Binary indicators of nearby obstacles in each direction.
- Combined Obstacle Body: Binary indicators of nearby obstacles and segments of the snake's body in each direction.
  
### Outputs

