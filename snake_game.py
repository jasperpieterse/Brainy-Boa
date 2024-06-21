import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import neat
import warnings
import multiprocessing
import os
import sys
import shutil
import pickle
import configparser

from random import randint, seed, choice
from enum import Enum

# Initialize random seed for reproducibility
seed(42)

class Direction(Enum):
    """Cartesian coordinates with inverted y-axis."""
    NORTH = (0, -1)
    SOUTH = (0, 1)
    EAST = (1, 0)
    WEST = (-1, 0)


class Action(Enum):
    """Enum for the actions the snake can take."""
    STRAIGHT = 0
    LEFT = 1
    RIGHT = 2

class SnakeGame:
    """Class for the Snake game."""

    # Game Constants
    NUM_ROWS = 10
    NUM_COLS = 10
    INTERVAL = 100
    NODE_SIZE = 20
    NETWORK_WIDTH = 700
    NETWORK_HEIGHT = 900
    GAME_WIDTH = 700
    GAME_HEIGHT = 700
    WINDOW_BUFFER = 25
    BLOCK_WIDTH = GAME_WIDTH / NUM_COLS
    BLOCK_HEIGHT = GAME_HEIGHT / NUM_ROWS
    GAME_TOP_LEFT = (2 * WINDOW_BUFFER + NETWORK_WIDTH, WINDOW_BUFFER)
    BUFFER = 4

    # Colors
    RED = (235, 64, 52)
    WHITE = (245, 245, 245)
    YELLOW = (252, 227, 109)
    BLACK = (20, 20, 20)
    BLUE = (58, 117, 196)
    ORANGE = (255, 140, 0)

    # Direction encoding
    DIRECTION_ENCODING_NSEW = {
    (0, -1): 0,  # North
    (0, 1): 1,   # South
    (1, 0): 2,   # East
    (-1, 0): 3   # West
    }

    def __init__(self,
                 input_features=['wall', 'relative_body', 'relative_apple', 'relative_obstacle'],
                 input_frame_of_reference='NSEW',
                 output_frame_of_reference='NSEW',
                 history_length=3,
                 use_obstacles=True,
                 fitness_iters=10,
                 min_time_to_eat_apple=100,
                 n_runs=1,
                 n_generations=10,
                 time_interval=100,
                 checkpoint_interval=10,
                 use_dummy_inputs=False):
        
        """Initializes the game with default parameters."""
        # Game parameters
        self.INPUT_FEATURES = input_features
        self.INPUT_FRAME_OF_REFERENCE = input_frame_of_reference
        self.OUTPUT_FRAME_OF_REFERENCE = output_frame_of_reference
        self.HISTORY_LENGTH = history_length
        self.USE_OBSTACLES = use_obstacles
        self.FITNESS_ITERS = fitness_iters
        self.MIN_TIME_TO_EAT_APPLE = min_time_to_eat_apple
        self.N_RUNS = n_runs
        self.N_GENERATIONS = n_generations
        self.CHECKPOINT_INTERVAL = checkpoint_interval
        self.USE_DUMMY_INPUTS = use_dummy_inputs

        # Compute input size based on features and frame of reference
        if 'history' in self.INPUT_FEATURES:
            self.NR_INPUT_FEATURES = 4 * (len(self.INPUT_FEATURES) - 1)  if self.INPUT_FRAME_OF_REFERENCE == 'NSEW' else 3 * (len(self.INPUT_FEATURES) - 1)
            self.NR_INPUT_FEATURES += self.HISTORY_LENGTH
        else:
            self.NR_INPUT_FEATURES = 4 * len(self.INPUT_FEATURES) if (self.INPUT_FRAME_OF_REFERENCE == 'NSEW' or self.USE_DUMMY_INPUTS) else 3 * len(self.INPUT_FEATURES)

        # Store output size based on frame of reference
        self.NR_OUTPUT_FEATURES = 4 if (self.OUTPUT_FRAME_OF_REFERENCE == 'NSEW' or self.USE_DUMMY_INPUTS) else 3

        # Create an object to update neat parameters with later
        self.neat_params = {
            'input_output': {
                'DefaultGenome': {
                    'num_inputs': f'{self.NR_INPUT_FEATURES}',
                    'num_outputs': f'{self.NR_OUTPUT_FEATURES}'
                }
            },
        }

        # Initialize game state
        self.snake = []
        self.snake_set = set()
        self.apple = ()
        self.apples_eaten = 0
        self.obstacle = ()
        self.dead = False
        self.step_count = 0
        self.v_x, self.v_y = 1, 0
        self.encoded_history = [0] * self.HISTORY_LENGTH
        self.last_ate_apple = 0
        self.reset()

        # Animation parameters
        self.INTERVAL = time_interval
        self.SCREEN_WIDTH = self.WINDOW_BUFFER + self.NETWORK_WIDTH + self.WINDOW_BUFFER + self.GAME_WIDTH + self.WINDOW_BUFFER
        self.SCREEN_HEIGHT = self.NETWORK_HEIGHT + 2 * self.WINDOW_BUFFER

        # Pygame initialization
        self.screen = None
        self.font = None

        # Preprocess sensory functions
        self.sensory_functions = self.preprocess_sensory_functions()

    def reset(self):
        """Resets the game to its initial state."""
        self.snake = [self._get_random_position(avoid_edges = True)] # Snake can't start at edges
        self.snake_set = set(self.snake)
        self.v_x, self.v_y = choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
        self.front, self.left, self.right = self.get_adjacent_positions()
        self.apple = self._get_random_position(exclude=self.snake_set)
        self.obstacle = self._get_random_position(exclude=self.snake_set | {self.apple} | {self.front, self.left, self.right}) if self.USE_OBSTACLES else ()
        self.apples_eaten = 0
        self.step_count = 0
        self.last_ate_apple = 0
        self.encoded_history = [0] * self.HISTORY_LENGTH
        self.dead = False
        self.t = 0

    def _get_random_position(self, exclude=None, avoid_edges=False) -> tuple:
        """Returns a random position that is not in the exclude set."""
        
        if avoid_edges:
            min_x, max_x = 1, self.NUM_COLS - 2
            min_y, max_y = 1, self.NUM_ROWS - 2
        else:
            min_x, max_x = 0, self.NUM_COLS - 1
            min_y, max_y = 0, self.NUM_ROWS - 1

        pos = (randint(min_x, max_x), randint(min_y, max_y))
        if exclude is None:
            return pos
        while pos in exclude:
            pos = (randint(min_x, max_x), randint(min_y, max_y))
        
        return pos

    def get_adjacent_positions(self) -> tuple:
        """Returns the positions in front, left, and right of the head position."""
        front = (self.snake[-1][0] + self.v_x, self.snake[-1][1] + self.v_y)
        left = (self.snake[-1][0] + self.v_y, self.snake[-1][1] - self.v_x)
        right = (self.snake[-1][0] - self.v_y, self.snake[-1][1] + self.v_x)
        return front, left, right
    
    def simulate_headless(self, net) -> float:
        """Simulates the game without rendering it and returns the fitness score."""
        scores = []

        for _ in range(self.FITNESS_ITERS):
            self.reset()
            while not self.dead and (self.t - self.last_ate_apple <= self.MIN_TIME_TO_EAT_APPLE):
                sensory_vector = self.sensory_function()
                activations = net.activate(sensory_vector)
                action = np.argmax(activations)
                self.change_direction(action)
                apple_eaten = self.step(action)
                self.t += 1

                if apple_eaten:
                    self.last_ate_apple = self.t
            scores.append(len(self.snake))

        return np.mean(scores)

    def change_direction(self, action: int):
        """Changes the direction of the snake based on the action (output of the neural network)."""
        if self.OUTPUT_FRAME_OF_REFERENCE == 'NSEW':
            directions = [
                Direction.NORTH.value,  # North
                Direction.SOUTH.value,  # South
                Direction.EAST.value,   # East
                Direction.WEST.value    # West
            ]
            self.v_x, self.v_y = directions[action]
        elif self.OUTPUT_FRAME_OF_REFERENCE == 'SNAKE':
            if action == Action.STRAIGHT.value:  # Go straight
                pass
            elif action == Action.LEFT.value:  # Turn left
                self.v_x, self.v_y =  self.v_y, -self.v_x
            elif action == Action.RIGHT.value:  # Turn right
                self.v_x, self.v_y =  -self.v_y, self.v_x
            elif action == 3 and self.USE_DUMMY_INPUTS:
                pass
        else:
            print('Invalid frame of reference selected')


    def step(self, action) -> bool:
        """Moves the snake one step forward and checks for collisions and apple eating."""
        # Calculate the new head position
        x, y = self.snake[-1]
        new_head = (x + self.v_x, y + self.v_y)

        # Check if the snake has collided with the wall, itself, or an obstacle
        if self._is_collision(new_head):
            self.dead = True
        # If not, move the snake forward
        self.snake.append(new_head)
        self.snake_set.add(new_head)
        self.front, self.left, self.right = self.get_adjacent_positions()

        # Check if the snake has eaten the apple and if so, generate a new apple and obstacle
        ate_apple = new_head == self.apple
        if ate_apple:
            self.apples_eaten += 1
            self.apple = self._get_random_position(exclude=self.snake_set | {self.obstacle})
            if self.USE_OBSTACLES:
                self.obstacle = self._get_random_position(exclude=self.snake_set | {self.apple})
        else:
            # Remove tail
            tail = self.snake.pop(0)
            self.snake_set.remove(tail)

        # Update the history of moves
        if 'history' in self.INPUT_FEATURES:
            self.update_encoded_history(action)

        self.step_count += 1
        return ate_apple
    
    def _is_collision(self, position: tuple) -> bool:
        """Checks if the given position collides with the walls, itself, or an obstacle."""
        x, y = position
        return (
            x < 0 or x >= self.NUM_COLS or
            y < 0 or y >= self.NUM_ROWS or
            position in self.snake_set or
            (self.USE_OBSTACLES and position == self.obstacle)
        )
    
    def preprocess_sensory_functions(self):
        """Preprocesses and stores the sensory functions in a dictionary."""
        sensory_functions = {}
        for feature in self.INPUT_FEATURES:
            if feature == 'relative_wall':
                sensory_functions[feature] = self.get_relative_distance_to_wall
            elif feature == 'binary_wall':
                sensory_functions[feature] = self.get_binary_distance_to_wall
            elif feature == 'relative_body':
                sensory_functions[feature] = self.get_relative_distance_to_body
            elif feature == 'binary_body':
                sensory_functions[feature] = self.get_binary_distance_to_body
            elif feature == 'relative_apple':
                sensory_functions[feature] = self.get_relative_distance_to_apple
            elif feature == 'binary_apple':
                sensory_functions[feature] = self.get_binary_distance_to_apple
            elif feature == 'relative_obstacle' and self.USE_OBSTACLES:
                sensory_functions[feature] = self.get_relative_distance_to_obstacle
            elif feature == 'binary_obstacle' and self.USE_OBSTACLES:
                sensory_functions[feature] = self.get_binary_distance_to_obstacle
        
        return sensory_functions
    
    def sensory_function(self):
        """Returns the sensory input for the neural network."""
        x, y = self.snake[-1]
        sensory_input = []
        
        for feature, func in self.sensory_functions.items():
            info = func(x, y)
            if self.INPUT_FRAME_OF_REFERENCE == 'SNAKE':
                direction = (self.v_x, self.v_y)
                info = self.convert_to_snake(info, direction)
            sensory_input.extend(info)

        if 'history' in self.INPUT_FEATURES:
            sensory_input.extend(reversed(self.encoded_history))

        if self.INPUT_FRAME_OF_REFERENCE == 'SNAKE' and self.OUTPUT_FRAME_OF_REFERENCE == 'SNAKE' and self.USE_DUMMY_INPUTS:
            sensory_input.extend(self.get_dummy_inputs())

        return np.array(sensory_input, dtype=float)

    def convert_to_snake(self, info, direction):
        """Converts the sensory information from NSWE frame to snake's frame of reference."""
        if direction == Direction.NORTH.value:  # Facing north
            return [info[0], info[2], info[3]]  # [north, east, west]
        elif direction == Direction.SOUTH.value:  # Facing south
            return [info[1], info[3], info[2]]  # [south, west, east]
        elif direction == Direction.WEST.value:  # Facing west
            return [info[3], info[1], info[0]]  # [west, south, north]
        elif direction == Direction.EAST.value:  # Facing east
            return [info[2], info[0], info[1]]  # [east, north, south]

        
    def get_relative_distance_to_wall(self, x, y):
        """Returns the relative distance to the wall in the north, south, east, and west directions."""
        return [1 / (y + 1), # North
                1 / (self.NUM_ROWS - y), # South
                1 / (self.NUM_COLS - x), # East
                1 / (x + 1)]  # West
    
    def get_binary_distance_to_wall(self, x, y):
        """Return 1 if the snake head is directly adjacent to the wall in the north, south, east, and west directions, otherwise 0."""
        return [1 if (y == 0) else 0, # North
                1 if (y == self.NUM_ROWS - 1) else 0, # South
                1 if (x == self.NUM_COLS - 1) else 0, # East
                1 if (x == 0) else 0]
    
    def get_relative_distance_to_body(self, x, y):
        """Returns the relative distance of the closest body part in the north, south, east, and west directions."""
        dist_to_body = [0, 0, 0, 0]
        for (body_x, body_y) in self.snake[:-1]: 
            if body_x == x:
                if body_y < y: # Body is north
                    dist_to_body[0] = min(dist_to_body[0], 1 / (y - body_y + 1))
                else: # Body is south
                    dist_to_body[1] = min(dist_to_body[1], 1 / (body_y - y + 1))
            elif body_y == y:
                if body_x > x: # Body is east
                    dist_to_body[2] = 1 / (body_x - x + 1) if dist_to_body[2] == 0 else min(dist_to_body[2], 1 / (body_x - x + 1))
                else: # Body is west
                    dist_to_body[3] = 1 / (x - body_x + 1) if dist_to_body[3] == 0 else min(dist_to_body[3], 1 / (x - body_x + 1))
        return dist_to_body
    
    def get_binary_distance_to_body(self, x, y):
        """Returns the binary distance to the body in the north, south, east, and west directions."""
        body_info = [0, 0, 0, 0]
        for (body_x, body_y) in self.snake[:-1]:
            if body_x == x:
                if body_y < y:
                    body_info[0] = 1 # North
                elif body_y > y:
                    body_info[1] = 1 # South
            elif body_y == y:
                if body_x > x:
                    body_info[2] = 1 # East
                elif body_x < x:
                    body_info[3] = 1 # West
        return body_info

    def get_relative_distance_to_obstacle(self, x, y):
        """Returns the relative distance to the obstacle in the north, south, east, and west directions."""
        dist_to_obstacle = [0, 0, 0, 0]
        obstacle_x, obstacle_y = self.obstacle
        if obstacle_x == x:
            if obstacle_y < y:
                dist_to_obstacle[0] = 1 / (y - obstacle_y + 1)
            else:
                dist_to_obstacle[1] = 1 / (obstacle_y - y + 1)
        elif obstacle_y == y:
            if obstacle_x > x:
                dist_to_obstacle[2] = 1 / (obstacle_x - x + 1)
            else:
                dist_to_obstacle[3] = 1 / (x - obstacle_x + 1)
        return dist_to_obstacle

    def get_binary_distance_to_obstacle(self, x, y):
        """Returns the binary distance to the obstacle in the north, south, east, and west directions."""
        obstacle_info = [0, 0, 0, 0]
        obstacle_x, obstacle_y = self.obstacle
        if obstacle_x == x:
            if obstacle_y < y:
                obstacle_info[0] = 1
            elif obstacle_y > y:
                obstacle_info[1] = 1
        elif obstacle_y == y:
            if obstacle_x > x:
                obstacle_info[2] = 1
            elif obstacle_x < x:
                obstacle_info[3] = 1
        return obstacle_info
    
    def get_relative_distance_to_apple(self, x, y):
        dist_to_apple = [0, 0, 0, 0]
        apple_x, apple_y = self.apple
        if apple_x == x:
            if apple_y < y: # Apple is north
                dist_to_apple[0] = 1 / (y - apple_y + 1)
            else: # Apple is south
                dist_to_apple[1] = 1 / (apple_y - y + 1)
        elif apple_y  == y:
            if apple_x > x: # Apple is east
                dist_to_apple[2] = 1 / (apple_x - x + 1)
            else: # Apple is west
                dist_to_apple[3] = 1 / (x - apple_x + 1)
        return dist_to_apple

    def get_binary_distance_to_apple(self, x, y):
        dist_to_apple = [0, 0, 0, 0]
        apple_x, apple_y = self.apple
        if apple_x == x:
            if apple_y < y: # Apple is north
                dist_to_apple[0] = 1
            else: # Apple is south
                dist_to_apple[1] = 1
        elif apple_y  == y:
            if apple_x > x: # Apple is east
                dist_to_apple[2] = 1 
            else: # Apple is west
                dist_to_apple[3] = 1
        return dist_to_apple
    
    def get_dummy_inputs(self):
        """Adds an amount of dummy inputs equivalent to the different between NSEW and SNAKE frame of reference."""
        nr = (len(self.INPUT_FEATURES))
        if 'history' in self.INPUT_FEATURES:
            nr -= 1 #Nr of history nodes is not affected by frame of reference
        return [0.5] * nr
    
    def update_encoded_history(self, action):
        """Updates the encoded move history with the current move."""
        if len(self.encoded_history) >= self.HISTORY_LENGTH:
            self.encoded_history.pop(0)
        self.encoded_history.append(1 / (action + 1))

    
    #===================== ANIMATION STUFF =====================
    def get_feed_forward_layers(self, inputs, connections, genome):
        required = set(genome.nodes)

        layers = []
        s = set(inputs)
        while True:
            c = set(b for (a, b) in connections if a in s and b not in s)
            t = set()
            for n in c:
                if n in required and all(a in s for (a, b) in connections if b == n):
                    t.add(n)
            if not t:
                break

            layers.append(t)
            s = s.union(t)

        return layers

    def modify_eval_functions(self, net, genome, config):
        connections = [cg.key for cg in genome.connections.values() if cg.enabled]

        layers = self.get_feed_forward_layers(config.genome_config.input_keys, connections, genome)
        node_evals = []
        for layer in layers:
            for node in layer:
                inputs = []
                for conn_key in connections:
                    inode, onode = conn_key
                    if onode == node:
                        cg = genome.connections[conn_key]
                        inputs.append((inode, cg.weight))

                ng = genome.nodes[node]
                aggregation_function = config.genome_config.aggregation_function_defs.get(ng.aggregation)
                activation_function = config.genome_config.activation_defs.get(ng.activation)
                node_evals.append((node, activation_function, aggregation_function, ng.bias, ng.response, inputs))

        net.node_evals = node_evals

    def simulate_animation(self, net, genome, config):
        global screen, font
        self.reset()

        self.modify_eval_functions(net, genome, config)
        has_eval = set(eval[0] for eval in net.node_evals)
        has_input = set(con[1] for con in genome.connections)
        hidden_nodes = [node for node in genome.nodes if not 0 <= node <= 3 and node in has_input and node in has_eval]
        node_centers = self.get_node_centers(net, genome, hidden_nodes)

        # Pygame initialization
        pygame.init()
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.font = pygame.font.SysFont(None, 24)
        screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        STEP = pygame.USEREVENT + 1
        pygame.time.set_timer(STEP, self.INTERVAL)

        font = pygame.font.Font(None, 24)
        running = True
        ts = 0
        while running:
            if self.dead:
                running = False
            if ts - self.last_ate_apple > self.MIN_TIME_TO_EAT_APPLE:
                running = False

            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == STEP:
                    sensory_vector = self.sensory_function()
                    activations = net.activate(sensory_vector)
                    action = np.argmax(activations)
                    self.change_direction(action)
                    apple = self.step(action)
                    if apple:
                        self.last_ate_apple = ts
                    ts += 1

            screen.fill(self.BLACK)
            self.draw_square() 
            self.draw_snake() 
            self.draw_apple() 
            if self.USE_OBSTACLES:
                self.draw_obstacle()
            self.draw_network(net, genome, node_centers, hidden_nodes)
            self.draw_fitness()
            pygame.display.flip()
        
        pygame.display.quit()
        pygame.quit()
        sys.exit()

    def get_node_centers(self, net, genome, hidden_nodes):
        node_centers = {}
        start_y = self.WINDOW_BUFFER
        start_x = self.WINDOW_BUFFER

        for i, input_node in enumerate(net.input_nodes):
            node_centers[input_node] = start_x + 8 * self.NODE_SIZE, start_y + i * 3 * self.NODE_SIZE + 10

        start_x = self.WINDOW_BUFFER + 0.5 * self.NETWORK_WIDTH
        start_y = self.WINDOW_BUFFER + self.NODE_SIZE * 6

        for i, hidden_node in enumerate(hidden_nodes):
            x = start_x + 2 * self.NODE_SIZE if i % 2 == 0 else start_x - 2 * self.NODE_SIZE
            if i == 2: 
                x += self.NODE_SIZE * 3
            node_centers[hidden_node] = x, start_y + i * 5 * self.NODE_SIZE + 10

        start_y = self.WINDOW_BUFFER + 12 * self.NODE_SIZE
        start_x = self.SCREEN_WIDTH - self.GAME_WIDTH - self.WINDOW_BUFFER * 3 - self.NODE_SIZE

        for i, output_node in enumerate(net.output_nodes):
            node_centers[output_node] = start_x - 2 * self.NODE_SIZE, start_y + i * 3 * self.NODE_SIZE + 10

        return node_centers

    def draw_connections(self, first_set, second_set, net, genome, node_centers):
        for first in first_set:
            for second in second_set:
                if (first, second) in genome.connections:
                    start = node_centers[first]
                    stop = node_centers[second]
                    weight = genome.connections[(first, second)].weight
                    color = self.ORANGE if weight >= 0 else self.BLUE

                    surf = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
                    alpha = 255 * (0.3 + net.values[first] * 0.7)
                    pygame.draw.line(surf, color + (alpha,), start, stop, width=5)
                    screen.blit(surf, (0, 0))

    def draw_network(self, net, genome, node_centers, hidden_nodes):
        node_names = {}
        if self.OUTPUT_FRAME_OF_REFERENCE == 'NSEW':
            node_names.update({0: 'North', 1: 'South', 2: 'East', 3: 'West'})
        if self.OUTPUT_FRAME_OF_REFERENCE == 'SNAKE':
            if self.USE_DUMMY_INPUTS:
                node_names.update({0: 'Straight', 1: 'Left', 2: 'Right', 3: 'Dummy'})
            else:
                node_names.update({0: 'Straight', 1: 'Left', 2: 'Right'})
        
        if self.INPUT_FRAME_OF_REFERENCE == 'NSEW':
            for i, feature in enumerate(self.INPUT_FEATURES):
                if feature == 'relative_wall' or feature == 'binary_wall':
                    node_names.update({
                        -i*4-1: "Wall_N",
                        -i*4-2: "Wall_S",
                        -i*4-3: "Wall_E",
                        -i*4-4: "Wall_W"
                    })
                elif feature == 'relative_body' or feature == 'binary_body' or feature == 'combined_obstacle_body' or feature == 'nearest_body_direction':
                    node_names.update({
                        -i*4-1: "Body_N",
                        -i*4-2: "Body_S",
                        -i*4-3: "Body_E",
                        -i*4-4: "Body_W"
                    })
                elif feature == 'relative_apple' or feature == 'binary_apple':
                    node_names.update({
                        -i*4-1: "Apple_N",
                        -i*4-2: "Apple_S",
                        -i*4-3: "Apple_E",
                        -i*4-4: "Apple_W"
                    })
                elif feature == 'binary_obstacle' or feature == 'relative_obstacle':
                    node_names.update({
                        -i*4-1: "Obst_N",
                        -i*4-2: "Obst_S",
                        -i*4-3: "Obst_E",
                        -i*4-4: "Obst_W"
                    })
                elif feature == 'history':
                    for j in range(self.HISTORY_LENGTH):
                        node_names.update({
                            -i*4-j-1: f"History_{j}"
                        })

        elif self.INPUT_FRAME_OF_REFERENCE == 'SNAKE':
            for i, feature in enumerate(self.INPUT_FEATURES):
                if feature == 'relative_wall' or feature == 'binary_wall':
                    node_names.update({
                        -i*3-1: "Wall_F",
                        -i*3-2: "Wall_L",
                        -i*3-3: "Wall_R"
                    })
                elif feature == 'binary_body' or feature == 'relative_body' or feature == 'combined_obstacle_body':
                    node_names.update({
                        -i*3-1: "Body_F",
                        -i*3-2: "Body_L",
                        -i*3-3: "Body_R"
                    })
                elif feature == 'binary_apple' or feature == 'relative_apple':
                    node_names.update({
                        -i*3-1: "Apple_F",
                        -i*3-2: "Apple_L",
                        -i*3-3: "Apple_R"
                    })
                elif feature == 'binary_obstacle' or feature == 'relative_obstacle':
                    node_names.update({
                        -i*3-1: "Obst_F",
                        -i*3-2: "Obst_L",
                        -i*3-3: "Obst_R"
                    })
                elif feature == 'nearest_body_direction':
                    node_names.update({
                        -i*3-1: "Body_F",
                        -i*3-2: "Body_L",
                        -i*3-3: "Body_R"
                    })
                elif feature == 'history':
                        for j in range(self.HISTORY_LENGTH):
                            node_names.update({
                                -i*3-j-1: f"History_{j}"
                            })
            if self.USE_DUMMY_INPUTS:
                nr = (len(self.INPUT_FEATURES))
                if 'history' in self.INPUT_FEATURES:
                    nr -= 1
                for j in range(nr):
                    node_names.update({
                        -(i+1)*3-j-1: f"Dummy_{j}"
                    })

        self.draw_connections(net.input_nodes, net.output_nodes, net, genome, node_centers)
        self.draw_connections(net.input_nodes, hidden_nodes, net, genome, node_centers)
        self.draw_connections(hidden_nodes, hidden_nodes, net, genome, node_centers)
        self.draw_connections(hidden_nodes, net.output_nodes, net, genome, node_centers)

        for i, input_node in enumerate(net.input_nodes):
            center = node_centers[input_node]
            center2 = center[0] - 5.5 * self.NODE_SIZE, center[1] - 10
            img = font.render(node_names[input_node], True, self.WHITE)
            screen.blit(img, center2)
            color = (net.values[input_node] * 255, 0, 0)
            pygame.draw.circle(screen, color, center, self.NODE_SIZE)
            pygame.draw.circle(screen, self.WHITE, center, self.NODE_SIZE, width=1)

        for i, output_node in enumerate(net.output_nodes):
            center = node_centers[output_node]
            color = (net.values[output_node] * 255, 0, 0)
            pygame.draw.circle(screen, color, center, self.NODE_SIZE)
            pygame.draw.circle(screen, self.WHITE, center, self.NODE_SIZE, width=1)
            center2 = center[0] + 1.5 * self.NODE_SIZE, center[1] - 10
            img = font.render(node_names[output_node], True, self.WHITE)
            screen.blit(img, center2)

        for hidden in hidden_nodes:
            center = node_centers[hidden]
            color = (net.values[hidden] * 255, 0, 0)
            center2 = center[0] - 5.5 * self.NODE_SIZE, center[1] - 10
            img = font.render(str(hidden), True, self.WHITE)
            screen.blit(img, center2)
            pygame.draw.circle(screen, color, center, self.NODE_SIZE)
            pygame.draw.circle(screen, self.WHITE, center, self.NODE_SIZE, width=1)

    def draw_snake(self):
        for i, (x, y) in enumerate(self.snake):
            rect = pygame.Rect(self.getLeftTop(x, y), (self.BLOCK_WIDTH - self.BUFFER * 2, self.BLOCK_HEIGHT - self.BUFFER * 2))
            pygame.draw.rect(screen, self.YELLOW if i == len(self.snake) - 1 else self.WHITE, rect)

    def draw_square(self):
        rect = pygame.Rect((self.GAME_TOP_LEFT[0] - self.BUFFER, self.GAME_TOP_LEFT[1] - self.BUFFER), 
                           (self.GAME_WIDTH + 2 * self.BUFFER, self.GAME_HEIGHT + 2 * self.BUFFER))
        pygame.draw.rect(screen, self.WHITE, rect, width=self.BUFFER // 2)

    def getLeftTop(self, x, y):
        return (x / self.NUM_ROWS) * self.GAME_WIDTH + self.BUFFER + self.GAME_TOP_LEFT[0], (y / self.NUM_ROWS) * self.GAME_HEIGHT + self.BUFFER + self.GAME_TOP_LEFT[1]

    def draw_apple(self):
        x, y = self.apple
        rect = pygame.Rect(self.getLeftTop(x, y), (self.BLOCK_WIDTH - self.BUFFER * 2, self.BLOCK_HEIGHT - self.BUFFER * 2))
        pygame.draw.rect(screen, self.RED, rect)

    def draw_obstacle(self):
        x, y = self.obstacle
        pygame.draw.rect(screen, self.ORANGE, pygame.Rect(self.getLeftTop(x, y), (self.BLOCK_WIDTH - self.BUFFER * 2, self.BLOCK_HEIGHT - self.BUFFER * 2)))

    def draw_fitness(self):
        fitness_text = font.render(f"Fitness: {self.apples_eaten}", True, self.WHITE)
        screen.blit(fitness_text, (self.SCREEN_WIDTH - 130, 30))

        
    #===================== NEAT STUFF =====================
    def eval_genome(self, genome, config): 
        """
        Fitness function to evaluate single genome, used with ParallelEvaluator.
        """
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness = self.simulate_headless(net)  # Evaluate the genome in a headless simulation.
        return fitness                          # For the ParallelEvaluator to work, the fitness must be returned.


    def eval_genomes(self, genomes, config):
        """
        Fitness function used to assign fitness to all genomees. This is different from eval_genome in that 
        it does not use the ParallelEvaluator and thus goes through each genome in the population one by one.

        Args:
        genomes (list of tuples): List of (genome_id, genome) tuples.
        config (neat.Config): NEAT configuration settings for network creation.
        """
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config) # Create a neural network from the genome.
            fitness = self.simulate_headless(net)  # Evaluate the genome in a headless simulation.
            genome.fitness = fitness  # Assign the fitness to the genome.


    def run_NEAT(self, config_file):
        """
        Run the NEAT algorithm to find the best performing genome.
        """
        # Load configuration into a NEAT object.
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_file)

        #Set results directory
        if not os.path.exists(f"{Paths.RESULTS_PATH}/checkpoints"):
            os.makedirs(f"{Paths.RESULTS_PATH}/checkpoints")

        # Create the population
        p = neat.Population(config)

        # Add statistics reporter to the population
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)

        # Show progress in the terminal and add a checkpointer
        p.add_reporter(neat.StdOutReporter(True))
        p.add_reporter(neat.Checkpointer(generation_interval = self.CHECKPOINT_INTERVAL, filename_prefix=f"{Paths.RESULTS_PATH}/checkpoints/population-"))

        # Add a parallel evaluator to evaluate the population in parallel.
        parallel_evaluator = neat.ParallelEvaluator(multiprocessing.cpu_count(), self.eval_genome) #parallelized fitness function

        # Run the NEAT algorithm for n generations
        winner = p.run(parallel_evaluator.evaluate, n=self.N_GENERATIONS)  

        # Visualize statistics and species progression over generations.
        plot_stats(stats, ylog=False, view=True, filename=f"{Paths.RESULTS_PATH}/fitness_graph.png")
        plot_species(stats, view=False, filename=f"{Paths.RESULTS_PATH}/species_graph.png")

        # Save the winner.
        with open('results/winner_genome', 'wb') as f:
            pickle.dump(winner, f)
        
        return winner, stats

    def run_NEAT_repeated(self, config_file):
        """Runs multiple instances of the NEAT algorithm and returns the winners and statistics of each run."""
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_file)

        # Ensure the results directory exists.
        if not os.path.exists(f"{Paths.RESULTS_PATH}/checkpoints"):
            os.makedirs(f"{Paths.RESULTS_PATH}/checkpoints")

        #Clear the output directory
        shutil.rmtree(Paths.RESULTS_PATH)

        stats_list = []

        for i in range(self.N_RUNS):  # Run the NEAT algorithm n times
            print(f"Running NEAT algorithm, run {i}")
            p = neat.Population(config)
            stats = neat.StatisticsReporter()
            p.add_reporter(stats)

            if not os.path.exists(f"{Paths.RESULTS_PATH}/checkpoints/run{i}"):
                os.makedirs(f"{Paths.RESULTS_PATH}/checkpoints/run{i}")
            p.add_reporter(neat.Checkpointer(self.CHECKPOINT_INTERVAL, filename_prefix=f"{Paths.RESULTS_PATH}/checkpoints/run{i}/population-"))

            parallel_evaluator = neat.ParallelEvaluator(multiprocessing.cpu_count(), self.eval_genome)
            winner = p.run(parallel_evaluator.evaluate, n = self.N_GENERATIONS)
            # winner = p.run(self.eval_genomes, n = self.N_GENERATIONS)
            stats_list.append(stats)

            # Save the winner of each run
            with open(f'{Paths.RESULTS_PATH}/checkpoints/run{i}/winner_genome', 'wb') as f:
                pickle.dump(winner, f)

            #Print results
            print(f"Run {i} completed, best fitness: {winner.fitness}")

        return stats_list

    def test_winner(self, genome, config_path):
        """Visualizes the genome passed playing the snake game"""

        # Load configuration into a NEAT object.
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_path)
        
        net = neat.nn.FeedForwardNetwork.create(genome, config) # Initialize the neural network from the passed genome.s

        # run the simulation
        self.simulate_animation(net, genome, config) # Simulate the environment with a GUI.


def plot_stats(statistics, ylog=False, view=False, filename='avg_fitness.svg'):
    """ Plots the population's average and best fitness. """
    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = np.array(statistics.get_fitness_mean())
    stdev_fitness = np.array(statistics.get_fitness_stdev())

    plt.plot(generation, avg_fitness, 'b-', label="average")
    plt.plot(generation, avg_fitness - stdev_fitness, 'g-.', label="-1 sd")
    plt.plot(generation, avg_fitness + stdev_fitness, 'g-.', label="+1 sd")
    plt.plot(generation, best_fitness, 'r-', label="best")

    plt.title("Population's average and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        plt.gca().set_yscale('symlog')

    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()



class Paths:
    RESULTS_PATH = 'results/'      # PATH to the results directory
    CONFIG_PATH = 'config.py'  # Path to the configuration file
    NEAT_CONFIG_PATH = 'config-neat'  # Path to the NEAT configuration file
    DRAW_NET_PATH = 'target_pursuit_2000_results/winner-feedforward.gv'  # Path to the neural network visualization
    WINNER_PATH = 'results/winner-feedforward'  # Path to the winner genome



def plot_species(statistics, view=False, filename='speciation.svg'):
    """ Visualizes speciation throughout evolution. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    species_sizes = statistics.get_species_sizes()
    num_generations = len(species_sizes)
    curves = np.array(species_sizes).T

    fig, ax = plt.subplots()
    ax.stackplot(range(num_generations), *curves)

    plt.title("Speciation")
    plt.ylabel("Size per Species")
    plt.xlabel("Generations")

    plt.savefig(filename)

    if view:
        plt.show()

    plt.close()


#Function to update configurtion
def update_config(file_path, section, variables):
    config = configparser.ConfigParser()
    config.read(file_path)
    for key, value in variables.items():
        config.set(section, key, value)
    with open(file_path, 'w') as configfile:
        config.write(configfile)