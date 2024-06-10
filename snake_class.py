import numpy as np
import pygame
from random import randint, seed, choice
from config import *

seed(42)

class SnakeGame:
    """Class for the Snake game."""
    def __init__(self, 
                 input_features = ['wall', 'relative_body', 'relative_apple', 'relative_obstacle'], 
                 input_frame_of_reference = 'nswe', output_frame_of_reference = 'snake', 
                 history_length = 3, 
                 use_obstacles = True,
                 fitness_iters = 10):
        """Initializes the game with default parameters."""

        # Game parameters
        self.INPUT_FEATURES = input_features
        self.INPUT_FRAME_OF_REFERENCE = input_frame_of_reference 
        self.OUTPUT_FRAME_OF_REFERENCE = output_frame_of_reference
        self.HISTORY_LENGTH = history_length
        self.USE_OBSTACLES =  use_obstacles
        self.FITNESS_ITERS = fitness_iters
        self.MIN_TIME_TO_EAT_APPLE = 100

        #Compute input and output size based on features and frame of reference
        if self.INPUT_FRAME_OF_REFERENCE == 'nswe':
            self.NR_INPUT_FEATURES = 4 * len(self.INPUT_FEATURES)
        elif self.INPUT_FRAME_OF_REFERENCE == 'snake':
            self.NR_INPUT_FEATURES = 3 * len(self.INPUT_FEATURES)
        if 'history' in self.INPUT_FEATURES:
            self.NR_INPUT_FEATURES += self.HISTORY_LENGTH - 1
        if self.OUTPUT_FRAME_OF_REFERENCE == 'nswe':
            self.NR_OUTPUT_FEATURES = 4
        elif self.OUTPUT_FRAME_OF_REFERENCE == 'snake':
            self.NR_OUTPUT_FEATURES = 3

        
        #create a object to update neat parameters with later
        self.neat_params = {
            'input_output': {
                'DefaultGenome': {
                    'num_inputs': f'{self.NR_INPUT_FEATURES}',
                    'num_outputs': f'{self.NR_OUTPUT_FEATURES}'
                }
            },
        }


        # Set constants
        self.NUM_ROWS = 10
        self.NUM_COLS = 10
        self.snake = []
        self.snake_set = set()
        self.apple = ()
        self.apples_eaten = 0
        self.obstacle = ()
        self.dead = False
        self.step_count = 0
        self.v_x, self.v_y = 1, 0
        self.movement_history = []
        self.reset()

        # Animation parameters
        self.INTERVAL = 300
        self.NODE_SIZE = 280 / self.NR_INPUT_FEATURES
        if self.NODE_SIZE > 20:
            self.NODE_SIZE = 20
        self.NETWORK_WIDTH, self.NETWORK_HEIGHT = 700, 900
        self.GAME_WIDTH, self.GAME_HEIGHT = 700, 700
        self.WINDOW_BUFFER = 25
        self.SCREEN_WIDTH = self.WINDOW_BUFFER + self.NETWORK_WIDTH + self.WINDOW_BUFFER + self.GAME_WIDTH + self.WINDOW_BUFFER
        self.SCREEN_HEIGHT = self.NETWORK_HEIGHT + 2 * self.WINDOW_BUFFER
        self.BLOCK_WIDTH = self.GAME_WIDTH / self.NUM_COLS
        self.BLOCK_HEIGHT = self.GAME_HEIGHT / self.NUM_ROWS
        self.GAME_TOP_LEFT = (2 * self.WINDOW_BUFFER + self.NETWORK_WIDTH, self.WINDOW_BUFFER)
        self.BUFFER = 4

        self.RED = (235, 64, 52)
        self.WHITE = (245, 245, 245)
        self.YELLOW = (252, 227, 109)
        self.BLACK = (20, 20, 20)
        self.BLUE = (58, 117, 196)
        self.ORANGE = (255, 140, 0)

        self.screen = None
        self.font = None


    def reset(self):
        """Resets the game to its initial state."""
        self.snake = [(randint(1, self.NUM_COLS - 2), randint(1, self.NUM_ROWS - 2))]
        self.snake_set = set(self.snake)
        self.apple = self._get_random_position(exclude=self.snake_set)
        self.obstacle = self._get_random_position(exclude=self.snake_set | {self.apple}) if self.USE_OBSTACLES else ()
        self.v_x, self.v_y = choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
        self.apples_eaten = 0
        self.step_count = 0
        self.dead = False

    def _get_random_position(self, exclude=set()):
        """Returns a random position that is not in the exclude set."""
        pos = (randint(0, self.NUM_COLS - 1), randint(0, self.NUM_ROWS - 1))
        while pos in exclude:
            pos = (randint(0, self.NUM_COLS - 1), randint(0, self.NUM_ROWS - 1))
        return pos

    def simulate_headless(self, net):
        """Simulates the game without rendering it and returns the fitness score."""
        sensory_function = self.create_sensory_function(self.INPUT_FEATURES)
        scores = []

        for _ in range(self.FITNESS_ITERS):
            self.reset()
            last_ate_apple = 0
            t = 0

            while not self.dead and (t - last_ate_apple <= self.MIN_TIME_TO_EAT_APPLE):
                sensory_vector = sensory_function()
                activations = net.activate(sensory_vector)
                action = np.argmax(activations)
                self.change_direction(action)
                apple_eaten = self.step()
                t += 1

                if apple_eaten:
                    last_ate_apple = t

            scores.append(len(self.snake))

        return np.mean(scores)

    def change_direction(self, action):
        """Changes the direction of the snake based on the action (output of the neural network)."""
        if self.OUTPUT_FRAME_OF_REFERENCE == 'nswe':
            directions = [
                (0, -1),  # North
                (0, 1),   # South
                (1, 0),   # East
                (-1, 0)   # West
            ]
            self.v_x, self.v_y = directions[action]

        elif self.OUTPUT_FRAME_OF_REFERENCE == 'snake':
            if action == 1:
                self.v_x, self.v_y = -self.v_y, self.v_x
            elif action == 2:
                self.v_x, self.v_y = self.v_y, -self.v_x
            # 0 does nothing

    def step(self):
        """Moves the snake one step forward and checks for collisions and apple eating"""
        x, y = self.snake[-1]
        new_head = (x + self.v_x, y + self.v_y)
        self.snake.append(new_head)
        self.snake_set.add(new_head)

        if self._is_collision(new_head):
            self.dead = True

        ate_apple = new_head == self.apple
        if ate_apple:
            self.apple = self._get_random_position(exclude=self.snake_set | {self.obstacle})
            if self.USE_OBSTACLES:
                self.obstacle = self._get_random_position(exclude=self.snake_set | {self.apple})
        else:
            tail = self.snake.pop(0)
            self.snake_set.remove(tail)

        self.step_count += 1
        return ate_apple

    def _is_collision(self, position):
        """Returns True if the position is a collision (with the wall, snake, or obstacle)."""
        x, y = position
        return (x < 0 or x >= self.NUM_COLS or y < 0 or y >= self.NUM_ROWS or
                position in self.snake_set or (self.USE_OBSTACLES and position == self.obstacle))

    def create_sensory_function(self, input_features):
        """Creates a function that returns the sensory input for the neural network."""
        def sensory_function():
            """Returns the sensory input for the neural network."""
            x, y = self.snake[-1]
            sensory_input = []

            for feature in input_features:
                if self.INPUT_FRAME_OF_REFERENCE == 'nswe':
                    if feature == 'wall':
                        sensory_input.extend(self.get_wall_info(x, y))
                    elif feature == 'relative_body':
                        sensory_input.extend(self.get_body_info(x, y))
                    elif feature == 'binary_body':
                        sensory_input.extend(self.get_relative_distance_to_body(x, y))
                    elif feature == 'combined_obstacle_body':
                        sensory_input.extend(self.get_combined_obstacle_body_info(x, y))
                    elif feature == 'relative_apple':
                        sensory_input.extend(self.get_relative_distance_to_apple(x, y))
                    elif feature == 'binary_apple':
                        sensory_input.extend(self.get_apple_info(x, y))
                    elif feature == 'binary_obstacle':
                        sensory_input.extend(self.get_obstacle_info(x, y))
                    elif feature == 'relative_obstacle':
                        sensory_input.extend(self.get_relative_distance_to_obstacle(x, y))
                    elif feature == 'nearest_body_direction':
                        sensory_input.extend(self.get_direction_of_nearest_body_segment(x, y))
                elif self.INPUT_FRAME_OF_REFERENCE == 'snake':
                    if feature == 'wall':
                        sensory_input.extend(self.get_wall_info_snake(x, y))
                    elif feature == 'body':
                        sensory_input.extend(self.get_body_info_snake(x, y))
                    elif feature == 'relative_body':
                        sensory_input.extend(self.get_relative_distance_to_body_snake(x, y))
                    elif feature == 'combined_obstacle_body':
                        sensory_input.extend(self.get_combined_obstacle_body_info_snake(x, y))
                    elif feature == 'apple':
                        sensory_input.extend(self.get_apple_info_snake(x, y))
                    elif feature == 'relative_apple':
                        sensory_input.extend(self.get_relative_distance_to_apple_snake(x, y))
                    elif feature == 'obstacle':
                        sensory_input.extend(self.get_obstacle_info_snake(x, y))
                    elif feature == 'relative_obstacle':
                        sensory_input.extend(self.get_relative_distance_to_obstacle_snake(x, y))
                    elif feature == 'nearest_body_direction':
                        sensory_input.extend(self.get_direction_of_nearest_body_segment_snake(x, y))

            return np.array(sensory_input, dtype=float)

        return sensory_function

    def get_wall_info(self, x, y):
        return [1 / (y + 1), 1 / (self.NUM_ROWS - y), 1 / (self.NUM_COLS - x), 1 / (x + 1)]

    def get_body_info(self, x, y):
        body_info = [0, 0, 0, 0]
        for (body_x, body_y) in self.snake[:-1]:
            if body_x == x:
                if body_y < y:
                    body_info[0] = 1
                elif body_y > y:
                    body_info[1] = 1
            elif body_y == y:
                if body_x > x:
                    body_info[2] = 1
                elif body_x < x:
                    body_info[3] = 1
        return body_info

    def get_relative_distance_to_body(self, x, y):
        dist_to_body = [0, 0, 0, 0]
        for (body_x, body_y) in self.snake[:-1]:
            if body_x == x:
                if body_y > y:
                    dist_to_body[1] = max(dist_to_body[1], 1 / (body_y - y + 1))
                else:
                    dist_to_body[0] = max(dist_to_body[0], 1 / (y - body_y + 1))
            elif body_y == y:
                if body_x > x:
                    dist_to_body[2] = max(dist_to_body[2], 1 / (body_x - x + 1))
                else:
                    dist_to_body[3] = max(dist_to_body[3], 1 / (x - body_x + 1))
        return dist_to_body

    def get_obstacle_info(self, x, y):
        obstacle_info = [0, 0, 0, 0]
        if self.obstacle[0] == x:
            if self.obstacle[1] < y:
                obstacle_info[0] = 1
            elif self.obstacle[1] > y:
                obstacle_info[1] = 1
        elif self.obstacle[1] == y:
            if self.obstacle[0] > x:
                obstacle_info[2] = 1
            elif self.obstacle[0] < x:
                obstacle_info[3] = 1
        return obstacle_info

    def get_relative_distance_to_obstacle(self, x, y):
        dist_to_obstacle = [0, 0, 0, 0]
        if self.obstacle[0] == x:
            if self.obstacle[1] > y:
                dist_to_obstacle[1] = 1 / (self.obstacle[1] - y + 1)
            else:
                dist_to_obstacle[0] = 1 / (y - self.obstacle[1] + 1)
        elif self.obstacle[1] == y:
            if self.obstacle[0] > x:
                dist_to_obstacle[2] = 1 / (self.obstacle[0] - x + 1)
            else:
                dist_to_obstacle[3] = 1 / (x - self.obstacle[0] + 1)
        return dist_to_obstacle

    def get_combined_obstacle_body_info(self, x, y):
        combined_info = [0, 0, 0, 0]
        dist_to_obstacle = self.get_relative_distance_to_obstacle(x, y)
        dist_to_body = self.get_relative_distance_to_body(x, y)
        for i in range(4):
            combined_info[i] = max(dist_to_body[i], dist_to_obstacle[i])
        return combined_info

    def get_apple_info(self, x, y):
        apple_info = [0, 0, 0, 0]
        if self.apple[0] == x:
            if self.apple[1] < y:
                apple_info[0] = 1
            elif self.apple[1] > y:
                apple_info[1] = 1
        elif self.apple[1] == y:
            if self.apple[0] > x:
                apple_info[2] = 1
            elif self.apple[0] < x:
                apple_info[3] = 1
        return apple_info

    def get_relative_distance_to_apple(self, x, y):
        dist_to_apple = [0, 0, 0, 0]
        if self.apple[0] == x:
            if self.apple[1] > y:
                dist_to_apple[1] = 1 / (self.apple[1] - y + 1)
            else:
                dist_to_apple[0] = 1 / (y - self.apple[1] + 1)
        elif self.apple[1] == y:
            if self.apple[0] > x:
                dist_to_apple[2] = 1 / (self.apple[0] - x + 1)
            else:
                dist_to_apple[3] = 1 / (x - self.apple[0] + 1)
        return dist_to_apple

    def get_direction_of_nearest_body_segment(self, x, y):
        nearest_body_dist = float('inf')
        nearest_body_direction = [0, 0, 0, 0]
        for (body_x, body_y) in self.snake[:-1]:
            dist = abs(body_x - x) + abs(body_y - y)
            if dist < nearest_body_dist:
                nearest_body_dist = dist
                if body_x == x:
                    if body_y < y:
                        nearest_body_direction = [1, 0, 0, 0]
                    else:
                        nearest_body_direction = [0, 1, 0, 0]
                elif body_y == y:
                    if body_x > x:
                        nearest_body_direction = [0, 0, 1, 0]
                    else:
                        nearest_body_direction = [0, 0, 0, 1]
        return nearest_body_direction
    
    def convert_to_snake(self, info):
        """Converts the sensory information to the snake's frame of reference."""
        if (self.v_x, self.v_y) == (1, 0):  # Facing right
            return info
        elif (self.v_x, self.v_y) == (-1, 0):  # Facing left
            return info[2:] + info[:2]
        elif (self.v_x, self.v_y) == (0, 1):  # Facing down
            return info[1:2] + info[:1] + info[3:] + info[2:3]
        elif (self.v_x, self.v_y) == (0, -1):  # Facing up
            return info[3:] + info[:3]


    def get_wall_info_snake(self, x, y):
        wall_info = self.get_wall_info(x, y)
        return self.convert_to_snake(wall_info)

    def get_body_info_snake(self, x, y):
        body_info = self.get_body_info(x, y)
        return self.convert_to_snake(body_info)

    def get_snake_distance_to_body_snake(self, x, y):
        rel_body_info = self.get_snake_distance_to_body(x, y)
        return self.convert_to_snake(rel_body_info)

    def get_combined_obstacle_body_info_snake(self, x, y):
        comb_info = self.get_combined_obstacle_body_info(x, y)
        return self.convert_to_snake(comb_info)

    def get_apple_info_snake(self, x, y):
        apple_info = self.get_apple_info(x, y)
        return self.convert_to_snake(apple_info)

    def get_relative_distance_to_apple_snake(self, x, y):
        rel_apple_info = self.get_relative_distance_to_apple(x, y)
        return self.convert_to_snake(rel_apple_info)

    def get_obstacle_info_snake(self, x, y):
        obstacle_info = self.get_obstacle_info(x, y)
        return self.convert_to_snake(obstacle_info)

    def get_relative_distance_to_obstacle_snake(self, x, y):
        rel_obstacle_info = self.get_relative_distance_to_obstacle(x, y)
        return self.convert_to_snake(rel_obstacle_info)

    def get_direction_of_nearest_body_segment_snake(self, x, y):
        nearest_body_dir = self.get_direction_of_nearest_body_segment(x, y)
        return self.convert_to_snake(nearest_body_dir)

    
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
        last_ate_apple = 0
        sensory_function = self.create_sensory_function(self.INPUT_FEATURES)

        self.modify_eval_functions(net, genome, config)
        has_eval = set(eval[0] for eval in net.node_evals)
        has_input = set(con[1] for con in genome.connections)
        hidden_nodes = [node for node in genome.nodes if not 0 <= node <= 3 and node in has_input and node in has_eval]
        node_centers = self.get_node_centers(net, genome, hidden_nodes)

        pygame.init()
        screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        STEP = pygame.USEREVENT + 1
        pygame.time.set_timer(STEP, self.INTERVAL)

        font = pygame.font.Font(None, 24)
        running = True
        ts = 0
        while running:
            if self.dead:
                running = False
                pygame.quit()
            if ts - last_ate_apple > self.MIN_TIME_TO_EAT_APPLE:
                running = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    self.dead = True
                elif event.type == STEP:
                    sensory_vector = sensory_function()
                    activations = net.activate(sensory_vector)
                    action = np.argmax(activations)
                    self.change_direction(action)
                    apple = self.step()
                    if apple:
                        last_ate_apple = ts
                        self.apples_eaten += 1
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
        pygame.quit()

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
                    color = self.BLUE if weight >= 0 else self.ORANGE

                    surf = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
                    alpha = 255 * (0.3 + net.values[first] * 0.7)
                    pygame.draw.line(surf, color + (alpha,), start, stop, width=5)
                    screen.blit(surf, (0, 0))

    def draw_network(self, net, genome, node_centers, hidden_nodes):
        node_names = {}
        
        if self.INPUT_FRAME_OF_REFERENCE == 'nswe':
            node_names.update({0: 'Up', 1: 'Left', 2: 'Down', 3: 'Right'})
            for i, feature in enumerate(self.INPUT_FEATURES):
                if feature == 'wall':
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

        elif self.INPUT_FRAME_OF_REFERENCE == 'snake':
            node_names.update({0: 'Left', 1: 'Straight', 2: 'Right'})
            for i, feature in enumerate(self.INPUT_FEATURES):
                if feature == 'wall':
                    node_names.update({
                        -i*3-1: "Wall_F",
                        -i*3-2: "Wall_L",
                        -i*3-3: "Wall_R"
                    })
                elif feature == 'body' or feature == 'relative_body' or feature == 'combined_obstacle_body':
                    node_names.update({
                        -i*3-1: "Body_F",
                        -i*3-2: "Body_L",
                        -i*3-3: "Body_R"
                    })
                elif feature == 'apple' or feature == 'relative_apple':
                    node_names.update({
                        -i*3-1: "Apple_F",
                        -i*3-2: "Apple_L",
                        -i*3-3: "Apple_R"
                    })
                elif feature == 'obstacle' or feature == 'relative_obstacle':
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


