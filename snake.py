''' code for the baseline snake model '''
import numpy as np
import pygame

from random import randint, seed, choice
from config import *

seed(42)

#===================== GAME PARAMETERS =====================
# Constants
NUM_ROWS, NUM_COLS = 10, 10

# Actions
GO_STRAIGHT = 0
TURN_LEFT = 1
TURN_RIGHT = 2

# Global variables
snake = []
snake_set = set()
apple = ()
apples_eaten = 0
obstacle = ()
dead = False
step_count = 0
v_x, v_y = 1, 0
movement_history = []

def reset():
    """Reset the game environment to the start state."""
    global snake, snake_set, apple, obstacle, v_x, v_y, dead, apples_eaten, step_count

    # Initialize snake (can't start at the edges)
    snake = [(randint(1, NUM_COLS - 2), randint(1, NUM_ROWS - 2))]
    snake_set = set(snake)

    # Initialize apple
    apple = (randint(0, NUM_COLS - 1), randint(0, NUM_ROWS - 1))
    while apple in snake_set:
        apple = (randint(0, NUM_COLS - 1), randint(0, NUM_ROWS - 1))

    # Initialize obstacle
    if Config.USE_OBSTACLES:
        obstacle = (randint(0, NUM_COLS - 1), randint(0, NUM_ROWS - 1))
        while obstacle in snake_set or obstacle == apple:
            obstacle = (randint(0, NUM_COLS - 1), randint(0, NUM_ROWS - 1))

    v_x, v_y = choice([(1, 0), (-1, 0), (0, 1), (0, -1)])  # Random initial direction
    apples_eaten = 0
    step_count = 0
    dead = False

def simulate_headless(net):
    """Simulate the game without GUI to train or evaluate the neural network."""
    sensory_function = create_sensory_function(Config.INPUT_FEATURES)
    scores = []

    for _ in range(Config.FITNESS_ITERS):
        reset()
        last_ate_apple = 0
        t = 0

        while not dead and (t - last_ate_apple <= Config.MIN_TIME_TO_EAT_APPLE):
            sensory_vector = sensory_function()
            activations = net.activate(sensory_vector)
            action = np.argmax(activations)
            change_direction(action)
            apple_eaten = step()
            t += 1

            if apple_eaten:
                last_ate_apple = t

        scores.append(len(snake))

    return np.mean(scores)

def change_direction(action):
    global v_x, v_y

    if Config.OUTPUT_FRAME_OF_REFERENCE == 'nswe':
        assert 0 <= action <= 3
        directions = [
            (0, -1),  # North
            (0, 1),   # South
            (1, 0),   # East
            (-1, 0)   # West
        ]
        v_x, v_y = directions[action]

    elif Config.OUTPUT_FRAME_OF_REFERENCE == 'snake':
        assert 0 <= action <= 2
        if action == TURN_LEFT:
            if v_x == 1 and v_y == 0:    # East to North
                v_x, v_y = 0, -1
            elif v_x == -1 and v_y == 0: # West to South
                v_x, v_y = 0, 1
            elif v_y == 1 and v_x == 0:  # South to East
                v_x, v_y = 1, 0
            elif v_y == -1 and v_x == 0: # North to West
                v_x, v_y = -1, 0
        elif action == GO_STRAIGHT:
            pass  # No change in direction
        elif action == TURN_RIGHT:
            if v_x == 1 and v_y == 0:    # East to South
                v_x, v_y = 0, 1
            elif v_x == -1 and v_y == 0: # West to North
                v_x, v_y = 0, -1
            elif v_y == 1 and v_x == 0:  # South to West
                v_x, v_y = -1, 0
            elif v_y == -1 and v_x == 0: # North to East
                v_x, v_y = 1, 0


def step():
    """Move the snake one step in the game and check for apples eaten or collisions."""
    global apple, dead, obstacle, step_count, apples_eaten, movement_history

    # Update the snake's position
    x, y = snake[-1]
    new_head = (x + v_x, y + v_y)
    snake.append(new_head)
    snake_set.add(new_head)

    # Check for wall collisions
    x, y = new_head
    if x < 0 or x >= NUM_COLS or y < 0 or y >= NUM_ROWS:
        dead = True

    # Check for body collisions
    if len(snake) != len(snake_set):
        dead = True

    # Check for obstacle collisions
    if Config.USE_OBSTACLES:
        if new_head == obstacle:
            dead = True

    # Update apple position and obstacle when apple is eaten
    ate_apple = False
    if new_head == apple:
        apple = (randint(0, NUM_COLS - 1), randint(0, NUM_ROWS - 1))
        while apple in snake_set or apple == obstacle:
            apple = (randint(0, NUM_COLS - 1), randint(0, NUM_ROWS - 1))
        ate_apple = True

        if Config.USE_OBSTACLES:
            obstacle = (randint(0, NUM_COLS - 1), randint(0, NUM_ROWS - 1))
            while obstacle in snake_set or obstacle == apple:
                obstacle = (randint(0, NUM_COLS - 1), randint(0, NUM_ROWS - 1))
    else:
        tail = snake.pop(0)
        snake_set.remove(tail)  # Remove the tail part if no apple was eaten


    step_count += 1
    return ate_apple


#===================== ANIMATION STUFF =====================
# Constants
INTERVAL = 300
NODE_SIZE = 280 / Config.NR_INPUT_FEATURES
if NODE_SIZE > 20:
    NODE_SIZE = 20
NETWORK_WIDTH, NETWORK_HEIGHT = 700, 900
GAME_WIDTH, GAME_HEIGHT = 700, 700
WINDOW_BUFFER = 25
SCREEN_WIDTH = WINDOW_BUFFER + NETWORK_WIDTH + WINDOW_BUFFER + GAME_WIDTH + WINDOW_BUFFER
SCREEN_HEIGHT = NETWORK_HEIGHT + 2 * WINDOW_BUFFER
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

# Global variables
screen = None
font = None


def get_feed_forward_layers(inputs, connections, genome):
  """
  Sort nodes into layers based on their connectivity.
  """
  required = set(genome.nodes)

  layers = []
  s = set(inputs)
  while 1:
      # Find candidate nodes c for the next layer.  These nodes should connect
      # a node in s to a node not in s.
      c = set(b for (a, b) in connections if a in s and b not in s)
      # Keep only the used nodes whose entire input set is contained in s.
      t = set()
      for n in c:
          if n in required and all(a in s for (a, b) in connections if b == n):
              t.add(n)

      if not t:
          break

      layers.append(t)
      s = s.union(t)

  return layers

def modify_eval_functions(net, genome, config):
  """
  Organizes the nodes into layers, gathers their input connections, and prepares the necessary 
  evaluation details (activation functions, aggregation functions, biases, and responses) for each node. 
  """
  # Gather expressed connections.
  connections = [cg.key for cg in genome.connections.values() if cg.enabled]

  layers = get_feed_forward_layers(config.genome_config.input_keys, connections, genome)
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

def simulate_animation(net, genome, config):
    global apples_eaten, screen, font, dead
    reset()
    last_ate_apple = 0
    sensory_function = create_sensory_function(Config.INPUT_FEATURES)

    # Initialize the neural network to visualize
    modify_eval_functions(net, genome, config)
    has_eval = set(eval[0] for eval in net.node_evals)
    has_input = set(con[1] for con in genome.connections)
    hidden_nodes = [node for node in genome.nodes if not 0 <= node <= 3 and node in has_input and node in has_eval]
    node_centers = get_node_centers(net, genome, hidden_nodes)

    # Initialize the pygame window
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    STEP = pygame.USEREVENT + 1
    pygame.time.set_timer(STEP, INTERVAL)

    pygame.init()
    font = pygame.font.Font(None, 24)
    running = True
    ts = 0
    while running:
        if dead:
            running = False
            pygame.quit()
        if ts - last_ate_apple > Config.MIN_TIME_TO_EAT_APPLE:
            running = False

        for event in pygame.event.get():
            if (event.type == pygame.QUIT):
                running = False
                dead = True  # Ensure the loop exits
            elif (event.type == STEP):
                sensory_vector = sensory_function()
                activations = net.activate(sensory_vector)
                action = np.argmax(activations)
                change_direction(action)
                apple = step()
                if apple:
                    last_ate_apple = ts
                    apples_eaten += 1  # Increment apples eaten
                ts += 1  # Increment time step

        screen.fill(BLACK)
        draw_square() 
        draw_snake() 
        draw_apple() 
        if Config.USE_OBSTACLES:
            draw_obstacle()
        draw_network(net, genome, node_centers, hidden_nodes)
        draw_fitness()  
        pygame.display.flip()
    pygame.quit()


def get_node_centers(net, genome, hidden_nodes):
    """Get the center coordinates of each node in the neural network."""
    node_centers = {}

    start_y = WINDOW_BUFFER
    start_x = WINDOW_BUFFER

    for i, input_node in enumerate(net.input_nodes):
        node_centers[input_node] = start_x + 8 * NODE_SIZE, start_y + i * 3 * NODE_SIZE + 10

    start_x = WINDOW_BUFFER + 0.5 * NETWORK_WIDTH
    start_y = WINDOW_BUFFER + NODE_SIZE * 6

    for i, hidden_node in enumerate(hidden_nodes):
        x = start_x + 2 * NODE_SIZE if i % 2 == 0 else start_x - 2 * NODE_SIZE
        if i == 2: 
            x += NODE_SIZE * 3
        node_centers[hidden_node] = x, start_y + i * 5 * NODE_SIZE + 10

    start_y = WINDOW_BUFFER + 12 * NODE_SIZE
    start_x = SCREEN_WIDTH - GAME_WIDTH - WINDOW_BUFFER * 3 - NODE_SIZE

    for i, output_node in enumerate(net.output_nodes):
        node_centers[output_node] = start_x - 2 * NODE_SIZE, start_y + i * 3 * NODE_SIZE + 10

    return node_centers

def draw_connections(first_set, second_set, net, genome, node_centers):
  """Draw connections between nodes in the neural network."""
  for first in first_set:
      for second in second_set:
          if (first, second) in genome.connections:
              start = node_centers[first]
              stop = node_centers[second]
              weight = genome.connections[(first, second)].weight
              color = BLUE if weight >= 0 else ORANGE

              surf = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
              alpha = 255 * (0.3 + net.values[first] * 0.7)
              pygame.draw.line(surf, color + (alpha,), start, stop, width=5)
              screen.blit(surf, (0, 0))

def draw_network(net, genome, node_centers, hidden_nodes):
    """Draw the neural network on the screen."""
    node_names = {}
    
    # Map input features based on the frame of reference
    if Config.INPUT_FRAME_OF_REFERENCE == 'nswe':
        node_names.update({0: 'Up', 1: 'Left', 2: 'Down', 3: 'Right'})
        
        # Add node names based on the input features [in order of the input features list]
        for i, feature in enumerate(Config.INPUT_FEATURES):
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

    elif Config.INPUT_FRAME_OF_REFERENCE == 'snake':
        node_names.update({0: 'Left', 1: 'Straight', 2: 'Right'})
        
        for i, feature in enumerate(Config.INPUT_FEATURES):
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

    # Draw connections between input and output nodes
    draw_connections(net.input_nodes, net.output_nodes, net, genome, node_centers)
    draw_connections(net.input_nodes, hidden_nodes, net, genome, node_centers)
    draw_connections(hidden_nodes, hidden_nodes, net, genome, node_centers)
    draw_connections(hidden_nodes, net.output_nodes, net, genome, node_centers)

    # Draw input nodes
    for i, input_node in enumerate(net.input_nodes):
        center = node_centers[input_node]
        center2 = center[0] - 5.5 * NODE_SIZE, center[1] - 10
        img = font.render(node_names[input_node], True, WHITE)
        screen.blit(img, center2)
        color = (net.values[input_node] * 255, 0, 0)
        pygame.draw.circle(screen, color, center, NODE_SIZE)
        pygame.draw.circle(screen, WHITE, center, NODE_SIZE, width=1)

    # Draw output nodes
    for i, output_node in enumerate(net.output_nodes):
        center = node_centers[output_node]
        color = (net.values[output_node] * 255, 0, 0)
        pygame.draw.circle(screen, color, center, NODE_SIZE)
        pygame.draw.circle(screen, WHITE, center, NODE_SIZE, width=1)
        center2 = center[0] + 1.5 * NODE_SIZE, center[1] - 10
        img = font.render(node_names[output_node], True, WHITE)
        screen.blit(img, center2)

    # Draw hidden nodes
    for hidden in hidden_nodes:
        center = node_centers[hidden]
        color = (net.values[hidden] * 255, 0, 0)
        center2 = center[0] - 5.5 * NODE_SIZE, center[1] - 10
        img = font.render(str(hidden), True, WHITE)
        screen.blit(img, center2)
        pygame.draw.circle(screen, color, center, NODE_SIZE)
        pygame.draw.circle(screen, WHITE, center, NODE_SIZE, width=1)



def draw_snake():
    for i, (x, y) in enumerate(snake):
        rect = pygame.Rect(getLeftTop(x, y), (BLOCK_WIDTH - BUFFER * 2, BLOCK_HEIGHT - BUFFER * 2))
        pygame.draw.rect(screen, YELLOW if i == len(snake) - 1 else WHITE, rect)

def draw_square():
    rect = pygame.Rect((GAME_TOP_LEFT[0] - BUFFER, GAME_TOP_LEFT[1] - BUFFER), 
                       (GAME_WIDTH + 2 * BUFFER, GAME_HEIGHT + 2 * BUFFER))
    pygame.draw.rect(screen, WHITE, rect, width=BUFFER // 2)

def getLeftTop(x, y):
    return (x / NUM_ROWS) * GAME_WIDTH + BUFFER + GAME_TOP_LEFT[0], (y / NUM_ROWS) * GAME_HEIGHT + BUFFER + GAME_TOP_LEFT[1]

def draw_apple():
    x, y = apple
    rect = pygame.Rect(getLeftTop(x, y), (BLOCK_WIDTH - BUFFER * 2, BLOCK_HEIGHT - BUFFER * 2))
    pygame.draw.rect(screen, RED, rect)

def draw_obstacle():
  '''set up an obstacle that the snake also has to go around'''
  x, y = obstacle
  pygame.draw.rect(screen, ORANGE, pygame.Rect(getLeftTop(x, y), (BLOCK_WIDTH - BUFFER * 2, BLOCK_HEIGHT - BUFFER * 2)))

def draw_fitness():
  fitness_text = font.render(f"Fitness: {apples_eaten}", True, WHITE)
  screen.blit(fitness_text, (SCREEN_WIDTH - 130, 30))

# =========================INPUT FEATURE FUNCTIONS NSWE==========================
#Make a function that compiles a get_sensory function only once, such that you don't have to go through all these if statements for every evualation of a genome.


def create_sensory_function(input_features):
    """Create a sensory function based on the selected features."""
    def sensory_function():
        """Get the sensory input for the neural network."""
        x, y = snake[-1]
        sensory_input = []

        for feature in input_features:
            if Config.INPUT_FRAME_OF_REFERENCE == 'nswe':
                if feature == 'wall':
                    sensory_input.extend(get_wall_info(x, y))
                elif feature == 'relative_body':
                    sensory_input.extend(get_body_info(x, y))
                elif feature == 'binary_body':
                    sensory_input.extend(get_relative_distance_to_body(x, y))
                elif feature == 'combined_obstacle_body':
                    sensory_input.extend(get_combined_obstacle_body_info(x, y))
                elif feature == 'relative_apple':
                    sensory_input.extend(get_relative_distance_to_apple(x, y))
                elif feature == 'binary_apple':
                    sensory_input.extend(get_apple_info(x, y))
                elif feature == 'binary_obstacle':
                    sensory_input.extend(get_obstacle_info(x, y))
                elif feature == 'relative_obstacle':
                    sensory_input.extend(get_relative_distance_to_obstacle(x, y))
                elif feature == 'nearest_body_direction':
                    sensory_input.extend(get_direction_of_nearest_body_segment(x, y))
                elif feature == 'dummy':
                    sensory_input.extend([0, 0, 0, 0])
            elif Config.INPUT_FRAME_OF_REFERENCE == 'snake':
                if feature == 'wall':
                    sensory_input.extend(get_wall_info_relative(x, y))
                elif feature == 'body':
                    sensory_input.extend(get_body_info_relative(x, y))
                elif feature == 'relative_body':
                    sensory_input.extend(get_relative_distance_to_body_relative(x, y))
                elif feature == 'combined_obstacle_body':
                    sensory_input.extend(get_combined_obstacle_body_info_relative(x, y))
                elif feature == 'apple':
                    sensory_input.extend(get_apple_info_relative(x, y))
                elif feature == 'relative_apple':
                    sensory_input.extend(get_relative_distance_to_apple_relative(x, y))
                elif feature == 'obstacle':
                    sensory_input.extend(get_obstacle_info_relative(x, y))
                elif feature == 'relative_obstacle':
                    sensory_input.extend(get_relative_distance_to_obstacle_relative(x, y))
                elif feature == 'nearest_body_direction':
                    sensory_input.extend(get_direction_of_nearest_body_segment_relative(x, y))
                elif feature == 'dummy':
                    sensory_input.extend([0, 0, 0])

            if feature == 'history':
                sensory_input.extend(movement_history)

        return np.array(sensory_input, dtype=float)

    return sensory_function



def get_wall_info(x, y):
    """ Get inverted distance to wall. """
    return [1 / (y + 1),
            1 / (NUM_ROWS - y),
            1 / (NUM_COLS - x),
            1 / (x + 1)]


def get_body_info(x, y):
    """ Get binary information about the snake's body in each direction. """
    body_info = [0, 0, 0, 0]  # North, South, East, West
    for (body_x, body_y) in snake[:-1]:
        if body_x == x:
            if body_y < y:
                body_info[0] = 1  # Body to the north
            elif body_y > y:
                body_info[1] = 1  # Body to the south
        elif body_y == y:
            if body_x > x:
                body_info[2] = 1  # Body to the east
            elif body_x < x:
                body_info[3] = 1  # Body to the west
    return body_info

def get_relative_distance_to_body(x, y):
    """ Get the relative distance to the nearest body segment in each direction. """
    dist_to_body = [0, 0, 0, 0]  # Using 0 to indicate no body part in that direction
    for (body_x, body_y) in snake[:-1]:
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

def get_obstacle_info(x, y):
    """ Get binary information about obstacles in each direction. """
    obstacle_info = [0, 0, 0, 0]  # North, South, East, West
    if obstacle[0] == x:
        if obstacle[1] < y:
            obstacle_info[0] = 1  # Obstacle to the north
        elif obstacle[1] > y:
            obstacle_info[1] = 1  # Obstacle to the south
    elif obstacle[1] == y:
        if obstacle[0] > x:
            obstacle_info[2] = 1  # Obstacle to the east
        elif obstacle[0] < x:
            obstacle_info[3] = 1  # Obstacle to the west
    return obstacle_info

def get_relative_distance_to_obstacle(x, y):
    """ Get the relative distance to obstacles in each direction. """
    dist_to_obstacle = [0, 0, 0, 0]  # Using 0 to indicate no obstacle in that direction
    if obstacle[0] == x:
        if obstacle[1] > y:
            dist_to_obstacle[1] = max(dist_to_obstacle[1], 1 / (obstacle[1] - y + 1))
        else:
            dist_to_obstacle[0] = max(dist_to_obstacle[0], 1 / (y - obstacle[1] + 1))
    elif obstacle[1] == y:
        if obstacle[0] > x:
            dist_to_obstacle[2] = max(dist_to_obstacle[2], 1 / (obstacle[0] - x + 1))
        else:
            dist_to_obstacle[3] = max(dist_to_obstacle[3], 1 / (x - obstacle[0] + 1))
    return dist_to_obstacle


def get_apple_info(x, y):
    """ Get binary information about the apple's position relative to the snake's head. """
    a_x, a_y = apple
    return [
        a_x == x and a_y < y,  # Apple to the north
        a_x == x and a_y > y,  # Apple to the south
        a_y == y and a_x > x,  # Apple to the east
        a_y == y and a_x < x   # Apple to the west
    ]

def get_relative_distance_to_apple(x, y):
    """ Get the relative distance to the apple. """
    a_x, a_y = apple
    return [
        1 / (abs(a_y - y) + 1) if a_y < y else 0,  # Apple to the north
        1 / (abs(a_y - y) + 1) if a_y > y else 0,  # Apple to the south
        1 / (abs(a_x - x) + 1) if a_x > x else 0,  # Apple to the east
        1 / (abs(a_x - x) + 1) if a_x < x else 0   # Apple to the west
    ]

def get_combined_obstacle_body_info(x, y):
    """ Get combined binary information about obstacles and the snake's body in each direction. """
    combined_info = [0, 0, 0, 0]  # North, South, East, West

    for (body_x, body_y) in snake[:-1]:
        if body_x == x:
            if body_y < y:
                combined_info[0] = 1  # Body to the north
            elif body_y > y:
                combined_info[1] = 1  # Body to the south
        elif body_y == y:
            if body_x > x:
                combined_info[2] = 1  # Body to the east
            elif body_x < x:
                combined_info[3] = 1  # Body to the west

    if obstacle[0] == x:
        if obstacle[1] < y:
            combined_info[0] = 1  # Obstacle to the north
        elif obstacle[1] > y:
            combined_info[1] = 1  # Obstacle to the south
    elif obstacle[1] == y:
        if obstacle[0] > x:
            combined_info[2] = 1  # Obstacle to the east
        elif obstacle[0] < x:
            combined_info[3] = 1  # Obstacle to the west

    return combined_info

def get_direction_of_nearest_body_segment(x, y):
    """ Get the direction of the nearest body segment in the NSWE frame of reference. """
    directions = ['N', 'S', 'W', 'E']
    nearest_body_dir = [0, 0, 0, 0]  # N, S, W, E
    min_distance = [float('inf'), float('inf'), float('inf'), float('inf')]

    for (body_x, body_y) in snake[:-1]:
        if body_x == x:
            if body_y < y:
                dist = y - body_y
                if dist < min_distance[0]:
                    min_distance[0] = dist
                    nearest_body_dir[0] = 1
            elif body_y > y:
                dist = body_y - y
                if dist < min_distance[1]:
                    min_distance[1] = dist
                    nearest_body_dir[1] = 1
        elif body_y == y:
            if body_x < x:
                dist = x - body_x
                if dist < min_distance[2]:
                    min_distance[2] = dist
                    nearest_body_dir[2] = 1
            elif body_x > x:
                dist = body_x - x
                if dist < min_distance[3]:
                    min_distance[3] = dist
                    nearest_body_dir[3] = 1

    return nearest_body_dir



#===================== INPUT FEATURE FUNCTIONS SNAKE FRAME OF REFERENCE ==========================
def get_relative_directions():
    """ Get the directions relative to the snake's current heading. """
    if v_x == 1 and v_y == 0:    # Heading East
        return ['front', 'left', 'right', 'back']
    elif v_x == -1 and v_y == 0: # Heading West
        return ['front', 'right', 'left', 'back']
    elif v_x == 0 and v_y == 1:  # Heading South
        return ['front', 'left', 'right', 'back']
    elif v_x == 0 and v_y == -1: # Heading North
        return ['front', 'right', 'left', 'back']


def get_wall_info_relative(x, y):
    """ Get binary information about walls in the snake's frame of reference. """
    directions = get_relative_directions()
    wall_info = [0, 0, 0]  # Front, Left, Right

    if directions[0] == 'front':
        wall_info[0] = y == 0 if v_y == -1 else y == NUM_ROWS - 1 if v_y == 1 else x == NUM_COLS - 1 if v_x == 1 else x == 0
    if directions[1] == 'left':
        wall_info[1] = x == 0 if v_y == -1 else x == NUM_COLS - 1 if v_y == 1 else y == 0 if v_x == 1 else y == NUM_ROWS - 1
    if directions[2] == 'right':
        wall_info[2] = x == NUM_COLS - 1 if v_y == -1 else x == 0 if v_y == 1 else y == NUM_ROWS - 1 if v_x == 1 else y == 0
    
    return wall_info

def get_body_info_relative(x, y):
    """ Get binary information about the snake's body in the snake's frame of reference. """
    directions = get_relative_directions()
    body_info = [0, 0, 0]  # Front, Left, Right

    for (body_x, body_y) in snake[:-1]:
        if directions[0] == 'front':
            if v_y == -1 and body_x == x and body_y < y or v_y == 1 and body_x == x and body_y > y or v_x == 1 and body_x > x and body_y == y or v_x == -1 and body_x < x and body_y == y:
                body_info[0] = 1
        if directions[1] == 'left':
            if v_y == -1 and body_x < x and body_y == y or v_y == 1 and body_x > x and body_y == y or v_x == 1 and body_x == x and body_y > y or v_x == -1 and body_x == x and body_y < y:
                body_info[1] = 1
        if directions[2] == 'right':
            if v_y == -1 and body_x > x and body_y == y or v_y == 1 and body_x < x and body_y == y or v_x == 1 and body_x == x and body_y < y or v_x == -1 and body_x == x and body_y > y:
                body_info[2] = 1

    return body_info

def get_obstacle_info_relative(x, y):
    """ Get binary information about obstacles in the snake's frame of reference. """
    directions = get_relative_directions()
    obstacle_info = [0, 0, 0]  # Front, Left, Right

    if directions[0] == 'front':
        if v_y == -1 and obstacle[0] == x and obstacle[1] < y or v_y == 1 and obstacle[0] == x and obstacle[1] > y or v_x == 1 and obstacle[0] > x and obstacle[1] == y or v_x == -1 and obstacle[0] < x and obstacle[1] == y:
            obstacle_info[0] = 1
    if directions[1] == 'left':
        if v_y == -1 and obstacle[0] < x and obstacle[1] == y or v_y == 1 and obstacle[0] > x and obstacle[1] == y or v_x == 1 and obstacle[0] == x and obstacle[1] > y or v_x == -1 and obstacle[0] == x and obstacle[1] < y:
            obstacle_info[1] = 1
    if directions[2] == 'right':
        if v_y == -1 and obstacle[0] > x and obstacle[1] == y or v_y == 1 and obstacle[0] < x and obstacle[1] == y or v_x == 1 and obstacle[0] == x and obstacle[1] < y or v_x == -1 and obstacle[0] == x and obstacle[1] > y:
            obstacle_info[2] = 1

    return obstacle_info

def get_combined_obstacle_body_info_relative(x, y):
    """ Get combined binary information about obstacles and the snake's body in the snake's frame of reference. """
    directions = get_relative_directions()
    combined_info = [0, 0, 0]  # Front, Left, Right

    for (body_x, body_y) in snake[:-1]:
        if directions[0] == 'front':
            if v_y == -1 and body_x == x and body_y < y or v_y == 1 and body_x == x and body_y > y or v_x == 1 and body_x > x and body_y == y or v_x == -1 and body_x < x and body_y == y:
                combined_info[0] = 1
        if directions[1] == 'left':
            if v_y == -1 and body_x < x and body_y == y or v_y == 1 and body_x > x and body_y == y or v_x == 1 and body_x == x and body_y > y or v_x == -1 and body_x == x and body_y < y:
                combined_info[1] = 1
        if directions[2] == 'right':
            if v_y == -1 and body_x > x and body_y == y or v_y == 1 and body_x < x and body_y == y or v_x == 1 and body_x == x and body_y < y or v_x == -1 and body_x == x and body_y > y:
                combined_info[2] = 1

    if directions[0] == 'front':
        if v_y == -1 and obstacle[0] == x and obstacle[1] < y or v_y == 1 and obstacle[0] == x and obstacle[1] > y or v_x == 1 and obstacle[0] > x and obstacle[1] == y or v_x == -1 and obstacle[0] < x and obstacle[1] == y:
            combined_info[0] = 1
    if directions[1] == 'left':
        if v_y == -1 and obstacle[0] < x and obstacle[1] == y or v_y == 1 and obstacle[0] > x and obstacle[1] == y or v_x == 1 and obstacle[0] == x and obstacle[1] > y or v_x == -1 and obstacle[0] == x and obstacle[1] < y:
            combined_info[1] = 1
    if directions[2] == 'right':
        if v_y == -1 and obstacle[0] > x and obstacle[1] == y or v_y == 1 and obstacle[0] < x and obstacle[1] == y or v_x == 1 and obstacle[0] == x and obstacle[1] < y or v_x == -1 and obstacle[0] == x and obstacle[1] > y:
            combined_info[2] = 1

    return combined_info

def get_apple_info_relative(x, y):
    """ Get binary information about the apple's position relative to the snake's head in the snake's frame of reference. """
    directions = get_relative_directions()
    a_x, a_y = apple
    apple_info = [0, 0, 0]  # Front, Left, Right

    if directions[0] == 'front':
        if v_y == -1 and a_x == x and a_y < y or v_y == 1 and a_x == x and a_y > y or v_x == 1 and a_x > x and a_y == y or v_x == -1 and a_x < x and a_y == y:
            apple_info[0] = 1
    if directions[1] == 'left':
        if v_y == -1 and a_x < x and a_y == y or v_y == 1 and a_x > x and a_y == y or v_x == 1 and a_x == x and a_y > y or v_x == -1 and a_x == x and a_y < y:
            apple_info[1] = 1
    if directions[2] == 'right':
        if v_y == -1 and a_x > x and a_y == y or v_y == 1 and a_x < x and a_y == y or v_x == 1 and a_x == x and a_y < y or v_x == -1 and a_x == x and a_y > y:
            apple_info[2] = 1

    return apple_info

def get_relative_distance_to_body_relative(x, y):
    """ Get the relative distance to the nearest body segment in the snake's frame of reference. """
    directions = get_relative_directions()
    dist_to_body = [0, 0, 0]  # Front, Left, Right

    for (body_x, body_y) in snake[:-1]:
        if directions[0] == 'front':
            if v_y == -1 and body_x == x and body_y < y:
                dist_to_body[0] = max(dist_to_body[0], 1 / (y - body_y + 1))
            elif v_y == 1 and body_x == x and body_y > y:
                dist_to_body[0] = max(dist_to_body[0], 1 / (body_y - y + 1))
            elif v_x == 1 and body_x > x and body_y == y:
                dist_to_body[0] = max(dist_to_body[0], 1 / (body_x - x + 1))
            elif v_x == -1 and body_x < x and body_y == y:
                dist_to_body[0] = max(dist_to_body[0], 1 / (x - body_x + 1))
        if directions[1] == 'left':
            if v_y == -1 and body_x < x and body_y == y:
                dist_to_body[1] = max(dist_to_body[1], 1 / (x - body_x + 1))
            elif v_y == 1 and body_x > x and body_y == y:
                dist_to_body[1] = max(dist_to_body[1], 1 / (body_x - x + 1))
            elif v_x == 1 and body_x == x and body_y > y:
                dist_to_body[1] = max(dist_to_body[1], 1 / (body_y - y + 1))
            elif v_x == -1 and body_x == x and body_y < y:
                dist_to_body[1] = max(dist_to_body[1], 1 / (y - body_y + 1))
        if directions[2] == 'right':
            if v_y == -1 and body_x > x and body_y == y:
                dist_to_body[2] = max(dist_to_body[2], 1 / (body_x - x + 1))
            elif v_y == 1 and body_x < x and body_y == y:
                dist_to_body[2] = max(dist_to_body[2], 1 / (x - body_x + 1))
            elif v_x == 1 and body_x == x and body_y < y:
                dist_to_body[2] = max(dist_to_body[2], 1 / (y - body_y + 1))
            elif v_x == -1 and body_x == x and body_y > y:
                dist_to_body[2] = max(dist_to_body[2], 1 / (body_y - y + 1))

    return dist_to_body

def get_relative_distance_to_obstacle_relative(x, y):
    """ Get the relative distance to obstacles in the snake's frame of reference. """
    directions = get_relative_directions()
    dist_to_obstacle = [0, 0, 0]  # Front, Left, Right

    if directions[0] == 'front':
        if v_y == -1 and obstacle[0] == x and obstacle[1] < y:
            dist_to_obstacle[0] = max(dist_to_obstacle[0], 1 / (y - obstacle[1] + 1))
        elif v_y == 1 and obstacle[0] == x and obstacle[1] > y:
            dist_to_obstacle[0] = max(dist_to_obstacle[0], 1 / (obstacle[1] - y + 1))
        elif v_x == 1 and obstacle[0] > x and obstacle[1] == y:
            dist_to_obstacle[0] = max(dist_to_obstacle[0], 1 / (obstacle[0] - x + 1))
        elif v_x == -1 and obstacle[0] < x and obstacle[1] == y:
            dist_to_obstacle[0] = max(dist_to_obstacle[0], 1 / (x - obstacle[0] + 1))
    if directions[1] == 'left':
        if v_y == -1 and obstacle[0] < x and obstacle[1] == y:
            dist_to_obstacle[1] = max(dist_to_obstacle[1], 1 / (x - obstacle[0] + 1))
        elif v_y == 1 and obstacle[0] > x and obstacle[1] == y:
            dist_to_obstacle[1] = max(dist_to_obstacle[1], 1 / (obstacle[0] - x + 1))
        elif v_x == 1 and obstacle[0] == x and obstacle[1] > y:
            dist_to_obstacle[1] = max(dist_to_obstacle[1], 1 / (obstacle[1] - y + 1))
        elif v_x == -1 and obstacle[0] == x and obstacle[1] < y:
            dist_to_obstacle[1] = max(dist_to_obstacle[1], 1 / (y - obstacle[1] + 1))
    if directions[2] == 'right':
        if v_y == -1 and obstacle[0] > x and obstacle[1] == y:
            dist_to_obstacle[2] = max(dist_to_obstacle[2], 1 / (obstacle[0] - x + 1))
        elif v_y == 1 and obstacle[0] < x and obstacle[1] == y:
            dist_to_obstacle[2] = max(dist_to_obstacle[2], 1 / (x - obstacle[0] + 1))
        elif v_x == 1 and obstacle[0] == x and obstacle[1] < y:
            dist_to_obstacle[2] = max(dist_to_obstacle[2], 1 / (y - obstacle[1] + 1))
        elif v_x == -1 and obstacle[0] == x and obstacle[1] > y:
            dist_to_obstacle[2] = max(dist_to_obstacle[2], 1 / (obstacle[1] - y + 1))

    return dist_to_obstacle

def get_relative_distance_to_apple_relative(x, y):
    """ Get the relative distance to the apple in the snake's frame of reference. """
    directions = get_relative_directions()
    a_x, a_y = apple
    dist_to_apple = [0, 0, 0]  # Front, Left, Right

    if directions[0] == 'front':
        if v_y == -1 and a_x == x and a_y < y:
            dist_to_apple[0] = max(dist_to_apple[0], 1 / (y - a_y + 1))
        elif v_y == 1 and a_x == x and a_y > y:
            dist_to_apple[0] = max(dist_to_apple[0], 1 / (a_y - y + 1))
        elif v_x == 1 and a_x > x and a_y == y:
            dist_to_apple[0] = max(dist_to_apple[0], 1 / (a_x - x + 1))
        elif v_x == -1 and a_x < x and a_y == y:
            dist_to_apple[0] = max(dist_to_apple[0], 1 / (x - a_x + 1))
    if directions[1] == 'left':
        if v_y == -1 and a_x < x and a_y == y:
            dist_to_apple[1] = max(dist_to_apple[1], 1 / (x - a_x + 1))
        elif v_y == 1 and a_x > x and a_y == y:
            dist_to_apple[1] = max(dist_to_apple[1], 1 / (a_x - x + 1))
        elif v_x == 1 and a_x == x and a_y > y:
            dist_to_apple[1] = max(dist_to_apple[1], 1 / (a_y - y + 1))
        elif v_x == -1 and a_x == x and a_y < y:
            dist_to_apple[1] = max(dist_to_apple[1], 1 / (y - a_y + 1))
    if directions[2] == 'right':
        if v_y == -1 and a_x > x and a_y == y:
            dist_to_apple[2] = max(dist_to_apple[2], 1 / (a_x - x + 1))
        elif v_y == 1 and a_x < x and a_y == y:
            dist_to_apple[2] = max(dist_to_apple[2], 1 / (x - a_x + 1))
        elif v_x == 1 and a_x == x and a_y < y:
            dist_to_apple[2] = max(dist_to_apple[2], 1 / (y - a_y + 1))
        elif v_x == -1 and a_x == x and a_y > y:
            dist_to_apple[2] = max(dist_to_apple[2], 1 / (a_y - y + 1))

    return dist_to_apple

def get_direction_of_nearest_body_segment_relative(x, y):
    """ Get the direction of the nearest body segment in the snake's frame of reference. """
    directions = get_relative_directions()
    nearest_body_dir = [0, 0, 0]  # Front, Left, Right
    min_distance = float('inf')

    for (body_x, body_y) in snake[:-1]:
        if directions[0] == 'front':
            if v_y == -1 and body_x == x and body_y < y and y - body_y < min_distance:
                nearest_body_dir = [1, 0, 0]
                min_distance = y - body_y
            elif v_y == 1 and body_x == x and body_y > y and body_y - y < min_distance:
                nearest_body_dir = [1, 0, 0]
                min_distance = body_y - y
            elif v_x == 1 and body_x > x and body_y == y and body_x - x < min_distance:
                nearest_body_dir = [1, 0, 0]
                min_distance = body_x - x
            elif v_x == -1 and body_x < x and body_y == y and x - body_x < min_distance:
                nearest_body_dir = [1, 0, 0]
                min_distance = x - body_x
        if directions[1] == 'left':
            if v_y == -1 and body_x < x and body_y == y and x - body_x < min_distance:
                nearest_body_dir = [0, 1, 0]
                min_distance = x - body_x
            elif v_y == 1 and body_x > x and body_y == y and body_x - x < min_distance:
                nearest_body_dir = [0, 1, 0]
                min_distance = body_x - x
            elif v_x == 1 and body_x == x and body_y > y and body_y - y < min_distance:
                nearest_body_dir = [0, 1, 0]
                min_distance = body_y - y
            elif v_x == -1 and body_x == x and body_y < y and y - body_y < min_distance:
                nearest_body_dir = [0, 1, 0]
                min_distance = y - body_y
        if directions[2] == 'right':
            if v_y == -1 and body_x > x and body_y == y and body_x - x < min_distance:
                nearest_body_dir = [0, 0, 1]
                min_distance = body_x - x
            elif v_y == 1 and body_x < x and body_y == y and x - body_x < min_distance:
                nearest_body_dir = [0, 0, 1]
                min_distance = x - body_x
            elif v_x == 1 and body_x == x and body_y < y and y - body_y < min_distance:
                nearest_body_dir = [0, 0, 1]
                min_distance = y - body_y
            elif v_x == -1 and body_x == x and body_y > y and body_y - y < min_distance:
                nearest_body_dir = [0, 0, 1]
                min_distance = body_y - y

    return nearest_body_dir

