import configparser
class Config:   
    #Neat parameters
    N_RUNS = 1  # Number of runs of the NEAT algorithm
    N_GENERATIONS = 5 # Number of generations for each run of the NEAT algorithm
    MIN_TIME_TO_EAT_APPLE = 100  # Minimum time to eat an apple
    FITNESS_ITERS = 10   # Number of iterations to compute a genomes fitness with

    #snake settings
    #options: wall, relative_body, binary_body, combined_obstacle_body, relative_apple, binary_apple, binary_obstacle, relative_obstacle
    INPUT_FEATURES = ['wall', 'relative_body', 'relative_apple', 'relative_obstacle']   
    FRAME_OF_REFERENCE = 'nswe' #nswe or snake
    HISTORY_LENGTH = 3 # number of previous moves to remember  
    USE_OBSTACLES = True
    
    #Compute input and output size based on features and frame of reference
    if FRAME_OF_REFERENCE == 'nswe':
        NR_INPUT_FEATURES = 4 * len(INPUT_FEATURES)
        NR_OUTPUT_FEATURES = 4
    elif FRAME_OF_REFERENCE == 'snake':
        NR_INPUT_FEATURES = 3 * len(INPUT_FEATURES)
        NR_OUTPUT_FEATURES = 3
    if 'history' in INPUT_FEATURES:
        NR_INPUT_FEATURES += HISTORY_LENGTH - 1

class Paths:
    RESULTS_PATH = 'results/'      # PATH to the results directory
    CONFIG_PATH = 'config.py'  # Path to the configuration file
    NEAT_CONFIG_PATH = 'config-neat'  # Path to the NEAT configuration file
    DRAW_NET_PATH = 'target_pursuit_2000_results/winner-feedforward.gv'  # Path to the neural network visualization
    WINNER_PATH = 'results/winner-feedforward'  # Path to the winner genome


#Function to update configurtion
def update_config(file_path, section, variables):
    config = configparser.ConfigParser()
    config.read(file_path)
    for key, value in variables.items():
        config.set(section, key, value)
    with open(file_path, 'w') as configfile:
        config.write(configfile)

#dictionary of dictionaries
neat_params = {
    'input_output': {
        'DefaultGenome': {
            'num_inputs': f'{Config.NR_INPUT_FEATURES}',
            'num_outputs': f'{Config.NR_OUTPUT_FEATURES}'
        }
    },
}

update_config('config-neat','DefaultGenome', neat_params['input_output']['DefaultGenome'])