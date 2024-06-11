import configparser
class Config:   
    #Neat parameters
    N_RUNS = 1  # Number of runs of the NEAT algorithm
    N_GENERATIONS = 10 # Number of generations for each run of the NEAT algorithm

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


