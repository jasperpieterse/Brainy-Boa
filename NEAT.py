"""All functions and classes related to the NEAT algorithm are defined in this file. """
from __future__ import print_function
import pickle
import os
import neat
import visualize
import multiprocessing
import shutil
from config import *


os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"


def simulate_headless(net):
    global current_game_instance
    if current_game_instance is None:
        raise ValueError("Current game instance is not set.")
    return current_game_instance.simulate_headless(net)

def eval_genome(genome, config): 
    """
    Fitness function to evaluate single genome, used with ParallelEvaluator.
    """
    global current_game_instance
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitness = current_game_instance.simulate_headless(net)  # Evaluate the genome in a headless simulation.
    return fitness                          # For the ParallelEvaluator to work, the fitness must be returned.


def eval_genomes(genomes, config):
    """
    Fitness function used to assign fitness to all genomees. This is different from eval_genome in that 
    it does not use the ParallelEvaluator and thus goes through each genome in the population one by one.

    Args:
    genomes (list of tuples): List of (genome_id, genome) tuples.
    config (neat.Config): NEAT configuration settings for network creation.
    """
    global current_game_instance
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config) # Create a neural network from the genome.
        fitness = current_game_instance.simulate_headless(net)  # Evaluate the genome in a headless simulation.
        genome.fitness = fitness  # Assign the fitness to the genome.


def run_NEAT(config_file, n_generations=10):
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
    p.add_reporter(neat.Checkpointer(10, filename_prefix=f"{Paths.RESULTS_PATH}/checkpoints/population-"))

    # Add a parallel evaluator to evaluate the population in parallel.
    parallel_evaluator = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome) #parallelized fitness function

    # Run the NEAT algorithm for n generations
    winner = p.run(parallel_evaluator.evaluate, n=n_generations)  

    # Visualize statistics and species progression over generations.
    visualize.plot_stats(stats, ylog=False, view=True, filename=f"{Paths.RESULTS_PATH}/fitness_graph.png")
    visualize.plot_species(stats, view=False, filename=f"{Paths.RESULTS_PATH}/species_graph.png")

    # Save the winner.
    with open('results/winner_genome', 'wb') as f:
        pickle.dump(winner, f)
    
    return winner, stats

def run_NEAT_repeated(config_file, n_runs = 1, n_generations = 10):
    """Runs multiple instances of the NEAT algorithm and returns the winners and statistics of each run."""
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Ensure the results directory exists.
    if not os.path.exists(f"{Paths.RESULTS_PATH}/checkpoints"):
        os.makedirs(f"{Paths.RESULTS_PATH}/checkpoints")

    #Clear the output directory
    shutil.rmtree(Paths.RESULTS_PATH)

    winners = []
    stats_list = []

    for i in range(n_runs):  # Run the NEAT algorithm n times
        print(f"Running NEAT algorithm, run {i}")
        p = neat.Population(config)
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)

        if not os.path.exists(f"{Paths.RESULTS_PATH}/checkpoints/run{i}"):
            os.makedirs(f"{Paths.RESULTS_PATH}/checkpoints/run{i}")
        p.add_reporter(neat.Checkpointer(1, filename_prefix=f"{Paths.RESULTS_PATH}/checkpoints/run{i}/population-"))

        parallel_evaluator = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
        winner = p.run(parallel_evaluator.evaluate, n = n_generations)

        winners.append(winner)
        stats_list.append(stats)

        # Save the winner of each run
        with open(f'{Paths.RESULTS_PATH}/checkpoints/run{i}/winner_genome', 'wb') as f:
            pickle.dump(winner, f)

        #Print results
        print(f"Run {i} completed, best fitness: {winner.fitness}")

    return winners, stats_list

def test_winner(genome, config_path):
    """Visualizes the genome passed playing the snake game"""

    # Load configuration into a NEAT object.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)
    
    net = neat.nn.FeedForwardNetwork.create(genome, config) # Initialize the neural network from the passed genome.s

    # run the simulation
    current_game_instance.simulate_animation(net, genome, config) # Simulate the environment with a GUI.


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, Paths.Config_PATH)
    winner, stats = run_NEAT(config_path)
    visualize.test_winner(winner, config_path) 
