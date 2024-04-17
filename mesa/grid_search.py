from simulation import Simulation
import numpy as np

import pygad

from infant_abm.agents.infant import Params as InfantParams


MAX_ITER = 2000
REPEATS = 25
SUCCESS_DIST = 40


def get_fitness(result):
    try:
        return np.where(result.goal_dist < SUCCESS_DIST)[0][0]
    except IndexError:
        return MAX_ITER + 1


def fitness_func(_ga_instance, genotype, _solution_idx):
    parameter_set = {
        "toy_count": 4,
        "responsiveness": 50,
        "relevance": 50,
        "infant_params": InfantParams(
            precision=genotype[0], coordination=genotype[1], exploration=genotype[2]
        ),
    }

    simulation = Simulation(
        model_param_sets=[parameter_set],
        max_iterations=MAX_ITER,
        repeats=REPEATS,
        display=False,
    )

    result = simulation.run()[0]
    fitness = get_fitness(result)

    return 1 / fitness


if __name__ == "__main__":
    fitness_function = fitness_func

    num_generations = 20
    num_parents_mating = 8

    sol_per_pop = 40
    num_genes = 3

    init_range_low = 0
    init_range_high = 1
    gene_space = {"low": 0, "high": 1}

    parent_selection_type = "sss"
    keep_parents = 3
    keep_elitism = 1

    crossover_type = "single_point"

    mutation_type = "random"
    mutation_probability = 0.2
    random_mutation_min_val = 0.001
    random_mutation_max_val = 0.02

    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_function,
        sol_per_pop=sol_per_pop,
        num_genes=num_genes,
        init_range_low=init_range_low,
        init_range_high=init_range_high,
        gene_space=gene_space,
        parent_selection_type=parent_selection_type,
        keep_parents=keep_parents,
        keep_elitism=keep_elitism,
        crossover_type=crossover_type,
        mutation_type=mutation_type,
        mutation_probability=mutation_probability,
        random_mutation_min_val=random_mutation_min_val,
        random_mutation_max_val=random_mutation_max_val,
        parallel_processing=20,
        save_solutions=False,
        save_best_solutions=True,
        suppress_warnings=True,
    )

    ga_instance.run()
    ga_instance.save("grid_results/ga_instance_2")
