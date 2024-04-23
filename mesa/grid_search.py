from simulation import Simulation
import numpy as np

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
