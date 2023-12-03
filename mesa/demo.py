from simulation import Simulation
import pygad
import numpy as np

from infant_abm.agents.infant import Params as InfantParams


MAX_ITER = 2000
REPEATS = 2
SUCCESS_DIST = 40


def get_fitness(result):
    # print(result)
    try:
        return np.where(result.goal_dist < SUCCESS_DIST)[0][0]
    except IndexError:
        return MAX_ITER + 1


def fitness_func(_ga_instance, genotype, _solution_idx):
    # output = numpy.sum(solution*function_inputs)
    # fitness = 1.0 / numpy.abs(output - desired_output)

    parameter_set = {
        "width": 300,
        "height": 300,
        "speed": 2,
        "lego_count": 4,
        "responsiveness": 50,
        "relevance": 50,
        "infant_params": InfantParams(precision=genotype[0], coordination=genotype[1], exploration=genotype[2])
    }

    simulation = Simulation(
        model_param_sets=[parameter_set],
        max_iterations=MAX_ITER,
        repeats=REPEATS,
        display=False,
    )

    result = simulation.run()[0]
    fitness = get_fitness(result)

    # print("FITNESS", fitness)

    return 1 / fitness


fitness_function = fitness_func

num_generations = 10
num_parents_mating = 2

sol_per_pop = 4
num_genes = 3

init_range_low = 0
init_range_high = 1

# gene_space = range(0, 1)

parent_selection_type = "sss"
keep_parents = 1

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = 10

ga_instance = pygad.GA(
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    fitness_func=fitness_function,
    sol_per_pop=sol_per_pop,
    num_genes=num_genes,
    init_range_low=init_range_low,
    init_range_high=init_range_high,
    parent_selection_type=parent_selection_type,
    keep_parents=keep_parents,
    crossover_type=crossover_type,
    mutation_type=mutation_type,
    mutation_percent_genes=mutation_percent_genes,
)

ga_instance.run()


solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print(
    "Fitness value of the best solution = {solution_fitness}".format(
        solution_fitness=solution_fitness
    )
)

prediction = InfantParams(
    precision=solution[0], coordination=solution[1], exploration=solution[2])
print(
    "Predicted output based on the best solution : {prediction} with fitness {fitness}".format(
        prediction=prediction,
        fitness=1 / solution_fitness
    )
)
