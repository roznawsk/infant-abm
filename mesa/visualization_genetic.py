import pygad

import matplotlib.pyplot as plt

from infant_abm.agents.infant import Params as InfantParams


if __name__ == "__main__":
    # new_solution_rate = ga_instance.plot_new_solution_rate()
    # fitness_plot = ga_instance.plot_fitness()

    ga_instance = pygad.load("genetic_results/ga_instance_0")
    ga_instance.plot_genes(graph_type="plot", plot_type="scatter")

    all_solutions = ga_instance.best_solutions_fitness

    plt.plot([1 / s for s in all_solutions])
    plt.show()

    print(all_solutions)

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print(
        "Fitness value of the best solution = {solution_fitness}".format(
            solution_fitness=solution_fitness
        )
    )

    prediction = InfantParams(
        precision=solution[0], coordination=solution[1], exploration=solution[2]
    )
    print(
        "Predicted output based on the best solution : {prediction} with fitness {fitness}".format(
            prediction=prediction, fitness=1 / solution_fitness
        )
    )
