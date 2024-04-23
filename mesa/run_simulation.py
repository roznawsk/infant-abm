import numpy as np
import itertools
import warnings

from simulation import Simulation
from infant_abm.agents.infant import Params as InfantParams

warnings.simplefilter(action="ignore", category=FutureWarning)


def get_model_param_sets(num):
    p1, p2, p3 = num
    perception = np.linspace(0, 1, p1)
    persistence = np.linspace(0, 1, p2)
    coordination = np.linspace(0, 1, p3)

    params = []

    for param_set in itertools.product(*[perception, persistence, coordination]):
        i_params = InfantParams.from_array(param_set)

        params.append({"infant_params": i_params})

    return params


if __name__ == "__main__":
    repeats = 7
    iterations = 2000
    output_path = "../results/test_run_temp.hdf"

    parameter_sets = get_model_param_sets((8, 8, 8))
    # parameter_sets = [{"infant_params": InfantParams(perception=0.66, persistence=1, coordination=1)}]

    simulation = Simulation(
        model_param_sets=parameter_sets,
        iterations=iterations,
        repeats=repeats,
        output_path=output_path,
        display=True,
    )

    results = simulation.run()
    simulation.save()
