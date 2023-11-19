import numpy as np
import itertools

from simulation import Simulation
from infant_abm.agents.infant import Params as InfantParams


def get_model_param_sets(default_params):
    prec = np.linspace(0.2, 1, 2)
    exp = np.linspace(0, 1, 2)
    coord = np.linspace(0, 1, 2)
    resp = np.linspace(0, 1, 1)
    rel = np.linspace(0, 1, 1)

    params = []

    for param_set in itertools.product(*[prec, exp, coord, resp, rel]):
        p, e, c, rs, rl = param_set

        i_params = InfantParams(precision=p, coordination=c, exploration=e)

        params.append(
            {
                **default_params,
                **{"infant_params": i_params, "responsiveness": rs, "relevance": rl},
            }
        )

    return params


if __name__ == "__main__":
    grid_size = 300
    repeats = 10
    max_iter = 5000

    output_path = "../results/test_run_temp.hdf"

    default_model_params = {
        "width": grid_size,
        "height": grid_size,
        "speed": 2,
        "lego_count": 4,
        # 'precision': 50,
        # 'exploration': 50,
        # 'coordination': 50,
        "responsiveness": 50,
        "relevance": 50,
    }

    parameter_sets = get_model_param_sets(default_model_params)

    simulation = Simulation(
        model_param_sets=parameter_sets,
        max_iterations=max_iter,
        repeats=repeats,
        output_path=output_path,
        display=True,
    )
    print(simulation.run())
    # simulation.save()
