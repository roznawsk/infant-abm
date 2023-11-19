import numpy as np
import itertools

from simulation import Simulation


def get_model_param_sets(default_params):
    prec = np.linspace(20, 100, 2)
    exp = np.linspace(0, 100, 2)
    coord = np.linspace(0, 100, 2)
    resp = np.linspace(0, 100, 1)
    rel = np.linspace(0, 100, 1)

    params = []

    for param_set in itertools.product(*[prec, exp, coord, resp, rel]):
        p, e, c, rs, rl = param_set

        params.append(
            {
                **default_params,
                **{
                    "precision": p,
                    "exploration": e,
                    "coordination": c,
                    "responsiveness": rs,
                    "relevance": rl,
                },
            }
        )

    return params


if __name__ == "__main__":
    grid_size = 300
    repeats = 10
    max_iter = 1000
    epochs = 500

    output_path = "../results/test_run_temp.hdf"

    default_model_params = {
        "width": grid_size,
        "height": grid_size,
        "speed": 2,
        "lego_count": 4,
        "precision": 50,
        "exploration": 50,
        "coordination": 50,
        "responsiveness": 50,
        "relevance": 50,
    }

    parameter_sets = get_model_param_sets(default_model_params)

    simulation = Simulation(
        model_param_sets=parameter_sets,
        max_iterations=max_iter,
        repeats=repeats,
        output_path=output_path,
    )
    simulation.run()
    simulation.save()
