import numpy as np
import itertools
import warnings

from simulation import Simulation
from infant_abm.agents.infant import Params as InfantParams

warnings.simplefilter(action="ignore", category=FutureWarning)


def get_model_param_sets(linspace, base_params=dict()):
    lo, hi, num = linspace
    perception = np.linspace(lo, hi, num)
    persistence = np.linspace(lo, hi, num)
    coordination = np.linspace(lo, hi, num)

    params = []

    for param_set in itertools.product(*[perception, persistence, coordination]):
        i_params = InfantParams.from_array(param_set)

        # print(i_params.to_array())

        params.append({**base_params, "infant_params": i_params})

    return params


def run_basic_simulation():
    repeats = 11
    iterations = 2000
    output_path = "../results/run_no_eye_contact.hdf"

    parameter_sets = get_model_param_sets((0, 1, 8))

    simulation = Simulation(
        model_param_sets=parameter_sets,
        iterations=iterations,
        repeats=repeats,
        output_path=output_path,
        display=True,
    )

    simulation.run()
    simulation.save()


def run_comparative_simulation():
    repeats = 11
    iterations = 15000

    for infant_class in ["SeqVisionInfant"]:
        base_params = {"infant_class": infant_class}
        parameter_sets = get_model_param_sets(
            (0.1, 0.9, 5), base_params=base_params
        )
        output_path = f"../results/final_final_3/{infant_class}.hdf"

        simulation = Simulation(
            model_param_sets=parameter_sets,
            iterations=iterations,
            repeats=repeats,
            output_path=output_path,
            display=True,
        )

        simulation.run()
        simulation.save()


if __name__ == "__main__":
    # run_basic_simulation()
    run_comparative_simulation()
