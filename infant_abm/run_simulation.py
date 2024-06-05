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
    repeats = 7
    iterations = 5000

    for parent_class in ["MoverParent", "VisionOnlyParent"]:
        for infant_class in ["NoVisionInfant", "SeqVisionInfant"]:
            base_params = {"parent_class": parent_class, "infant_class": infant_class}
            parameter_sets = get_model_param_sets(
                (0.4, 0.6, 3), base_params=base_params
            )
            output_path = f"../results/comparative2/{parent_class}_{infant_class}.hdf"

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
