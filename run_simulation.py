import numpy as np
import itertools
import warnings

from pathlib import Path

from infant_abm import Simulation, Config, InfantParams

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


def run_basic_simulation(filename, parameter_sets, repeats=13, iterations=10000):
    simulation = Simulation(
        model_param_sets=parameter_sets,
        iterations=iterations,
        repeats=repeats,
        output_path=filename,
        display=True,
    )

    simulation.run()
    simulation.save()

    return simulation


def run_comparative_agent_simulation():
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


def run_comparative_boost_simulation():
    repeats = 11
    iterations = 20000
    linspace = (0.05, 0.95, 10)
    boosted_parameter = "coordination"
    boost_name = f"{boosted_parameter}_boost_value"
    boost_values = [0.15, 0.45]

    default_boosts = {"persistence_boost_value": 0.5, "coordination_boost_value": 0.2}

    linspace_str = (
        str(linspace)
        .replace("(", "_")
        .replace(")", "")
        .replace(" ", "")
        .replace(".", "")
        .replace(",", "_")
    )

    dir_path = f"./results/{boosted_parameter}{linspace_str}"
    Path(dir_path).mkdir(parents=False, exist_ok=False)

    for boost_value in boost_values:
        boosts = default_boosts
        boosts[boost_name] = boost_value

        parameter_sets = get_model_param_sets(
            linspace, base_params={"config": Config(**boosts)}
        )

        boost_value_str = f"{boosted_parameter[:5]}{round(boost_value * 100):03d}"

        filename = f"{dir_path}/{boost_value_str}.hdf"

        run_basic_simulation(
            filename=filename,
            parameter_sets=parameter_sets,
            repeats=repeats,
            iterations=iterations,
        )


if __name__ == "__main__":
    # run_basic_simulation()
    # run_comparative_simulation()
    run_comparative_boost_simulation()
