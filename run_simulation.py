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


def get_linspace_str(linspace):
    return (
        str(linspace)
        .replace("(", "_")
        .replace(")", "")
        .replace(" ", "")
        .replace(".", "")
        .replace(",", "_")
    )


def run_basic_simulation(output_dir, parameter_sets, repeats=13, iterations=20000):
    simulation = Simulation(
        model_param_sets=parameter_sets,
        iterations=iterations,
        repeats=repeats,
        output_dir=output_dir,
        display=True,
    )

    simulation.run()

    return simulation


def run_comparative_simulation():
    repeats = 11
    iterations = 15000

    for infant_class in ["SeqVisionInfant"]:
        base_params = {"infant_class": infant_class}
        parameter_sets = get_model_param_sets((0.1, 0.9, 5), base_params=base_params)
        output_path = f"../results/final_final_3/{infant_class}.hdf"

        simulation = Simulation(
            model_param_sets=parameter_sets,
            iterations=iterations,
            repeats=repeats,
            output_path=output_path,
            display=True,
        )

        simulation.run()


def run_comparative_boost_simulation():
    repeats = 11
    iterations = 20000
    linspace = (0.05, 0.95, 10)
    boosted_parameter = "coordination"
    boost_name = f"{boosted_parameter}_boost_value"
    boost_values = [0.15, 0.45]

    default_boosts = {"persistence_boost_value": 0.5, "coordination_boost_value": 0.2}

    dir_path = f"./results/{boosted_parameter}{get_linspace_str(linspace)}"
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
    linspace = (0.1, 0.9, 5)

    output_dir = "./results/model1/boost_improvement"
    Path(output_dir).mkdir(parents=False, exist_ok=True)

    lo, hi, num = linspace

    perception, persistence, coordination = [
        np.round(np.linspace(lo, hi, num), 3) for _ in range(3)
    ]

    boost = np.linspace(0, 1, 5)
    params = []

    for param_set in itertools.product(*[perception, persistence, coordination, boost]):
        # for bst in boost:
        prc, prs, crd, bst = param_set

        # prc, prs, crd = param_set

        i_params = InfantParams.from_array([prc, prs, crd])
        base_params = {
            "config": Config(persistence_boost_value=bst, coordination_boost_value=bst)
        }

        params.append({**base_params, "infant_params": i_params})

    run_basic_simulation(
        output_dir=output_dir,
        parameter_sets=params,
        iterations=20000,
        repeats=13,
    )
