import numpy as np
import itertools
import warnings

from pathlib import Path

from infant_abm import Simulation, Config, InfantParams
from infant_abm.agents import (
    NoVisionInfant,
    NoVisionParent,
    SpatialVisionInfant,
    SpatialVisionParent,
    AbstractVisionInfant,
    AbstractVisionParent,
    QLearnInfant,
    QLearnParent,
)

warnings.simplefilter(action="ignore", category=FutureWarning)


class Model_0_1_0:
    infant_class = NoVisionInfant
    parent_class = NoVisionParent
    output_dir = "v0.1.0"


class Model_0_1_1:
    infant_class = SpatialVisionInfant
    parent_class = SpatialVisionParent
    output_dir = "v0.1.1"


class Model_0_1_2:
    infant_class = AbstractVisionInfant
    parent_class = AbstractVisionParent
    output_dir = "v0.1.2"


class Model_0_2_0:
    infant_class = QLearnInfant
    parent_class = QLearnParent
    output_dir = "v0.2.0"


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
    linspace = (0.1, 0.9, 5)

    output_dir = "./results/model1/boost_improvement"
    Path(output_dir).mkdir(parents=False, exist_ok=False)

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


if __name__ == "__main__":
    model = Model_0_2_0()
    output_dir = f"./results/{model.output_dir}/test"

    params = [
        {
            "infant_params": InfantParams.from_array([0.5, 0.5, 0.5]),
            "infant_class": model.infant_class,
            "parent_class": model.parent_class,
            "config": Config(),
        }
    ]

    run_basic_simulation(
        output_dir=output_dir,
        parameter_sets=params,
        iterations=100,
        repeats=1,
    )
