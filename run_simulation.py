import numpy as np
import itertools
import warnings
import os

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


ITERATIONS = 20000


def get_model_param_sets(model, linspace, base_params=dict()):
    lo, hi, num = linspace
    perception = np.linspace(lo, hi, num)
    persistence = np.linspace(lo, hi, num)
    coordination = np.linspace(lo, hi, num)

    params = []

    for param_set in itertools.product(*[perception, persistence, coordination]):
        i_params = InfantParams.from_array(param_set)

        params.append(
            {
                **base_params,
                "infant_params": i_params,
                "infant_class": model.infant_class,
                "parent_class": model.parent_class,
            }
        )

    return params


def run_basic_simulation(output_dir, parameter_sets, repeats=100):
    simulation = Simulation(
        model_param_sets=parameter_sets,
        iterations=ITERATIONS,
        repeats=repeats,
        output_dir=output_dir,
        display=True,
        processes=os.cpu_count() - 1,
    )

    simulation.run()

    return simulation


def run_comparative_boost_simulation(
    model, repeats, output_dir, linspace, boost_linspace
):
    output_dir = f"./results/{model.output_dir}/{output_dir}"
    Path(output_dir).mkdir(parents=True, exist_ok=False)

    perception, persistence, coordination = [
        np.round(np.linspace(*linspace), 3) for _ in range(3)
    ]

    boost = np.linspace(*boost_linspace)
    params = []

    for param_set in itertools.product(*[perception, persistence, coordination, boost]):
        prc, prs, crd, bst = param_set

        i_params = InfantParams.from_array([prc, prs, crd])
        base_params = {
            "config": Config(persistence_boost_value=bst, coordination_boost_value=bst),
            "infant_class": model.infant_class,
            "parent_class": model.parent_class,
        }

        params.append({**base_params, "infant_params": i_params})

    run_basic_simulation(
        output_dir=output_dir,
        parameter_sets=params,
        repeats=repeats,
    )


def run_from_description(model, output_dir, repeats):
    output_dir = f"./results/{model.output_dir}/{output_dir}"

    simulation = Simulation.from_description(
        output_dir=output_dir,
        iterations=ITERATIONS,
        repeats=repeats,
        display=True,
        processes=os.cpu_count() - 1,
    )

    simulation.run()

    return simulation


if __name__ == "__main__":
    model = Model_0_1_1()

    grid = 3
    boost = 1
    repeats = 1
    output_dir = f"success_{grid}_grid_{boost}_boost_test_no_array"

    linspace = (0.05, 0.95, grid)
    boost_linspace = (0, 1, boost)

    run_comparative_boost_simulation(
        model, repeats, output_dir, linspace, boost_linspace
    )

    # run_from_description(model, output_dir, repeats)
