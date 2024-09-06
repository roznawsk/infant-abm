import numpy as np
import itertools
import warnings
import os
import platform

from infant_abm import Config, InfantParams
from infant_abm.simulation import (
    Simulation,
    DataCollector,
    Model_0_1_0,  # noqa: F401
    Model_0_1_1,  # noqa: F401
    Model_0_1_2,  # noqa: F401
    Model_0_2_0,  # noqa: F401
)


warnings.simplefilter(action="ignore", category=FutureWarning)


ITERATIONS = 20000
SUCCESS_DIST = 10

is_mac = "macOS" in platform.platform()
PROCESSES = os.cpu_count() - 1 if is_mac else os.cpu_count()


def run_basic_simulation(
    model, run_name, collector, parameter_sets, iterations, repeats=100
):
    simulation = Simulation(
        model=model,
        model_param_sets=parameter_sets,
        iterations=iterations,
        repeats=repeats,
        run_name=run_name,
        datacollector=collector,
        display=True,
        processes=PROCESSES,
    )

    simulation.run()

    return simulation


def run_comparative_boost_simulation(
    model,
    collector,
    run_name,
    iterations,
    repeats,
    linspace,
    boost_linspace=(0, 0, 1),
):
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
        }

        params.append({**base_params, "infant_params": i_params})

    run_basic_simulation(
        run_name=run_name,
        model=model,
        iterations=iterations,
        collector=collector,
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


class v1Collector(DataCollector):
    def __init__(self, model):
        super().__init__(model)
        self.goal_dist_iteration = None

    def after_step(self):
        if self.model.get_middle_dist() < SUCCESS_DIST:
            self.goal_dist_iteration = self.model._steps
            return False
        return True

    def to_dict(self):
        return {
            "goal_dist": self.goal_dist_iteration,
        }


if __name__ == "__main__":
    model = Model_0_1_0()
    collector = v1Collector

    grid = 7
    boost = 1
    repeats = 13
    run_name = "test_collect"

    linspace = (0.35, 0.65, grid)
    boost_linspace = (0, 1, boost)

    run_comparative_boost_simulation(
        model=model,
        iterations=100,
        collector=collector,
        run_name=run_name,
        repeats=repeats,
        linspace=linspace,
    )

    # run_from_description(model, output_dir, repeats)
