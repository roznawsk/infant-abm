import numpy as np
import itertools
import warnings
import os
import platform

from infant_abm import Config, InfantParams
from infant_abm.simulation import (
    Simulation,
    DataCollector,
    Model_0_2_0,  # noqa: F401
    Model_0_2_1,  # noqa: F401
)

warnings.simplefilter(action="ignore", category=FutureWarning)

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
    model, collector, run_name, iterations, repeats, linspace, q_learn_params
):
    perception, persistence, coordination = [
        np.round(np.linspace(*linspace), 3) for _ in range(3)
    ]

    params = []

    for param_set in itertools.product(
        *[perception, persistence, coordination, q_learn_params]
    ):
        prc, prs, crd, q_lrn = param_set

        i_params = InfantParams.from_array([prc, prs, crd])
        base_params = {
            "config": Config(),
        }

        a, g, e = q_lrn
        kwargs = {"alpha": a, "gamma": g, "epsilon": e}

        params.append(
            {**base_params, "infant_params": i_params, "infant_kwargs": kwargs}
        )

    run_basic_simulation(
        run_name=run_name,
        model=model,
        iterations=iterations,
        collector=collector,
        parameter_sets=params,
        repeats=repeats,
    )


def run_from_description(model, iterations, output_dir, repeats):
    output_dir = f"./results/{model.output_dir}/{output_dir}"

    simulation = Simulation.from_description(
        output_dir=output_dir,
        iterations=iterations,
        repeats=repeats,
        display=True,
        processes=os.cpu_count() - 1,
    )

    simulation.run()

    return simulation


class v2Collector(DataCollector):
    def __init__(self, model):
        super().__init__(model)

        self.rewards = []

    def after_step(self):
        self.rewards.append(self.model.infant.last_reward)
        return True

    def to_dict(self):
        return {
            "rewards": np.array(self.rewards),
            "q_table": self.model.infant.q_learning_agent.q_table,
        }


if __name__ == "__main__":
    model = Model_0_2_1()
    collector = v2Collector

    iterations = 100_00
    grid = 2
    repeats = 10
    run_name = "test_q_learnv2_10k"
    q_learn_params = list(
        itertools.product(*[[0.08, 0.13], [0.5, 0.7, 0.8], [0.005, 0.01, 0.05]])
    )

    linspace = (0.3, 0.7, grid)

    run_comparative_boost_simulation(
        model=model,
        iterations=iterations,
        collector=collector,
        run_name=run_name,
        repeats=repeats,
        linspace=linspace,
        q_learn_params=q_learn_params,
    )

    # run_from_description(model, output_dir, repeats)
