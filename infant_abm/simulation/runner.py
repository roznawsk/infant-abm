import itertools
import os
import platform
import warnings

import numpy as np

from infant_abm import Config, InfantParams
from infant_abm.simulation.simulation import Simulation
from infant_abm.simulation import collectors

warnings.simplefilter(action="ignore", category=FutureWarning)


is_mac = "macOS" in platform.platform()
PROCESSES = os.cpu_count() - 1 if is_mac else os.cpu_count()


def run_basic_simulation(
    model,
    run_name,
    collector,
    parameter_sets,
    iterations,
    repeats,
    chunksize,
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
        chunksize=chunksize,
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
    q_learn_params=[None],
    chunksize=10,
):
    perception, persistence, coordination = [
        np.round(np.linspace(*linspace), 3) for _ in range(3)
    ]

    boost = np.linspace(*boost_linspace)
    params_list = []

    for param_set in itertools.product(
        *[perception, persistence, coordination, boost, q_learn_params]
    ):
        prc, prs, crd, bst, q_lrn = param_set

        i_params = InfantParams.from_array([prc, prs, crd])
        base_params = {
            "config": Config(persistence_boost_value=bst, coordination_boost_value=bst),
        }

        params = {
            **base_params,
            "infant_params": i_params,
        }

        if q_lrn is not None:
            a, g, e = q_lrn
            kwargs = {"alpha": a, "gamma": g, "epsilon": e}

            params = {**params, "infant_kwargs": kwargs}

        params_list.append(params)

    run_basic_simulation(
        run_name=run_name,
        model=model,
        iterations=iterations,
        collector=collector,
        parameter_sets=params_list,
        repeats=repeats,
        chunksize=chunksize,
    )


def run_from_description(model, output_dir, repeats):
    output_dir = f"./results/{model.output_dir}/{output_dir}"

    simulation = Simulation.from_description(
        output_dir=output_dir,
        iterations=collectors.ITERATIONS,
        repeats=repeats,
        display=True,
        processes=os.cpu_count() - 1,
    )

    simulation.run()

    return simulation
