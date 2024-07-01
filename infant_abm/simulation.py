from dataclasses import dataclass

import multiprocessing
import os
import numpy as np
import pandas as pd
import tqdm
import warnings

from copy import deepcopy
from collections import Counter

from infant_abm.model import InfantModel
from infant_abm.utils import moving_average


@dataclass
class RunResult:
    parameter_set: dict
    repeats: int
    iterations: int
    goal_dist: np.ndarray
    actions: np.ndarray
    parent_tps: np.ndarray
    infant_tps: np.ndarray

    def fitness(self, metric, goal_dist=None, average_steps=None):
        if metric == "goal_dist":
            try:
                return np.where(self.goal_dist < goal_dist)[0][0]
            except IndexError:
                return np.NaN

        elif metric == "parent_tps":
            return moving_average(self.parent_tps, average_steps)[-1]

        elif metric == "infant_tps":
            return moving_average(self.infant_tps, average_steps)[-1]

    def get_columns(self):
        return [
            "perception",
            "persistence",
            "coordination",
            "repeats",
            "iterations",
            "goal_distance",
            "actions",
            "parent_tps",
            "infant_tps",
        ]

    def to_list(self):
        return list(self.parameter_set["infant_params"].to_array()) + [
            self.repeats,
            self.iterations,
            self.goal_dist,
            self.actions,
            self.parent_tps,
            self.infant_tps,
        ]


class Simulation:
    def __init__(
        self, model_param_sets, iterations, repeats, output_path=None, display=False
    ):
        self.parameter_sets = model_param_sets

        self.iterations: int = iterations
        self.repeats: int = repeats

        self.display = display

        if output_path is not None and os.path.exists(output_path):
            raise ValueError("Output path already exists")

        self.output_path = output_path

        self.results: list[RunResult] = None

    def run(self):
        n_runs = len(self.parameter_sets)

        file_size = 8 * self.iterations * 3 * n_runs / 1024 / 1024 + 1
        if self.display:
            print(f"Runs no: {n_runs}, estimated output size: {file_size:.2f}MB")

        pool = multiprocessing.Pool()
        results = []

        for res in tqdm.tqdm(
            pool.imap(self._run_param_set, self.parameter_sets, chunksize=1),
            total=len(self.parameter_sets),
            disable=not self.display,
        ):
            results.append(res)

        self.results = results
        return self.results

    def save(self):
        columns = self.results[0].get_columns()
        results_np = [r.to_list() for r in self.results]

        out_df = pd.DataFrame(results_np, columns=columns)

        # We use hdf, because writing arrays into csv is troublesome
        warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
        out_df.to_hdf(self.output_path, key="hdfkey")

    def _run_param_set(self, param_set):
        run_results = []

        result_param_set = deepcopy(param_set)

        for _ in range(self.repeats):
            run_results.append(self._single_run_param_set(param_set))

        goal_dist, infant_actions, infant_satisfaction, parent_satisfaction = zip(
            *run_results
        )

        return RunResult(
            parameter_set=result_param_set,
            repeats=self.repeats,
            iterations=self.iterations,
            goal_dist=np.average(goal_dist, axis=0),
            actions=dict(sum(infant_actions, Counter())),
            parent_tps=np.average(infant_satisfaction, axis=0),
            infant_tps=np.average(parent_satisfaction, axis=0),
        )

    def _single_run_param_set(self, param_set):
        model = InfantModel(**param_set)

        goal_dist = []

        for _ in range(self.iterations):
            model.step()
            goal_dist.append(model.get_middle_dist())

        return (
            goal_dist,
            model.infant.actions,
            model.infant.satisfaction,
            model.parent.satisfaction,
        )
