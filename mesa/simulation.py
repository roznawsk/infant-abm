from dataclasses import dataclass

import multiprocessing
import os
import numpy as np
import pandas as pd
import tqdm


from infant_abm.model import InfantModel
from infant_abm.utils import moving_average


@dataclass
class RunResult:
    parameter_set: dict
    repeats: int
    max_iterations: int
    goal_dist: np.ndarray
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
            "width",
            "height",
            "toy_count",
            "responsiveness",
            "relevance",
            "precision",
            "coordination",
            "exploration",
            "repeats",
            "max_iter",
            "goal_distance",
            "parent_tps",
            "infant_tps",
        ]

    def to_list(self):
        return (
            [
                self.parameter_set["width"],
                self.parameter_set["height"],
                self.parameter_set["toy_count"],
                self.parameter_set["responsiveness"],
                self.parameter_set["relevance"],
            ]
            + list(self.parameter_set["infant_params"].to_numpy())
            + [
                self.repeats,
                self.max_iterations,
                self.goal_dist,
                self.parent_tps,
                self.infant_tps,
            ]
        )


class Simulation:
    def __init__(
        self, model_param_sets, max_iterations, repeats, output_path=None, display=False
    ):
        self.parameter_sets = model_param_sets

        self.max_iterations: int = max_iterations
        self.repeats: int = repeats

        self.display = display

        if output_path is not None and os.path.exists(output_path):
            raise ValueError("Output path already exists")

        self.output_path = output_path

        self.results: list[RunResult] = None

    def run(self):
        n_runs = len(self.parameter_sets)

        file_size = 8 * self.max_iterations * 3 * n_runs / 1024 / 1024
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

        out_df.to_hdf(self.output_path, "hdfkey")

    def _run_param_set(self, param_set):
        run_results = []

        for _ in range(self.repeats):
            run_results.append(self._single_run_param_set(param_set))

        return RunResult(
            parameter_set=param_set,
            repeats=self.repeats,
            max_iterations=self.max_iterations,
            goal_dist=np.average([s["goal_dist"] for s in run_results], axis=0),
            parent_tps=np.average([s["parent"] for s in run_results], axis=0),
            infant_tps=np.average([s["infant"] for s in run_results], axis=0),
        )

    def _single_run_param_set(self, param_set):
        model = InfantModel(**param_set)

        goal_dist = []

        for _ in range(self.max_iterations):
            model.step()
            goal_dist.append(model.get_middle_dist())

        return {
            "goal_dist": goal_dist,
            "parent": model.parent.satisfaction,
            "infant": model.infant.satisfaction,
        }
