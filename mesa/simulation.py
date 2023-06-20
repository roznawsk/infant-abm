import numpy as np
import pandas as pd
import multiprocessing
import tqdm
import os

from matplotlib import pyplot as plt

from dataclasses import dataclass

from infant_abm.model import InfantModel
from infant_abm.genetic_model.infant_genome import InfantGenome


@dataclass
class RunResult:
    parameter_set: dict
    repeats: int
    max_iterations: int
    goal_dist: list[float]
    parent_tps: list[int]
    infant_tps: list[int]


class Simulation:
    def __init__(self, model_param_sets, max_iterations, repeats, output_path=None):
        self.parameter_sets = model_param_sets

        self.max_iterations: int = max_iterations
        self.repeats: int = repeats

        if os.path.exists(output_path):
            raise ValueError("Output path already exists")

        self.output_path = output_path

        self.result: list[RunResult] = None

    def run(self):
        n_runs = len(self.parameter_sets)

        file_size = 8 * self.max_iterations * 3 * n_runs / 1024 / 1024
        print(f'Runs no: {n_runs}, estimated output size: {file_size:.2f}MB')

        pool = multiprocessing.Pool()
        result = []

        for res in tqdm.tqdm(
                pool.imap_unordered(self._run_param_set, self.parameter_sets, chunksize=1),
                total=len(self.parameter_sets)):
            result.append(res)

        self.results = pool.map(self._run_param_set, self.parameter_sets)
        return self.results

    def save(self):
        columns = list(self.parameter_sets[0].keys()) + \
            ['repeats', 'max_iter', 'goal_distance',
             'parent_satisfaction', 'infant_satisfaction']

        out_df = pd.DataFrame(self.results, columns=columns)
        print(out_df)

        out_df.to_hdf(self.output_path, 'hdfkey')

    def _run_param_set(self, param_set):
        run_results = []

        for _ in range(self.repeats):
            run_results.append(self._single_run_param_set(param_set))

        run_results = {
            'goal_dist': np.average([s['goal_dist'] for s in run_results], axis=0),
            'parent': np.average([s['parent'] for s in run_results], axis=0),
            'infant': np.average([s['infant'] for s in run_results], axis=0)
        }

        return list(param_set.values()) \
            + [self.repeats, self.max_iterations] \
            + [run_results['goal_dist'],
               run_results['parent'],
               run_results['infant']]

    def _single_run_param_set(self, param_set):
        model = InfantModel(**param_set)

        goal_dist = []

        for _ in range(self.max_iterations):
            model.step()
            goal_dist.append(model.get_middle_dist())

        return {
            'goal_dist': goal_dist,
            'parent': model.parent.satisfaction,
            'infant': model.infant.satisfaction
        }
