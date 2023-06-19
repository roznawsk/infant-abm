import mesa

import itertools
import numpy as np
import pandas as pd
import multiprocessing
import tqdm
import os

from matplotlib import pyplot as plt

from infant_abm.model import InfantModel


class Simulation:
    def __init__(self, model_param_sets, max_iterations, repeats):
        self.parameter_sets = parameter_sets

        self.max_iterations = max_iterations
        self.repeats = repeats

        self.result = None

    def _single_run_param_set(self, param_set):
        model = InfantModel(**param_set)

        goal_dist = []

        for _ in range(self.max_iterations):
            goal_dist.append(model.get_middle_dist())

            model.step()

        return {
            'goal_dist': goal_dist,
            'parent': model.parent.satisfaction,
            'infant': model.infant.satisfaction
        }

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

    def run(self):
        pool = multiprocessing.Pool()
        result = []

        for res in tqdm.tqdm(
                pool.imap_unordered(self._run_param_set, self.parameter_sets, chunksize=1),
                total=len(parameter_sets)):
            result.append(res)

        results = pool.map(self._run_param_set, parameter_sets)
        self.results = results
        return self.results


def get_model_param_sets(default_params):
    prec = np.linspace(20, 100, 2)
    exp = np.linspace(0, 100, 2)
    coord = np.linspace(0, 100, 2)
    resp = np.linspace(0, 100, 1)
    rel = np.linspace(0, 100, 1)

    params = []

    for param_set in itertools.product(*[prec, exp, coord, resp, rel]):
        p, e, c, rs, rl = param_set

        params.append({**default_params, **{
            'precision': p,
            'exploration': e,
            'coordination': c,
            'responsiveness': rs,
            'relevance': rl
        }})

    return params


if __name__ == '__main__':
    grid_size = 300
    repeats = 10
    max_iter = 5000

    output_path = '../results/test_run_temp.hdf'

    if os.path.exists(output_path):
        raise ValueError("Output path already exists")

    default_model_params = {
        'width': grid_size,
        'height': grid_size,
        'speed': 2,
        'lego_count': 4,
        'precision': 50,
        'exploration': 50,
        'coordination': 50,
        'responsiveness': 50,
        'relevance': 50
    }

    columns = list(default_model_params.keys()) + ['repeats', 'max_iter', 'goal_distance',
                                                   'parent_satisfaction', 'infant_satisfaction']

    parameter_sets = get_model_param_sets(default_model_params)
    n_runs = len(parameter_sets)

    file_size = 8 * max_iter * 3 * n_runs / 1024 / 1024
    print(f'Runs no: {n_runs}, estimated file size: {file_size:.2f}MB')

    simulation = Simulation(model_param_sets=parameter_sets, max_iterations=max_iter, repeats=repeats)
    result = simulation.run()
    out_df = pd.DataFrame(result, columns=columns)

    out_df.to_hdf(output_path, 'hdfkey')
