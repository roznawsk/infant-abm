import mesa

import itertools
import numpy as np
import pandas as pd
import time
import multiprocessing
import tqdm

from ast import literal_eval

from matplotlib import pyplot as plt

from boid_flockers.model import InfantModel
from boid_flockers.SimpleContinuousModule import SimpleCanvas
from boid_flockers.agents.infant import Infant
from boid_flockers.agents.parent import Parent
from boid_flockers.agents.toy import Toy


def single_run_param_set(param_set):
    model = InfantModel(**param_set['model_params'])

    goal_dist = []

    for _ in range(param_set['max_iter']):
        goal_dist.append(model.get_middle_dist())

        model.step()

    return {
        'goal_dist': goal_dist,
        'parent': model.parent.satisfaction,
        'infant': model.infant.satisfaction
    }


def run_param_set(param_set):
    run_results = []

    for _ in range(param_set['repeats']):
        run_results.append(single_run_param_set(param_set))

    run_results = {
        'goal_dist': np.average([s['goal_dist'] for s in run_results], axis=0),
        'parent': np.average([s['parent'] for s in run_results], axis=0),
        'infant': np.average([s['infant'] for s in run_results], axis=0)
    }

    return list(param_set['model_params'].values()) + [param_set['repeats'],
                                                       param_set['max_iter']] + [run_results['goal_dist'],
                                                                                 run_results['parent'],
                                                                                 run_results['infant']]


def perform_simulation(parameter_sets):
    pool = multiprocessing.Pool()
    result = []

    for res in tqdm.tqdm(pool.imap_unordered(run_param_set, parameter_sets, chunksize=1), total=len(parameter_sets)):
        result.append(res)

    results = pool.map(run_param_set, parameter_sets)
    return results


def get_model_param_sets(default_params, sim_params):
    prec = np.linspace(20, 100, 5)
    exp = np.linspace(0, 100, 6)
    coord = np.linspace(0, 100, 6)
    resp = np.linspace(0, 100, 6)
    rel = np.linspace(0, 100, 4)

    params = []

    for param_set in itertools.product(*[prec, exp, coord, resp, rel]):
        p, e, c, rs, rl = param_set

        param_dict = {
            'precision': p,
            'exploration': e,
            'coordination': c,
            'responsiveness': rs,
            'relevance': rl
        }

        params.append({'model_params': {**default_params, **param_dict}, **sim_params})

    return params


if __name__ == '__main__':
    grid_size = 300
    repeats = 100
    max_iter = 5000

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

    sim_params = {
        'max_iter': max_iter,
        'repeats': repeats
    }

    parameter_sets = get_model_param_sets(default_model_params, sim_params)
    n_runs = len(parameter_sets)

    file_size = 8 * max_iter * 3 * n_runs / 1024 / 1024
    print(f'Runs no: {n_runs}, estimated file size: {file_size:.2f}MB')

    result = perform_simulation(parameter_sets)
    out_df = pd.DataFrame(result, columns=columns)

    out_df.to_hdf('results/run_grid_6.hdf', 'hdfkey')
