import mesa

import itertools
import numpy as np
import pandas as pd
import time
import multiprocessing


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


def worker_function(queue, param_set):
    result = single_run_param_set(param_set)
    queue.put(result)


def perform_parallel_run(param_set):
    pool = multiprocessing.Pool()
    manager = multiprocessing.Manager()
    queue = manager.Queue()

    repeats = param_set['repeats']

    for i in range(repeats):
        pool.apply_async(worker_function, args=(queue, param_set))

    pool.close()

    run_results = []
    for _ in range(repeats):
        result = queue.get()
        run_results.append(result)

    pool.join()

    return {
        'goal_dist': np.average([s['goal_dist'] for s in run_results], axis=0),
        'parent': np.average([s['parent'] for s in run_results], axis=0),
        'infant': np.average([s['infant'] for s in run_results], axis=0)
    }


def moving_average(a, n=3):
    a = np.concatenate([([0] * (n - 1)), a])

    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def plot_run_results(param_set, run_results):
    x = list(range(param_set['max_iter']))

    average_over_steps = 500

    step_stats = run_results['goal_dist']
    parent_stats = run_results['parent']
    infant_stats = run_results['infant']

    parent_stats = moving_average(parent_stats, average_over_steps)
    infant_stats = moving_average(infant_stats, average_over_steps)

    fig, ax1 = plt.subplots(figsize=(16, 10))

    ax1.plot(x, step_stats, color='r')
    ax1.set_ylim(bottom=0, top=150)
    ax1.set_ylabel('goal distance')
    ax1.set_xlabel('step')

    mp = param_set['model_params']

    title = f' \
    lego={mp["lego_count"]},\
    prec={mp["precision"]},\
    exp={mp["exploration"]},\
    coord={mp["coordination"]},\
    resp={mp["responsiveness"]},\
    rel={mp["relevance"]}, \
    avg for {param_set["repeats"]} runs \
    '
    ax1.set_title(title)

    # ax1.plot(x, r_steps, linestyle='dashed', marker='s', color='r')
    # ax1.set_xlabel(current_param)
    # ax1.set_ylim(bottom=0, top=1000)
    # ax1.set_ylabel('steps to goal', color='r')

    ax2 = ax1.twinx()
    ax2.plot(x, parent_stats, color='b')
    ax2.set_ylabel('satisfaction')

    ax2.plot(x, infant_stats, color='orange')
    ax2.legend(['parent', 'infant'])
    ax2.set_ylim(bottom=0)

    ax2.axvline(x=average_over_steps - 1, color='grey', label='axvline - full height')

    fig.tight_layout()
    # plt.savefig(f'../../plots/big_{current_param}.png', dpi=300)
    plt.show()


def get_model_param_sets(default_params):
    prec = np.linspace(0, 100, 2)
    exp = np.linspace(0, 100, 2)
    coord = np.linspace(0, 100, 2)
    resp = np.linspace(0, 100, 2)
    rel = np.linspace(0, 100, 2)

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

        params.append({**default_params, **param_dict})

    return params


if __name__ == '__main__':
    grid_size = 300
    success_dist = 40
    repeats = 50
    max_iter = 1000

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
    result = []

    for model_param_set in get_model_param_sets(default_model_params):

        param_set = {
            'model_params': model_param_set,
            'max_iter': max_iter,
            'repeats': repeats
        }

        start = time.time()

        run_results = perform_parallel_run(param_set)

        print('run t = {:.2f} ms'.format((time.time() - start) * 1000))

        new_entry = list(model_param_set.values()) + [repeats, max_iter] + list(run_results.values())

        result.append(new_entry)

    out_df = pd.DataFrame(result, columns=columns)
    out_df.to_csv('../results/run.csv')

    # for current_param in ['precision', 'coordination', 'responsiveness', 'relevance']:
    # for current_param in ['coordination']:
    #     model_params = dict(default_model_params)

    #     x = np.arange(0, 101, 10)
    #     model_params[current_param] = 50

    #     r_steps = []
    #     r_parent = []
    #     r_infant = []
    #     r_goal_dist = []

    #     params_results = {'steps': [], 'parent': [], 'infant': [], 'goal_dist': []}

    #     plot_run_results(param_set, run_results)

    # print(results)

    # r_steps.append(np.average(params_results['steps']))
    # r_parent.append(np.average(params_results['parent']))
    # r_infant.append(np.average(params_results['infant']))
    # # r_goal_dist.append()

    # fig, ax1 = plt.subplots()

    # ax1.plot(x, r_steps, linestyle='dashed', marker='s', color='r')
    # ax1.set_xlabel(current_param)
    # ax1.set_ylim(bottom=0, top=1000)
    # ax1.set_ylabel('steps to goal', color='r')

    # ax2 = ax1.twinx()
    # ax2.plot(x, r_parent, linestyle='dashed', marker='s', color='b')
    # ax2.set_ylabel('satisfaction')

    # ax2.plot(x, r_infant, linestyle='dashed', marker='s', color='orange')
    # ax2.legend(['parent', 'infant'])
    # ax2.set_ylim(bottom=0)

    # fig.tight_layout()
    # plt.savefig(f'../../plots/big_{current_param}.png', dpi=300)
    # plt.show()
