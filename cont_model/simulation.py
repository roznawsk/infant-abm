import mesa

import itertools
import numpy as np
import time
import multiprocessing


from matplotlib import pyplot as plt


from boid_flockers.model import ToddlerModel
from boid_flockers.SimpleContinuousModule import SimpleCanvas
from boid_flockers.agents.toddler import Toddler
from boid_flockers.agents.parent import Parent
from boid_flockers.agents.toy import Toy


def single_run_param_set(param_set):
    model = ToddlerModel(**param_set['model_params'])

    goal_dist = []

    for _ in range(param_set['max_iter']):
        goal_dist.append(model.get_middle_dist())

        model.step()

    return {
        'goal_dist': goal_dist,
        'parent': model.parent.satisfaction,
        'infant': model.toddler.satisfaction
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

    return run_results


def moving_average(a, n=3):
    a = np.concatenate([([0] * (n - 1)), a])

    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def plot_run_results(param_set, run_results):
    x = list(range(param_set['max_iter']))

    average_over_steps = 500

    step_stats = np.average([s['goal_dist'] for s in run_results], axis=0)
    parent_stats = np.average([s['parent'] for s in run_results], axis=0)
    infant_stats = np.average([s['infant'] for s in run_results], axis=0)

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


if __name__ == '__main__':
    grid_size = 300
    success_dist = 40
    repeats = 50
    max_iter = 10000

    default_model_params = {
        'width': grid_size,
        'height': grid_size,
        'speed': 2,
        'lego_count': 4,
        'precision': 70,
        'exploration': 70,
        'coordination': 70,
        'responsiveness': 70,
        'relevance': 70
    }

    # for current_param in ['precision', 'coordination', 'responsiveness', 'relevance']:
    for current_param in ['coordination']:
        model_params = dict(default_model_params)

        x = np.arange(0, 101, 10)
        model_params[current_param] = 50

        r_steps = []
        r_parent = []
        r_toddler = []
        r_goal_dist = []

        params_results = {'steps': [], 'parent': [], 'toddler': [], 'goal_dist': []}

        print(model_params)

        param_set = {
            'model_params': model_params,
            'max_iter': max_iter,
            'repeats': repeats
        }

        start = time.time()

        model = ToddlerModel(**(param_set['model_params']))

        run_results = perform_parallel_run(param_set)
        plot_run_results(param_set, run_results)

        print('t = {:2f} ms'.format((time.time() - start) * 1000))

        # print(results)

        # r_steps.append(np.average(params_results['steps']))
        # r_parent.append(np.average(params_results['parent']))
        # r_toddler.append(np.average(params_results['toddler']))
        # # r_goal_dist.append()

        # fig, ax1 = plt.subplots()

        # ax1.plot(x, r_steps, linestyle='dashed', marker='s', color='r')
        # ax1.set_xlabel(current_param)
        # ax1.set_ylim(bottom=0, top=1000)
        # ax1.set_ylabel('steps to goal', color='r')

        # ax2 = ax1.twinx()
        # ax2.plot(x, r_parent, linestyle='dashed', marker='s', color='b')
        # ax2.set_ylabel('satisfaction')

        # ax2.plot(x, r_toddler, linestyle='dashed', marker='s', color='orange')
        # ax2.legend(['parent', 'toddler'])
        # ax2.set_ylim(bottom=0)

        # fig.tight_layout()
        # plt.savefig(f'../../plots/big_{current_param}.png', dpi=300)
        # plt.show()