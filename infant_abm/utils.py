import numpy as np


def moving_average(a, n=3):
    # Add zeros at the beginning, so the result doesn't change dimensions
    a = np.concatenate([([0] * (n - 1)), a])

    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def chance(total_chance, n_steps):
    return 1 - np.power((1 - total_chance), 1.0 / n_steps) > np.random.rand()