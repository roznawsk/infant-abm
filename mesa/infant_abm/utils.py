import numpy as np

from infant_abm.agents.toy import Toy


def calc_norm_vector(p1, p2):
    vec = p2 - p1
    norm = np.linalg.norm(vec)
    if norm == 0:
        return np.ones(2)
    else:
        return vec / norm


def correct_out_of_bounds(pos, dimensions):
    for idx, value in enumerate(dimensions):
        pos[idx] = max(0, pos[idx])
        pos[idx] = min(value - 1e-10, pos[idx])
    return pos


def get_toys(model, pos=None, range=None):
    toys = []

    if range is None:
        toys = model.schedule.agents
    else:
        toys = model.space.get_neighbors(pos, range, False)

    return [a for a in toys if type(a) == Toy]


def moving_average(a, n=3):
    # Add zeros at the beginning, so the result doesn't change dimensions
    a = np.concatenate([([0] * (n - 1)), a])

    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
