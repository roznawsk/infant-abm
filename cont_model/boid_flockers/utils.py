import numpy as np

from boid_flockers.agents.toy import Toy


def calc_dist(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))


def calc_norm_vector(p1, p2):
    vec = p2 - p1
    norm = np.linalg.norm(vec)
    if norm == 0:
        return np.ones(2)
    else:
        return vec / norm


def correct_out_of_bounds(pos, space):
    for idx, value in enumerate([space.width, space.height]):
        pos[idx] = max(0, pos[idx])
        pos[idx] = min(value - .01, pos[idx])
    return pos


def get_toys(model, pos=None, range=None):
    toys = []

    if range is None:
        toys = model.schedule.agents
    else:
        toys = model.space.get_neighbors(pos, range, False)

    return [a for a in toys if type(a) == Toy]
