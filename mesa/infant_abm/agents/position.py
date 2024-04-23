import random
import numpy as np


class Position:
    x_max = None
    y_max = None

    @staticmethod
    def random():
        x = random.random() * Position.x_max
        y = random.random() * Position.y_max

        return np.array([x, y])

    @staticmethod
    def calc_norm_vector(first, second) -> np.ndarray:
        vec = second - first
        norm = np.linalg.norm(vec)
        if norm == 0:
            return np.ones(2)
        else:
            return vec / norm

    @staticmethod
    def correct_out_of_bounds(pos):
        for idx, value in enumerate([Position.x_max, Position.y_max]):
            pos[idx] = max(0, pos[idx])
            pos[idx] = min(value - 1e-10, pos[idx])

        return pos
