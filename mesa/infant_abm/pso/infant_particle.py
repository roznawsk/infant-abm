import numpy as np
from infant_abm.agents.infant import Params
from infant_abm.utils import correct_out_of_bounds


class InfantParticle():
    def __init__(self, momentum, cognitive_r, social_r, params: Params):
        self.pos = params.to_numpy()

        self.cognitive_r = cognitive_r
        self.social_r = social_r

        self.momentum = momentum
        self.velocity = np.zeros(len(self.pos))

        self.best_fitness = np.inf
        self.best_individual_pos = None

    def move(self, best_global_pos):
        self.velocity = self.momentum * self.velocity \
            + self.cognitive_r * (self.best_individual_pos - self.pos) \
            + self.social_r * (best_global_pos - self.pos)

        self.pos += self.velocity
        self.pos = correct_out_of_bounds(self.pos, np.ones(3))

    def set_best_fitness(self, best_fitness):
        self.best_fitness = best_fitness
        self.best_individual_pos = self.pos

    def __repr__(self) -> str:
        return f'particle: pos: {self.pos}, \n\t cogni: {self.cognitive_r:.3f}, social: {self.social_r:.3f} \
            velo: {np.round(self.velocity, 3)}, best: {self.best_fitness} / {np.round(self.best_individual_pos, 3)}'
