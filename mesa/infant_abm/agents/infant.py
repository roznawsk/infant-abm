from enum import Enum
from dataclasses import dataclass
import math

from infant_abm.utils import get_toys, calc_norm_vector, correct_out_of_bounds

import numpy as np
import mesa


class Action(Enum):
    CRAWL = 1
    LOOK_FOR_TOY = 2
    INTERACT_WITH_TOY = 3


@dataclass
class Params:
    precision: float
    coordination: float
    exploration: float

    def to_numpy(self):
        return np.array([self.precision, self.coordination, self.exploration])

    @staticmethod
    def from_numpy(nd_array):
        p, c, e = nd_array
        return Params(precision=p, coordination=c, exploration=e)

    @staticmethod
    def random():
        return Params.from_numpy(np.random.random(3))


class Infant(mesa.Agent):
    def __init__(
        self,
        unique_id,
        model,
        pos,
        speed,
        params: Params
    ):
        super().__init__(unique_id, model)
        self.pos = np.array(pos)
        self.speed = speed

        self.params: Params = params

        self.velocity = None

        self.toy_interaction_range = 10
        self.toy_throw_range = 20
        self.target = None
        self.bonus_target = None

        self.satisfaction = []

        self.next_action = Action.LOOK_FOR_TOY

        self.steps_until_distraction = None

    def step(self):
        """
        Get the Boid's neighbors, compute the new vector, and move accordingly.
        """

        self.satisfaction.append(0)

        if self.next_action == Action.CRAWL:
            self._step_crawl()
        elif self.next_action == Action.LOOK_FOR_TOY:
            self._step_change_target()
        elif self.next_action == Action.INTERACT_WITH_TOY:
            self._step_toy_interaction()

    def _step_crawl(self):
        local_toys = get_toys(self.model, self.pos, self.toy_interaction_range)

        if local_toys:
            if self.target in local_toys or self.params.precision < np.random.rand():
                self.target = local_toys[0]
                self.next_action = Action.INTERACT_WITH_TOY
                return

        if self.steps_until_distraction == 0:
            self.target = None

            self.next_action = Action.LOOK_FOR_TOY
            return

        self.velocity = calc_norm_vector(self.pos, self.target.pos)
        new_pos = self.pos + self.velocity * self.speed
        new_pos = correct_out_of_bounds(new_pos, self.model.get_dims())

        if self.steps_until_distraction:
            self.steps_until_distraction -= 1
        self.model.space.move_agent(self, new_pos)

    def _step_toy_interaction(self):
        throw_direction = None

        if self.params.coordination > np.random.rand():
            throw_direction = calc_norm_vector(self.pos, self.model.parent.pos) * self.toy_throw_range
        else:
            throw_direction = np.random.rand(2)
            throw_direction = throw_direction / np.linalg.norm(throw_direction) * self.toy_throw_range

        new_pos = self.pos + throw_direction
        new_pos = correct_out_of_bounds(new_pos, self.model.get_dims())

        self.model.space.move_agent(self.target, new_pos)

        self.model.parent.respond(self.target)

        self.target.times_interacted_with += 1
        self.model.parent.bonus_target = self.target
        if self.target == self.bonus_target:
            self.satisfaction[-1] += 1
        self.target = None
        self.bonus_target = None

        self.next_action = Action.LOOK_FOR_TOY

    def _step_change_target(self):
        toys = get_toys(self.model, self.pos)

        probabilities = np.array([self._toy_probability(toy) for toy in toys])
        probabilities /= probabilities.sum()

        [target] = np.random.choice(toys, size=1, p=probabilities)
        self.velocity = calc_norm_vector(self.pos, target.pos)
        self.target = target

        if self.params.precision > np.random.rand():
            self.steps_until_distraction = None
        else:
            target_dist = math.dist(self.pos, self.target.pos)
            steps_to_target = max(1, np.floor(target_dist - self.toy_interaction_range) / self.speed)
            self.steps_until_distraction = np.random.randint(steps_to_target)

        self.next_action = Action.CRAWL

    def _toy_probability(self, toy):
        return np.power((toy.times_interacted_with + 1), 1 - 2 * self.params.exploration)
