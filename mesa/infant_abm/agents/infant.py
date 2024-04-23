from enum import Enum
from dataclasses import dataclass
import math

from infant_abm.agents.agent import Agent

import numpy as np

from infant_abm.agents.position import Position


class Action(Enum):
    CRAWL = 1
    LOOK_FOR_TOY = 2
    INTERACT_WITH_TOY = 3


@dataclass
class Params:
    perception: float
    persistence: float
    coordination: float

    @staticmethod
    def from_array(array):
        p, c, e = array
        return Params(perception=p, persistence=c, coordination=e)

    @staticmethod
    def from_slider(perception, persistence, coordination):
        return Params.from_array(
            np.array([perception, persistence, coordination]) / 100
        )

    def to_array(self):
        return np.array([self.perception, self.persistence, self.coordination])


class Infant(Agent):
    def __init__(self, unique_id, model, pos, params: Params):
        super().__init__(unique_id, model, pos)

        self.speed = 1

        self.params: Params = params

        self.velocity = None

        self.toy_interaction_range = 2
        self.toy_throw_range = 10
        self.target = None
        self.bonus_target = None

        self.satisfaction = []

        self.next_action = Action.LOOK_FOR_TOY

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
        if math.dist(self.pos, self.target.pos) < self.toy_interaction_range:
            self.next_action = Action.INTERACT_WITH_TOY
            return

        if self._gets_distracted():
            self.target = None
            self.next_action = Action.LOOK_FOR_TOY
            return

        self.velocity = Position.calc_norm_vector(self.pos, self.target.pos)
        new_pos = self.pos + self.velocity * self.speed
        self.move_agent(new_pos)

    def _step_toy_interaction(self):
        throw_direction = None

        if self.params.coordination < np.random.rand():
            parent_dist = math.dist(self.pos, self.model.parent.pos)
            throw_range = min(self.toy_throw_range, parent_dist)
            throw_direction = (
                Position.calc_norm_vector(self.pos, self.model.parent.pos) * throw_range
            )
        else:
            throw_direction = np.random.rand(2)
            throw_direction = (
                throw_direction / np.linalg.norm(throw_direction) * self.toy_throw_range
            )

        new_pos = self.target.pos + throw_direction
        self.target.move_agent(new_pos)

        self.model.parent.respond(self.target)

        self.target.interact()
        self.model.parent.bonus_target = self.target
        if self.target == self.bonus_target:
            self.satisfaction[-1] += 1
        self.target = None
        self.bonus_target = None

        self.next_action = Action.LOOK_FOR_TOY

    def _step_change_target(self):
        toys = self.model.get_toys()

        probabilities = np.array([self._toy_probability(toy) for toy in toys])
        probabilities /= probabilities.sum()

        [target] = np.random.choice(toys, size=1, p=probabilities)
        self.velocity = Position.calc_norm_vector(self.pos, target.pos)
        self.target = target

        self.next_action = Action.CRAWL

    def _toy_probability(self, toy):
        return np.power((toy.times_interacted_with + 1), 1 - 2 * self.params.perception)

    def _gets_distracted(self):
        if self.params.persistence == 1:
            return False
        return (1 - self.params.persistence) ** (1 / 25) < np.random.rand()
