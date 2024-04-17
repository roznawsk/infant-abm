from enum import Enum

import numpy as np
import math

from infant_abm.agents.agent import Agent
from infant_abm.agents.position import Position


class Action(Enum):
    WAIT = 1
    FETCH_TOY = 2
    PASS_TOY = 3


class Parent(Agent):
    """ """

    def __init__(self, unique_id, model, pos):
        """
        Create a new Boid flocker agent.

        Args:
        """
        super().__init__(unique_id, model, pos)
        self.speed = 5

        self.velocity = None

        self.toy_interaction_range = 10
        self.toy_throw_range = 20
        self.target = None
        self.bonus_target = None

        self.satisfaction = []

        self.next_action = Action.WAIT

    def step(self):
        """
        Get the Boid's neighbors, compute the new vector, and move accordingly.
        """

        self.satisfaction.append(0)

        if self.next_action == Action.WAIT:
            pass
        elif self.next_action == Action.FETCH_TOY:
            self._step_fetch_toy()
        elif self.next_action == Action.PASS_TOY:
            self._step_pass_toy()

    def _step_fetch_toy(self):
        toys = self.model.get_toys(self.pos, self.toy_interaction_range)

        if self.target in toys:
            self.next_action = Action.PASS_TOY
            return

        self.velocity = Position.calc_norm_vector(self.pos, self.target.pos)
        new_pos = self.pos + self.velocity * self.speed
        self.move_agent(new_pos)

    def _step_pass_toy(self):
        throw_direction = self.pos.calc_norm_vector(self.model.infant.pos) * min(
            self.toy_throw_range, math.dist(self.pos, self.model.infant.pos)
        )

        new_pos = self.pos + throw_direction
        self.target.move_agent(new_pos)

        self.model.infant.bonus_target = self.target
        if self.target == self.bonus_target:
            self.satisfaction[-1] += 1
        self.target = None
        self.bonus_target = None
        self.next_action = Action.WAIT

    def respond(self, toy):
        if self.model.responsiveness > np.random.rand():
            if 0.5 > np.random.rand():
                self._respond_relevant(toy)
            else:
                self._respond_irrelevant()
            self.next_action = Action.FETCH_TOY

    def _respond_relevant(self, toy):
        self.target = toy

    def _respond_irrelevant(self):
        toys = self.model.get_toys(self.model, self.pos)

        probabilities = np.array([1 for _ in toys])
        probabilities = probabilities / probabilities.sum()

        [target] = np.random.choice(toys, size=1, p=probabilities)
        self.target = target
