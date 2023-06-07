import mesa
import numpy as np
from enum import Enum

from utils import *


class Action(Enum):
    WAIT = 1
    FETCH_TOY = 2
    PASS_TOY = 3


class Parent(mesa.Agent):
    """
    """

    def __init__(
        self,
        unique_id,
        model,
        pos,
        speed
    ):
        """
        Create a new Boid flocker agent.

        Args:
        """
        super().__init__(unique_id, model)
        self.pos = np.array(pos)
        self.speed = speed

        self.velocity = None

        self.toy_interaction_range = 20
        self.toy_throw_range = 40
        self.target = None
        self.bonus_target = None

        self.satisfaction = 0

        self.next_action = Action.WAIT

    def step(self):
        """
        Get the Boid's neighbors, compute the new vector, and move accordingly.
        """

        if self.next_action == Action.WAIT:
            pass
        elif self.next_action == Action.FETCH_TOY:
            self._step_fetch_toy()
        elif self.next_action == Action.PASS_TOY:
            self._step_pass_toy()

    def _step_fetch_toy(self):
        toys = get_toys(self.model, self.pos, self.toy_interaction_range)

        if self.target in toys:
            self.next_action = Action.PASS_TOY
            return

        self.velocity = calc_norm_vector(self.pos, self.target.pos)
        new_pos = self.pos + self.velocity * self.speed
        new_pos = correct_out_of_bounds(new_pos, self.model.space)

        self.model.space.move_agent(self, new_pos)

    def _step_pass_toy(self):
        throw_direction = calc_norm_vector(self.pos, self.model.toddler.pos) \
            * min(self.toy_throw_range, calc_dist(self.pos, self.model.toddler.pos))

        new_pos = self.pos + throw_direction
        new_pos = correct_out_of_bounds(new_pos, self.model.space)

        # print(f'throw = {throw_direction}, {type(self.target.pos)}, {self.target.pos}, {new_pos}')

        self.model.space.move_agent(self.target, new_pos)

        self.model.toddler.bonus_target = self.target
        if self.target == self.bonus_target:
            self.satisfaction += 1
        self.target = None
        self.bonus_target = None
        self.next_action = Action.WAIT

    def respond(self, toy):
        if self.model.responsiveness > np.random.rand():
            if self.model.relevance > np.random.rand():
                self._respond_relevant(toy)
            else:
                self._respond_irrelevant()
            self.next_action = Action.FETCH_TOY

    def _respond_relevant(self, toy):
        self.target = toy

    def _respond_irrelevant(self):
        toys = get_toys(self.model, self.pos)

        probabilities = np.array(
            [(1 / (calc_dist(toy.pos, self.pos) + 0.01)) for toy in toys])

        probabilities = probabilities / probabilities.sum()

        [target] = np.random.choice(toys, size=1, p=probabilities)
        self.target = target
