import math
import numpy as np

from enum import Enum
from dataclasses import dataclass

from infant_abm.agents.agent import Agent
from infant_abm.agents.position import Position


class Action(Enum):
    CRAWL = 1
    LOOK_FOR_TOY = 2
    INTERACT_WITH_TOY = 3
    EVALUATE_TOY = 4


@dataclass
class Params:
    perception: float
    persistence: float
    coordination: float

    @staticmethod
    def from_array(array):
        p, c, e = array
        return Params(perception=p, persistence=c, coordination=e)

    def to_array(self):
        return np.array([self.perception, self.persistence, self.coordination])


class InfantBase(Agent):
    # Agent constants

    speed = 1
    toy_interaction_range = 2
    toy_throw_range = 10

    distraction_exponent = 1 / 25

    def __init__(self, unique_id, model, pos, params: Params):
        super().__init__(unique_id, model, pos)

        self.params: Params = params
        self.velocity = None
        self.target = None
        self.bonus_target = None
        self.satisfaction = []

        self.next_action = Action.LOOK_FOR_TOY

    def step(self):
        self.satisfaction.append(0)

        self._before_step()

        match self.next_action:
            case Action.CRAWL:
                self._step_crawl()
            case Action.LOOK_FOR_TOY:
                self._step_look_for_toy()
            case Action.INTERACT_WITH_TOY:
                self._step_interact_with_toy()
            case Action.EVALUATE_TOY:
                self._step_evaluate_toy()

    def _before_step(self):
        pass

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

    def _toy_probability(self, toy):
        return np.power(
            (toy.times_interacted_with + 1e-5), 2 * self.params.perception - 1
        )
