import math
import numpy as np

from dataclasses import dataclass

from infant_abm.agents.infant import actions, Parameter
from infant_abm.agents.agent import Agent
from infant_abm.agents.position import Position


@dataclass
class Params:
    perception: Parameter
    persistence: Parameter
    coordination: Parameter

    @staticmethod
    def from_array(array):
        c, s, o = array
        return Params(
            perception=Parameter(c), persistence=Parameter(s), coordination=Parameter(o)
        )

    def to_array(self):
        return np.array([self.perception.e1, self.persistence.e1, self.coordination.e1])


class InfantBase(Agent):
    # Agent constants

    SPEED = 1
    TOY_INTERACTION_RANGE = 2
    TOY_THROW_RANGE = 10

    DISTRACTION_EXPONENT = 1 / 25

    def __init__(self, unique_id, model, pos, params: Params):
        super().__init__(unique_id, model, pos)

        self.params: Params = params
        self.velocity = None
        self.target = None
        self.bonus_target = None
        self.satisfaction = []

        self.next_action = actions.LookForToy()

    def step(self):
        self.satisfaction.append(0)

        self._before_step()

        next_action = self._perform_action(self.next_action)

        assert issubclass(type(next_action), actions.Action)

        self.next_action = next_action

    def _perform_action(self, action):
        match action:
            case actions.LookForToy():
                return self._step_look_for_toy(action)
            case actions.EvaluateToy():
                return self._step_evaluate_toy(action)
            case actions.Crawl():
                return self._step_crawl(action)
            case actions.InteractWithToy():
                return self._step_interact_with_toy(action)
            case actions.EvaluateThrow():
                return self._step_evaluate_throw(action)

    def _before_step(self):
        pass

    def _move(self):
        self.velocity = Position.calc_norm_vector(self.pos, self.target.pos)
        new_pos = self.pos + self.velocity * self.SPEED
        self.move_agent(new_pos)

    def _toy_probability(self, toy):
        return np.power(
            (toy.times_interacted_with + 1e-5), 2 * self.params.perception.e2 - 1
        )

    def _gets_distracted(self):
        if self.params.persistence.e1 == 1:
            return True
        return self.params.persistence.e2**self.DISTRACTION_EXPONENT < np.random.rand()

    def _target_in_range(self):
        return math.dist(self.pos, self.target.pos) < self.TOY_INTERACTION_RANGE
