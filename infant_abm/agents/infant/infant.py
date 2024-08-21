import math
import numpy as np
import dataclasses

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict


from infant_abm.agents.infant import actions, Parameter
from infant_abm.agents import Agent, Position


@dataclass
class Params:
    perception: Parameter
    persistence: Parameter
    coordination: Parameter

    @staticmethod
    def new(perception, persistence, coordination):
        return Params(
            perception=Parameter(perception),
            persistence=Parameter(persistence),
            coordination=Parameter(coordination),
        )

    @staticmethod
    def from_array(array):
        pc, pr, co = array
        return Params(
            perception=Parameter(pc),
            persistence=Parameter(pr),
            coordination=Parameter(co),
        )

    def to_array(self):
        return np.array([self.perception.e2, self.persistence.e2, self.coordination.e2])

    def to_dict(self):
        return {k: parameter.e2 for k, parameter in asdict(self).items()}

    def reset(self):
        fields = dataclasses.fields(self)
        for f in fields:
            getattr(self, f.name).reset()

    def __repr__(self):
        return f"Params({self.perception}, {self.persistence}, {self.coordination})"

    def __eq__(self, other) -> bool:
        assert isinstance(other, Params)
        return self.__dict__ == other.__dict__


class Infant(Agent, ABC):
    # Agent constants

    SPEED = 1
    TOY_INTERACTION_RANGE = 2
    TOY_THROW_RANGE = 10

    DISTRACTION_EXPONENT = 1 / 25

    def __init__(self, unique_id, model, pos, params: Params):
        super().__init__(unique_id, model, pos)

        self.params: Params = params
        self.velocity = None

    @abstractmethod
    def step(self):
        pass

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
