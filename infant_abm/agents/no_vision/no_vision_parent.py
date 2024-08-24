import math
import numpy as np

from infant_abm.agents.parent import Parent, Action
from infant_abm.agents.events import ToyThrown
from infant_abm.agents.position import Position


class NoVisionParent(Parent):
    ALLOWED_ACTIONS = [Action.WAIT, Action.FETCH_TOY, Action.PASS_TOY]

    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model, pos)

        self.velocity = None
        self.target = None

        self.next_action = Action.WAIT

    def step(self):
        next_action = super()._perform_action(self.next_action)

        assert next_action in self.ALLOWED_ACTIONS

        self.next_action = next_action

    def _step_pass_toy(self):
        throw_direction = Position.calc_norm_vector(
            self.pos, self.model.infant.pos
        ) * min(self.TOY_THROW_RANGE, math.dist(self.pos, self.model.infant.pos))

        new_pos = self.pos + throw_direction
        self.target.move_agent(new_pos)
        self.rotate_towards(new_pos)

        self.target = None
        return Action.WAIT

    def _step_fetch_toy(self):
        toys = self.model.get_toys(self.pos, self.TOY_INTERACTION_RANGE)

        if self.target in toys:
            self.next_action = Action.PASS_TOY
            return

        self.velocity = Position.calc_norm_vector(self.pos, self.target.pos)
        new_pos = self.pos + self.velocity * self.SPEED
        self.move_agent(new_pos)

    def _handle_event_toy_thrown(self, event: ToyThrown):
        if self.responsiveness > np.random.rand():
            if self.relevant_response_probability > np.random.rand():
                self._respond_relevant(event.toy)
            else:
                self._respond_irrelevant()
            return Action.FETCH_TOY

    def _respond_relevant(self, toy):
        self.target = toy

    def _respond_irrelevant(self):
        toys = self.model.get_toys()

        probabilities = np.array([1 for _ in toys])
        probabilities = probabilities / probabilities.sum()

        [target] = np.random.choice(toys, size=1, p=probabilities)
        self.target = target
