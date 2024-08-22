import math
import numpy as np

from infant_abm.agents.parent.parent import Parent, Action
from infant_abm.agents.infant.events import ToyThrown, ToySelected, ThrowEvaluation
from infant_abm.agents import Toy
from infant_abm.agents.position import Position


class AbstractVisionParent(Parent):
    ALLOWED_ACTIONS = [Action.WAIT, Action.PASS_TOY]

    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model, pos)

        self.infant_visible = False
        self.target = None

        self.next_action = Action.WAIT

    def step(self):
        super()._update_infant_visible()

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

    def _handle_event_toy_thrown(self, event: ToyThrown):
        if self.responsiveness > np.random.rand():
            if self.relevant_response_probability > np.random.rand():
                self._find_toy_nearby(event.toy)

    def _handle_event_throw_evaluation(self, event: ThrowEvaluation):
        if self.relevant_response_probability > np.random.rand():
            self.infant_visible = True

    def _handle_event_toy_selected(self, event: ToySelected):
        if self.relevant_response_probability > np.random.rand():
            self.infant_visible = True

    def _find_toy_nearby(self, toy: Toy):
        toys = self.model.get_toys(self.pos, self.TOY_INTERACTION_RANGE)

        if not toys:
            return
        elif toy in toys:
            target = toy
        else:
            [target] = np.random.choice(toys, size=1)
        self.target = target
        self.next_action = Action.PASS_TOY
