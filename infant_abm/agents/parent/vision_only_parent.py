import numpy as np

from infant_abm.agents.parent_base import ParentBase, Action
from infant_abm.agents.infant.events import ToyThrown, ToySelected, ThrowEvaluation


class VisionOnlyParent(ParentBase):
    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model, pos)

    def _handle_event_toy_thrown(self, event: ToyThrown):
        if self.responsiveness > np.random.rand():
            self.rotate_towards(self.model.infant.pos)

            if self.relevant_response_probability > np.random.rand():
                self._find_toy_nearby()

    def _handle_event_throw_evaluation(self, event: ThrowEvaluation):
        if self.relevant_response_probability > np.random.rand():
            self.rotate_towards(self.model.infant.pos)

    def _handle_event_toy_selected(self, event: ToySelected):
        if self.relevant_response_probability > np.random.rand():
            self.rotate_towards(self.model.infant.pos)

    def _find_toy_nearby(self):
        toys = self.model.get_toys(self.pos, self.TOY_INTERACTION_RANGE)

        if not toys:
            return

        [target] = np.random.choice(toys, size=1)
        self.target = target
        self.next_action = Action.PASS_TOY
