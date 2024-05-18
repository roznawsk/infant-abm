import numpy as np

from infant_abm.agents.parent_base import ParentBase, Action
from infant_abm.agents.infant.events import ToyThrown


class MoverParent(ParentBase):
    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model, pos)

    def _handle_event_toy_thrown(self, event: ToyThrown):
        if self.responsiveness > np.random.rand():
            if self.relevant_response_probability > np.random.rand():
                self._respond_relevant(event.toy)
            else:
                self._respond_irrelevant()
            self.next_action = Action.FETCH_TOY

    def _respond_relevant(self, toy):
        self.target = toy

    def _respond_irrelevant(self):
        toys = self.model.get_toys()

        probabilities = np.array([1 for _ in toys])
        probabilities = probabilities / probabilities.sum()

        [target] = np.random.choice(toys, size=1, p=probabilities)
        self.target = target
