import numpy as np

from infant_abm.agents.parent_base import ParentBase


class VisionOnlyParent(ParentBase):
    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model, pos)

    def respond(self, toy):
        if self.responsiveness > np.random.rand():
            self.rotate_towards(self.model.infant.pos)
