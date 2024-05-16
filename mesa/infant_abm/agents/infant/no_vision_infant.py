import math
import numpy as np

from infant_abm.agents.infant_base import InfantBase, Params, Action
from infant_abm.agents.infant.events import ToySelected, ToyThrown
from infant_abm.agents.position import Position


class NoVisionInfant(InfantBase):
    def __init__(self, unique_id, model, pos, params: Params):
        super().__init__(unique_id, model, pos, params)

    def _step_interact_with_toy(self):
        throw_direction = None

        if self.params.coordination > np.random.rand():
            parent_dist = math.dist(self.pos, self.model.parent.pos)
            throw_range = min(self.toy_throw_range, parent_dist)
            throw_direction = (
                Position.calc_norm_vector(self.pos, self.model.parent.pos) * throw_range
            )
        else:
            throw_direction = np.random.rand(2)
            throw_direction = (
                throw_direction / np.linalg.norm(throw_direction) * self.toy_throw_range
            )

        new_pos = self.target.pos + throw_direction
        self.target.move_agent(new_pos)
        self.rotate_towards(new_pos)

        self.model.parent.handle_event(ToyThrown(self.target))

        self.target.interact()
        self.model.parent.bonus_target = self.target
        if self.target == self.bonus_target:
            self.satisfaction[-1] += 1
        self.target = None
        self.bonus_target = None

        self.next_action = Action.LOOK_FOR_TOY

    def _step_look_for_toy(self):
        toys = self.model.get_toys()

        probabilities = np.array([self._toy_probability(toy) for toy in toys])
        probabilities /= probabilities.sum()

        [target] = np.random.choice(toys, size=1, p=probabilities)
        self.velocity = Position.calc_norm_vector(self.pos, target.pos)
        self.target = target
        self.next_action = Action.CRAWL
        self.model.parent.handle_event(ToySelected(self.target))

    def _gets_distracted(self):
        if self.params.persistence == 0:
            return True
        return self.params.persistence**self.distraction_exponent < np.random.rand()
