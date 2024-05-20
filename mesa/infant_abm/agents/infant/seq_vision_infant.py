import math
import numpy as np

from infant_abm.agents.infant_base import InfantBase, Params
from infant_abm.agents.infant.events import ToySelected, ToyThrown
from infant_abm.agents.position import Position
from infant_abm.agents.infant import actions


class SeqVisionInfant(InfantBase):
    # Agent constants

    TOY_EVALUATION_DURATION = 3
    THROW_EVALUATION_DURATION = 3

    persistence_boost_duration = 20
    boost_value = 0.2

    def __init__(self, unique_id, model, pos, params: Params):
        super().__init__(unique_id, model, pos, params)

        self.parent_visible = False

        self.current_persistence_boost_duration = 0

    def _before_step(self):
        self._update_parent_visible()

    def _step_look_for_toy(self, _action):
        self.current_persistence_boost_duration = 0
        self.params.persistence.reset()

        toys = self.model.get_toys()

        probabilities = np.array([self._toy_probability(toy) for toy in toys])
        probabilities /= probabilities.sum()

        [target] = np.random.choice(toys, size=1, p=probabilities)
        self.velocity = Position.calc_norm_vector(self.pos, target.pos)
        self.target = target
        self.rotate_towards(target.pos)

        self._start_evaluating_toy()
        return actions.EvaluateToy()

    def _step_evaluate_toy(self, action: actions.EvaluateToy):
        if self.parent_visible and self.model.parent.infant_visible:
            self.params.persistence.boost(self.boost_value)
            return actions.Crawl()
        elif action.duration == self.TOY_EVALUATION_DURATION:
            return actions.Crawl()
        else:
            return actions.EvaluateToy(action.duration + 1)

    def _step_interact_with_toy(self, _action):
        throw_direction = None

        if self.params.coordination.e2 > np.random.rand():
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

        self.target.interact()
        self.model.parent.handle_event(ToyThrown(self.target))

        self.model.parent.bonus_target = self.target
        if self.target == self.bonus_target:
            self.satisfaction[-1] += 1
        self.target = None
        self.bonus_target = None

        return actions.LookForToy()

    def _step_crawl(self, action):
        next_action = super()._step_crawl(action)

        self.current_persistence_boost_duration += 1
        if self.current_persistence_boost_duration == self.persistence_boost_duration:
            self.params.persistence.reset()

        return next_action

    # def _step_evaluate_throw(self, action: actions.EvaluateThrow):
    #     if self.parent_visible and self.model.parent.infant_visible:
    #         self.params.coordination.boost(self.boost_value)
    #         return actions.InteractWithToy()
    #     elif action.duration == self.TOY_EVALUATION_DURATION:
    #         return actions.InteractWithToy()
    #     else:
    #         return actions.EvaluateThrow(action.duration + 1)

    # Helper functions

    def _start_evaluating_toy(self):
        self.current_evaluation_steps = 0

        if 0.5 > np.random.rand():
            self.rotate_towards(self.model.parent.pos)
            self.model.parent.handle_event(ToySelected(self.target))

    def _update_parent_visible(self):
        parent_angle = Position.angle(self.pos, self.model.parent.pos)
        self.parent_visible = abs(parent_angle - self.direction) < self.sight_angle
