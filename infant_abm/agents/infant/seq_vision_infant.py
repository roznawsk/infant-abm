import math
import numpy as np

from infant_abm.agents.infant.infant_base import InfantBase, Params
from infant_abm.agents.infant.events import ToySelected, ToyThrown, ThrowEvaluation
from infant_abm.agents.position import Position
from infant_abm.agents.infant import actions

from infant_abm.utils import chance


class SeqVisionInfant(InfantBase):
    # Agent constants

    TOY_EVALUATION_DURATION = 3
    THROW_EVALUATION_DURATION = 20

    PERSISTENCE_BOOST_DURATION = 20

    COORDINATION_BOOST_VALUE = 0.2
    PERSISTENCE_BOOST_VALUE = 0.2

    TOY_EVALUATION_PARENT_CHANCE = 0.7
    TOY_EVALUATION_INFANT_CHANCE = 0.7

    THROW_EVALUATION_PARENT_CHANCE = 0.7
    THROW_EVALUATION_INFANT_CHANCE = 0.7

    def __init__(self, unique_id, model, pos, params: Params):
        super().__init__(unique_id, model, pos, params)

        self.parent_visible = False

        self.current_persistence_boost_duration = 0

    def _before_step(self):
        pass

    def _step_look_for_toy(self, _action):
        self.current_persistence_boost_duration = 0
        self.params.persistence.reset()

        toys = self.model.get_toys()

        probabilities = np.array([self._toy_probability(toy) for toy in toys])
        probabilities /= probabilities.sum()

        [target] = np.random.choice(toys, size=1, p=probabilities)
        self.velocity = Position.calc_norm_vector(self.pos, target.pos)
        self.target = target

        return actions.EvaluateToy()

    def _step_evaluate_toy(self, action: actions.EvaluateToy):
        if self.parent_visible and self.model.parent.infant_visible:
            self.params.persistence.boost(self.PERSISTENCE_BOOST_VALUE)

            self._reset_visible()
            return actions.Crawl(metadata="persistence_boost")
        elif action.duration == self.TOY_EVALUATION_DURATION:
            self._reset_visible()
            return actions.Crawl(metadata="no_boost")
        else:
            if chance(self.TOY_EVALUATION_INFANT_CHANCE, self.TOY_EVALUATION_DURATION):
                self.parent_visible = True

            if chance(self.TOY_EVALUATION_PARENT_CHANCE, self.TOY_EVALUATION_DURATION):
                self.model.parent.handle_event(ToySelected(self.target))

            return actions.EvaluateToy(action.duration + 1)

    def _step_interact_with_toy(self, _action):
        self.params.coordination.reset()
        throw_direction = None

        if self.params.coordination.e2 > np.random.rand():
            parent_dist = math.dist(self.pos, self.model.parent.pos)
            throw_range = min(self.TOY_THROW_RANGE, parent_dist)
            throw_direction = (
                Position.calc_norm_vector(self.pos, self.model.parent.pos) * throw_range
            )
        else:
            throw_angle = np.random.uniform(0, 2 * np.pi)
            throw_direction = np.array([np.cos(throw_angle), np.sin(throw_angle)])
            throw_direction *= self.TOY_THROW_RANGE

        new_pos = self.target.pos + throw_direction
        self.target.move_agent(new_pos)

        self.target.interact()
        self.model.parent.handle_event(ToyThrown(self.target))

        self.model.parent.bonus_target = self.target
        if self.target == self.bonus_target:
            self.satisfaction[-1] += 1
        self.target = None
        self.bonus_target = None

        return actions.LookForToy()

    def _step_crawl(self, _action):
        if self._target_in_range():
            self._start_evaluating_throw()
            return actions.EvaluateThrow()

        if self._gets_distracted():
            self.target = None
            return actions.LookForToy()

        self._move()

        self.current_persistence_boost_duration += 1
        if self.current_persistence_boost_duration == self.PERSISTENCE_BOOST_DURATION:
            self.params.persistence.reset()

        return actions.Crawl()

    def _step_evaluate_throw(self, action: actions.EvaluateThrow):
        if self.parent_visible and self.model.parent.infant_visible:
            self.params.coordination.boost(self.COORDINATION_BOOST_VALUE)

            self._reset_visible()
            return actions.InteractWithToy(metadata="coordination_boost")
        elif action.duration == self.TOY_EVALUATION_DURATION:
            self._reset_visible()
            return actions.InteractWithToy()
        else:
            if chance(
                self.THROW_EVALUATION_INFANT_CHANCE, self.THROW_EVALUATION_DURATION
            ):
                self.parent_visible = True

            if chance(
                self.THROW_EVALUATION_PARENT_CHANCE, self.THROW_EVALUATION_DURATION
            ):
                self.model.parent.handle_event(ThrowEvaluation())

            return actions.EvaluateThrow(action.duration + 1)

    # Helper functions

    def _start_evaluating_throw(self):
        if 0.5 > np.random.rand():
            self.parent_visible = True
            self.model.parent.handle_event(ThrowEvaluation())

    def _reset_visible(self):
        self.parent_visible = False
        self.model.parent.infant_visible = False
