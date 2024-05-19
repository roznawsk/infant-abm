import math
import numpy as np

from infant_abm.agents.infant_base import InfantBase, Params, Action
from infant_abm.agents.infant.events import ToySelected, ToyThrown
from infant_abm.agents.position import Position


class SeqVisionInfant(InfantBase):
    # Agent constants

    toy_evaluation_steps = 3

    persistence_boost_duration = 20
    boost_value = 0.2

    def __init__(self, unique_id, model, pos, params: Params):
        super().__init__(unique_id, model, pos, params)

        self.parent_visible = False

        self.current_persistence_boost_duration = 0
        self.current_evaluation_steps = 0

    def _before_step(self):
        self._update_parent_visible()

    def _step_look_for_toy(self):
        self.current_persistence_boost_duration = 0
        self.params.persistence.reset()

        toys = self.model.get_toys()

        probabilities = np.array([self._toy_probability(toy) for toy in toys])
        probabilities /= probabilities.sum()

        [target] = np.random.choice(toys, size=1, p=probabilities)
        self.velocity = Position.calc_norm_vector(self.pos, target.pos)
        self.target = target
        self.rotate_towards(target.pos)

        self.current_evaluation_steps = 0
        self._start_evaluating_toy()

    def _step_evaluate_toy(self):
        self.current_evaluation_steps += 1

        if self.parent_visible and self.model.parent.infant_visible:
            self.params.persistence.boost(self.boost_value)
            self.next_action = Action.CRAWL
        elif self.current_evaluation_steps == self.toy_evaluation_steps:
            self.next_action = Action.CRAWL

    def _step_interact_with_toy(self):
        # TODO: add new step, where we try to establish eye contact with the parent
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

        self.next_action = Action.LOOK_FOR_TOY

    def _step_crawl(self):
        super()._step_crawl()

        self.current_persistence_boost_duration += 1
        if self.current_persistence_boost_duration == self.persistence_boost_duration:
            self.params.persistence.reset()

    def _toy_probability(self, toy):
        return np.power(
            (toy.times_interacted_with + 1e-5), 2 * self.params.perception.e1 - 1
        )

    def _gets_distracted(self):
        if self.params.persistence.e1 == 1:
            return True
        return self.params.persistence.e2**self.distraction_exponent < np.random.rand()

    def _update_parent_visible(self):
        parent_angle = Position.angle(self.pos, self.model.parent.pos)
        self.parent_visible = abs(parent_angle - self.direction) < self.sight_angle

    def _start_evaluating_toy(self):
        self.next_action = Action.EVALUATE_TOY

        if 0.5 > np.random.rand():
            self.rotate_towards(self.model.parent.pos)

            self.model.parent.handle_event(ToySelected(self.target))
