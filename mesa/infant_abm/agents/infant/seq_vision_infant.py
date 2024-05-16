import math
import numpy as np

from infant_abm.agents.infant_base import InfantBase, Params, Action
from infant_abm.agents.infant.events import ToySelected, ToyThrown
from infant_abm.agents.position import Position


class SeqVisionInfant(InfantBase):
    # Agent constants

    toy_evaluation_steps = 3

    def __init__(self, unique_id, model, pos, params: Params):
        super().__init__(unique_id, model, pos, params)

        self.parent_visible = False
        self.explore_exploit_ratio = 0.5
        self.steps_since_eye_contact = 0
        self.current_evaluation_steps = 0

    def _before_step(self):
        self._update_parent_visible()

    def _step_interact_with_toy(self):
        throw_direction = None
        coordination = self._get_updated_param(self.params.coordination)

        if coordination > np.random.rand():
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

    def _step_look_for_toy(self):
        toys = self.model.get_toys()

        probabilities = np.array([self._toy_probability(toy) for toy in toys])
        probabilities /= probabilities.sum()

        [target] = np.random.choice(toys, size=1, p=probabilities)
        self.velocity = Position.calc_norm_vector(self.pos, target.pos)
        self.target = target
        self.rotate_towards(target.pos)

        self.current_evaluation_steps = 0
        self.explore_exploit_ratio = 0.5

        self._start_evaluating_toy()

    def _step_evaluate_toy(self):
        self.current_evaluation_steps += 1

        if self.parent_visible and self.model.parent.infant_visible:
            self.explore_exploit_ratio = 1.0
            self.next_action = Action.CRAWL
        elif self.current_evaluation_steps == self.toy_evaluation_steps:
            self.next_action = Action.CRAWL

    def _toy_probability(self, toy):
        perception = self._get_updated_param(self.params.perception)
        return np.power((toy.times_interacted_with + 1e-5), 2 * perception - 1)

    def _gets_distracted(self):
        persistence = self._get_updated_param(self.params.persistence)

        if persistence == 0:
            return True
        return persistence**self.distraction_exponent < np.random.rand()

    def _get_updated_param(self, value):
        if self.explore_exploit_ratio >= 0.5:
            return value + (1 - value) * (2 * self.explore_exploit_ratio - 1)
        else:
            return 2 * value * self.explore_exploit_ratio

    def _update_parent_visible(self):
        parent_angle = Position.angle(self.pos, self.model.parent.pos)
        self.parent_visible = abs(parent_angle - self.direction) < self.sight_angle

    def _start_evaluating_toy(self):
        self.next_action = Action.EVALUATE_TOY

        if 0.5 > np.random.rand():
            self.rotate_towards(self.model.parent.pos)

            self.model.parent.handle_event(ToySelected(self.target))
