from enum import Enum
from dataclasses import dataclass
import math

from infant_abm.agents.agent import Agent

import numpy as np

from infant_abm.agents.position import Position


class Action(Enum):
    CRAWL = 1
    LOOK_FOR_TOY = 2
    INTERACT_WITH_TOY = 3


@dataclass
class Params:
    perception: float
    persistence: float
    coordination: float

    @staticmethod
    def from_array(array):
        p, c, e = array
        return Params(perception=p, persistence=c, coordination=e)

    def to_array(self):
        return np.array([self.perception, self.persistence, self.coordination])


class Infant(Agent):
    # Agent constants

    speed = 1
    toy_interaction_range = 2
    toy_throw_range = 10

    sight_angle = 10 / 180 * np.pi
    sight_exploitation_gain = 0.05
    explore_exploit_ratio_reset_steps = 15

    distraction_exponent = 1 / 25

    def __init__(self, unique_id, model, pos, params: Params):
        super().__init__(unique_id, model, pos)

        self.params: Params = params
        self.explore_exploit_ratio = 0.5
        self.parent_visible = False
        self.steps_since_eye_contact = 0
        self.velocity = None
        self.target = None
        self.bonus_target = None
        self.satisfaction = []

        self.next_action = Action.LOOK_FOR_TOY

    def step(self):
        self.satisfaction.append(0)

        self._update_parent_visible()
        self._update_explore_exploit_ratio()

        match self.next_action:
            case Action.CRAWL:
                self._step_crawl()
            case Action.LOOK_FOR_TOY:
                self._step_change_target()
            case Action.INTERACT_WITH_TOY:
                self._step_toy_interaction()

    def _step_crawl(self):
        if math.dist(self.pos, self.target.pos) < self.toy_interaction_range:
            self.next_action = Action.INTERACT_WITH_TOY
            return

        if self._gets_distracted():
            self.target = None
            self.next_action = Action.LOOK_FOR_TOY
            return

        self.velocity = Position.calc_norm_vector(self.pos, self.target.pos)
        new_pos = self.pos + self.velocity * self.speed
        self.move_agent(new_pos)

    def _step_toy_interaction(self):
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

        self.model.parent.respond(self.target)

        self.target.interact()
        self.model.parent.bonus_target = self.target
        if self.target == self.bonus_target:
            self.satisfaction[-1] += 1
        self.target = None
        self.bonus_target = None

        self.next_action = Action.LOOK_FOR_TOY

    def _step_change_target(self):
        toys = self.model.get_toys()

        probabilities = np.array([self._toy_probability(toy) for toy in toys])
        probabilities /= probabilities.sum()

        [target] = np.random.choice(toys, size=1, p=probabilities)
        self.velocity = Position.calc_norm_vector(self.pos, target.pos)
        self.target = target

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

    def _update_explore_exploit_ratio(self):
        if self.parent_visible:
            self.steps_since_eye_contact = 0
            self.explore_exploit_ratio = min(
                1.0, self.explore_exploit_ratio + self.sight_exploitation_gain
            )
        else:
            self.steps_since_eye_contact += 1
            if self.steps_since_eye_contact == self.explore_exploit_ratio_reset_steps:
                self.explore_exploit_ratio = 0.5
