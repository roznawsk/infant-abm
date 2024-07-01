from enum import Enum

import math

from infant_abm.agents.agent import Agent
from infant_abm.agents.position import Position
from infant_abm.agents.infant.events import ToySelected, ToyThrown, ThrowEvaluation


class Action(Enum):
    WAIT = 1
    FETCH_TOY = 2
    PASS_TOY = 3


class ParentBase(Agent):
    # Agent constants

    responsiveness = 0.5
    relevant_response_probability = 0.5

    SPEED = 5
    TOY_INTERACTION_RANGE = 10
    TOY_THROW_RANGE = 20

    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model, pos)

        self.velocity = None
        self.target = None
        self.bonus_target = None

        self.satisfaction = []
        self.infant_visible = False

        self.next_action = Action.WAIT

    def step(self):
        self.satisfaction.append(0)

        self._update_infant_visible()

        match self.next_action:
            case Action.WAIT:
                pass
            case Action.FETCH_TOY:
                self._step_fetch_toy()
            case Action.PASS_TOY:
                self._step_pass_toy()

    def handle_event(self, event):
        """
        Respond to infant's interaction with a toy
        """
        match event:
            case ToyThrown():
                self._handle_event_toy_thrown(event)
            case ToySelected():
                self._handle_event_toy_selected(event)
            case ThrowEvaluation():
                self._handle_event_throw_evaluation(event)

    def _step_fetch_toy(self):
        toys = self.model.get_toys(self.pos, self.TOY_INTERACTION_RANGE)

        if self.target in toys:
            self.next_action = Action.PASS_TOY
            return

        self.velocity = Position.calc_norm_vector(self.pos, self.target.pos)
        new_pos = self.pos + self.velocity * self.SPEED
        self.move_agent(new_pos)

    def _step_pass_toy(self):
        throw_direction = Position.calc_norm_vector(
            self.pos, self.model.infant.pos
        ) * min(self.TOY_THROW_RANGE, math.dist(self.pos, self.model.infant.pos))

        new_pos = self.pos + throw_direction
        self.target.move_agent(new_pos)
        self.rotate_towards(new_pos)

        self.model.infant.bonus_target = self.target
        if self.target == self.bonus_target:
            self.satisfaction[-1] += 1
        self.target = None
        self.bonus_target = None
        self.next_action = Action.WAIT

    def _update_infant_visible(self):
        infant_angle = Position.angle(self.pos, self.model.infant.pos)
        self.infant_visible = abs(infant_angle - self.direction) < self.sight_angle

    def _handle_event_toy_thrown(self, event):
        pass

    def _handle_event_toy_selected(self, event):
        pass

    def _handle_event_throw_evaluation(self, event):
        pass
