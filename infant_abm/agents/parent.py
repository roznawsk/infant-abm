from enum import Enum

from abc import ABC, abstractmethod

from infant_abm.agents.agent import Agent
from infant_abm.agents.position import Position
from infant_abm.agents.events import ToySelected, ToyThrown, ThrowEvaluation


class Action(Enum):
    WAIT = 1
    FETCH_TOY = 2
    PASS_TOY = 3


class Parent(Agent, ABC):
    responsiveness = 0.5
    relevant_response_probability = 0.5

    SPEED = 5
    TOY_INTERACTION_RANGE = 10
    TOY_THROW_RANGE = 20

    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model, pos)

    @abstractmethod
    def step(self):
        pass

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

    def _perform_action(self, action):
        match action:
            case Action.WAIT:
                return Action.WAIT
            case Action.FETCH_TOY:
                return self._step_fetch_toy()
            case Action.PASS_TOY:
                return self._step_pass_toy()

    def _update_infant_visible(self):
        infant_angle = Position.angle(self.pos, self.model.infant.pos)
        self.infant_visible = abs(infant_angle - self.direction) < self.sight_angle

    def _handle_event_toy_thrown(self, event):
        pass

    def _handle_event_toy_selected(self, event):
        pass

    def _handle_event_throw_evaluation(self, event):
        pass
