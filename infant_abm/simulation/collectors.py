from abc import ABC, abstractmethod

import numpy as np

from infant_abm.model import InfantModel


class DataCollector(ABC):
    def __init__(self, model: InfantModel):
        self.model = model

    @abstractmethod
    def after_step(self):
        pass

    @abstractmethod
    def to_dict(self):
        pass


SUCCESS_DIST = 10


class v1Collector(DataCollector):
    def __init__(self, model):
        super().__init__(model)
        self.goal_dist_iteration = None

    def after_step(self):
        if self.model.get_middle_dist() < SUCCESS_DIST:
            self.goal_dist_iteration = self.model._steps
            return False
        return True

    def to_dict(self):
        return {
            "goal_dist": self.goal_dist_iteration,
        }


class v1CollectoTrails(DataCollector):
    def __init__(self, model):
        super().__init__(model)

        self.goal_dists = []
        self.infant_positions = []
        self.infant_actions = []

        self.goal_dist_iteration = None

    def after_step(self):
        if not self.goal_dist_iteration and self.model.get_middle_dist() < SUCCESS_DIST:
            self.goal_dist_iteration = self.model._steps

        self.goal_dists.append(self.model.get_middle_dist())
        self.infant_positions.append(self.model.infant.pos)
        self.infant_actions.append(type(self.model.infant.next_action).__name__)

        return True

    def to_dict(self):
        return {
            "goal_dist": self.goal_dist_iteration,
            "goal_dists": np.array(self.goal_dists),
            "infant_positions": np.array(self.infant_positions),
            "infant_actions": np.array(self.infant_actions),
        }


class v2Collector(DataCollector):
    def __init__(self, model):
        super().__init__(model)

        self.rewards = []

    def after_step(self):
        self.rewards.append(self.model.infant.last_reward)
        return True

    def to_dict(self):
        return {
            "rewards": np.array(self.rewards),
            "q_table": self.model.infant.q_learning_agent.q_table,
        }
