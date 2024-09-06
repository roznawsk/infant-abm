from abc import ABC, abstractmethod

import numpy as np


class DataCollector(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def after_step(self):
        pass

    @abstractmethod
    def to_dict(self):
        pass


SUCCESS_DIST = 20000


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
