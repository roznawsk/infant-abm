"""
Infant Model
=============================================================
A Mesa implementation of Infant ABM Model
"""

import math
from typing import List
import mesa
import numpy as np
import warnings

from infant_abm.agents.infant import Params as InfantParams
from infant_abm.agents.infant import Infant
from infant_abm.agents.toy import Toy
from infant_abm.agents.position import Position
from infant_abm.agents.parent import Parent
from infant_abm.config import Config


class InfantModel(mesa.Model):
    """
    Flocker model class. Handles agent creation, placement and scheduling.
    """

    WIDTH = 100
    HEIGHT = 100

    def __init__(
        self,
        infant_class,
        parent_class,
        config=Config(),
        infant_params=None,
        perception=None,
        persistence=None,
        coordination=None,
        infant_kwargs=dict(),
    ):
        mesa.Model.__init__(self)

        if infant_params is None:
            infant_params = InfantParams.from_array(
                [perception, persistence, coordination]
            )
        self.next_agent_id = 0

        self.space = mesa.space.ContinuousSpace(self.WIDTH, self.HEIGHT, False)
        self.schedule = mesa.time.SimultaneousActivation(model=self)

        Position.x_max = self.WIDTH
        Position.y_max = self.HEIGHT

        assert issubclass(infant_class, Infant)
        self.infant_class = infant_class

        assert issubclass(parent_class, Parent)
        self.parent_class = parent_class

        self.parent: Parent = None
        self.infant: Infant = None
        self.toys: List[Toy] = None

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self.make_agents(infant_params, infant_kwargs)

        self.config = config
        self._apply_config(config)

    def make_agents(self, infant_params, infant_kwargs):
        self.toys = self._create_toys()

        parent_x = np.random.uniform(0.25, 0.75) * Position.x_max
        parent_y = np.random.uniform(0.25, 0.75) * Position.y_max

        self.parent = self.parent_class(
            model=self,
            unique_id=self._next_agent_id(),
            pos=np.array([parent_x, parent_y]),
        )

        x, y = (0.5 * Position.x_max, 0.5 * Position.y_max)

        self.infant = self.infant_class(
            model=self,
            unique_id=self._next_agent_id(),
            pos=np.array([x, y]),
            params=infant_params,
            **infant_kwargs,
        )

        for agent in self.toys + [self.infant] + [self.parent]:
            self.schedule.add(agent)
            self.space.place_agent(agent, agent.pos)

    def step(self):
        self.schedule.step()

    def get_middle_dist(self) -> float:
        middle_point = (self.parent.pos + self.infant.pos) / 2

        total_dist = 0
        toys = self.get_toys()
        for toy in toys:
            total_dist += math.dist(middle_point, toy.pos)

        return np.round(total_dist / len(toys), 5)

    def get_toys(self, pos=None, range=None):
        if pos is None or range is None:
            return self.toys
        else:
            toys = self.space.get_neighbors(pos, range, False)
            return [t for t in toys if isinstance(t, Toy)]

    def _create_toys(self):
        toys = []

        for x in [1 / 4, 3 / 4]:
            for y in [1 / 4, 3 / 4]:
                toy_pos = np.array([x * self.space.x_max, y * self.space.y_max])
                toys.append(Toy(self._next_agent_id(), self, toy_pos))

        return toys

    def _next_agent_id(self):
        agent_id = self.next_agent_id
        self.next_agent_id += 1
        return agent_id

    def _apply_config(self, config):
        if config.coordination_boost_value:
            self.infant_class.COORDINATION_BOOST_VALUE = config.coordination_boost_value
        if config.persistence_boost_value:
            self.infant_class.PERSISTENCE_BOOST_VALUE = config.persistence_boost_value
