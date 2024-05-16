"""
Infant Model
=============================================================
A Mesa implementation of Infant ABM Model
"""

import math
import mesa
import numpy as np

from infant_abm.agents.infant_base import InfantBase
from infant_abm.agents.infant.no_vision_infant import NoVisionInfant
from infant_abm.agents.infant.vision_infant import VisionInfant
from infant_abm.agents.infant.seq_vision_infant import SeqVisionInfant

from infant_abm.agents.infant_base import Params as InfantParams

from infant_abm.agents.parent import Parent
from infant_abm.agents.toy import Toy


from infant_abm.agents.position import Position


class InfantModel(mesa.Model):
    """
    Flocker model class. Handles agent creation, placement and scheduling.
    """

    WIDTH = 100
    HEIGHT = 100

    # infant_class = NoVisionInfant
    infant_class = VisionInfant
    # infant_class = SeqVisionInfant

    def __init__(
        self,
        visualization_average_steps=300,
        infant_params=None,
        perception=None,
        persistence=None,
        coordination=None,
    ):
        """
        Create a new Infant model.
        """

        mesa.Model.__init__(self)

        if infant_params is None:
            infant_params = InfantParams(perception, persistence, coordination)
        self.next_agent_id = 0

        self.visualization_average_steps = visualization_average_steps

        self.schedule = mesa.time.RandomActivation(self)
        self.space = mesa.space.ContinuousSpace(self.WIDTH, self.HEIGHT, False)
        Position.x_max = self.WIDTH
        Position.y_max = self.HEIGHT

        self.parent: Parent = None
        self.infant: InfantBase = None
        self.toys = []
        self.make_agents(infant_params)

        self.explore_exploit_ratio = getattr(self.infant, "explore_exploit_ratio", -1.0)

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "explore-exploit-ratio": "explore_exploit_ratio",
                "parent-visible": "parent_visible",
                "infant-visible": "infant_visible",
            },
        )

        self.datacollector.collect(self)

    def make_agents(self, infant_params):
        """
        Create self.population agents, with random positions and starting headings.
        """

        self._make_toys()

        parent = Parent(
            model=self,
            unique_id=self._next_agent_id(),
            pos=Position.random(),
        )
        self.parent = parent

        self.space.place_agent(parent, parent.pos)
        self.schedule.add(parent)

        x = 0.5 * Position.x_max
        y = 0.5 * Position.y_max

        infant = self.infant_class(
            model=self,
            unique_id=self._next_agent_id(),
            pos=np.array([x, y]),
            params=infant_params,
        )
        self.infant = infant
        self.space.place_agent(infant, infant.pos)
        self.schedule.add(infant)

    def step(self):
        self.schedule.step()

        self.explore_exploit_ratio = getattr(self.infant, "explore_exploit_ratio", -1.0)
        self.parent_visible = int(getattr(self.infant, "parent_visible", 0))
        self.infant_visible = int(self.parent.infant_visible) / 2

        self.datacollector.collect(self)

    def get_infant_satisfaction(self):
        return np.average(self.infant.satisfaction[-self.visualization_average_steps :])

    def get_parent_satisfaction(self):
        return np.average(self.parent.satisfaction[-self.visualization_average_steps :])

    def get_middle_dist(self):
        middle_point = (self.parent.pos + self.infant.pos) / 2

        total_dist = 0
        toys = self.get_toys()
        for toy in toys:
            total_dist += math.dist(middle_point, toy.pos)

        return total_dist / len(toys)

    def get_toys(self, pos=None, range=None):
        toys = []

        if pos is None or range is None:
            toys = self.schedule.agents
        else:
            toys = self.space.get_neighbors(pos, range, False)

        return [a for a in toys if type(a) == Toy]

    def _make_toys(self):
        for x in [1 / 4, 3 / 4]:
            for y in [1 / 4, 3 / 4]:
                toy_pos = np.array([x * self.space.x_max, y * self.space.y_max])
                toy = Toy(self._next_agent_id(), self, toy_pos)
                self.space.place_agent(toy, toy.pos)
                self.toys.append(toy)
                self.schedule.add(toy)

    def _next_agent_id(self):
        agent_id = self.next_agent_id
        self.next_agent_id += 1
        return agent_id
