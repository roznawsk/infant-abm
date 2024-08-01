"""
Infant Model
=============================================================
A Mesa implementation of Infant ABM Model
"""

import math
import mesa
import numpy as np
import warnings

from infant_abm.agents import Toy, Position

from infant_abm.agents.infant import Params as InfantParams
from infant_abm.agents.infant import InfantBase, NoVisionInfant, SeqVisionInfant

from infant_abm.agents.parent_base import ParentBase
from infant_abm.agents.parent import MoverParent, VisionOnlyParent

from infant_abm.config import Config


class InfantModel(mesa.Model):
    """
    Flocker model class. Handles agent creation, placement and scheduling.
    """

    WIDTH = 100
    HEIGHT = 100

    def __init__(
        self,
        visualization_average_steps=300,
        infant_class="SeqVisionInfant",
        parent_class="VisionOnlyParent",
        config=Config(),
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
            infant_params = InfantParams.from_array(
                [perception, persistence, coordination]
            )
        self.next_agent_id = 0

        self.visualization_average_steps = visualization_average_steps

        self.schedule = mesa.time.RandomActivation(self)
        self.space = mesa.space.ContinuousSpace(self.WIDTH, self.HEIGHT, False)
        Position.x_max = self.WIDTH
        Position.y_max = self.HEIGHT

        self.parent: ParentBase = None
        match parent_class:
            case "MoverParent":
                self.parent_class = MoverParent
            case "VisionOnlyParent":
                self.parent_class = VisionOnlyParent

        self.infant: InfantBase = None
        match infant_class:
            case "NoVisionInfant":
                self.infant_class = NoVisionInfant
            case "SeqVisionInfant":
                self.infant_class = SeqVisionInfant

        self.config = config
        self._apply_config(config)

        self.toys = []

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self.make_agents(infant_params)

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "parent-visible": lambda m: int(getattr(m.infant, "parent_visible", 0)),
                "infant-visible": lambda m: int(m.parent.infant_visible) / 2,
                "heading": lambda m: m.infant.params.persistence.e2,
                "throwing": lambda m: m.infant.params.coordination.e2,
                "goal_dist": lambda m: m.get_middle_dist(),
            },
        )

        self.datacollector.collect(self)

    def make_agents(self, infant_params):
        """
        Create self.population agents, with random positions and starting headings.
        """

        self._make_toys()

        parent_x = np.random.uniform(0.25, 0.75) * Position.x_max
        parent_y = np.random.uniform(0.25, 0.75) * Position.y_max

        parent = self.parent_class(
            model=self,
            unique_id=self._next_agent_id(),
            pos=np.array([parent_x, parent_y]),
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

        self.datacollector.collect(self)

    def get_infant_satisfaction(self):
        return np.average(self.infant.satisfaction[-self.visualization_average_steps :])

    def get_parent_satisfaction(self):
        return np.average(self.parent.satisfaction[-self.visualization_average_steps :])

    def get_middle_dist(self) -> float:
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

    def _apply_config(self, config):
        if config.coordination_boost_value:
            self.infant_class.COORDINATION_BOOST_VALUE = config.coordination_boost_value
        if config.persistence_boost_value:
            self.infant_class.PERSISTENCE_BOOST_VALUE = config.persistence_boost_value
