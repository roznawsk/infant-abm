"""
Infant Model
=============================================================
A Mesa implementation of Infant ABM Model
"""

from infant_abm.agents.infant import Infant
from infant_abm.agents.infant import Params as InfantParams

from infant_abm.agents.parent import Parent
from infant_abm.agents.toy import Toy

import numpy as np
import mesa

from infant_abm.agents.position import Position


class InfantModel(mesa.Model):
    """
    Flocker model class. Handles agent creation, placement and scheduling.
    """

    def __init__(
        self,
        width,
        height,
        toy_count,
        responsiveness,
        visualization_average_steps=300,
        exploration=None,
        precision=None,
        coordination=None,
        infant_params=None,
    ):
        """
        Create a new Infant model.

        Args:
        """

        mesa.Model.__init__(self)

        self.toy_count = toy_count
        self.toys = []

        self.responsiveness = responsiveness / 100

        if infant_params is None:
            infant_params = InfantParams(
                # precision=precision / 100,
                # coordination=coordination / 100,
                exploration=exploration / 100,
            )

        self.visualization_average_steps = visualization_average_steps

        self.schedule = mesa.time.RandomActivation(self)
        self.space = mesa.space.ContinuousSpace(width, height, False)
        Position.x_max = width
        Position.y_max = height

        self.datacollector = mesa.DataCollector(
            {
                "Infant TPS": self.get_infant_satisfaction,
                "Parent TPS": self.get_parent_satisfaction,
                # "dist_middle": self.get_middle_dist,
            }
        )

        self.make_agents(infant_params)
        self.running = True

    def make_agents(self, infant_params):
        """
        Create self.population agents, with random positions and starting headings.
        """

        for i in range(self.toy_count):
            toy_pos = Position.random()
            print(f"toy pos, {toy_pos}")
            toy = Toy(i, self, toy_pos)
            self.space.place_agent(toy, toy.pos)
            self.toys.append(toy)
            self.schedule.add(toy)

        parent = Parent(
            model=self,
            unique_id=self.toy_count + 1,
            pos=Position.random(),
        )
        self.parent = parent

        self.space.place_agent(parent, parent.pos)
        self.schedule.add(parent)

        x = 0.5 * Position.x_max
        y = 0.5 * Position.y_max

        infant = Infant(
            model=self,
            unique_id=self.toy_count,
            pos=np.array([x, y]),
            params=infant_params,
        )
        self.infant = infant
        self.space.place_agent(infant, infant.pos)
        self.schedule.add(infant)

        print("all agents")
        agents = self.schedule.agents
        agents = self.space.get_neighbors((0, 0), 1000)
        for a in agents:
            print(a.pos)
            print(type(a.pos))

    def step(self):
        if self.get_middle_dist() < 10:
            print("target achieved")
            return

        self.schedule.step()

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
            total_dist += Position.dist(middle_point, toy.pos)

        return total_dist / len(toys)

    def get_toys(self, pos=None, range=None):
        toys = []

        if pos is None or range is None:
            toys = self.schedule.agents
        else:
            toys = self.space.get_neighbors(pos, range, False)

        return [a for a in toys if type(a) == Toy]
