"""
Infant Model
=============================================================
A Mesa implementation of Infant ABM Model
"""

from infant_abm.agents.infant import Infant
from infant_abm.agents.infant import Params as InfantParams

from infant_abm.agents.parent import Parent
from infant_abm.agents.toy import Toy

from infant_abm.utils import get_toys

import math
import numpy as np
import mesa


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
            x = self.random.random() * self.space.x_max
            y = self.random.random() * self.space.y_max
            pos = np.array((x, y))

            toy = Toy(i, self, pos)
            self.space.place_agent(toy, pos)
            self.toys.append(toy)
            self.schedule.add(toy)

        parent = Parent(
            model=self,
            unique_id=self.toy_count + 1,
            pos=pos,
        )
        self.parent = parent

        x = self.random.random() * self.space.x_max
        y = self.random.random() * self.space.y_max
        pos = np.array((x, y))
        self.space.place_agent(parent, pos)
        self.schedule.add(parent)

        x = 0.5 * self.space.x_max
        y = 0.5 * self.space.y_max
        pos = np.array((x, y))

        infant = Infant(
            model=self,
            unique_id=self.toy_count,
            pos=pos,
            params=infant_params,
        )
        self.infant = infant
        self.space.place_agent(infant, pos)
        self.schedule.add(infant)

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
        toys = get_toys(self)
        for toy in toys:
            total_dist += math.dist(middle_point, toy.pos)

        return total_dist / len(toys)

    def get_dims(self):
        return [self.space.width, self.space.height]
