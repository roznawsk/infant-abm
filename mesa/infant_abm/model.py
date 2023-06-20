"""
Flockers
=============================================================
A Mesa implementation of Craig Reynolds's Boids flocker model.
Uses numpy arrays to represent vectors.
"""

import mesa
import numpy as np
import random


from infant_abm.agents.infant import Infant
from infant_abm.agents.parent import Parent
from infant_abm.agents.toy import Toy
from infant_abm.genetic_model.infant_genome import InfantGenome

from infant_abm.utils import *


class InfantModel(mesa.Model):
    """
    Flocker model class. Handles agent creation, placement and scheduling.
    """

    def __init__(
        self,
        width,
        height,
        speed,
        lego_count,
        responsiveness,
        relevance,
        visualization_average_steps=300,
        exploration=None,
        precision=None,
        coordination=None,
        infant_genome=None,
    ):
        """
        Create a new Infant model.

        Args:
            """

        self.lego_count = lego_count
        self.speed = speed
        self.parent_speed = 2 * speed

        self.responsiveness = responsiveness / 100
        self.relevance = relevance / 100

        if infant_genome is None:
            infant_genome = InfantGenome(
                precision=precision / 100,
                coordination=coordination / 100,
                exploration=exploration / 100
            )

        print('infant_genome', infant_genome)

        self.visualization_average_steps = visualization_average_steps

        self.schedule = mesa.time.RandomActivation(self)
        self.space = mesa.space.ContinuousSpace(width, height, False)

        self.datacollector = mesa.DataCollector(
            {
                "Infant satisfaction": self.get_infant_satisfaction,
                "Parent satisfaction": self.get_parent_satisfaction
                # "dist_middle": self.get_middle_dist,
                # "dist_parent_infant": self.get_parent_infant_dist
            }
        )

        self.make_agents(infant_genome)
        self.running = True

    def make_agents(self, infant_genome):
        """
        Create self.population agents, with random positions and starting headings.
        """
        for i in range(self.lego_count):
            x = self.random.random() * self.space.x_max
            y = self.random.random() * self.space.y_max
            pos = np.array((x, y))

            brick = Toy(i, self, pos)
            self.space.place_agent(brick, pos)
            self.schedule.add(brick)

        parent = Parent(
            model=self,
            unique_id=self.lego_count + 1,
            pos=pos,
            speed=self.parent_speed
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

        print('creating_infant', infant_genome)
        infant = Infant(
            model=self,
            unique_id=self.lego_count,
            pos=pos,
            speed=self.speed,
            genome=infant_genome
        )
        self.infant = infant
        self.space.place_agent(infant, pos)
        self.schedule.add(infant)

    def step(self):
        self.schedule.step()

        self.datacollector.collect(self)

    def get_infant_satisfaction(self):
        return np.average(self.infant.satisfaction[-self.visualization_average_steps:])

    def get_parent_satisfaction(self):
        return np.average(self.parent.satisfaction[-self.visualization_average_steps:])

    def get_middle_dist(self):
        middle_point = (self.parent.pos + self.infant.pos) / 2

        total_dist = 0
        toys = get_toys(self)
        for toy in toys:
            total_dist += calc_dist(middle_point, toy.pos)

        return total_dist / len(toys)
