"""
Flockers
=============================================================
A Mesa implementation of Craig Reynolds's Boids flocker model.
Uses numpy arrays to represent vectors.
"""

import mesa
import numpy as np
import random


from agents.toddler import Toddler
from agents.parent import Parent
from agents.toy import Toy

from utils import *


class ToddlerModel(mesa.Model):
    """
    Flocker model class. Handles agent creation, placement and scheduling.
    """

    def __init__(
        self,
        width,
        height,
        speed,
        lego_count,
        exploration,
        precision,
        coordination,
        responsiveness,
        relevance,
        success_dist
    ):
        """
        Create a new Toddler model.

        Args:
            """

        self.lego_count = lego_count
        self.speed = speed
        self.parent_speed = 2 * speed
        self.exploration = exploration / 100
        self.precision = precision / 100
        self.coordination = coordination / 100
        self.responsiveness = responsiveness / 100
        self.relevance = relevance / 100
        self.success_dist = success_dist

        self.schedule = mesa.time.RandomActivation(self)
        self.space = mesa.space.ContinuousSpace(width, height, False)

        # self.datacollector = mesa.DataCollector(
        #     {
        #         "Toddler satisfaction": get_toddler_satisfaction,
        #         "Parent satisfaction": get_parent_satisfaction
        #     }
        # )

        self.make_agents()
        self.running = True

    def make_agents(self):
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
        toddler = Toddler(
            model=self,
            unique_id=self.lego_count,
            pos=pos,
            speed=self.speed
        )
        self.toddler = toddler
        self.space.place_agent(toddler, pos)
        self.schedule.add(toddler)

    def step(self):
        self.schedule.step()

        # self.datacollector.collect(self)

    def success(self):
        total_dist = self.goal_dist()
        return total_dist <= self.success_dist

    def goal_dist(self):
        middle_point = (self.parent.pos + self.toddler.pos) / 2

        total_dist = 0
        toys = get_toys(self)
        for toy in toys:
            total_dist += calc_dist(middle_point, toy.pos)

        return total_dist / len(toys)

    def get_toddler_satisfaction(self):
        return self.toddler.satisfaction / self.schedule.steps

    def get_parent_satisfaction(self):
        return self.parent.satisfaction / self.schedule.steps
