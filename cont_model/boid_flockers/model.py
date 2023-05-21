"""
Flockers
=============================================================
A Mesa implementation of Craig Reynolds's Boids flocker model.
Uses numpy arrays to represent vectors.
"""

import mesa
import numpy as np
import random


from .agents.toddler import Toddler
from .agents.parent import Parent
from .agents.toy import Toy


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
        relevance
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

        self.schedule = mesa.time.RandomActivation(self)
        self.space = mesa.space.ContinuousSpace(width, height, False)

        self.datacollector = mesa.DataCollector(
            {
                "Steps/Interaction": get_steps
            }
        )

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
        self.datacollector.collect(self)


def get_steps(m):
    ksteps = m.schedule.steps / (sum([a.times_interacted_with for a in m.schedule.agents if type(
        a) == Toy]) + 1)

    return ksteps
