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
        perception,
        precision,
        coordination
    ):
        """
        Create a new Toddler model.

        Args:
            """

        self.lego_count = lego_count
        self.speed = speed
        self.perception = perception / 100 * width
        self.precision = precision / 100
        self.coordination = coordination / 100

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

        x = 0.5 * self.space.x_max
        y = 0.5 * self.space.y_max
        pos = np.array((x, y))
        toddler = Toddler(
            model=self,
            unique_id=self.lego_count,
            pos=pos,
            speed=self.speed)
        self.space.place_agent(toddler, pos)
        self.schedule.add(toddler)

    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)


def get_steps(m):
    ksteps = m.schedule.steps / (sum([a.times_interacted_with for a in m.schedule.agents if type(
        a) == Toy]) + 1)

    return ksteps
