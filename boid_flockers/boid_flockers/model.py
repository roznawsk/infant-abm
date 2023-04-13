"""
Flockers
=============================================================
A Mesa implementation of Craig Reynolds's Boids flocker model.
Uses numpy arrays to represent vectors.
"""

import mesa
import numpy as np
import random

from .agents import Toddler, Toy


class BoidFlockers(mesa.Model):
    """
    Flocker model class. Handles agent creation, placement and scheduling.
    """

    def __init__(
        self,
        population=10,
        width=100,
        height=100,
        speed=1,
        vision=10,
        persistence=90,
    ):
        """
        Create a new Flockers model.

        Args:
            population: Number of Boids
            width, height: Size of the space.
            speed: How fast should the Boids move.
            vision: How far around should each Boid look for its neighbors
            separation: What's the minimum distance each Boid will attempt to
                    keep from any other
            cohere, separate, match: factors for the relative importance of
                    the three drives."""
        self.population = population
        self.vision = vision
        self.speed = speed
        self.persistence = persistence

        self.schedule = mesa.time.RandomActivation(self)
        self.space = mesa.space.ContinuousSpace(width, height, False)
        self.make_agents()
        self.running = True

    def make_agents(self):
        """
        Create self.population agents, with random positions and starting headings.
        """
        for i in range(self.population):
            x = self.random.random() * self.space.x_max
            y = self.random.random() * self.space.y_max
            pos = np.array((x, y))

            color = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])

            brick = Toy(
                i,
                self,
                pos,
                color
            )
            self.space.place_agent(brick, pos)
            self.schedule.add(brick)

        x = 0.5 * self.space.x_max
        y = 0.5 * self.space.y_max
        pos = np.array((x, y))
        velocity = self.rand_velocity()
        toddler = Toddler(
            self.population,
            self,
            pos,
            self.speed,
            velocity,
            self.vision
        )
        self.space.place_agent(toddler, pos)
        self.schedule.add(toddler)

    def step(self):
        self.schedule.step()

    def rand_velocity(self):
        velocity = np.random.random(2) * 2 - 1
        return velocity / np.linalg.norm(velocity)
