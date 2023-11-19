"""
Wolf-Sheep Predation Model
================================

Replication of the model found in NetLogo:
    Wilensky, U. (1997). NetLogo Wolf Sheep Predation model.
    http://ccl.northwestern.edu/netlogo/models/WolfSheepPredation.
    Center for Connected Learning and Computer-Based Modeling,
    Northwestern University, Evanston, IL.
"""

import mesa
import random

from toddler_abm.scheduler import RandomActivationByTypeFiltered
from toddler_abm.agents import Toddler, LegoBrick


class ToddlerABM(mesa.Model):
    """
    Wolf-Sheep Predation Model
    """

    verbose = False  # Print-monitoring

    description = (
        "A model for simulating wolf and sheep (predator-prey) ecosystem modelling."
    )

    def __init__(
        self,
        width=15,
        height=15,
        initial_bricks=4,
        drops_brick=10,
    ):
        """
        Create a new Wolf-Sheep model with the given parameters.

        Args:
            initial_sheep: Number of sheep to start with
            initial_wolves: Number of wolves to start with
            sheep_reproduce: Probability of each sheep reproducing each step
            wolf_reproduce: Probability of each wolf reproducing each step
            wolf_gain_from_food: Energy a wolf gains from eating a sheep
            grass: Whether to have the sheep eat grass for energy
            grass_regrowth_time: How long it takes for a grass patch to regrow
                                 once it is eaten
            sheep_gain_from_food: Energy sheep gain from grass, if enabled.
        """
        super().__init__()
        # Set parameters
        self.width = width
        self.height = height
        self.initial_bricks = initial_bricks
        self.drops_brick = drops_brick / 100

        self.schedule = RandomActivationByTypeFiltered(self)
        self.grid = mesa.space.MultiGrid(self.width, self.height, torus=True)
        self.datacollector = mesa.DataCollector(
            {
                "Toddler": lambda m: m.schedule.get_type_count(Toddler),
                "LegoBricks": lambda m: m.schedule.get_type_count(LegoBrick),
            }
        )

        # Create sheep:
        for i in range(self.initial_bricks):
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)
            color = "#" + "".join([random.choice("0123456789ABCDEF") for j in range(6)])

            brick = LegoBrick(self.next_id(), (x, y), self, color)
            self.grid.place_agent(brick, (x, y))
            self.schedule.add(brick)

        x = self.random.randrange(self.width)
        y = self.random.randrange(self.height)

        toddler = Toddler(self.next_id(), (x, y), self, True)
        self.grid.place_agent(toddler, (x, y))
        self.schedule.add(toddler)

        self.running = True
        self.datacollector.collect(self)

    def step(self):
        self.schedule.step()
        # collect data
        self.datacollector.collect(self)

    def run_model(self, step_count=200):
        # if self.verbose:
        #     print("Initial number wolves: ", self.schedule.get_type_count(Wolf))
        #     print("Initial number sheep: ", self.schedule.get_type_count(Sheep))
        #     print(
        #         "Initial number grass: ",
        #         self.schedule.get_type_count(GrassPatch, lambda x: x.fully_grown),
        #     )

        for i in range(step_count):
            self.step()

        # if self.verbose:
        #     print("")
        #     print("Final number wolves: ", self.schedule.get_type_count(Wolf))
        #     print("Final number sheep: ", self.schedule.get_type_count(Sheep))
        #     print(
        #         "Final number grass: ",
        #         self.schedule.get_type_count(GrassPatch, lambda x: x.fully_grown),
        #     )
