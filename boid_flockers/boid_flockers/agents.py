import mesa
import numpy as np


class Toddler(mesa.Agent):
    """
    """

    def __init__(
        self,
        unique_id,
        model,
        pos,
        speed,
        velocity,
        vision,
    ):
        """
        Create a new Boid flocker agent.

        Args:
            unique_id: Unique agent identifyer.
            pos: Starting position
            speed: Distance to move per step.
            heading: numpy vector for the Boid's direction of movement.
            vision: Radius to look around for nearby Boids.
            separation: Minimum distance to maintain from other Boids.
            cohere: the relative importance of matching neighbors' positions
            separate: the relative importance of avoiding close neighbors
            match: the relative importance of matching neighbors' headings
        """
        super().__init__(unique_id, model)
        self.pos = np.array(pos)
        self.speed = speed
        self.velocity = velocity
        self.target = None

    def step(self):
        """
        Get the Boid's neighbors, compute the new vector, and move accordingly.
        """

        toys_in_neighbourhood = self.model.space.get_neighbors(self.pos, self.model.vision, False)

        if np.random.random() > self.model.persistence / 100:
            if toys_in_neighbourhood and np.random.random() > 0.5:
                [target] = np.random.choice(toys_in_neighbourhood, 1)
                self.velocity = calc_velocity(self.pos, target.pos)
                self.target = target
            else:
                self.velocity = np.random.random(2) * 2 - 1
                self.velocity /= np.linalg.norm(self.velocity)

        new_pos = self.pos + self.velocity * self.speed
        self._correct_out_of_bounds(new_pos)

        self.model.space.move_agent(self, new_pos)

        if self.target and dist(self.pos, self.target.pos) < 5:
            print('removing target', self.target, self.target.pos)
            self.model.space.remove_agent(self.target)
            self.model.schedule.remove(self.target)
            self.target = None

    def _correct_out_of_bounds(self, new_pos):
        if new_pos[0] < 0:
            new_pos[0] = -new_pos[0]
            self.velocity[0] = -self.velocity[0]
        elif new_pos[0] > self.model.space.width:
            new_pos[0] = 2 * self.model.space.width - new_pos[0]
            self.velocity[0] = -self.velocity[0]

        if new_pos[1] < 0:
            new_pos[1] = -new_pos[1]
            self.velocity[1] = -self.velocity[1]
        elif new_pos[1] > self.model.space.height:
            new_pos[1] = 2 * self.model.space.height - new_pos[1]
            self.velocity[1] = -self.velocity[1]

    # def step(self):
    #     """
    #     A model step. Move, then eat grass and reproduce.
    #     """
    #     self.random_move()

    #     # If there is grass available, eat it
    #     this_cell = self.model.grid.get_cell_list_contents([self.pos])
    #     lego_brick = [obj for obj in this_cell if isinstance(obj, LegoBrick)]

    #     if self.brick is None and lego_brick:
    #         lego_brick = lego_brick[0]

    #         print(f'removing agent {lego_brick}')
    #         self.model.grid.remove_agent(lego_brick)
    #         self.brick = lego_brick

    #         # self.energy += self.model.sheep_gain_from_food
    #         # grass_patch.fully_grown = False

    #     elif self.brick and self.random.random() < self.model.drops_brick:
    #         # Create a new sheep:
    #         # if self.model.grass:
    #         #     self.energy /= 2

    #         # lamb = Sheep(
    #         #     self.model.next_id(), self.pos, self.model, self.moore, self.energy
    #         # )
    #         self.brick.pos = self.pos
    #         self.model.grid.place_agent(self.brick, self.pos)
    #         self.brick = None


class Toy(mesa.Agent):
    """
    A Boid-style flocker agent.

    The agent follows three behaviors to flock:
        - Cohesion: steering towards neighboring agents.
        - Separation: avoiding getting too close to any other agent.
        - Alignment: try to fly in the same direction as the neighbors.

    Boids have a vision that defines the radius in which they look for their
    neighbors to flock with. Their speed (a scalar) and velocity (a vector)
    define their movement. Separation is their desired minimum distance from
    any other Boid.
    """

    def __init__(
        self,
        unique_id,
        model,
        pos,
        color
    ):
        """
        Create a new Boid flocker agent.

        Args:
            unique_id: Unique agent identifyer.
            pos: Starting position
            speed: Distance to move per step.
            heading: numpy vector for the Boid's direction of movement.
            vision: Radius to look around for nearby Boids.
            separation: Minimum distance to maintain from other Boids.
            cohere: the relative importance of matching neighbors' positions
            separate: the relative importance of avoiding close neighbors
            match: the relative importance of matching neighbors' headings
        """
        super().__init__(unique_id, model)
        self.pos = np.array(pos)
        self.color = color

    def step(self):
        """
        Get the Boid's neighbors, compute the new vector, and move accordingly.
        """

        # neighbors = self.model.space.get_neighbors(self.pos, self.vision, False)
        # self.velocity += (
        #     self.cohere(neighbors) * self.cohere_factor
        #     + self.separate(neighbors) * self.separate_factor
        #     + self.match_heading(neighbors) * self.match_factor
        # ) / 2
        # self.velocity /= np.linalg.norm(self.velocity)
        # new_pos = self.pos + self.velocity * self.speed
        # self.model.space.move_agent(self, new_pos)
        pass


def dist(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5


def calc_velocity(p1, p2):
    velocity = [p2[0] - p1[0], p2[1] - p1[1]]

    return velocity / np.linalg.norm(velocity)
