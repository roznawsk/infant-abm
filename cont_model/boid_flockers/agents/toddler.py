import random
import uuid
import mesa
import numpy as np

from .toy import Toy


class Toddler(mesa.Agent):
    """
    """

    def __init__(
        self,
        unique_id,
        model,
        pos,
        speed,
        velocity=None
    ):
        """
        Create a new Boid flocker agent.

        Args:
        """
        super().__init__(unique_id, model)
        self.pos = np.array(pos)
        self.speed = speed

        if velocity is None:
            velocity = self._rand_velocity()

        self.velocity = velocity

        self.inertia = 5
        self.toy_interaction_range = 10
        self.toy_throw_range = 20
        self.target = None

    def step(self):
        """
        Get the Boid's neighbors, compute the new vector, and move accordingly.
        """

        if self.target and dist(self.pos, self.target.pos) < self.toy_interaction_range:
            self._interact_with_toy()

        if self.target is None:
            self._maybe_find_new_target()

        self._set_new_velocity()
        new_pos = self.pos + self.velocity * self.speed
        self._correct_out_of_bounds(new_pos)

        self.model.space.move_agent(self, new_pos)

    def _set_new_velocity(self):
        new_velocity = None
        inertia = self.inertia

        if self.target:
            new_velocity = calc_velocity(self.pos, self.target.pos)
            new_velocity = self._random_rotate(new_velocity)

            toy_dist = dist(self.pos, self.target.pos)

            # print(toy_dist, inertia, self.toy_interaction_range, self.speed)

            # inertia = inertia * \
            #     min(1, max(0, toy_dist / self.speed - self.toy_interaction_range))
        else:
            new_velocity = self._random_rotate(self.velocity)

        self.velocity = (self.velocity * inertia +
                         new_velocity) / (inertia + 1)

    def _random_rotate(self, vector):
        angle = np.random.uniform(-np.pi, np.pi) * (1 - self.model.precision)

        sin = np.sin(angle)
        cos = np.cos(angle)

        rotation_matrix = np.array([[cos, -sin], [sin, cos]])
        return rotation_matrix @ vector

    def _interact_with_toy(self):
        self.target.pos[0] -= self.toy_throw_range * \
            np.random.uniform(-1, 1)

        self.target.pos[0] = max(0, self.target.pos[0])
        self.target.pos[0] = min(self.model.space.width, self.target.pos[0])

        self.target.pos[1] -= self.toy_throw_range * \
            np.random.uniform((-1 + self.model.coordination), 1)

        self.target.pos[1] = max(0, self.target.pos[1])
        self.target.pos[1] = min(self.model.space.height, self.target.pos[1])

        self.target.deactivate()
        self.target = None

    def _maybe_find_new_target(self):
        toys_in_neighbourhood = self.model.space.get_neighbors(
            self.pos, self.model.perception, False)

        x = np.random.random() * self.model.space.x_max
        y = np.random.random() * self.model.space.y_max
        pos = np.array((x, y))

        toys_in_neighbourhood.append(
            Toy(model=self.model, unique_id=uuid.uuid4(), pos=pos)
        )

        toys_in_neighbourhood[-1].times_interacted_with = max(
            [agent.times_interacted_with for agent in self.model.schedule.agents if type(agent) == Toy])

        probabilities = np.array(
            [1 / (toy.times_interacted_with + 1) for toy in toys_in_neighbourhood])

        probabilities = probabilities / probabilities.sum()

        if toys_in_neighbourhood and np.random.random() > self.model.precision:
            [target] = np.random.choice(
                toys_in_neighbourhood, size=1, p=probabilities)
            self.velocity = calc_velocity(self.pos, target.pos)
            self.target = target
            self.target.activate()

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

    def _rand_velocity(self):
        velocity = np.random.random(2) * 2 - 1
        return velocity / np.linalg.norm(velocity)


def dist(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5


def calc_velocity(p1, p2):
    velocity = [p2[0] - p1[0], p2[1] - p1[1]]

    return velocity / np.linalg.norm(velocity)

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
