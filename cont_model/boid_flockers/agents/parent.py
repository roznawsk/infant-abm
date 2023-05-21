import mesa
import numpy as np
from enum import Enum

from .toy import Toy


class Action(Enum):
    CRAWL = 1
    LOOK_FOR_TOY = 2
    INTERACT_WITH_TOY = 3


class Parent(mesa.Agent):
    """
    """

    def __init__(
        self,
        unique_id,
        model,
        pos,
        speed
    ):
        """
        Create a new Boid flocker agent.

        Args:
        """
        super().__init__(unique_id, model)
        self.pos = np.array(pos)
        self.speed = speed

        self.velocity = None

        self.toy_interaction_range = 20
        self.toy_throw_range = 40
        self.target = None

        self.next_action = Action.LOOK_FOR_TOY

        self.steps_until_distraction = None

    def step(self):
        """
        Get the Boid's neighbors, compute the new vector, and move accordingly.
        """

        pass

        # if self.next_action == Action.CRAWL:
        #     self._step_crawl()
        # elif self.next_action == Action.LOOK_FOR_TOY:
        #     self._step_change_target()
        # elif self.next_action == Action.INTERACT_WITH_TOY:
        #     self._step_toy_interaction()

        # if self.target is None:
        #     self._maybe_find_new_target()

    def _step_crawl(self):

        toys = self.model.space.get_neighbors(
            self.pos, self.toy_interaction_range, False)
        print(f'distra: {self.steps_until_distraction}, Toys in range: {toys}')
        if self.steps_until_distraction == 0:
            self.target = None

            self.next_action = Action.LOOK_FOR_TOY
            return

        self.velocity = calc_velocity(self.pos, self.target.pos)
        new_pos = self.pos + self.velocity * self.speed
        self._correct_out_of_bounds(new_pos)

        if self.steps_until_distraction:
            self.steps_until_distraction -= 1
        self.model.space.move_agent(self, new_pos)

        toys = self.model.space.get_neighbors(
            self.pos, self.toy_interaction_range, False)

        if toys:
            self.target = toys[0]
            self.next_action = Action.INTERACT_WITH_TOY

    def _step_toy_interaction(self):
        throw_direction = None

        if self.model.coordination > np.random.rand():
            throw_direction = np.array([0, self.toy_throw_range])
        else:
            throw_direction = np.array([np.random.randint(1) * 2 - 1, 0]) * self.toy_throw_range
            # throw_direction = np.random.rand(2)
            # throw_direction = throw_direction / np.linalg.norm(throw_direction) * self.toy_throw_range

        # self.target.pos[0] -= self.toy_throw_range * \
        #     np.random.uniform(-1, 1)

        new_pos = self.target.pos + throw_direction

        for idx, value in enumerate([self.model.space.width, self.model.space.height]):
            new_pos[idx] = max(0, new_pos[idx])
            new_pos[idx] = min(value - .01, new_pos[idx])

        print(f'throw = {throw_direction}, {type(self.target.pos)}, {self.target.pos}, {new_pos}')

        self.model.space.move_agent(self.target, new_pos)

        self.target.times_interacted_with += 1
        self.target = None

        self.next_action = Action.LOOK_FOR_TOY

    def _step_change_target(self):
        toys = [a for a in self.model.schedule.agents if type(a) == Toy]

        probabilities = np.array(
            [1 / (toy.times_interacted_with * self.model.exploration + 1) for toy in toys])

        probabilities = probabilities / probabilities.sum()

        [target] = np.random.choice(toys, size=1, p=probabilities)
        self.velocity = calc_velocity(self.pos, target.pos)
        self.target = target

        print(f'prec = {self.model.precision}')

        if self.model.precision > np.random.rand():
            self.steps_until_distraction = None
        else:
            dist = calc_dist(self.pos, self.target.pos)
            steps_to_target = max(1, np.floor(dist - self.toy_interaction_range) / self.speed)
            self.steps_until_distraction = np.random.randint(steps_to_target)

        self.next_action = Action.CRAWL

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


def calc_dist(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5


def calc_velocity(p1, p2):
    velocity = [p2[0] - p1[0], p2[1] - p1[1]]

    return velocity / np.linalg.norm(velocity)
