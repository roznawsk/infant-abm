import mesa
import numpy as np

from infant_abm.agents.position import Position


class Agent(mesa.Agent):
    sight_angle = 25 / 180 * np.pi

    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model)

        self.pos = pos

        # Direction of the agent ranging from 0 to +2π
        self.direction = np.random.uniform(0, 2 * np.pi)

    def move_agent(self, new_pos):
        new_pos = Position.correct_out_of_bounds(new_pos)
        self.rotate_towards(new_pos)

        self.model.space.move_agent(self, new_pos)

    def rotate_towards(self, pos):
        self.direction = Position.angle(self.pos, pos)
