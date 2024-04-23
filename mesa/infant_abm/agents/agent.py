import mesa
from infant_abm.agents.position import Position


class Agent(mesa.Agent):
    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model)

        self.pos = pos

    def move_agent(self, new_pos):
        new_pos = Position.correct_out_of_bounds(new_pos)
        self.model.space.move_agent(self, new_pos)
