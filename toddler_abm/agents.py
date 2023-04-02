import mesa
from toddler_abm.random_walk import RandomWalker


class Toddler(RandomWalker):
    """
    Toddler
    """

    energy = None

    def __init__(self, unique_id, pos, model, moore):
        super().__init__(unique_id, pos, model, moore=moore)
        self.brick = None
        self.

    def step(self):
        """
        A model step. Move, then eat grass and reproduce.
        """
        self.random_move()

        # If there is grass available, eat it
        this_cell = self.model.grid.get_cell_list_contents([self.pos])
        lego_brick = [obj for obj in this_cell if isinstance(obj, LegoBrick)]

        if self.brick is None and lego_brick:
            lego_brick = lego_brick[0]

            print(f'removing agent {lego_brick}')
            self.model.grid.remove_agent(lego_brick)
            self.brick = lego_brick

            # self.energy += self.model.sheep_gain_from_food
            # grass_patch.fully_grown = False

        elif self.brick and self.random.random() < self.model.drops_brick:
            # Create a new sheep:
            # if self.model.grass:
            #     self.energy /= 2

            # lamb = Sheep(
            #     self.model.next_id(), self.pos, self.model, self.moore, self.energy
            # )
            self.brick.pos = self.pos
            self.model.grid.place_agent(self.brick, self.pos)
            self.brick = None


class LegoBrick(mesa.Agent):
    """
    A patch of grass that grows at a fixed rate and it is eaten by sheep
    """

    def __init__(self, unique_id, pos, model, color):
        """
        Creates a new patch of grass

        Args:
            grown: (boolean) Whether the patch of grass is fully grown or not
            countdown: Time for the patch of grass to be fully grown again
        """
        super().__init__(unique_id, model)
        self.pos = pos
        self.color = color

    def __repr__(self) -> str:
        return f'Brick Object at {self.pos} with color {self.color}'
