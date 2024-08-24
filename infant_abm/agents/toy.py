from infant_abm.agents.agent import Agent


class Toy(Agent):
    def __init__(self, unique_id, model, pos, color=None):
        super().__init__(unique_id, model, pos)

        self.model = model
        self.color = "#00FF00"
        self.times_interacted_with = 0

    def step(self):
        pass

    def interact(self):
        self.times_interacted_with += 1

        self.update_color()

    def update_color(self, max_interactions=None):
        if max_interactions is None:
            max_interactions = max([t.times_interacted_with for t in self.model.toys])
            for toy in self.model.toys:
                toy.update_color(max_interactions)

        intensity = self.times_interacted_with / max_interactions
        color_red = (round(255 * intensity), round(255 * (1 - intensity)), 0)
        self.color = "#" + ("%02x%02x%02x" % color_red)
