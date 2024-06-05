from infant_abm.agents.agent import Agent


class Toy(Agent):
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

    def __init__(self, unique_id, model, pos, color=None):
        """
        Create a new Boid flocker agent.

        Args:
        """

        super().__init__(unique_id, model, pos)

        self.model = model
        self.color = "#00FF00"
        self.times_interacted_with = 0

    def step(self):
        """
        Get the Boid's neighbors, compute the new vector, and move accordingly.
        """
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
