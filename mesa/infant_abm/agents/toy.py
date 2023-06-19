import mesa
import random
import numpy as np


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
        color=None
    ):
        """
        Create a new Boid flocker agent.

        Args:
        """

        super().__init__(unique_id, model)
        self.pos = np.array(pos)

        if color is None:
            color = self._random_color()

        self.color_activated = color
        self.color_deactivated = self._deactivated_color()

        self.color = self.color_deactivated

        self.times_interacted_with = 0

    def step(self):
        """
        Get the Boid's neighbors, compute the new vector, and move accordingly.
        """
        pass

    def _deactivated_color(self):
        color_rgb = self.color_activated.lstrip('#')
        color_rgb = tuple(int(color_rgb[i:i+2], 16) for i in [0, 2, 4])
        color_rgb = tuple(c + 88 for c in color_rgb)

        return '#' + ('%02x%02x%02x' % color_rgb)

    def _random_color(self):
        return "#"+''.join([random.choice('345')
                            for j in range(6)])
