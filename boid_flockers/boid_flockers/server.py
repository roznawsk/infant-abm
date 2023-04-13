import mesa

from .model import BoidFlockers
from .SimpleContinuousModule import SimpleCanvas
from .agents import Toddler, Toy


def portrayal(agent):
    if type(agent) is Toddler:
        # portrayal["Shape"] = "./resources/toddler.png"
        # portrayal["Shape"] = "circle"
        # portrayal["Color"] = "red"

        # # https://icons8.com/web-app/433/sheep
        # portrayal["scale"] = 0.8
        # portrayal["Layer"] = 1

        return {'Shape': 'circle', 'Layer': 1, 'Color': 'grey', 'Filled': 'true', 'r': 16}

    elif type(agent) is Toy:
        return {'Shape': 'rect', 'w': 0.025, 'h': 0.025, 'Layer': 1, 'Color': agent.color, 'Filled': 'true'}


boid_canvas = SimpleCanvas(portrayal, 900, 900)
model_params = {
    "title": mesa.visualization.StaticText("Grid size: 300"),
    "population": 8,
    "width": 300,
    "height": 300,
    "speed": 2,
    "vision": mesa.visualization.Slider("Vision", 70, 1, 100),
    "persistence": mesa.visualization.Slider("Persistence", 95, 0, 100)
}

server = mesa.visualization.ModularServer(
    BoidFlockers, [boid_canvas], "Boids", model_params
)
