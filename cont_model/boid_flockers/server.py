import mesa

from .model import ToddlerModel
from .SimpleContinuousModule import SimpleCanvas
from .agents.toddler import Toddler
from .agents.toy import Toy


def portrayal(agent):
    if type(agent) is Toddler:

        return {'Shape': 'boid_flockers/resources/toddler_2.png', 'Layer': 1, 'w': 50, 'h': 50}

    elif type(agent) is Toy:
        return {'Shape': 'rect', 'w': 0.025, 'h': 0.025, 'Layer': 2, 'Color': agent.color, 'Filled': 'true'}


model_canvas = SimpleCanvas(portrayal, 900, 900)

chart_element = mesa.visualization.ChartModule(
    [
        {"Label": "Steps/Interaction", "Color": "#AA0000"},
    ]
)

model_params = {
    "title": mesa.visualization.StaticText("Grid size: 300"),
    "lego_count": 8,
    "width": 300,
    "height": 300,
    "speed": 2,
    "precision": mesa.visualization.Slider("Precision", 50, 0, 100),
    "perception": mesa.visualization.Slider("Perception", 50, 0, 100),
    "coordination": mesa.visualization.Slider("Coordination", 50, 0, 100)
}

server = mesa.visualization.ModularServer(
    ToddlerModel, [model_canvas, chart_element], "Boids", model_params
)
