import mesa

from .model import ToddlerModel
from .SimpleContinuousModule import SimpleCanvas
from .agents.toddler import Toddler
from .agents.toy import Toy


def portrayal(agent):
    if type(agent) is Toddler:

        return {'Shape': 'boid_flockers/resources/toddler_2.png', 'Layer': 1, 'w': 50, 'h': 50}

    elif type(agent) is Toy:
        color = None
        if agent.model.toddler.target == agent:
            color = agent.color_activated
        else:
            color = agent.color_deactivated
        return {'Shape': 'rect', 'w': 0.025, 'h': 0.025, 'Layer': 2, 'Color': color, 'Filled': 'true'}


model_canvas = SimpleCanvas(portrayal, 900, 900)

chart_element = mesa.visualization.ChartModule(
    [
        {"Label": "Steps/Interaction", "Color": "#AA0000"},
    ]
)

grid_size = 300

model_params = {
    "title": mesa.visualization.StaticText(f'Grid size: {grid_size}'),
    "lego_count": 8,
    "width": grid_size,
    "height": grid_size,
    "speed": 2,
    "precision": mesa.visualization.Slider("Precision", 100, 0, 100),
    "exploration": mesa.visualization.Slider("Exploration", 50, 0, 100),
    "coordination": mesa.visualization.Slider("Coordination", 100, 0, 100)
}

server = mesa.visualization.ModularServer(
    ToddlerModel, [model_canvas, chart_element], "Boids", model_params
)
