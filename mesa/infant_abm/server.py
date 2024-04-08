from infant_abm.model import InfantModel
from infant_abm.canvas import Canvas
from infant_abm.agents.infant import Infant
from infant_abm.agents.parent import Parent
from infant_abm.agents.toy import Toy

import mesa

GRID_SIZE = 100


def portrayal(agent):
    if type(agent) is Infant:
        return {
            "Shape": "infant_abm/resources/stickfigure.png",
            "Layer": 1,
            "w": 24,
            "h": 50,
        }

    elif type(agent) is Toy:
        color = agent.color

        return {
            "Shape": "rect",
            "w": 0.025,
            "h": 0.025,
            "Layer": 2,
            "Color": color,
            "Filled": "true",
        }

    elif type(agent) is Parent:
        return {
            "Shape": "infant_abm/resources/stickfigure.png",
            "Layer": 1,
            "w": 32,
            "h": 68,
        }


model_canvas = Canvas(portrayal, 900, 900)

chart_element = mesa.visualization.ChartModule(
    [
        {"Label": "Infant TPS", "Color": "#991144"},
        {"Label": "Parent TPS", "Color": "#441199"},
    ]
)

model_params = {
    "width": GRID_SIZE,
    "height": GRID_SIZE,
    "toy_count": mesa.visualization.Slider("Toy count", 5, 1, 10),
    "exploration": mesa.visualization.Slider("Exploration / Exploatation", 50, 0, 100),
    "responsiveness": mesa.visualization.Slider("Parent responsiveness", 50, 0, 100),
}

server = mesa.visualization.ModularServer(
    InfantModel, [model_canvas, chart_element], "Boids", model_params
)
