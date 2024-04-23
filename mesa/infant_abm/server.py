from infant_abm.model import InfantModel
from infant_abm.canvas import Canvas
from infant_abm.agents.infant import Infant
from infant_abm.agents.parent import Parent
from infant_abm.agents.toy import Toy

import mesa


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

model_params = {
    "perception": mesa.visualization.Slider("Perception", 50, 0, 100),
    "persistence": mesa.visualization.Slider("Persistence", 50, 0, 100),
    "coordination": mesa.visualization.Slider("Coordination", 50, 0, 100),
    "explore_exploit_ratio": mesa.visualization.Slider(
        "Explore-Exploit Ratio", 50, 0, 100
    ),
}

server = mesa.visualization.ModularServer(
    InfantModel, [model_canvas], "Infant", model_params
)
