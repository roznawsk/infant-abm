from infant_abm.model import InfantModel
from infant_abm.canvas import Canvas
from infant_abm.agents.infant_base import InfantBase
from infant_abm.agents.parent import Parent
from infant_abm.agents.toy import Toy

import mesa


# def get_parent_visible(model):
#     return f"parent visible: {model.infant.parent_visible}"

# def get_infant_visible(model):
#     return f"infant visible: {model.parent.infant_visible}"


def portrayal(agent):
    if issubclass(type(agent), InfantBase):
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


model_canvas = Canvas(portrayal, 650, 650)

model_params = {
    "perception": mesa.visualization.Slider("Perception", 0.5, 0.0, 1.0, 0.01),
    "persistence": mesa.visualization.Slider("Persistence", 0.5, 0.0, 1.0, 0.01),
    "coordination": mesa.visualization.Slider("Coordination", 0.5, 0.0, 1.0, 0.01),
}

visibility_chart = mesa.visualization.ChartModule(
    [
        {"Label": "parent-visible", "Color": "Blue"},
        {"Label": "infant-visible", "Color": "Orange"},
    ],
    canvas_height=120,
)

explore_exploit_chart = mesa.visualization.ChartModule(
    [
        {"Label": "explore-exploit-ratio", "Color": "Black"},
    ],
    canvas_height=120,
)


server = mesa.visualization.ModularServer(
    InfantModel,
    [model_canvas, visibility_chart, explore_exploit_chart],
    "Infant ABM",
    model_params,
)
