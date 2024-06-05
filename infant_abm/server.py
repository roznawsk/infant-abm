from infant_abm.model import InfantModel
from infant_abm.canvas import Canvas
from infant_abm.agents.infant import InfantBase
from infant_abm.agents.parent_base import ParentBase
from infant_abm.agents.toy import Toy

import mesa


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

    elif issubclass(type(agent), ParentBase):
        return {
            "Shape": "infant_abm/resources/stickfigure.png",
            "Layer": 1,
            "w": 32,
            "h": 68,
        }


# infant_class = "NoVisionInfant"
infant_class = "SeqVisionInfant"


def get_infant_class(model):
    return infant_class


model_canvas = Canvas(portrayal, 650, 650)

model_params = {
    "infant_class": infant_class,
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
        {"Label": "perception", "Color": "Black"},
        {"Label": "persistence", "Color": "Blue"},
        {"Label": "coordination", "Color": "Green"},
    ],
    canvas_height=120,
)


server = mesa.visualization.ModularServer(
    InfantModel,
    [
        get_infant_class,
        model_canvas,
        visibility_chart,
        explore_exploit_chart,
    ],
    "Infant ABM",
    model_params,
)
