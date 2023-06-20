import mesa

from infant_abm.model import InfantModel
from infant_abm.canvas import Canvas
from infant_abm.agents.infant import Infant
from infant_abm.agents.parent import Parent
from infant_abm.agents.toy import Toy
from infant_abm.genetic_model.infant_genome import InfantGenome


def portrayal(agent):
    if type(agent) is Infant:

        return {'Shape': 'infant_abm/resources/infant_2.png', 'Layer': 1, 'w': 50, 'h': 50}

    elif type(agent) is Toy:
        color = None
        if agent.model.infant.target == agent:
            color = agent.color_activated
        else:
            color = agent.color_deactivated
        return {'Shape': 'rect', 'w': 0.025, 'h': 0.025, 'Layer': 2, 'Color': color, 'Filled': 'true'}

    elif type(agent) is Parent:
        return {'Shape': 'infant_abm/resources/parent_2.png', 'Layer': 1, 'w': 50, 'h': 50}


model_canvas = Canvas(portrayal, 900, 900)

chart_element = mesa.visualization.ChartModule(
    [
        {"Label": "Infant satisfaction", "Color": "#991144"},
        {"Label": "Parent satisfaction", "Color": "#441199"},
        # {"Label": "dist_middle", "Color": "#991144"},
        # {"Label": "dist_parent_infant", "Color": "#441199"},
    ]
)

grid_size = 300

model_params = {
    "title": mesa.visualization.StaticText(f'Grid size: {grid_size}'),
    "width": grid_size,
    "height": grid_size,
    "speed": 2,
    "lego_count": mesa.visualization.Slider("Brick count", 5, 1, 15),
    "precision": mesa.visualization.Slider("Infant Precision", 50, 0, 100),
    "exploration": mesa.visualization.Slider("Infant Exploration", 50, 0, 100),
    "coordination": mesa.visualization.Slider("Infant Coordination", 50, 0, 100),
    "responsiveness": mesa.visualization.Slider("Parent responsiveness", 50, 0, 100),
    "relevance": mesa.visualization.Slider("Parent relevance", 50, 0, 100)
}

server = mesa.visualization.ModularServer(
    InfantModel, [model_canvas, chart_element], "Boids", model_params
)
