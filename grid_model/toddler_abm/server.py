import mesa

from toddler_abm.agents import Toddler, LegoBrick
from toddler_abm.model import ToddlerABM


def toddler_abm_portrayal(agent):
    if agent is None:
        return

    portrayal = {}

    if type(agent) is Toddler:
        portrayal["Shape"] = "toddler_abm/resources/toddler.png"
        # https://icons8.com/web-app/433/sheep
        portrayal["scale"] = 0.8
        portrayal["Layer"] = 1

    elif type(agent) is LegoBrick:
        portrayal["Color"] = [agent.color for _ in range(3)]

        portrayal["Shape"] = "rect"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["w"] = 1
        portrayal["h"] = 1

    return portrayal


canvas_element = mesa.visualization.CanvasGrid(toddler_abm_portrayal, 15, 15, 900, 900)
# chart_element = mesa.visualization.ChartModule(
#     [
#         {"Label": "Bricks", "Color": "#666666"},
#         {"Label": "Toddler", "Color": "#00AA00"},
#     ]
# )

model_params = {
    # The following line is an example to showcase StaticText.
    "title": mesa.visualization.StaticText("Parameters:"),
    "initial_bricks": mesa.visualization.Slider(
        "Initial Brick Population", 5, 1, 50
    ),
    "drops_brick": mesa.visualization.Slider("Brick drop probability", 40, 1, 100)
}

server = mesa.visualization.ModularServer(
    ToddlerABM, [canvas_element], "Wolf Sheep Predation", model_params
)
server.port = 8521
