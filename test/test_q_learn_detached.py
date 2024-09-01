from infant_abm import InfantParams
from infant_abm.config import Config


from infant_abm.model import InfantModel
from infant_abm.agents import QLearnDetachedInfant, QLearnDetachedParent


model = InfantModel(
    infant_class=QLearnDetachedInfant,
    parent_class=QLearnDetachedParent,
    config=Config(),
    infant_params=InfantParams.from_array([0.5, 0.5, 0.5]),
)


def test_q_learning_agent_get_state():
    agent = model.infant.q_learning_agent

    model.infant.gaze_directions = [None] * 11
    model.parent.gaze_directions = [None] * 11
    assert agent.get_state() == 0

    model.infant.gaze_directions = [model.toys[0]] + [None] * 10
    model.parent.gaze_directions = [None] * 11
    assert agent.get_state() == 4

    model.infant.gaze_directions = [model.toys[0]] + [None] * 10
    model.parent.gaze_directions = [None] + [model.toys[0]] + [None] * 9
    assert agent.get_state() == 6

    model.infant.gaze_directions = [model.toys[0]] + [None] * 10
    model.parent.gaze_directions = [None] * 6 + [model.toys[0]] + [None] * 4
    assert agent.get_state() == 4

    model.infant.gaze_directions = [model.toys[0]] + [None] * 10
    model.parent.gaze_directions = [None] * 4 + [model.toys[0]] + [None] * 6
    assert agent.get_state() == 6

    model.infant.gaze_directions = [model.toys[0]] + [None] * 9 + [model.parent]
    model.parent.gaze_directions = (
        [None] * 5 + [model.toys[0]] + [None] * 4 + [model.infant]
    )
    assert agent.get_state() == 7

    model.infant.gaze_directions = [None] * 10 + [model.parent]
    model.parent.gaze_directions = [None] * 10 + [model.infant]
    assert agent.get_state() == 1
