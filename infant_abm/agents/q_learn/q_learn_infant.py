import math
import numpy as np

from infant_abm.agents.infant import Infant, Params
from infant_abm.agents.events import ToyThrown
from infant_abm.agents.position import Position
from infant_abm.agents.infant import infant_actions

from infant_abm.agents.q_learn.q_learning_agent import QLearningAgent


class QLearnInfant(Infant):
    # Agent constants

    # TOY_EVALUATION_DURATION = 3
    # THROW_EVALUATION_DURATION = 20

    PERSISTENCE_BOOST_DURATION = 20

    # COORDINATION_BOOST_VALUE = 0.2
    # PERSISTENCE_BOOST_VALUE = 0.2

    # TOY_EVALUATION_PARENT_CHANCE = 0.7
    # TOY_EVALUATION_INFANT_CHANCE = 0.7

    # THROW_EVALUATION_PARENT_CHANCE = 0.7
    # THROW_EVALUATION_INFANT_CHANCE = 0.7

    GAZE_HISTORY_SIZE = 11

    ALLOWED_ACTIONS = [
        infant_actions.LookForToy,
        infant_actions.Crawl,
        infant_actions.InteractWithToy,
    ]

    def __init__(self, unique_id, model, pos, params: Params):
        super().__init__(unique_id, model, pos, params)

        self.gaze_directions = [None] * self.GAZE_HISTORY_SIZE
        self.current_persistence_boost_duration = 0
        self.q_learning_state = None

        self.q_learning_agent = QLearningAgent(
            model=model, actions=self.get_q_actions()
        )

        self.next_action = infant_actions.LookForToy()

    def step(self):
        self.q_learning_state = self.q_learning_agent.get_state()
        new_q_action = self.q_learning_agent.choose_action()

        self.gaze_directions.append(new_q_action)
        self.gaze_directions.pop(0)

        next_action = super()._perform_action(self.next_action)

        assert type(next_action) in self.ALLOWED_ACTIONS

        self.next_action = next_action

    def advance(self):
        next_state = self.q_learning_agent.get_state()
        reward = self.q_learning_agent.reward(next_state)
        self.q_learning_agent.update_q_table(
            self.q_learning_state, self.gaze_directions[-1], reward, next_state
        )

        # print(f"{self.gaze_directions[-2:]}, {self.model.parent.gaze_directions[-2:]}")

        if np.random.rand() < 0.005:
            # print(next_state)
            # print(self.model.q_learning_agent.q_table)
            print(
                {
                    state: np.argmax(self.q_learning_agent.q_table[state])
                    for state in range(8)
                }
            )

    def get_q_actions(self):
        return [None, self.model.parent] + self.model.get_toys()

    def _step_look_for_toy(self, _action):
        self.current_persistence_boost_duration = 0
        self.params.persistence.reset()

        toys = self.model.get_toys()

        probabilities = np.array([self._toy_probability(toy) for toy in toys])
        probabilities /= probabilities.sum()

        [target] = np.random.choice(toys, size=1, p=probabilities)
        self.velocity = Position.calc_norm_vector(self.pos, target.pos)
        self.target = target

        return infant_actions.Crawl()

    def _step_interact_with_toy(self, _action):
        self.params.coordination.reset()
        throw_direction = None

        if self.params.coordination.e2 > np.random.rand():
            parent_dist = math.dist(self.pos, self.model.parent.pos)
            throw_range = min(self.TOY_THROW_RANGE, parent_dist)
            throw_direction = (
                Position.calc_norm_vector(self.pos, self.model.parent.pos) * throw_range
            )
        else:
            throw_angle = np.random.uniform(0, 2 * np.pi)
            throw_direction = np.array([np.cos(throw_angle), np.sin(throw_angle)])
            throw_direction *= self.TOY_THROW_RANGE

        new_pos = self.target.pos + throw_direction
        self.target.move_agent(new_pos)

        self.target.interact()
        self.model.parent.handle_event(ToyThrown(self.target))

        self.target = None

        return infant_actions.LookForToy()

    def _step_crawl(self, _action):
        if self._target_in_range():
            return infant_actions.InteractWithToy()

        if self._gets_distracted():
            self.target = None
            return infant_actions.LookForToy()

        self._move()

        self.current_persistence_boost_duration += 1
        if self.current_persistence_boost_duration == self.PERSISTENCE_BOOST_DURATION:
            self.params.persistence.reset()

        return infant_actions.Crawl()

    def _get_q_state(self):
        return (
            self.last_action.number,
            self._get_agent_gaze_direction(self),
            self._get_agent_gaze_direction(self.model.parent),
        )

    def _get_agent_gaze_direction(self, agent):
        if agent.gaze_direction is None:
            return 0
        elif agent.gaze_direction == self.model.parent:
            return 1
        else:
            gaze_direction = agent.gaze_direction.unique_id + 2
            assert gaze_direction >= 2 and gaze_direction <= 5
            return gaze_direction
