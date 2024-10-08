import math
import numpy as np

from infant_abm.agents import infant_actions
from infant_abm.agents.infant import Infant, Params
from infant_abm.agents.events import ToySelected, ToyThrown
from infant_abm.agents.position import Position
from infant_abm.agents.toy import Toy


class NoVisionInfant(Infant):
    ALLOWED_ACTIONS = [
        infant_actions.LookForToy,
        infant_actions.Crawl,
        infant_actions.InteractWithToy,
    ]

    def __init__(self, unique_id, model, pos, params: Params):
        super().__init__(unique_id, model, pos, params)

        self.target: Toy = None

        self.next_action = infant_actions.LookForToy()

    def step(self):
        next_action = super()._perform_action(self.next_action)

        assert type(next_action) in self.ALLOWED_ACTIONS

        self.next_action = next_action

    def _step_interact_with_toy(self, _action):
        throw_direction = None

        if self.params.coordination.e2 > np.random.rand():
            parent_dist = math.dist(self.pos, self.model.parent.pos)
            throw_range = min(self.TOY_THROW_RANGE, parent_dist)
            throw_direction = (
                Position.calc_norm_vector(self.pos, self.model.parent.pos) * throw_range
            )
        else:
            throw_direction = np.random.rand(2)
            throw_direction = (
                throw_direction / np.linalg.norm(throw_direction) * self.TOY_THROW_RANGE
            )

        new_pos = self.target.pos + throw_direction
        self.target.move_agent(new_pos)
        self.rotate_towards(new_pos)

        self.model.parent.handle_event(ToyThrown(self.target))

        self.target.interact()
        self.target = None

        return infant_actions.LookForToy()

    def _step_look_for_toy(self, _action):
        toys = self.model.get_toys()

        probabilities = np.array([self._toy_probability(toy) for toy in toys])
        probabilities /= probabilities.sum()

        [target] = np.random.choice(toys, size=1, p=probabilities)
        self.velocity = Position.calc_norm_vector(self.pos, target.pos)
        self.target = target
        self.model.parent.handle_event(ToySelected(self.target))
        return infant_actions.Crawl()

    def _step_crawl(self, _action):
        if self._target_in_range():
            return infant_actions.InteractWithToy()

        if self._gets_distracted():
            self.target = None
            return infant_actions.LookForToy()

        self._move()
        return infant_actions.Crawl()
