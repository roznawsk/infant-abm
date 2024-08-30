import numpy as np

from infant_abm.agents.toy import Toy

STATE_SPACE = np.array([2, 2, 2])
STATE_SPACE_SIZE = np.multiply.reduce(STATE_SPACE)
GOAL_STATE = np.array([1, 1, 1])


class QLearningAgent:
    def __init__(self, model, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.model = model
        self.q_table = np.random.rand(STATE_SPACE_SIZE, len(actions))

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = {action: num for num, action in enumerate(actions)}
        self.number_actions = {num: action for action, num in self.actions.items()}

    def choose_action(self):
        state = self.get_state()

        if np.random.rand() < self.epsilon:
            return np.random.choice(list(self.actions.keys()))  # Explore
        else:
            return self.number_actions[np.argmax(self.q_table[state])]  # Exploit

    def update_q_table(self, state, action, reward, next_state):
        action = self.actions[action]
        best_next_action = np.max(self.q_table[next_state])

        self.q_table[state, action] += self.alpha * (
            reward + self.gamma * best_next_action - self.q_table[state, action]
        )

    def get_state(self):
        # [infant_looked_at_toy, parent_looked_at_toy, mutual_gaze]

        raw_state = np.array(
            [
                int(self._infant_looked_at_toy()),
                int(self._parent_looked_at_toy_after_infant()),
                int(self._mutual_gaze()),
            ]
        )

        multiplier = np.array([4, 2, 1])
        return np.sum(raw_state * multiplier)

    def reward(self, state):
        if np.all(state == GOAL_STATE):
            return 1
        else:
            return 0

    def _infant_looked_at_toy(self):
        return any([isinstance(obj, Toy) for obj in self.model.infant.gaze_directions])

    def _parent_looked_at_toy_after_infant(self):
        for i, obj in enumerate(self.model.infant.gaze_directions):
            if (
                isinstance(obj, Toy)
                and obj in self.model.parent.gaze_directions[i : i + 5]
            ):
                return True

        return False

    def _mutual_gaze(self):
        return (
            self.model.infant.gaze_directions[-1] == self.model.parent
            and self.model.parent.gaze_directions[-1] == self.model.infant
        )
