from gymnasium.spaces import Discrete

from pettingzoo.utils import BaseParallelWraper


class AssertOutOfBoundsParallelWrapper(BaseParallelWraper):
    """
    this wrapper crashes for out of bounds actions
    Should be used for Discrete spaces
    """

    def __init__(self, env):
        super().__init__(env)
        assert all(
            isinstance(self.action_space(agent), Discrete)
            for agent in getattr(self, "possible_agents", [])
        ), "should only use AssertOutOfBoundsWrapper for Discrete spaces"

    def step(self, actions):
        assert all(
            (action is None and agent not in self.agents) or  # None action for dead agent
            self.action_space(agent).contains(action)  # action within bounds for live agent
            for agent, action in actions.items()
        )

        return super().step(actions)

    def seed(self, seed=None):
        return self.env.seed(seed)
