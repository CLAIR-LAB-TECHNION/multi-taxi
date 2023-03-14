from pettingzoo.utils import BaseParallelWraper
from pettingzoo.utils.env_logger import EnvLogger


class OrderEnforcingParallelWrapper(BaseParallelWraper):

    def __init__(self, env):
        self._has_reset = False
        super().__init__(env)

    def __getattr__(self, value):
        if value == "possible_agents":
            EnvLogger.error_possible_agents_attribute_missing("possible_agents")
        elif value == "observation_spaces":
            raise AttributeError(
                "The base environment does not have an possible_agents attribute. Use the environments "
                "`observation_space` method instead"
            )
        elif value == "action_spaces":
            raise AttributeError(
                "The base environment does not have an possible_agents attribute. Use the environments `action_space` "
                "method instead"
            )
        else:
            return getattr(self.unwrapped, value)

    def render(self, mode="human"):
        if not self._has_reset:
            EnvLogger.error_render_before_reset()
        else:
            self.env.render()

    def step(self, action):
        if not self._has_reset:
            EnvLogger.error_step_before_reset()
        elif not self.agents:
            EnvLogger.warn_step_after_terminated_truncated()
        else:
            return super().step(action)

    def state(self):
        if not self._has_reset:
            EnvLogger.error_state_before_reset()
        return super().state()

    def reset(self, **kwargs):
        self._has_reset = True
        return super().reset(**kwargs)

    def __str__(self):
        str(self.unwrapped)

    def seed(self, seed=None):
        return self.env.seed(seed)
