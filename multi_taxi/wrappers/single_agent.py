import gymnasium
from pettingzoo.utils.wrappers import BaseParallelWraper


class SingleAgentParallelEnvToGymWrapper(BaseParallelWraper, gymnasium.Env):
    """
    A wrapper for single-agent parallel environments aligning the environments'
    API with OpenAI Gym.
    """
    # gym API class variables
    reward_range = (-float("inf"), float("inf"))
    spec = None

    def __init__(self, env):
        super().__init__(env)

        # assert single agent environment
        assert len(env.possible_agents) == 1

    @property
    def render_mode(self):
        return self.env.unwrapped.render_mode

    def reset(self, **kwargs):
        # get pettingzoo specific reset arguments
        seed = kwargs.pop('seed', None)  # random seed (also common in gym)
        return_info = kwargs.pop('return_info', False)  # pettingzoo exclusive

        # run `reset` as usual.
        out = self.env.reset(seed=seed,
                             return_info=return_info,
                             options=kwargs or None)

        # check if infos are a part of the reset return
        if return_info:
            obs, infos = out
        else:
            obs = out
            infos = {k: {} for k in obs.keys()}

        # return the single entry value as is.
        # no need for the key (only one agent)
        return next(iter(obs.values())), next(iter(infos.values()))

    def step(self, action):
        # step using "joint action" of a single agnet as a dictionary
        step_rets = self.env.step({self.env.agents[0]: action})

        # unpack step return values from their dictionaries
        return tuple(next(iter(ret.values())) for ret in step_rets)

    @property  # make property for gym-like access
    def action_space(self, _=None):  # ignore second argument in API
        # get action space of the single agent
        return self.env.action_space(self.env.possible_agents[0])

    @property  # make property for gym-like access
    def observation_space(self, _=None):  # ignore second argument in API
        # get observation space of the single agent
        return self.env.observation_space(self.env.possible_agents[0])

    def seed(self, seed=None):
        return self.env.seed(seed)


class SingleTaxiWrapper(SingleAgentParallelEnvToGymWrapper):
    def __init__(self, env):
        super().__init__(env)

        # override environment extra APIs

        # rename original functions
        self.unwrapped.state_action_transitions_ = self.unwrapped.state_action_transitions
        self.unwrapped.observe_all_ = self.unwrapped.observe_all
        self.unwrapped.get_observation_meanings_ = self.unwrapped.get_observation_meanings
        self.unwrapped.get_action_meanings_ = self.unwrapped.get_action_meanings
        self.unwrapped.get_action_map_ = self.unwrapped.get_action_map

        # set original name to wrapper overrides
        self.unwrapped.state_action_transitions = self.__state_action_transitions
        self.unwrapped.observe_all = self.__observe_all
        self.unwrapped.get_observation_meanings = self.__get_observation_meanings
        self.unwrapped.get_action_meanings = self.__get_action_meanings
        self.unwrapped.get_action_map = self.__get_action_map

    def __state_action_transitions(self, state, action):
        transitions = self.unwrapped.state_action_transitions_(state, {self.possible_agents[0]: action})
        single_agent_transitions = []
        for res in transitions:
            new_state, rewards, terms, truncs, infos, prob = res

            # extract single agent values
            rewards = next(iter(rewards.values()))
            terms = next(iter(terms.values()))
            truncs = next(iter(truncs.values()))
            infos = next(iter(infos.values()))

            single_agent_transitions.append((new_state, rewards, terms, truncs, infos))

        return single_agent_transitions

    def __observe_all(self):
        return next(iter(self.unwrapped.observe_all_().values()))

    def __get_observation_meanings(self):
        return self.unwrapped.get_observation_meanings_(self.possible_agents[0])

    def __get_action_meanings(self):
        return self.unwrapped.get_action_meanings_(self.possible_agents[0])

    def __get_action_map(self):
        return self.unwrapped.get_action_map_(self.possible_agents[0])
