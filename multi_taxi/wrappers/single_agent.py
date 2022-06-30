from pettingzoo.utils.wrappers import BaseParallelWraper


class SingleAgentParallelEnvToGymWrapper(BaseParallelWraper):
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

    def reset(self, **kwargs):
        # run `reset` as usual.
        # returned value is a dictionary of observations with a single entry
        obs = self.env.reset(**kwargs)

        # return the single entry value as is.
        # no need for the key (only one agent)
        return next(iter(obs.values()))

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
        self.unwrapped.get_observation_meanings_ = self.unwrapped.get_observation_meanings
        self.unwrapped.get_action_meanings_ = self.unwrapped.get_action_meanings
        self.unwrapped.get_action_map_ = self.unwrapped.get_action_map

        # set original name to wrapper overrides
        self.unwrapped.state_action_transitions = self.__state_action_transitions
        self.unwrapped.get_observation_meanings = self.__get_observation_meanings
        self.unwrapped.get_action_meanings = self.__get_action_meanings
        self.unwrapped.get_action_map = self.__get_action_map

    def __state_action_transitions(self, state, action):
        transitions = self.unwrapped.state_action_transitions_(state, {self.possible_agents[0]: action})
        single_agent_transitions = []
        for res in transitions:
            new_state, rewards, dones, infos, prob = res

            # extract single agent values
            rewards = next(iter(rewards.values()))
            dones = next(iter(dones.values()))
            infos = next(iter(infos.values()))

            single_agent_transitions.append((new_state, rewards, dones, infos))

        return single_agent_transitions

    def __get_observation_meanings(self):
        return self.unwrapped.get_observation_meanings_(self.possible_agents[0])

    def __get_action_meanings(self):
        return self.unwrapped.get_action_meanings_(self.possible_agents[0])

    def __get_action_map(self):
        return self.unwrapped.get_action_map_(self.possible_agents[0])
