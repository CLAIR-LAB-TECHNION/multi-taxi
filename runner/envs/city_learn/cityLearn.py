from env_creator import EnvCreator

# pip install git+https://github.com/CLAIR-LAB-TECHNION/CLAIR_grid
# from clair_grid.environment.citylearn_wrapper import CityLearnWrapper

from citylearn.citylearn import CityLearnEnv
from citylearn.reward_function import SolarPenaltyReward


class CityLearnWrapper:
    def __init__(self, building_indices=None):
        # Path to the environment schema
        schema_path = 'envs/city_learn/data/schema.json'
        max_num_buildings = 17
        # dataset_name = 'citylearn_challenge_2022_phase_1'
        env = CityLearnEnv(schema=schema_path)
        env.reward_function = SolarPenaltyReward(env)

        if building_indices:
            env.buildings = [env.buildings[i] for i in building_indices]

        self.env = env

    def reset(self):
        return self.env.reset()

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def agents(self):
        return self.env.buildings


class CityLearnCreator(EnvCreator):
    ENV_NAME = "city_learn_env"

    @staticmethod
    def get_env_name():
        return "city_learn_env"

    @staticmethod
    def create_env():
        building_indices = None
        # schema_path = '~/data/schema.json'

        # Initialize the environment
        # dataset_name = 'citylearn_challenge_2022_phase_1'
        # env = CityLearnEnv(schema=schema_path)
        # env = CityLearnEnv(schema=dataset_name)
        # env.reward_function = SolarPenaltyReward(env)
        # env = CityLearnWrapper(env)

        # create the agents
        # num_of_agents = len(env.buildings)

        # # initialize the coodinator
        # b_random_order = False
        # coordinator = SGDecentralizedCoordinator(env_wrapper, agents, agent_ids, b_random_order)
        #
        # # run training
        # coordinator.run(1000, b_log=True, b_train=True, b_evaluate=False)

        # If building_indices are provided, select only the indicated buildings.

        return CityLearnWrapper(building_indices)