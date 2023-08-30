from runner.envs.env_creator import EnvCreator
from copy import deepcopy

from citylearn.citylearn import CityLearnEnv
from problem.CityLearn.CityState import CityState
from problem.CityLearn.kpi__ import plot_simulation_summary


# Assuming the rest of the required imports...

class SmartGridCreator(EnvCreator):
    ENV_NAME = "city_learn_env"

    @staticmethod
    def get_env_name():
        return "city_learn_env"

    @staticmethod
    def create_env(building_indices=None, schema_path='data/schema.json'):
        """
        Create a CityLearn environment with the given settings.

        Args:
            building_indices: A list of indices for the buildings to consider in the problem.
            schema_path: Path to the environment schema. Default is 'data/schema.json'.

        Returns:
            An instance of the CityLearnEnv environment.
        """

        # Initialize the environment
        env = CityLearnEnv(schema=schema_path)

        # If building_indices are provided, select only the indicated buildings.
        if building_indices:
            env.buildings = [env.buildings[i] for i in building_indices]

        # Other environment customizations can be added here as needed...

        return env
