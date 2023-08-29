from .env_creator import EnvCreator


class CityLearnCreator(EnvCreator):
    ENV_NAME = "citylearn_env"

    def get_env_name(self):
        return self.ENV_NAME

    def create_env(self):
        # Your custom configuration for the environment
        config = {
            "data_path": "path/to/data/folder",
            "building_attributes": "path/to/building_attributes.csv",
            "weather_file": "path/to/weather_file.csv",
            "solar_profile": "path/to/solar_profile.csv",
            "carbon_intensity": "path/to/carbon_intensity.csv",
            "building_ids": ["building1", "building2"],
            # Add more configuration options as needed
        }

        return CityLearn(**config)