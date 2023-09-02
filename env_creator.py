from abc import ABC, abstractmethod


class EnvCreator(ABC):

    @staticmethod
    @abstractmethod
    def get_env_name():
        pass

    @staticmethod
    @abstractmethod
    def create_env():
        pass

