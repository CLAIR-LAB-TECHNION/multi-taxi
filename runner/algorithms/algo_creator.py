from abc import ABC, abstractmethod


class AlgoCreator(ABC):

    @abstractmethod
    def get_algo():
        pass

    @abstractmethod
    def get_algo_name():
        pass

    @abstractmethod
    def get_config(env_name):
        pass