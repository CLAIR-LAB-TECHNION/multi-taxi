from .algo_creator import AlgoCreator

from ray.rllib.algorithms.dqn import DQN, DQNConfig

class DQNCreator(AlgoCreator):

    def get_algo():
        return DQN
    
    def get_algo_name():
        return "DQN"
    
    def get_config(env_name):
        return DQNConfig()                                              \
            .environment(env=env_name, disable_env_checking=True)       \
            .framework(framework="tf")                                  \
            .rollouts(num_rollout_workers=0, enable_connectors=False)   \
            .training(
                    train_batch_size=512,
                    lr=2e-5,
                    gamma=0.99,
                )