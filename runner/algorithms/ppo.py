from .algo_creator import AlgoCreator

from ray.rllib.algorithms.ppo import PPO, PPOConfig

class PPOCreator(AlgoCreator):

    def get_algo():
        return PPO
    
    def get_algo_name():
        return "PPO"
    
    def get_config(env_name):
        return PPOConfig()                                              \
            .environment(env=env_name, disable_env_checking=True)       \
            .framework(framework="tf")                                  \
            .rollouts(num_rollout_workers=0, enable_connectors=False)   \
            .training(
                    train_batch_size=512,
                    lr=2e-5,
                    gamma=0.99,
                    lambda_=0.9,
                    use_gae=True,
                    clip_param=0.4,
                    grad_clip=None,
                    entropy_coeff=0.1,
                    vf_loss_coeff=0.25,
                    sgd_minibatch_size=64,
                    num_sgd_iter=10,
                )