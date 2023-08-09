#!/usr/bin/env python3

import argparse

from typing import Dict
from pathlib import Path

import ray
from ray import tune
from ray.rllib.env import ParallelPettingZooEnv

from pettingzoo import ParallelEnv
from envs.multi_taxi import MultiTaxiCreator
from algorithms.ppo import PPOCreator

class ParallelEnvRunner:

    def __init__(self, env_name: str, env: ParallelEnv, algorithm_name, config):
        ray.init()

        self.env = env
        self.env_name = env_name

        self.config = config
        self.algorithm_name = algorithm_name
        
        tune.register_env(self.env_name, lambda config: ParallelPettingZooEnv(self.create_env(config)))


    def __del__(self):
        ray.shutdown()

    def create_env(self, config):
        '''
        This is a function called when registering a new env.
        '''
        return self.env
    
    def print_actions(self, actions: Dict[str, str]):
        '''
        Prints the actions the taxis have taken

        @param - a dictionary of taxis and the action they made
        '''

        for taxi, action in actions.items():
            for action_str, action_num in self.env.get_action_map(taxi).items():
                if action_num == action:
                    print(f"{taxi} - {action_str}")

    def train(self):
        '''
        This is the function used to train the policy
        '''
        tune.run(
            self.algorithm_name,
            name=self.algorithm_name,
            stop={"timesteps_total": 5000000},
            checkpoint_freq=10,
            checkpoint_score_attr="episode_reward_mean",
            local_dir="ray_results/" + self.env_name,
            config=self.config.to_dict(),
        )


    def evaluate(self, algorithm, checkpoint_path: str = None, seed: int = 42):
        # Create an agent to handle the environment
        agent = algorithm(config=self.config)
        if checkpoint_path is not None:
            agent.restore(checkpoint_path)

        # Setup the environment
        obs = self.env.reset(seed=seed)
        self.env.render()

        reward_sum = 0
        i = 1

        # TODO: Change this to stop when trunc or term say you should stop
        while True:
            # Get actions from the policy
            action_dict = agent.compute_actions(obs)
            self.print_actions(action_dict)
            
            # Step the environment with the chosen actions
            next_obs, rewards, term, trunc, info = self.env.step(action_dict)
            
            # Update the episode reward
            reward_sum += sum(rewards.values())
            
            # Check if we need to stop the evaluation
            if all(term.values()):
                print("Termineting")
                break

            if all(trunc.values()):
                print("Truncating")
                break    

            obs = next_obs

            # time.sleep(0.1)
            self.env.render()

            print(f"Step {i} - Total Reward: {reward_sum}")
            i += 1

def validate_path(path: str):
    '''
    Check if the path exists
    '''
    path = Path(path)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Path '{path}' does not exist.")
    return path

def parse_args():
    parser = argparse.ArgumentParser(description="Train or evaluate an agent")
    parser.add_argument("--mode", choices=["train", "evaluate"], required=True, help="Choose 'train' or 'evaluate' mode")
    parser.add_argument("--checkpoint-path", type=validate_path, help="Checkpoint path for evaluation")

    try:
        import argcomplete
        argcomplete.autocomplete(parser)
    except ImportError:
        print("Please install argcomplete for auto completion")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    runner = ParallelEnvRunner(MultiTaxiCreator.get_env_name(), MultiTaxiCreator.create_env(), 
                               PPOCreator.get_algo_name(), PPOCreator.get_config(MultiTaxiCreator.get_env_name()))
    if args.mode == 'train':
        runner.train()
    elif args.mode == 'evaluate':
        runner.evaluate(PPOCreator.get_algo(), args.checkpoint_path)
