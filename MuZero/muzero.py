import importlib
import math
import os
import pickle
import sys
import time

import numpy
import torch

class Muzero:
    def __init__(self, game_name, config=None, split_resources_in=1):
        # Load the game and the config from the module with the game name
        try:
            game_module = importlib.import_module("games." + game_name)
            self.Game = game_module.Game
            self.config = game_module.MuZeroConfig()
        except ModuleNotFoundError as err:
            print(
                f'{game_name} is not a supported game name, try "cartpole" or refer to the documentation for adding a new game.'
            )
            raise err

        # Overwrite the config
        if config:
            if type(config) is dict:
                for param, value in config.items():
                    setattr(self.config, param, value)
            else:
                self.config = config

        # Fix random generator seed
        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        # Manage GPUs
        total_gpus = (
            self.config.max_num_gpus
            if self.config.max_num_gpus is not None
            else torch.cuda.device_count()
        )
        self.num_gpus = total_gpus / split_resources_in
        if 1 < self.num_gpus:
            self.num_gpus = math.floor(self.num_gpus)

        #ray.init(num_gpus=total_gpus, ignore_reinit_error=True)

        # Checkpoint and replay buffer used to initialize workers
        self.checkpoint = {
            "weights": None,
            "optimizer_state": None,
            "total_reward": 0,
            "muzero_reward": 0,
            "opponent_reward": 0,
            "episode_length": 0,
            "mean_value": 0,
            "training_step": 0,
            "lr": 0,
            "total_loss": 0,
            "value_loss": 0,
            "reward_loss": 0,
            "policy_loss": 0,
            "num_played_games": 0,
            "num_played_steps": 0,
            "num_reanalysed_games": 0,
            "terminate": False,
        }
        self.replay_buffer = {}

        model = models.MuZeroNetwork(config)
        weights = model.get_weights()
        summary = str(model).replace("\n", " \n\n")
   
        self.checkpoint["weights"] = weights
        self.summary = summary
        
        # Workers
        self.self_play_workers = None
        self.test_worker = None
        self.training_worker = None
        self.reanalyse_worker = None
        self.replay_buffer_worker = None
        self.shared_storage_worker = None

    def train(self):
        pass

    def terminate_workers(self):
        pass

    def test(self, render=True, opponent=None, muzero_player=None, num_tests=1, num_gpus=0):
        pass

    def load_model(self, checkpoint_path=None, replay_buffer_path=None):
        '''
            Load model from storage -- use torch.load()
        '''
        pass

if __name__ == "__main__":


    muzero = muzero('tictactoe')
    
    # Select Train, Load and Play
    