import copy
import importlib
import math
import os
import pickle
import sys
import time
import threading

import numpy
import torch

sys.path.insert(0, '..')
from TrafficEnvironment import traffic_environment

import muzero_config
import models
import replay_buffer
import self_play
import shared_storage
import trainer

class Muzero:
    def __init__(self, game_name, config=None, split_resources_in=1):
        # Load the game and the config from the module with the game name
        '''
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
        '''
        self.Game = traffic_environment.TrafficEnv()
        self.config = muzero_config.MuZeroConfig()
        #self.config.observation_shape = (1, 1, len(self.Game.observation()))
        self.config.observation_shape = (1, 1, len(self.Game.observation_space.sample()))
        self.config.action_space = list(range(0, 2 ** len(self.Game.action_space.sample()))
        #self.config.action_space = list(range(2**self.Game.action_space.shape[0]))


        # Fix random generator seed
        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        # Manage GPUs
        # TODO could trim this out
        total_gpus = (
            self.config.max_num_gpus
            if self.config.max_num_gpus is not None
            else torch.cuda.device_count()
        )
        self.num_gpus = total_gpus / split_resources_in
        if 1 < self.num_gpus:
            self.num_gpus = math.floor(self.num_gpus)


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
            "terminate": False
        }
        self.replay_buffer = {}

        model = models.MuZeroNetwork(self.config)
        weights = model.get_weights()
        self.summary = str(model).replace("\n", " \n\n") 
        self.checkpoint["weights"] = copy.deepcopy(weights)

        
        # Workers
        self.self_play_workers = []
        self.test_worker = None
        self.training_worker = None
        self.reanalyse_worker = None
        self.replay_buffer_worker = None
        self.shared_storage_worker = None

    def train(self):
        # Manage GPUs
        '''
        if 0 < self.num_gpus:
            num_gpus_per_worker = self.num_gpus / (
                self.config.train_on_gpu
                + self.config.num_workers * self.config.selfplay_on_gpu
                + log_in_tensorboard * self.config.selfplay_on_gpu
                + self.config.use_last_model_value * self.config.reanalyse_on_gpu
            )
            if 1 < num_gpus_per_worker:
                num_gpus_per_worker = math.floor(num_gpus_per_worker)
        else:
            num_gpus_per_worker = 0
        '''

        # Initialize Worker Threads
        for SP_worker_index in range(self.config.num_workers):
            self.self_play_workers.append(
                self_play.SelfPlay(self.checkpoint, self.Game, self.config, self.config.seed + SP_worker_index)
            )
        self.training_worker = trainer.Trainer(self.checkpoint, self.config)
       
        self.replay_buffer_worker = replay_buffer.ReplayBuffer(self.checkpoint, self.replay_buffer, self.config)
        self.shared_storage_worker = shared_storage.SharedStorage(self.checkpoint, self.config)
        self.shared_storage_worker.set_info("terminate", False)
        #Launch Workers

        for SP_worker in self.self_play_workers:
            self.self_play_workers[SP_worker_index].continuous_self_play(self.shared_storage_worker, self.replay_buffer_worker)
        self.training_worker.continuous_update_weights(self.shared_storage_worker, self.replay_buffer_worker, self.shared_storage_worker)

    def terminate_workers(self):
        if self.shared_storage_worker:
            self.shared_storage_worker.set_info("terminate", True)
            self.checkpoint = self.shared_storage_worker.get_checkpoint()
            
        if self.replay_buffer_worker:
            self.replay_buffer = self.replay_buffer_worker.get_buffer()

        print("\nShutting down workers...")

        self.self_play_workers = None
        self.test_worker = None
        self.training_worker = None
        self.reanalyse_worker = None
        self.replay_buffer_worker = None
        self.shared_storage_worker = None

    def test(self, render=True, opponent=None, muzero_player=None, num_tests=1, num_gpus=0):
        pass

    def load_model(self, checkpoint_path=None, replay_buffer_path=None):
        '''
            Load model from storage -- use torch.load()
        '''
        pass

if __name__ == "__main__":
    print(torch.cuda.is_available())
    muzero = Muzero('traffic sim')
    muzero.train()

    choice = input("Press a key to terminate operation: ")
    muzero.terminate_workers()

    if self.config.save_model:
        # Persist replay buffer to disk
        print("\n\nPersisting replay buffer games to disk...")
        pickle.dump(
            self.replay_buffer,
            open(os.path.join(self.config.results_path, "replay_buffer.pkl"), "wb"),
        )
    # Select Train, Load and Play

    # Need either of the following if using DistributedDataParallel()  - DDP
    # reference https://github.com/pytorch/examples/blob/master/imagenet/main.py

    # Spawn a thread for multiprocessing will pickle model and save to desk
    #torch.multiprocessing.spawn(function_name, args=(arg1,),nprocs=#,join=True)

    # sync with nccl, mpi, 
    #torch.distributed.init_process_group()

    

    