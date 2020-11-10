import copy
import time

import numpy
import torch

import models

class ReplayBuffer:
    """
    Class which run in a dedicated thread to store played games and generate batch.
    """

    def __init__(self, initial_checkpoint, initial_buffer, config):
        self.config = config

        # initialized in muzero as {}, and loaded wtih muzero.load_module(...)
        self.buffer = copy.deepcopy(initial_buffer)

        pass

    def save_game(self, game_history, shared_storage=None):
        # Append game to buffer
        pass

    def get_buffer(self):
        return self.buffer

    def get_batch(self):
        # select X amount of random games at random states and return for training
        pass

    def sample_game(self, force_uniform=False):
        pass

    def sample_position(self, game_history, force_uniform=False):
        pass      

    def update_game_history(self, game_id, game_history):
        pass     

    def update_priorities(self, priorities, index_info):
        pass
    
    def compute_target_value(self, game_history, index):
         pass       

    def make_target(self, game_history, state_index):
        pass     


# Keeping this for now, may delete later.
class Reanalyse:
    """
    Class which run in a dedicated thread to update the replay buffer with fresh information.
    See paper appendix Reanalyse.
    """

    def __init__(self, initial_checkpoint, config):
        self.config = config

        # Fix random generator seed
        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        # Initialize the network
        self.model = models.MuZeroNetwork(self.config)
        self.model.set_weights(initial_checkpoint["weights"])
        self.model.to(torch.device("cuda" if self.config.reanalyse_on_gpu else "cpu"))
        self.model.eval()

        self.num_reanalysed_games = initial_checkpoint["num_reanalysed_games"]

    def reanalyse(self, replay_buffer, shared_storage):
        while shared_storage.get_info.remote("num_played_games") < 1:
            time.sleep(0.1)

        while shared_storage.get_info.remote("training_step") < self.config.training_steps and not shared_storage.get_info.remote("terminate"):
            self.model.set_weights(shared_storage.get_info.remote("weights"))

            game_id, game_history, _ = replay_buffer.sample_game.remote(force_uniform=True)
            

            # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
            if self.config.use_last_model_value:
                observations = [
                    game_history.get_stacked_observations(
                        i, self.config.stacked_observations
                    )
                    for i in range(len(game_history.root_values))
                ]

                observations = (
                    torch.tensor(observations)
                    .float()
                    .to(next(self.model.parameters()).device)
                )
                values = models.support_to_scalar(
                    self.model.initial_inference(observations)[0],
                    self.config.support_size,
                )
                game_history.reanalysed_predicted_root_values = (
                    torch.squeeze(values).detach().numpy()
                )

            replay_buffer.update_game_history.remote(game_id, game_history)
            self.num_reanalysed_games += 1
            shared_storage.set_info.remote(
                "num_reanalysed_games", self.num_reanalysed_games
            )
