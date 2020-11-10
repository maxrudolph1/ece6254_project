import math
from abc import ABC, abstractmethod

import torch

class MuZeroNetwork:
    def __new__(cls, config):
        if config.network == "fullyconnected":
            return MuZeroFullyConnectedNetwork(
                config.observation_shape,
                config.stacked_observations,
                len(config.action_space),
                config.encoding_size,
                config.fc_reward_layers,
                config.fc_value_layers,
                config.fc_policy_layers,
                config.fc_representation_layers,
                config.fc_dynamics_layers,
                config.support_size,
            )
        else:
            raise NotImplementedError(
                'The network parameter should be "fullyconnected" or "resnet".'
            )

def dict_to_cpu(dictionary):
    cpu_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            cpu_dict[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_dict[key] = dict_to_cpu(value)
        else:
            cpu_dict[key] = value
    return cpu_dict

class AbstractNetwork(ABC, torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def initial_inference(self, observation):
        pass

    @abstractmethod
    def recurrent_inference(self, encoded_state, action):
        pass

    def get_weights(self):
        return dict_to_cpu(self.state_dict())

    def set_weights(self, weights):
        self.load_state_dict(weights)

##################################
######## Fully Connected #########

class MuZeroFullyConnectedNetwork(AbstractNetwork):
    def __init__(
        self,
        observation_shape,
        stacked_observations,
        action_space_size,
        encoding_size,
        fc_reward_layers,
        fc_value_layers,
        fc_policy_layers,
        fc_representation_layers,
        fc_dynamics_layers,
        support_size,
    ):
        super().__init__()
        self.action_space_size = action_space_size
        self.full_support_size = 2 * support_size + 1

        self.representation_network = None #mlp

        self.dynamics_encoded_state_network = None #mlp

        self.dynamics_reward_network = None #mlp

        self.prediction_policy_network = None #mlp

        self.prediction_value_network = None # mlp

    def prediction(self, encoded_state):
        policy_logits = None
        value = None
        return policy_logits, value

    def representation(self, observation):
        pass

    def dynamics(self, encoded_state, action):
        pass

    def initial_inference(self, observation):
        pass

    def recurrent_inference(self, encoded_state, action):
        pass

###### End Fully Connected #######
##################################

def mlp(
    input_size,
    layer_sizes,
    output_size,
    output_activation=torch.nn.Identity,
    activation=torch.nn.ELU,
):
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    # create layers for multilayer perceptron here with pytorch
    return torch.nn.Sequential(*layers)

def support_to_scalar(logits, support_size):
    """
    Transform a categorical representation to a scalar
    See paper appendix Network Architecture
    """
    # Decode to a scalar
    probabilities = torch.softmax(logits, dim=1)
    support = (
        torch.tensor([x for x in range(-support_size, support_size + 1)])
        .expand(probabilities.shape)
        .float()
        .to(device=probabilities.device)
    )
    x = torch.sum(support * probabilities, dim=1, keepdim=True)

    # Invert the scaling (defined in https://arxiv.org/abs/1805.11593)
    x = torch.sign(x) * (
        ((torch.sqrt(1 + 4 * 0.001 * (torch.abs(x) + 1 + 0.001)) - 1) / (2 * 0.001))
        ** 2
        - 1
    )
    return x

def scalar_to_support(x, support_size):
    """
    Transform a scalar to a categorical representation with (2 * support_size + 1) categories
    See paper appendix Network Architecture
    """
    # Reduce the scale (defined in https://arxiv.org/abs/1805.11593)
    x = torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + 0.001 * x

    # Encode on a vector
    x = torch.clamp(x, -support_size, support_size)
    floor = x.floor()
    prob = x - floor
    logits = torch.zeros(x.shape[0], x.shape[1], 2 * support_size + 1).to(x.device)
    logits.scatter_(
        2, (floor + support_size).long().unsqueeze(-1), (1 - prob).unsqueeze(-1)
    )
    indexes = floor + support_size + 1
    prob = prob.masked_fill_(2 * support_size < indexes, 0.0)
    indexes = indexes.masked_fill_(2 * support_size < indexes, 0.0)
    logits.scatter_(2, indexes.long().unsqueeze(-1), prob.unsqueeze(-1))
    return logits
