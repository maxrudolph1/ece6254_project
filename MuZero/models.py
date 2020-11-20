# Code borrows heavily from werner-duvaud/muzero-general

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
                #config.action_space,
                config.encoding_size,
                config.fc_reward_layers,
                config.fc_value_layers,
                config.fc_policy_layers,
                config.fc_representation_layers,
                config.fc_dynamics_layers,
                config.support_size,
            )
        else:
            raise NotImplementedError('The network parameter should be "fullyconnected".')

# Note: called in trainer.py
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
    def recurrent_inference(self, hidden_state, action):
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

        
        # TODO look into DataParallel() usage.
        self.representation_network = mlp(observation_shape[0] * observation_shape[1] * observation_shape[2] * (stacked_observations + 1)
                + stacked_observations * observation_shape[1] * observation_shape[2],
                fc_representation_layers,
                encoding_size
            )
        #torch.nn.parallel.DistributedDataParallel(#)

        self.dynamics_hidden_state_network = mlp( encoding_size + action_space_size,
                fc_dynamics_layers,
                encoding_size   
            )
        #torch.nn.parallel.DistributedDataParallel( #)

        self.dynamics_reward_network = mlp(encoding_size, fc_reward_layers, self.full_support_size)
        #torch.nn.parallel.DistributedDataParallel(#)

        self.prediction_policy_network = mlp(encoding_size, fc_reward_layers, action_space_size)
        #torch.nn.parallel.DistributedDataParallel(#)

        self.prediction_value_network = mlp(encoding_size, fc_value_layers, self.full_support_size)
        #torch.nn.parallel.DistributedDataParallel(#)

    # perform prediction inference hidden_state-> policy, value
    def prediction(self, hidden_state):
        policy_logits = None
        value = None
        policy_logits = self.prediction_policy_network(hidden_state)
        value = self.prediction_value_network(hidden_state)
        return policy_logits, value

    def representation(self, observation):

        # From arxiv muzero paper training appendix
        # "To improve the learning process and bound the activations, we also scale the hidden state to the same range as
        # the action input ([0, 1]): sscaled = s−min(s)/(max(s)−min(s))""

        hidden_state = self.representation_network(observation.view(observation.shape[0], -1))

        min_hidden_state = hidden_state.min(1, keepdim=True)[0]
        max_hidden_state = hidden_state.max(1, keepdim=True)[0]
        scale_hidden_state = max_hidden_state - min_hidden_state
        scale_hidden_state[scale_hidden_state < 1e-5] += 1e-5
        hidden_state_normalized = (
            hidden_state - min_hidden_state
        ) / scale_hidden_state
        
        return hidden_state_normalized

    def dynamics(self, hidden_state, action):
        # From arxiv muzero paper network architecture appendix
        # "For the dynamics function (which always operates at the downsampled resolution of 6x6), the action is first
        # encoded as an image, then stacked with the hidden state of the previous step along the plane dimension"
        action_one_hot = (
            torch.zeros((action.shape[0], self.action_space_size))
            .to(action.device)
            .float()
        )
        action_one_hot.scatter_(1, action.long(), 1.0)
        model_input = torch.cat((hidden_state, action_one_hot), dim=1)
        next_hidden_state = self.dynamics_hidden_state_network(model_input)
        reward = self.dynamics_reward_network(next_hidden_state)

        #normalize hidden_state
        min_next_hidden_state = next_hidden_state.min(1, keepdim=True)[0]
        max_next_hidden_state = next_hidden_state.max(1, keepdim=True)[0]
        scale_next_hidden_state = max_next_hidden_state - min_next_hidden_state
        scale_next_hidden_state[scale_next_hidden_state < 1e-5] += 1e-5
        next_hidden_state_normalized = (
            next_hidden_state - min_next_hidden_state
        ) / scale_next_hidden_state

        return next_hidden_state_normalized, reward

    def initial_inference(self, observation):
        hidden_state = self.representation(observation)
        policy, value = self.prediction(hidden_state)

        # Generate dummy reward of 0
        reward = torch.log(
            (
                torch.zeros(1, self.full_support_size)
                .scatter(1, torch.tensor([[self.full_support_size // 2]]).long(), 1.0)
                .repeat(len(observation), 1)
                .to(observation.device)
            )
        )

        return value, reward, policy, hidden_state

    def recurrent_inference(self, hidden_state, action):
        next_hidden_state, reward = self.dynamics(hidden_state, action)
        policy, value = self.prediction(next_hidden_state)
        return value, reward, policy, hidden_state

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

    # create layers for multilayer perceptron
    for i in range (len(sizes) - 1):
        # set activation functions for each layer
        activation_funct = None
        if (i < len(sizes) - 2):
            activation_funct = activation
        else:
            activation_funct = output_activation
        # add linear transform layer with args in_features and out_featrures, and corresponding activation function 
        layers += [torch.nn.Linear(sizes[i], sizes[i+1]), activation_funct()]
    # build and return multilayer perceptron NN
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
