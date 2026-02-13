import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    """ A simple feedforward neural network. """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        activation: nn.Module = None
    ):
        super().__init__()

        if activation is None:
            activation = nn.ReLU()

        self.linear_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            activation,
            nn.Linear(hidden_size, hidden_size),
            activation,
            nn.Linear(hidden_size, output_size)
        )

        self.initialize_weights()

    def forward(self, x):
        out = self.linear_stack(x)
        return out
   
    def initialize_weights(self):
        for layer in self.linear_stack:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias) 