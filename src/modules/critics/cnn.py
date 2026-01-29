import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def _build_layers(input_shape, hidden_dim):
    h, w, c = input_shape
    n_layers = max(int(np.log2(min(h, w))) - 2, 0)
    layers = []
    for _ in range(n_layers):
        layers.extend([
            nn.Conv2d(c, c, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(c, c, 2, 2),
            nn.ReLU(),
        ])
    layers.append(nn.Flatten())
    cnn_layers = nn.Sequential(*layers)

    dummy_input = torch.randn(*input_shape).permute(2, 0, 1)
    dummy_output = cnn_layers(dummy_input)

    output_dim = np.prod(dummy_output.shape)
    return cnn_layers, nn.Linear(output_dim, hidden_dim)

class CNN(nn.Module):
    def __init__(self, input_shape, hidden_dim):
        """Build a CNN that take `input_dim` and output (flatten) `hidden_dim`"""
        super(CNN, self).__init__()
        assert len(input_shape) == 3, "`input_shape` should be of shape (h, w, c)"
        self.cnn, self.fc = _build_layers(input_shape, hidden_dim)

    def forward(self, inputs):
        assert len(inputs.shape) > 3, inputs.shape
        input_shape = inputs.shape
        input_dim = np.arange(len(inputs.shape))

        h, w, c = input_dim[-3:]
        input_dim[-3:] = c, h, w

        inputs = inputs.permute(*input_dim)
        inputs = inputs.reshape(-1, *inputs.shape[-3:])

        x = self.cnn(inputs)
        x = x.reshape(*input_shape[:-3], -1)
        x = self.fc(x)
        return x
