import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_shape, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_shape, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q


class CNN_MLP(nn.Module):
    """A drop-in replacement of the `MLP` class"""
    def __init__(self, input_shape, hidden_dim, output_dim):
        super(CNN_MLP, self).__init__()
        from modules.critics.cnn import CNN
        self.cnn = CNN(input_shape, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs):
        x = F.relu(self.cnn(inputs))
        x = self.fc(x)
        return x