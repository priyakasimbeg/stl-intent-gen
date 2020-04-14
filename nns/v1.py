# Reference: from CS236 course material Fall 2018

import torch
from torch import autograd, nn, optimi
from utils import prob_utils as ut


class Encoder(nn.Module):
    def __init__(self, z_dim, x_dim=50):
        super().__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.model = nn.Sequential(
            nn.Linear(self.y_dim + self.x_dim, 100),
            nn.ELU(),
            nn.linear(100, 100),
            nn.ELU(),
            nn.Linear(100, 2 * z_dim)
        )

    def encode(self, y, x):
        xy = torch.cat((x, y), dim=1)
        h = self.model(xy)
        m, v = ut.gaussian_parameters(h, dim=1)
        return m, v


class Decoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim

        self.model = nn.Sequential(
            nn.Linear(z_dim, 100),
            nn.ELU(),
            nn.Linear(100, 100),
            nn.ELU(),
            nn.Linear(100, 50)
        )

    def decode(self, z):
        return self.model(z)

