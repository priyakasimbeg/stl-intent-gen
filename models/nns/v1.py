# Reference: from CS236 course material Fall 2018

import torch
from torch import autograd, nn, optim
from utils import prob_utils as ut

from utils import data_generator as dg

class Encoder(nn.Module):
    def __init__(self, z_dim, x_dim=dg.HISTORY_SIZE, y_dim=dg.PREDICTION_SIZE):
        super().__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.model = nn.Sequential(
            nn.Linear(self.y_dim + self.x_dim, 100),
            nn.ELU(),
            nn.Linear(100, 100),
            nn.ELU(),
            nn.Linear(100, 2 * z_dim)
        )

    def encode(self, y, x):
        xy = torch.cat((x, y), dim=1)
        print('Type of xy')
        print(type(xy.data))
        h = self.model(xy)
        m, v = ut.gaussian_parameters(h, dim=1)
        return m, v


class Decoder(nn.Module):
    def __init__(self, z_dim, y_dim=dg.PREDICTION_SIZE):
        super().__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim

        self.model = nn.Sequential(
            nn.Linear(z_dim, 100),
            nn.ELU(),
            nn.Linear(100, 100),
            nn.ELU(),
            nn.Linear(100, y_dim)
        )

    def decode(self, z):
        return self.model(z)

