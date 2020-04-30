import torch
from torch import nn

from utils import dataset as ds

class Encoder(nn.Module):
    def __init__(self, z_dim, x_dim, y_dim):
        super().__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.y_dim = y_dim

        self.xy_model = nn.Sequential(
            nn.Linear(2 * (self.x_dim + self.y_dim), 100),
            nn.ELU(),
            nn.Linear(100, 100),
            nn.ELU(),
            nn.Linear(100, 30),
            nn.ELU(),
            nn.Linear(30, z_dim)
        )

        self.x_model = nn.Sequential(
            nn.Linear(2 * self.x_dim, 100),
            nn.ELU(),
            nn.Linear(100, 100),
            nn.ELU(),
            nn.Linear(100, 30),
            nn.ELU(),
            nn.Linear(30, z_dim)
        )

        self.xy_model = torch.nn.LSTM(self.x_dim + self.y)

    def q_encode(self, y, x):
        xy = torch.cat((x, y), dim=1)
        encoded_xy = self.xy_model(xy)
        return encoded_xy

    def p_encode(self, x):
        encoded_x = self.x_model(x)
        return encoded_x


class Decoder(nn.Module):
    def __init__(self, z_dim, x_dim, y_dim):
        super().__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.x_dim = x_dim

        self.model = nn.Sequential(
            nn.Linear(z_dim + 2 * x_dim, 100),
            nn.ELU(),
            nn.Linear(100, 100),
            nn.ELU(),
            nn.Linear(100, 100),
            nn.ELU(),
            nn.Linear(100, 2 * y_dim)
        )

    def decode(self, z, x):
        xz = torch.cat((x, z), dim=1)
        return self.model(xz)

