import torch
from torch import nn

from utils import dataset as ds

class Encoder(nn.Module):
    def __init__(self, z_dim, x_dim, y_dim, hidden_dim):
        super().__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.hidden_dim = hidden_dim

        self.x_rnn = nn.LSTM(self.x_dim, self.hidden_dim, batch_first=True)
        self.y_rnn = nn.LSTM(self.y_dim, self.hidden_dim, batch_first=True)
        self.xz_proj_network = nn.Linear(self.hidden_dim, self.z_dim)
        self.xyz_proj_network = nn.Linear(self.hidden_dim * 2, self.z_dim)

    # Todo return encoded instead to pass into
    def q_encode(self, y, x):
        encoded_x, x_state = self.x_rnn(x)
        encoded_y, y_state = self.y_rnn(y)

        #
        xy = torch.cat((encoded_x[:,-1,:], encoded_y[:,-1,:]), dim=1)
        z_vec = self.xyz_proj_network(xy)
        return z_vec, x_state #Todo encorporate y_state

    def p_encode(self, x):
        encoded_x, x_state = self.x_rnn(x)
        z_vec = self.xz_proj_network(encoded_x[:,-1,:])
        return z_vec, x_state


class Decoder(nn.Module):
    def __init__(self, z_dim, x_dim, y_dim, hidden_dim):
        super().__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.x_dim = x_dim
        self.hidden_dim = hidden_dim

        # Todo: LSTMCell instead
        self.rnn_network = torch.nn.LSTM(self.x_dim, self.hidden_dim, batch_first=True)
        self.proj_network = torch.nn.Linear(self.z_dim, self.hidden_dim)
        self.projout_network = torch.nn.Linear(self.hidden_dim, self.y_dim)

    def decode(self, z, x, y_pred_len, train=False, y=None):
        y_out = []
        rnn_state = self.proj_network(z) # Todo: how to get state?
        rnn_state = torch.unsqueeze(rnn_state, 0)
        rnn_state = (rnn_state, rnn_state) # Todo:help
        y_prev = x[:, -1:, :]

        if train:
            for t in range(y_pred_len):
                out, rnn_state = self.rnn_network(y_prev, rnn_state)
                u = self.projout_network(out)  # predict velocity
                y_current = y_prev + u  # integrate
                y_out.append(y_current)
                y_prev = y[:, t:t+1, :]
        else:
            for t in range(y_pred_len):
                out, rnn_state = self.rnn_network(y_prev, rnn_state)
                u = self.projout_network(out)
                y_current = y_prev + u
                y_out.append(y_current)
                y_prev = y_current

        return torch.cat(y_out, dim=1)