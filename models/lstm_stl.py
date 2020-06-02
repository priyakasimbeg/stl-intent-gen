import numpy as np
import torch
from torch import nn
import torch.distributions as td

from models.nns import v5 as net

import stlcg


class CVAE(nn.Module):

    def __init__(self, x_dim, y_dim,
                 z_dim=1,
                 hidden_dim=10,
                 pred_len=20,
                 name='vae',
                 version='v5',
                 beta=100,
                 beta_r=10,
                 c_dim=1,
                 stl=True,
                 robustness_type='curvature'):
        super().__init__()

        self.name = name
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.hidden_dim = hidden_dim
        self.pred_len = pred_len

        # nn refers to specific architecture file found in models/nns/*.py
        #nn = getattr(nns, nn) Doesnt work for some reason
        self.enc = net.Encoder(self.z_dim, self.x_dim, self.y_dim, self.hidden_dim)
        self.dec = net.Decoder(self.z_dim, self.x_dim, self.y_dim, self.hidden_dim, c_dim)
        self.beta = beta

        if stl:
            self.beta_r = beta_r
        else:
            self.beta_r = 0

        self.version = version

        self.robustness_type = robustness_type


    def negative_elbo_bound(self, y, x, k):
        """
        Computes Evidence Lower Bound, KL and Reconstruction loss

        :param x: tensor (batch, dim) observations
        :return: nelbo: tensor
                 kl: tensor
                 rec: tensor
        """
        B, H, _ = np.shape(x)
        _, P, _ = np.shape(y)

        # y = np.reshape(y, (B, -1))
        # x = np.reshape(x, (B, -1))

        q_logits, x_state = self.enc.q_encode(y, x)
        p_logits, x_state = self.enc.p_encode(x)
        # cite Gumbel-Softmax (Jang et al., 2016) and Concrete (Maddison et al., 2016) if using Relaxed version
        # Todo: implement annealing temperature from 2.0
        backpropable_posterior = td.RelaxedOneHotCategorical(1.0, logits=q_logits)
        z = backpropable_posterior.rsample()

        posterior = td.OneHotCategorical(logits=q_logits)
        prior = td.OneHotCategorical(logits=p_logits)

        # Todo: predict u means instead and sigma coefficients
        y_pred_means = self.dec.decode(k, z, x, self.pred_len, train=True, y=y)

        n, m, _ = y_pred_means.shape

        y_pred = torch.zeros((B, self.pred_len, self.x_dim))
        rec = 0

        #Todo: incorporate covariance matrix
        for i in range(self.pred_len):
            y_pred_mean = y_pred_means[:, i, :]
            p_y = td.MultivariateNormal(y_pred_mean, torch.eye(self.x_dim) * 0.001)
            y_pred[:, i, :] = p_y.sample()
            rec = rec + torch.mean(- p_y.log_prob(y[:, i, :]))

        kl = torch.mean(td.kl.kl_divergence(posterior, prior))
        nelbo = rec + self.beta * kl

        # curvature
        robustness = self.robustness_loss(y_pred, k, type=self.robustness_type)

        return nelbo, kl, rec, robustness

    def loss(self, y, x, k):
        """
        Compute loss
        :param y: tensor (batch_size, future_length)  Future trajectory
        :param x: tensor (batch_size, history_length) History
        :return:
        """
        nelbo, kl, rec, robustness = self.negative_elbo_bound(y, x, k)
        loss = nelbo + self.beta_r * robustness
        #loss = nelbo

        summaries = dict((
            ('train/loss', nelbo),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl),
            ('gen/rec', rec),
            ('gen/robustness', robustness)
        ))

        return loss, summaries

    def sample_z(self, x):
        B, H, _ = np.shape(x)
        p_logits, x_state = self.enc.p_encode(x)
        prior = td.OneHotCategorical(logits=p_logits)

        return prior.sample()

    def sample_y_given(self, x, z, c):
        B, H, _ = np.shape(x)
        y_pred_means = self.dec.decode(c, z, x, self.pred_len)

        p_y_pred = np.zeros((B, self.pred_len, self.x_dim))

        for i in range(self.pred_len):
            p_y = td.MultivariateNormal(y_pred_means[:, i, :], torch.eye(self.x_dim) * 0.001)
            p_y_pred[:, i, :] = p_y.sample()

        return p_y_pred

    def sample_y(self, x, c):
        z = self.sample_z(x)
        y_pred = self.sample_y_given(x, z, c)

        return y_pred

    def curvature(self, y_pred):
        # estimate curvature
        y = torch.as_tensor(y_pred).float()
        y_next = y[:, 1:, :]
        y_prev = y[:, :-1, :]

        v = y_next - y_prev
        v_next = v[:, 1:, :]
        v_prev = v[:, :-1, :]
        a = v_next - v_prev

        v_centered = y[:, 2:, :] - y[:, :-2, :]

        dx = v_centered[:, :, 0]
        ddx = a[:, :, 0]
        dy = v_centered[:, :, 1]
        ddy = a[:, :, 1]

        k = (dx * ddy - dy * ddx) / ((dx ** 2 + dy ** 2) ** (3 / 2))

        return k

    def robustness_loss(self, y_pred, k, type='curvature'):
        """
        Calculate robustness loss for curvature
        :param y_pred:
        :param k:
        :return:
        """

        if type == 'curvature':
            robustness = self.robustness_curvature(y_pred, k)

        elif type == 'stop':
            robustness = self.robustness_stop(y_pred, k)

        else:
            raise ValueError('Invalid argument of robustness type')

        return robustness

    def robustness_curvature(self, y_pred, k):

        # Estimate curvature from predictions
        k_pred = self.curvature(y_pred)
        # Set up STL formula
        kf = stlcg.Expression('yf')
        phi_2 = kf > torch.as_tensor(k.unsqueeze(-1) - 0.3).float()
        phi_3 = kf < torch.as_tensor(k.unsqueeze(-1) + 0.3).float()

        phi = phi_2 & phi_3
        psi = stlcg.Always(subformula=phi, interval=[0, 8])

        # Calculate robustness
        robustness = torch.relu(-psi.robustness((k_pred.unsqueeze(-1), k_pred.unsqueeze(-1)), scale=-1)).sum()

        return robustness

    def robustness_stop(self, y_pred, k, eps=0.01):
        """
        Calculate robustness based on stopping condition at (0,0)
        :param y_pred:
        :param k:
        :return:
        """
        _, N, _ = y_pred.shape
        # stopping formula

        x_coord = stlcg.Expression('x', y_pred[:, :, 0].unsqueeze(-1))
        y_coord = stlcg.Expression('y', y_pred[:, :, 1].unsqueeze(-1))

        min = torch.as_tensor([-eps]).float()
        max = torch.as_tensor([eps]).float()

        # psi = car is in box
        phi1 = (x_coord > min) & (x_coord < max)
        phi2 = (y_coord > min) & (y_coord < max)
        phi_A = (phi1 & phi2)

        psi1 = stlcg.Always(subformula=phi_A, interval=[0, 20])

        robustness = (k.unsqueeze(-1) * torch.relu(-psi1.robustness(((x_coord, x_coord), (y_coord, y_coord)), scale=-1)))
        robustness = robustness.sum()
        return robustness



