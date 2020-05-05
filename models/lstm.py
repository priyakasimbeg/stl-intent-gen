import numpy as np

import torch
from torch import nn
import torch.distributions as td

from models.nns import v4 as net


class CVAE(nn.Module):

    def __init__(self, x_dim, y_dim, z_dim=1, hidden_dim=10, pred_len=5,
                 name='vae', version='v4', beta=10000):
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
        self.dec = net.Decoder(self.z_dim, self.x_dim, self.y_dim, self.hidden_dim)
        self.beta = beta

        self.version = version

        # Set prior attributes
        self.set_priors()

    def negative_elbo_bound(self, y, x):
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
        y_pred_means = self.dec.decode(z, x, self.pred_len, train=True, y=y)

        n, m, _ = y_pred_means.shape

        p_y_pred = []
        rec = 0

        #Todo: incorporate covariance matrix
        for i in range(self.pred_len):
            p_y = td.MultivariateNormal(y_pred_means[:, i, :], torch.eye(self.x_dim))
            p_y_pred.append(p_y)
            #
            rec = rec + torch.mean(- p_y.log_prob(y[:, i, :]))

        kl = torch.mean(td.kl.kl_divergence(posterior, prior))

        nelbo = rec + self.beta * kl

        return nelbo, kl, rec

    def loss(self, y, x):
        """
        Compute loss
        :param y: tensor (batch_size, future_length)  Future trajectory
        :param x: tensor (batch_size, history_length) History
        :return:
        """
        nelbo, kl, rec = self.negative_elbo_bound(y, x)
        loss = nelbo

        summaries = dict((
            ('train/loss', nelbo),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl),
            ('gen/rec', rec)
        ))

        return loss, summaries

    def sample_z(self, x):
        B, H, _ = np.shape(x)
        p_logits, x_state = self.enc.p_encode(x)
        prior = td.OneHotCategorical(logits=p_logits)

        return prior.sample()

    def sample_y_given(self, x, z):
        B, H, _ = np.shape(x)
        y_pred_means = self.dec.decode(z, x, self.pred_len)

        p_y_pred = np.zeros((B, self.pred_len, self.x_dim))

        for i in range(self.pred_len):
            p_y = td.MultivariateNormal(y_pred_means[:, i, :], torch.eye(self.x_dim))
            p_y_pred[:, i, :] = p_y.sample()

        return p_y_pred

    def sample_y(self, x):
        z = self.sample_z(x)
        y_pred = self.sample_y_given(x, z)

        return y_pred

    def set_priors(self):
        if self.version == 'v1':
            self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
            self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
            self.z_prior = (self.z_prior_m, self.z_prior_v)