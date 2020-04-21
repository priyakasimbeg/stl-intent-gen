import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
import torch.distributions as td

from models.nns import v2 as net
from utils import prob_utils as ut


class CVAE(nn.Module):

    def __init__(self, name='vae', version='v2', z_dim=1, beta=1):
        super().__init__()

        self.name = name
        self.z_dim = z_dim

        # nn refers to specific architecture file found in models/nns/*.py
        #nn = getattr(nns, nn) Doesnt work for some reason

        self.enc = net.Encoder(self.z_dim)
        self.dec = net.Decoder(self.z_dim)
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

        if self.version == 'v1':
            qm, qv = self.enc.encode(y, x)
            posterior = td.normal(qm, qv)
            z = posterior.rsample()
            means = self.dec.decode(z, x)
            print('here')

        if self.version == 'v2':
            logits = self.enc.encode(y, x)
            # cite Gumbel-Softmax (Jang et al., 2016) and Concrete (Maddison et al., 2016) if using Relaxed version
            # Todo: Also estimate temperature?
            # Todo: KL not implemented for RelaxedOneHotCategorical?
            posterior = td.OneHotCategorical(logits=logits)
            z = posterior.sample()
            means = self.dec.decode(z, x)

        n, m = means.shape
        p_y = td.MultivariateNormal(means, torch.eye(m))


        rec = torch.mean(- p_y.log_prob(y))
        kl = torch.mean(td.kl.kl_divergence(posterior, self.prior))

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

    def sample_z(self, batch_size):

        return self.prior.sample(torch.tensor([batch_size]))

    def sample_y_given(self, x, z):
        params = self.dec.decode(z, x)
        N, M = params.shape

        if self.version == 'v1':
            dist = td.MultivariateNormal(params, torch.eye(M))

        elif self.version == 'v2':
            dist = td.OneHotCategorical(logits=params)

        else:
            raise ValueError('Version should be v1 or v2')

        return dist.sample()

    def set_priors(self):
        if self.version == 'v1':
            self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
            self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
            self.z_prior = (self.z_prior_m, self.z_prior_v)
            self.prior = td.Normal(self.z_prior_m, self.z_prior_v)

        if self.version == 'v2':
            self.z_prior_logprobs = torch.nn.Parameter(torch.ones(self.z_dim), requires_grad=False)
            self.prior = td.OneHotCategorical(self.z_prior_logprobs)


