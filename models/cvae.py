import torch
from models import nns
from utils import prob_utils as ut
from torch import nn
from torch.nn import functional as F
import torch.distributions.multivariate_normal as mn
import torch.distributions as td
from models.nns import v1 as net


class CVAE(nn.Module):

    def __init__(self, name='vae', z_dim=2, beta=1):
        super().__init__()

        self.name = name
        self.z_dim = z_dim

        # nn refers to specific architecture file found in models/nns/*.py
        #nn = getattr(nns, nn) Doesnt work for some reason

        self.enc = net.Encoder(self.z_dim)
        self.dec = net.Decoder(self.z_dim)
        self.beta = beta

        # Set prior as fixed parameter
        # Todo: incorporate relaxed one-hot categorical distribution
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

    def negative_elbo_bound(self, y, x):
        """
        Computes Evidence Lower Bound, KL and Reconstruction loss

        :param x: tensor (batch, dim) observations
        :return: nelbo: tensor
                 kl: tensor
                 rec: tensor
        """

        m = self.z_prior_m
        v = self.z_prior_v
        prior = td.Normal(m, v)

        qm, qv = self.enc.encode(y, x)

        posterior = td.Normal(qm, qv)
        z = posterior.rsample()
        means = self.dec.decode(z, x)
        N, M = means.shape
        p_y = mn.MultivariateNormal(means, torch.eye(M))
        rec = torch.mean(- p_y.log_prob(y))
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

    def sample_z(self, batch):
        return ut.sample_gaussian(
            self.z_prior[0].expand(batch, self.z_dim),
            self.z_prior[1].expand(batch, self.z_dim)
        )

    def sample_y_given(self, x, z):
        means = self.dec.decode(z, x)
        N, M = means.shape
        m = mn.MultivariateNormal(means, torch.eye(M))
        return m.sample()

