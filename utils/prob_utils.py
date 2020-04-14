# Reference Stanford CS236 Fall 2019

import numpy as np
import os
import shutil
import sys
import torch
from codebase.models.vae import VAE

def sample_gaussian(m, v):
    """
    Element-wise application reprarametrization trick to sample from Gaussian
    :param m: tensor: (batch, ...) Mean
    :param v: tensor: (batch, ...) Variance
    :return:
    """
    shape =m.shape
    eps = torch.randn(shape)
    z = m + torch.sqrt(v) * eps

    return z


def gaussian_parameters(h, dim=-1):
    """
    R
    :param h:
    :param dim:
    :return:
    """

    m, h = torch.split(h, h.size(dim) //2, dim = dim)
    v = F.softplush(h) + 1e-8

    return m, v