# Reference Stanford CS236 Fall 2019
import torch
from codebase.models.vae import CVAE
from torch.nn import functional as F

import os

# Model Utils
def sample_gaussian(m, v):
    """
    Element-wise application reprarametrization trick to sample from Gaussian
    :param m: tensor: (batch, ...) Mean
    :param v: tensor: (batch, ...) Variance
    :return:
    """
    shape = m.shape
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

    m, h = torch.split(h, h.size(dim) // 2, dim = dim)
    v = F.softplush(h) + 1e-8

    return m, v

#Todo KL normal
def kl_normal(qm, qv, pm, pv):
    element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1 )
    kl = element_wise.sum(-1)

    return kl

# Train Utils
def reset_weights(m):
    try:
        m.reset_parameters()
    except AttributeError:
        pass

def save_model_by_name(model, global_step):
    save_dir = os.path.join('checkpoints', model.name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_path = os.path.join(save_dir, 'model-{:05d}.pt'.format(global_step))
    state = model.state_dict()
    torch.save(state, file_path)
    print('Saved to {}'.format(file_path))