# Reference Stanford CS236 Fall 2019
import torch
from torch.nn import functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import tensorflow as tf

from utils import dataset as ds

import numpy as np
import os
import shutil

BATCH_SIZE = 16
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

## Model Utils
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
    v = F.softplus(h) + 1e-8

    return m, v


## Train Utils
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


def get_data_loaders(shuffle_dataset=True,
                     batch_size=16, validation_split=0.2):

    dataset = ds.Dataset()

    # Generate indices for training and validation
    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    split = int(np.floor(validation_split * dataset_size))

    if shuffle_dataset:
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]


    # Data samplers and loaders
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    validation_loader = DataLoader(dataset, batch_size=1, sampler=valid_sampler)

    return train_loader, validation_loader


def prepare_writer(model_name, overwrite_existing=False):
    log_dir = os.path.join('logs', model_name)
    save_dir = os.path.join('checkpoints', model_name)
    if overwrite_existing:
        delete_existing(log_dir)
        delete_existing(save_dir)

    writer = tf.summary.create_file_writer(log_dir)
    return writer


def log_summaries(writer, summaries, global_step):
    # with writer.as_default():
    #     for tag in summaries:
    #         val = summaries[tag]
    #         tf.summary.scalar(tag, val, step=global_step)
    #     #writer.add_summary(tf.summary(value=[tf_summary]), global_step)
    # writer.flush()
    pass


def delete_existing(path):
    if os.path.exists(path):
        print("Deleting existing path: {}".format(path))
        shutil.rmtree(path)
