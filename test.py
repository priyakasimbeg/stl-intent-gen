import sys
sys.path.append('..')
import os
import torch
from models.cvae import CVAE
import numpy as np
from utils import test_utils as tut
from utils import prob_utils as put
import matplotlib.pyplot as plt


# Load model
z = 3
run = 1

layout = [
    ('model={:s}', 'cvae'),
    ('z={:02d}', z),
    ('run={:04d}', run)
]

model_name = '_'.join([t.format(v) for (t, v) in layout])

cvae = CVAE(z_dim=z, name=os.path.join(model_name))
tut.load_model_by_name(cvae, global_step=20000)

# Load data
train_loader, valid_loader = put.get_data_loaders()

for i, (x, y) in enumerate(valid_loader):
    batch = len(x)
    z = cvae.sample_z(x)
    y_pred = cvae.sample_y_given(x, z)
    break