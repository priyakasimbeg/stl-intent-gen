import torch
from torch.utils import data
import numpy as np
import os

import utils.data_generator as dg

DATA_FOLDER = 'data'
HISTORY_SIZE = 20

class Dataset(data.Dataset): # Todo: pytorch batch

    def __init__(self, path=os.path.join(DATA_FOLDER, 'exp.npy'),
                       history_size = HISTORY_SIZE):
        self.tracks = np.load(path)
        self.ids = range(0, len(self.tracks))
        self.history_size = history_size
        self.trajectory_size = len(self.tracks[0])
        self.prediction_size = self.trajectory_size - self.history_size

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, index):
        X = self.tracks[index].astype('float32')
        x = X[: self.history_size].astype('float32')
        y = X[self.history_size:].astype('float32')

        return x, y
