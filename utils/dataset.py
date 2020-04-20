import torch
from torch.utils import data
import numpy as np
import os

from utils import data_generator as dg

DATA_FOLDER = 'data'


class Dataset(data.Dataset): # Todo: pytorch batch

    def __init__(self, path=os.path.join(DATA_FOLDER,'exp.npy')):
        self.tracks = np.load(path)
        self.ids = range(0, len(self.tracks))

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, index):
        X = self.tracks[index].astype('float')
        x = np.expand_dims(X[: dg.HISTORY_SIZE].astype('float'), 1)
        y = np.expand_dims(X[dg.HISTORY_SIZE :].astype('float'), 1)

        return x, y