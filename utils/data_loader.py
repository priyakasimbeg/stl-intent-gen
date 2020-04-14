import torch
from torch.utils import data
import numpy as np
import os

DATA_FOLDER = 'data'

class Dataset(data.Dataset):

    def __init__(self, path=os.path.join(DATA_FOLDER,'exp.npy')):
        self.tracks = np.load(path)
        self.ids = range(0, len(self.tracks))

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, index):
        X = self.tracks[index]

        return X