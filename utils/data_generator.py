import numpy as np
import os

MODES = ['functional', 'step']
DATA_FOLDER = 'data'
HISTORY_SIZE = 25
TRAJECTORY_SIZE = 50
SIZE = 50
PREDICTION_SIZE = TRAJECTORY_SIZE - HISTORY_SIZE
END = 5
DT = 0.1

class DataGenerator:

    def __init__(self):
        self.mode = MODES[0]
        self.size = SIZE
        self.end = 5
        self.dx = 0.1
        self.history_size = HISTORY_SIZE


    def generateExp(self, path=os.path.join(DATA_FOLDER,'exp')):

        tracks = []


        for i in range(self.size):
            alpha = np.random.random()

            if (np.random.random() < 0.5):
                sign = 1
            else:
                sign = -1

            x = np.arange(0, self.end, self.dx)
            y = sign * np.exp(alpha * x)

            tracks.append(y)

        np.save(path, tracks)


        # Circles












