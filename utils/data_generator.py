import numpy as np
import os

MODES = ['functional', 'step']
DATA_FOLDER = '../data'
SIZE = 600
END = 5
DT = 0.1

class DataGenerator:

    def __init__(self):
        self.mode = MODES[0]
        self.size = SIZE
        self.end = 5
        self.dt = 0.1

    def generate(self, path):
        pass


class DataGenerator1D(DataGenerator):

    def __init__(self):
        super().__init__()

    def generate(self, path=os.path.join(DATA_FOLDER, 'exp')):

        tracks = []


        for i in range(self.size):
            alpha = np.random.random()

            if (np.random.random() < 0.5):
                sign = 1
            else:
                sign = -1

            x = np.arange(0, self.end, self.dt)
            y = sign * np.exp(alpha * x)

            tracks.append(y)

        np.save(path, tracks)


class DataGenerator2D(DataGenerator):
    def __init__(self):
        super().__init__()
        self.y_int = 0
        self.v = np.array([0.0, 0.8])
        self.pos = np.array([0, -1])
        self.directions = ['left', 'right', 'straight']
        self.t_steps = np.arange(0, self.end, 0.1)
        self.max_steps = len(self.t_steps)


    def generate(self, path=os.path.join(DATA_FOLDER, 'fork')):

        tracks = []

        partition_size = self.size // len(self.directions)

        # Left
        for direction in self.directions:
            for i in range(partition_size):
                track = self.generate_track(direction)
                tracks.append(track)

        np.save(path, tracks)

    def generate_track(self, direction):
        pos = self.pos

        # Randomness
        v = self.v + np.random.random() * 0.2

        # Diretion settings
        if direction == 'right':
            r = 1.0 + np.random.random() * 0.2
            v_end = np.array([1.0, 0.])
            a_v = - np.linalg.norm(v) / r  # sign
            phi_0 = np.pi

        if direction == 'left':
            r = 1.5 + np.random.random() * 0.2
            v_end = np.array([-1.0, 0.])
            a_v = np.linalg.norm(v) / r  # sign
            phi_0 = 0

        # Start generating
        num_steps = 0

        path = [pos]

        while pos[1] < self.y_int:
            pos = pos + self.dt * v
            path.append(pos)
            num_steps += 1

        if direction == 'right' or direction == 'left':
            t = 0
            while abs(v[1] - v_end[1]) > 0.1:
                # for i in range(50):
                vx = - a_v * r * np.sin(phi_0 + a_v * t)
                vy = a_v * r * np.cos(phi_0 + a_v * t)
                v = np.array([vx, vy])
                pos = pos + v * self.dt

                t = t + self.dt
                path.append(pos)
                num_steps += 1

        while num_steps < self.max_steps:
            pos = pos + v * self.dt
            path.append(pos)
            num_steps += 1

        path = np.array(path)

        return path

    # Helper functions for











