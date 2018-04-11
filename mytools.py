
from keras.datasets import mnist  # Keras is ONLY used for the dataset
import numpy as np  # calculation is NumPy-only

class DataSource:  # one class for train and test
    def __init__(self, x, y):
        self.size = len(x)
        self.x = np.array((x / 255.).reshape(self.size, 28 * 28), dtype=np.float32)
        self.y = np.zeros((self.size, 10), dtype=np.float32)  # convert index to pattern. examp.: 3 -> [0,0,0,1,..]
        self.y[range(self.size), y] = 1.

    def get_x(self, i):
        return self.x[i % self.size]

    def get_y(self, i):
        return self.y[i % self.size]


class Measure:  # this is just a class to measure True/False restus and finaly get a precentage
    def __init__(self, name, floating=True):
        self.name = name
        self.right = 0.
        self.counter = 0.
        self.counter_i = 0
        self.floating = floating

    def reset(self):
        self.right = 0.
        self.counter = 0.
        self.counter_i = 0

    def compare(self, seen, wanted):
        right = np.argmax(seen)==np.argmax(wanted)
        self.right += right
        self.counter += 1.
        self.counter_i += 1
        if self.counter > 500 and self.floating:
            self.right *= .95
            self.counter *= .95
        if self.counter_i % 1000 == 0:  # print every 10k
            print(self.name, self.counter_i, self.get_prediction())

    def get_prediction(self):
        return self.right / self.counter

