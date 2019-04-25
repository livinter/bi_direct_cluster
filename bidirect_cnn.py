"""
BI-DIRECTIONAL CLUSTERING
=========================
By adding a  simplified binary CNN layer success goes to 97,01%
to have this running fast enough the kernel-filter is composed
of 5 pixels that are used as bits and translated to a number of 0..31
"""

from keras.datasets import mnist  # Keras is ONLY used for the dataset
import numpy as np  # calculation is NumPy-only

from numba import jit


@jit(nopython=True)
def maxpool4(q, matrix9, maxo):
    step = 2
    q2 = np.zeros((22 // step, 26 // step, maxo), dtype=np.float32)
    for y in range(22 // step):
        idx = y * 28 * step
        for x in range(26 // step):
            for i in matrix9:
                q2[y, x, q[idx + i]] += 1.
            idx += step
    return (q2[:, :, 1:].ravel() ** .02) / (len(matrix9) ** .02)


class ConvNet:
    def __init__(self):
        index_bits = [(0, 0), (2, 0), (1, 1), (0, 2), (2, 2)]

        self.index_len = len(index_bits)
        self.kernel_idx = np.zeros((self.index_len, 4), dtype=np.int32)
        self.kernel_mul = np.zeros((self.index_len, 4), dtype=np.float32)
        for i, (ix, iy) in enumerate(index_bits):
            cnt = 0
            for y in range(3):
                for x in range(3):
                    v = 1. - ((ix - x) ** 2 + (iy - y) ** 2) ** .5 * .5
                    if v > (ix == 1) * .49 and cnt < 4:
                        self.kernel_idx[i, cnt] = y * 28 + x
                        self.kernel_mul[i, cnt] = v
                        cnt += 1
            self.kernel_mul[i] /= sum(self.kernel_mul[i])
        n = 28 * 28 - 28 * 3
        self.conv_index = np.stack([self.kernel_idx] * n) + (np.arange(n)[None][None]).T

        self.arange5 = np.arange(self.index_len)[None]
        self.matrix16 = ((np.arange(4))[None] + (np.arange(4)[None].T * 28)).ravel()

    def process(self, x):
        q1 = (((x[self.conv_index] * self.kernel_mul).sum(axis=-1) > .5) << self.arange5).sum(axis=-1)
        return maxpool4(q1, self.matrix16, 2 ** self.index_len)


class DataSource:  # one class for train and test
    def __init__(self, x, y):
        self.size = len(x)
        self.x = np.array((x / 255.).reshape(self.size, 28 * 28), dtype=np.float32)
        self.y = np.zeros((self.size, 10), dtype=np.float32)  # convert index to pattern. examp.: 3 -> [0,0,0,1,..]
        self.y[range(self.size), y] = 1.
        self.cycle = 0
        self.move = [3, 28 * 3, 2, 28 * 2, 0, 28, 28 * 2 + 3, 30]
        self.conv = ConvNet()

    def get_x(self, i):
        # mv = self.move[self.cycle]
        # self.cycle += 1
        # self.cycle %= len(self.move)
        # return self.x[i % self.size][mv:mv+25*28]
        return self.conv.process(self.x[i % self.size])

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
        right = np.argmax(seen) == np.argmax(wanted)
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


class Layer:
    def __init__(self, id, n_layer, load):
        self.id, self.load_src, self.n_layer = id, load, n_layer
        self.n_dest = load(0).shape[0]
        self.full = np.ones((self.n_dest, self.n_layer), dtype=np.float32) * 0.75  # full_layer init.
        self.measure = Measure(self.id)

    def info(self):
        print(">", self.id)
        print("   SUCC: ", self.measure.get_prediction())
        print("   IN  : ", self.n_dest)

    def load(self, n):
        self.is_in = self.load_src(n)  # load source
        activation = np.dot(self.is_in, self.full)  # data DOT fully_connected_layer
        activation -= np.max(activation) * .95  # focus in top (higher value, faster learning curve)
        self.activation = np.maximum(activation / np.max(activation), 0.)  # normalize & ReLU

    def store(self, pair_data, measure=False):
        self.new_data = np.dot(pair_data, self.full.T)  # send back by full_weights.tranpose
        if measure:
            self.measure.compare(self.is_in, self.new_data)

    def learn(self, n, best, q=1.0):  # update weights just for best matching
        lr = 0.04 * 4000 / (4000 + n)  # learn rate decrease learning rate over time(n)
        # mixing of best-matching-pattern with actual pattern
        # + 0.986 making future activation more difficult to give all neurons a 'chance'
        self.full[:, best] *= lr * q * self.is_in - lr * 2. + 0.986  # *= acts like AND, -2 to neutralize +lr
        self.full[:, best] += lr * (2. - q) * self.is_in  # += acts like OR


class EntanglementLayer:
    def __init__(self, num, size, load_a, load_b):  # size = amount of twin-neurons
        self.a = Layer(num, size, load=load_a)  # picture
        self.b = Layer(num, size, load=load_b)  # number

    def run(self, count, learn=True):
        for i in range(count):
            self.a.load(i)
            self.b.load(i)
            self.b.store(self.a.activation, measure=True)

            best = np.argmax(
                self.a.activation * 1.75 + self.b.activation)  # final learning decision based on both sides
            if learn:
                self.a.learn(i, best, 1.0)
                self.b.learn(i, best)

    def info(self):
        self.b.info()  # only measure output


(x_train, y_train), (x_test, y_test) = mnist.load_data()
train_data = DataSource(x_train, y_train)  # 60k
test_data = DataSource(x_test, y_test)  # 10k

layer = EntanglementLayer(0, 1000, train_data.get_x, train_data.get_y)

print("Train...")
layer.run(train_data.size * 2)  # 3 epochs

print("Run Actual Test Data")
layer.a.load_src = test_data.get_x  # switch from train to test..
layer.b.load_src = test_data.get_y
layer.b.measure.reset()
layer.b.measure.floating = False
layer.run(test_data.size, learn=False)
layer.info()
