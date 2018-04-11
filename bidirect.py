
from keras.datasets import mnist  # Keras is ONLY used for the dataset
import numpy as np  # calculation is NumPy-only
from mytools import DataSource, Measure

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
        self.full[:, best] *= lr *q* self.is_in -lr*2.+ 0.986  # *= acts like AND, -2 to neutralize +lr
        self.full[:, best] += lr *(2.-q)* self.is_in   # += acts like OR


class EntanglementLayer:
    def __init__(self, num, size, load_a, load_b):  # size = amount of twin-neurons
        self.a = Layer(num, size, load=load_a)  # picture
        self.b = Layer(num, size, load=load_b)  # number

    def run(self, count, learn=True):
        for i in range(count):
            self.a.load(i)
            self.b.load(i)
            self.b.store(self.a.activation, measure=True)

            best = np.argmax(self.a.activation * 1.75 + self.b.activation)  # final learning decision based on both sides
            if learn:
                self.a.learn(i, best, 1.2)
                self.b.learn(i, best)

    def info(self):
        self.b.info()  # only measure output


(x_train, y_train), (x_test, y_test) = mnist.load_data()
train_data = DataSource(x_train, y_train)  # 60k
test_data = DataSource(x_test, y_test)  # 10k

layer = EntanglementLayer(0, 784, train_data.get_x, train_data.get_y)

print("Train...")
layer.run(train_data.size * 3)  # 3 epochs

print("Run Actual Test Data")
layer.a.load_src = test_data.get_x  # switch from train to test..
layer.b.load_src = test_data.get_y
layer.b.measure.reset()
layer.b.measure.floating = False
layer.run(test_data.size, learn=False)
layer.info()
