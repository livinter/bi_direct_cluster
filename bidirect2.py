"""
I tried to get a very simple solution to MNIST. Numpy-Only


Implementation-Text:
    - Neurons have activation from 0..1
    - Weights are between 0..1, all start at .9
    - Activations are calculated based on input AND expectation
    - Learning is only applied to the most active(s) neuron()
      - input AND output- weights are modified to match better in the future.
      - the learning is simply done by multiplying and adding activation to the weight.
   
Implementation-Code:

The main lines to get to 96% are:

    0) init all weights to 0.9  

    1) activation with dot() # from forward and backward

    2) # get sparse relu activation
    activation -= np.max(activation) * .95
    return np.maximum(activation / np.max(activation), 0.)

    3) select only best maching neuron from: 1.75*input + 1*expectation
    best = np.argmax(layer1.out * 1.75 + layer2.in_)

    4) # learn only on neuron from 3), applyed symetricaly
    weights *= learn_rate * activation - learn_rate * 2. + 0.99
    weights += learn_rate * activation
    


Intuition 1: The first layer in CNN could be extracted from input only
When you watch the first layer in a picture-watching-neuronal-network:
https://www.researchgate.net/profile/Okan_Kopuklu/publication/331134795/figure/fig2/AS:726640602673156@1550256028863/Learned-convolutional-kernels-in-the-first-convolutional-layer-of-AlexNet-Some-of-the.jpg
you realize that no backpropagation is needed. Instead, the pattern is just the most common pattern
in the seen pictures. Sure, the more you go further you got in the layer, the more patterns are related to the expected output.


Intuition 2: "seeing" based on the expectation
When a baby tuches and rotate a toy-brick, several senses make the brick experience.
All the pictures and the senses get connected. 
A higher layer can just stay with its conclusion "brick", and while new frames come in,
they can get connected to the same high-level conclusion, as we can expect that there is still a brick.
This kind of learning is in strong contrast to a batch of different objects.
Instead, we look at one object and get it in all its variations. It is not one shoot,
its several shoots from one thing.


Deep learning using backpropagation has a strict separation from the input flow on one side,
and desired output on the other side where each neuron activation is only defined by the 
the sum of its inputs and each neuron "desired activation" is defined only by the outputs to the following layers.

When we process videos we throw away all activations each frame. I suspect that those activations could be useful. And that it could make sense to use them in the calculation of the activation of the neurons.


Takeaways:
 - The experiment showed best results taking when calculating connections based on
     `input * 1.75 +output`
      Better than just using output, and better than just using input.   
 - Normally all weights get updated. In my code, I only modify weights from one or a few neurons. Finding a way to do this in DL could help to speed up.
 
"""

from keras.datasets.mnist import load_data
#from keras.datasets.fashion_mnist import load_data
import numpy as np
(x_train, y_train), (x_test, y_test) = load_data()

EPOCHS=8     # change to 3 for faster run
HIDDEN=1500  # change to 750 for faster run

# this is just a class to measure True/False restus and finaly get a precentage
# not needed for understanding
class Measure:      
    def __init__(self, name, floating=True):
        self.name = name
        self.right = 0.
        self.counter = 0.
        self.counter_i = 0
        self.floating = floating
        self.seen = np.zeros((10, ), )

    def reset(self):
        self.right = 0.
        self.counter = 0.
        self.counter_i = 0

    def compare(self, seen, wanted):
        right = np.argmax(seen) == np.argmax(wanted)
        self.seen[np.argmax(seen)] += 1

        self.right += right
        self.counter += 1.
        self.counter_i += 1
        if self.counter > 500 and self.floating:
            self.right *= .97
            self.counter *= .97
        if self.counter_i % 10000 == 0:  # print every 10k
            print(self.name, self.counter_i, self.get_prediction())
        return right

    def get_prediction(self):
        return self.right / self.counter


# convert x,y to 0..1
# not needed for understanding
def convert(x, y):
    size = len(x)
    x = np.array((x / 255.).reshape(size, 28 * 28), dtype=np.float32)
    y1 = np.zeros((size, 10), dtype=np.float32)  # convert index to pattern. examp.: 3 -> [0,0,0,1,..]
    y1[range(size), y] = 1.
    return x, y1, size


class Layer:
    def __init__(self, n_src, n_dest):
        self.full = np.ones((n_src, n_dest), dtype=np.float32) * .9

    @staticmethod
    def GReLu(activation):
        # Global ReLU, only focus on strongest 5% in Layer
        activation -= np.max(activation) * .95
        return np.maximum(activation / np.max(activation), 0.)

    def forward(self, in_):
        self.in_ = in_
        self.out = self.GReLu(np.dot(in_, self.full))

    def backward(self, out):
        self.out = out
        self.in_ = self.GReLu(np.dot(out, self.full.T))

    def learn(self, activation, weights, n):
        # slow down learn rate over time
        lr = 0.03 * 8000 / (8000 + n)

        # `0.99` means: this value is <1 and result that an activation will
        # decrease the possibility of future activations.
        # `*= means: if the source neuron is not activated the weight will be reduced
        weights *= lr * activation - lr * 2. + 0.99
        # `+= means: if the source neuron is activated the weight will be increased
        weights += lr * activation


def symetric_learn(l1, l2, i):
    # learn only one neuron that has strongest activation taking in concideration
    # input*1.75 AND output.
    best = np.argmax(l1.out * 1.75 + l2.in_)
    # modify weights to both sides.
    l1.learn(l1.in_, l1.full[:, best], i)
    l2.learn(l2.out, l2.full[best, :], i)


# 
a = Layer(784, HIDDEN)
b = Layer(HIDDEN, 10)

for learning in [True, False]:  # first learn, than test
    if learning == True:
        x, y, size = convert(x_train, y_train)
        measure = Measure("Train:", floating=True)
        lsize = size*EPOCHS+1  # * Epochs

    if learning == False:
        x, y, size = convert(x_test, y_test)
        measure = Measure("Test: ", floating=False)
        lsize = size+1
        
    for i in range(lsize):
        a.forward(x[i % size])
        b.forward(a.out)
        measure.compare(b.out, y[i % size])

        if learning:
            b.backward(y[i % size])
            symetric_learn(a, b, i) 
 
