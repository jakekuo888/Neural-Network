# this test is me manually plugging in w&b from a pre-trained model
# this is me before applying things like back-prop
# I want to see whether or not it works out!

import numpy as np

def relu(x):
    return np.maximum(0, x)
def sMax(x):
    x = np.array(x)
    e = np.exp(x-np.max(x))
    return e/e.sum()

class Layer:
    def __init__(self, in_shape, out_shape):
        self.weights = np.random.uniform(-0.5, 0.5, (out_shape, in_shape))
        self.biases = np.random.uniform(-0.5, 0.5, out_shape)
    
    def forward(self, input):
        return relu(self.weights @ input + self.biases)

class Neural_net:
    def __init__(self, shape):
        self.shape = shape
        self.layers = []
        for i in range(1, len(shape)):
            self.layers.append(Layer(shape[i-1], shape[i]))

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

nn = Neural_net([1,2,3,4])
test_ = np.array([0.1])
print(sMax(nn.forward(test_)))