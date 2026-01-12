import numpy as np

def relu(x):
    return np.maximum(0, x)
def sMax(x):
    x = np.array(x)
    e = np.exp(x-np.max(x))
    return e/e.sum()
def crossEntropy(pred, true):
    return -np.log(pred[true]+1e-15)
def squaredError(pred, true):
    return np.sum((pred - true)**2)

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
    
    def backward(self, input, true):
        # Placeholder for backward pass
        pass