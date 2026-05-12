import numpy as np

def relu(x):
    return np.maximum(0, x)

def d_relu(x):
    return (x > 0).astype(float)

def sMax(x):
    x = np.array(x)
    e = np.exp(x-np.max(x))
    return e/e.sum()

class Layer:
    def __init__(self, input_shape, output_shape):
        self.weights = np.random.uniform(-0.5, 0.5, (output_shape, input_shape))
        self.biases = np.random.uniform(-0.5, 0.5, output_shape)

        self.input = None
        self.z = None
    
    def forward(self, input, last=False):
        self.input = input
        self.z = self.weights @ self.input + self.biases
        if not last:
            return relu(self.z)
        return sMax(self.z)

class Neural_net:
    def __init__(self, shape):
        self.shape = shape
        self.layers = []
        for i in range(1, len(self.shape)):
            self.layers.append(Layer(self.shape[i-1], self.shape[i]))

    def forward(self, input):
        for l in range(0, len(self.layers)-1):
            input = self.layers[l].forward(input)
        input = self.layers[-1].forward(input, True)
        return input

NN = Neural_net(np.array([2,3,3,2]))
print(NN.forward(np.array([2,3])))
