#this only works for forward passes; check main_.py for further progress.
import random as rd
import numpy as np
import math

def relu(x):
    return max(0, x)
def sMax(x):
    x = [math.exp(i) for i in x]
    tot = sum(x)
    return [i/tot for i in x]

class Neuron:
    def __init__(self, bias, weights):
        self.bias = bias
        self.weights = weights
        self.output = 0

    def forward(self, inputs):
        s = 0
        for i in range(len(self.weights)):
            s += self.weights[i]*inputs[i]
        self.output = relu(s+self.bias)
        return self.output

class Layer:
    def __init__(self, size, n_weight):
        self.size = size
        self.n_weight = n_weight
        self.neurons = []
        for i in range(self.size):
            n_neuron = Neuron(rd.uniform(-0.5, 0.5), [rd.uniform(-0.5, 0.5) for i in range(self.n_weight)])
            self.neurons.append(n_neuron)

    def forward(self, inputs):
        return [neuron.forward(inputs) for neuron in self.neurons]

class Neural_network:
    def __init__(self, shape):
        self.shape = shape
        self.inputs = shape[0]
        self.layers = []
        for i in range(1, len(self.shape)):
            size = shape[i]
            layer = Layer(size, shape[i-1])
            self.layers.append(layer)
    
    def forward(self, inputs):
        for layer in self.layers:
            print(inputs)
            inputs = layer.forward(inputs)
        print(inputs)
        return inputs
    
nn = Neural_network([3,2,3,4])
print(sMax(nn.forward([0.1, 0.2, 0.3])))