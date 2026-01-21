import numpy as np

def relu(x):
    return np.maximum(0, x)
def d_relu(x):
    return (x > 0).astype(float)
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

        self.input = None
        self.z = None
    
    def forward(self, input):
        self.input = input
        self.z = self.weights @ input + self.biases
        return relu(self.z)

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
    
    def predict(self, input):
        out = self.forward(input)
        return sMax(out)
    
    def backward(self, input, true, l_rate=0.01):
        out = self.forward(input)
        pred = sMax(out)
        
        # Gradient at output for softmax+cross-entropy
        grad = pred.copy()
        grad[true] -= 1  # true is the label
        
        # Backpropagation
        for i, layer in enumerate(reversed(self.layers)):
            # Only apply ReLU derivative for hidden layers
            if i != 0:
                dz = grad * d_relu(layer.z)
            else:
                dz = grad  # output layer

            dw = np.outer(dz, layer.input)
            db = dz

            layer.weights -= l_rate * dw
            layer.biases -= l_rate * db

            grad = layer.weights.T @ dz