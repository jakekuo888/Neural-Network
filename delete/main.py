import numpy as np

# -----------------------------
# Activation Functions & Losses
# -----------------------------
def relu(x):
    return np.maximum(0, x)

def d_relu(x):
    # Fully vectorized derivative
    return (x > 0).astype(float)

def sMax(x):
    x = np.array(x, dtype=np.float64)
    e = np.exp(x - np.max(x))  # numerical stability
    return e / e.sum()

def crossEntropy(pred, true):
    # pred: softmax probabilities
    # true: integer label
    return -np.log(pred[true] + 1e-15)

def squaredError(pred, true):
    return np.sum((pred - true)**2)

# -----------------------------
# Layer Class
# -----------------------------
class Layer:
    def __init__(self, in_shape, out_shape):
        # He initialization for ReLU
        self.weights = np.random.randn(out_shape, in_shape) * np.sqrt(2 / in_shape)
        self.biases = np.zeros(out_shape)

        self.input = None
        self.z = None

    def forward(self, input):
        self.input = input
        self.z = self.weights @ input + self.biases
        return relu(self.z)

# -----------------------------
# Neural Network Class
# -----------------------------
class Neural_net:
    def __init__(self, shape):
        self.shape = shape
        self.layers = [Layer(shape[i-1], shape[i]) for i in range(1, len(shape))]

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def predict(self, input):
        out = self.forward(input)
        return sMax(out)

    def backward(self, input, true, l_rate=0.001):
        # Forward pass
        out = self.forward(input)
        pred = sMax(out)

        # Gradient at output for softmax + cross-entropy
        grad = pred.copy()
        grad[true] -= 1  # true is the integer label

        # Backpropagation
        for i, layer in enumerate(reversed(self.layers)):
            # Only apply ReLU derivative for hidden layers
            if i != 0:
                dz = grad * d_relu(layer.z)
            else:
                dz = grad  # output layer

            # Weight and bias gradients
            dw = np.outer(dz, layer.input)
            db = dz

            # Gradient descent step
            layer.weights -= l_rate * dw
            layer.biases -= l_rate * db

            # Gradient to propagate to previous layer
            grad = layer.weights.T @ dz
