import numpy as np
from scipy.io import loadmat

# ---- Activation functions ----
def relu(x):
    return np.maximum(0, x)

def sMax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

# ---- NumPy NN classes ----
class Layer:
    def __init__(self, in_shape, out_shape):
        self.weights = np.random.uniform(-0.5, 0.5, (out_shape, in_shape))
        self.biases = np.random.uniform(-0.5, 0.5, out_shape)

class Neural_net:
    def __init__(self, shape):
        self.shape = shape
        self.layers = []
        for i in range(1, len(shape)):
            self.layers.append(Layer(shape[i-1], shape[i]))

    def forward(self, input):
        for i, layer in enumerate(self.layers):
            input = layer.weights @ input + layer.biases
            if i != len(self.layers) - 1:  # ReLU only on hidden layers
                input = relu(input)
        return input

# ---- Initialize network and load pre-trained weights ----
nn = Neural_net([784, 128, 10])
nn.layers[0].weights = np.loadtxt("tests/test_data/fc1_weights.txt", delimiter=",")
nn.layers[0].biases = np.loadtxt("tests/test_data/fc1_biases.txt", delimiter=",")
nn.layers[1].weights = np.loadtxt("tests/test_data/fc2_weights.txt", delimiter=",")
nn.layers[1].biases = np.loadtxt("tests/test_data/fc2_biases.txt", delimiter=",")

# ---- Load MNIST test data ----
mat = loadmat("data/mnist-original.mat")
X = mat["data"].T / 255.0          # shape (70000, 784)
y = mat["label"][0].astype(int)    # shape (70000,)
X_test = X[60000:]                 # last 10000 samples
y_test = y[60000:]

# ---- Evaluate accuracy ----
correct = 0
for i in range(len(X_test)):
    sample = X_test[i]
    output = nn.forward(sample)
    pred = np.argmax(sMax(output))
    if pred == y_test[i]:
        correct += 1

accuracy = correct / len(X_test)
print(f"Accuracy on MNIST test set: {accuracy*100:.2f}%")
