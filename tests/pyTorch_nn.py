import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import loadmat
import numpy as np

# ---- Load MNIST from .mat ----
mat = loadmat("data/mnist-original.mat")
X = mat["data"].T / 255.0          # shape: (70000, 784)
y = mat["label"][0].astype(int)    # shape: (70000,)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

X_train, y_train = X[:60000], y[:60000]

# ---- Define NN ----
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = Net()

# ---- Training setup ----
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ---- Train (short) ----
for epoch in range(5):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# ---- Save weights & biases to separate CSV-style files ----
np.savetxt("tests/test_data/fc1_weights.txt", model.fc1.weight.detach().numpy(), delimiter=",")
np.savetxt("tests/test_data/fc1_biases.txt", model.fc1.bias.detach().numpy()[None, :], delimiter=",")  # save as row
np.savetxt("tests/test_data/fc2_weights.txt", model.fc2.weight.detach().numpy(), delimiter=",")
np.savetxt("tests/test_data/fc2_biases.txt", model.fc2.bias.detach().numpy()[None, :], delimiter=",")  # save as row

X_test, y_test = X[60000:], y[60000:]  # last 10000 samples for testing

with torch.no_grad():  # turn off gradients
    outputs = model(X_test)            # forward pass
    _, preds = torch.max(outputs, dim=1)  # predicted classes
    correct = (preds == y_test).sum().item()
    total = y_test.size(0)
    accuracy = correct / total

print(f"Accuracy of the PyTorch model on MNIST test set: {accuracy*100:.2f}%")
