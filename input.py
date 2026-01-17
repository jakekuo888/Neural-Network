import main  # your neural network code
import numpy as np
import scipy.io

# -----------------------------
# 1. Load MNIST from .mat file
# -----------------------------
mnist = scipy.io.loadmat('data/mnist-original.mat')

X = mnist['data'].T.astype(np.float32)  # shape (70000, 784)
Y = mnist['label'].flatten().astype(int)  # shape (70000,)

# -----------------------------
# 2. Normalize images
# -----------------------------
X /= 255.0  # scale pixels to [0,1]

# -----------------------------
# 3. Split into train/test manually
# -----------------------------
num_samples = X.shape[0]
num_train = int(num_samples * 0.8)

# Shuffle indices once
indices = np.arange(num_samples)
np.random.shuffle(indices)

train_idx = indices[:num_train]
test_idx = indices[num_train:]

X_train = X[train_idx]
Y_train = Y[train_idx]

X_test = X[test_idx]
Y_test = Y[test_idx]

# -----------------------------
# 4. Initialize neural network
# -----------------------------
nn = main.Neural_net([784, 128, 64, 10])

# -----------------------------
# 5. Training parameters
# -----------------------------
epochs = 5
batch_size = 32
learning_rate = 0.001  # smaller for stability

# -----------------------------
# 6. Training loop (mini-batch)
# -----------------------------
for epoch in range(epochs):
    # Shuffle training data at the start of each epoch
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    X_train = X_train[indices]
    Y_train = Y_train[indices]

    total_loss = 0

    for start in range(0, len(X_train), batch_size):
        end = start + batch_size
        X_batch = X_train[start:end]
        Y_batch = Y_train[start:end]

        for x, y in zip(X_batch, Y_batch):
            pred = nn.predict(x)
            # cross-entropy loss
            loss = -np.log(pred[y] + 1e-15)
            total_loss += loss
            # backward pass
            nn.backward(x, y, l_rate=learning_rate)

    avg_loss = total_loss / len(X_train)

    # Evaluate test accuracy at the end of the epoch
    correct = sum(np.argmax(nn.predict(x)) == y for x, y in zip(X_test, Y_test))
    accuracy = correct / len(X_test)

    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")

# -----------------------------
# 7. Final Test Accuracy
# -----------------------------
correct = sum(np.argmax(nn.predict(x)) == y for x, y in zip(X_test, Y_test))
accuracy = correct / len(X_test)
print("Final Test Accuracy:", accuracy)
