import numpy as np
import wandb
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt

# Initialize wandb
wandb.init(project="DL-Project01", name="feedforward_nn")

# Load Fashion-MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize images to range [0,1]
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0  # Flatten images
x_test = x_test.reshape(x_test.shape[0], -1) / 255.0

# One-hot encode labels
def one_hot_encode(y, num_classes=10):
    return np.eye(num_classes)[y]

y_train_onehot = one_hot_encode(y_train)
y_test_onehot = one_hot_encode(y_test)

# Neural Network class
class FeedforwardNeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.01):
        """
        Initializes a neural network with given structure.

        :param input_size: Number of input neurons (e.g., 784 for Fashion-MNIST)
        :param hidden_layers: List containing number of neurons in each hidden layer
        :param output_size: Number of output neurons (10 for classification)
        :param learning_rate: Learning rate for gradient descent
        """
        self.learning_rate = learning_rate
        self.layers = [input_size] + hidden_layers + [output_size]

        # Initialize weights and biases
        self.weights = [np.random.randn(self.layers[i], self.layers[i + 1]) * 0.01 for i in range(len(self.layers) - 1)]
        self.biases = [np.zeros((1, self.layers[i + 1])) for i in range(len(self.layers) - 1)]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stability trick
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        """
        Forward pass through the network.
        :param X: Input data
        :return: Output probabilities, Activations at each layer
        """
        activations = [X]
        for i in range(len(self.weights) - 1):
            X = self.sigmoid(np.dot(X, self.weights[i]) + self.biases[i])
            activations.append(X)

        # Output layer uses softmax
        X = self.softmax(np.dot(X, self.weights[-1]) + self.biases[-1])
        activations.append(X)

        return activations

    def backward(self, X, y_true, activations):
        """
        Backpropagation algorithm to compute gradients.
        :param X: Input data
        :param y_true: One-hot encoded true labels
        :param activations: Activations from forward pass
        """
        m = X.shape[0]
        deltas = [activations[-1] - y_true]  # Output layer error

        # Backpropagate errors through hidden layers
        for i in range(len(self.weights) - 2, -1, -1):
            delta = np.dot(deltas[0], self.weights[i + 1].T) * (activations[i + 1] * (1 - activations[i + 1]))
            deltas.insert(0, delta)

        # Compute gradients
        dW = [np.dot(activations[i].T, deltas[i]) / m for i in range(len(self.weights))]
        dB = [np.sum(deltas[i], axis=0, keepdims=True) / m for i in range(len(self.biases))]

        return dW, dB

    def update_weights(self, dW, dB):
        """
        Updates weights and biases using gradient descent.
        """
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dW[i]
            self.biases[i] -= self.learning_rate * dB[i]

    def compute_loss(self, y_pred, y_true):
        """
        Computes cross-entropy loss.
        """
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))  # Avoid log(0)

    def accuracy(self, X, y_true):
        """
        Computes classification accuracy.
        """
        y_pred = np.argmax(self.forward(X)[-1], axis=1)
        y_true_labels = np.argmax(y_true, axis=1)
        return np.mean(y_pred == y_true_labels)

    def train(self, X, y, epochs=10, batch_size=128):
        """
        Trains the model using mini-batch gradient descent.
        """
        for epoch in range(epochs):
            # Shuffle dataset
            indices = np.random.permutation(X.shape[0])
            X_shuffled, y_shuffled = X[indices], y[indices]

            # Mini-batch training
            for i in range(0, X.shape[0], batch_size):
                X_batch, y_batch = X_shuffled[i:i + batch_size], y_shuffled[i:i + batch_size]
                activations = self.forward(X_batch)
                if i + batch_size >= X.shape[0]:  # Check if it's the last batch
                    print(f"Epoch {epoch+1}, last batch, first 5 samples' output probabilities:")
                    for j in range(min(5, X_batch.shape[0])):  # Print up to 5 samples
                        probs = activations[-1][j]
                        print(f"Sample {j+1}: {np.round(probs, 4)}, sum: {np.round(np.sum(probs), 4)}")
                dW, dB = self.backward(X_batch, y_batch, activations)
                self.update_weights(dW, dB)

            # Compute loss & accuracy
            loss = self.compute_loss(self.forward(X)[-1], y)
            acc = self.accuracy(X, y)

            # Log to wandb
            wandb.log({"Epoch": epoch + 1, "Loss": loss, "Accuracy": acc})



    def predict(self, X):
        """
        Predicts class labels for given input.
        """
        return np.argmax(self.forward(X)[-1], axis=1)

# Hyperparameters
input_size = 784  # 28x28 images
hidden_layers = [128, 64]  # Two hidden layers
output_size = 10  # 10 classes
learning_rate = 0.01
epochs = 10
batch_size = 128

# Initialize and train the neural network
nn = FeedforwardNeuralNetwork(input_size, hidden_layers, output_size, learning_rate)
nn.train(x_train, y_train_onehot, epochs, batch_size)

# Evaluate on test set
test_acc = nn.accuracy(x_test, y_test_onehot)
print(f"Test Accuracy: {test_acc:.4f}")

wandb.log({"Test Accuracy": test_acc})
wandb.finish()
