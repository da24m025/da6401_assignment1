import numpy as np
import wandb
from keras.datasets import fashion_mnist

class Optimizer:
    """
    Optimizer class implementing various optimization algorithms for updating neural network parameters.
    Supported optimizers: SGD, Momentum, Nesterov, RMSProp, Adam, Nadam.
    """
    def __init__(self, optimizer_type="sgd", lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initialize the optimizer with the specified type and hyperparameters.

        Args:
            optimizer_type (str): Type of optimizer ('sgd', 'momentum', 'nesterov', 'rmsprop', 'adam', 'nadam')
            lr (float): Learning rate
            beta1 (float): Momentum factor (first moment decay rate)
            beta2 (float): RMSProp/Adam factor (second moment decay rate)
            epsilon (float): Small constant to prevent division by zero
        """
        self.optimizer_type = optimizer_type.lower()
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # Time step for bias correction in Adam/Nadam
        self.weight_states = None
        self.bias_states = None

    def update(self, weights, biases, dW, dB):
        """
        Update weights and biases using the specified optimization algorithm.

        Args:
            weights (list): List of weight matrices
            biases (list): List of bias vectors
            dW (list): List of weight gradients
            dB (list): List of bias gradients

        Returns:
            tuple: Updated weights and biases
        """
        # Initialize state dictionaries for each layer
        if self.weight_states is None:
            if self.optimizer_type in ["momentum", "nesterov", "rmsprop"]:
                self.weight_states = [{'v_t': np.zeros_like(w)} for w in weights]
                self.bias_states = [{'v_t': np.zeros_like(b)} for b in biases]
            elif self.optimizer_type in ["adam", "nadam"]:
                self.weight_states = [{'m_t': np.zeros_like(w), 'v_t': np.zeros_like(w)} for w in weights]
                self.bias_states = [{'m_t': np.zeros_like(b), 'v_t': np.zeros_like(b)} for b in biases]

        # Ensure gradients are NumPy arrays
        dW = [np.array(dw) for dw in dW]
        dB = [np.array(db) for db in dB]

        if self.optimizer_type == "sgd":
            for i in range(len(weights)):
                weights[i] -= self.lr * dW[i]
                biases[i] -= self.lr * dB[i]
        elif self.optimizer_type == "momentum":
            for i in range(len(weights)):
                self.weight_states[i]['v_t'] = self.beta1 * self.weight_states[i]['v_t'] + dW[i]
                weights[i] -= self.lr * self.weight_states[i]['v_t']
                self.bias_states[i]['v_t'] = self.beta1 * self.bias_states[i]['v_t'] + dB[i]
                biases[i] -= self.lr * self.bias_states[i]['v_t']
        elif self.optimizer_type == "nesterov":
            for i in range(len(weights)):
                v_w = self.weight_states[i]['v_t']
                self.weight_states[i]['v_t'] = self.beta1 * v_w + dW[i]
                update_w = dW[i] + self.beta1 * self.weight_states[i]['v_t']
                weights[i] -= self.lr * update_w
                v_b = self.bias_states[i]['v_t']
                self.bias_states[i]['v_t'] = self.beta1 * v_b + dB[i]
                update_b = dB[i] + self.beta1 * self.bias_states[i]['v_t']
                biases[i] -= self.lr * update_b
        elif self.optimizer_type == "rmsprop":
            for i in range(len(weights)):
                self.weight_states[i]['v_t'] = self.beta2 * self.weight_states[i]['v_t'] + (1 - self.beta2) * (dW[i] ** 2)
                weights[i] -= self.lr * dW[i] / (np.sqrt(self.weight_states[i]['v_t']) + self.epsilon)
                self.bias_states[i]['v_t'] = self.beta2 * self.bias_states[i]['v_t'] + (1 - self.beta2) * (dB[i] ** 2)
                biases[i] -= self.lr * dB[i] / (np.sqrt(self.bias_states[i]['v_t']) + self.epsilon)
        elif self.optimizer_type == "adam":
            self.t += 1
            for i in range(len(weights)):
                m_t_w = self.weight_states[i]['m_t'] = self.beta1 * self.weight_states[i]['m_t'] + (1 - self.beta1) * dW[i]
                v_t_w = self.weight_states[i]['v_t'] = self.beta2 * self.weight_states[i]['v_t'] + (1 - self.beta2) * (dW[i] ** 2)
                m_hat_w = m_t_w / (1 - self.beta1 ** self.t)
                v_hat_w = v_t_w / (1 - self.beta2 ** self.t)
                weights[i] -= self.lr * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
                m_t_b = self.bias_states[i]['m_t'] = self.beta1 * self.bias_states[i]['m_t'] + (1 - self.beta1) * dB[i]
                v_t_b = self.bias_states[i]['v_t'] = self.beta2 * self.bias_states[i]['v_t'] + (1 - self.beta2) * (dB[i] ** 2)
                m_hat_b = m_t_b / (1 - self.beta1 ** self.t)
                v_hat_b = v_t_b / (1 - self.beta2 ** self.t)
                biases[i] -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)
        elif self.optimizer_type == "nadam":
            self.t += 1
            for i in range(len(weights)):
                m_t_w = self.weight_states[i]['m_t'] = self.beta1 * self.weight_states[i]['m_t'] + (1 - self.beta1) * dW[i]
                v_t_w = self.weight_states[i]['v_t'] = self.beta2 * self.weight_states[i]['v_t'] + (1 - self.beta2) * (dW[i] ** 2)
                m_hat_w = m_t_w / (1 - self.beta1 ** self.t)
                v_hat_w = v_t_w / (1 - self.beta2 ** self.t)
                m_bar_w = self.beta1 * m_hat_w + (1 - self.beta1) * dW[i]
                weights[i] -= self.lr * m_bar_w / (np.sqrt(v_hat_w) + self.epsilon)
                m_t_b = self.bias_states[i]['m_t'] = self.beta1 * self.bias_states[i]['m_t'] + (1 - self.beta1) * dB[i]
                v_t_b = self.bias_states[i]['v_t'] = self.beta2 * self.bias_states[i]['v_t'] + (1 - self.beta2) * (dB[i] ** 2)
                m_hat_b = m_t_b / (1 - self.beta1 ** self.t)
                v_hat_b = v_t_b / (1 - self.beta2 ** self.t)
                m_bar_b = self.beta1 * m_hat_b + (1 - self.beta1) * dB[i]
                biases[i] -= self.lr * m_bar_b / (np.sqrt(v_hat_b) + self.epsilon)
        return weights, biases

class NeuralNetwork:
    """
    Neural network class with customizable architecture and optimization.
    Supports sigmoid, ReLU, and tanh activations for hidden layers and softmax for the output layer.
    """
    def __init__(self, input_size, hidden_layers, output_size, optimizer, learning_rate=0.01, weight_init="xavier", weight_decay=0, activation='relu'):
        """
        Initialize the neural network with the specified architecture and parameters.

        Args:
            input_size (int): Number of input features
            hidden_layers (int or list): Number of neurons in hidden layers (int for single layer, list for multiple)
            output_size (int): Number of output classes
            optimizer (str): Optimizer type for the Optimizer class
            learning_rate (float): Learning rate for the optimizer
            weight_init (str): Weight initialization method ('random' or 'xavier')
            weight_decay (float): Weight decay coefficient for L2 regularization
            activation (str): Activation function for hidden layers ('relu', 'sigmoid', or 'tanh')
        """
        self.layers = [input_size] + (hidden_layers if isinstance(hidden_layers, list) else [hidden_layers]) + [output_size]
        self.optimizer = Optimizer(optimizer, learning_rate)
        self.weights = []
        self.biases = []
        self.activation = activation
        self.weight_decay = weight_decay
        for i in range(len(self.layers) - 1):
            if weight_init == "random":
                W = np.random.randn(self.layers[i], self.layers[i+1]) * 0.01
            elif weight_init == "xavier":
                W = np.random.randn(self.layers[i], self.layers[i+1]) * np.sqrt(1 / self.layers[i])
            else:
                raise ValueError("Unsupported weight initialization")
            b = np.zeros((1, self.layers[i+1]))
            self.weights.append(W)
            self.biases.append(b)

    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        """Tanh activation function."""
        return np.tanh(x)

    def softmax(self, x):
        """Softmax activation function for the output layer."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        """
        Forward pass through the network.

        Args:
            X (np.ndarray): Input data

        Returns:
            list: List of activations at each layer
        """
        activations = [X]
        for i in range(len(self.weights) - 1):
            if self.activation == 'relu':
                X = np.maximum(0, np.dot(X, self.weights[i]) + self.biases[i])
            elif self.activation == 'sigmoid':
                X = self.sigmoid(np.dot(X, self.weights[i]) + self.biases[i])
            elif self.activation == 'tanh':
                X = self.tanh(np.dot(X, self.weights[i]) + self.biases[i])
            else:
                raise ValueError("Unsupported activation")
            activations.append(X)
        X = self.softmax(np.dot(X, self.weights[-1]) + self.biases[-1])
        activations.append(X)
        return activations

    def backward(self, X, y_true, activations):
        """
        Backward pass to compute gradients with weight decay.

        Args:
            X (np.ndarray): Input data
            y_true (np.ndarray): True labels (one-hot encoded)
            activations (list): List of activations from the forward pass

        Returns:
            tuple: Gradients for weights (dW) and biases (dB)
        """
        m = X.shape[0]
        deltas = [activations[-1] - y_true]  # Output layer uses softmax with cross-entropy
        for i in range(len(self.weights) - 2, -1, -1):
            if self.activation == 'relu':
                delta = np.dot(deltas[0], self.weights[i+1].T) * (activations[i+1] > 0)
            elif self.activation == 'sigmoid':
                delta = np.dot(deltas[0], self.weights[i+1].T) * (activations[i+1] * (1 - activations[i+1]))
            elif self.activation == 'tanh':
                delta = np.dot(deltas[0], self.weights[i+1].T) * (1 - activations[i+1]**2)
            else:
                raise ValueError("Unsupported activation")
            deltas.insert(0, delta)
        # Apply weight decay to weight gradients (L2 regularization)
        dW = [(np.dot(activations[i].T, deltas[i]) / m) + self.weight_decay * self.weights[i] for i in range(len(self.weights))]
        dB = [np.sum(deltas[i], axis=0, keepdims=True) / m for i in range(len(self.biases))]
        return dW, dB

    def train(self, X, y, X_val=None, y_val=None, epochs=10, batch_size=128):
        """
        Train the neural network using mini-batch gradient descent with validation metrics.

        Args:
            X (np.ndarray): Training data
            y (np.ndarray): Training labels (one-hot encoded)
            X_val (np.ndarray, optional): Validation data
            y_val (np.ndarray, optional): Validation labels (one-hot encoded)
            epochs (int): Number of training epochs
            batch_size (int): Size of each mini-batch
        """
        for epoch in range(1, epochs + 1):
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X, y = X[indices], y[indices]
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                activations = self.forward(X_batch)
                dW, dB = self.backward(X_batch, y_batch, activations)
                self.weights, self.biases = self.optimizer.update(self.weights, self.biases, dW, dB)
            # Compute training loss
            y_pred = self.forward(X)[-1]
            train_loss = -np.mean(np.sum(y * np.log(y_pred + 1e-8), axis=1))
            # Compute validation loss if validation data is provided
            if X_val is not None and y_val is not None:
                y_val_pred = self.forward(X_val)[-1]
                val_loss = -np.mean(np.sum(y_val * np.log(y_val_pred + 1e-8), axis=1))


# Example usage:
"""
# Load and preprocess Fashion MNIST data
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train = X_train.reshape(-1, 784) / 255.0
X_test = X_test.reshape(-1, 784) / 255.0
y_train_onehot = np.zeros((y_train.size, 10))
y_train_onehot[np.arange(y_train.size), y_train] = 1

# Initialize wandb
wandb.init(project="neural-network-optimizers")

# Create and train the model
model = NeuralNetwork(
    input_size=784,
    hidden_layers=[256, 128],
    output_size=10,
    optimizer="adam",
    learning_rate=0.001,
    activation="relu",
    weight_init="xavier"
)
model.train(X_train, y_train_onehot, epochs=10, batch_size=128)
wandb.finish()
"""
