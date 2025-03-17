import numpy as np
import wandb
from keras.datasets import fashion_mnist


class Optimizer:
    def __init__(self, optimizer_type="sgd", lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.optimizer_type = optimizer_type.lower()
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.weight_states = None
        self.bias_states = None

    def update(self, weights, biases, dW, dB):
        if self.weight_states is None:
            if self.optimizer_type in ["adam", "nadam"]:
                self.weight_states = [{'m_t': np.zeros_like(w), 'v_t': np.zeros_like(w)} for w in weights]
                self.bias_states = [{'m_t': np.zeros_like(b), 'v_t': np.zeros_like(b)} for b in biases]
        if self.optimizer_type == "nadam":
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

# Neural Network class with loss type support
class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size, optimizer, learning_rate=0.01, weight_init="xavier", weight_decay=0, activation='tanh', loss_type='cross_entropy'):
        self.layers = [input_size] + hidden_layers + [output_size]
        self.optimizer = Optimizer(optimizer, learning_rate)
        self.weights = []
        self.biases = []
        self.activation = activation
        self.weight_decay = weight_decay
        self.loss_type = loss_type
        for i in range(len(self.layers) - 1):
            if weight_init == "xavier":
                W = np.random.randn(self.layers[i], self.layers[i+1]) * np.sqrt(1 / self.layers[i])
            else:
                W = np.random.randn(self.layers[i], self.layers[i+1]) * 0.01
            b = np.zeros((1, self.layers[i+1]))
            self.weights.append(W)
            self.biases.append(b)

    def tanh(self, x):
        return np.tanh(x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        activations = [X]
        for i in range(len(self.weights) - 1):
            X = self.tanh(np.dot(X, self.weights[i]) + self.biases[i])
            activations.append(X)
        X = self.softmax(np.dot(X, self.weights[-1]) + self.biases[-1])
        activations.append(X)
        return activations

    def compute_loss(self, y_pred, y_true):
        if self.loss_type == 'cross_entropy':
            return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))
        elif self.loss_type == 'squared_error':
            return np.mean(np.sum((y_true - y_pred)**2, axis=1) / 2)
        else:
            raise ValueError("Unsupported loss type")

    def backward(self, X, y_true, activations):
        m = X.shape[0]
        # For simplicity, use p - y as delta for both losses (exact for CE, approximate for MSE)
        deltas = [activations[-1] - y_true]
        for i in range(len(self.weights) - 2, -1, -1):
            delta = np.dot(deltas[0], self.weights[i+1].T) * (1 - activations[i+1]**2)
            deltas.insert(0, delta)
        dW = [(np.dot(activations[i].T, deltas[i]) / m) + self.weight_decay * self.weights[i] for i in range(len(self.weights))]
        dB = [np.sum(deltas[i], axis=0, keepdims=True) / m for i in range(len(self.biases))]
        return dW, dB

    def train(self, X, y, X_val, y_val, epochs=10, batch_size=32):
        for epoch in range(epochs):
            indices = np.random.permutation(X.shape[0])
            X, y = X[indices], y[indices]
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                activations = self.forward(X_batch)
                dW, dB = self.backward(X_batch, y_batch, activations)
                self.weights, self.biases = self.optimizer.update(self.weights, self.biases, dW, dB)
            # Log metrics
            y_train_pred = self.forward(X)[-1]
            train_loss = self.compute_loss(y_train_pred, y)
            train_acc = np.mean(np.argmax(y_train_pred, axis=1) == np.argmax(y, axis=1))
            y_val_pred = self.forward(X_val)[-1]
            val_loss = self.compute_loss(y_val_pred, y_val)
            val_acc = np.mean(np.argmax(y_val_pred, axis=1) == np.argmax(y_val, axis=1))
            wandb.log({
                "Epoch": epoch + 1,
                "Loss": train_loss,
                "Accuracy": train_acc,
                "Val Loss": val_loss,
                "Val Accuracy": val_acc
            })

# Load and preprocess data
(X_train_full, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train_full = X_train_full.reshape(-1, 784) / 255.0
X_test = X_test.reshape(-1, 784) / 255.0
y_train_onehot = np.zeros((y_train.size, 10))
y_train_onehot[np.arange(y_train.size), y_train] = 1
y_test_onehot = np.zeros((y_test.size, 10))
y_test_onehot[np.arange(y_test.size), y_test] = 1

# Split into train and validation (90% train, 10% val)
split_idx = int(0.9 * X_train_full.shape[0])
X_train, X_val = X_train_full[:split_idx], X_train_full[split_idx:]
y_train, y_val = y_train_onehot[:split_idx], y_train_onehot[split_idx:]

# Function to train and log
def train_and_log(loss_type):
    wandb.init(project="fashion-mnist-loss-comparison", name=f"{loss_type}_loss")
    model = NeuralNetwork(
        input_size=784,
        hidden_layers=[128] * 5,
        output_size=10,
        optimizer="nadam",
        learning_rate=0.0005,
        weight_decay=0,
        activation="tanh",
        weight_init="xavier",
        loss_type=loss_type
    )
    model.train(X_train, y_train, X_val, y_val, epochs=10, batch_size=32)
    wandb.finish()

# Run experiments
train_and_log('cross_entropy')
train_and_log('squared_error')
