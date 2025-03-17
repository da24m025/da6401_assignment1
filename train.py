import argparse
import numpy as np
import wandb
from keras.datasets import mnist, fashion_mnist

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
            if self.optimizer_type in ["momentum", "nesterov", "rmsprop"]:
                self.weight_states = [{'v_t': np.zeros_like(w)} for w in weights]
                self.bias_states = [{'v_t': np.zeros_like(b)} for b in biases]
            elif self.optimizer_type in ["adam", "nadam"]:
                self.weight_states = [{'m_t': np.zeros_like(w), 'v_t': np.zeros_like(w)} for w in weights]
                self.bias_states = [{'m_t': np.zeros_like(b), 'v_t': np.zeros_like(b)} for b in biases]

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
        elif self.optimizer_type == "nesterov" or self.optimizer_type == "nag":
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
    def __init__(self, input_size, hidden_layers, output_size, optimizer, learning_rate=0.01, weight_init="xavier", weight_decay=0, activation='relu'):
        self.layers = [input_size] + hidden_layers + [output_size]
        self.optimizer = Optimizer(optimizer, learning_rate, beta1=args.beta1, beta2=args.beta2, epsilon=args.eps)
        self.weights = []
        self.biases = []
        self.activation = activation.lower()
        self.weight_decay = weight_decay
        for i in range(len(self.layers) - 1):
            if weight_init.lower() == "random":
                W = np.random.randn(self.layers[i], self.layers[i+1]) * 0.01
            elif weight_init.lower() == "xavier":
                W = np.random.randn(self.layers[i], self.layers[i+1]) * np.sqrt(1 / self.layers[i])
            else:
                raise ValueError("Unsupported weight initialization")
            b = np.zeros((1, self.layers[i+1]))
            self.weights.append(W)
            self.biases.append(b)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        activations = [X]
        for i in range(len(self.weights) - 1):
            if self.activation == 'relu':
                X = self.relu(np.dot(X, self.weights[i]) + self.biases[i])
            elif self.activation == 'sigmoid':
                X = self.sigmoid(np.dot(X, self.weights[i]) + self.biases[i])
            elif self.activation == 'tanh':
                X = self.tanh(np.dot(X, self.weights[i]) + self.biases[i])
            elif self.activation == 'identity':
                X = np.dot(X, self.weights[i]) + self.biases[i]
            else:
                raise ValueError("Unsupported activation")
            activations.append(X)
        X = self.softmax(np.dot(X, self.weights[-1]) + self.biases[-1])
        activations.append(X)
        return activations

    def backward(self, X, y_true, activations):
        m = X.shape[0]
        deltas = [activations[-1] - y_true]
        for i in range(len(self.weights) - 2, -1, -1):
            if self.activation == 'relu':
                delta = np.dot(deltas[0], self.weights[i+1].T) * (activations[i+1] > 0)
            elif self.activation == 'sigmoid':
                delta = np.dot(deltas[0], self.weights[i+1].T) * (activations[i+1] * (1 - activations[i+1]))
            elif self.activation == 'tanh':
                delta = np.dot(deltas[0], self.weights[i+1].T) * (1 - activations[i+1]**2)
            elif self.activation == 'identity':
                delta = np.dot(deltas[0], self.weights[i+1].T)
            else:
                raise ValueError("Unsupported activation")
            deltas.insert(0, delta)
        dW = [(np.dot(activations[i].T, deltas[i]) / m) + self.weight_decay * self.weights[i] for i in range(len(self.weights))]
        dB = [np.sum(deltas[i], axis=0, keepdims=True) / m for i in range(len(self.biases))]
        return dW, dB

    def train(self, X, y, X_val=None, y_val=None, epochs=10, batch_size=128):
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
            y_pred = self.forward(X)[-1]
            train_loss = -np.mean(np.sum(y * np.log(y_pred + 1e-8), axis=1))
            train_accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))
            if X_val is not None and y_val is not None:
                y_val_pred = self.forward(X_val)[-1]
                val_loss = -np.mean(np.sum(y_val * np.log(y_val_pred + 1e-8), axis=1))
                val_accuracy = np.mean(np.argmax(y_val_pred, axis=1) == np.argmax(y_val, axis=1))
                wandb.log({
                    "Epoch": epoch,
                    "Loss": train_loss,
                    "Accuracy": train_accuracy,
                    "Val Loss": val_loss,
                    "Val Accuracy": val_accuracy
                })
            else:
                wandb.log({
                    "Epoch": epoch,
                    "Loss": train_loss,
                    "Accuracy": train_accuracy
                })

    def evaluate(self, X, y):
        y_pred = self.forward(X)[-1]
        accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))
        return accuracy

def main(args):
    # Load and preprocess data
    if args.dataset == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    elif args.dataset == "fashion_mnist":
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError("Unsupported dataset")

    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    y_train_onehot = np.zeros((y_train.size, 10))
    y_train_onehot[np.arange(y_train.size), y_train] = 1
    y_test_onehot = np.zeros((y_test.size, 10))
    y_test_onehot[np.arange(y_test.size), y_test] = 1

    # Split training data into train and validation
    split_idx = int(0.9 * X_train.shape[0])
    X_train, X_val = X_train[:split_idx], X_train[split_idx:]
    y_train_onehot, y_val_onehot = y_train_onehot[:split_idx], y_train_onehot[split_idx:]

    # Initialize wandb
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))

    # Create hidden layers list
    hidden_layers = [args.hidden_size] * args.num_layers

    # Initialize and train the model
    model = NeuralNetwork(
        input_size=784,
        hidden_layers=hidden_layers,
        output_size=10,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        activation=args.activation,
        weight_init=args.weight_init
    )
    model.train(X_train, y_train_onehot, X_val, y_val_onehot, epochs=args.epochs, batch_size=args.batch_size)

    # Evaluate on test set
    test_accuracy = model.evaluate(X_test, y_test_onehot)
    wandb.log({"Test Accuracy": test_accuracy})
    print(f"Test Accuracy: {test_accuracy:.4f}")

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a feedforward neural network on MNIST or Fashion-MNIST")
    parser.add_argument("--wandb_project", "-wp", default="DL", help="Wandb project name")
    parser.add_argument("--wandb_entity", "-we", default="da24m025", help="Wandb entity name")
    parser.add_argument("--dataset", "-d", default="fashion_mnist", choices=["mnist", "fashion_mnist"], help="Dataset to use")
    parser.add_argument("--epochs", "-e", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size")
    parser.add_argument("--loss", "-l", default="cross_entropy", choices=["mean_squared_error", "cross_entropy"], help="Loss function")
    parser.add_argument("--optimizer", "-o", default="nadam", choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], help="Optimizer")
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--momentum", "-m", type=float, default=0.5, help="Momentum for momentum and nag optimizers")
    parser.add_argument("--beta", type=float, default=0.5, help="Beta for rmsprop optimizer")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 for adam and nadam optimizers")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 for adam and nadam optimizers")
    parser.add_argument("--eps", "--epsilon", type=float, default=1e-8, help="Epsilon for optimizers")
    parser.add_argument("--weight_decay", "-w_d", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--weight_init", "-w_i", default="Xavier", choices=["random", "Xavier"], help="Weight initialization")
    parser.add_argument("--num_layers", "-nhl", type=int, default=5, help="Number of hidden layers")
    parser.add_argument("--hidden_size", "-sz", type=int, default=128, help="Hidden layer size")
    parser.add_argument("--activation", "-a", default="tanh", choices=["identity", "sigmoid", "tanh", "ReLU"], help="Activation function")

    args = parser.parse_args()
    main(args)