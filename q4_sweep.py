import wandb
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical

# Load and preprocess data
(X_full_train, y_full_train), (X_test, y_test) = fashion_mnist.load_data()
X_full_train = X_full_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
X_full_train = X_full_train.reshape(X_full_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
y_full_train = to_categorical(y_full_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
split_idx = int(0.9 * X_full_train.shape[0])
X_train, X_val = X_full_train[:split_idx], X_full_train[split_idx:]
y_train, y_val = y_full_train[:split_idx], y_full_train[split_idx:]

# Sweep configuration
sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'Val Loss', 'goal': 'minimize'},
    'parameters': {
        'epochs': {'values': [5, 10]},
        'hidden_layers': {'values': [3, 4, 5]},
        'hidden_size': {'values': [32, 64, 128]},
        'weight_decay': {'values': [0, 0.0005, 0.5]},
        'learning_rate': {'values': [1e-3, 1e-4]},
        'optimizer': {'values': ['sgd', 'momentum', 'nesterov', 'rmsprop', 'adam', 'nadam']},
        'batch_size': {'values': [16, 32, 64]},
        'weight_init': {'values': ['random', 'xavier']},
        'activation': {'values': ['sigmoid', 'tanh', 'relu']}
    }
}
sweep_id = wandb.sweep(sweep_config, project="fashion-mnist-optimizers")

# Training function
def train():
    with wandb.init() as run:
        config = wandb.config
        run.name = f"hl_{config.hidden_layers}_hs_{config.hidden_size}_bs_{config.batch_size}_ac_{config.activation}"
        hidden_layer_sizes = [config.hidden_size] * config.hidden_layers
        model = NeuralNetwork(
            input_size=784,
            hidden_layers=hidden_layer_sizes,
            output_size=10,
            optimizer=config.optimizer,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            activation=config.activation,
            weight_init=config.weight_init
        )
        for epoch in range(1, config.epochs + 1):
            model.train(X_train, y_train, X_val, y_val, epochs=1, batch_size=config.batch_size)
            y_train_pred = model.forward(X_train)[-1]
            train_loss = -np.mean(np.sum(y_train * np.log(y_train_pred + 1e-8), axis=1))
            train_accuracy = np.mean(np.argmax(y_train_pred, axis=1) == np.argmax(y_train, axis=1))
            y_val_pred = model.forward(X_val)[-1]
            val_loss = -np.mean(np.sum(y_val * np.log(y_val_pred + 1e-8), axis=1))
            val_accuracy = np.mean(np.argmax(y_val_pred, axis=1) == np.argmax(y_val, axis=1))
            wandb.log({
                "Loss": train_loss,
                "Accuracy": train_accuracy,
                "Val Loss": val_loss,
                "Val Accuracy": val_accuracy
            })
            print(f"Epoch {epoch}/{config.epochs} - Train Loss: {train_loss:.4f} - Train Acc: {train_accuracy:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_accuracy:.4f}")

# Run sweep
wandb.agent(sweep_id, function=train)