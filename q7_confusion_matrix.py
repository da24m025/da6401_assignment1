import numpy as np
import wandb
from keras.datasets import fashion_mnist

# Initialize wandb
wandb.init(project="DL-Project01", name="best_config_run")

# Load and preprocess Fashion-MNIST dataset
(X_full_train, y_full_train), (X_test, y_test) = fashion_mnist.load_data()
X_full_train = X_full_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
X_full_train = X_full_train.reshape(X_full_train.shape[0], -1)  # Flatten to (num_samples, 784)
X_test = X_test.reshape(X_test.shape[0], -1)

# One-hot encode labels
def one_hot_encode(y, num_classes=10):
    return np.eye(num_classes)[y]

y_full_train_onehot = one_hot_encode(y_full_train)
y_test_onehot = one_hot_encode(y_test)

# Split into train and validation sets (90% train, 10% validation)
split_idx = int(0.9 * X_full_train.shape[0])
X_train, X_val = X_full_train[:split_idx], X_full_train[split_idx:]
y_train, y_val = y_full_train_onehot[:split_idx], y_full_train_onehot[split_idx:]

# Best configuration
best_config = {
    'activation': 'tanh',
    'batch_size': 64,
    'epochs': 10,
    'hidden_layers': 5,
    'hidden_size': 128,
    'learning_rate': 0.001,
    'optimizer': 'nadam',
    'weight_decay': 0,
    'weight_init': 'xavier'
}

# Define hidden layer sizes: 5 layers of 128 neurons each
hidden_layer_sizes = [best_config['hidden_size']] * best_config['hidden_layers']

# Initialize the neural network (using your NeuralNetwork class)
model = NeuralNetwork(
    input_size=784,  # 28x28 flattened
    hidden_layers=hidden_layer_sizes,
    output_size=10,
    optimizer=best_config['optimizer'],
    learning_rate=best_config['learning_rate'],
    weight_decay=best_config['weight_decay'],
    activation=best_config['activation'],
    weight_init=best_config['weight_init']
)

# Train the model
model.train(
    X_train, y_train,
    X_val, y_val,
    epochs=best_config['epochs'],
    batch_size=best_config['batch_size']
)

# Evaluate on the test set
y_test_pred = model.forward(X_test)[-1]  # Get output probabilities
test_accuracy = np.mean(np.argmax(y_test_pred, axis=1) == np.argmax(y_test_onehot, axis=1))
print(f"Test Accuracy: {test_accuracy:.4f}")

# Log the test accuracy to wandb
wandb.log({"Test Accuracy": test_accuracy})

from sklearn.metrics import confusion_matrix

# Get predicted and true labels
y_test_true_labels = np.argmax(y_test_onehot, axis=1)
y_test_pred_labels = np.argmax(y_test_pred, axis=1)

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test_true_labels, y_test_pred_labels)

# Define Fashion-MNIST class labels
class_labels = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

import seaborn as sns
import matplotlib.pyplot as plt

# Normalize confusion matrix (for better visualization)
conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

# Plot
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_norm, annot=True, cmap="coolwarm", fmt=".2f", xticklabels=class_labels, yticklabels=class_labels)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Fashion-MNIST Confusion Matrix")



# Log to WandB
step = best_config['epochs']
wandb.log({"Confusion Matrix": wandb.Image(plt), "step": step})


plt.show()
