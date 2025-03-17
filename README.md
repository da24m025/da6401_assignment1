
---

# Assignment 1: Feedforward Neural Network with Gradient Descent on Fashion-MNIST

This repository contains the implementation of a feedforward neural network with backpropagation and various gradient descent optimizers (SGD, Momentum, Nesterov, RMSProp, Adam, Nadam) for classifying the Fashion-MNIST dataset. The project leverages Wandb for experiment tracking, hyperparameter sweeps, and result visualization, fulfilling the assignment's dual goals of implementation and tool familiarity.

## Submission Details
- **Zip File**: `A1_DA24M025.zip` 
- **Wandb Report**: [Wandb Report](https://wandb.ai/da24m025-iit-madras/fashion-mnist-optimizers/reports/DA6401-Assignment-1-Report--VmlldzoxMTcwNjUwMQ)
- **GitHub Repository**: [GitHub Repo](https://github.com/da24m025/da6401_assignment1) 


## Code Organization
The repository is structured to reflect a modular and evolutionary development process, adhering to good software engineering practices:

```
dl-assignment1/
├── q1_visualization.py        # Code for Question 1: Plots one sample image per Fashion-MNIST class
├── q2_feedforward.py          # Code for Question 2: Basic feedforward neural network implementation
├── q3_optimizers.py           # Code for Question 3: Neural network with multiple optimizers
├── q4_sweep.py                # Code for Question 4: Hyperparameter sweep using Wandb
├── train.py                   # Main script supporting command-line arguments for training
├── requirements.txt           # List of required Python packages
└── README.md                  # Project documentation (this file)
```

### File Descriptions
- **`q1_visualization.py`**: Loads Fashion-MNIST and logs one sample image per class to Wandb for visualization.
- **`q2_feedforward.py`**: Implements a basic feedforward neural network with backpropagation using SGD, supporting flexible layer configurations.
- **`q3_optimizers.py`**: Extends the neural network to include SGD, Momentum, Nesterov, RMSProp, Adam, and Nadam optimizers with backpropagation.
- **`q4_sweep.py`**: Conducts a hyperparameter sweep on Fashion-MNIST to optimize network performance, logging results to Wandb.
- **`train.py`**: A flexible script accepting command-line arguments for dataset (Fashion-MNIST/MNIST), optimizer, activation, and other hyperparameters, used for final experiments.
- **`requirements.txt`**: Lists dependencies (`numpy`, `wandb`, `keras`) required to run the project.

## Setup Instructions

### Prerequisites
- Python 3.7 or higher
- Git (for repository management)
- Wandb account (for experiment tracking)

### Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/da24m025/dl-assignment1.git
   cd dl-assignment1
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Log in to Wandb**:
   ```bash
   wandb login
   ```
   Follow the prompt to enter your API key (available from your Wandb account settings).

### Running the Code

#### Question 1: Visualize Fashion-MNIST Samples
To generate and log sample images:
```bash
python q1_visualization.py
```
This creates a grid of one image per class in the Wandb dashboard under the `mnist-visualizations` project.

#### Question 2: Train Basic Feedforward Network
To train the initial neural network:
```bash
python q2_feedforward.py
```
This trains a network with two hidden layers (128, 64 neurons) on Fashion-MNIST, logging accuracy and loss to Wandb.

#### Question 3: Train with Multiple Optimizers
To test the network with different optimizers:
```bash
python q3_optimizers.py
```
This script trains with the `adam` optimizer by default; modify the `optimizer` parameter to test others (e.g., `nadam`).

#### Question 4: Run Hyperparameter Sweep
To perform the sweep and find the best configuration:
```bash
python q4_sweep.py
```
The sweep explores epochs (5, 10), hidden layers (3, 4, 5), hidden size (32, 64, 128), weight decay (0, 0.0005, 0.5), learning rate (1e-3, 1e-4), optimizer (sgd, momentum, nesterov, rmsprop, adam, nadam), batch size (16, 32, 64), weight initialization (random, xavier), and activation (sigmoid, tanh, relu). Results are logged to the `fashion-mnist-optimizers` project with meaningful run names (e.g., `hl_5_hs_128_bs_64_ac_tanh`).

#### Train with Recommended Configuration
Based on the sweep, the best configuration (3 hidden layers, 128 neurons, `nadam`, `tanh`, batch size 64, learning rate 0.001, Xavier initialization, no weight decay) can be run using:
```bash
python train.py --wandb_entity da24m025-iit-madras --wandb_project fashion-mnist-optimizers --dataset fashion_mnist --epochs 10 --batch_size 64 --optimizer nadam --learning_rate 0.001 --num_layers 5 --hidden_size 128 --weight_decay 0 --weight_init xavier --activation tanh --beta1 0.9 --beta2 0.999 --eps 1e-8
```


## Dependencies
The `requirements.txt` file includes:
```
numpy>=1.19.0
wandb>=0.12.0
keras>=2.4.0
```

## Notes
- All implementations use NumPy for matrix operations, adhering to the restriction against Keras/TensorFlow optimizers and layers.
- Commit history reflects incremental development (e.g., initial network, optimizer additions, sweep implementation).
- Verify Wandb logs under `da24m025-iit-madras/fashion-mnist-optimizers` and `DL` projects.
- The `train.py` script supports the required command-line arguments for TA verification.

---

