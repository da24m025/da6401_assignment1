# Assignment 1: Feedforward Neural Network with Gradient Descent on Fashion-MNIST

This repository implements a feedforward neural network trained with backpropagation and various gradient descent optimizers (SGD, Momentum, Nesterov, RMSProp, Adam, Nadam) for Fashion-MNIST classification. It utilizes WandB for experiment tracking, compares loss functions, and evaluates the best model with a confusion matrix.

## Submission Details
- **Zip File**: `A1_<ROLL_NUMBER>.zip` 
- **WandB Report**: [View Report](https://wandb.ai/da24m025-iit-madras/fashion-mnist-optimizers/reports/DA6401-Assignment-1-Report--VmlldzoxMTcwNjUwMQ)
- I'm unable to give public access of my Project Wandb Report , nor am i able to invite people to give access, I instead added TA Mr Siva Sankar S ch20b103@smail.iitm.ac.in as collaborator to the team as full Admin as suggested by him, If any other TA is grading my assignment please mail me at da24m025@smail.iitm.ac.in, so that i will be able to add you to the project team. Thank you.
- **GitHub Repository**: [View Repo](https://github.com/da24m025/da6401_assignment1) 



---

## Code Organization
```
dl-assignment1/
├── q1_visualization.py        # Visualizes sample images per class
├── q2_feedforward.py          # Basic feedforward network with SGD
├── q3_optimizers.py           # Implements multiple optimizers
├── q4_sweep.py                # Hyperparameter tuning with WandB
├── q7_confusion_matrix.py     # Generates confusion matrix & test accuracy
├── q8_loss_comparison.py      # Compares cross-entropy vs. squared error loss
├── train.py                   # Main training script with CLI arguments
├── requirements.txt           # Required dependencies
└── README.md                  # Documentation
```

---

## Setup Instructions

### Prerequisites
- Python 3.7+
- Git
- WandB account

### Installation
```bash
git clone https://github.com/da24m025/dl-assignment1.git
cd dl-assignment1
pip install -r requirements.txt
wandb login  # Enter API key when prompted
```

---

## Running the Code

### 1. Visualize Fashion-MNIST Samples
```bash
python q1_visualization.py
```
Logs sample images per class to the `mnist-visualizations` project.

### 2. Train a Basic Feedforward Network
```bash
python q2_feedforward.py
```
Trains a two-hidden-layer network (128, 64) on Fashion-MNIST and logs results.

### 3. Train with Different Optimizers
```bash
python q3_optimizers.py
```
Trains with `adam` by default. Edit the script to test other optimizers (e.g., `nadam`).

### 4. Run Hyperparameter Sweep
```bash
python q4_sweep.py
```
Sweeps over key hyperparameters, including:
- Epochs: [5, 10]
- Hidden Layers: [3, 4, 5]
- Hidden Size: [32, 64, 128]
- Weight Decay: [0, 0.0005, 0.5]
- Learning Rate: [1e-3, 1e-4]
- Optimizers: [SGD, Momentum, Nesterov, RMSProp, Adam, Nadam]
- Batch Size: [16, 32, 64]
- Activation: [Sigmoid, Tanh, ReLU]
- Weight Initialization: [Random, Xavier]

Logs results with run names like `hl_5_hs_128_bs_64_ac_tanh`.

### 5. Evaluate the Best Model & Confusion Matrix
```bash
python q7_confusion_matrix.py
```
Trains the best configuration (5 layers, 128 neurons, `nadam`, `tanh`, batch size 64, Xavier init, learning rate 0.001) and logs results.

### 6. Compare Loss Functions
```bash
python q8_loss_comparison.py
```
Trains a 5-layer model with `nadam` and `tanh` to compare cross-entropy and squared error losses.

### 7. Train with Best Configuration
```bash
python train.py --wandb_entity da24m025-iit-madras \
    --wandb_project fashion-mnist-optimizers --dataset fashion_mnist \
    --epochs 10 --batch_size 64 --optimizer nadam --learning_rate 0.001 \
    --num_layers 3 --hidden_size 128 --weight_decay 0 --weight_init xavier \
    --activation tanh --beta1 0.9 --beta2 0.999 --eps 1e-8 --loss cross_entropy
```

---

## Results Overview

### Best Hyperparameter Configuration (From Sweep)
- **Activation**: `tanh`
- **Batch Size**: 64
- **Epochs**: 10
- **Hidden Layers**: 5
- **Hidden Size**: 128
- **Learning Rate**: 0.001
- **Optimizer**: `nadam`
- **Weight Decay**: 0
- **Weight Initialization**: `xavier`
- **Validation Accuracy**: 89.47%
- **Validation Loss**: 0.3178

### Test Accuracy & Confusion Matrix
- **Test Accuracy**: ~89-90% (Check WandB logs)
- **Confusion Matrix**: Normalized heatmap with diagonal highlighting for correct predictions, logged under `fashion-mnist-best-model`.

### Loss Function Comparison
- **Cross-Entropy**: ~89-90% Validation Accuracy, ~0.3-0.35 Validation Loss
- **Squared Error**: ~85-87% Validation Accuracy, ~0.1-0.15 Validation Loss

---

## Recommended Configurations for MNIST
| Configuration | Hidden Layers | Activation | Optimizer | Learning Rate | Batch Size | Test Accuracy |
|--------------|--------------|------------|------------|--------------|------------|--------------|
| **Config 1** | 3 | Tanh | Nadam | 0.0005 | 32 | **97.78%** |
| **Config 2** | 3 | ReLU | Nadam | 0.0005 | 32 | 97.65% |
| **Config 3** | 5 | Tanh | Adam | 0.0005 | 32 | 97.49% |

Run with:
```bash
python train.py --dataset mnist --epochs 10 --batch_size 32 --optimizer nadam \
    --learning_rate 0.0005 --num_layers 3 --hidden_size 128 --weight_init xavier \
    --activation tanh --loss cross_entropy
```

---

## Dependencies
```bash
numpy>=1.19.0
wandb>=0.12.0
keras>=2.4.0
scikit-learn>=0.24.0
seaborn>=0.11.0
matplotlib>=3.3.0
```

---



---

This refined version enhances readability, structure, and clarity while keeping all details intact.



