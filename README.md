
---

# Assignment 1: Feedforward Neural Network with Gradient Descent on Fashion-MNIST

This repository contains the implementation of a feedforward neural network with backpropagation and various gradient descent optimizers (SGD, Momentum, Nesterov, RMSProp, Adam, Nadam) for classifying the Fashion-MNIST dataset. The project leverages Wandb for experiment tracking, hyperparameter sweeps, and result visualization, fulfilling the assignment's dual goals of implementation and tool familiarity.

## Submission Details
- **Zip File**: `A1_DA24M025.zip` (replace `DA24M025` with your actual roll number)
- **Wandb Report**: [Link to Wandb Report](https://wandb.ai/da24m025-iit-madras/fashion-mnist-classification/reports/Assignment-1-Report--Vmlldzo5MzM0NTY) *(Note: Replace with the actual Wandb report link after creating it in your Wandb dashboard)*
- **GitHub Repository**: [Link to GitHub Repo](https://github.com/da24m025/dl-assignment1) *(Note: Replace with your actual GitHub repository link after uploading your code)*



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
Based on the sweep, the best configuration (5 hidden layers, 128 neurons, `nadam`, `tanh`, batch size 64, learning rate 0.001, Xavier initialization, no weight decay) can be run using:
```bash
python train.py --wandb_entity da24m025-iit-madras --wandb_project fashion-mnist-optimizers --dataset fashion_mnist --epochs 10 --batch_size 64 --optimizer nadam --learning_rate 0.001 --num_layers 5 --hidden_size 128 --weight_decay 0 --weight_init xavier --activation tanh --beta1 0.9 --beta2 0.999 --eps 1e-8
```
**Validation Accuracy**: 89.47%  
**Validation Loss**: 0.3178

## Experiment Results

### Question 4 Sweep Results
The best configuration from the sweep:
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

### Question 6 Insights
- **Activation Functions**: `tanh` (purple lines) outperforms `sigmoid` (yellow lines) due to stable gradients, while `ReLU` shows variability due to the dying neuron issue.
- **Learning Rate**: Low rates (0.0001–0.0005) enhance stability; 0.001 (baseline) risks overshooting but still performs well.
- **Batch Size**: Moderate sizes (32–50) improve generalization; 64 (baseline) may reduce noise excessively.
- **Optimizers**: `nadam` and `adam` (purple lines) offer stable convergence, while `sgd` and `rmsprop` (yellow lines) are less consistent.
- **Recommendation for 95% Accuracy**: Use `tanh` activation, batch size 32, 5 hidden layers, 128 neurons, learning rate 0.0005, `nadam` optimizer, no weight decay, and Xavier initialization.

### MNIST Recommendations (Based on Fashion-MNIST Learnings)
Leveraging Fashion-MNIST insights, three configurations were tested on MNIST:
1. **Configuration 1**: 3 hidden layers, `nadam`, `tanh`  
   - Command: `python train.py --wandb_entity da24m025-iit-madras --wandb_project DL --dataset mnist --epochs 10 --batch_size 32 --optimizer nadam --learning_rate 0.0005 --num_layers 3 --hidden_size 128 --weight_decay 0 --weight_init xavier --activation tanh --beta1 0.9 --beta2 0.999 --eps 1e-8`  
   - **Test Accuracy**: 97.78%  
   - **Analysis**: A shallow network excels on MNIST’s simpler patterns.
2. **Configuration 2**: 3 hidden layers, `nadam`, `ReLU`  
   - Command: `python train.py --wandb_entity da24m025-iit-madras --wandb_project DL --dataset mnist --epochs 10 --batch_size 32 --optimizer nadam --learning_rate 0.0005 --num_layers 3 --hidden_size 128 --weight_decay 0 --weight_init xavier --activation ReLU --beta1 0.9 --beta2 0.999 --eps 1e-8`  
   - **Test Accuracy**: 97.65%  
   - **Analysis**: `ReLU` offers efficiency with near-identical performance.
3. **Configuration 3**: 5 hidden layers, `adam`, `tanh`  
   - Command: `python train.py --wandb_entity da24m025-iit-madras --wandb_project DL --dataset mnist --epochs 10 --batch_size 32 --optimizer adam --learning_rate 0.0005 --num_layers 5 --hidden_size 128 --weight_decay 0 --weight_init xavier --activation tanh --beta1 0.9 --beta2 0.999 --eps 1e-8`  
   - **Test Accuracy**: 97.49%  
   - **Analysis**: Deeper network with `adam` lags slightly.

**Conclusion**: Configuration 1 (97.78%) is recommended for MNIST, with Configuration 2 as an efficient alternative.

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

### Steps to Finalize the README
1. **Replace Placeholder Links**:
   - Update the Wandb report link and GitHub repository link with actual URLs after setting up your project. Create a Wandb report in the `fashion-mnist-classification` or `fashion-mnist-optimizers` project, summarizing your experiments, and copy the link.
2. **Replace Roll Number**:
   - Update `A1_DA24M025.zip` with your actual roll number.
3. **Upload to GitHub**:
   - Create a repository named `dl-assignment1` (or similar).
   - Add all files (`q1_visualization.py`, `q2_feedforward.py`, `q3_optimizers.py`, `q4_sweep.py`, `train.py`, `requirements.txt`, `README.md`) with a commit history reflecting your work progression.
   - Push to GitHub and ensure it’s public or accessible to the TA.
4. **Create the Zip File**:
   - Zip the repository contents (excluding `.git`) into `A1_DA24M025.zip`.
   - Submit on Gradescope as instructed.

This `README.md` provides a comprehensive overview, meeting all assignment requirements while ensuring clarity and reproducibility. Let me know if further adjustments are needed!
