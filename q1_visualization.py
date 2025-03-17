import numpy as np
import wandb
from wandb import Image
from keras.datasets import fashion_mnist
import os

# Path to store the run ID
RUN_ID_FILE = "run_id.txt"

# Function to get or create a run ID
def get_run_id():
    if os.path.exists(RUN_ID_FILE):
        with open(RUN_ID_FILE, "r") as f:
            run_id = f.read().strip()
    else:
        run_id = wandb.util.generate_id()  # Generate a new unique run ID
        with open(RUN_ID_FILE, "w") as f:
            f.write(run_id)
    return run_id

# Get the run ID
run_id = get_run_id()

# Initialize or resume the wandb run
wandb.init(project="DL-Project01", id=run_id, resume=True)

# Load Fashion-MNIST dataset
(x_train, y_train), (_, _) = fashion_mnist.load_data()

# Define class labels for Fashion-MNIST
class_labels = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Select one image per class
wandb_images = []
for i in range(10):
    idx = np.where(y_train == i)[0][0]  # First image of class i
    img = x_train[idx]
    wandb_images.append(Image(img, caption=class_labels[i]))

# Log the images as a new step
wandb.log({"Fashion-MNIST Samples": wandb_images})