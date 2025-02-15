import numpy as np
import torch
import random

# Set random seed for all usages
def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)