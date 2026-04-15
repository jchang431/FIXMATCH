import random, os
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils


class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)

def get_device():
    # if you want to default to cuda first change order.
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA (gpu: {torch.cuda.get_device_name(0)}).")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def compute_params(model):
    total = 0
    for param in model.parameters():
        total += param.numel()
    print(f'total parameters {total}')



class AverageMeter(object):
    """Computes and stores the average and current value. Code credit:CS7643 A2"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # print(f'set seed to {seed}')

def load_pt_data(load_path):
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"dataset not found at {load_path}")
    
    data = torch.load(load_path)
    return data
