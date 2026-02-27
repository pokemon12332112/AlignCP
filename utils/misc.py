import torch
import numpy as np
import random


def set_seeds(seed_value, use_cuda):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value) 
    random.seed(seed_value)  
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) 
        torch.backends.cudnn.deterministic = True  
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)