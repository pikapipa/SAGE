import torch
import random
import numpy as np
import os

def init_device(prefer_gpu: bool = True, verbose: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            props = torch.cuda.get_device_properties(device)
            total_mem = props.total_memory / (1024 ** 3)
            print(f"[EnvInit] ✅ Using GPU: {props.name} ({total_mem:.2f} GB)")
    else:
        device = torch.device("cpu")
        if verbose:
            print("[EnvInit] ⚠️ Using CPU (No GPU available)")
    return device

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[EnvInit] 🌱 Seed set to {seed}")

def check_gpu_memory():
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
        allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
        free = total - reserved
        print(f"[EnvInit] 📊 GPU Memory - Total: {total:.2f} GB | Reserved: {reserved:.2f} GB | Allocated: {allocated:.2f} GB | Free: {free:.2f} GB")
    else:
        print("[EnvInit] ❌ GPU not available.")

def enable_amp_if_available():
    return torch.cuda.is_available()

def setup_environment(seed: int = 42, verbose: bool = True):
    device = init_device(verbose=verbose)
    set_seed(seed)
    if verbose:
        check_gpu_memory()
    return device