import yaml
import os
import logging
import torch
import time
from typing import Dict, Any, Optional, Union, List
import numpy as np
import json

def load_config(path: str = "config/config.yaml") -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found at: {path}")
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config

def setup_logger(name: str = "trigger", 
                level: int = logging.INFO, 
                log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_device_info() -> Dict[str, Any]:
    device_info = {
        "device": "cpu",
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
    }
    
    if torch.cuda.is_available():
        device_info["device"] = "cuda"
        device_info["cuda_device_name"] = torch.cuda.get_device_name(0)
        device_info["cuda_device_capability"] = torch.cuda.get_device_capability(0)
        device_info["cuda_memory_allocated"] = torch.cuda.memory_allocated(0) / (1024**3)
        device_info["cuda_memory_reserved"] = torch.cuda.memory_reserved(0) / (1024**3)
        props = torch.cuda.get_device_properties(0)
        device_info["cuda_total_memory"] = props.total_memory / (1024**3)
    
    return device_info

def to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_device(v, device) for v in data]
    else:
        return data

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"[TIMER] {func.__name__} execute: {end_time - start_time:.4f} second")
        return result
    return wrapper

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

def load_json(filepath: str) -> Any:
    import json
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def tensor_to_list(obj):
    if isinstance(obj, dict):
        return {k: tensor_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_list(v) for v in obj]
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj