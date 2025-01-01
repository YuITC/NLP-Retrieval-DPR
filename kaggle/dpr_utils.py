import re
import yaml
import types
import torch
import faiss
import random
import numpy as np
from loguru import logger
from pathlib import Path
from torch.optim.lr_scheduler import LambdaLR

def check_device():
    print("=== PyTorch Check ===")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Version   : {torch.version.cuda}")
    print(f"Number of GPUs : {torch.cuda.device_count()}")
    print("\n".join(
        [f"- ID {i}: {torch.cuda.get_device_name(i)}" 
            for i in range(torch.cuda.device_count())]
    ))

    print("\n=== FAISS Check ===")
    print(f"FAISS Version       : {faiss.__version__}")
    print(f"FAISS GPU Support   : {'Yes' if hasattr(faiss, 'StandardGpuResources') else 'No'}")
    index = faiss.IndexFlatL2(128)
    print("FAISS Index Creation: Successful")
    
def get_best_model(file_path):
    # Extract data from dpr_train.log
    step_pattern     = re.compile(r"Step: (\d+)")
    avg_rank_pattern = re.compile(r"Average Rank: ([\d\.]+)")
    loss_pattern     = re.compile(r"Loss: ([\d\.]+)")

    steps, avg_ranks, losses = [], [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            step_match     = step_pattern.search(line)
            avg_rank_match = avg_rank_pattern.search(line)
            loss_match     = loss_pattern.search(line)

            if step_match and avg_rank_match and loss_match:
                steps.append(int(step_match.group(1)))
                avg_ranks.append(float(avg_rank_match.group(1)))
                losses.append(float(loss_match.group(1)))

    # Find the index of the best model
    best_index = min(range(len(avg_ranks)), key=lambda i: (avg_ranks[i], losses[i]))
    print(f"Best model is at Step: {steps[best_index]}")
    print(f"Average Rank         : {avg_ranks[best_index]}")
    print(f"Loss                 : {losses[best_index]}")

def get_logger(log_name, log_dir=Path("logs")):
    """
    Initialize a customized logger for structured and formatted logging.

    Args:
        log_name (str): Name of the file using the logger.
        log_dir (Path): Directory to save log files (default: "logs").
    """
    log_dir.mkdir(exist_ok=True)
    logger.remove()
    logger.add(
        sink=log_dir / f"{Path(log_name).stem}.log",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{file}:{line} - {function}</cyan> | <yellow>{message}</yellow>",
        level="INFO",
        enqueue=True, # Ensure thread-safe logging
    )
    return logger

def set_seed(seed=42): 
    """
    Set a fixed seed for reproducibility.
    """
    random.seed(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def normalize_document(document):
    document = re.sub(r"[’‘“”]", "'", document)
    document = re.sub(r"\n", " ", document)
    return re.sub(r'^\s*"?|"?\s*$', '', document)

def normalize_query(question):
    return re.sub(r"[’‘“”]", "'", question)

def get_config(path):
    """
    Load a configuration file and convert it into a namespace object for easy access.
    
    Notes:
        Usage: config.key -> value
    """
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return types.SimpleNamespace(**config)

def get_linear_scheduler(optimizer, warmup_steps, total_training_steps, steps_shift=0, last_epoch=-1):
    """
    Create a linear lr scheduler using "Linear Warmup With Linear Decay" strategy.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to apply the lr schedule to.
        warmup_steps                (int): Number of steps for the warmup phase.
        total_training_steps        (int): Total number of training steps.
        steps_shift                 (int): Additional steps to adjust for continued training from a checkpoint.
        last_epoch                  (int): Index of the last completed epoch for resuming training (default: -1).
    """
    
    def lr_warmup_and_decay(current_step):
        current_step += steps_shift
        
        # Linearly scale up the learning rate during warmup
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        
        # Linearly decay the learning rate after warmup
        return max(1e-7, float(total_training_steps - current_step) / float(max(1, total_training_steps - warmup_steps)))
    
    return LambdaLR(optimizer, lr_warmup_and_decay, last_epoch)