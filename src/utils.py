import torch
import numpy as np
import random
import os
import json


def save_results_to_json(results, output_path):
    """
    Save evaluation results to JSON file.

    Args:
        results: List of dicts with keys: target, type, sentences, output
        output_path: Path to save the JSON file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  Results saved to: {output_path}")


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)