import unsloth

import torch
import gc

def free_gpu():
    """Dọn bộ nhớ GPU cho training/inference"""
    for _ in range(5):
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
    print("[INFO] ✅ GPU memory cleaned!")