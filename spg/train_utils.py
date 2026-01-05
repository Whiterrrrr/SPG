import os
import re
import glob
from typing import Optional
from dataclasses import dataclass, field
from transformers import TrainerCallback
from datasets import load_dataset, Dataset


# Helper: find latest checkpoint-{step} under a directory
def find_latest_checkpoint(base_path: str = None):
    if base_path is None:
        return None, None
    
    # If base_path itself is a checkpoint dir like .../checkpoint-123
    m = re.search(r"checkpoint-(\d+)$", base_path)
    if m:
        return base_path, int(m.group(1))

    # Otherwise look for checkpoint-* subdirs
    pattern = os.path.join(base_path, "checkpoint-*")
    candidates = glob.glob(pattern)
    max_step = -1
    max_path = None
    for c in candidates:
        mm = re.search(r"checkpoint-(\d+)$", c)
        if mm:
            step = int(mm.group(1))
            if step > max_step:
                max_step = step
                max_path = c
    if max_path:
        return max_path, max_step
    return None, None


class ResumeStepCallback(TrainerCallback):
    def __init__(self, resume_step, max_steps):
        super().__init__()
        self.resume_step = resume_step
        self.max_steps = max_steps

    def on_train_begin(self, args, state, control, **kwargs):
        if self.resume_step is not None:
            try:
                state.global_step = int(self.resume_step)
            except Exception:
                pass
        if self.max_steps is not None:
            try:
                state.max_steps = int(self.max_steps)
            except Exception:
                pass