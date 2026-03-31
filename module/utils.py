import os
import re
import numpy as np
import random
from numpy.random import RandomState
from numpy.random import seed as seednp

from module.models import Model
from module.datasets.esnli import ESNLI

import torch

def set_seed(seed: int) -> RandomState:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # set to false for reproducibility, True to boost performance
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    random_state = random.getstate()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return random_state

def get_language_model(model_tag, max_tokens=256, temperature=0.7, batch_size=4, load_in_4bit=False, thinking=False):
    if model_tag == "Llama3.2_1B":
        return Model(name="meta-llama/Llama-3.2-1B-Instruct", max_tokens=max_tokens, temperature=temperature, batch_size=batch_size, load_in_4bit=load_in_4bit, padding_side="left")
    elif model_tag == "Llama3.1_8B":
        return Model(name="meta-llama/Llama-3.1-8B-Instruct", max_tokens=max_tokens, temperature=temperature, batch_size=batch_size, load_in_4bit=load_in_4bit, padding_side="left")
    elif model_tag == "Llama3.3_70B":
        return Model(name="meta-llama/Llama-3.3-70B-Instruct", max_tokens=max_tokens, temperature=temperature, batch_size=batch_size, load_in_4bit=load_in_4bit, padding_side="left")
    elif model_tag == "Qwen3_4B":
        return Model(name="Qwen/Qwen3-4B", max_tokens=max_tokens, temperature=temperature, batch_size=batch_size, load_in_4bit=load_in_4bit, padding_side="left", thinking=thinking)
    elif model_tag == "Qwen3_8B":
        return Model(name="Qwen/Qwen3-8B", max_tokens=max_tokens, temperature=temperature, batch_size=batch_size, load_in_4bit=load_in_4bit, padding_side="left", thinking=thinking)
    elif model_tag == "Qwen3_14B":
        return Model(name="Qwen/Qwen3-14B", max_tokens=max_tokens, temperature=temperature, batch_size=batch_size, load_in_4bit=load_in_4bit, padding_side="left", thinking=thinking)
    elif model_tag == "Qwen3_32B":
        return Model(name="Qwen/Qwen3-32B", max_tokens=max_tokens, temperature=temperature, batch_size=batch_size, load_in_4bit=load_in_4bit, padding_side="left", thinking=thinking)
    else:
        return Model(name=model_tag, max_tokens=max_tokens, temperature=temperature, batch_size=batch_size, load_in_4bit=load_in_4bit, padding_side="left", thinking=thinking)
    
def get_dataset(dataset_name, dataset_path, split="train"):
    if dataset_name == "esnli":
        return ESNLI(filepath=dataset_path, split=split)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
