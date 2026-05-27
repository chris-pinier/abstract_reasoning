import os
from pathlib import Path
from typing import *
import torch
from nnterp import StandardizedTransformer

SNELLIUS_CACHE_DIR = Path('/scratch-shared') / os.environ.get('USER')
IS_SNELLIUS = Path('/scratch-shared').exists()

def load_model(model_name: str, dispatch: bool = IS_SNELLIUS) -> StandardizedTransformer:
    model = StandardizedTransformer(
        model_name,
        cache_dir = SNELLIUS_CACHE_DIR if IS_SNELLIUS else None,
        dispatch = dispatch,
        dtype = torch.float16
    )
    return model