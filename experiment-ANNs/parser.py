import argparse
from pathlib import Path
import json
import code

wd = Path(__file__).parent

parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B")
parser.add_argument("--device", type=str, default=None)
parser.add_argument("--dtype", type=str, default=None)
parser.add_argument("--layer", type=str, default="layers.31.mlp.act_fn")
parser.add_argument("--hf_api_key", type=str, default=None)


args = parser.parse_args()


