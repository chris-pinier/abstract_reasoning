import yaml
from pathlib import Path

# Define the path to the globals file
base_dir = Path(__file__).resolve().parent
globals_path = (base_dir / '../../' / 'globals.yml').resolve()

# Load data from YAML
with open(globals_path, "r") as stream:
    data = yaml.safe_load(stream)

# Variables are named after the keys in the YAML file
for key, value in data.items():
    globals()[key] = (base_dir / '../../' / value).resolve()