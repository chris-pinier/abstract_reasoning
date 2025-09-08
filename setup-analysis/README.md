
# Manual Installation

## Python version
python = ">3.12"

## Using in a Python virtual environment with uv
[See more information about uv here.](https://docs.astral.sh/uv/)
uv works across Windows, macOS, and Linux

On Linux:
```bash
# Git clone if necessary, uncomment the next line
# git clone https://github.com/chris-pinier/abstract_reasoning.git
# Install uv if necessary, uncomment the next line
# curl -LsSf https://astral.sh/uv/install.sh | sh
cd setup-analysis
uv sync # Creates virtual environment
# Change analysis_lab_config.py to local data locations
uv run analysis_lab.py
# Run interactive ipython if you want, uncomment the next line
+uv run ipython
```

## Using a conda virtual environment
[See more information about conda environments here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
On Linux:
```bash
# Git clone if necessary, uncomment the next line
# git clone https://github.com/chris-pinier/abstract_reasoning.git
cd setup-analysis
conda create --name pinier python=3.12 pip
conda activate pinier
pip install -r requirements.txt
python analysis_lab.py
```



