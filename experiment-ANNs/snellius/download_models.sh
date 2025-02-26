#!/bin/bash

#SBATCH --job-name=download_model
#SBATCH --partition=cbuild
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_output/SLURM-%j-run_models.out
#SBATCH --error=slurm_output/SLURM-%j-run_models.err

# Load modules (before Python environment setup)
module purge
module load 2024
module load CUDA/12.6.0

# Activate Python environment (after modules!)
source .venv/bin/activate

# Navigate to the correct directory
cd $HOME/test  # Or wherever your script is

# # Get the model ID (passed as a command-line argument)
# model_id="$1"  # Access the first command-line argument

# Run the Python script, passing the model ID
python download_models.py