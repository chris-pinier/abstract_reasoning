#!/bin/bash

#SBATCH --job-name=download_model
#SBATCH --partition=cbuild
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/home4/cpinier/abstract_reasoning/experiment-ANNs/snellius/slurm_output/SLURM-%j-run_models.out
#SBATCH --error=/gpfs/home4/cpinier/abstract_reasoning/experiment-ANNs/snellius/slurm_output/SLURM-%j-run_models.err

# Load modules (before Python environment setup)
module purge
module load 2024
module load CUDA/12.6.0


# Navigate to the correct directory
cd $HOME/abstract_reasoning/experiment-ANNs  # Or wherever your script is

# Activate Python environment (after modules!)
source .venv/bin/activate

# # Get the model ID (passed as a command-line argument)
# model_id="$1"  # Access the first command-line argument

# Run the Python script, passing the model ID
python snellius/download_models.py