#!/bin/bash

#SBATCH --job-name=my_model_job  # Default job name (will be overridden)
#SBATCH --time=02:00:00          # Default time (will be overridden)
#SBATCH --output=/gpfs/home4/cpinier/abstract_reasoning/experiment-ANNs/snellius/slurm_output/SLURM-%j-run_models.out # Default output (will be overridden); %j will be replaced by the job id
#SBATCH --error=/gpfs/home4/cpinier/abstract_reasoning/experiment-ANNs/snellius/slurm_output/SLURM-%j-run_models.err  # Default error (will be overridden); %j will be replaced by the job id

# Load modules (before Python environment setup)
module purge
module load 2024
module load CUDA/12.6.0

cd $HOME/abstract_reasoning/experiment-ANNs/

# Activate Python environment (after modules!)
source .venv/bin/activate

# Navigate to the correct directory
# cd snellius
# cd $HOME/abstract_reasoning/experiment-ANNs/snellius

# Get the model ID (passed as a command-line argument)
model_id="$1"  # Access the first command-line argument

# Run the Python script, passing the model ID
# uv run python run_models.py --model_id "$model_id"
python snellius/run_models.py --model_id "$model_id"