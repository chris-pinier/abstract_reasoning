import tomllib
import subprocess
from pathlib import Path

WD = Path(__file__).resolve().parent

def submit_slurm_job(model_id, config_file, slurm_script):
    try:
        with open(config_file, "rb") as f:
            config = tomllib.load(f)
            
        model_config = config.get(model_id)

        if model_config is None:
            raise ValueError(f"No Slurm configuration found for model: {model_id}")

        sbatch_cmd = ["sbatch"]

        for arg, value in model_config.items():
            if isinstance(value, list):
                for v in value:
                    sbatch_cmd.extend([f"--{arg}", str(v)])
            else:
                sbatch_cmd.extend([f"--{arg}", str(value)])

        sbatch_cmd.append(str(slurm_script)) # Path to the SLURM script
        sbatch_cmd.append(model_id) # Pass the model_id as an argument

        print(f"Submitting Slurm job for {model_id} with command: \n{' '.join(sbatch_cmd)}\n")
        print("-" * 80)
        subprocess.run(sbatch_cmd, check=True)

    except (FileNotFoundError, ValueError, subprocess.CalledProcessError, toml.TomlDecodeError) as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    model_ids = [
        # "Qwen/Qwen2.5-72B-Instruct",
        # "Qwen/Qwen2.5-72B-Instruct-GGUF",
        # "Qwen/Qwen2.5-7B-Instruct",
        # "google/gemma-2-2b-it",
        # "google/gemma-2-9b-it".
        # "meta-llama/Llama-3.2-3B-Instruct",
        # "Qwen/QwQ-32B-Preview",
        # "meta-llama/Meta-Llama-3-8B-Instruct",
        # "meta-llama/Llama-3.2-3B-Instruct",
        # "meta-llama/Llama-3.3-70B-Instruct",
        # "Qwen/Qwen2.5-72B-Instruct",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        
    ]

    for model_id in model_ids:
        submit_slurm_job(
            model_id, 
            config_file=WD / "config/slurm_config.toml", 
            slurm_script=WD / "run_models.sh"
            )