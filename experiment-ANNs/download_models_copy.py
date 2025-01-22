import argparse
from transformers import AutoModel, AutoTokenizer
import logging
import os
import argparse
from transformers import AutoModel, AutoTokenizer
import logging
import tomllib
from pathlib import Path

WD = Path(__file__).resolve().parent

with open(WD / "credentials.toml", "rb") as f:
    credentials = tomllib.load(f)

hf_token = credentials.get("api_keys", {}).get("HF")

# * Set up a basic logger without a file handler initially
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")


def download_model(model_name, hf_token, cache_dir):
    try:
        print(f"Attempting to download model and tokenizer for: {model_name}")
        # * Download tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, token=hf_token, cache_dir=cache_dir
        )
        # * Download model
        model = AutoModel.from_pretrained(
            model_name, token=hf_token, cache_dir=cache_dir
        )
        print(f"Successfully downloaded {model_name} to {cache_dir}")

    except Exception as e:
        # * Check if a file handler is already added, if not add one
        if not any(
            isinstance(handler, logging.FileHandler) for handler in logger.handlers
        ):
            file_handler = logging.FileHandler("./logs/model_download_errors.log")
            file_handler.setFormatter(formatter)

            logger.addHandler(file_handler)
        logger.error(f"An error occurred while downloading {model_name}: {str(e)}")
        if "403 Client Error" in str(e):
            logger.error(
                "403 Client Error. This might be a gated repository. "
                "Ensure you have access and are logged in with the correct token."
            )


def main(model_names, hf_token, cache_dir):
    for name in model_names:
        download_model(name, hf_token, cache_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download models from Hugging Face.")

    # * Argument to specify the model names to download
    parser.add_argument(
        "model_names", nargs="+", help="List of model names to download"
    )

    # * Optional arguments
    # * Argument to specify the Hugging Face API token
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face API token for accessing gated models",
    )

    # * Argument to specify the cache directory
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./models",  # Default to a directory named "models" in the current working directory
        help="Directory to download models to",
    )

    args = parser.parse_args()

    hf_token = args.hf_token or hf_token

    if hf_token is None:
        print(
            "Warning: HF_TOKEN is not set. Download might fail if the model is gated."
        )
    main(args.model_names, hf_token, args.cache_dir)
