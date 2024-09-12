import argparse
from transformers import AutoModel, AutoTokenizer
import logging

# Set up a basic logger without a file handler initially
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")


def download_model(model_name):
    try:
        print(f"Attempting to download model and tokenizer for: {model_name}")
        # Download tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Download model
        model = AutoModel.from_pretrained(model_name)
        print(f"Successfully downloaded {model_name}")
    except Exception as e:
        # Check if a file handler is already added, if not add one
        if not any(
            isinstance(handler, logging.FileHandler) for handler in logger.handlers
        ):
            file_handler = logging.FileHandler("model_download_errors.log")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        logger.error(f"An error occurred while downloading {model_name}: {str(e)}")


def main(model_names):
    for name in model_names:
        download_model(name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download models from Hugging Face.")
    parser.add_argument(
        "model_names", nargs="+", help="List of model names to download"
    )
    args = parser.parse_args()
    main(args.model_names)
