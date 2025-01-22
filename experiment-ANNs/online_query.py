from threading import Lock
import time
from dataclasses import dataclass
import requests
from pathlib import Path
import pandas as pd
from openai import OpenAI
import json
from tqdm.auto import tqdm

WD = Path(__file__).parent

# * load API key
with open(WD / "config/private.json", "r") as f:
    private = json.load(f)

api_keys = {k: v["key"] for k, v in private["API"].items()}


@dataclass
class OpenRouter:
    api_key: str
    base_url: str = "https://openrouter.ai/api/v1"

    def __post_init__(self):
        self.models = self.list_models()
        self.usage = self.fetch_usage()
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        self.rate_limit = self.usage.get("rate_limit")
        self.lock = Lock()
        self.request_count = 0
        self.reset_time = time.time()

    def set_rate_limit(self, requests: int, interval: str):
        self.rate_limit = {"requests": requests, "interval": interval}

    def list_models(self):
        url = f"{self.base_url}/models"
        headers = {"accept": "application/json"}
        response = requests.get(url, headers=headers)
        df_models = None

        if response.status_code == 200:
            response_json = response.json()
            dict_models = response_json["data"]
            df_models = pd.DataFrame(dict_models)
        else:
            print(f"Failed to fetch models: {response.status_code} - {response.text}")

        self.models = df_models

        return df_models

    def fetch_usage(self):
        url = f"{self.base_url}/auth/key"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "accept": "application/json",
        }
        response = requests.get(url, headers=headers).json()
        return response.get("data", {})

    def chat_completion(self, model_id, messages, **kwargs):
        with self.lock:
            # *  Parse the rate limit configuration
            max_requests = self.rate_limit["requests"]
            interval_seconds = self._parse_interval(self.rate_limit["interval"])

            # * Check if the interval has passed and reset if necessary
            current_time = time.time()
            if current_time - self.reset_time > interval_seconds:
                self.request_count = 0
                self.reset_time = current_time

            # * Enforce the rate limit
            if self.request_count >= max_requests:
                sleep_time = self.reset_time + interval_seconds - current_time
                print(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)

                # * Reset after sleeping
                self.request_count = 0
                self.reset_time = time.time()

            # * Increment the request count
            self.request_count += 1

        # * Make the API request
        completion = self.client.chat.completions.create(
            model=model_id,
            messages=messages,
            **kwargs,
        )

        return completion

    def _parse_interval(self, interval: str):
        """Helper function to parse the interval string."""
        time_unit = interval[-1]
        value = int(interval[:-1])

        if time_unit == "s":  # Seconds
            return value
        elif time_unit == "m":  # Minutes
            return value * 60
        elif time_unit == "h":  # Hours
            return value * 3600
        else:
            raise ValueError(f"Unsupported time unit in interval: {interval}")

    def __repr__(self):
        """Return a string representation of the OpenRouter object."""
        return "OpenRouter API client"


def main():
    open_router = OpenRouter(api_key=api_keys["openrouter"])
    # open_router.rate_limit['interval']
    # open_router.rate_limit['requests']

    completion_parameters = dict(
        temperature=1,
        max_tokens=50,
    )

    df_sequence_prompts = pd.read_csv(WD / "sequences/sequence_prompts.csv")
    df_sequence_prompts.reset_index(drop=True, inplace=True)

    # sequence_prompts = df_sequence_prompts["prompt"].tolist()

    models = [
        # "anthropic/claude-3.5-sonnet-20240620",
        # "google/gemini-2.0-flash-exp:free",
        # "google/gemini-2.0-flash-thinking-exp:free",
        "google/gemma-2-27b-it",
        "google/gemma-2-9b-it",
        # "meta-llama/llama-3.1-405b-instruct",
        "meta-llama/llama-3.2-3b-instruct",
        "meta-llama/llama-3.3-70b-instruct",
        "microsoft/phi-4",
        # "openai/gpt-4o-2024-11-20'",
        # "openai/gpt-4o-mini-2024-07-18",
        # "openai/o1-mini-2024-09-12",
        # "openai/o1-preview-2024-09-12",
        "qwen/qwen-2.5-72b-instruct",
        "qwen/qwen-2.5-7b-instruct",
        "qwen/qwq-32b-preview",
    ]

    # models = [
    #     "google/gemini-2.0-flash-thinking-exp:free",
    # ]

    pb_models = tqdm(models)
    pb_prompts = tqdm(total=len(df_sequence_prompts))

    for model_id in pb_models:
        errors = []
        
        pb_models.set_description(model_id)
        model_id_str = model_id.replace("/", "--")
        model_resps = []

        try:
            for row_idx, row in df_sequence_prompts.iterrows():
                # print(f"{'-' * 30}\n{row_idx}\n{row['prompt']}\n{'-' * 30}\n")

                completion = open_router.chat_completion(
                    model_id=model_id,
                    messages=[{"role": "user", "content": row["prompt"]}],
                    **completion_parameters,
                )

                # * Check if the 'error' attribute exists and handle accordingly
                if hasattr(completion, "error") and completion.error is not None:
                    print(
                        f"Error for prompt #{row_idx} (item_id:{row['item_id']}): {completion.error}"
                    )

                    errors.append(
                        [
                            model_id,
                            row["item_id"],
                            completion.error,
                        ]
                    )

                    if len(errors) > 5:
                        print("Too many errors, breaking")
                        break
                    else:
                        continue

                resp = completion.choices[0].message.content
                model_resps.append([row["item_id"], resp])

                pb_prompts.update(1)

            pb_prompts.reset()

        except Exception as e:
            print("AN ERROR OCCURED, details below:")
            raise e

        finally:
            save_dir = WD / "results/online_queries"
            save_dir.mkdir(exist_ok=True, parents=True)

            fpath = save_dir / f"{model_id_str}.csv"

            models_resps_df = pd.DataFrame(model_resps, columns=["item_id", "response"])
            models_resps_df.insert(0, "model_id", model_id)
            models_resps_df.to_csv(fpath, index=False)

            if len(errors) > 0:
                df_errors = pd.DataFrame(
                    errors, columns=["model_id", "item_id", "error"]
                )

                fpath = save_dir / f"errors-{model_id_str}.csv"
                df_errors.to_csv(fpath, index=False)


if __name__ == "__main__":
    main()
