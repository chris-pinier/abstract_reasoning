from dataclasses import dataclass
import requests
from threading import Lock
from openai import OpenAI
import time
import pandas as pd


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

    def chat_completion_image(self, model_id, **kwargs):
        raise NotImplementedError("Image completions not implemented yet.")

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
