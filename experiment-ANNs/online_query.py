from pathlib import Path
import os
import pandas as pd
import json
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from typing import Union, Optional
import traceback
from online_providers import OpenRouter
import time

WD = Path(__file__).parent
# os.chdir(WD)
assert WD == Path.cwd()

STORAGE_LOCATION = Path("/Volumes/Realtek 1Tb/PhD Data/experiment1/data/ANNs")

if STORAGE_LOCATION.exists():
    EXPORT_DIR = STORAGE_LOCATION / "online_queries"
else:
    print("WARNING: Storage location not found, using local storage")
    EXPORT_DIR = WD / "results/online_queries"

EXPORT_DIR.mkdir(exist_ok=True, parents=True)

# * load API key
with open(WD / "config/private.json", "r") as f:
    private = json.load(f)

api_keys = {k: v["key"] for k, v in private["API"].items()}


def get_timestamp(fmt="%Y_%m_%d-%H_%M_%S"):
    return time.strftime(fmt, time.localtime())


def clean_model_id(model_id):
    model_id = model_id.replace("/", "--")
    model_id = model_id.replace(":", "-")
    return model_id


def get_missing_items(
    model_resps: pd.DataFrame,
    df_prompts: pd.DataFrame,
    identifier: str = "item_id",
) -> pd.DataFrame:
    item_ids_resps = set(model_resps[identifier])
    item_ids_prompts = set(df_prompts[identifier])

    missing_item_ids = list(item_ids_prompts - item_ids_resps)

    print(f"{len(missing_item_ids)} missing items")

    missing_items = df_prompts.query(f"{identifier} in @missing_item_ids")

    return missing_items


def main(
    df_prompts: pd.DataFrame,
    models: list,
    completion_parameters: Optional[dict] = None,
    save: bool = True,
    save_dir: Optional[Union[str, Path]] = None,
    task_type: str = "text",
    err_threshold: int = 10,
    identifier: str = "item_id",
):
    assert task_type in ["text", "image"]

    open_router = OpenRouter(api_key=api_keys["openrouter"])

    if completion_parameters is None:
        completion_parameters = {}
    # open_router.rate_limit['interval']
    # open_router.rate_limit['requests']

    # sequence_prompts = df_prompts["prompt"].tolist()

    # models = [
    #     "google/gemini-2.0-flash-thinking-exp:free",
    # ]

    timestamp = get_timestamp()

    if save_dir is None:
        save_dir = EXPORT_DIR / timestamp
    else:
        save_dir = Path(save_dir)

    save_dir.mkdir(exist_ok=True, parents=True)

    pb_models = tqdm(models)
    pb_prompts = tqdm(total=len(df_prompts))

    models_resps_df = []
    models_errors = []

    for model_id in pb_models:
        model_errors = []

        pb_models.set_description(model_id)
        model_id_str = clean_model_id(model_id)
        model_resps = []

        try:
            for row_idx, row in df_prompts.iterrows():
                # print(f"{'-' * 30}\n{row_idx}\n{row['prompt']}\n{'-' * 30}\n")

                completion = open_router.chat_completion(
                    model_id=model_id,
                    messages=[{"role": "user", "content": row["prompt"]}],
                    **completion_parameters,
                )

                # * Check if the 'error' attribute exists and handle accordingly
                if hasattr(completion, "error") and completion.error is not None:
                    print(
                        f"Error for prompt #{row_idx} (identifier: {row[identifier]}): {completion.error}"
                    )

                    model_errors.append(
                        [
                            model_id,
                            row[identifier],
                            completion.error,
                        ]
                    )

                    if len(model_errors) > err_threshold:
                        print("Too many errors, breaking")
                        break
                    else:
                        continue

                resp = completion.choices[0].message.content
                model_resps.append([row[identifier], row["masked_idx"], resp])

                pb_prompts.update(1)

            pb_prompts.reset()

        except Exception as e:
            # print("AN ERROR OCCURED, details below:")
            # raise e
            print(f"Error occurred: {type(e).__name__}")
            print(f"Details: {str(e)}")
            print("\nTraceback:")
            print(traceback.format_exc())
            raise

        finally:
            model_resps_df = pd.DataFrame(
                model_resps, columns=[identifier, "masked_idx", "response"]
            )
            model_resps_df.insert(0, "model_id", model_id)
            models_resps_df.append(model_resps_df)

            if save:
                fpath = save_dir / f"{model_id_str}-responses.csv"
                model_resps_df.to_csv(fpath, index=False)

            # if len(errors) > 0:
            models_errors.append(model_errors)
            # save_dir.mkdir(exist_ok=True, parents=True)

            # df_errors = pd.DataFrame(
            #     errors, columns=["model_id", identifier, "error"]
            # )

            # fpath = save_dir / f"errors-{model_id_str}.csv"
            # df_errors.to_csv(fpath, index=False)

    models_resps_df = pd.concat(models_resps_df, ignore_index=True)
    models_resps_df.reset_index(drop=True, inplace=True)

    return models_resps_df, model_errors


def run_model_on_missing_items(
    model_id: str,
    model_resps: pd.DataFrame,
    df_prompts: pd.DataFrame,
    save_dir: Union[str, Path],
    completion_parameters: Optional[dict] = None,
    missing_items: Optional[pd.DataFrame] = None,
    identifier: str = "item_id",
):
    save_dir = Path(save_dir)

    if missing_items is None:
        missing_items = get_missing_items(model_resps, df_prompts)

    new_responses = main(
        missing_items,
        [model_id],
        completion_parameters,
        save=False,
        identifier=identifier,
    )

    responses = model_resps.copy()
    responses = pd.concat([responses, new_responses], ignore_index=True)

    df_prompts.reset_index(names="original_order", inplace=True)

    original_order = df_prompts[["original_order", identifier]]

    responses = responses.merge(original_order, on=identifier, how="left")
    responses.sort_values("original_order", inplace=True)

    model_id_cleaned = clean_model_id(responses["model_id"][0])
    responses.to_csv(
        save_dir / f"{model_id_cleaned}-responses-COMPLETE.csv", index=False
    )


if __name__ == "__main__":
    models = [
        # "anthropic/claude-3.5-sonnet",
        # "anthropic/claude-3.5-sonnet-20240620",
        # "deepseek/deepseek-chat",
        # "deepseek/deepseek-r1",
        # "deepseek/deepseek-r1-distill-llama-70b",
        # "deepseek/deepseek-r1-distill-qwen-1.5b",
        # "deepseek/deepseek-r1-distill-qwen-14b",
        # "deepseek/deepseek-r1-distill-qwen-32b",
        "deepseek/deepseek-r1",
        # # "google/gemini-2.0-flash-exp:free",
        # # "google/gemini-2.0-flash-thinking-exp:free",
        # "google/gemini-flash-1.5 ",
        # "google/gemma-2-27b-it",
        # "google/gemma-2-9b-it",
        # "meta-llama/llama-3.1-405b-instruct",
        # "meta-llama/llama-3.2-90b-vision-instruct",
        # "meta-llama/llama-3.2-11b-vision-instruc"t
        # "microsoft/phi-4",
        # "openai/gpt-4o-2024-11-20'",
        # "openai/gpt-4o-mini-2024-07-18",
        # "openai/o1-mini-2024-09-12",
        # "openai/o1-preview-2024-09-12",
        # "qwen/qwen-14b-chat",
        # "openai/o3-mini",
        # "meta-llama/llama-3.2-3b-instruct",
        # "meta-llama/llama-3.3-70b-instruct",
        # "minimax/minimax-01",
        # "mistralai/mistral-small-24b-instruct-2501",
        # "qwen/qwen-2.5-72b-instruct",
        # "qwen/qwen-2.5-7b-instruct",
        # "qwen/qwq-32b-preview",
    ]

    prompts_file = WD / "sequence_prompts/sequence_prompts-masked_idx(7).csv"
    df_prompts = pd.read_csv(prompts_file)

    # df_prompts = df_prompts.sample(2)

    completion_parameters = dict(
        temperature=0,
        tool_choice=None,
        seed=0,
        # max_tokens=500,
    )

    save_dir = EXPORT_DIR  # / prompts_file.stem

    models_resps, errors = main(
        df_prompts,
        models,
        completion_parameters,
        # identifier="unique_id",
    )

    # * ---- recover missing items ----
