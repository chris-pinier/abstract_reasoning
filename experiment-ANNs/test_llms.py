import json
import numpy as np
import pandas as pd
import os
import together
import openai
from openai import OpenAI
import anthropic
import json
from pathlib import Path
import re
from tabulate import tabulate
import time
from tqdm.auto import tqdm
from typing import List, Dict
import seaborn as sns
import matplotlib.pyplot as plt
from typing import NamedTuple

# from groq import Groq
# import ollama


# import pyperclip
rate_limits_info = {
    "openai": "Tier 1: 500 requests per minute, 300,000 Tokens per minute",
}

rate_limits = {
    "togetherai": together.error.RateLimitError,
    "anthropic": anthropic.RateLimitError,
    "chatgpt": openai.RateLimitError,
}
rate_limits = tuple(rate_limits.values())

os.chdir(Path(__file__).resolve().parents[0])  # ! TEMPORARY

wd = Path(__file__).parent

if not wd.parent.is_dir() and not wd.parent.name == "experiment1":
    raise FileNotFoundError("Please run this script from the 'experiment1' directory.")

config_file = Path(wd / "config/experiment_config.json")
with open(config_file, "r") as f:
    main_config = json.load(f)

# * load API key from environment variable
with open(wd / "config/private.json", "r") as f:
    private = json.load(f)

api_keys = {k: v["key"] for k, v in private["API"].items()}

for provider, api_info in private["API"].items():
    var_name = api_info.get("var_name")
    if var_name and var_name not in os.environ:
        os.environ[api_info["var_name"]] = api_info["key"]

together.api_key = api_keys["togetherai"]

api_funcs = {
    "together": together.Complete.create,
    "anthropic": anthropic.Anthropic(api_key=api_keys["anthropic"]).messages.create,
    "openai": OpenAI(api_key=api_keys["openai"]).chat.completions.create,
    "google": None,
    "groq": Groq(api_key=api_keys["groq"]).chat.completions.create,
    "ollama": ollama.chat,
}

inference_config = {
    "together": {
        "prompt": "",
        "model": "",
        "max_tokens": 10,
        # "stop": ,
        "temperature": 0.0,
        # "top_p": ,
        # "top_k": ,
        # "repetition_penalty": ,
        # "logprobs": ,
        # "api_key": ,
        # "cast": ,
        # "safety_model": ,
    },
    "anthropic": {
        "messages": [],
        "model": "",
        "max_tokens": 10,
        # 'metadata': ,
        # 'stop_sequences': ,
        # 'stream': ,
        # 'system': ,
        "temperature": 0.0,
        # 'top_k': ,
        # 'top_p': ,
        # 'extra_headers': ,
        # 'extra_query': ,
        # 'extra_body': ,
        # 'timeout': ,
    },
    "openai": {
        "model": "",
        # "frequency_penalty":,
        # "function_call":,
        # "functions":,
        # "logit_bias":,
        # "logprobs":,
        "max_tokens": 10,
        # "n":,
        # "presence_penalty":,
        # "response_format":,
        # "seed":,
        # "stop":,
        # "stream":,
        "temperature": 0,
        # "tool_choice":,
        # "tools":,
        # "top_logprobs":,
        # "top_p":,
        # "user":,
        # "extra_headers":,
        # "extra_query":,
        # "extra_body":,
        # "timeout":,
    },
    "ollama": {
        "model": "",
        "messages": [],
        "options": {
            # "mirostat":,
            # "mirostat_eta":,
            # "mirostat_tau":,
            # "num_ctx":,
            # "repeat_last_n":,
            # "repeat_penalty":,
            "temperature": 0.0,
            "seed": 123,
            # "stop":,
            # "tfs_z":,
            # "num_predict":,
            # "top_k":,
            # "top_p":,
        },
    },
}


def timestamp(mode="datetime"):
    current_time = time.localtime()
    return {
        "datetime": time.strftime("%Y_%m_%d-%H_%M_%S", current_time),
        "date": time.strftime("%Y_%m_%d", current_time),
        "time": time.strftime("%H_%M_%S", current_time),
    }.get(
        mode, "datetime"
    )  # ! Does not raise an error if mode is not found


def totable(l):
    print(tabulate(l, headers="keys", tablefmt="fancy_outline", showindex=True))


def prepare_prompt(
    row,
    instructions,
    seq_cols,
    choice_cols,
    unique_icons=None,
    prompt_format: str = None,
):
    sequence = row.loc[seq_cols].tolist()
    # mapped_sequence = [str(unique_icons[icon]) for icon in sequence]

    choices = row.loc[choice_cols].tolist()
    # solution = sequence[row.loc["maskedImgIdx"]]
    sequence[row.loc["maskedImgIdx"]] = "?"
    # mapped_sequence[row.loc["maskedImgIdx"]] = "?"

    sequence_string = "Sequence: " + " ".join(sequence)
    # choices_string = "Choices: " + " ".join(choices)
    choices_string = "Options: " + " ".join(choices)
    sequence_prompt = f"\n{sequence_string}\n{choices_string}"
    sequence_prompt = instructions + sequence_prompt

    if prompt_format is not None:
        sequence_prompt = prompt_format.replace("{prompt}", sequence_prompt)

    return sequence_prompt  # , solution


def api_model_call(
    func_call: callable,
    prompt_args: dict,
    wait_time: float = 2.0,
    n_tries: int = 3,
    error_value=None,
    verbose: bool = False,
):
    while n_tries > 0:
        try:
            return func_call(**prompt_args)

        except rate_limits as e:
            if verbose:
                txt = (
                    f"Rate limit reached. Waiting {wait_time} "
                    "seconds before trying again..."
                )
                print(f"{txt}\n")
                print(f"Details: {e}\n\n")

            time.sleep(wait_time)
            n_tries -= 1
        # * General exception handling for other unexpected errors
        except Exception as e:
            print("Unexpected error occurred. Please try again later.")
            print(f"Error: {e}\n\n")
            print("Prompt args:", prompt_args)
            return error_value

    if n_tries == 0:
        print(
            "Rate limit reached and maximum number of attemps reached. ",
            "Please try again later.",
        )
        return "break"


def access_output(api: str, output):
    commands = {
        "anthropic": "output.content[0].text",
        "openai": "output.choices[0].message.content",
        "together": "output['output']['choices'][0]['text']",
        "groq": "output.choices[0].message.content",
        "ollama": "output['message']['content']",
    }
    command = commands.get(api)
    if command is None:
        raise ValueError("Cannot access output, invalid API name.")
    else:
        return eval(command)


def main(api, models_list, sequences, wait_time: float = 1.5, error_val="NA"):
    #! TEMP
    wait_time: float = 0.15
    error_val = "NA"
    # models_list = together.Models.list()
    # models_list = [m for m in models_list if "llama 3" in m.get("display_name").lower()]
    # models_list = [m["name"] for m in models_list]
    # api = "together"
    api = "ollama"
    models_list = ["phi3"]  # gpt-4-turbo-2024-04-09
    sequences = pd.read_csv(wd / "config/sequences-format[names].csv")
    #! TEMP

    sequences = sequences.sample(frac=1).reset_index(drop=True)

    seq_cols = [col for col in sequences.columns if "figure" in col]
    choice_cols = [col for col in sequences.columns if re.search(r"choice\d{1,2}", col)]

    with open(wd / f"config/instructions.txt", "r") as f:
        instructions = f.read()

    results_cols = ["combinationID", "answer", "solution"]
    n_sequences = sequences.shape[0]

    api_func = api_funcs[api]
    default_params = inference_config[api]

    if api == "together":
        prompt_formats = get_together_prompt_formats(models_list)
    else:
        prompt_formats = {}

    pbar1 = tqdm(models_list)
    pbar2 = tqdm(total=n_sequences)

    for model in pbar1:
        params = default_params.copy()
        params["model"] = model

        prompt_format = prompt_formats.get(model)

        for masked_idx in [0, 4, 7]:
            # for masked_idx in [7]:
            sequences["maskedImgIdx"] = masked_idx
            results = np.zeros((n_sequences, len(results_cols)), dtype="object")

            for idx, row in sequences.iterrows():
                start_time = time.time()

                combinationID = row["combinationID"]
                solution = row[seq_cols].iloc[row["maskedImgIdx"]]

                prompt = prepare_prompt(
                    row,
                    instructions,
                    seq_cols,
                    choice_cols,
                    prompt_format=prompt_format,
                )

                if api == "together":
                    # TODO: This is no longer valid, together.chat.completions.create
                    # TODO: uses the same format as the other APIs
                    params["prompt"] = prompt
                else:
                    params["messages"] = [{"role": "user", "content": prompt}]

                output = api_model_call(
                    func_call=api_func,
                    prompt_args=params,
                    wait_time=2.0,
                    n_tries=3,
                    error_value=error_val,
                )

                if output == "break":
                    break
                elif output == error_val:
                    print(f"Error with model {model}\n")
                    results[idx, :] = [combinationID, error_val, solution]
                else:
                    text = access_output(api=api, output=output)
                    results[idx, :] = [combinationID, text, solution]

                end_time = time.time()

                if (duration := end_time - start_time) < wait_time:
                    time.sleep(wait_time - duration)
                pbar2.update()

            pbar2.reset()

            results = pd.DataFrame(results, columns=results_cols)
            results["masked_idx"] = masked_idx
            results["model"] = model

            model_name = model.replace("/", "_")
            fname = f"api[{api}]-{model_name}-masked_idx[{masked_idx}]-{timestamp()}"

            results.to_csv(wd / f"results/{fname}.csv", index=False)

        pbar2.close()


def get_together_prompt_formats(models_names: list, save=False):
    m_names_lower = [m.lower() for m in models_names]
    models_list = together.Models.list()
    models_list = [m for m in models_list if m["name"].lower() in m_names_lower]

    prompt_formats = {}
    for model_info in models_list:
        # * Safeguard against 'config' being None
        config = model_info.get("config", {})
        if config and config.get("prompt_format"):
            prompt_formats[model_info["name"]] = config["prompt_format"]

    if save:
        fpath = wd / "config/prompt_formats.json"

        with open(fpath, "w") as f:
            json.dump(prompt_formats, f, indent=4)

    return prompt_formats


def get_together_unavailable_models(
    models_dict: Dict, wait_time: float = 2.0
) -> List[str]:
    unavailable_models = {}
    for model, model_info in tqdm(models_dict.items()):
        t1 = time.time()

        try:
            together.Complete.create(prompt="test prompt", model=model, max_tokens=1)

        except Exception as e:
            print(f"Model {model} is not available.\nError: {e}\n\n")
            unavailable_models[model] = str(e)

        duration = time.time() - t1

        if duration < wait_time:
            time.sleep(wait_time - duration)

    fpath = wd / "config/unavailable_models.json"

    with open(fpath, "w") as f:
        json.dump(unavailable_models, f, indent=4)

    return unavailable_models


def run_hf_models(model, prompt):
    # ! TEMP
    model = "toshi456/llava-jp-1.3b-v1.1"
    # ! TEMP
    # * More info at: https://huggingface.co/docs/api-inference/en/quicktour

    import requests

    headers = {"Authorization": f"Bearer {api_keys['huggingface']}"}

    API_URL = f"https://api-inference.huggingface.co/models/{model}"

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    data = query({"inputs": prompt})

    return data


def preprocess_answer(answer):
    chars_to_remove = [
        r"\n+",
        r"\t+",
        r"\s+",
        r"\.",
        r"\[",
        r"\]",
        r"\{",
        r"\}",
        r"\)",
        r"\(",
        r"\"",
        r"\'",
        "<",
        ">",
        ";",
        ",",
        # ":",
        # "?"
    ]
    chars_to_remove = "|".join(chars_to_remove)
    chars_to_remove = re.compile(chars_to_remove)

    # answer = re.sub(chars_to_remove, " ", str(answer))
    # answer = re.sub(r"\s+", " ", answer)
    # answer = re.sub(r"Answer:?\s?", " ", answer, flags=re.IGNORECASE)
    # answer = answer.strip()
    # answer = re.findall(r"\S+", answer, flags=re.IGNORECASE)

    # if len(answer) > 1:
    #     answer = f"INVALID: {' '.join(answer)}"
    # elif len(answer) == 0:
    #     answer = None
    # else:
    #     answer = answer[0]

    answer = re.sub(chars_to_remove, " ", str(answer))
    answer = re.findall(r"Answer:\s?\S+", answer, flags=re.IGNORECASE)
    if len(answer) > 0:
        answer = re.sub(r"Answer:\s?", "", answer[0], flags=re.IGNORECASE)
    else:
        answer = "invalid"

    return answer
    # if len(answer.split(" ")) > 2:
    # return "invalid"
    # else:
    # str_search = re.compile(r"Answer:\s*(\S+)", re.IGNORECASE)
    # result = re.findall(str_search, answer)
    # return result[0] if result else "invalid"


def results_analysis(files, sequences):
    # file = Path("./ANNs/results/Austism_chronos-hermes-13b-masked_idx[7].csv")
    wd = Path(__file__).parent
    files = [f for f in Path(wd / "results/batch3").glob("*.csv")]
    # files = [f for f in files if "masked_idx[7]" in f.stem]

    models = [f.stem.split("-masked_idx")[0] for f in files]
    set(sorted(models))
    # files = [f for f in Path("./ANNs/results/").glob("*.csv")]
    sequences = pd.read_csv(wd / "config/sequences-format[names].csv")

    # dfs = pd.concat([pd.read_csv(f) for f in files], axis=0)
    # dfs['model'].unique().shape
    # dfs.groupby(['model'])['masked_idx'].nunique()

    results_cleaned = pd.DataFrame()
    cols = ["model", "masked_idx", "accuracy"]
    results_acccuracy = pd.DataFrame(columns=cols)
    results_by_pattern = pd.DataFrame()

    for file in files:
        # file = "./ANNs/results/chatgpt-masked_idx[7]-2024_04_04-11_04_52.csv"
        # file = Path(file)
        model_name, info = file.stem.split("-masked_")
        masked_idx = int(re.search(r"idx\[(\d+)\]", info)[1])

        # print(model_name, masked_idx)

        df = pd.read_csv(file)
        df = pd.merge(df, sequences[["combinationID", "pattern"]], on="combinationID")

        # df['answer'].apply(preprocess_answer)
        # df.drop(columns=["clean_answer", "correct"], inplace=True)

        df.insert(2, "clean_answer", df["answer"].apply(preprocess_answer))

        df["clean_answer"] = df["clean_answer"].str.lower()
        df["solution"] = df["solution"].str.lower()
        df.insert(4, "correct", df["clean_answer"] == df["solution"])
        results_cleaned = pd.concat([results_cleaned, df])

        # print(f"\n\n{model_name}:")
        # display(df.iloc[:, 1:5].head(55))

        pct_correct = (df["correct"].sum() / df.shape[0]) * 100
        # print(f"{model_name}: {pct_correct:.2f} % correct answers.")
        model_accuracy = pd.DataFrame(
            [[model_name, masked_idx, pct_correct]], columns=cols
        )

        pattern_res = df.groupby("pattern")["correct"].sum()
        pattern_res.name = model_name
        results_by_pattern = pd.concat([results_by_pattern, pattern_res], axis=1)

        results_acccuracy = pd.concat([results_acccuracy, model_accuracy])

    results_cleaned = results_cleaned.reset_index(names="trial_n")

    def temp_mathpsych_pres(results_cleaned):
        results_cleaned.query('model.str.contains("claude")').groupby(
            ["model", "masked_idx"]
        )["correct"].sum()
        results_cleaned = results_cleaned.query(
            'not model.str.contains("claude", case=False)'
        )

        # * make sure that all selected models have the same number of responses
        assert (
            len(
                results_cleaned.query("masked_idx in [0,4,7]")["model"]
                .value_counts()
                .unique()
            )
            == 1
        )

        results_cleaned

    results_acccuracy = results_acccuracy.reset_index(drop=True)
    results_acccuracy.sort_values(
        ["masked_idx", "accuracy"],
        ascending=False,
        inplace=True,
    )

    data = results_acccuracy.query("masked_idx in [0,4,7]")  # & accuracy>=30")
    fig, ax = plt.subplots(figsize=(14, 11), dpi=300)
    sns.barplot(
        data=data,
        x="accuracy",
        y="model",
        ax=ax,
        hue="masked_idx",
        palette=sns.color_palette(),
    )
    ax.grid(axis="x")
    title = f"Accuracy by model"
    ax.set_title(title)
    ax.set_xlim(0, 100)
    plt.tight_layout()
    plt.show()
    plt.close()

    # for masked_idx in [0,4,7]:#results["masked_idx"].unique():
    #     fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    #     data = results.query("masked_idx==@masked_idx & accuracy>=30")
    #     sns.barplot(
    #         data=data,
    #         x="accuracy",
    #         y="model",
    #         ax=ax,
    #         hue="masked_idx",
    #         palette=sns.color_palette(),
    #     )
    #     title = f"Accuracy by model - Masked index: {masked_idx}"
    #     ax.grid(axis="x")
    #     ax.set_title(title)
    #     ax.set_xlim(0, 100)
    #     plt.tight_layout()
    #     plt.show()

    # results.groupby(["model", ""])
    results_by_pattern = results_by_pattern.T
    results_by_pattern = (
        results_by_pattern.reset_index(names="model")
        .melt(id_vars="model", var_name="pattern", value_name="score")
        .sort_values("model")
    )

    export_dir = wd / "results/Analysis"
    export_dir.mkdir(exist_ok=True)

    results_cleaned.to_excel(export_dir / "results_cleaned.xlsx", index=False)

    for res in ["results_cleaned", "results_acccuracy", "results_by_pattern"]:
        fpath = export_dir / f"{res}"
        res = eval(res)
        res.to_csv(f"{fpath}.csv", index=False)

    # * ########### PLOTTING ############
    n_by_pattern = 6
    n_models = results_acccuracy["model"].nunique()

    figParams = NamedTuple("fig_params", [("figsize", tuple), ("dpi", int)])
    fig_params = figParams((10, 6), 300)

    def plots():
        # * ALL MODELS
        fig, ax = plt.subplots(figsize=(6, 12), dpi=fig_params.dpi)
        # data = results.query("model.str.lower().str.contains('qwen')")
        data = results
        g = sns.barplot(data=data, x="accuracy", y="model", ax=ax)  # hue="masked_idx")
        # for p in g.patches:
        #     width = p.get_width()
        #     ax.text(
        #         width + 0.1,
        #         p.get_y() + p.get_height() / 2.0,
        #         f"{width:.2f}%",
        #         ha="center",
        #         va="center",
        #     )
        title = "Accuracy by model"
        ax.set_title(title)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Accuracy (%)")
        ax.grid(axis="x")
        fname = title.replace(" ", "_").lower()
        plt.savefig(
            f"./ANNs/results/Analysis/{fname}.png",
            bbox_inches="tight",
            dpi=fig_params.dpi,
        )
        plt.show()
        plt.close()

        # * Highest performing models
        fig, ax = plt.subplots(figsize=(7, 12), dpi=fig_params.dpi)
        # data = results.query("model.str.lower().str.contains('qwen')")
        data = results.query("accuracy>=30")
        g = sns.barplot(data=data, x="accuracy", y="model", ax=ax)  # hue="masked_idx")
        # for p in g.patches:
        #     width = p.get_width()
        #     ax.text(
        #         width + 0.1,
        #         p.get_y() + p.get_height() / 2.0,
        #         f"{width:.2f}%",
        #         ha="center",
        #         va="center",
        #     )
        title = "Accuracy by model (top performers)"
        ax.set_xticks(range(0, 101, 10))
        ax.set_title(title)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Accuracy (%)")
        ax.grid(axis="x", linestyle="--")
        fname = title.replace(" ", "_").lower()
        plt.savefig(
            f"./ANNs/results/Analysis/{fname}.png",
            bbox_inches="tight",
            dpi=fig_params.dpi,
        )
        plt.show()
        plt.close()

        # *
        fig, ax = plt.subplots(figsize=(12, 8), dpi=fig_params.dpi)
        data = results_by_pattern.groupby("pattern")["score"].sum()
        data = (data / (n_models * n_by_pattern)) * 100
        sns.barplot(data=data, orient="h", ax=ax)  # hue="masked_idx")
        title = "Accuracy by pattern (Across all models)"
        ax.set_title(title)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Accuracy (%)")
        fname = title.replace(" ", "_").lower()
        plt.savefig(
            f"./ANNs/results/Analysis/{fname}.png",
            bbox_inches="tight",
            dpi=fig_params.dpi,
        )
        plt.show()
        plt.close()

        # *
        fig, ax = plt.subplots(figsize=(12, 8), dpi=fig_params.dpi)
        data = results_by_pattern.groupby("pattern")["score"].sum()
        data = (data / (n_models * n_by_pattern)) * 100
        sns.barplot(data=data, orient="h", ax=ax)  # hue="masked_idx")
        title = "Accuracy by pattern (Across all models)"
        ax.set_title(title)
        # ax.set_xlim(0, 100)
        ax.set_xlabel("Accuracy (%)")
        fname = title.replace(" ", "_").lower() + "-zoomed"
        plt.savefig(
            f"./ANNs/results/Analysis/{fname}.png",
            bbox_inches="tight",
            dpi=fig_params.dpi,
        )
        plt.show()
        plt.close()

        # *
        fig, ax = plt.subplots(figsize=(12, 8), dpi=fig_params.dpi)
        accuracy_threshold = 46
        selected_models = results.query("accuracy>=@accuracy_threshold")["model"]
        data = results_by_pattern.query("model.isin(@selected_models)")
        data = data.groupby("pattern")["score"].sum()
        # data = data.groupby(["model", "pattern"])["score"].sum()
        data = (data / (len(selected_models) * n_by_pattern)) * 100
        sns.barplot(data=data, orient="h", ax=ax)  # hue="masked_idx")
        title = f"Accuracy by pattern (models with overall accuracy >= {accuracy_threshold}%)"
        ax.set_title(title)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Accuracy (%)")
        fname = title.replace(" ", "_").lower()
        plt.savefig(
            f"./ANNs/results/Analysis/{fname}.png",
            bbox_inches="tight",
            dpi=fig_params.dpi,
        )
        plt.show()
        plt.close()

        # *
        # fig, ax = plt.subplots(figsize=(12, 8), dpi=fig_params.dpi)
        # data = results_by_pattern.groupby(["pattern"])["score"].mean()
        # data = (data / n_by_pattern) * 100
        # sns.barplot(data=data, orient="h", ax=ax)  # hue="masked_idx")
        # title = "Accuracy by pattern (Avg across models)"
        # ax.set_title(title)
        # ax.set_xlim(0, 100)
        # ax.set_xlabel("Accuracy (%)")
        # plt.show()
        # plt.close()

        # *
        for model in results_by_pattern["model"].unique():
            model_results = results_by_pattern[results_by_pattern["model"] == model]
            model_results = model_results.sort_values("pattern")
            fig, ax = plt.subplots(figsize=(10, 6), dpi=fig_params.dpi)
            sns.barplot(data=model_results, x="score", y="pattern", ax=ax)
            title = f"{model} - Correct answers by pattern"
            ax.set_title(title)
            fname = title.replace(" ", "_").lower()
            plt.show()
            plt.close()

    def model_comparison(model_names: list):
        # combined_res = results_by_pattern.query("model.isin([@model1, @model2])").copy()
        combined_res = results_by_pattern.query("model.isin(@model_names)").copy()
        combined_res["score"] = (combined_res["score"] / n_by_pattern) * 100
        combined_res.sort_values(["pattern", "model"], inplace=True)
        fig, ax = plt.subplots(figsize=(10, 6), dpi=fig_params.dpi)
        sns.barplot(data=combined_res, x="score", y="pattern", hue="model", ax=ax)
        title = f"Correct predictions by pattern"
        ax.set_title(title)
        ax.set_title(title)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Accuracy (%)")
        ax.legend(title="Model", bbox_to_anchor=(1, 1))
        plt.show()
        plt.close()

        return combined_res, fig

    comparison_list = [
        ["chatgpt", "claude"],
        ["chatgpt", "claude", "Qwen_Qwen1.5-72B-Chat"],
        ["Qwen_Qwen1.5-72B-Chat", "NousResearch_Nous-Hermes-2-Yi-34B"],
        [
            "Qwen_Qwen1.5-72B-Chat",
            "NousResearch_Nous-Hermes-2-Yi-34B",
            "garage-bAInd_Platypus2-70B-instruct",
        ],
    ]

    for model_names in comparison_list:
        data_comp, fig = model_comparison(model_names)


# ! OLD


def compare_models():
    import seaborn as sns

    results = [f for f in Path("./ANNs/results/masked_idx[4]").glob("*.txt")]
    results = [
        pd.read_table(f, names=[f.stem.replace("_results", "")]) for f in results
    ]
    results = pd.concat(results, axis=1)
    results["solution"] = solutions
    results.insert(1, "gpt_correct", results["gpt"] == results["solution"])
    results.insert(3, "claude_correct", results["claude"] == results["solution"])

    results["gpt_correct"].sum()
    results["claude_correct"].sum()

    sequences = pd.concat([sequences, results.drop("solution", axis=1)], axis=1)

    gpt_correct = sequences.groupby("pattern")["gpt_correct"].sum()
    claude_correct = sequences.groupby("pattern")["claude_correct"].sum()

    df_analysis = pd.concat(
        [gpt_correct, claude_correct], axis=1, keys=["gpt", "claude"]
    )
    df_analysis["gpt"] == gpt_correct
    df_analysis["claude"] == claude_correct

    melted_df = df_analysis.reset_index().melt(
        id_vars="pattern", var_name="model", value_name="score"
    )

    fname = input("filename:")
    sequences.to_csv(f"./ANNs/results/{fname}.csv", index=False)

    fig, ax = plt.subplots()
    sns.barplot(data=melted_df, x="score", y="pattern", hue="model", ax=ax)
    plt.legend(bbox_to_anchor=(1, 1))
    ax.set_title("Correct predictions by pattern")


def compare_pattern_by_model(results):
    results_dir = Path("./ANNs/results")
    results_files = [f for f in results_dir.rglob("pattern_results*")]

    results = {
        re.findall("idx\[(\d)\]", f.stem)[0]: pd.read_csv(f) for f in results_files
    }

    combined_results = pd.DataFrame()
    for masked_idx, res in results.items():

        melted_df = res.reset_index(drop=True).melt(
            id_vars="pattern", var_name="model", value_name="score"
        )
        melted_df["masked_idx"] = masked_idx
        combined_results = pd.concat([combined_results, melted_df], axis=0)

    for model in combined_results["model"].unique():
        model_results = combined_results[combined_results["model"] == model]

        fig, ax = plt.subplots()
        sns.barplot(data=model_results, x="score", y="pattern", hue="masked_idx", ax=ax)
        plt.legend(bbox_to_anchor=(1, 1))
        ax.set_title(f"Correct predictions by pattern - {model}")
        plt.show()
        plt.close()

        fig, ax = plt.subplots()
        n = 126
        sns.barplot((model_results.groupby("masked_idx")["score"].sum() / n) * 100)
        ax.set_title(f"Correct predictions by masked index - {model}")
        ax.set_ylabel("Score (%)")
        ax.set_xlabel("Masked Image Index")
        ax.set_ylim(0, 100)
        ax.grid(axis="y")
        plt.show()
        plt.close()

    fig, ax = plt.subplots()
    n = 126
    data = combined_results.groupby(["model", "masked_idx"])["score"].sum()
    data = data.reset_index()
    sns.barplot(data=data, x="model", y="score", hue="masked_idx")
    ax.set_title(f"Correct predictions by masked index - {model}")
    ax.set_ylabel("Score (%)")
    ax.set_xlabel("Masked Image Index")
    ax.set_ylim(0, 100)
    ax.grid(axis="y")
    plt.legend(bbox_to_anchor=(1, 1))
    plt.show()
    plt.close()


def revert_sequences_to_old_format(sequences):
    sequences = pd.read_csv("./ANNs/config/sequences-format[names].csv")
    sequences.rename(
        columns={"figure8": "solution", "combinationID": "itemid"}, inplace=True
    )
    unique_icons = sequences.iloc[:, 1:13].stack().nunique()

    wrong_inds = []
    for idx, row in sequences.iterrows():
        sequence = row.iloc[1:9]
        choices = row.iloc[9:13]
        if not sequence["solution"] in choices.values:
            wrong_inds.append(idx)
            c = [c for c in choices if c not in sequence.values]
            if c:
                c = np.random.choice(c)
            else:
                c = np.random.choice(choices)

            idx_c = 9 + list(choices).index(c)

            sequences.iloc[idx, idx_c] = row["solution"]

    # sequences.iloc[wrong_inds, :]
    sequences.drop(columns="maskedImgIdx", inplace=True)

    sequences.drop(
        index=sequences.query("pattern in ['ABCDEFGH','ABCDEFAB']").index, inplace=True
    )
    sequences["seq_order"] = sequences["seq_order"].astype(str)
    sequences["choice_order"] = sequences["choice_order"].astype(str)

    new_seq_order = sequences.loc[sequences["seq_order"].str.len() < 8]["seq_order"]
    sequences.loc[new_seq_order.index, "seq_order"] = [
        "0" + row for row in new_seq_order.tolist()
    ]

    new_choice_order = sequences.loc[sequences["choice_order"].str.len() < 4][
        "choice_order"
    ]
    sequences.loc[new_choice_order.index, "choice_order"] = [
        "0" + row for row in new_choice_order.tolist()
    ]

    sequences[sequences["seq_order"].str.len() < 8]
    sequences[sequences["choice_order"].str.len() < 4]
    sequences.groupby("pattern").sample(n=4)
    new_sequences = sequences.groupby("pattern").sample(n=4).copy()

    new_sequences["seq_order"] = [f"'{v}'" for v in new_sequences["seq_order"].values]
    new_sequences["choice_order"] = [
        f"'{v}'" for v in new_sequences["choice_order"].values
    ]

    wrong_inds = []
    for idx, row in new_sequences.iterrows():
        sequence = row.iloc[1:9]
        choices = row.iloc[9:13]
        if not sequence["solution"] in choices.values:
            wrong_inds.append(idx)

    assert len(wrong_inds) == 0

    new_sequences.to_csv(
        f"sequences-{timestamp()}.csv",
        # "sequences-2024_04_18.csv",
        index=False,
    )


def format_instructions(unique_icons):
    unique_icons_str = "\n".join([f"- {k}: {v}" for k, v in unique_icons.items()])
    np.random.seed(32)

    with open("./ANNs/config/instructions.txt", "r") as f:
        instructions = f.read()

    el1, el2 = "smile", "eye"
    example_seq = [el1, el2, el1, el2, el1, el2]

    example_choices = [i for i in unique_icons.keys() if i not in [el1, el2]]
    example_choices = np.random.choice(example_choices, 2, replace=False).tolist()
    example_choices += [el1, el2]
    np.random.shuffle(example_choices)
    example_solution = example_seq[-1]

    mapped_seq = [str(unique_icons[icon]) for icon in example_seq]
    example_seq[-1] = "?"
    mapped_seq[-1] = "?"
    mapped_choices = [str(unique_icons[icon]) for icon in example_choices]
    mapped_solution = str(unique_icons[example_solution])

    example_seq = " ".join(example_seq)
    example_choices = " ".join(example_choices)
    mapped_seq = " ".join(mapped_seq)
    mapped_choices = " ".join(mapped_choices)

    return instructions.format(
        unique_icons_str,
        example_seq,
        example_choices,
        mapped_seq,
        mapped_choices,
        example_solution,
        el1,
        el2,
        mapped_solution,
        example_solution,
    )


def seq_for_chat_interface(model_name, sequences=None, masked_idx=7):
    import pyperclip

    with open("./ANNs/config/instructions_chat_gpt.txt", "r") as f:
        instructions = f.read()

    sequences = pd.read_csv("./ANNs/config/sequences-format[names].csv")
    sequences = sequences.sample(frac=1).reset_index(drop=True)
    # masked_idx = 7
    sequences["maskedImgIdx"] = masked_idx

    seq_cols = [col for col in sequences.columns if "figure" in col]
    choice_cols = [col for col in sequences.columns if re.search("choice\d{1,2}", col)]

    # sequences.iloc[:, 1:13].stack().unique()
    cols = ["combinationID", "solution", "pattern"]
    df_solutions = pd.DataFrame(columns=cols)
    text = instructions
    for idx, row in sequences.iterrows():
        combinationID = row["combinationID"]
        solution = row[seq_cols][row["maskedImgIdx"]]
        pattern = row["pattern"]
        df = pd.DataFrame([[combinationID, solution, pattern]], columns=cols)
        df_solutions = pd.concat([df_solutions, df])
        seq = prepare_prompt(
            row, instructions="", seq_cols=seq_cols, choice_cols=choice_cols
        )
        text += f"\nPuzzle {idx + 1} (ID: {combinationID}):{seq}\n"

    df_solutions.reset_index(drop=True, inplace=True)

    print(text)
    pyperclip.copy(text)

    user_inp = input("Model results copied? (y/n)").lower()
    if user_inp == "n":
        return

    model_results = pd.read_clipboard(
        sep=",", header=None, names=["idx", "combinationID", "answer"]
    )
    model_results = model_results.merge(df_solutions, on="combinationID", how="left")
    model_results["masked_idx"] = masked_idx
    model_results["model"] = model_name

    model_results = model_results.loc[
        :, ["combinationID", "answer", "solution", "masked_idx", "model"]
    ]
    # clean_answer = model_results['answer'].apply(lambda x: x.lower().replace("answer:", "").strip())
    # model_results["clean_answer"] = clean_answer
    # model_results.insert(5, "correct", model_results["clean_answer"] == model_results["solution"])
    # model_results["correct"] = model_results["clean_answer"] == model_results["solution"]
    # model_results["correct"].sum() / model_results.shape[0]

    fpath = f"./ANNs/results/{model_name}-masked_idx[{masked_idx}]-{timestamp()}.csv"
    model_results.to_csv(fpath, index=False)

    return model_results, df_solutions


def run_together_models(models_list=None):
    #! TEMP
    models_list = [
        m for m in together.Models.list() if "llama 3" in m.get("display_name").lower()
    ]
    #! TEMP
    # att_list = []
    # for model in together.Models.list():
    #     att_list.extend(list(model.keys()))
    # att_list = list(set(att_list))

    api = "together"

    if models_list is None:
        model_type = "chat"
        models_list = together.Models.list()
        models_list = [m for m in models_list if m.get("display_type") == model_type]

    models_dict = {m["display_name"]: m for m in models_list}
    models_names = list(models_dict.keys())

    sequences = pd.read_csv(wd / "config/sequences-format[names].csv")
    sequences = sequences.sample(frac=1).reset_index(drop=True)

    seq_cols = [col for col in sequences.columns if "figure" in col]
    choice_cols = [col for col in sequences.columns if re.search(r"choice\d{1,2}", col)]

    unique_icons = sorted(sequences.loc[:, seq_cols + choice_cols].stack().unique())
    unique_icons = {icon: 101 + i for i, icon in enumerate(unique_icons)}

    config_file = Path(wd / "config/experiment_config.json")
    with open(config_file, "r") as f:
        config = json.load(f)

    with open(wd / f"config/instructions.txt", "r") as f:
        instructions = f.read()
    # instructions = instructions.replace("\n", " ")

    wait_time = 1.5
    error_val = "NA"
    results_cols = ["combinationID", "answer", "solution"]
    n_sequences = sequences.shape[0]

    # if (f := Path("./ANNs/config/unavailable_models.json")).exists():
    #     with open(f) as file:
    #         unavailable_models = json.load(file)
    # else:
    #     unavailable_models = get_unavailable_models(models_dict)
    unavailable_models = get_unavailable_models(models_dict)

    unavailable_models = {
        k: v
        for k, v in unavailable_models.items()
        if not v.startswith("Too many requests")
    }

    api_func = api_funcs[api]
    default_config = inference_config[api]

    prompt_formats = get_prompt_formats(models_dict)
    models_dict = {k: v for k, v in models_dict.items() if k not in unavailable_models}

    for model, model_info in tqdm(models_dict.items()):
        config = default_config.copy()
        config["model"] = model

        prompt_format = prompt_formats.get(model)

        # for masked_idx in [0, 4, 7]:
        for masked_idx in [4, 7]:
            sequences["maskedImgIdx"] = masked_idx
            results = np.zeros((n_sequences, len(results_cols)), dtype="object")

            iter_seq = tqdm(sequences.iterrows(), total=n_sequences, leave=False)

            for idx, row in iter_seq:
                t1 = time.time()

                combinationID = row["combinationID"]
                solution = row[seq_cols][row["maskedImgIdx"]]

                config["prompt"] = prepare_prompt(
                    row,
                    instructions,
                    seq_cols,
                    choice_cols,
                    prompt_format=prompt_format,
                )

                output = api_model_call(
                    func_call=api_func,
                    prompt_args=config,
                    wait_time=2.0,
                    n_tries=3,
                    error_value=error_val,
                )

                if output == error_val:
                    print(f"Error with model {model}\n")
                    results[idx, :] = [combinationID, error_val, solution]
                else:
                    text = output["output"]["choices"][0]["text"]
                    results[idx, :] = [combinationID, text, solution]

                duration = time.time() - t1
                if duration < wait_time:
                    time.sleep(wait_time - duration)

            results = pd.DataFrame(results, columns=results_cols)
            results["masked_idx"] = masked_idx
            results["model"] = model

            model_name = model.replace("/", "_")
            fname = f"{model_name}-masked_idx[{masked_idx}]-{timestamp()}"

            results.to_csv(wd / f"results/{fname}.csv", index=False)
