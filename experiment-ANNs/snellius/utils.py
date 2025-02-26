
import pickle
from pathlib import Path
from tqdm.auto import tqdm

def reformat_act_files(res_dir: Path, verbose=True, delete_original=True):
    act_files = sorted(res_dir.glob("layers_acts*.pkl"))

    acts = []

    for act_f in act_files:
        with open(act_f, "rb") as f:
            acts.append(pickle.load(f))

    acts_by_layer: dict = {l: [] for l in acts[0].keys()}

    # print("Loading models' activations...", end=" ") if verbose else None
    for act in acts:
        for l, v in act.items():
            acts_by_layer[l].append(v)
    # print("Done!") if verbose else None

    # print("Saving activations by layer...", end=" ") if verbose else None
    for layer, acts in tqdm(acts_by_layer.items()):
        with open(res_dir / f"acts_by_layer-{layer}.pkl", "wb") as f:
            pickle.dump(acts, f)
    # print("Done !") if verbose else None

    if delete_original:
        [f.unlink() for f in act_files]

if __name__ == '__main__':
    dir_list = [
        "/home/cpinier/test/results/20250207-095625/meta-llama--Llama-3.2-3B-Instruct",
        "/home/cpinier/test/results/20250207-101604/meta-llama--Meta-Llama-3-8B-Instruct",
    ]
    dir_list = [Path(d) for d in dir_list]

    for d in dir_list:
        reformat_act_files(d, verbose=True, delete_original=True)