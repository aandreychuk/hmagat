import argparse
import pickle
import pathlib
import numpy as np

from hmagat.run_expert import add_expert_dataset_args, get_expert_dataset_file_name

from hmagat.dataset_loading import load_dataset


def get_and_save_expert_makespans(args, file_name=None):
    dataset = load_dataset(
        [get_expert_dataset_file_name],
        "raw_expert_predictions",
        args,
    )

    if isinstance(dataset, tuple):
        dataset, seed_mask = dataset

    makespans = np.zeros(len(seed_mask))
    seed_mask = np.array(seed_mask, dtype=bool)
    makespans[~seed_mask] = np.inf

    ids = np.arange(len(seed_mask))
    ids = ids[seed_mask]

    for i, id in enumerate(ids):
        makespans[id] = len(dataset[i][1])

    if file_name is None:
        file_name = get_expert_dataset_file_name(args)

    path = pathlib.Path(args.dataset_dir, "expert_makespans", file_name)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(makespans, f)

    return makespans


def check_or_create_expert_makespans(args):
    file_name = get_expert_dataset_file_name(args)
    path = pathlib.Path(args.dataset_dir, "expert_makespans", file_name)

    if path.exists():
        print(f"Loaded expert makespans.")
        with open(path, "rb") as f:
            makespans = pickle.load(f)
        return makespans
    else:
        print(f"Expert makespans not found, generating them.")
        return get_and_save_expert_makespans(args, file_name)


def main():
    parser = argparse.ArgumentParser(description="Save expert makespans.")
    parser = add_expert_dataset_args(parser)

    args = parser.parse_args()
    print(args)

    get_and_save_expert_makespans(args)
