import numpy as np
import torch


def generate_target_vec(
    dataset,
    num_samples,
    print_prefix="",
    stack_with_np=True,
):
    dataset_target_vec = []

    for id, (sample_observations, actions, terminated) in enumerate(dataset):
        if print_prefix is not None:
            print(
                f"{print_prefix}"
                f"Generating Graph Dataset for map {id + 1}/{num_samples}"
            )
        for observations in sample_observations:
            global_xys = np.array([obs["global_xy"] for obs in observations])
            global_target_xys = np.array(
                [obs["global_target_xy"] for obs in observations]
            )
            target_vec = global_target_xys - global_xys

            dataset_target_vec.append(target_vec)

    if not stack_with_np:
        dataset_target_vec = [
            torch.from_numpy(np.array(data)) for data in dataset_target_vec
        ]
        return dataset_target_vec

    dataset_target_vec = np.stack(dataset_target_vec)
    return torch.from_numpy(dataset_target_vec)
