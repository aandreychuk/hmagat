import argparse
import json
import pathlib
import time

import numpy as np
import torch
import yaml

from hmagat.run_expert import add_expert_dataset_args
from hmagat.training_args import add_training_args
from hmagat.convert_to_imitation_dataset import add_imitation_dataset_args
from hmagat.generate_hypergraphs import add_hypergraph_generation_args
from hmagat.generate_additional_data import add_additional_data_args
from hmagat.temperature_training import (
    add_temperature_sampling_args,
    get_temperature_sampling_model,
)
from hmagat.modules.agents import get_model
from hmagat.modules.model.run_model import run_model_on_grid

from pogema import GridConfig


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


class TimingModelWrapper:
    def __init__(self, base_obj):
        self.base_obj = base_obj
        self.timings = []

    def __call__(self, *args, **kwargs):
        start_time = time.time()
        results = self.base_obj(*args, **kwargs)
        end_time = time.time()
        self.timings.append(end_time - start_time)
        return results

    def __getattr__(self, name):
        return getattr(self.base_obj, name)


def _result_dict(
    map_name,
    seed,
    num_agents,
    success,
    makespan,
    partial_success_rate,
    sum_of_costs,
    runtime,
    algorithm,
):
    """One entry for the JSON results list."""
    return {
        "metrics": {
            "ISR": partial_success_rate,
            "CSR": 1.0 if success else 0.0,
            "ep_length": makespan,
            "SoC": sum_of_costs,
            "makespan": makespan,
            "runtime": runtime,
        },
        "env_grid_search": {
            "num_agents": num_agents,
            "map_name": map_name,
            "seed": seed,
        },
        "algorithm": algorithm,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test HMAGAT on maps from a YAML file."
    )

    # ── HMAGAT model argument groups ──
    parser = add_expert_dataset_args(parser)
    parser = add_imitation_dataset_args(parser)
    parser = add_hypergraph_generation_args(parser)
    parser = add_additional_data_args(parser)
    parser = add_training_args(parser)
    parser = add_temperature_sampling_args(parser)

    # ── Pogema benchmark settings ──
    # NOTE: --num_agents, --max_episode_steps, --obs_radius, --collision_system,
    #       --on_target are already registered by add_grid_config_args
    #       (via add_expert_dataset_args). Use set_defaults() to override.
    parser.add_argument(
        "--maps_path",
        type=str,
        default="maps.yaml",
        help="Path to YAML file with named maps.",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1,
        help="Episodes per map. Seeds used: 0, 1, ..., num_episodes-1",
    )

    # ── HMAGAT checkpoint loading ──
    parser.add_argument("--model_epoch_num", type=int, default=None)
    parser.add_argument(
        "--temperature_sampling_model_epoch_num", type=int, default=43
    )

    # ── Output options ──
    parser.add_argument(
        "--svg_save_dir",
        type=str,
        default=None,
        help="If set, save SVG animations here (one per task).",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="If set, write results (list of dicts) to this JSON file.",
    )
    parser.add_argument("--record_timings", action="store_true", default=False)

    # ── Override defaults to match README evaluation section ──
    parser.set_defaults(
        # Expert dataset
        obs_radius=5,
        save_termination_state=True,
        # Hypergraph
        hypergraph_comm_radius=7,
        hyperedge_generation_method="kmeans",
        hypergraph_num_updates=10,
        hypergraph_wait_one=True,
        hypergraph_initial_colperc=0.1,
        hypergraph_final_colperc=0.1,
        # Additional data
        add_data_cost_to_go=True,
        normalize_cost_to_go=True,
        clamp_cost_to_go=1.0,
        # Imitation dataset
        use_lists=True,
        # Training / model
        checkpoints_dir="checkpoints/hmagat",
        run_name="hmagat",
        device=-1,
        run_online_expert=True,
        imitation_learning_model="DirectionalHMAGAT",
        hyperedge_feature_generator="magat",
        final_feature_generator="magat",
        model_residuals="all",
        use_edge_attr=True,
        use_edge_attr_for_messages="positions+manhattan",
        edge_attr_cnn_mode="MLP",
        load_positions_separately=True,
        train_on_terminated_agents=True,
        recursive_oe=True,
        cnn_mode="ResNetLarge_withMLP",
        collision_shielding="pibt",
        action_sampling="probabilistic",
        # Temperature sampling
        rl_based_temperature_sampling=True,
        temperature_checkpoints_dir="checkpoints/hmagat_temperature_module",
        temperature_run_name="simple_rl",
        temperature_actor_critic="simple-local-val-init",
        temperature_optimize="only-all-on-goal",
    )

    args = parser.parse_args()
    print(args)

    assert args.save_termination_state

    # ── Device ──
    if args.device == -1:
        device = torch.device("cuda")
    elif args.device is not None:
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")

    # ── Load HMAGAT model ──
    model, hypergraph_model, dataset_kwargs = get_model(args, device)

    if args.temperature_sampling_model_epoch_num is None:
        if args.rl_based_temperature_sampling:
            # Only here to count the number of parameters
            model = get_temperature_sampling_model(model, args, device, state_dict=None)

        num_parameters = count_parameters(model)
        print(f"Num Parameters: {num_parameters}")

    if args.model_epoch_num is None:
        checkpoint_path = pathlib.Path(args.checkpoints_dir, "best.pt")
        if not checkpoint_path.exists():
            checkpoint_path = pathlib.Path(args.checkpoints_dir, "best_low_val.pt")
    else:
        checkpoint_path = pathlib.Path(
            args.checkpoints_dir, f"epoch_{args.model_epoch_num}.pt"
        )

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.eval()

    if args.rl_based_temperature_sampling:
        assert args.temperature_sampling_model_epoch_num is not None
        checkpoint_path = pathlib.Path(
            args.temperature_checkpoints_dir,
            f"epoch_{args.temperature_sampling_model_epoch_num}.pt",
        )
        state_dict = torch.load(checkpoint_path, map_location=device)
        model = get_temperature_sampling_model(model, args, device, state_dict)

        num_parameters = count_parameters(model)
        print(f"Num Parameters: {num_parameters}")

    use_target_vec = args.use_target_vec
    if use_target_vec is None and args.rl_based_temperature_sampling:
        use_target_vec = "target-vec"

    if args.record_timings:
        model = TimingModelWrapper(model)

    # ── Load maps ──
    maps_path = pathlib.Path(args.maps_path)
    if not maps_path.exists():
        raise FileNotFoundError(f"Maps file not found: {maps_path}")
    with open(maps_path, "r") as f:
        maps = yaml.safe_load(f)
    map_items = list(maps.items())
    if not map_items:
        raise ValueError("No maps to run.")

    seeds_per_map = list(range(0, args.num_episodes))
    # --num_agents is a "+"-delimited string (e.g. "16+24+32") from add_grid_config_args_mixed
    num_agents_list = [int(x) for x in args.num_agents.split("+")]
    total_tasks = len(map_items) * len(seeds_per_map) * len(num_agents_list)

    save_svg = bool(args.svg_save_dir)
    if save_svg:
        pathlib.Path(args.svg_save_dir).mkdir(parents=True, exist_ok=True)

    # ── Evaluation loop ──
    num_completed = 0
    all_makespan = []
    all_soc = []
    all_partial_success_rate = []
    json_results = []

    task_id = 0
    for map_idx, (map_name, map_str) in enumerate(map_items):
        for ep_idx, seed in enumerate(seeds_per_map):
            for na in num_agents_list:
                grid_config = GridConfig(
                    map=map_str,
                    num_agents=na,
                    obs_radius=args.obs_radius,
                    collision_system=args.collision_system,
                    on_target=args.on_target,
                    max_episode_steps=args.max_episode_steps,
                    observation_type="MAPF",
                    seed=int(seed),
                )

                # aux_func to track makespan and costs
                def aux_func(env, observations, actions, **kwargs):
                    if actions is None:
                        aux_func.makespan = 0
                        aux_func.costs = np.ones(env.get_num_agents())
                    else:
                        at_goals = np.array(env.was_on_goal)
                        aux_func.makespan += 1
                        aux_func.costs[~at_goals] = aux_func.makespan + 1

                t0 = time.perf_counter()
                success, env, _ = run_model_on_grid(
                    model,
                    device,
                    grid_config,
                    args,
                    hypergraph_model,
                    dataset_kwargs=dataset_kwargs,
                    use_target_vec=use_target_vec,
                    aux_func=aux_func,
                    animation_monitor=save_svg,
                )
                runtime = time.perf_counter() - t0

                makespan = aux_func.makespan
                costs = aux_func.costs
                partial_success_rate = float(np.mean(env.was_on_goal))
                sum_of_costs = int(np.sum(costs))

                num_completed += 1 if success else 0
                all_makespan.append(makespan)
                all_soc.append(sum_of_costs)
                all_partial_success_rate.append(partial_success_rate)

                json_results.append(
                    _result_dict(
                        map_name,
                        seed,
                        na,
                        success,
                        makespan,
                        partial_success_rate,
                        sum_of_costs,
                        runtime,
                        "HMAGAT",
                    )
                )

                if save_svg:
                    out = pathlib.Path(args.svg_save_dir)
                    env.save_animation(out / f"anim_{task_id}.svg")

                timing_str = ""
                if args.record_timings:
                    timings = np.mean(model.timings)
                    model.timings = []
                    timing_str = f" | mean_timing={timings:.4f}s"

                print(
                    f"Task {task_id + 1}/{total_tasks} "
                    f"map={map_name} seed={seed} n_agents={na} "
                    f"success={success} makespan={makespan} SoC={sum_of_costs} "
                    f"ISR={partial_success_rate:.3f} "
                    f"rate={num_completed}/{task_id + 1}{timing_str}"
                )
                task_id += 1

    print(
        f"\nDone. Success {num_completed}/{total_tasks}, "
        f"avg makespan={np.mean(all_makespan):.1f} "
        f"avg SoC={np.mean(all_soc):.1f} "
        f"avg ISR={np.mean(all_partial_success_rate):.3f}"
    )

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(json_results, f, indent=2)
        print(f"Wrote {len(json_results)} results to {args.output_json}")


if __name__ == "__main__":
    main()
