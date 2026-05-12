"""
AM model — Single-instance inference from a CSV file.

Loads one CSV scenario, runs the AM model, and prints detailed per-vehicle
routes, costs, constraint violations, and diagnostics.

Usage (all params explicit):
    python3 am/infer_single.py --data-csv data/datasets/100/h100c101.csv \\
        --model-weight data/_AM/chkpt_best.pyth \\
        --customers-count 100 --vehicles-count 5 --veh-capa 1300 --veh-speed 1

Usage (auto-configure from training args.json):
    python3 am/infer_single.py --data-csv data/datasets/100/h100c101.csv \\
        --model-weight data/_AM/chkpt_best.pyth \\
        --model-args data/_AM/args.json

When --model-args is given, model architecture, problem, and environment
parameters are loaded from the file.  Any individual CLI flag still overrides
the corresponding value in the JSON (CLI takes precedence).
"""

import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import set_random_seed
from utils.infer_utils import (
    add_infer_args,
    build_dataset,
    build_env,
    build_env_params,
    clone_dataset,
    compute_cost_components,
    dataset_cls,
    environment_cls,
    init_am_model,
    load_model_weights,
    parse_infer_args,
    replay_routes_cost,
    route_diag_for_instance,
    run_inference,
    run_single_inference,
    save_json,
    warmup_model,
)
from utils import routes_to_string


def main():
    args = parse_infer_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    set_random_seed(args.rng_seed, deterministic=True)

    # --- Resolve dataset class & environment class ---
    dset_cls = dataset_cls(args.problem_type)
    env_cls = environment_cls(args.problem_type)
    if dset_cls is None or env_cls is None:
        raise ValueError(f"Unsupported problem type '{args.problem_type}'")

    # --- Load dataset ---
    data = build_dataset(args, dset_cls)
    raw_data = clone_dataset(data)

    print(f"Loaded dataset: batch_size={data.batch_size}, nodes={data.nodes_count}, "
          f"vehicles={data.veh_count}, features={data.nodes.size(-1)}")

    # If we generated on-the-fly, there may be more than 1 instance; cap to show
    n_instances = data.batch_size
    if n_instances > 1:
        print(f"Note: dataset has {n_instances} instances. Showing first instance in detail.")

    if not args.no_normalize:
        data.normalize()
        print("Dataset normalized.")

    # --- Build environment ---
    env_params = build_env_params(args)
    env = build_env(args, data, env_cls, env_params, device)
    print(f"Environment built: minibatch_size={env.minibatch_size}, "
          f"veh_count={env.veh_count}, nodes_count={env.nodes_count}")

    # --- Initialize model ---
    learner = init_am_model(args, env_cls, device)
    learner.eval()
    print(f"AM model initialised (model_size={args.model_size}, "
          f"layers={args.layer_count}, heads={args.head_count})")

    # --- Warmup + load weights ---
    warmup_model(args, data, env_cls, env_params, learner)
    load_model_weights(args.model_weight, learner)
    learner.eval()

    # --- Run inference ---
    routes, costs = run_inference(args, env, learner)
    print(f"\nInference complete: {costs.numel()} instance(s)")

    # --- Detailed per-instance output ---
    for inst_idx in range(min(costs.numel(), args.max_print_instances)):
        print("\n" + "=" * 70)
        print(f"INSTANCE #{inst_idx}  |  cost = {costs[inst_idx].item():.4f}")
        print("=" * 70)

        # Per-vehicle routes
        for veh_idx, route in enumerate(routes[inst_idx]):
            route_str = "0 → " + " → ".join(str(n) for n in route) if route else "0 (unused)"
            print(f"  Vehicle {veh_idx}: {route_str}")

        # Diagnostics
        diag = route_diag_for_instance(data, routes, inst_idx)
        print(f"\n  Route diagnostics:")
        print(f"    Active customers : {diag['active_customers']}")
        print(f"    Visited customers: {diag['visited_customers']}")
        print(f"    Missing count    : {diag['missing_count']}")
        print(f"    Duplicate count  : {diag['duplicate_count']}")
        if diag["missing_count"] > 0:
            print(f"    Missing (sample) : {diag['missing_head']}")
        if diag["duplicate_count"] > 0:
            print(f"    Duplicate (samp) : {diag['duplicate_head']}")

    # --- Replay cost on raw (un-normalized) data ---
    raw_replay_costs = replay_routes_cost(raw_data, env_cls, env_params, routes,
                                          rollouts=args.verify_rollouts)
    print(f"\nRaw replay costs: mean={raw_replay_costs.mean().item():.4f}")

    for idx in range(min(3, costs.numel())):
        print(f"  Instance #{idx}: normalized_cost={costs[idx].item():.4f}, "
              f"raw_replay_cost={raw_replay_costs[idx].item():.4f}, "
              f"diff={abs(costs[idx].item() - raw_replay_costs[idx].item()):.6f}")

    # --- Cost components ---
    raw_cc = compute_cost_components(raw_data, routes, args.pending_cost, args.late_cost)
    print(f"\nCost components (raw scale, first {min(3, len(raw_cc))} instances):")
    for idx in range(min(3, len(raw_cc))):
        c = raw_cc[idx]
        print(f"  Instance #{idx}: distance={c['distance']:.4f}, "
              f"late_time={c['late_time']:.4f}, late_penalty={c['late_penalty']:.4f}, "
              f"skipped={c['skipped_orders']}, skipped_penalty={c['skipped_penalty']:.4f}, "
              f"total_cost={c['total_cost']:.4f}")

    # --- Optional JSON save ---
    if args.save_json is not None:
        route_diagnostics = [route_diag_for_instance(data, routes, idx)
                             for idx in range(len(routes))]
        from utils.infer_utils import check_route_constraints, compute_cost_components, save_json
        constraint_diagnostics = check_route_constraints(raw_data, routes)
        norm_cc = compute_cost_components(data, routes, args.pending_cost, args.late_cost)
        save_json(
            args.save_json,
            routes,
            costs,
            raw_replay_costs,
            route_diagnostics,
            constraint_diagnostics,
            raw_cc,
            norm_cc,
        )


if __name__ == "__main__":
    main()
