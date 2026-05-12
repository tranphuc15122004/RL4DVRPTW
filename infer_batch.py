"""
Batch inference for AM and PolyNet models.

Supports:
  - Single .pyth file inference (batch)
  - Single .csv file inference
  - Directory of .csv files (batch over multiple scenarios)
  - Multiple .pyth files
  - Both AM and PolyNet models
  - Comparison mode (AM vs PolyNet on same data)

Usage examples:
    # Batch over a directory of CSV files with AM (all params explicit)
    python3 infer_batch.py --model am \\
        --model-weight data/_AM/chkpt_best.pyth \\
        --csv-dir data/datasets/100 --vehicles-count 5 --veh-capa 1300 --veh-speed 1 \\
        --customers-count 100 --output-dir output/batch_am_100

    # Same but auto-configure from training args.json
    python3 infer_batch.py --model am \\
        --model-weight data/_AM/chkpt_best.pyth \\
        --csv-dir data/datasets/100 --model-args data/_AM/args.json \\
        --output-dir output/batch_am_100

    # Batch over a directory of CSV files with PolyNet
    python3 infer_batch.py --model polynet \\
        --model-weight data/_PolyNet/chkpt_best.pyth \\
        --csv-dir data/datasets/100 --model-args data/_PolyNet/args.json \\
        --output-dir output/batch_polynet_100

    # Compare AM vs PolyNet on the same CSV directory
    python3 infer_batch.py --model compare \\
        --am-weight data/_AM/chkpt_best.pyth \\
        --polynet-weight data/_PolyNet/chkpt_best.pyth \\
        --csv-dir data/datasets/100 --am-args data/_AM/args.json \\
        --polynet-args data/_PolyNet/args.json \\
        --output-dir output/batch_compare_100

    # Single .pyth file with AM
    python3 infer_batch.py --model am \\
        --model-weight data/_AM/chkpt_best.pyth \\
        --data-file data/dvrptw_n100m5_10240.pyth \\
        --output-dir output/batch_am_pyth

When --model-args / --am-args / --polynet-args are given, the corresponding
model architecture, problem, and environment parameters are loaded from the
training args.json file.  Any individual CLI flag still overrides the JSON.
"""

import csv
import json
import os
import sys
import time
from argparse import ArgumentParser
from collections import OrderedDict

import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import parse_args, set_random_seed
from utils.infer_utils import (
    add_infer_args,
    build_dataset,
    build_env,
    build_env_params,
    clone_dataset,
    compute_cost_components,
    dataset_cls,
    discover_csv_files,
    environment_cls,
    init_am_model,
    init_polynet_model,
    load_model_weights,
    parse_infer_args,
    replay_routes_cost,
    route_diag_for_instance,
    run_inference,
    run_single_inference,
    save_json,
    warmup_model,
)


# ---------------------------------------------------------------------------
# Batch argument parser
# ---------------------------------------------------------------------------

def parse_batch_args(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = ArgumentParser(description="Batch inference for AM / PolyNet")

    # Model selection
    parser.add_argument("--model", type=str, choices=["am", "polynet", "compare"],
                        default="am", help="Model type to use for inference")
    parser.add_argument("--model-weight", type=str, default=None,
                        help="Path to model weights (for single-model mode)")
    parser.add_argument("--am-weight", type=str, default=None,
                        help="Path to AM model weights (for compare mode)")
    parser.add_argument("--polynet-weight", type=str, default=None,
                        help="Path to PolyNet model weights (for compare mode)")

    # Data sources
    parser.add_argument("--data-file", type=str, default=None,
                        help="Path to a .pyth dataset file")
    parser.add_argument("--data-csv", type=str, default=None,
                        help="Path to a single CSV scenario file")
    parser.add_argument("--csv-dir", type=str, default=None,
                        help="Directory containing CSV scenario files (batch mode)")
    parser.add_argument("--pyth-dir", type=str, default=None,
                        help="Directory containing .pyth dataset files (batch mode)")

    # Output
    parser.add_argument("--output-dir", type=str, default="output/batch_infer",
                        help="Directory to write per-file and aggregated results")
    parser.add_argument("--csv-output", type=str, default=None,
                        help="Aggregated CSV output path (default: {output_dir}/aggregated.csv)")

    # Filtering
    parser.add_argument("--max-files", type=int, default=None,
                        help="Limit number of CSV/pyth files to process")
    parser.add_argument("--file-pattern", type=str, default=None,
                        help="Only process files whose name contains this substring")

    # Model args (auto-configure from training args.json)
    parser.add_argument("--model-args", type=str, default=None,
                        help="Path to a training args.json to auto-configure model/problem params")
    parser.add_argument("--am-args", type=str, default=None,
                        help="Path to AM training args.json (for compare mode)")
    parser.add_argument("--polynet-args", type=str, default=None,
                        help="Path to PolyNet training args.json (for compare mode)")

    # Config fallback
    parser.add_argument("--config-file", type=str, default=None,
                        help="Path to a JSON config file for model hyperparameters")

    args, remain = parser.parse_known_args(argv)

    # Merge with general args via parse_args
    general_args = parse_args(remain)
    for key in vars(general_args):
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, getattr(general_args, key))

    # Add inference args
    infer_parser = ArgumentParser(add_help=False)
    add_infer_args(infer_parser)
    infer_args, _ = infer_parser.parse_known_args(argv)
    for key in vars(infer_args):
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, getattr(infer_args, key))

    # Set defaults for greedy/sample
    if not hasattr(args, "greedy"):
        args.greedy = True
    if not hasattr(args, "sample"):
        args.sample = False

    # Resolve config file (standard --config-file, for training hyperparams)
    if args.config_file is not None:
        with open(args.config_file) as f:
            config = json.load(f)
            for key, value in config.items():
                if not hasattr(args, key) or getattr(args, key) is None:
                    setattr(args, key, value)

    # Load model args file (--model-args or per-model args in compare mode)
    if args.model_args is not None:
        from utils.infer_utils import load_model_args_from_file, merge_model_args_into_namespace
        model_dict = load_model_args_from_file(args.model_args)
        merge_model_args_into_namespace(args, model_dict)

    return args


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def create_model(args, model_type, env_cls, device):
    if model_type == "am":
        return init_am_model(args, env_cls, device)
    elif model_type == "polynet":
        return init_polynet_model(args, env_cls, device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def resolve_weight_path(args, model_type):
    if model_type == "am":
        return args.am_weight or args.model_weight
    elif model_type == "polynet":
        return args.polynet_weight or args.model_weight
    return args.model_weight


# ---------------------------------------------------------------------------
# Per-file inference runner
# ---------------------------------------------------------------------------

def infer_on_file(args, model_type, data_path, is_csv, device, learner=None, env_cls=None):
    """Run inference on a single data file.

    Args:
        args: parsed arguments
        model_type: 'am' or 'polynet'
        data_path: path to .pyth or .csv file
        is_csv: True if CSV, False if .pyth
        device: torch device
        learner: optional pre-loaded model (reused across files)
        env_cls: optional pre-resolved environment class

    Returns:
        dict with inference results, or None on error
    """
    try:
        start_t = time.time()

        # Override data source args
        args.data_csv = data_path if is_csv else None
        args.data_file = data_path if not is_csv else None

        # Resolve dataset/environment classes
        if env_cls is None:
            dset_cls = dataset_cls(args.problem_type)
            env_cls = environment_cls(args.problem_type)
            if dset_cls is None or env_cls is None:
                raise ValueError(f"Unsupported problem type '{args.problem_type}'")
        else:
            dset_cls = dataset_cls(args.problem_type)

        # Load dataset
        data = build_dataset(args, dset_cls)
        raw_data = clone_dataset(data)

        if not args.no_normalize:
            data.normalize()

        # Build environment
        env_params = build_env_params(args)
        env = build_env(args, data, env_cls, env_params, device)

        # Run inference
        routes, costs = run_inference(args, env, learner)

        # Replay on raw data
        raw_replay_costs = replay_routes_cost(raw_data, env_cls, env_params, routes,
                                              rollouts=args.verify_rollouts)

        # Diagnostics
        route_diags = [route_diag_for_instance(data, routes, idx)
                       for idx in range(len(routes))]
        raw_cc = compute_cost_components(raw_data, routes, args.pending_cost, args.late_cost)

        elapsed = time.time() - start_t

        result = {
            "file": os.path.basename(data_path),
            "file_path": data_path,
            "model": model_type,
            "num_instances": costs.numel(),
            "mean_cost": costs.mean().item(),
            "std_cost": costs.std().item() if costs.numel() > 1 else 0.0,
            "mean_raw_replay_cost": raw_replay_costs.mean().item(),
            "total_skipped": sum(d["missing_count"] for d in route_diags),
            "elapsed_seconds": elapsed,
            "routes": routes,
            "costs": costs,
            "raw_replay_costs": raw_replay_costs,
            "route_diagnostics": route_diags,
            "raw_cost_components": raw_cc,
        }
        return result

    except Exception as e:
        print(f"  ERROR on {data_path}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Aggregated CSV writer
# ---------------------------------------------------------------------------

def write_aggregated_csv(results, output_path):
    """Write per-file aggregated results to a CSV."""
    fieldnames = [
        "file", "model", "num_instances", "mean_cost", "std_cost",
        "mean_raw_replay_cost", "total_skipped", "elapsed_seconds"
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"Wrote aggregated CSV to '{output_path}'")


# ---------------------------------------------------------------------------
# Comparison runner
# ---------------------------------------------------------------------------

def run_comparison(args, data_path, is_csv, device, am_learner, pn_learner, env_cls):
    """Run both AM and PolyNet on the same data and return comparison results."""
    am_result = infer_on_file(args, "am", data_path, is_csv, device,
                              learner=am_learner, env_cls=env_cls)
    pn_result = infer_on_file(args, "polynet", data_path, is_csv, device,
                              learner=pn_learner, env_cls=env_cls)

    comp = {
        "file": os.path.basename(data_path),
        "am_mean_cost": am_result["mean_cost"] if am_result else None,
        "pn_mean_cost": pn_result["mean_cost"] if pn_result else None,
        "am_std_cost": am_result["std_cost"] if am_result else None,
        "pn_std_cost": pn_result["std_cost"] if pn_result else None,
        "am_raw_replay": am_result["mean_raw_replay_cost"] if am_result else None,
        "pn_raw_replay": pn_result["mean_raw_replay_cost"] if pn_result else None,
        "am_skipped": am_result["total_skipped"] if am_result else None,
        "pn_skipped": pn_result["total_skipped"] if pn_result else None,
        "am_elapsed": am_result["elapsed_seconds"] if am_result else None,
        "pn_elapsed": pn_result["elapsed_seconds"] if pn_result else None,
        "cost_diff": (am_result["mean_cost"] - pn_result["mean_cost"])
                     if (am_result and pn_result) else None,
        "am_result": am_result,
        "pn_result": pn_result,
    }
    return comp


# ---------------------------------------------------------------------------
# Main batch logic
# ---------------------------------------------------------------------------

def main():
    args = parse_batch_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    set_random_seed(args.rng_seed, deterministic=True)

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Collect data files ---
    data_files = []  # list of (path, is_csv)

    if args.data_file is not None:
        data_files.append((args.data_file, False))

    if args.data_csv is not None:
        data_files.append((args.data_csv, True))

    if args.csv_dir is not None:
        csv_files = discover_csv_files(args.csv_dir)
        if args.file_pattern:
            csv_files = [f for f in csv_files if args.file_pattern in os.path.basename(f)]
        if args.max_files:
            csv_files = csv_files[:args.max_files]
        for cf in csv_files:
            data_files.append((cf, True))
        print(f"Discovered {len(csv_files)} CSV files in '{args.csv_dir}'")

    if args.pyth_dir is not None:
        pyth_files = sorted([
            os.path.join(args.pyth_dir, fn)
            for fn in os.listdir(args.pyth_dir)
            if fn.endswith(".pyth") and not fn.startswith(".")
        ])
        if args.file_pattern:
            pyth_files = [f for f in pyth_files if args.file_pattern in os.path.basename(f)]
        if args.max_files:
            pyth_files = pyth_files[:args.max_files]
        for pf in pyth_files:
            data_files.append((pf, False))
        print(f"Discovered {len(pyth_files)} .pyth files in '{args.pyth_dir}'")

    if not data_files:
        print("No data files found. Use --data-file, --data-csv, --csv-dir, or --pyth-dir.")
        sys.exit(1)

    print(f"Total files to process: {len(data_files)}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {device}")

    # --- Resolve env class once ---
    dset_cls = dataset_cls(args.problem_type)
    env_cls = environment_cls(args.problem_type)
    if dset_cls is None or env_cls is None:
        raise ValueError(f"Unsupported problem type '{args.problem_type}'")

    # --- Load model(s) ---
    if args.model == "compare":
        from utils.infer_utils import load_model_args_from_file, merge_model_args_into_namespace

        # Prepare separate args for AM and PolyNet (if per-model args provided)
        am_args = args
        pn_args = args
        if args.am_args is not None:
            import copy
            am_args = copy.copy(args)
            am_dict = load_model_args_from_file(args.am_args)
            merge_model_args_into_namespace(am_args, am_dict)
        if args.polynet_args is not None:
            import copy
            pn_args = copy.copy(args)
            pn_dict = load_model_args_from_file(args.polynet_args)
            merge_model_args_into_namespace(pn_args, pn_dict)

        print("Loading AM model...")
        am_learner = init_am_model(am_args, env_cls, device)
        am_learner.eval()
        warmup_model(am_args, dset_cls.generate(1, am_args.customers_count, am_args.vehicles_count,
                                                 am_args.veh_capa, am_args.veh_speed),
                     env_cls, build_env_params(am_args), am_learner)
        load_model_weights(resolve_weight_path(args, "am"), am_learner)
        am_learner.eval()

        print("Loading PolyNet model...")
        pn_learner = init_polynet_model(pn_args, env_cls, device)
        pn_learner.eval()
        warmup_model(pn_args, dset_cls.generate(1, pn_args.customers_count, pn_args.vehicles_count,
                                                 pn_args.veh_capa, pn_args.veh_speed),
                     env_cls, build_env_params(pn_args), pn_learner)
        load_model_weights(resolve_weight_path(args, "polynet"), pn_learner)
        pn_learner.eval()

        # Run comparison
        comparisons = []
        for idx, (fpath, is_csv) in enumerate(data_files):
            print(f"\n[{idx+1}/{len(data_files)}] Comparing on: {os.path.basename(fpath)}")
            comp = run_comparison(args, fpath, is_csv, device, am_learner, pn_learner, env_cls)
            comparisons.append(comp)

            if comp["am_result"]:
                print(f"  AM  | mean_cost={comp['am_mean_cost']:.4f} | "
                      f"skipped={comp['am_skipped']} | {comp['am_elapsed']:.2f}s")
            if comp["pn_result"]:
                print(f"  PolyNet | mean_cost={comp['pn_mean_cost']:.4f} | "
                      f"skipped={comp['pn_skipped']} | {comp['pn_elapsed']:.2f}s")
            if comp["cost_diff"] is not None:
                print(f"  Δ (AM - PolyNet) = {comp['cost_diff']:.4f}")

            # Save per-file comparison JSON
            file_tag = os.path.splitext(os.path.basename(fpath))[0]
            out_json = os.path.join(args.output_dir, f"{file_tag}_comparison.json")
            json.dump({
                "file": comp["file"],
                "am": {
                    "mean_cost": comp["am_mean_cost"],
                    "std_cost": comp["am_std_cost"],
                    "mean_raw_replay_cost": comp["am_raw_replay"],
                    "total_skipped": comp["am_skipped"],
                    "elapsed": comp["am_elapsed"],
                } if comp["am_result"] else None,
                "polynet": {
                    "mean_cost": comp["pn_mean_cost"],
                    "std_cost": comp["pn_std_cost"],
                    "mean_raw_replay_cost": comp["pn_raw_replay"],
                    "total_skipped": comp["pn_skipped"],
                    "elapsed": comp["pn_elapsed"],
                } if comp["pn_result"] else None,
                "cost_diff_am_minus_pn": comp["cost_diff"],
            }, f, indent=2)
            print(f"  Saved comparison to '{out_json}'")

        # Write aggregated comparison CSV
        agg_path = args.csv_output or os.path.join(args.output_dir, "comparison_aggregated.csv")
        with open(agg_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "file", "am_mean_cost", "pn_mean_cost", "cost_diff",
                "am_std_cost", "pn_std_cost",
                "am_raw_replay", "pn_raw_replay",
                "am_skipped", "pn_skipped",
                "am_elapsed", "pn_elapsed",
            ])
            writer.writeheader()
            for comp in comparisons:
                writer.writerow({
                    "file": comp["file"],
                    "am_mean_cost": comp["am_mean_cost"],
                    "pn_mean_cost": comp["pn_mean_cost"],
                    "cost_diff": comp["cost_diff"],
                    "am_std_cost": comp["am_std_cost"],
                    "pn_std_cost": comp["pn_std_cost"],
                    "am_raw_replay": comp["am_raw_replay"],
                    "pn_raw_replay": comp["pn_raw_replay"],
                    "am_skipped": comp["am_skipped"],
                    "pn_skipped": comp["pn_skipped"],
                    "am_elapsed": comp["am_elapsed"],
                    "pn_elapsed": comp["pn_elapsed"],
                })
        print(f"Wrote aggregated comparison CSV to '{agg_path}'")

    else:
        # Single model mode
        learner = create_model(args, args.model, env_cls, device)
        learner.eval()
        # Warmup with generated data
        warmup_model(args, dset_cls.generate(1, args.customers_count, args.vehicles_count,
                                               args.veh_capa, args.veh_speed),
                     env_cls, build_env_params(args), learner)
        weight_path = resolve_weight_path(args, args.model)
        load_model_weights(weight_path, learner)
        learner.eval()

        # Process each file
        results = []
        for idx, (fpath, is_csv) in enumerate(data_files):
            print(f"\n[{idx+1}/{len(data_files)}] Inferring {args.model.upper()} on: "
                  f"{os.path.basename(fpath)}")
            result = infer_on_file(args, args.model, fpath, is_csv, device,
                                   learner=learner, env_cls=env_cls)
            if result is None:
                continue

            results.append(result)
            print(f"  instances={result['num_instances']}, "
                  f"mean_cost={result['mean_cost']:.4f}, "
                  f"std={result['std_cost']:.4f}, "
                  f"raw_replay={result['mean_raw_replay_cost']:.4f}, "
                  f"skipped={result['total_skipped']}, "
                  f"elapsed={result['elapsed_seconds']:.2f}s")

            # Save per-file JSON with full details
            file_tag = os.path.splitext(os.path.basename(fpath))[0]
            out_json = os.path.join(args.output_dir, f"{file_tag}_{args.model}.json")
            from utils.infer_utils import check_route_constraints
            constraint_diags = check_route_constraints(
                clone_dataset(build_dataset(args, dset_cls)) if not is_csv else
                build_dataset(args, dset_cls),
                result["routes"]
            ) if False else []  # skip heavy constraint check for speed; enable if needed
            save_json(
                out_json,
                result["routes"],
                result["costs"],
                result["raw_replay_costs"],
                result["route_diagnostics"],
                [],
                result["raw_cost_components"],
                [],
            )
            print(f"  Saved to '{out_json}'")

        # Write aggregated CSV
        agg_path = args.csv_output or os.path.join(args.output_dir, "aggregated.csv")
        write_aggregated_csv(results, agg_path)

        # Print summary
        if results:
            mean_costs = [r["mean_cost"] for r in results]
            mean_replays = [r["mean_raw_replay_cost"] for r in results]
            print("\n" + "=" * 60)
            print(f"BATCH INFERENCE SUMMARY ({args.model.upper()})")
            print(f"  Files processed : {len(results)}/{len(data_files)}")
            print(f"  Mean cost range : {min(mean_costs):.4f} – {max(mean_costs):.4f}")
            print(f"  Avg mean cost   : {sum(mean_costs)/len(mean_costs):.4f}")
            print(f"  Avg replay cost : {sum(mean_replays)/len(mean_replays):.4f}")
            print(f"  Total skipped   : {sum(r['total_skipped'] for r in results)}")
            print("=" * 60)


if __name__ == "__main__":
    main()
