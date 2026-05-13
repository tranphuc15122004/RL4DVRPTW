#!/usr/bin/env python3
"""
Aggregate paper evaluation results into publication-ready tables.

Reads per-file JSON outputs from the paper_eval directory structure:
    output/paper_eval/{model}_{decode}/dvrptw_*.json

Produces:
    1. paper_summary.csv       — Per (model, decode, dataset) summary
    2. paper_comparison.csv    — AM vs PolyNet head-to-head
    3. paper_stats.json        — Full nested statistics for plotting

Usage:
    python3 scripts/aggregate_paper_stats.py
    python3 scripts/aggregate_paper_stats.py --input-dir output/paper_eval
    python3 scripts/aggregate_paper_stats.py --output-dir output/paper_eval
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_ORDER = ["n20m1", "n50m3", "n100m5", "n200m10", "n400m20"]
MODEL_ORDER = ["am", "polynet"]
DECODE_ORDER = ["greedy", "sample"]

DATASET_CUST_VEH = {
    "n20m1":  (20, 1),
    "n50m3":  (50, 3),
    "n100m5": (100, 5),
    "n200m10": (200, 10),
    "n400m20": (400, 20),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_dataset_tag(filename: str) -> str | None:
    """Extract dataset tag like 'n20m1' from filename."""
    m = re.search(r'(n\d+m\d+)', filename)
    return m.group(1) if m else None


def discover_result_files(base_dir: str) -> list[dict]:
    """Walk the paper_eval directory and discover all result JSON files.

    Returns a list of dicts with keys: path, model, decode, dataset, filename.
    """
    results = []
    base = Path(base_dir)

    if not base.exists():
        print(f"ERROR: Directory not found: {base_dir}", file=sys.stderr)
        return results

    for subdir in sorted(base.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith("_"):
            continue

        # Parse model_decode from directory name
        parts = subdir.name.split("_", 1)
        if len(parts) != 2:
            continue
        model, decode = parts
        if model not in MODEL_ORDER:
            continue
        if decode not in DECODE_ORDER:
            continue

        for json_file in sorted(subdir.glob("*.json")):
            if json_file.name.startswith("aggregated"):
                continue

            ds_tag = parse_dataset_tag(json_file.stem)
            if ds_tag is None:
                continue

            results.append({
                "path": str(json_file),
                "model": model,
                "decode": decode,
                "dataset": ds_tag,
                "filename": json_file.name,
            })

    return results


def load_result_json(path: str) -> dict | None:
    """Load a single result JSON file."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  WARN: Failed to load {path}: {e}", file=sys.stderr)
        return None


def compute_summary(data: dict) -> dict:
    """Compute summary statistics from a single result JSON.

    Args:
        data: the loaded JSON dict from save_json()

    Returns:
        dict with summary statistics.
    """
    costs = data.get("costs", data.get("normalized_costs", []))
    replay_costs = data.get("raw_replay_costs", costs)
    n = len(costs)

    # Basic stats
    mean_cost = sum(costs) / n if n > 0 else 0.0
    std_cost = (sum((c - mean_cost) ** 2 for c in costs) / n) ** 0.5 if n > 1 else 0.0
    min_cost = min(costs) if n > 0 else 0.0
    max_cost = max(costs) if n > 0 else 0.0

    mean_replay = sum(replay_costs) / n if n > 0 else 0.0
    std_replay = (sum((c - mean_replay) ** 2 for c in replay_costs) / n) ** 0.5 if len(replay_costs) > 1 else 0.0

    # Cost components (from raw_cost_components)
    cc_list = data.get("raw_cost_components", [])
    if cc_list:
        distances = [cc["distance"] for cc in cc_list]
        late_penalties = [cc["late_penalty"] for cc in cc_list]
        skipped_penalties = [cc["skipped_penalty"] for cc in cc_list]
        skipped_orders = [cc["skipped_orders"] for cc in cc_list]
        total_costs = [cc["total_cost"] for cc in cc_list]
        late_times = [cc["late_time"] for cc in cc_list]

        def _mean(lst): return sum(lst) / len(lst)
        def _sum(lst): return sum(lst)

        mean_distance = _mean(distances)
        mean_late_penalty = _mean(late_penalties)
        mean_skipped_penalty = _mean(skipped_penalties)
        mean_late_time = _mean(late_times)
        total_skipped = _sum(skipped_orders)
        mean_skipped = _mean(skipped_orders)
        mean_total_cost = _mean(total_costs)
    else:
        mean_distance = 0.0
        mean_late_penalty = 0.0
        mean_skipped_penalty = 0.0
        mean_late_time = 0.0
        total_skipped = data.get("total_skipped_customers", 0)
        mean_skipped = total_skipped / n if n > 0 else 0.0
        mean_total_cost = mean_replay

    # Constraint violations
    total_tw_violations = data.get("total_tw_violations", 0)
    total_appearance_violations = data.get("total_appearance_violations", 0)

    # Route diagnostics
    route_diags = data.get("route_diagnostics", [])
    total_missing = sum(d.get("missing_count", 0) for d in route_diags) if route_diags else total_skipped
    total_duplicates = sum(d.get("duplicate_count", 0) for d in route_diags) if route_diags else 0

    # Per-instance cost percentiles
    sorted_costs = sorted(costs) if costs else []
    p25 = sorted_costs[int(n * 0.25)] if n >= 4 else sorted_costs[0] if sorted_costs else 0.0
    p50 = sorted_costs[int(n * 0.50)] if n >= 2 else sorted_costs[0] if sorted_costs else 0.0
    p75 = sorted_costs[int(n * 0.75)] if n >= 4 else sorted_costs[-1] if sorted_costs else 0.0
    p95 = sorted_costs[int(n * 0.95)] if n >= 20 else sorted_costs[-1] if sorted_costs else 0.0

    return {
        "num_instances": n,
        # Normalized costs
        "mean_cost": mean_cost,
        "std_cost": std_cost,
        "min_cost": min_cost,
        "max_cost": max_cost,
        "p25_cost": p25,
        "p50_cost": p50,
        "p75_cost": p75,
        "p95_cost": p95,
        # Replay costs
        "mean_replay_cost": mean_replay,
        "std_replay_cost": std_replay,
        # Cost components (raw)
        "mean_distance": mean_distance,
        "mean_late_penalty": mean_late_penalty,
        "mean_skipped_penalty": mean_skipped_penalty,
        "mean_late_time": mean_late_time,
        "mean_total_cost_raw": mean_total_cost,
        # Constraint diagnostics
        "total_skipped": int(total_skipped),
        "mean_skipped": mean_skipped,
        "total_missing": int(total_missing),
        "total_duplicates": int(total_duplicates),
        "total_tw_violations": int(total_tw_violations),
        "total_appearance_violations": int(total_appearance_violations),
    }


# ---------------------------------------------------------------------------
# Output generators
# ---------------------------------------------------------------------------

def generate_summary_csv(results: list[dict], output_path: str):
    """Write paper_summary.csv with one row per (model, decode, dataset)."""
    fieldnames = [
        "model", "decode", "dataset", "customers", "vehicles",
        "num_instances",
        "mean_cost", "std_cost", "min_cost", "max_cost",
        "p25_cost", "p50_cost", "p75_cost", "p95_cost",
        "mean_replay_cost", "std_replay_cost",
        "mean_distance", "mean_late_penalty", "mean_skipped_penalty",
        "mean_late_time", "mean_total_cost_raw",
        "total_skipped", "mean_skipped",
        "total_missing", "total_duplicates",
        "total_tw_violations", "total_appearance_violations",
    ]

    # Sort: model, decode, then by dataset order
    ds_order = {tag: i for i, tag in enumerate(DATASET_ORDER)}
    results_sorted = sorted(results, key=lambda r: (
        MODEL_ORDER.index(r["model"]) if r["model"] in MODEL_ORDER else 99,
        DECODE_ORDER.index(r["decode"]) if r["decode"] in DECODE_ORDER else 99,
        ds_order.get(r["dataset"], 99),
    ))

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for r in results_sorted:
            cust, veh = DATASET_CUST_VEH.get(r["dataset"], ("?", "?"))
            row = {**r, "customers": cust, "vehicles": veh}
            writer.writerow(row)

    print(f"✓ Summary CSV written to: {output_path}")


def generate_comparison_csv(results: list[dict], output_path: str):
    """Write paper_comparison.csv — AM vs PolyNet head-to-head.

    Pairs results by (dataset, decode) and compares.
    """
    # Build lookup: (dataset, decode) -> {model: summary}
    lookup: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for r in results:
        key = (r["dataset"], r["decode"])
        lookup[key][r["model"]] = r

    fieldnames = [
        "dataset", "decode",
        "am_mean_cost", "pn_mean_cost", "delta_cost", "winner",
        "am_std", "pn_std",
        "am_replay", "pn_replay",
        "am_distance", "pn_distance",
        "am_late_pen", "pn_late_pen",
        "am_skipped_pen", "pn_skipped_pen",
        "am_skipped_total", "pn_skipped_total",
        "am_tw_violations", "pn_tw_violations",
    ]

    rows = []
    for ds in DATASET_ORDER:
        for dec in DECODE_ORDER:
            key = (ds, dec)
            if key not in lookup:
                continue
            pair = lookup[key]
            am = pair.get("am", {})
            pn = pair.get("polynet", {})

            if not am or not pn:
                continue

            am_cost = am.get("mean_cost", 0)
            pn_cost = pn.get("mean_cost", 0)
            delta = am_cost - pn_cost
            winner = "PolyNet" if delta > 0 else "AM" if delta < 0 else "Tie"

            rows.append({
                "dataset": ds,
                "decode": dec,
                "am_mean_cost": f"{am_cost:.4f}",
                "pn_mean_cost": f"{pn_cost:.4f}",
                "delta_cost": f"{delta:+.4f}",
                "winner": winner,
                "am_std": f"{am.get('std_cost', 0):.4f}",
                "pn_std": f"{pn.get('std_cost', 0):.4f}",
                "am_replay": f"{am.get('mean_replay_cost', 0):.4f}",
                "pn_replay": f"{pn.get('mean_replay_cost', 0):.4f}",
                "am_distance": f"{am.get('mean_distance', 0):.4f}",
                "pn_distance": f"{pn.get('mean_distance', 0):.4f}",
                "am_late_pen": f"{am.get('mean_late_penalty', 0):.4f}",
                "pn_late_pen": f"{pn.get('mean_late_penalty', 0):.4f}",
                "am_skipped_pen": f"{am.get('mean_skipped_penalty', 0):.4f}",
                "pn_skipped_pen": f"{pn.get('mean_skipped_penalty', 0):.4f}",
                "am_skipped_total": am.get("total_skipped", 0),
                "pn_skipped_total": pn.get("total_skipped", 0),
                "am_tw_violations": am.get("total_tw_violations", 0),
                "pn_tw_violations": pn.get("total_tw_violations", 0),
            })

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"✓ Comparison CSV written to: {output_path}")


def generate_master_json(results: list[dict], output_path: str):
    """Write paper_stats.json — full nested statistics for plotting/analysis.

    Structure:
    {
        "metadata": { ... },
        "datasets": {
            "n20m1": {
                "customers": 20, "vehicles": 1,
                "models": {
                    "am": {
                        "greedy": { summary... },
                        "sample": { summary... }
                    },
                    "polynet": { ... }
                }
            },
            ...
        }
    }
    """
    master: dict[str, Any] = {
        "metadata": {
            "description": "AM vs PolyNet evaluation on 5 DVRPTW datasets",
            "num_datasets": len(DATASET_ORDER),
            "models": MODEL_ORDER,
            "decodes": DECODE_ORDER,
        },
        "datasets": {},
    }

    for ds in DATASET_ORDER:
        cust, veh = DATASET_CUST_VEH.get(ds, ("?", "?"))
        ds_entry: dict[str, Any] = {
            "customers": cust,
            "vehicles": veh,
            "models": {},
        }

        for model in MODEL_ORDER:
            model_entry = {}
            for decode in DECODE_ORDER:
                # Find matching result
                match = None
                for r in results:
                    if r["dataset"] == ds and r["model"] == model and r["decode"] == decode:
                        match = r
                        break
                model_entry[decode] = match if match else None
            ds_entry["models"][model] = model_entry

        master["datasets"][ds] = ds_entry

    # Also include flat results array for easy pandas loading
    master["results_flat"] = results

    with open(output_path, "w") as f:
        json.dump(master, f, indent=2, default=str)

    print(f"✓ Master JSON written to: {output_path}")


def print_console_table(results: list[dict]):
    """Print a pretty console summary table."""
    # Group by dataset, then model+decode
    ds_order = {tag: i for i, tag in enumerate(DATASET_ORDER)}
    results_sorted = sorted(results, key=lambda r: (
        ds_order.get(r["dataset"], 99),
        MODEL_ORDER.index(r["model"]) if r["model"] in MODEL_ORDER else 99,
        DECODE_ORDER.index(r["decode"]) if r["decode"] in DECODE_ORDER else 99,
    ))

    print()
    print("╔══════════════════════════════════════════════════════════════════════════════════════════╗")
    print("║                         PAPER EVALUATION — CONSOLE SUMMARY                              ║")
    print("╠══════════╤════════╤════════╤══════════╤═══════════╤═══════════╤══════════╤═══════════════╣")
    print("║ Dataset  │ Model  │ Decode │ Mean Cost│ Std Cost  │ Replay    │ Skipped  │ TW Violations ║")
    print("╟──────────┼────────┼────────┼──────────┼───────────┼───────────┼──────────┼───────────────╢")

    for r in results_sorted:
        ds = r["dataset"]
        model = r["model"].upper()
        decode = r["decode"]
        mc = r.get("mean_cost", 0)
        sc = r.get("std_cost", 0)
        rc = r.get("mean_replay_cost", 0)
        sk = r.get("total_skipped", 0)
        tw = r.get("total_tw_violations", 0)

        print(f"║ {ds:<8} │ {model:<6} │ {decode:<6} │ {mc:>8.4f} │ {sc:>9.4f} │ {rc:>9.4f} │ {sk:>8} │ {tw:>13} ║")

    print("╚══════════╧════════╧════════╧══════════╧═══════════╧═══════════╧══════════╧═══════════════╝")
    print()

    # AM vs PolyNet comparison per dataset
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║                      AM vs PolyNet — Head-to-Head                           ║")
    print("╠══════════╤════════╤════════════════╤════════════════╤════════════╤══════════╣")
    print("║ Dataset  │ Decode │ AM Mean Cost   │ PN Mean Cost   │ Δ (AM-PN)  │ Winner   ║")
    print("╟──────────┼────────┼────────────────┼────────────────┼────────────┼──────────╢")

    lookup: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for r in results:
        lookup[(r["dataset"], r["decode"])][r["model"]] = r

    for ds in DATASET_ORDER:
        for dec in DECODE_ORDER:
            pair = lookup.get((ds, dec), {})
            am = pair.get("am", {})
            pn = pair.get("polynet", {})
            if not am or not pn:
                continue

            am_c = am.get("mean_cost", 0)
            pn_c = pn.get("mean_cost", 0)
            delta = am_c - pn_c
            winner = "PolyNet" if delta > 0.0001 else "AM" if delta < -0.0001 else "Tie  "

            print(f"║ {ds:<8} │ {dec:<6} │ {am_c:>14.4f} │ {pn_c:>14.4f} │ {delta:>+10.4f} │ {winner:<8} ║")

    print("╚══════════╧════════╧════════════════╧════════════════╧════════════╧══════════╝")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate paper evaluation results into tables"
    )
    parser.add_argument(
        "--input-dir", type=str, default="output/paper_eval",
        help="Root directory containing per-model/decode subdirectories"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for CSV/JSON files (default: same as --input-dir)"
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir
    os.makedirs(output_dir, exist_ok=True)

    # Discover
    print(f"Scanning: {input_dir}")
    file_list = discover_result_files(input_dir)
    print(f"Found {len(file_list)} result JSON files")

    if not file_list:
        print("No result files found. Run run_paper_eval.sh first.")
        sys.exit(1)

    # Load & summarize
    summaries = []
    loaded = 0
    for finfo in file_list:
        data = load_result_json(finfo["path"])
        if data is None:
            continue
        loaded += 1
        summary = compute_summary(data)
        summary["model"] = finfo["model"]
        summary["decode"] = finfo["decode"]
        summary["dataset"] = finfo["dataset"]
        summaries.append(summary)

    print(f"Loaded {loaded}/{len(file_list)} files successfully")

    # Generate outputs
    summary_path = os.path.join(output_dir, "paper_summary.csv")
    comparison_path = os.path.join(output_dir, "paper_comparison.csv")
    master_json_path = os.path.join(output_dir, "paper_stats.json")

    generate_summary_csv(summaries, summary_path)
    generate_comparison_csv(summaries, comparison_path)
    generate_master_json(summaries, master_json_path)

    # Console display
    print_console_table(summaries)

    print("Done.")
    print(f"  Summary CSV    : {summary_path}")
    print(f"  Comparison CSV : {comparison_path}")
    print(f"  Master JSON    : {master_json_path}")


if __name__ == "__main__":
    main()
