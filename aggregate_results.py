import os
import json
import csv
import argparse
from pathlib import Path

def aggregate_results(input_dir, output_csv):
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Directory {input_dir} does not exist.")
        return

    json_files = list(input_path.glob("*.json"))
    if not json_files:
        print(f"No json files found in {input_dir}")
        return

    all_data = []
    
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract basic info from filename, assuming format like h100c101_am.json
            filename = file_path.stem
            
            row = {
                "instance": filename,
                "cost": data.get("costs", [None])[0] if data.get("costs") else None,
                "normalized_cost": data.get("normalized_costs", [None])[0] if data.get("normalized_costs") else None,
                "raw_replay_cost": data.get("raw_replay_costs", [None])[0] if data.get("raw_replay_costs") else None,
                "skipped_customers_count": data.get("skipped_customers_count", [None])[0] if data.get("skipped_customers_count") else None,
                "total_skipped_customers": data.get("total_skipped_customers"),
                "total_tw_violations": data.get("total_tw_violations"),
                "total_appearance_violations": data.get("total_appearance_violations")
            }
            
            # Extract raw component costs if available
            raw_components = data.get("raw_cost_components", [])
            if raw_components and len(raw_components) > 0:
                comp = raw_components[0]
                row["reward"] = comp.get("reward")
                row["total_cost_comp"] = comp.get("total_cost")
                row["distance"] = comp.get("distance")
                row["late_time"] = comp.get("late_time")
                row["late_penalty"] = comp.get("late_penalty")
                row["skipped_orders"] = comp.get("skipped_orders")
                row["skipped_penalty"] = comp.get("skipped_penalty")
                row["time_limit_penalty"] = comp.get("time_limit_penalty")
            
            # Extract route diagnostics if available
            route_diag = data.get("route_diagnostics", [])
            if route_diag and len(route_diag) > 0:
                diag = route_diag[0]
                row["active_customers"] = diag.get("active_customers")
                row["visited_customers"] = diag.get("visited_customers")
                row["visit_steps"] = diag.get("visit_steps")
                row["missing_count"] = diag.get("missing_count")
                row["duplicate_count"] = diag.get("duplicate_count")
                row["extra_count"] = diag.get("extra_count")

            # Extract solve time if available (or time taken)
            row["solve_time"] = data.get("time") if data.get("time") is not None else data.get("solve_time")
                
            all_data.append(row)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if not all_data:
        print("No valid data could be extracted.")
        return

    # Get all unique keys for CSV header
    all_keys = set()
    for row in all_data:
        all_keys.update(row.keys())
    
    # Define a preferred column order
    preferred_order = [
        "instance", "cost", "normalized_cost", "raw_replay_cost", 
        "distance", "late_time", "late_penalty", "skipped_orders", "skipped_penalty", "time_limit_penalty", "reward", "total_cost_comp",
        "solve_time",
        "total_skipped_customers", "skipped_customers_count", 
        "total_tw_violations", "total_appearance_violations",
        "active_customers", "visited_customers", "visit_steps", "missing_count", "duplicate_count", "extra_count"
    ]
    
    header = [k for k in preferred_order if k in all_keys]
    header.extend([k for k in all_keys if k not in header])

    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        all_data = sorted(all_data, key=lambda x: x.get("instance", ""))
        writer.writerows(all_data)
        
    print(f"Aggregated {len(all_data)} files into {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate inference JSON results into a CSV.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing JSON files.", default="output/am_infer")
    parser.add_argument("--output_csv", type=str, required=True, help="Output CSV file path.", default="output/am_infer/aggregated_full.csv")
    args = parser.parse_args()
    
    aggregate_results(args.input_dir, args.output_csv)
