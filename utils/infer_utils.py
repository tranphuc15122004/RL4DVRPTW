"""
Shared inference utilities for AM and PolyNet models.

Provides common functions for:
  - Dataset loading (.pyth / .csv / generate)
  - Model initialization & weight loading
  - Environment setup
  - Running inference (batch, single-instance)
  - Route extraction, verification, diagnostics
  - Cost computation and JSON output
"""

import json
import os
import sys
from argparse import ArgumentParser
from collections import Counter
from typing import Any

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from problems import (
    DVRPTW_Dataset,
    DVRPTW_Environment,
    SDVRPTW_Dataset,
    SDVRPTW_Environment,
    SVRPTW_Environment,
    VRP_Dataset,
    VRP_Environment,
    VRPTW_Dataset,
    VRPTW_Environment,
)
from utils import actions_to_routes, eval_apriori_routes, routes_to_string, set_random_seed

# ---------------------------------------------------------------------------
# Model args file loading
# ---------------------------------------------------------------------------

# Keys from a training args.json that are relevant to inference.
# When --model-args is provided, these fields are loaded from the file and
# set as defaults; any corresponding CLI flag still overrides.
MODEL_ARGS_MAP = {
    # Model architecture
    "model_size": "model_size",
    "layer_count": "layer_count",
    "head_count": "head_count",
    "ff_size": "ff_size",
    "tanh_xplor": "tanh_xplor",
    "cust_k": "cust_k",
    "dropout": "dropout",
    "edge_feat_size": "edge_feat_size",
    "memory_size": "memory_size",
    "lookahead_hidden": "lookahead_hidden",
    "ablation_profile": "ablation_profile",
    "use_edge_features": "use_edge_features",
    "use_memory": "use_memory",
    "use_ownership": "use_ownership",
    "use_lookahead": "use_lookahead",
    "fusion_mode": "fusion_mode",
    "linear_fusion_weights": "linear_fusion_weights",
    # Problem configuration
    "problem_type": "problem_type",
    "customers_count": "customers_count",
    "vehicles_count": "vehicles_count",
    "veh_capa": "veh_capa",
    "veh_speed": "veh_speed",
    "horizon": "horizon",
    "min_cust_count": "min_cust_count",
    "loc_range": "loc_range",
    "dem_range": "dem_range",
    "dur_range": "dur_range",
    "tw_ratio": "tw_ratio",
    "tw_range": "tw_range",
    "deg_of_dyna": "deg_of_dyna",
    "appear_early_ratio": "appear_early_ratio",
    # Environment
    "pending_cost": "pending_cost",
    "late_cost": "late_cost",
    "speed_var": "speed_var",
    "late_prob": "late_prob",
    "slow_down": "slow_down",
    "late_var": "late_var",
}

# Fields that should never be imported from an args file
MODEL_ARGS_SKIP = {
    "config_file", "verbose", "no_cuda", "rng_seed",
    "epoch_count", "iter_count", "batch_size", "learning_rate",
    "rate_decay", "weight_decay", "max_grad_norm", "grad_norm_decay",
    "loss_use_cumul", "amp", "num_workers", "pin_memory",
    "smoke_test", "resource_safe", "max_vram_fraction",
    "regen_train_data_each_epoch",
    "baseline_type", "rollout_count", "rollout_threshold",
    "critic_use_qval", "critic_rate", "critic_decay",
    "adv_norm", "entropy_coef",
    "test_batch_size",
    "ppo_epochs", "ppo_clip_range", "ppo_value_coef", "ppo_entropy_coef",
    "ppo_gamma", "ppo_gae_lambda", "ppo_adv_norm", "ppo_target_kl",
    "output_dir", "checkpoint_period", "resume_state", "model_weight",
    "pend_cost_growth", "late_cost_growth",
}


def load_model_args_from_file(path):
    """Load a training args.json and return a dict of inference-relevant keys.

    Only keys listed in MODEL_ARGS_MAP are returned; training-only fields
    are silently skipped.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model args file not found: '{path}'")

    with open(path, "r") as f:
        raw = json.load(f)

    extracted = {}
    for src_key, dst_key in MODEL_ARGS_MAP.items():
        if src_key in raw and raw[src_key] is not None:
            extracted[dst_key] = raw[src_key]

    # Report what was loaded
    print(f"[Model args] Loaded {len(extracted)} parameters from '{path}'")
    for k, v in sorted(extracted.items()):
        print(f"  {k} = {v}")

    return extracted


# Module-level defaults from _args.py; used to detect whether a value was
# explicitly provided via CLI or left at the built-in default.
# The mapping below links argparse attribute names → module-level constant names.
import utils._args as _args_module
_ARGS_DEFAULT_MAP = {
    "customers_count": "CUST_COUNT",
    "vehicles_count": "VEH_COUNT",
    "veh_capa": "VEH_CAPA",
    "veh_speed": "VEH_SPEED",
    "horizon": "HORIZON",
    "min_cust_count": "MIN_CUST_COUNT",
    "loc_range": "LOC_RANGE",
    "dem_range": "DEM_RANGE",
    "dur_range": "DUR_RANGE",
    "tw_ratio": "TW_RATIO",
    "tw_range": "TW_RANGE",
    "deg_of_dyna": "DEG_OF_DYN",
    "appear_early_ratio": "APPEAR_EARLY_RATIO",
    "pending_cost": "PEND_COST",
    "late_cost": "LATE_COST",
    "speed_var": "SPEED_VAR",
    "late_prob": "LATE_PROB",
    "slow_down": "SLOW_DOWN",
    "late_var": "LATE_VAR",
    "model_size": "MODEL_SIZE",
    "layer_count": "LAYER_COUNT",
    "head_count": "HEAD_COUNT",
    "ff_size": "FF_SIZE",
    "tanh_xplor": "TANH_XPLOR",
    "cust_k": "CUST_K",
    "edge_feat_size": "EDGE_FEAT_SIZE",
    "memory_size": "MEMORY_SIZE",
    "lookahead_hidden": "LOOKAHEAD_HIDDEN",
    "dropout": "MODEL_DROPOUT",
    "ablation_profile": "ABLATION_PROFILE",
    "use_edge_features": "USE_EDGE_FEATURES",
    "use_memory": "USE_MEMORY",
    "use_ownership": "USE_OWNERSHIP",
    "use_lookahead": "USE_LOOKAHEAD",
    "fusion_mode": "FUSION_MODE",
    "linear_fusion_weights": "LINEAR_FUSION_WEIGHTS",
    "test_batch_size": "TEST_BATCH_SIZE",
    "output_dir": "OUTPUT_DIR",
    "checkpoint_period": "CHECKPOINT_PERIOD",
    "model_weight": "MODEL_WEIGHT",
}
_MODULE_DEFAULTS = {}
for _attr, _const_name in _ARGS_DEFAULT_MAP.items():
    if hasattr(_args_module, _const_name):
        _MODULE_DEFAULTS[_attr] = getattr(_args_module, _const_name)


def merge_model_args_into_namespace(args, model_args_dict):
    """Merge model-args values into *args*, preserving values that were
    explicitly set by the user on the command line.

    Heuristic: a field is considered "user-set" if its current value differs
    from the module-level default or is a scalar/list that was likely provided
    via CLI.  Because the module-level defaults are all uppercase constants in
    ``_args.py``, we compare against ``_MODULE_DEFAULTS``.
    """
    if model_args_dict is None:
        return args

    for key, value in model_args_dict.items():
        current = getattr(args, key, None)
        # If current is None, the key doesn't exist → always apply
        if current is None:
            setattr(args, key, value)
            continue

        # Check if current matches the module-level default
        default_val = _MODULE_DEFAULTS.get(key)
        if default_val is not None:
            if isinstance(current, (list, tuple)) and isinstance(default_val, (list, tuple)):
                # For compound defaults (e.g. loc_range), compare element-wise
                if list(current) == list(default_val):
                    setattr(args, key, value)
            elif current == default_val:
                setattr(args, key, value)
            # If different from default, user likely set it via CLI → keep current
        # If there's no known default, keep current (conservative)

    return args


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

DATASET_REGISTRY = {
    "vrp": VRP_Dataset,
    "vrptw": VRPTW_Dataset,
    "svrptw": VRPTW_Dataset,
    "sdvrptw": SDVRPTW_Dataset,
    "dvrptw": DVRPTW_Dataset,
}

ENVIRONMENT_REGISTRY = {
    "vrp": VRP_Environment,
    "vrptw": VRPTW_Environment,
    "svrptw": SVRPTW_Environment,
    "sdvrptw": SDVRPTW_Environment,
    "dvrptw": DVRPTW_Environment,
}


def dataset_cls(problem_type: str):
    return DATASET_REGISTRY.get(problem_type)


def environment_cls(problem_type: str):
    return ENVIRONMENT_REGISTRY.get(problem_type)


def build_gen_params(args):
    gen_params = [
        args.customers_count,
        args.vehicles_count,
        args.veh_capa,
        args.veh_speed,
        args.min_cust_count,
        args.loc_range,
        args.dem_range,
    ]
    if args.problem_type != "vrp":
        gen_params.extend([args.horizon, args.dur_range, args.tw_ratio, args.tw_range])
    if args.problem_type in ("sdvrptw", "dvrptw"):
        gen_params.extend([args.deg_of_dyna, args.appear_early_ratio])
    return gen_params


def load_dataset_from_file(dataset_cls, path):
    """Load a dataset from a .pyth file (torch-serialized)."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dataset_cls):
        return obj
    if isinstance(obj, dict):
        return dataset_cls(**obj)
    raise ValueError(f"Unsupported dataset file format at '{path}'")


def build_dataset(args, dataset_cls):
    """Build a dataset from --data-csv, --data-file, or generate on-the-fly."""
    if args.data_csv is not None:
        if args.problem_type != "dvrptw":
            raise ValueError("--data-csv is currently supported for --problem-type dvrptw only")
        return dataset_cls.from_csv(
            args.data_csv,
            veh_count=args.vehicles_count,
            veh_capa=args.veh_capa,
            veh_speed=args.veh_speed,
        )
    if args.data_file is not None:
        return load_dataset_from_file(dataset_cls, args.data_file)
    gen_params = build_gen_params(args)
    return dataset_cls.generate(args.test_batch_size, *gen_params)


def clone_dataset(data):
    """Clone a dataset (used to keep un-normalized copy)."""
    nodes = data.nodes.clone()
    cust_mask = None if data.cust_mask is None else data.cust_mask.clone()
    return data.__class__(data.veh_count, data.veh_capa, data.veh_speed, nodes, cust_mask)


def build_env_params(args):
    env_params = [args.pending_cost]
    if args.problem_type != "vrp":
        env_params.append(args.late_cost)
        if args.problem_type not in ("vrptw", "dvrptw"):
            env_params.extend([args.speed_var, args.late_prob, args.slow_down, args.late_var])
    return env_params


def build_env(args, data, env_cls, env_params, device):
    """Create the environment and move its tensors to the target device."""
    env = env_cls(data, None, None, *env_params)
    env.nodes = env.nodes.to(device)
    if env.init_cust_mask is not None:
        env.init_cust_mask = env.init_cust_mask.to(device)
    return env


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def init_am_model(args, env_cls, device):
    """Initialize an AM_DVRPTW model."""
    from am import AM_DVRPTW
    learner = AM_DVRPTW(
        cust_feat_size=env_cls.CUST_FEAT_SIZE,
        veh_state_size=env_cls.VEH_STATE_SIZE,
        model_size=args.model_size,
        layer_count=args.layer_count,
        head_count=args.head_count,
        ff_size=args.ff_size,
        tanh_xplor=args.tanh_xplor,
        greedy=args.greedy,
    )
    learner.to(device)
    return learner


def init_polynet_model(args, env_cls, device):
    """Initialize a PolyNet_DVRPTW model."""
    from polynet import PolyNet_DVRPTW
    poly_k = args.cust_k if args.cust_k is not None else (args.customers_count + 1)
    learner = PolyNet_DVRPTW(
        env_cls.CUST_FEAT_SIZE,
        env_cls.VEH_STATE_SIZE,
        model_size=args.model_size,
        layer_count=args.layer_count,
        head_count=args.head_count,
        ff_size=args.ff_size,
        tanh_xplor=args.tanh_xplor,
        greedy=args.greedy,
        k=max(2, int(poly_k)),
    )
    learner.to(device)
    return learner


def load_model_weights(path, learner):
    """Load model weights from a checkpoint, with informative logging."""
    if path is None:
        raise ValueError("Please provide --model-weight path for inference")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint and isinstance(checkpoint["model"], dict):
        state_dict = checkpoint["model"]
        source = "checkpoint['model']"
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
        source = "state_dict"
    else:
        raise ValueError(f"Unsupported model checkpoint format at '{path}'")

    try:
        res = learner.load_state_dict(state_dict, strict=False)
    except RuntimeError as e:
        raise RuntimeError(
            "Model checkpoint is incompatible with current architecture. "
            f"Please use matching model hyperparameters/config.\n{e}"
        )

    missing = res.missing_keys if hasattr(res, "missing_keys") else []
    unexpected = res.unexpected_keys if hasattr(res, "unexpected_keys") else []
    matched = sum(1 for k in state_dict if k in learner.state_dict())
    print(
        f"Loaded model from '{path}' ({source}) | "
        f"provided={len(state_dict)}, matched={matched}, "
        f"missing={len(missing)}, unexpected={len(unexpected)}"
    )


def warmup_model(args, data, env_cls, env_params, learner):
    """Run a single warmup forward pass to initialize lazy modules."""
    with torch.no_grad():
        device = next(learner.parameters()).device
        warmup_env = env_cls(data, data.nodes[:1].to(device), None, *env_params)
        _ = learner(warmup_env)


# ---------------------------------------------------------------------------
# Inference runners
# ---------------------------------------------------------------------------

def run_single_inference(env, learner):
    """Run inference on an already-initialized environment.

    Returns:
        routes: list of routes per instance
        costs: 1-D tensor of (negative total reward) per instance
    """
    with torch.no_grad():
        actions, _, rewards = learner(env)
    costs = -torch.stack(rewards).sum(dim=0).squeeze(-1)
    routes = actions_to_routes(actions, env.minibatch_size, env.veh_count)
    return routes, costs


def run_inference(args, env, learner):
    """Run inference; for stochastic problems run multiple rollouts."""
    if args.problem_type.startswith("s"):
        all_costs = []
        last_routes = None
        for _ in range(args.stoch_rollouts):
            routes, costs = run_single_inference(env, learner)
            all_costs.append(costs)
            last_routes = routes
        mean_costs = torch.stack(all_costs, dim=0).mean(dim=0)
        return last_routes, mean_costs
    return run_single_inference(env, learner)


def run_single_instance_inference(args, data, env_cls, env_params, learner, device):
    """Run inference on a single-instance dataset (batch_size=1).

    This is a convenience wrapper for single CSV inference.
    """
    env = build_env(args, data, env_cls, env_params, device)
    return run_inference(args, env, learner)


# ---------------------------------------------------------------------------
# Route verification and diagnostics
# ---------------------------------------------------------------------------

def active_customer_set(data, inst_idx):
    if data.cust_mask is None:
        return set(range(1, data.nodes_count))
    mask = data.cust_mask[inst_idx].cpu()
    return {j for j in range(1, data.nodes_count) if not bool(mask[j].item())}


def route_diag_for_instance(data, routes, inst_idx):
    active = active_customer_set(data, inst_idx)
    visited = []
    for route in routes[inst_idx]:
        visited.extend([node for node in route if node != 0])

    counts = Counter(visited)
    dup = sorted([node for node, c in counts.items() if c > 1])
    missing = sorted(active - set(visited))
    extra = sorted(set(visited) - active)
    return {
        "active_customers": len(active),
        "visited_customers": len(set(visited)),
        "visit_steps": len(visited),
        "missing_count": len(missing),
        "duplicate_count": len(dup),
        "extra_count": len(extra),
        "missing_head": missing[:10],
        "duplicate_head": dup[:10],
        "extra_head": extra[:10],
    }


def verify_routes_cost(data, env_cls, env_params, routes, model_costs, rollouts=1):
    """Replay routes and compare costs with model predictions."""
    verify_env = env_cls(data, None, None, *env_params)
    verify_costs = eval_apriori_routes(verify_env, routes, max(1, int(rollouts)))
    abs_diff = (verify_costs - model_costs.cpu()).abs()

    print(f"Route verification: replay_mean={verify_costs.mean().item():.4f}, "
          f"model_mean={model_costs.mean().item():.4f}, "
          f"max_abs_diff={abs_diff.max().item():.6f}")

    for idx in range(min(3, len(routes))):
        diag = route_diag_for_instance(data, routes, idx)
        print(
            f"  Instance #{idx} | replay={verify_costs[idx].item():.4f} "
            f"model={model_costs[idx].item():.4f} diff={abs_diff[idx].item():.6f} "
            f"missing={diag['missing_count']} dup={diag['duplicate_count']} "
            f"extra={diag['extra_count']}"
        )
        if diag["missing_count"] > 0:
            print(f"    missing sample: {diag['missing_head']}")
        if diag["duplicate_count"] > 0:
            print(f"    duplicate sample: {diag['duplicate_head']}")
        if diag["extra_count"] > 0:
            print(f"    extra sample: {diag['extra_head']}")

    return verify_costs


def replay_routes_cost(data, env_cls, env_params, routes, rollouts=1):
    replay_env = env_cls(data, None, None, *env_params)
    return eval_apriori_routes(replay_env, routes, max(1, int(rollouts)))


def check_route_constraints(data, routes, eps=1e-9):
    """Check time-window and appearance violations for all routes."""
    nodes = data.nodes.detach().cpu()
    veh_speed = float(data.veh_speed)
    if veh_speed <= 0:
        raise ValueError("veh_speed must be positive for constraint checking")

    all_reports = []
    for inst_idx, inst_routes in enumerate(routes):
        node = nodes[inst_idx]
        depot_xy = node[0, :2]

        tw_violations = []
        appearance_violations = []

        for veh_idx, route in enumerate(inst_routes):
            cur_xy = depot_xy.clone()
            cur_time = 0.0
            for step_idx, cust_id in enumerate(route):
                if cust_id < 0 or cust_id >= node.size(0):
                    continue

                dest = node[cust_id]
                dist = torch.dist(cur_xy, dest[:2], p=2).item()
                arrival = cur_time + dist / veh_speed
                open_t = float(dest[3].item())
                close_t = float(dest[4].item())
                service_t = float(dest[5].item())
                appear_t = float(dest[6].item()) if dest.numel() >= 7 else 0.0

                start_service = max(arrival, open_t)

                if cust_id != 0:
                    if start_service > close_t + eps:
                        tw_violations.append({
                            "vehicle": int(veh_idx),
                            "step": int(step_idx),
                            "customer": int(cust_id),
                            "start_service": float(start_service),
                            "close": float(close_t),
                            "late_by": float(start_service - close_t),
                        })
                    if start_service + eps < appear_t:
                        appearance_violations.append({
                            "vehicle": int(veh_idx),
                            "step": int(step_idx),
                            "customer": int(cust_id),
                            "start_service": float(start_service),
                            "appearance": float(appear_t),
                            "early_by": float(appear_t - start_service),
                        })

                cur_time = start_service + service_t
                cur_xy = dest[:2]

        all_reports.append({
            "instance": int(inst_idx),
            "tw_violation_count": len(tw_violations),
            "appearance_violation_count": len(appearance_violations),
            "tw_violations_head": tw_violations[:10],
            "appearance_violations_head": appearance_violations[:10],
        })

    return all_reports


def compute_cost_components(data, routes, pending_cost, late_cost, eps=1e-9):
    """Compute detailed cost breakdown (distance, late penalty, skipped orders)."""
    nodes = data.nodes.detach().cpu()
    veh_speed = float(data.veh_speed)
    if veh_speed <= 0:
        raise ValueError("veh_speed must be positive for component cost checking")

    reports = []
    for inst_idx, inst_routes in enumerate(routes):
        node = nodes[inst_idx]
        depot_xy = node[0, :2]
        visited_customers = set()

        total_distance = 0.0
        total_late_time = 0.0

        for route in inst_routes:
            cur_xy = depot_xy.clone()
            cur_time = 0.0
            for cust_id in route:
                if cust_id < 0 or cust_id >= node.size(0):
                    continue

                dest = node[cust_id]
                dist = torch.dist(cur_xy, dest[:2], p=2).item()
                arrival = cur_time + dist / veh_speed
                open_t = float(dest[3].item())
                close_t = float(dest[4].item())
                service_t = float(dest[5].item())

                start_service = max(arrival, open_t)
                late = max(0.0, start_service - close_t)

                total_distance += dist
                total_late_time += late

                if cust_id != 0:
                    visited_customers.add(int(cust_id))

                cur_time = start_service + service_t
                cur_xy = dest[:2]

        active = active_customer_set(data, inst_idx)
        skipped_orders = len(active - visited_customers)

        late_penalty = float(late_cost) * total_late_time
        skipped_penalty = float(pending_cost) * skipped_orders
        total_cost = total_distance + late_penalty + skipped_penalty
        reward = -total_cost

        reports.append({
            "instance": int(inst_idx),
            "reward": float(reward),
            "total_cost": float(total_cost),
            "distance": float(total_distance),
            "late_time": float(total_late_time),
            "late_penalty": float(late_penalty),
            "skipped_orders": int(skipped_orders),
            "skipped_penalty": float(skipped_penalty),
        })

    return reports


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_routes(routes, costs, max_instances=3):
    """Print routes and costs for a limited number of instances."""
    n_show = min(max_instances, len(routes))
    for idx in range(n_show):
        print("=" * 60)
        print(f"Instance #{idx} | cost={costs[idx].item():.4f}")
        print(routes_to_string(routes[idx]))


def save_json(path, routes, normalized_costs,
              raw_replay_costs=None,
              route_diagnostics=None,
              constraint_diagnostics=None,
              raw_cost_components=None,
              normalized_cost_components=None):
    """Save full inference results to a JSON file."""
    if raw_replay_costs is None:
        raw_replay_costs = normalized_costs
    if route_diagnostics is None:
        route_diagnostics = []
    if constraint_diagnostics is None:
        constraint_diagnostics = []
    if raw_cost_components is None:
        raw_cost_components = []
    if normalized_cost_components is None:
        normalized_cost_components = []

    payload = {
        "costs": [float(v) for v in normalized_costs.cpu().tolist()],
        "normalized_costs": [float(v) for v in normalized_costs.cpu().tolist()],
        "raw_replay_costs": [float(v) for v in raw_replay_costs.cpu().tolist()],
        "skipped_customers_count": [int(d.get("missing_count", 0)) for d in route_diagnostics],
        "total_skipped_customers": int(sum(int(d.get("missing_count", 0)) for d in route_diagnostics)),
        "route_diagnostics": route_diagnostics,
        "tw_violations_count": [int(d.get("tw_violation_count", 0)) for d in constraint_diagnostics],
        "appearance_violations_count": [int(d.get("appearance_violation_count", 0)) for d in constraint_diagnostics],
        "total_tw_violations": int(sum(int(d.get("tw_violation_count", 0)) for d in constraint_diagnostics)),
        "total_appearance_violations": int(sum(int(d.get("appearance_violation_count", 0)) for d in constraint_diagnostics)),
        "constraint_diagnostics": constraint_diagnostics,
        "raw_cost_components": raw_cost_components,
        "normalized_cost_components": normalized_cost_components,
        "routes": routes,
    }
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# Full inference pipeline (reusable by both AM and PolyNet)
# ---------------------------------------------------------------------------

def full_inference_pipeline(args, model_init_fn, device=None):
    """Run the full inference pipeline: load data, init model, infer, verify, report.

    Args:
        args: parsed arguments (from parse_args + parse_infer_args)
        model_init_fn: callable (args, env_cls, device) -> nn.Module
        device: torch device (auto-detected if None)

    Returns:
        dict with keys: routes, costs, raw_replay_costs, route_diagnostics,
                        constraint_diagnostics, raw_cost_components,
                        normalized_cost_components
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    set_random_seed(args.rng_seed, deterministic=True)

    dset_cls = dataset_cls(args.problem_type)
    env_cls = environment_cls(args.problem_type)
    if dset_cls is None or env_cls is None:
        raise ValueError(f"Unsupported problem type '{args.problem_type}'")

    # Load / generate dataset
    data = build_dataset(args, dset_cls)
    raw_data = clone_dataset(data)

    if not args.no_normalize:
        data.normalize()

    # Build environment
    env_params = build_env_params(args)
    env = build_env(args, data, env_cls, env_params, device)

    # Initialize model, warmup, load weights
    learner = model_init_fn(args, env_cls, device)
    learner.eval()

    warmup_model(args, data, env_cls, env_params, learner)
    load_model_weights(args.model_weight, learner)
    learner.eval()

    # Run inference
    routes, costs = run_inference(args, env, learner)

    # Replay on raw (un-normalized) data
    raw_replay_costs = replay_routes_cost(raw_data, env_cls, env_params, routes,
                                          rollouts=args.verify_rollouts)

    # Diagnostics
    route_diagnostics = [route_diag_for_instance(data, routes, idx)
                         for idx in range(len(routes))]
    constraint_diagnostics = check_route_constraints(raw_data, routes)
    raw_cost_components = compute_cost_components(raw_data, routes,
                                                  args.pending_cost, args.late_cost)
    normalized_cost_components = compute_cost_components(data, routes,
                                                          args.pending_cost, args.late_cost)

    # Summary statistics
    total_skipped = sum(d["missing_count"] for d in route_diagnostics)
    total_tw_viol = sum(d["tw_violation_count"] for d in constraint_diagnostics)
    total_appear_viol = sum(d["appearance_violation_count"] for d in constraint_diagnostics)

    mean = costs.mean().item()
    std = costs.std().item() if costs.numel() > 1 else 0.0
    print(f"Inference done on {costs.numel()} instance(s): mean={mean:.4f}, std={std:.4f}")
    print(f"Cost summary: normalized_cost_mean={costs.mean().item():.4f}, "
          f"raw_replay_cost_mean={raw_replay_costs.mean().item():.4f}")

    for idx in range(min(3, costs.numel())):
        print(f"  Instance #{idx} | normalized_cost={costs[idx].item():.4f} | "
              f"raw_replay_cost={raw_replay_costs[idx].item():.4f}")

    print("Cost components (raw scale):")
    for idx in range(min(3, len(raw_cost_components))):
        c = raw_cost_components[idx]
        print(f"  Instance #{idx} | reward={c['reward']:.4f} | "
              f"distance={c['distance']:.4f} | late_time={c['late_time']:.4f} | "
              f"skipped_orders={c['skipped_orders']}")

    print(f"Skipped customers: total={total_skipped} | per_instance="
          f"{[d['missing_count'] for d in route_diagnostics[:min(10, len(route_diagnostics))]]}")
    print(f"Constraint violations: total_tw={total_tw_viol} | "
          f"total_appearance={total_appear_viol}")

    for idx in range(min(3, len(constraint_diagnostics))):
        rep = constraint_diagnostics[idx]
        print(f"  Instance #{idx} | tw_violations={rep['tw_violation_count']} | "
              f"appearance_violations={rep['appearance_violation_count']}")

    print_routes(routes, costs, args.max_print_instances)

    if args.verify_routes:
        verify_routes_cost(data, env_cls, env_params, routes, costs,
                           rollouts=args.verify_rollouts)

    if args.save_json is not None:
        save_json(
            args.save_json,
            routes,
            costs,
            raw_replay_costs,
            route_diagnostics,
            constraint_diagnostics,
            raw_cost_components,
            normalized_cost_components,
        )
        print(f"Saved inference outputs to '{args.save_json}'")

    return {
        "routes": routes,
        "costs": costs,
        "raw_replay_costs": raw_replay_costs,
        "route_diagnostics": route_diagnostics,
        "constraint_diagnostics": constraint_diagnostics,
        "raw_cost_components": raw_cost_components,
        "normalized_cost_components": normalized_cost_components,
    }


# ---------------------------------------------------------------------------
# Argument parser for inference-specific flags
# ---------------------------------------------------------------------------

def add_infer_args(parser):
    """Add inference-specific arguments to an existing ArgumentParser."""
    parser.add_argument("--data-csv", type=str, default=None,
                        help="Path to a CSV scenario file (supported for dvrptw)")
    parser.add_argument("--data-file", type=str, default=None,
                        help="Path to a saved .pyth dataset file")
    parser.add_argument("--model-args", type=str, default=None,
                        help="Path to a training args.json to load model/config defaults from")
    parser.add_argument("--no-normalize", action="store_true", default=False,
                        help="Disable dataset normalization before inference")
    decode_group = parser.add_mutually_exclusive_group()
    decode_group.add_argument("--greedy", action="store_true", default=False,
                              help="Use greedy decoding during inference")
    decode_group.add_argument("--sample", action="store_true", default=False,
                              help="Use sampling decoding instead of greedy")
    parser.add_argument("--stoch-rollouts", type=int, default=100,
                        help="Rollout count for stochastic problems (svrptw/sdvrptw)")
    parser.add_argument("--max-print-instances", type=int, default=3,
                        help="Number of instances to print routes for")
    parser.add_argument("--save-json", type=str, default=None,
                        help="Optional path to save routes/costs JSON")
    parser.add_argument("--verify-routes", action="store_true", default=True,
                        help="Replay returned routes and compare replayed cost with model cost")
    parser.add_argument("--no-verify-routes", action="store_false", dest="verify_routes")
    parser.add_argument("--verify-rollouts", type=int, default=1,
                        help="Rollout count for route replay verification")
    return parser


def parse_infer_args(argv=None):
    """Parse inference-specific args merged with general args.

    If ``--model-args`` is provided, model architecture and problem parameters
    from that training ``args.json`` are loaded as defaults.  Any explicit CLI
    flag still overrides the JSON value.
    """
    if argv is None:
        argv = sys.argv[1:]

    from utils import parse_args

    # ---- 1. Parse general + inference args normally ----
    infer_parser = ArgumentParser(add_help=False)
    add_infer_args(infer_parser)
    infer_args, remain = infer_parser.parse_known_args(argv)
    args = parse_args(remain)

    # Copy inference args into the main args namespace
    for key in vars(infer_args):
        setattr(args, key, getattr(infer_args, key))

    # Set greedy/sample
    args.greedy = not infer_args.sample
    args.sample = infer_args.sample

    # ---- 2. Load model args file and merge (CLI values take precedence) ----
    if args.model_args is not None:
        model_dict = load_model_args_from_file(args.model_args)
        merge_model_args_into_namespace(args, model_dict)

    return args


# ---------------------------------------------------------------------------
# CSV batch helpers
# ---------------------------------------------------------------------------

def discover_csv_files(root_dir):
    """Recursively discover all CSV files under *root_dir*.

    Returns a list of absolute paths.
    """
    csv_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith(".csv") and not fn.startswith("."):
                csv_files.append(os.path.join(dirpath, fn))
    return sorted(csv_files)


def guess_problem_params_from_csv_num_nodes(csv_path):
    """Rough heuristic: read node count from a CSV to guess customers_count."""
    with open(csv_path, "r") as f:
        # Subtract 1 for header, 1 for depot → number of customers
        n = sum(1 for _ in f) - 1
    return max(1, n - 1)  # customers = nodes - 1 (depot)
