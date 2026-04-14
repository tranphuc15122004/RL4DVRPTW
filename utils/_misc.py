import torch
import math
import os.path
import os
import csv
import random
import numpy as np
from itertools import repeat, zip_longest


def set_random_seed(seed, deterministic = True):
    if seed is None:
        return
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only = True)

def actions_to_routes(actions, batch_size, veh_count):
    routes = [[[] for i in range(veh_count)] for b in range(batch_size)]
    for veh_idx, cust_idx in actions:
        for b, (i,j) in enumerate(zip(veh_idx, cust_idx)):
            routes[b][i.item()].append(j.item())
    return routes


def routes_to_string(routes):
    return '\n'.join(
        '0 -> ' + ' -> '.join(str(j) for j in route)
        for route in routes
        )


""" def export_train_test_stats(args, start_ep, train_stats, test_stats):
    fpath = os.path.join(args.output_dir, "loss_gap.csv")
    with open(fpath, 'a') as f:
        f.write( (' '.join("{: >16}" for _ in range(9)) + '\n').format(
            "#EP", "#LOSS", "#PROB", "#VAL", "#BL", "#NORM", "#TEST_MU", "#TEST_STD", "#TEST_GAP"
            ))
        for ep, (tr,te) in enumerate( zip_longest(train_stats, test_stats, fillvalue=float('nan')), start = start_ep):
            f.write( ("{: >16d}" + ' '.join("{: >16.3g}" for _ in range(8)) + '\n').format(
                ep, *tr, *te)) """


def export_train_test_stats(args, start_ep, train_stats, test_stats):
    os.makedirs(args.output_dir, exist_ok=True)
    fpath = os.path.join(args.output_dir, "loss_gap.csv")

    header = [
        "#EP", "#LOSS", "#PROB", "#VAL", "#BL", "#NORM",
        "#TEST_MU", "#TEST_STD", "#TEST_GAP"
    ]

    write_header = not os.path.exists(fpath)

    def safe_vals(vals, n):
        """Pad hoặc cắt để đủ n giá trị"""
        if vals is None or (isinstance(vals, float) and math.isnan(vals)):
            return [float("nan")] * n
        vals = list(vals)
        return (vals + [float("nan")] * n)[:n]

    with open(fpath, "a") as f:
        if write_header:
            f.write(" ".join(f"{h:>16}" for h in header) + "\n")

        for ep, (tr, te) in enumerate(
            zip_longest(train_stats, test_stats, fillvalue=None),
            start=start_ep
        ):
            tr_vals = safe_vals(tr, 5)
            te_vals = safe_vals(te, 3)

            f.write(
                f"{ep:>16d}" +
                "".join(f"{v:>16.3g}" for v in (*tr_vals, *te_vals)) +
                "\n"
            )
    
def update_train_test_stats(args, ep, train_stats, val_stats):
    os.makedirs(args.output_dir, exist_ok=True)
    fpath = os.path.join(args.output_dir, "train_statistics.csv")

    header = [
        "EP", "LOSS", "PROB", "VAL", "BL", "NORM",
        "POLICY_LOSS", "CRITIC_LOSS", "ENTROPY_LOSS",
        "VAL_MU", "VAL_STD"
    ]

    def to_float(x):
        if isinstance(x, torch.Tensor):
            return x.detach().float().cpu().item()
        try:
            return float(x)
        except Exception:
            return float("nan")

    def get_latest(vals):
        if vals is None:
            return None
        if isinstance(vals, (list, tuple)):
            return vals[-1]
        return vals

    def safe_vals(vals, n):
        vals = get_latest(vals)
        if vals is None:
            return [float("nan")] * n

        vals = list(vals)
        vals = [to_float(v) for v in vals]
        return (vals + [float("nan")] * n)[:n]

    def migrate_train_statistics_file_if_needed():
        if not os.path.exists(fpath):
            return True
        with open(fpath, "r", newline = "") as f:
            first_line = f.readline().strip()
        if first_line == ",".join(header):
            return False

        rows = []
        with open(fpath, "r", newline = "") as f:
            reader = csv.reader(f)
            old_header = next(reader, None)
            for record in reader:
                if not record:
                    continue
                rows.append(record)

        with open(fpath, "w", newline = "") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for record in rows:
                ep_value = record[0] if record else ""
                train_vals = list(record[1:6]) if len(record) > 1 else []
                val_vals = list(record[6:8]) if len(record) > 6 else []
                writer.writerow(
                    [ep_value]
                    + (train_vals + ["nan"] * 5)[:5]
                    + ["nan", "nan", "nan"]
                    + (val_vals + ["nan"] * 2)[:2]
                )
        return False

    write_header = migrate_train_statistics_file_if_needed()

    tr_vals = safe_vals(train_stats, 8)
    va_vals = safe_vals(val_stats, 2)

    row = [ep] + tr_vals + va_vals

    with open(fpath, "a") as f:
        if write_header:
            f.write(",".join(header) + "\n")

        f.write(",".join(f"{v:.6g}" if i > 0 else str(v)
                         for i, v in enumerate(row)) + "\n")

def _pad_with_zeros(src_it):
    yield from src_it
    yield from repeat(0)


def eval_apriori_routes(dyna, routes, rollout_count):
    mean_cost = dyna.nodes.new_zeros(dyna.minibatch_size)
    for c in range(rollout_count):
        dyna.reset()
        routes_it = [[_pad_with_zeros(route) for route in inst_routes] for inst_routes in routes]
        rewards = []
        while not dyna.done:
            cust_idx = dyna.nodes.new_tensor([[next(routes_it[n][i.item()])]
                for n,i in enumerate(dyna.cur_veh_idx)], dtype = torch.int64)
            rewards.append( dyna.step(cust_idx) )
        mean_cost += -torch.stack(rewards).sum(dim = 0).squeeze(-1)
    return mean_cost / rollout_count


def load_old_weights(learner, state_dict):
    learner.load_state_dict(state_dict)
    for layer in learner.cust_encoder.children():
        layer.mha._inv_sqrt_d = layer.mha.key_size_per_head**0.5
    learner.fleet_attention._inv_sqrt_d = learner.fleet_attention.key_size_per_head**0.5
    learner.veh_attention._inv_sqrt_d = learner.veh_attention.key_size_per_head**0.5

