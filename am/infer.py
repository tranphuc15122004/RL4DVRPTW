from argparse import ArgumentParser
import json
import os
import sys
from collections import Counter

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

from am import AM_DVRPTW
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
from utils import actions_to_routes, eval_apriori_routes, parse_args, routes_to_string, set_random_seed


'''
python3 am/infer.py --problem-type dvrptw --config-file data/_AM/args.json --model-weight data/_AM/chkpt_best.pyth --customers-count 100 --vehicles-count 5 --veh-capa 1300 --veh-speed 1 --test-batch-size 256 --save-json output/infer_result.json
'''


def parse_infer_args(argv = None):
	if argv is None:
		argv = sys.argv[1:]

	infer_parser = ArgumentParser(add_help = False)
	infer_parser.add_argument("--data-csv", type = str, default = None,
			help = "Path to a CSV scenario file (supported for dvrptw)")
	infer_parser.add_argument("--data-file", type = str, default = None,
			help = "Path to a saved .pyth dataset file")
	infer_parser.add_argument("--model-args", type = str, default = None,
			help = "Path to a training args.json to auto-configure model/problem params")
	infer_parser.add_argument("--no-normalize", action = "store_true", default = False,
			help = "Disable dataset normalization before inference")
	decode_group = infer_parser.add_mutually_exclusive_group()
	decode_group.add_argument("--greedy", action = "store_true", default = False,
			help = "Use greedy decoding during inference")
	decode_group.add_argument("--sample", action = "store_true", default = False,
			help = "Use sampling decoding instead of greedy")
	infer_parser.add_argument("--stoch-rollouts", type = int, default = 100,
			help = "Rollout count for stochastic problems (svrptw/sdvrptw)")
	infer_parser.add_argument("--max-print-instances", type = int, default = 3,
			help = "Number of instances to print routes for")
	infer_parser.add_argument("--save-json", type = str, default = None,
			help = "Optional path to save routes/costs JSON")
	infer_parser.add_argument("--verify-routes", action = "store_true", default = True,
			help = "Replay returned routes and compare replayed cost with model cost")
	infer_parser.add_argument("--no-verify-routes", action = "store_false", dest = "verify_routes")
	infer_parser.add_argument("--verify-rollouts", type = int, default = 1,
			help = "Rollout count for route replay verification")

	infer_args, remain = infer_parser.parse_known_args(argv)
	args = parse_args(remain)

	args.data_csv = infer_args.data_csv
	args.data_file = infer_args.data_file
	args.model_args = infer_args.model_args
	args.no_normalize = infer_args.no_normalize
	args.greedy = not infer_args.sample
	args.sample = infer_args.sample
	args.stoch_rollouts = infer_args.stoch_rollouts
	args.max_print_instances = infer_args.max_print_instances
	args.save_json = infer_args.save_json
	args.verify_routes = infer_args.verify_routes
	args.verify_rollouts = infer_args.verify_rollouts

	# Load model args file if provided
	if args.model_args is not None:
		from utils.infer_utils import load_model_args_from_file, merge_model_args_into_namespace
		model_dict = load_model_args_from_file(args.model_args)
		merge_model_args_into_namespace(args, model_dict)

	return args


def _dataset_cls(problem_type):
	return {
		"vrp": VRP_Dataset,
		"vrptw": VRPTW_Dataset,
		"svrptw": VRPTW_Dataset,
		"sdvrptw": SDVRPTW_Dataset,
		"dvrptw": DVRPTW_Dataset,
	}.get(problem_type)


def _environment_cls(problem_type):
	return {
		"vrp": VRP_Environment,
		"vrptw": VRPTW_Environment,
		"svrptw": SVRPTW_Environment,
		"sdvrptw": SDVRPTW_Environment,
		"dvrptw": DVRPTW_Environment,
	}.get(problem_type)


def _build_gen_params(args):
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


def _load_dataset_from_file(dataset_cls, path):
	obj = torch.load(path, map_location = "cpu", weights_only = False)
	if isinstance(obj, dataset_cls):
		return obj
	if isinstance(obj, dict):
		return dataset_cls(**obj)
	raise ValueError("Unsupported dataset file format at '{}'".format(path))


def _build_dataset(args, dataset_cls):
	if args.data_csv is not None:
		if args.problem_type != "dvrptw":
			raise ValueError("--data-csv is currently supported for --problem-type dvrptw only")
		data = dataset_cls.from_csv(
			args.data_csv,
			veh_count = args.vehicles_count,
			veh_capa = args.veh_capa,
			veh_speed = args.veh_speed,
		)
		return data

	if args.data_file is not None:
		return _load_dataset_from_file(dataset_cls, args.data_file)

	gen_params = _build_gen_params(args)
	return dataset_cls.generate(args.test_batch_size, *gen_params)


def _clone_dataset(data):
	nodes = data.nodes.clone()
	cust_mask = None if data.cust_mask is None else data.cust_mask.clone()
	return data.__class__(data.veh_count, data.veh_capa, data.veh_speed, nodes, cust_mask)


def _build_env_params(args):
	env_params = [args.pending_cost]
	if args.problem_type != "vrp":
		env_params.append(args.late_cost)
		if args.problem_type not in ("vrptw", "dvrptw"):
			env_params.extend([args.speed_var, args.late_prob, args.slow_down, args.late_var])
	return env_params


def _init_model(args, env_cls, device):
	learner = AM_DVRPTW(
		cust_feat_size = env_cls.CUST_FEAT_SIZE,
		veh_state_size = env_cls.VEH_STATE_SIZE,
		model_size = args.model_size,
		layer_count = args.layer_count,
		head_count = args.head_count,
		ff_size = args.ff_size,
		tanh_xplor = args.tanh_xplor,
		greedy = args.greedy,
	)
	learner.to(device)
	return learner


def _load_model_weights_or_raise(path, learner):
	if path is None:
		raise ValueError("Please provide --model-weight path for inference")

	checkpoint = torch.load(path, map_location = "cpu", weights_only = False)
	if isinstance(checkpoint, dict) and "model" in checkpoint and isinstance(checkpoint["model"], dict):
		state_dict = checkpoint["model"]
		source = "checkpoint['model']"
	elif isinstance(checkpoint, dict):
		state_dict = checkpoint
		source = "state_dict"
	else:
		raise ValueError("Unsupported model checkpoint format at '{}'".format(path))

	try:
		res = learner.load_state_dict(state_dict, strict = False)
	except RuntimeError as e:
		raise RuntimeError(
			"Model checkpoint is incompatible with current architecture. "
			"Please use matching model hyperparameters/config.\n{}".format(e)
		)

	missing = res.missing_keys if hasattr(res, "missing_keys") else []
	unexpected = res.unexpected_keys if hasattr(res, "unexpected_keys") else []
	matched = sum(1 for k in state_dict if k in learner.state_dict())
	print(
		"Loaded model from '{}' ({}) | provided={}, matched={}, missing={}, unexpected={}".format(
			path,
			source,
			len(state_dict),
			matched,
			len(missing),
			len(unexpected),
		)
	)


def _run_single_inference(env, learner):
	with torch.no_grad():
		actions, _, rewards = learner(env)
	costs = -torch.stack(rewards).sum(dim = 0).squeeze(-1)
	routes = actions_to_routes(actions, env.minibatch_size, env.veh_count)
	return routes, costs


def _run_inference(args, env, learner):
	if args.problem_type.startswith("s"):
		all_costs = []
		last_routes = None
		for _ in range(args.stoch_rollouts):
			routes, costs = _run_single_inference(env, learner)
			all_costs.append(costs)
			last_routes = routes
		mean_costs = torch.stack(all_costs, dim = 0).mean(dim = 0)
		return last_routes, mean_costs
	return _run_single_inference(env, learner)


def _print_routes(routes, costs, max_instances):
	n_show = min(max_instances, len(routes))
	for idx in range(n_show):
		print("=" * 60)
		print("Instance #{} | cost={:.4f}".format(idx, costs[idx].item()))
		print(routes_to_string(routes[idx]))


def _save_json(path, routes, normalized_costs, raw_replay_costs = None,
		route_diagnostics = None, constraint_diagnostics = None,
		raw_cost_components = None, normalized_cost_components = None):
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
	os.makedirs(os.path.dirname(path), exist_ok = True) if os.path.dirname(path) else None
	with open(path, "w") as f:
		json.dump(payload, f, indent = 2)


def _active_customer_set(data, inst_idx):
	if data.cust_mask is None:
		return set(range(1, data.nodes_count))
	mask = data.cust_mask[inst_idx].cpu()
	return {j for j in range(1, data.nodes_count) if not bool(mask[j].item())}


def _route_diag_for_instance(data, routes, inst_idx):
	active = _active_customer_set(data, inst_idx)
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


def _verify_routes_cost(data, env_cls, env_params, routes, model_costs, rollouts = 1):
	verify_env = env_cls(data, None, None, *env_params)
	verify_costs = eval_apriori_routes(verify_env, routes, max(1, int(rollouts)))
	abs_diff = (verify_costs - model_costs.cpu()).abs()

	print("Route verification: replay_mean={:.4f}, model_mean={:.4f}, max_abs_diff={:.6f}".format(
		verify_costs.mean().item(),
		model_costs.mean().item(),
		abs_diff.max().item(),
	))

	for idx in range(min(3, len(routes))):
		diag = _route_diag_for_instance(data, routes, idx)
		print(
			"  Instance #{} | replay={:.4f} model={:.4f} diff={:.6f} "
			"missing={} dup={} extra={}".format(
				idx,
				verify_costs[idx].item(),
				model_costs[idx].item(),
				abs_diff[idx].item(),
				diag["missing_count"],
				diag["duplicate_count"],
				diag["extra_count"],
			)
		)
		if diag["missing_count"] > 0:
			print("    missing sample:", diag["missing_head"])
		if diag["duplicate_count"] > 0:
			print("    duplicate sample:", diag["duplicate_head"])
		if diag["extra_count"] > 0:
			print("    extra sample:", diag["extra_head"])

	return verify_costs


def _replay_routes_cost(data, env_cls, env_params, routes, rollouts = 1):
	replay_env = env_cls(data, None, None, *env_params)
	return eval_apriori_routes(replay_env, routes, max(1, int(rollouts)))


def _check_route_constraints(data, routes, eps = 1e-9):
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
				dist = torch.dist(cur_xy, dest[:2], p = 2).item()
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


def _compute_cost_components(data, routes, pending_cost, late_cost, eps = 1e-9):
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
				dist = torch.dist(cur_xy, dest[:2], p = 2).item()
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

		active = _active_customer_set(data, inst_idx)
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


def main(args):
	device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
	set_random_seed(args.rng_seed, deterministic = True)

	dataset_cls = _dataset_cls(args.problem_type)
	env_cls = _environment_cls(args.problem_type)
	if dataset_cls is None or env_cls is None:
		raise ValueError("Unsupported problem type '{}'".format(args.problem_type))

	data = _build_dataset(args, dataset_cls)
	raw_data = _clone_dataset(data)
	if not args.no_normalize:
		data.normalize()

	env_params = _build_env_params(args)
	env = env_cls(data, None, None, *env_params)
	env.nodes = env.nodes.to(device)
	if env.init_cust_mask is not None:
		env.init_cust_mask = env.init_cust_mask.to(device)

	learner = _init_model(args, env_cls, device)
	learner.eval()

	with torch.no_grad():
		warmup_env = env_cls(data, data.nodes[:1].to(device), None, *env_params)
		_ = learner(warmup_env)

	_load_model_weights_or_raise(args.model_weight, learner)
	learner.eval()

	routes, costs = _run_inference(args, env, learner)
	raw_replay_costs = _replay_routes_cost(raw_data, env_cls, env_params, routes, rollouts = args.verify_rollouts)
	route_diagnostics = [_route_diag_for_instance(data, routes, idx) for idx in range(len(routes))]
	total_skipped = sum(d["missing_count"] for d in route_diagnostics)
	constraint_diagnostics = _check_route_constraints(raw_data, routes)
	raw_cost_components = _compute_cost_components(raw_data, routes, args.pending_cost, args.late_cost)
	normalized_cost_components = _compute_cost_components(data, routes, args.pending_cost, args.late_cost)
	total_tw_viol = sum(d["tw_violation_count"] for d in constraint_diagnostics)
	total_appear_viol = sum(d["appearance_violation_count"] for d in constraint_diagnostics)
	mean = costs.mean().item()
	std = costs.std().item() if costs.numel() > 1 else 0.0
	print("Inference done on {} instance(s): mean={:.4f}, std={:.4f}".format(costs.numel(), mean, std))
	print("Cost summary: normalized_cost_mean={:.4f}, raw_replay_cost_mean={:.4f}".format(
		costs.mean().item(),
		raw_replay_costs.mean().item(),
	))
	for idx in range(min(3, costs.numel())):
		print("  Instance #{} | normalized_cost={:.4f} | raw_replay_cost={:.4f}".format(
			idx,
			costs[idx].item(),
			raw_replay_costs[idx].item(),
		))
	print("Cost components (raw scale):")
	for idx in range(min(3, len(raw_cost_components))):
		c = raw_cost_components[idx]
		print(
			"  Instance #{} | reward={:.4f} | distance={:.4f} | late_time={:.4f} | skipped_orders={}".format(
				idx,
				c["reward"],
				c["distance"],
				c["late_time"],
				c["skipped_orders"],
			)
		)
	print("Skipped customers: total={} | per_instance={}".format(
		total_skipped,
		[d["missing_count"] for d in route_diagnostics[:min(10, len(route_diagnostics))]],
	))
	print("Constraint violations: total_tw={} | total_appearance={}".format(
		total_tw_viol,
		total_appear_viol,
	))
	for idx in range(min(3, len(constraint_diagnostics))):
		rep = constraint_diagnostics[idx]
		print("  Instance #{} | tw_violations={} | appearance_violations={}".format(
			idx,
			rep["tw_violation_count"],
			rep["appearance_violation_count"],
		))
	_print_routes(routes, costs, args.max_print_instances)

	if args.verify_routes:
		_verify_routes_cost(data, env_cls, env_params, routes, costs, rollouts = args.verify_rollouts)

	if args.save_json is not None:
		_save_json(
			args.save_json,
			routes,
			costs,
			raw_replay_costs,
			route_diagnostics,
			constraint_diagnostics,
			raw_cost_components,
			normalized_cost_components,
		)
		print("Saved inference outputs to '{}'".format(args.save_json))


if __name__ == "__main__":
	main(parse_infer_args())
