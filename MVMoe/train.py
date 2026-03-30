from tqdm import tqdm
from MVMoe import *
from problems import *
from baselines import *
from utils import *
from layers import reinforce_loss
from  baselines._base import Baseline
import torch
try:
    from torch.amp import autocast, GradScaler as _GradScaler
    def GradScaler(enabled=True): return _GradScaler('cuda', enabled=enabled)
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
_AMP_DEVICE = 'cuda'
from torch.utils.data import DataLoader
from torch.optim import Adam 
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_
from problems import *
import time
import os
import math
from itertools import chain


def _format_param_count(count: int) -> str:
    return "{:,}".format(int(count))


def print_model_param_report(model: torch.nn.Module, model_name: str = "MVMoE"):
    total_params = 0
    trainable_params = 0

    print("[model] {} parameter report".format(model_name))
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count

        print(
            "  - {:60s} shape={} numel={} {}".format(
                name,
                tuple(param.shape),
                _format_param_count(param_count),
                "trainable" if param.requires_grad else "frozen",
            )
        )

    frozen_params = total_params - trainable_params
    print(
        "[model] total={} trainable={} frozen={}".format(
            _format_param_count(total_params),
            _format_param_count(trainable_params),
            _format_param_count(frozen_params),
        )
    )
    if total_params > 0:
        print("[model] trainable ratio={:.2%}".format(trainable_params / float(total_params)))


def _is_cuda_device_assert_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "device-side assert triggered" in msg
        or "cuda error" in msg and "assert" in msg
    )


def _is_oom_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "out of memory" in msg
        or "cublas_status_alloc_failed" in msg
        or "defaultcpuallocator" in msg and "memory" in msg
    )


def _cleanup_after_oom(optim: Adam = None):
    if optim is not None:
        try:
            optim.zero_grad(set_to_none = True)
        except Exception:
            pass

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _apply_resource_profile(args, device, verbose_print):
    profile_enabled = getattr(args, "resource_safe", False) or getattr(args, "smoke_test", False)
    if not profile_enabled:
        return

    if getattr(args, "smoke_test", False):
        smoke_caps = {
            "epoch_count": 1,
            "iter_count": 1,
            "batch_size": 8,
            "test_batch_size": 8,
            "customers_count": 10,
            "vehicles_count": 2,
            "model_size": 64,
            "layer_count": 2,
            "head_count": 4,
            "ff_size": 128,
            "num_workers": 0,
        }
        changed = []
        for key, cap in smoke_caps.items():
            current = getattr(args, key)
            if current is None:
                continue
            if current > cap:
                setattr(args, key, cap)
                changed.append("{}:{}->{}".format(key, current, cap))
        args.pin_memory = False
        if changed:
            verbose_print("[safe] smoke profile active ({})".format(", ".join(changed)))

    if getattr(args, "resource_safe", False):
        args.num_workers = max(0, int(args.num_workers))
        if args.num_workers == 0:
            args.pin_memory = False

    if device.type == "cuda":
        vram_fraction = args.max_vram_fraction
        if vram_fraction is None and getattr(args, "smoke_test", False):
            vram_fraction = 0.9
        if vram_fraction is not None:
            vram_fraction = min(1.0, max(0.1, float(vram_fraction)))
            try:
                dev_index = device.index if device.index is not None else torch.cuda.current_device()
                torch.cuda.set_per_process_memory_fraction(vram_fraction, device = dev_index)
                verbose_print("[safe] capped process VRAM usage to {:.0%}".format(vram_fraction))
            except Exception as exc:
                print("[warn] failed to set max VRAM fraction: {}".format(exc))


def _is_finite_scalar(value) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    return math.isfinite(float(value))


def _detach_loss_components(components):
    detached = {}
    for key in ("policy_loss", "critic_loss", "entropy_loss"):
        value = components.get(key, 0.0)
        if isinstance(value, torch.Tensor):
            detached[key] = value.detach()
        else:
            detached[key] = value
    return detached


def _format_train_postfix(loss, prob, val, bl, grad_norm, processed_batches, skipped_nonfinite, skip_reason = None):
    base = "l={:.4g} p={:9.4g} val={:6.4g} bl={:6.4g} |g|={:.4g}".format(
        loss,
        prob,
        val,
        bl,
        grad_norm,
    )
    if skip_reason is None and skipped_nonfinite == 0:
        return base
    total_batches = processed_batches + skipped_nonfinite
    suffix = " skip={}: {}/{}".format(
        skip_reason if skip_reason is not None else "total",
        skipped_nonfinite,
        total_batches,
    )
    return base + suffix


def _compute_train_batch_stats(args, data, Environment, env_params, bl_wrapped_learner, custs, mask):
    dyna = Environment(data, custs, mask, *env_params)
    actions, logps, rewards, bl_vals = bl_wrapped_learner(dyna)
    loss, loss_components = reinforce_loss(
        logps,
        rewards,
        bl_vals,
        adv_norm = args.adv_norm,
        entropy_coef = args.entropy_coef,
        return_components = True,
    )

    prob = torch.stack(logps).sum(0).exp().mean()
    if isinstance(rewards, torch.Tensor):
        val = rewards.mean()
    else:
        val = torch.stack(rewards).sum(0).mean()

    if bl_vals is None:
        bl = loss.detach().new_zeros(())
    elif isinstance(bl_vals, torch.Tensor):
        bl = bl_vals.mean()
    elif len(bl_vals) == 0 or bl_vals[0] is None:
        bl = loss.detach().new_zeros(())
    else:
        bl = bl_vals[0].mean()

    return loss, prob, val, bl, _detach_loss_components(loss_components)


def save_best_val_checkpoint(args, ep, learner, optim, baseline = None, lr_sched = None, best_val_mu = None):
    checkpoint = {
            "ep": ep,
            "best_ep": ep,
            "best_val_mu": None if best_val_mu is None else float(best_val_mu),
            "model": learner.state_dict(),
            "optim": optim.state_dict()
            }
    if args.rate_decay is not None:
        checkpoint["lr_sched"] = lr_sched.state_dict()
    if args.baseline_type == "critic":
        checkpoint["critic"] = baseline.state_dict()
    torch.save(checkpoint, os.path.join(args.output_dir, "chkpt_best.pyth"))

def train_epoch(args, data, Environment : VRP_Environment, env_params, bl_wrapped_learner : Baseline, optim : Adam, device, ep, scaler: GradScaler):
    bl_wrapped_learner.learner.train()
    loader = DataLoader(
        data,
        args.batch_size,
        True,
        num_workers = args.num_workers,
        pin_memory = args.pin_memory,
        persistent_workers = args.num_workers > 0,
    )

    ep_loss = 0
    ep_prob = 0
    ep_val = 0
    ep_bl = 0
    ep_norm = 0
    ep_policy_loss = 0
    ep_critic_loss = 0
    ep_entropy_loss = 0
    processed_batches = 0
    skipped_nonfinite = 0
    amp_fallbacks = 0
    with tqdm(loader, desc = "Ep.#{: >3d}/{: <3d}".format(ep+1, args.epoch_count)) as progress:
        for minibatch in progress:
            if data.cust_mask is None:
                custs, mask = minibatch.to(device), None
            else:
                custs, mask = minibatch[0].to(device), minibatch[1].to(device)

            grad_norm = 0.0
            used_amp = args.amp
            with autocast(_AMP_DEVICE, enabled = used_amp):
                loss, prob, val, bl, loss_components = _compute_train_batch_stats(
                    args,
                    data,
                    Environment,
                    env_params,
                    bl_wrapped_learner,
                    custs,
                    mask,
                )

            if not all(_is_finite_scalar(stat) for stat in (loss, prob, val, bl)):
                optim.zero_grad(set_to_none = True)
                if used_amp:
                    with autocast(_AMP_DEVICE, enabled = False):
                        loss, prob, val, bl, loss_components = _compute_train_batch_stats(
                            args,
                            data,
                            Environment,
                            env_params,
                            bl_wrapped_learner,
                            custs,
                            mask,
                        )
                    used_amp = False
                    amp_fallbacks += 1
                if not all(_is_finite_scalar(stat) for stat in (loss, prob, val, bl)):
                    skipped_nonfinite += 1
                    progress.set_postfix_str(_format_train_postfix(
                        loss,
                        prob,
                        val,
                        bl,
                        grad_norm,
                        processed_batches,
                        skipped_nonfinite,
                        "nonfinite-forward",
                    ))
                    continue

            optim.zero_grad()
            if used_amp:
                scaler.scale(loss).backward()
                if args.max_grad_norm is not None:
                    scaler.unscale_(optim)
                    grad_norm = clip_grad_norm_(
                        chain.from_iterable(grp["params"] for grp in optim.param_groups),
                        args.max_grad_norm,
                    )
                    if not torch.isfinite(grad_norm):
                        optim.zero_grad(set_to_none = True)
                        scaler.update()
                        with autocast(_AMP_DEVICE, enabled = False):
                            loss, prob, val, bl, loss_components = _compute_train_batch_stats(
                                args,
                                data,
                                Environment,
                                env_params,
                                bl_wrapped_learner,
                                custs,
                                mask,
                            )
                        amp_fallbacks += 1
                        if not all(_is_finite_scalar(stat) for stat in (loss, prob, val, bl)):
                            skipped_nonfinite += 1
                            progress.set_postfix_str(_format_train_postfix(
                                loss,
                                prob,
                                val,
                                bl,
                                grad_norm,
                                processed_batches,
                                skipped_nonfinite,
                                "nonfinite-forward",
                            ))
                            continue
                        loss.backward()
                        if args.max_grad_norm is not None:
                            grad_norm = clip_grad_norm_(
                                chain.from_iterable(grp["params"] for grp in optim.param_groups),
                                args.max_grad_norm,
                            )
                            if not torch.isfinite(grad_norm):
                                optim.zero_grad(set_to_none = True)
                                skipped_nonfinite += 1
                                progress.set_postfix_str(_format_train_postfix(
                                    loss,
                                    prob,
                                    val,
                                    bl,
                                    grad_norm,
                                    processed_batches,
                                    skipped_nonfinite,
                                    "nonfinite-grad",
                                ))
                                continue
                        optim.step()
                        used_amp = False
                        processed_batches += 1
                        progress.set_postfix_str(_format_train_postfix(
                            loss,
                            prob,
                            val,
                            bl,
                            grad_norm,
                            processed_batches,
                            skipped_nonfinite,
                        ))
                        ep_loss += loss.item()
                        ep_prob += prob.item()
                        ep_val += val.item()
                        ep_bl += bl.item()
                        ep_norm += float(grad_norm)
                        ep_policy_loss += float(loss_components["policy_loss"])
                        ep_critic_loss += float(loss_components["critic_loss"])
                        ep_entropy_loss += float(loss_components["entropy_loss"])
                        continue
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                if args.max_grad_norm is not None:
                    grad_norm = clip_grad_norm_(
                        chain.from_iterable(grp["params"] for grp in optim.param_groups),
                        args.max_grad_norm,
                    )
                    if not torch.isfinite(grad_norm):
                        optim.zero_grad(set_to_none = True)
                        skipped_nonfinite += 1
                        progress.set_postfix_str(_format_train_postfix(
                            loss,
                            prob,
                            val,
                            bl,
                            grad_norm,
                            processed_batches,
                            skipped_nonfinite,
                            "nonfinite-grad",
                        ))
                        continue
                optim.step()

            processed_batches += 1
            progress.set_postfix_str(_format_train_postfix(
                loss,
                prob,
                val,
                bl,
                grad_norm,
                processed_batches,
                skipped_nonfinite,
            ))

            ep_loss += loss.item()
            ep_prob += prob.item()
            ep_val += val.item()
            ep_bl += bl.item()
            ep_norm += float(grad_norm)
            ep_policy_loss += float(loss_components["policy_loss"])
            ep_critic_loss += float(loss_components["critic_loss"])
            ep_entropy_loss += float(loss_components["entropy_loss"])

    if skipped_nonfinite > 0:
        print("[warn] epoch {} skipped {}/{} minibatches due to non-finite training stats/gradients (amp_fallbacks={})".format(
            ep + 1,
            skipped_nonfinite,
            processed_batches + skipped_nonfinite,
            amp_fallbacks,
        ))

    if processed_batches == 0:
        print("[error] epoch {} had no successful optimizer steps; returning NaN train metrics instead of misleading zeros.".format(ep + 1))
        return (float("nan"),) * 8

    return tuple(stat / processed_batches for stat in (
        ep_loss,
        ep_prob,
        ep_val,
        ep_bl,
        ep_norm,
        ep_policy_loss,
        ep_critic_loss,
        ep_entropy_loss,
    ))


def test_epoch(args, test_env, learner, ref_costs):
    learner.eval()
    prev_greedy = getattr(learner, "greedy", None)
    if prev_greedy is not None:
        learner.greedy = True
    try:
        with torch.no_grad():   # 🔥 BẮT BUỘC
            if args.problem_type[0] == "s":
                costs = test_env.nodes.new_zeros(test_env.minibatch_size)
                for _ in range(100):
                    _, _, rewards = learner(test_env)
                    costs -= torch.stack(rewards).sum(0).squeeze(-1)
                costs = costs / 100
            else:
                _, _, rs = learner(test_env)
                costs = -torch.stack(rs).sum(dim = 0).squeeze(-1)
            mean = costs.mean()
            std = costs.std()
            gap = (costs.to(ref_costs.device) / ref_costs - 1).mean()
    finally:
        if prev_greedy is not None:
            learner.greedy = prev_greedy

    print("Cost on test dataset: {:5.2f} +- {:5.2f} ({:.2%})".format(mean, std, gap))
    return mean.item(), std.item(), gap.item()

def val_epoch(args, test_env, learner):
    learner.eval()
    prev_greedy = getattr(learner, "greedy", None)
    if prev_greedy is not None:
        learner.greedy = True
    try:
        with torch.no_grad():   # 🔥 BẮT BUỘC
            if args.problem_type[0] == "s":
                costs = test_env.nodes.new_zeros(test_env.minibatch_size)
                for _ in range(100):
                    _, _, rewards = learner(test_env)
                    costs -= torch.stack(rewards).sum(0).squeeze(-1)
                costs = costs / 100
            else:
                _, _, rs = learner(test_env)
                costs = -torch.stack(rs).sum(dim = 0).squeeze(-1)
            mean = costs.mean()
            std = costs.std()
    finally:
        if prev_greedy is not None:
            learner.greedy = prev_greedy

    print("Cost on test dataset: {:5.2f} +- {:5.2f}".format(mean, std))
    return mean.item(), std.item()

def main(args):
    dev = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    torch.backends.cudnn.benchmark = not (getattr(args, "resource_safe", False) or getattr(args, "smoke_test", False))
    if args.rng_seed is not None:
        torch.manual_seed(args.rng_seed)

    if args.verbose:
        verbose_print = print
    else:
        def verbose_print(*args, **kwargs): pass

    _apply_resource_profile(args, dev, verbose_print)

    # PROBLEM
    Dataset = {
            "vrp": VRP_Dataset,
            "vrptw": VRPTW_Dataset,
            "svrptw": VRPTW_Dataset,
            "sdvrptw": SDVRPTW_Dataset,
            "dvrptw" : DVRPTW_Dataset
            }.get(args.problem_type)
    gen_params = [
            args.customers_count,
            args.vehicles_count,
            args.veh_capa,
            args.veh_speed,
            args.min_cust_count,
            args.loc_range,
            args.dem_range
            ]
    if args.problem_type != "vrp":
        gen_params.extend( [args.horizon, args.dur_range, args.tw_ratio, args.tw_range] )
    if args.problem_type == "sdvrptw" or  args.problem_type == "dvrptw":
        gen_params.extend( [args.deg_of_dyna, args.appear_early_ratio] )

    # TRAIN DATA
    verbose_print("Generating {} {} samples of training data...".format(
        args.iter_count * args.batch_size, args.problem_type.upper()),
        end = " ", flush = True)
    train_data = Dataset.generate(
            args.iter_count * args.batch_size,
            *gen_params
            )
    train_data.normalize()
    verbose_print("Done.")

    # TEST DATA AND COST REFERENCE
    verbose_print("Generating {} {} samples of test data...".format(
        args.test_batch_size, args.problem_type.upper()),
        end = " ", flush = True)
    test_data = Dataset.generate(
            args.test_batch_size,
            *gen_params
            )
    test_data.normalize()
    verbose_print("Done.")


    # ENVIRONMENT
    Environment = {
            "vrp": VRP_Environment,
            "vrptw": VRPTW_Environment,
            "svrptw": SVRPTW_Environment,
            "sdvrptw": SDVRPTW_Environment,
            "dvrptw": DVRPTW_Environment
            }.get(args.problem_type)
    env_params = [args.pending_cost]
    if args.problem_type != "vrp":
        env_params.append(args.late_cost)
        if args.problem_type != "vrptw" and args.problem_type != "dvrptw":
            env_params.extend( [args.speed_var, args.late_prob, args.slow_down, args.late_var] )
    print(env_params)
    test_env = Environment(test_data, None, None, *env_params)


    test_env.nodes = test_env.nodes.to(dev)
    if test_env.init_cust_mask is not None:
        test_env.init_cust_mask = test_env.init_cust_mask.to(dev)

    # MODEL
    verbose_print("Initializing attention model...",
        end = " ", flush = True)
    
    
    ## thay the model vao dau de test
    learner : torch.Module = MVMoE_DVRPTW(
        cust_feat_size = Environment.CUST_FEAT_SIZE,
        veh_state_size = Environment.VEH_STATE_SIZE,
        model_size = args.model_size,
        layer_count = args.layer_count,
        head_count = args.head_count,
        ff_size = args.ff_size,
        tanh_xplor = args.tanh_xplor,
    )
    
    learner.to(dev)
    print_model_param_report(learner, learner.__class__.__name__)
    verbose_print("Done.")

    # BASELINE
    verbose_print("Initializing '{}' baseline...".format(
        args.baseline_type),
        end = " ", flush = True)
    baseline : Baseline = None
    if args.baseline_type == "none":
        baseline = NoBaseline(learner)
    elif args.baseline_type == "nearnb":
        baseline = NearestNeighbourBaseline(learner, args.loss_use_cumul)
    elif args.baseline_type == "rollout":
        args.loss_use_cumul = True
        baseline = RolloutBaseline(learner, args.rollout_count, args.rollout_threshold)
    elif args.baseline_type == "critic":
        baseline = CriticBaseline(learner, args.customers_count, args.critic_use_qval, args.loss_use_cumul)
        if not args.adv_norm:
            print("[warn] baseline=critic but --adv-norm is disabled; training variance can be high.")
    baseline.to(dev)
    verbose_print("Done.")

    # OPTIMIZER AND LR SCHEDULER
    verbose_print("Initializing Adam optimizer...",
        end = " ", flush = True)
    lr_sched = None
    if args.baseline_type == "critic":
        optim = Adam([
            {"params": learner.parameters(), "lr": args.learning_rate, "weight_decay": args.weight_decay},
            {"params": baseline.parameters(), "lr": args.critic_rate, "weight_decay": args.weight_decay}
            ])
        if args.rate_decay is not None:
            critic_decay = args.rate_decay if args.critic_decay is None else args.critic_decay
            
            """ lr_sched = LambdaLR(optim,[
                lambda ep: args.learning_rate * args.rate_decay**ep,
                lambda ep: args.critic_rate * critic_decay**ep
                ])
             """
            
            lr_sched = LambdaLR(optim,[
                lambda ep: args.rate_decay**ep,
                lambda ep: critic_decay**ep
                ])
    else:
        optim = Adam(learner.parameters(), args.learning_rate, weight_decay = args.weight_decay)
        if args.rate_decay is not None:
            lr_sched = LambdaLR(optim, lambda ep: args.rate_decay**ep)
    verbose_print("Done.")

    # CHECKPOINTING
    verbose_print("Creating output dir...",
        end = " ", flush = True)
    args.output_dir = "./output/{}n{}m{}_{}".format(
            args.problem_type.upper(),
            args.customers_count,
            args.vehicles_count,
            time.strftime("%y%m%d-%H%M")
            ) if args.output_dir is None else args.output_dir
    os.makedirs(args.output_dir, exist_ok = True)
    write_config_file(args, os.path.join(args.output_dir, "args.json"))
    verbose_print("'{}' created.".format(args.output_dir))

    if args.resume_state is None:
        start_ep = 0
    else:
        start_ep = load_checkpoint(args, learner, optim, baseline, lr_sched)
        
    
    load_model_weights(args , learner )
    

    verbose_print("Running...")
    train_stats = []
    val_stats = []
    test_stats = []
    best_val_mu = float("inf")
    best_ep = -1

    best_ckpt_path = os.path.join(args.output_dir, "chkpt_best.pyth")
    if os.path.exists(best_ckpt_path):
        try:
            best_ckpt = torch.load(best_ckpt_path, map_location = "cpu", weights_only = False)
            loaded_best = best_ckpt.get("best_val_mu", None)
            loaded_best_ep = best_ckpt.get("best_ep", None)
            if loaded_best is not None:
                best_val_mu = float(loaded_best)
            if loaded_best_ep is not None:
                best_ep = int(loaded_best_ep)
        except Exception:
            pass

    scaler = GradScaler(enabled = args.amp)  # wrapper above passes device_type automatically
    ep = start_ep - 1
    try:
        for ep in range(start_ep, args.epoch_count):
            oom_retries = 0
            while True:
                try:
                    train_stats.append(train_epoch(args, train_data, Environment, env_params, baseline, optim, dev, ep, scaler))
                    break
                except RuntimeError as exc:
                    if not _is_oom_error(exc):
                        raise

                    oom_retries += 1
                    _cleanup_after_oom(optim)
                    old_batch = args.batch_size
                    old_workers = args.num_workers
                    if old_batch <= 1 or oom_retries > 5:
                        raise RuntimeError(
                            "OOM persisted after {} retries (batch_size={})".format(oom_retries, old_batch)
                        ) from exc
                    args.batch_size = max(1, old_batch // 2)
                    args.num_workers = max(0, old_workers // 2)
                    if args.num_workers == 0:
                        args.pin_memory = False
                    print("[warn] OOM at epoch {}. Retrying with batch_size {}->{} and num_workers {}->{}".format(
                        ep + 1,
                        old_batch,
                        args.batch_size,
                        old_workers,
                        args.num_workers,
                    ))

            try:
                val_stats.append(val_epoch(args, test_env, learner))
            except RuntimeError as exc:
                if not _is_oom_error(exc):
                    raise
                _cleanup_after_oom()
                print("[warn] OOM during validation at epoch {}; logging NaN val stats for this epoch".format(ep + 1))
                val_stats.append((float("nan"), float("nan")))

            cur_val_mu = val_stats[-1][0]
            if math.isfinite(cur_val_mu) and cur_val_mu < best_val_mu:
                best_val_mu = float(cur_val_mu)
                best_ep = ep
                save_best_val_checkpoint(args, ep, learner, optim, baseline, lr_sched, best_val_mu)
                verbose_print("[BEST] ep={} val_mu={:.6g} -> chkpt_best.pyth".format(ep + 1, best_val_mu))
            update_train_test_stats(args, ep, train_stats, val_stats)

            if args.rate_decay is not None:
                lr_sched.step()
            if args.pend_cost_growth is not None:
                env_params[0] *= args.pend_cost_growth
            if args.late_cost_growth is not None:
                env_params[1] *= args.late_cost_growth
            if args.grad_norm_decay is not None:
                args.max_grad_norm *= args.grad_norm_decay

            if (ep+1) % args.checkpoint_period == 0:
                save_checkpoint(args, ep, learner, optim, baseline, lr_sched)

    except KeyboardInterrupt:
        try:
            save_checkpoint(args, ep, learner, optim, baseline, lr_sched)
        except Exception as e:
            print("[warn] Skip checkpoint after interrupt due to save error: {}".format(e))
        export_train_test_stats(args, start_ep, train_stats, test_stats)
    except Exception as e:
        if _is_cuda_device_assert_error(e):
            print("[error] CUDA device-side assert detected. Skipping checkpoint save because CUDA context is invalid.")
        else:
            raise
        
    finally:
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            try:
                torch.cuda.synchronize()
            except Exception as e:
                if _is_cuda_device_assert_error(e):
                    print("[warn] Skipping final checkpoint save due to prior CUDA device-side assert.")
                    export_train_test_stats(args, start_ep, train_stats, test_stats)
                    if best_ep >= 0:
                        verbose_print("Best validation checkpoint: ep={} val_mu={:.6g}".format(best_ep + 1, best_val_mu))
                    return
                print("[warn] CUDA sync failed before final save: {}".format(e))

        try:
            save_checkpoint(args, ep, learner, optim, baseline, lr_sched)
        except Exception as e:
            if _is_cuda_device_assert_error(e):
                print("[warn] Final checkpoint skipped due to CUDA device-side assert.")
            else:
                print("[warn] Final checkpoint save failed: {}".format(e))
        export_train_test_stats(args, start_ep, train_stats, test_stats)
        if best_ep >= 0:
            verbose_print("Best validation checkpoint: ep={} val_mu={:.6g}".format(best_ep + 1, best_val_mu))


if __name__ == "__main__":
    main(parse_args())
