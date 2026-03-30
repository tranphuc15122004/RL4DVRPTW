import torch
import os.path

def save_checkpoint(args, ep, learner, optim, baseline = None, lr_sched = None):
    checkpoint = {
            "ep": ep,
            "model": learner.state_dict(),
            "optim": optim.state_dict()
            }
    if args.rate_decay is not None:
        checkpoint["lr_sched"] = lr_sched.state_dict()
    if args.baseline_type == "critic":
        checkpoint["critic"] = baseline.state_dict()
    torch.save(checkpoint, os.path.join(args.output_dir, "chkpt_ep{}.pyth".format(ep+1)))

KEEP_LAST = 5

def save_checkpoint_in_train(args, ep, learner, optim, baseline = None, lr_sched = None):
    checkpoint = {
            "ep": ep,
            "model": learner.state_dict(),
            "optim": optim.state_dict()
            }
    if args.rate_decay is not None:
        checkpoint["lr_sched"] = lr_sched.state_dict()
    if args.baseline_type == "critic":
        checkpoint["critic"] = baseline.state_dict()
    out_name = "chkpt_ep{}.pyth".format(ep+1)
    out_path = os.path.join(args.output_dir, out_name)
    torch.save(checkpoint, out_path)

    # keep only the latest KEEP_LAST checkpoints (or args.keep_checkpoints if provided)
    try:
        keep = getattr(args, 'keep_checkpoints', KEEP_LAST)
        if keep is None:
            keep = KEEP_LAST
        files = [f for f in os.listdir(args.output_dir) if f.startswith("chkpt_ep") and f.endswith('.pyth')]

        def epoch_from_name(name):
            try:
                s = name[len("chkpt_ep"):-len(".pyth")]
                return int(s)
            except Exception:
                return -1

        files_sorted = sorted(files, key=epoch_from_name, reverse=True)
        for old in files_sorted[int(keep):]:
            p = os.path.join(args.output_dir, old)
            try:
                os.remove(p)
            except Exception:
                pass
    except Exception:
        # do not fail saving checkpoint because of cleanup errors
        pass

def load_checkpoint(args, learner, optim, baseline = None, lr_sched = None):
    checkpoint = torch.load(args.resume_state, map_location='cpu', weights_only=False)

    # Try strict first; fall back to non-strict so checkpoints from nearby
    # architecture variants can still be resumed safely.
    try:
        learner.load_state_dict(checkpoint["model"], strict=True)
    except RuntimeError as e:
        print(f"[load_checkpoint] strict load failed ({e}); retrying with strict=False")
        result = learner.load_state_dict(checkpoint["model"], strict=False)
        if result.missing_keys:
            print(f"[load_checkpoint] Missing keys (will use random init): {result.missing_keys}")
        if result.unexpected_keys:
            print(f"[load_checkpoint] Unexpected keys (ignored): {result.unexpected_keys}")

    optim.load_state_dict(checkpoint["optim"])
    if args.rate_decay is not None and "lr_sched" in checkpoint:
        lr_sched.load_state_dict(checkpoint["lr_sched"])
    if args.baseline_type == "critic" and baseline is not None and "critic" in checkpoint:
        baseline.load_state_dict(checkpoint["critic"])
    return checkpoint["ep"]

import torch

def load_model_weights(args, learner, strict=False):
    """
    Load only model weights (policy network), in-place.

    Used for:
    - inference
    - fine-tune
    - eval
    - warm-start training (optimizer reset)

    Args:
        args.model_weight: path to checkpoint or state_dict
        learner: nn.Module (updated in-place)
        strict: whether to require exact key match (default False — safe for
            cross-version loads with minor key differences)
    """
    path = getattr(args, "model_weight", None)
    if not path:
        print("[load_model_weights] No --model-weight provided; skip loading.")
        return

    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"[load_model_weights] ERROR loading '{path}': {e}")
        return

    # Determine checkpoint format
    if isinstance(checkpoint, dict) and "model" in checkpoint and isinstance(checkpoint["model"], dict):
        state_dict = checkpoint["model"]
        source = "checkpoint['model']"
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
        source = "state_dict"
    else:
        print(f"[load_model_weights] Unexpected checkpoint format: {type(checkpoint)}")
        return

    # Load weights (in-place)
    try:
        res = learner.load_state_dict(state_dict, strict=strict)
    except Exception as e:
        print(f"[load_model_weights] ERROR applying state_dict (strict={strict}): {e}")
        return

    # Normalize missing / unexpected keys across PyTorch versions
    if isinstance(res, (tuple, list)):
        missing, unexpected = res
    else:
        missing = res.missing_keys
        unexpected = res.unexpected_keys

    provided = len(state_dict)
    model_keys = len(learner.state_dict())
    matched = sum(1 for k in state_dict if k in learner.state_dict())

    print(
        f"[load_model_weights] Loaded from '{path}' ({source}) | "
        f"provided={provided}, matched={matched}, model_keys={model_keys}, "
        f"missing={len(missing)}, unexpected={len(unexpected)}, strict={strict}"
    )

    if missing:
        print(f"[load_model_weights] Sample missing keys ({len(missing)}): {missing[:10]}")
    if unexpected:
        print(f"[load_model_weights] Sample unexpected keys ({len(unexpected)}): {unexpected[:10]}")
