import logging

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

from tensordict import TensorDict

from rl4co.data.transforms import StateAugmentation
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl.reinforce.reinforce import REINFORCE
from rl4co.models.zoo.polynet.policy import PolyNetPolicy
from rl4co.utils.ops import gather_by_index, unbatchify
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class PolyNet_DVRPTW(nn.Module):
    """Adapter that keeps the PolyNet policy core and exposes the stepwise API
    used by this repository's DVRPTW training pipeline and baselines.
    """

    def __init__(
        self,
        cust_feat_size,
        veh_state_size,
        model_size=128,
        layer_count=3,
        head_count=8,
        ff_size=512,
        tanh_xplor=10,
        greedy=False,
        k: int = 32,
        encoder_type: str = "AM",
    ):
        super().__init__()
        self.policy = PolyNetPolicy(
            env_name="cvrptw",
            k=max(2, int(k)),
            encoder_type=encoder_type,
            embed_dim=model_size,
            num_encoder_layers=layer_count,
            num_heads=head_count,
            feedforward_hidden=ff_size,
            tanh_clipping=tanh_xplor,
        )
        self.greedy = greedy

        self.cust_repr = None
        self._static_td = None
        self._decoder_cache = None
        self._vehicle_capacity = None
        self._skip_refresh_once = False

    def _build_static_td(self, customers: torch.Tensor, cust_mask: torch.Tensor | None):
        model_in = customers if cust_mask is None else customers.masked_fill(cust_mask.unsqueeze(-1), 0)
        return TensorDict(
            {
                "locs": model_in[:, :, :2],
                "demand": model_in[:, 1:, 2],
                "durations": model_in[:, :, 5],
                "time_windows": model_in[:, :, 3:5],
            },
            batch_size=[model_in.size(0)],
            device=model_in.device,
        )

    def _encode_customers(self, dyna):
        customers = dyna.nodes
        cust_mask = getattr(dyna, "cust_mask", None)
        td = self._build_static_td(customers, cust_mask)
        cust_repr, _ = self.policy.encoder(td)
        if cust_mask is not None:
            cust_repr = cust_repr.masked_fill(cust_mask.unsqueeze(-1), 0)

        self._static_td = td
        self.cust_repr = cust_repr
        self._decoder_cache = self.policy.decoder._precompute_cache(cust_repr)

    def step(self, dyna):
        if self._decoder_cache is None:
            raise RuntimeError("Call on_rollout_start before step")

        # Refresh customer cache when new dynamic customers are revealed.
        if getattr(dyna, "new_customers", False):
            if self._skip_refresh_once:
                self._skip_refresh_once = False
            else:
                self._encode_customers(dyna)

        veh_mask = dyna.cur_veh_mask
        if veh_mask.dim() == 3 and veh_mask.size(1) != 1:
            veh_mask = veh_mask.gather(1, dyna.cur_veh_idx[:, :, None].expand(-1, -1, veh_mask.size(-1)))

        cur_veh = dyna.vehicles.gather(
            1,
            dyna.cur_veh_idx[:, :, None].expand(-1, -1, dyna.vehicles.size(-1)),
        )
        cur_pos = cur_veh[:, 0, :2]
        locs = self._static_td["locs"]
        current_node = torch.cdist(cur_pos[:, None, :], locs).squeeze(1).argmin(dim=1, keepdim=True)

        step_td = TensorDict(
            {
                "locs": self._static_td["locs"],
                "demand": self._static_td["demand"],
                "durations": self._static_td["durations"],
                "time_windows": self._static_td["time_windows"],
                "current_node": current_node,
                "current_time": cur_veh[:, :, 3],
                "used_capacity": (self._vehicle_capacity - cur_veh[:, :, 2]).clamp_min(0),
                "vehicle_capacity": self._vehicle_capacity,
                "action_mask": ~veh_mask.squeeze(1),
            },
            batch_size=[dyna.vehicles.size(0)],
            device=dyna.vehicles.device,
        )

        compat, _ = self.policy.decoder(step_td, self._decoder_cache, num_starts=0)

        mask = veh_mask.squeeze(1)
        all_masked = mask.all(dim=1, keepdim=True)
        if all_masked.any():
            mask = mask.clone()
            mask[all_masked.squeeze(1), 0] = False
        scores = compat.clone()
        scores[mask] = -float("inf")
        logp_full = scores.log_softmax(dim=1)

        probs = logp_full.exp()
        bad = (~torch.isfinite(probs)).any(dim=1, keepdim=True) | (probs.sum(dim=1, keepdim=True) <= 0)
        if bad.any():
            safe = torch.zeros_like(probs)
            safe[:, 0] = 1.0
            probs = torch.where(bad, safe, probs)

        cust_idx = probs.argmax(dim=1, keepdim=True) if self.greedy else probs.multinomial(1)
        if cust_idx.dtype != torch.int64:
            cust_idx = cust_idx.long()
        chosen_mask = veh_mask.gather(2, cust_idx.unsqueeze(1)).squeeze(1)
        if chosen_mask.any():
            cust_idx = cust_idx.masked_fill(chosen_mask, 0)

        return cust_idx, logp_full.gather(1, cust_idx), {"compat": compat.unsqueeze(1)}

    def on_rollout_start(self, dyna):
        pointer = getattr(self.policy.decoder, "pointer", None)
        if pointer is not None and hasattr(pointer, "probs"):
            pointer.probs = None

        self._vehicle_capacity = dyna.vehicles.new_full(
            (dyna.minibatch_size, 1),
            float(dyna.veh_capa),
        )

        self._encode_customers(dyna)
        self._skip_refresh_once = True

    def forward(self, dyna):
        dyna.reset()
        self.on_rollout_start(dyna)

        actions, logps, rewards = [], [], []
        while not dyna.done:
            out = self.step(dyna)
            cust_idx, logp = out[0], out[1]
            actions.append((dyna.cur_veh_idx, cust_idx))
            logps.append(logp)
            rewards.append(dyna.step(cust_idx))
        return actions, logps, rewards


class PolyNet(REINFORCE):
    """PolyNet
    Based on Hottung et al. (2024) https://arxiv.org/abs/2402.14048.

    Note:
        PolyNet allows to learn diverse solution stratgies with a single model. This is achieved
        through a modified decoder and the Poppy loss (Grinsztajn et al. (2021)). PolyNet can be used with the attention model encoder or the MatNet encoder by
        setting encoder_type to "AM" or "MatNet", respectively.

    Args:
        env: TorchRL Environment
        policy: Policy to use for the algorithm
        k: Number of strategies to learn ("K" in the paper)
        val_num_solutions: Number of solutions that are generated per instance during validation
        encoder_type: Type of encoder that should be used. "AM" or "MatNet" are supported
        policy_kwargs: Keyword arguments for policy
        baseline: Baseline to use for the algorithm. Note that PolyNet only supports shared baseline,
            so we will throw an error if anything else is passed.
        num_augment: Number of augmentations (used only for validation and test)
        augment_fn: Function to use for augmentation, defaulting to dihedral8
        first_aug_identity: Whether to include the identity augmentation in the first position
        feats: List of features to augment
        **kwargs: Keyword arguments passed to the superclass
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: PolyNetPolicy = None,
        k: int = 128,
        val_num_solutions: int = 800,
        encoder_type="AM",
        base_model_checkpoint_path: str = None,
        policy_kwargs={},
        baseline: str = "shared",
        num_augment: int = 8,
        augment_fn: str | Callable = "dihedral8",
        first_aug_identity: bool = True,
        feats: list = None,
        **kwargs,
    ):
        self.save_hyperparameters(logger=False)

        self.k = k
        self.val_num_solutions = val_num_solutions

        assert encoder_type in [
            "AM",
            "MatNet",
        ], "Supported encoder types are 'AM' and 'MatNet'"

        assert baseline == "shared", "PolyNet only supports shared baseline"

        if (
            policy_kwargs.get("val_decode_type") == "greedy"
            or policy_kwargs.get("test_decode_type") == "greedy"
        ):
            assert val_num_solutions <= k, (
                "If greedy decoding is used val_num_solutions must be <= k"
            )

        if encoder_type == "MatNet":
            assert num_augment == 1, "MatNet does not use symmetric or dihedral augmentation"

        if policy is None:
            policy = PolyNetPolicy(
                env_name=env.name, k=k, encoder_type=encoder_type, **policy_kwargs
            )

        if base_model_checkpoint_path is not None:
            logging.info(f"Trying to load weights from baseline model {base_model_checkpoint_path}")
            checkpoint = torch.load(base_model_checkpoint_path, weights_only=False)
            state_dict = checkpoint["state_dict"]
            state_dict = {k.replace("policy.", "", 1): v for k, v in state_dict.items()}
            policy.load_state_dict(state_dict, strict=False)

        train_batch_size = kwargs["batch_size"] if "batch_size" in kwargs else 64
        kwargs_with_defaults = {
            "val_batch_size": train_batch_size,
            "test_batch_size": train_batch_size,
        }
        kwargs_with_defaults.update(kwargs)

        # Initialize with the shared baseline
        super().__init__(env, policy, baseline, **kwargs_with_defaults)

        self.num_augment = num_augment
        if self.num_augment > 1:
            self.augment = StateAugmentation(
                num_augment=self.num_augment,
                augment_fn=augment_fn,
                first_aug_identity=first_aug_identity,
                feats=feats,
            )
        else:
            self.augment = None

        # Add `_multistart` to decode type for train, val and test in policy
        # for phase in ["train", "val", "test"]:
        #    self.set_decode_type_multistart(phase)

    def shared_step(self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None):
        td = self.env.reset(batch)
        n_aug = self.num_augment

        # During training, we do not augment the data
        if phase == "train":
            n_aug = 0
        elif n_aug > 1:
            td = self.augment(td)

        if phase == "train":
            n_start = self.k
        else:
            n_start = self.val_num_solutions

        # Evaluate policy
        out = self.policy(
            td,
            self.env,
            phase=phase,
            num_starts=n_start,
            multisample=True,
        )

        # Unbatchify reward to [batch_size, num_augment, num_starts].
        reward = unbatchify(out["reward"], (n_aug, n_start))

        # Training phase
        if phase == "train":
            assert n_start > 1, "num_starts must be > 1 during training"
            log_likelihood = unbatchify(out["log_likelihood"], (n_aug, n_start))
            self.calculate_loss(td, batch, out, reward, log_likelihood)
            max_reward, max_idxs = reward.max(dim=-1)
            out.update({"max_reward": max_reward})
        # Get multi-start (=POMO) rewards and best actions only during validation and test
        else:
            if n_start > 1:
                # max multi-start reward
                max_reward, max_idxs = reward.max(dim=-1)
                out.update({"max_reward": max_reward})

                if out.get("actions", None) is not None:
                    # Reshape batch to [batch_size, num_augment, num_starts, ...]
                    actions = unbatchify(out["actions"], (n_aug, n_start))
                    out.update(
                        {
                            "best_multistart_actions": gather_by_index(
                                actions, max_idxs.unsqueeze(2), dim=2
                            )
                        }
                    )
                    out["actions"] = actions

            # Get augmentation score only during inference
            if n_aug > 1:
                # If multistart is enabled, we use the best multistart rewards
                reward_ = max_reward if n_start > 1 else reward
                max_aug_reward, max_idxs = reward_.max(dim=1)
                out.update({"max_aug_reward": max_aug_reward})

                if out.get("actions", None) is not None:
                    actions_ = out["best_multistart_actions"] if n_start > 1 else out["actions"]
                    out.update({"best_aug_actions": gather_by_index(actions_, max_idxs)})

        metrics = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)
        return {"loss": out.get("loss", None), **metrics}

    def calculate_loss(
        self,
        td: TensorDict,
        batch: TensorDict,
        policy_out: dict,
        reward: torch.Tensor | None = None,
        log_likelihood: torch.Tensor | None = None,
    ):
        """Calculate loss following Poppy (https://arxiv.org/abs/2210.03475).

        Args:
            td: TensorDict containing the current state of the environment
            batch: Batch of data. This is used to get the extra loss terms, e.g., REINFORCE baseline
            policy_out: Output of the policy network
            reward: Reward tensor. If None, it is taken from `policy_out`
            log_likelihood: Log-likelihood tensor. If None, it is taken from `policy_out`
        """
        # Extra: this is used for additional loss terms, e.g., REINFORCE baseline
        extra = batch.get("extra", None)
        reward = reward if reward is not None else policy_out["reward"]
        log_likelihood = (
            log_likelihood if log_likelihood is not None else policy_out["log_likelihood"]
        )

        # REINFORCE baseline
        bl_val, bl_loss = self.baseline.eval(td, reward, self.env) if extra is None else (extra, 0)

        # Log-likelihood mask. Mask everything but the best rollout per instance
        best_idx = (-reward).argsort(1).argsort(1)
        mask = best_idx < 1

        # Main loss function
        advantage = reward - bl_val  # advantage = reward - baseline
        reinforce_loss = -(advantage * log_likelihood * mask).mean()
        loss = reinforce_loss + bl_loss
        policy_out.update(
            {
                "loss": loss,
                "reinforce_loss": reinforce_loss,
                "bl_loss": bl_loss,
                "bl_val": bl_val,
            }
        )
        return policy_out
