from collections.abc import Callable

import torch
import torch.nn as nn
from tensordict import TensorDict

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl.reinforce.baselines import REINFORCEBaseline
from rl4co.models.zoo.am import AttentionModel, AttentionModelPolicy
from rl4co.models.zoo.pomo import POMO
from utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class MVMoE_DVRPTW(nn.Module):
    """Adapter that keeps the MVMoE-AM policy core and exposes the stepwise API
    used by this repository's DVRPTW training pipeline and baselines.
    """

    def __init__(
        self,
        cust_feat_size,
        veh_state_size,
        model_size = 128,
        layer_count = 3,
        head_count = 8,
        ff_size = 512,
        tanh_xplor = 10,
        greedy = False,
        moe_kwargs: dict = None,
    ):
        super().__init__()
        if moe_kwargs is None:
            moe_kwargs = {
                "encoder": {
                    "hidden_act": "ReLU",
                    "num_experts": 4,
                    "k": 2,
                    "noisy_gating": True,
                },
                "decoder": {
                    "light_version": True,
                    "out_bias": False,
                    "num_experts": 4,
                    "k": 2,
                    "noisy_gating": True,
                },
            }

        self.policy = AttentionModelPolicy(
            env_name = "cvrptw",
            embed_dim = model_size,
            num_encoder_layers = layer_count,
            num_heads = head_count,
            feedforward_hidden = ff_size,
            tanh_clipping = tanh_xplor,
            moe_kwargs = moe_kwargs,
        )
        self.greedy = greedy

        self.cust_repr = None
        self._customers = None
        self._static_td = None
        self._decoder_cache = None
        self._last_td = None
        self._active_dyna = None
        self._vehicle_capacity = None

    def _build_policy_td(self, customers):
        batch_size = customers.size(0)
        return TensorDict(
            {
                "locs": customers[:, :, :2],
                "demand": customers[:, 1:, 2],
                "durations": customers[:, :, 5],
                "time_windows": customers[:, :, 3:5],
            },
            batch_size = [batch_size],
            device = customers.device,
        )

    def _infer_current_node(self, vehicles, veh_idx):
        cur_veh = vehicles.gather(1, veh_idx[:, :, None].expand(-1, -1, vehicles.size(-1)))
        cur_pos = cur_veh[:, 0, :2]
        locs = self._static_td["locs"]
        dist = torch.cdist(cur_pos[:, None, :], locs).squeeze(1)
        current_node = dist.argmin(dim = 1, keepdim = True)
        return cur_veh, current_node

    def _build_step_td(self, vehicles, veh_idx, veh_mask):
        if veh_mask.dim() == 3 and veh_mask.size(1) != 1:
            veh_mask = veh_mask.gather(1, veh_idx[:, :, None].expand(-1, -1, veh_mask.size(-1)))

        cur_veh, current_node = self._infer_current_node(vehicles, veh_idx)
        remaining_capacity = cur_veh[:, :, 2]
        used_capacity = (self._vehicle_capacity - remaining_capacity).clamp_min(0)
        current_time = cur_veh[:, :, 3]
        action_mask = ~veh_mask.squeeze(1)

        return TensorDict(
            {
                "locs": self._static_td["locs"],
                "demand": self._static_td["demand"],
                "durations": self._static_td["durations"],
                "time_windows": self._static_td["time_windows"],
                "current_node": current_node,
                "current_time": current_time,
                "used_capacity": used_capacity,
                "vehicle_capacity": self._vehicle_capacity,
                "action_mask": action_mask,
            },
            batch_size = [vehicles.size(0)],
            device = vehicles.device,
        )

    def _maybe_refresh_customer_encoding(self):
        if self._active_dyna is None:
            return
        if getattr(self._active_dyna, "new_customers", False):
            cust_mask = getattr(self._active_dyna, "cust_mask", None)
            self._encode_customers(self._active_dyna.nodes, cust_mask)

    def _encode_customers(self, customers, mask = None):
        self._customers = customers

        model_in = customers
        if mask is not None:
            model_in = customers.clone()
            model_in[mask] = 0

        td = self._build_policy_td(model_in)
        cust_repr, _ = self.policy.encoder(td)
        if mask is not None:
            cust_repr = cust_repr.masked_fill(mask.unsqueeze(-1), 0)

        self._static_td = td
        self.cust_repr = cust_repr
        self._decoder_cache = self.policy.decoder._precompute_cache(cust_repr)
        self._last_td = None

    def _repr_vehicle(self, vehicles, veh_idx, mask):
        self._maybe_refresh_customer_encoding()
        if self._decoder_cache is None:
            raise RuntimeError("Call _encode_customers before _repr_vehicle")

        self._last_td = self._build_step_td(vehicles, veh_idx, mask)
        return self.policy.decoder._compute_q(self._decoder_cache, self._last_td)

    def _score_customers(self, veh_repr = None):
        if self._last_td is None:
            raise RuntimeError("Call _repr_vehicle before _score_customers")

        logits, _ = self.policy.decoder(self._last_td, self._decoder_cache, num_starts = 0)
        return logits.unsqueeze(1)

    def _get_logp(self, compat, veh_mask):
        if compat.dim() == 2:
            compat = compat.unsqueeze(1)

        mask = veh_mask
        if mask.dim() == 2:
            mask = mask.unsqueeze(1)

        all_masked = mask.all(dim = 2, keepdim = True)
        if all_masked.any():
            mask = mask.clone()
            all_masked_rows = all_masked.squeeze(2).squeeze(1)
            mask[all_masked_rows, 0, 0] = False

        scores = compat.clone()
        scores[mask] = -float("inf")
        return scores.log_softmax(dim = 2).squeeze(1)

    def _sample_action(self, logp, cur_veh_mask):
        probs = logp.exp()
        bad = (~torch.isfinite(probs)).any(dim = 1, keepdim = True) | (probs.sum(dim = 1, keepdim = True) <= 0)
        if bad.any():
            safe = torch.zeros_like(probs)
            safe[:, 0] = 1.0
            probs = torch.where(bad, safe, probs)

        if self.greedy:
            cust_idx = probs.argmax(dim = 1, keepdim = True)
        else:
            cust_idx = probs.multinomial(1)

        if cust_idx.dtype != torch.int64:
            cust_idx = cust_idx.long()

        chosen_mask = cur_veh_mask.gather(2, cust_idx.unsqueeze(1)).squeeze(1)
        if chosen_mask.any():
            cust_idx = cust_idx.masked_fill(chosen_mask, 0)
        return cust_idx

    def step(self, dyna):
        veh_repr = self._repr_vehicle(dyna.vehicles, dyna.cur_veh_idx, dyna.cur_veh_mask)
        compat = self._score_customers(veh_repr)
        logp = self._get_logp(compat, dyna.cur_veh_mask)
        cust_idx = self._sample_action(logp, dyna.cur_veh_mask)
        return cust_idx, logp.gather(1, cust_idx)

    def _reset_memory(self, dyna):
        self._active_dyna = dyna
        self._last_td = None
        self._vehicle_capacity = dyna.vehicles.new_full(
            (dyna.minibatch_size, 1),
            float(dyna.veh_capa),
        )

    def forward(self, dyna):
        dyna.reset()
        self._reset_memory(dyna)

        actions, logps, rewards = [], [], []
        while not dyna.done:
            if self.cust_repr is None or getattr(dyna, "new_customers", False):
                self._encode_customers(dyna.nodes, getattr(dyna, "cust_mask", None))

            cust_idx, logp = self.step(dyna)
            actions.append((dyna.cur_veh_idx, cust_idx))
            logps.append(logp)
            rewards.append(dyna.step(cust_idx))
        return actions, logps, rewards


class MVMoE_POMO(POMO):
    """MVMoE Model for neural combinatorial optimization based on POMO and REINFORCE
    Please refer to Zhou et al. (2024) <https://arxiv.org/abs/2405.01029>.
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: nn.Module = None,
        policy_kwargs={},
        baseline: str = "shared",
        num_augment: int = 8,
        augment_fn: str | Callable = "dihedral8",
        first_aug_identity: bool = True,
        feats: list = None,
        num_starts: int = None,
        moe_kwargs: dict = None,
        **kwargs,
    ):
        if moe_kwargs is None:
            moe_kwargs = {
                "encoder": {
                    "hidden_act": "ReLU",
                    "num_experts": 4,
                    "k": 2,
                    "noisy_gating": True,
                },
                "decoder": {
                    "light_version": True,
                    "num_experts": 4,
                    "k": 2,
                    "noisy_gating": True,
                },
            }

        if policy is None:
            policy_kwargs_ = {
                "num_encoder_layers": 6,
                "normalization": "instance",
                "use_graph_context": False,
                "moe_kwargs": moe_kwargs,
            }
            policy_kwargs.update(policy_kwargs_)
            policy = AttentionModelPolicy(env_name=env.name, **policy_kwargs)

        # Initialize with the shared baseline
        super().__init__(
            env,
            policy,
            policy_kwargs,
            baseline,
            num_augment,
            augment_fn,
            first_aug_identity,
            feats,
            num_starts,
            **kwargs,
        )


class MVMoE_AM(AttentionModel):
    """MVMoE Model for neural combinatorial optimization based on AM and REINFORCE
    Please refer to Zhou et al. (2024) <https://arxiv.org/abs/2405.01029>.
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: AttentionModelPolicy = None,
        baseline: REINFORCEBaseline | str = "rollout",
        policy_kwargs={},
        baseline_kwargs={},
        moe_kwargs: dict = None,
        **kwargs,
    ):
        if moe_kwargs is None:
            moe_kwargs = {
                "encoder": {
                    "hidden_act": "ReLU",
                    "num_experts": 4,
                    "k": 2,
                    "noisy_gating": True,
                },
                "decoder": {
                    "light_version": True,
                    "out_bias": False,
                    "num_experts": 4,
                    "k": 2,
                    "noisy_gating": True,
                },
            }

        if policy is None:
            policy_kwargs_ = {
                "moe_kwargs": moe_kwargs,
            }
            policy_kwargs.update(policy_kwargs_)
            policy = AttentionModelPolicy(env_name=env.name, **policy_kwargs)

        # Initialize with the shared baseline
        super().__init__(env, policy, baseline, policy_kwargs, baseline_kwargs, **kwargs)
