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
        self._static_td = None
        self._decoder_cache = None
        self._vehicle_capacity = None
        self._skip_refresh_once = False

    def step(self, dyna):
        if self._decoder_cache is None:
            raise RuntimeError("Call on_rollout_start before step")

        # Refresh customer cache when new dynamic customers are revealed.
        if getattr(dyna, "new_customers", False):
            if self._skip_refresh_once:
                self._skip_refresh_once = False
            else:
                customers = dyna.nodes
                cust_mask = getattr(dyna, "cust_mask", None)
                model_in = customers if cust_mask is None else customers.masked_fill(cust_mask.unsqueeze(-1), 0)

                td = TensorDict(
                    {
                        "locs": model_in[:, :, :2],
                        "demand": model_in[:, 1:, 2],
                        "durations": model_in[:, :, 5],
                        "time_windows": model_in[:, :, 3:5],
                    },
                    batch_size = [model_in.size(0)],
                    device = model_in.device,
                )
                cust_repr, _ = self.policy.encoder(td)
                if cust_mask is not None:
                    cust_repr = cust_repr.masked_fill(cust_mask.unsqueeze(-1), 0)
                self._static_td = td
                self.cust_repr = cust_repr
                self._decoder_cache = self.policy.decoder._precompute_cache(cust_repr)

        veh_mask = dyna.cur_veh_mask
        if veh_mask.dim() == 3 and veh_mask.size(1) != 1:
            veh_mask = veh_mask.gather(1, dyna.cur_veh_idx[:, :, None].expand(-1, -1, veh_mask.size(-1)))

        cur_veh = dyna.vehicles.gather(
            1,
            dyna.cur_veh_idx[:, :, None].expand(-1, -1, dyna.vehicles.size(-1)),
        )
        cur_pos = cur_veh[:, 0, :2]
        locs = self._static_td["locs"]
        current_node = torch.cdist(cur_pos[:, None, :], locs).squeeze(1).argmin(dim = 1, keepdim = True)

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
            batch_size = [dyna.vehicles.size(0)],
            device = dyna.vehicles.device,
        )

        compat, _ = self.policy.decoder(step_td, self._decoder_cache, num_starts = 0)
        mask = veh_mask.squeeze(1)
        all_masked = mask.all(dim = 1, keepdim = True)
        if all_masked.any():
            mask = mask.clone()
            mask[all_masked.squeeze(1), 0] = False
        scores = compat.clone()
        scores[mask] = -float("inf")
        logp_full = scores.log_softmax(dim = 1)

        probs = logp_full.exp()
        bad = (~torch.isfinite(probs)).any(dim = 1, keepdim = True) | (probs.sum(dim = 1, keepdim = True) <= 0)
        if bad.any():
            safe = torch.zeros_like(probs)
            safe[:, 0] = 1.0
            probs = torch.where(bad, safe, probs)

        cust_idx = probs.argmax(dim = 1, keepdim = True) if self.greedy else probs.multinomial(1)
        if cust_idx.dtype != torch.int64:
            cust_idx = cust_idx.long()
        chosen_mask = veh_mask.gather(2, cust_idx.unsqueeze(1)).squeeze(1)
        if chosen_mask.any():
            cust_idx = cust_idx.masked_fill(chosen_mask, 0)

        return cust_idx, logp_full.gather(1, cust_idx), {"compat": compat.unsqueeze(1)}

    def on_rollout_start(self, dyna):
        # Reset cached MoE gate probs so each rollout uses a fresh autograd graph.
        pointer = getattr(self.policy.decoder, "pointer", None)
        if pointer is not None and hasattr(pointer, "probs"):
            pointer.probs = None

        self._vehicle_capacity = dyna.vehicles.new_full(
            (dyna.minibatch_size, 1),
            float(dyna.veh_capa),
        )
        customers = dyna.nodes
        cust_mask = getattr(dyna, "cust_mask", None)
        model_in = customers if cust_mask is None else customers.masked_fill(cust_mask.unsqueeze(-1), 0)
        td = TensorDict(
            {
                "locs": model_in[:, :, :2],
                "demand": model_in[:, 1:, 2],
                "durations": model_in[:, :, 5],
                "time_windows": model_in[:, :, 3:5],
            },
            batch_size = [model_in.size(0)],
            device = model_in.device,
        )

        cust_repr, _ = self.policy.encoder(td)
        if cust_mask is not None:
            cust_repr = cust_repr.masked_fill(cust_mask.unsqueeze(-1), 0)

        self._static_td = td
        self.cust_repr = cust_repr
        self._decoder_cache = self.policy.decoder._precompute_cache(cust_repr)
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
