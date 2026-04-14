import torch
import torch.nn as nn

from tensordict import TensorDict

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl import REINFORCE
from rl4co.models.rl.reinforce.baselines import REINFORCEBaseline
from rl4co.models.zoo.am.policy import AttentionModelPolicy
from rl4co.utils.decoding import process_logits


class AM_DVRPTW(nn.Module):
    """Adapter exposing the stepwise API used by the repository DVRPTW pipeline."""

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
    ):
        super().__init__()
        self.policy = AttentionModelPolicy(
            env_name="cvrptw",
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

        invalid_mask = veh_mask.squeeze(1)
        feasible_mask = ~invalid_mask
        all_masked = ~feasible_mask.any(dim=1, keepdim=True)
        if all_masked.any():
            feasible_mask = feasible_mask.clone()
            feasible_mask[all_masked.squeeze(1), 0] = True

        logp_full = process_logits(
            compat.clone(),
            mask=feasible_mask,
            temperature=self.policy.temperature,
            tanh_clipping=self.policy.tanh_clipping,
            mask_logits=self.policy.mask_logits,
        )

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


class AttentionModel(REINFORCE):
    """Attention Model based on REINFORCE: https://arxiv.org/abs/1803.08475.
    Check :class:`REINFORCE` and :class:`rl4co.models.RL4COLitModule` for more details such as additional parameters  including batch size.

    Args:
        env: Environment to use for the algorithm
        policy: Policy to use for the algorithm
        baseline: REINFORCE baseline. Defaults to rollout (1 epoch of exponential, then greedy rollout baseline)
        policy_kwargs: Keyword arguments for policy
        baseline_kwargs: Keyword arguments for baseline
        **kwargs: Keyword arguments passed to the superclass
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: AttentionModelPolicy = None,
        baseline: REINFORCEBaseline | str = "rollout",
        policy_kwargs={},
        baseline_kwargs={},
        **kwargs,
    ):
        if policy is None:
            policy = AttentionModelPolicy(env_name=env.name, **policy_kwargs)

        super().__init__(env, policy, baseline, baseline_kwargs, **kwargs)
