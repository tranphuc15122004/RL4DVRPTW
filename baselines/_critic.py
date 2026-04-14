from baselines._base import Baseline

from itertools import chain

import torch
import torch.nn as nn


class CriticBaseline(Baseline):
    STATE_FEAT_SIZE = 8

    def __init__(self, learner, cust_count, use_qval = True, use_cumul_reward = False, hidden_size = None):
        super().__init__(learner, use_cumul_reward)
        self.use_qval = use_qval
        self.cust_count = cust_count
        if hidden_size is None:
            hidden_size = 128
        out_size = cust_count + 1 if use_qval else 1
        self.project_compat = nn.Sequential(
            nn.Linear(cust_count + 1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size)
        )
        self.project_state = nn.Sequential(
            nn.Linear(self.STATE_FEAT_SIZE, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size),
        )

    def _masked_min(self, values, valid_mask):
        inf = torch.full_like(values, float("inf"))
        masked = torch.where(valid_mask, values, inf)
        min_val = masked.min(dim = 1, keepdim = True).values
        has_valid = valid_mask.any(dim = 1, keepdim = True)
        return torch.where(has_valid, min_val, torch.zeros_like(min_val))

    def _build_state_features(self, vrp_dynamics):
        cur_veh = vrp_dynamics.cur_veh.squeeze(1)
        nodes = vrp_dynamics.nodes
        device = nodes.device

        served = getattr(vrp_dynamics, "served", None)
        if served is None:
            served = torch.zeros(nodes.size(0), nodes.size(1), dtype = torch.bool, device = device)

        hidden = getattr(vrp_dynamics, "cust_mask", None)
        if hidden is None:
            hidden = torch.zeros_like(served)

        pending = (~served) & (~hidden)
        if pending.size(1) > 0:
            pending[:, 0] = False

        pending_count = pending.float().sum(dim = 1, keepdim = True)
        denom = max(vrp_dynamics.nodes_count - 1, 1)
        pending_ratio = pending_count / float(denom)

        demand = nodes[:, :, 2]
        pending_demand = (demand * pending.float()).sum(dim = 1, keepdim = True)
        mean_pending_demand = pending_demand / pending_count.clamp_min(1.0)

        if nodes.size(-1) >= 5:
            cur_time = cur_veh[:, 3:4]
            tw_close = nodes[:, :, 4]
            slack = tw_close - cur_time
            min_slack = self._masked_min(slack, pending)
            late_count = ((slack < 0) & pending).float().sum(dim = 1, keepdim = True)
            late_ratio = late_count / pending_count.clamp_min(1.0)
        else:
            min_slack = cur_veh.new_zeros((cur_veh.size(0), 1))
            late_ratio = cur_veh.new_zeros((cur_veh.size(0), 1))

        return torch.cat([
            cur_veh,
            pending_ratio,
            mean_pending_demand,
            min_slack,
            late_ratio,
        ], dim = 1)

    def eval_step(self, vrp_dynamics, learner_compat, cust_idx):
        compat = learner_compat
        if isinstance(learner_compat, dict):
            compat = learner_compat.get("compat", None)

        if compat is not None and compat.dim() == 3 and compat.size(-1) == self.cust_count + 1:
            compat = compat.detach().clone()
            compat[vrp_dynamics.cur_veh_mask] = 0
            val = self.project_compat(compat)
            if self.use_qval:
                safe_idx = cust_idx.clamp(0, val.size(2) - 1)
                val = val.gather(2, safe_idx.unsqueeze(1).expand(-1, 1, -1))
            return val.squeeze(1)

        state_feat = self._build_state_features(vrp_dynamics).detach()
        val = self.project_state(state_feat)
        if self.use_qval:
            safe_idx = cust_idx.clamp(0, val.size(1) - 1)
            val = val.gather(1, safe_idx)
        return val

    def parameters(self):
        return chain(self.project_compat.parameters(), self.project_state.parameters())

    def state_dict(self):
        return {
            "project_compat": self.project_compat.state_dict(),
            "project_state": self.project_state.state_dict(),
        }

    def load_state_dict(self, state_dict):
        if "project_compat" in state_dict:
            self.project_compat.load_state_dict(state_dict["project_compat"])
            self.project_state.load_state_dict(state_dict.get("project_state", {}), strict = False)
            return

        # Backward compatibility with old checkpoints storing only one critic head.
        self.project_compat.load_state_dict(state_dict, strict = False)

    def to(self, device):
        self.project_compat.to(device = device)
        self.project_state.to(device = device)