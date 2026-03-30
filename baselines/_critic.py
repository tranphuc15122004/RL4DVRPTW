from baselines._base import Baseline

import torch
import torch.nn as nn


class CriticBaseline(Baseline):
    def __init__(self, learner, cust_count, use_qval = True, use_cumul_reward = False, hidden_size = None):
        super().__init__(learner, use_cumul_reward)
        self.use_qval = use_qval
        if hidden_size is None:
            hidden_size = 128
        out_size = cust_count + 1 if use_qval else 1
        self.project = nn.Sequential(
            nn.Linear(cust_count + 1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size)
        )

    def eval_step(self, vrp_dynamics, learner_compat, cust_idx):
        compat = learner_compat.detach().clone()
        compat[vrp_dynamics.cur_veh_mask] = 0
        val = self.project(compat)
        if self.use_qval:
            # Guard against rare invalid sampled indices on GPU to avoid device-side asserts.
            safe_idx = cust_idx.clamp_(0, val.size(2) - 1)
            val = val.gather(2, safe_idx.unsqueeze(1).expand(-1,1,-1))
        return val.squeeze(1)

    def __call__(self, vrp_dynamics):
        vrp_dynamics.reset()
        cust_mask = getattr(vrp_dynamics, "cust_mask", None)
        self.learner._encode_customers(vrp_dynamics.nodes, cust_mask)
        if hasattr(vrp_dynamics, "veh_speed"):
            self.learner.veh_speed = vrp_dynamics.veh_speed
        if hasattr(self.learner, "_reset_memory"):
            self.learner._reset_memory(vrp_dynamics)
        actions, logps, rewards, bl_vals = [], [], [], []
        while not vrp_dynamics.done:
            veh_repr = self.learner._repr_vehicle(
                    vrp_dynamics.vehicles,
                    vrp_dynamics.cur_veh_idx,
                    vrp_dynamics.mask)
            if hasattr(self.learner, "_compute_edge_embedding") and hasattr(self.learner, "_compute_owner_bias") and hasattr(self.learner, "_compute_lookahead"):
                edge_emb = self.learner._compute_edge_embedding(
                    vrp_dynamics.vehicles,
                    vrp_dynamics.nodes,
                    vrp_dynamics.cur_veh_idx,
                    vrp_dynamics.cur_veh_mask,
                )
                owner_bias = self.learner._compute_owner_bias(vrp_dynamics.cur_veh_idx)
                lookahead = self.learner._compute_lookahead(veh_repr, self.learner.cust_repr, edge_emb)
                compat = self.learner._score_customers(
                    veh_repr,
                    self.learner.cust_repr,
                    edge_emb,
                    owner_bias,
                    lookahead,
                    vrp_dynamics.cur_veh_mask)
            else:
                compat = self.learner._score_customers(veh_repr)
            logp = self.learner._get_logp(compat, vrp_dynamics.cur_veh_mask)
            probs = logp.exp()
            bad = (~torch.isfinite(probs)).any(dim = 1, keepdim = True) | (probs.sum(dim = 1, keepdim = True) <= 0)
            if bad.any():
                safe = torch.zeros_like(probs)
                safe[:, 0] = 1.0
                probs = torch.where(bad, safe, probs)
            cust_idx = probs.multinomial(1)
            # Keep actions in valid domain for all downstream gather/scatter calls.
            if cust_idx.dtype != torch.int64:
                cust_idx = cust_idx.long()
            if vrp_dynamics.nodes_count > 0:
                cust_idx = cust_idx.clamp(0, vrp_dynamics.nodes_count - 1)

            # If a masked customer was sampled due to numerical issues, route to depot.
            chosen_mask = vrp_dynamics.cur_veh_mask.gather(2, cust_idx.unsqueeze(1)).squeeze(1)
            if chosen_mask.any():
                cust_idx = cust_idx.masked_fill(chosen_mask, 0)

            if not(self.use_cumul and bl_vals):
                bl_vals.append( self.eval_step(vrp_dynamics, compat, cust_idx) )
            if hasattr(self.learner, "_update_memory") and "edge_emb" in locals():
                self.learner._update_memory(vrp_dynamics.cur_veh_idx, cust_idx, veh_repr, edge_emb)
            actions.append( (vrp_dynamics.cur_veh_idx, cust_idx) )
            logps.append( logp.gather(1, cust_idx) )
            r = vrp_dynamics.step(cust_idx)
            rewards.append(r)
        if self.use_cumul:
            rewards = torch.stack(rewards).sum(dim = 0)
            bl_vals = bl_vals[0]
        return actions, logps, rewards, bl_vals

    def parameters(self):
        return self.project.parameters()

    def state_dict(self):
        return self.project.state_dict()

    def load_state_dict(self, state_dict):
        return self.project.load_state_dict(state_dict)

    def to(self, device):
        self.project.to(device = device)