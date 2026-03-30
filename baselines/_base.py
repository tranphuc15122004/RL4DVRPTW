import torch

from utils.learner_protocol import supports_standard_rollout_api

class Baseline:
    def __init__(self, learner, use_cumul_reward = False):
        self.learner = learner
        self.use_cumul = use_cumul_reward

    def __call__(self, vrp_dynamics):
        if self.use_cumul:
            actions, logps, rewards = self.learner(vrp_dynamics)
            rewards = torch.stack(rewards).sum(dim = 0)
            bl_vals = self.eval(vrp_dynamics)
        else:
            vrp_dynamics.reset()
            self._prepare_rollout(vrp_dynamics)
            actions, logps, rewards, bl_vals = [], [], [], []
            while not vrp_dynamics.done:
                cust_idx, logp, compat = self._policy_step(vrp_dynamics)
                bl_vals.append( self.eval_step(vrp_dynamics, compat, cust_idx) )
                actions.append( (vrp_dynamics.cur_veh_idx, cust_idx) )
                logps.append(logp)
                r = vrp_dynamics.step(cust_idx)
                rewards.append(r)
        self.update(rewards, bl_vals)
        return actions, logps, rewards, bl_vals

    def _prepare_rollout(self, vrp_dynamics):
        if hasattr(vrp_dynamics, "veh_speed"):
            self.learner.veh_speed = vrp_dynamics.veh_speed

        # Preferred generic interface: each model can initialize rollout however it wants.
        if supports_standard_rollout_api(self.learner):
            self.learner.on_rollout_start(vrp_dynamics)
            return

        # Backward-compatible path for legacy models.
        if hasattr(self.learner, "_encode_customers"):
            cust_mask = getattr(vrp_dynamics, "cust_mask", None)
            self.learner._encode_customers(vrp_dynamics.nodes, cust_mask)
        if hasattr(self.learner, "_reset_memory"):
            self.learner._reset_memory(vrp_dynamics)

    def _sample_from_logp(self, logp, cur_veh_mask, nodes_count = None):
        probs = logp.exp()
        bad = (~torch.isfinite(probs)).any(dim = 1, keepdim = True) | (probs.sum(dim = 1, keepdim = True) <= 0)
        if bad.any():
            safe = torch.zeros_like(probs)
            safe[:, 0] = 1.0
            probs = torch.where(bad, safe, probs)

        if getattr(self.learner, "greedy", False):
            cust_idx = probs.argmax(dim = 1, keepdim = True)
        else:
            cust_idx = probs.multinomial(1)

        if cust_idx.dtype != torch.int64:
            cust_idx = cust_idx.long()
        if nodes_count is not None and nodes_count > 0:
            cust_idx = cust_idx.clamp(0, nodes_count - 1)

        chosen_mask = cur_veh_mask.gather(2, cust_idx.unsqueeze(1)).squeeze(1)
        if chosen_mask.any():
            cust_idx = cust_idx.masked_fill(chosen_mask, 0)
        return cust_idx

    def _legacy_policy_step(self, vrp_dynamics):
        veh_repr = self.learner._repr_vehicle(
                vrp_dynamics.vehicles,
                vrp_dynamics.cur_veh_idx,
                vrp_dynamics.mask)

        edge_emb = None
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

        logp_full = self.learner._get_logp(compat, vrp_dynamics.cur_veh_mask)
        cust_idx = self._sample_from_logp(logp_full, vrp_dynamics.cur_veh_mask, vrp_dynamics.nodes_count)

        if hasattr(self.learner, "_update_memory") and edge_emb is not None:
            self.learner._update_memory(vrp_dynamics.cur_veh_idx, cust_idx, veh_repr, edge_emb)

        return cust_idx, logp_full.gather(1, cust_idx), compat

    def _policy_step(self, vrp_dynamics):
        if supports_standard_rollout_api(self.learner):
            out = self.learner.step(vrp_dynamics)
            if not isinstance(out, (tuple, list)) or len(out) < 2:
                raise RuntimeError("learner.step must return at least (cust_idx, logp_or_selected_logp)")

            cust_idx, logp = out[0], out[1]
            compat = None
            if len(out) > 2:
                extra = out[2]
                if isinstance(extra, dict):
                    compat = extra.get("compat", None)
                elif torch.is_tensor(extra):
                    compat = extra

            if logp.dim() == 1:
                logp = logp.unsqueeze(1)
            elif logp.dim() == 2 and logp.size(1) > 1:
                logp = logp.gather(1, cust_idx)

            return cust_idx, logp, compat

        return self._legacy_policy_step(vrp_dynamics)

    def eval(self, vrp_dynamics):
        raise NotImplementedError()

    def eval_step(self, vrp_dynamics, learner_compat, cust_idx):
        raise NotImplementedError()

    def update(self, rewards, bl_vals):
        pass

    def parameters(self):
        return []

    def state_dict(self, destination = None):
        return {}

    def load_state_dict(self, state_dict):
        pass

    def to(self, device):
        pass