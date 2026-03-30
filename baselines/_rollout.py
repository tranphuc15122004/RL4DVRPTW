from baselines._base import Baseline
from dep import SCIPY_ENABLED, ttest_rel

import copy
import torch

class RolloutBaseline(Baseline):
    def __init__(self, learner, rollout_count = 1, update_threshold = 0.05):
        super().__init__(learner, True)

        if not SCIPY_ENABLED:
            raise RuntimeError("Cannot use rollout baseline without scipy.stats.ttest_rel")

        self.learner = learner
        self.policy = self._safe_deepcopy(learner)
        self.policy.eval()
        self.count = rollout_count
        self.thresh = update_threshold

    def _safe_deepcopy(self, learner):
        cached = {}
        for name, value in learner.__dict__.items():
            if torch.is_tensor(value) and value.requires_grad and not value.is_leaf:
                cached[name] = value
                setattr(learner, name, None)
            elif torch.is_tensor(value) and not value.is_leaf:
                cached[name] = value
                setattr(learner, name, None)
            elif isinstance(value, (list, dict)):
                continue
            elif hasattr(value, "requires_grad") and not getattr(value, "is_leaf", True):
                cached[name] = value
                setattr(learner, name, None)
        try:
            copied = copy.deepcopy(learner)
        finally:
            for name, value in cached.items():
                setattr(learner, name, value)
        return copied

    def eval(self, dyna):
        val = []
        with torch.no_grad():
            for it in range(self.count):
                _,_,rewards = self.policy(dyna)
                val.append( torch.stack(rewards).sum(dim = 0) )
        return torch.stack(val).mean(dim = 0)

    def update(self, rewards, bl_vals):
        if (rewards - bl_vals).mean() > 0:
            t, p = ttest_rel(rewards.numpy(), bl_vals.numpy())
            if p > 1-self.thresh:
                self.policy.load_state_dict(self.learner.state_dict())

    def to(self, device):
        self.policy.to(device = device)
