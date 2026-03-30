from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class StandardRolloutLearner(Protocol):
    """Minimal learner contract for this project training pipeline.

    Models implementing this protocol can run end-to-end with baselines without
    exposing model-specific private methods.
    """

    def on_rollout_start(self, dyna: Any) -> None:
        """Initialize per-episode state after environment reset."""

    def step(self, dyna: Any):
        """Return either (cust_idx, selected_logp) or (cust_idx, selected_logp, extra)."""

    def forward(self, dyna: Any):
        """Run a full episode and return (actions, logps, rewards)."""


def supports_standard_rollout_api(learner: Any) -> bool:
    return callable(getattr(learner, "on_rollout_start", None)) and callable(getattr(learner, "step", None))
