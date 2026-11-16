# SPDX-License-Identifier: Apache-2.0
# Adapted from the sum-state handoff in sgl-project/sglang at 9cdd6624fe.
"""Invocation lifetime and row-partitioned sum state, without serving policy."""

import torch

from flashinfer.hyperconnection import HCApplyContext, HCSumState

from .api import HCSumPlan


class HCSumRuntime:
    """Allocator-backed HC execution for a sequential inference lane.

    Create outside CUDA Graph capture. A small initialized Down scratch is
    reused on one ordered execution stream, including sequential graph replay.
    Concurrent streams/replays require separate runtimes (or caller-owned
    HCSumPlan workspaces). Invocation outputs use the current PyTorch allocator
    and may be allocated during capture; their lifetime follows the returned
    context/state. Kernel dispatch and arithmetic are identical to HCSumPlan.
    """

    def __init__(self, plan, *, deterministic=False):
        if not isinstance(plan, HCSumPlan) or type(deterministic) is not bool:
            raise ValueError("runtime requires an HCSumPlan and a bool mode")
        with torch.cuda.device(plan.up.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("create HC runtime outside CUDA Graph capture")
        self.plan = plan
        self.deterministic = deterministic
        self._small_down = torch.zeros(
            (24, plan.kpad), dtype=plan.up.dtype, device=plan.up.device
        )

    def mix(self, residual, prepared=None):
        plan = self.plan
        rows = plan._residual(residual)
        if prepared is not None and not isinstance(prepared, HCSumState):
            raise ValueError("prepared must be an HCSumState or None")
        sums = (
            prepared.sum_sq
            if prepared is not None and prepared.matches(residual)
            else plan.bootstrap(residual)
        )
        # A contiguous odd-row partition starts at a 16-byte offset. The
        # small-T CuTe vector load needs 32-byte alignment; keep exact carried
        # statistics instead of recomputing from rounded residuals.
        if not isinstance(sums, torch.Tensor):
            raise ValueError("prepared sum_sq must be a Tensor")
        if rows <= 24 and sums.data_ptr() % 32:
            sums = sums.clone(memory_format=torch.contiguous_format)
        ws = plan._allocate_workspace(
            rows,
            deterministic=self.deterministic,
            small_down=self._small_down[:rows] if rows <= 24 else None,
        )
        mixed, alpha = plan.mix(residual, sums, ws)
        return mixed, HCApplyContext(
            residual, alpha, ws.residual, ws.sum_sq, ws.partial_sums, ws, sums
        )
