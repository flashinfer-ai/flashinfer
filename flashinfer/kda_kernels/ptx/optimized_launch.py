# Copyright 2026 Co-developed by KDA (Kernel Design Agents)
"""Adapt the retained optimized schedules to the public state contract."""

import torch

from .optimized.kda_fwd_fused import prepare_kda_forward


_LAYOUTS = ((8192,), (1300, 547, 2048, 963, 271, 3063), (1024,) * 8)


def supports(q, offsets):
    lengths = tuple(b - a for a, b in zip(offsets, offsets[1:], strict=False))
    return (
        q.shape[1] == 8192
        and lengths in _LAYOUTS
        and torch.cuda.get_device_properties(q.device).multi_processor_count == 148
    )


class OptimizedLaunch:
    def __init__(self, *args, final_state, offsets, **kwargs):
        self.final_state = final_state
        self._plan = prepare_kda_forward(*args, final_state=final_state, **kwargs)
        self._fused = self._plan._fused

    @property
    def schedule(self):
        return self._plan.schedule

    def launch(self):
        self._plan.launch()
