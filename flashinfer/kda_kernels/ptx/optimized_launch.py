# Copyright 2026 Co-developed by KDA (Kernel Design Agents)
# Copyright (c) 2026 KDA Team
# SPDX-License-Identifier: MIT
# See LICENSE.kda-for-kda.txt for the full license.

"""Route the six validated INT21 layouts through the shared KDA front end."""

import torch

from .kda_fwd_fused import KDAForwardLaunch
from .optimized.scheduler import prepare_fwd_pieces

_LAYOUTS = ((8192,), (1300, 547, 2048, 963, 271, 3063), (1024,) * 8)


def supports(q, offsets):
    """Select only validated 8192-token layouts on a 148-SM B300."""
    lengths = tuple(b - a for a, b in zip(offsets, offsets[1:], strict=False))
    return (
        q.shape[1] == 8192
        and lengths in _LAYOUTS
        and torch.cuda.get_device_properties(q.device).multi_processor_count == 148
    )


def _prepare_exact(*args, allow_approximate_split, use_expected_norm, **kwargs):
    """Adapt shared front-end options to the exact-only optimized scheduler."""
    if allow_approximate_split or use_expected_norm:
        raise ValueError(
            "the optimized route requires exact planning and normalization"
        )
    return prepare_fwd_pieces(*args, **kwargs)


class OptimizedLaunch(KDAForwardLaunch):
    """Use the shared launch surface with the optimized exact scheduler."""

    def __init__(self, *args, **kwargs):
        """Prepare a private plan without executing or mutating its inputs."""
        super().__init__(*args, _prepare=_prepare_exact, **kwargs)
