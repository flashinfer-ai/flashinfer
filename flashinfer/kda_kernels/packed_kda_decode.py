"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Serving-native packed Kimi K3 recurrent decode for exact SM100a and SM103a.
"""

from typing import Optional

import torch

from ..jit.cpp_ext import is_cuda_version_at_least
from ..jit.flash_kda_packed_t1 import (
    FlashKDAPackedT1Target,
    _variant_for_batch,
    get_flash_kda_packed_t1_module,
)
from ..utils import get_compute_capability

_HEADS = 12
_HEAD_DIM = 128
_MIXED_WIDTH = 3 * _HEADS * _HEAD_DIM


def _target_for_device(device: torch.device) -> FlashKDAPackedT1Target:
    """Select an exact physical target and enforce its CUDA floor."""

    compute_capability = get_compute_capability(device)
    if compute_capability == (10, 0):
        if not is_cuda_version_at_least("12.8"):
            raise RuntimeError(
                "packed KDA T=1 on compute capability 10.0 requires CUDA 12.8 or newer"
            )
        return "sm100a"
    if compute_capability == (10, 3):
        if not is_cuda_version_at_least("12.9"):
            raise RuntimeError(
                "packed KDA T=1 on compute capability 10.3 requires CUDA 12.9 or newer"
            )
        return "sm103a"
    raise RuntimeError(
        "packed KDA T=1 requires exact compute capability 10.0 "
        "(SM100a) or 10.3 (SM103a); got "
        f"{compute_capability[0]}.{compute_capability[1]}"
    )


def run_packed_kda_decode(
    mixed_qkv: torch.Tensor,
    raw_gate: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Launch one packed recurrent update on the caller's current stream.

    The state pool is updated in place. Active values in ``state_indices`` are
    a device-side caller contract: they must be unique and in bounds. ``-1``
    marks an inactive graph-padding row, which produces zero output and does
    not access the state pool. Values are intentionally not inspected on the
    host so the call remains allocation-free and CUDA-graph compatible when a
    caller-owned output is supplied.
    """

    if not isinstance(mixed_qkv, torch.Tensor):
        raise TypeError("mixed_qkv must be a torch.Tensor")
    if not mixed_qkv.is_cuda:
        raise ValueError("mixed_qkv must be a CUDA tensor")
    if mixed_qkv.ndim != 2 or mixed_qkv.shape[1] != _MIXED_WIDTH:
        raise ValueError(f"mixed_qkv must have shape [B, {_MIXED_WIDTH}]")
    batch = int(mixed_qkv.shape[0])
    if batch <= 0 or batch > 65535:
        raise ValueError(
            "packed KDA T=1 batch must be in the CUDA grid.y range "
            f"[1, 65535], got {batch}"
        )

    if output is None:
        output = mixed_qkv.new_empty((batch, 1, _HEADS, _HEAD_DIM))
    elif not isinstance(output, torch.Tensor):
        raise TypeError("output must be a torch.Tensor")
    if (
        tuple(output.shape) != (batch, 1, _HEADS, _HEAD_DIM)
        or not output.is_contiguous()
    ):
        raise ValueError("output must be contiguous with shape [B,1,12,128]")
    output_view = output.view(batch, _HEADS, _HEAD_DIM)

    variant = _variant_for_batch(batch)
    target = _target_for_device(mixed_qkv.device)
    module = get_flash_kda_packed_t1_module(variant, target)
    module.run(
        mixed_qkv,
        raw_gate,
        raw_beta,
        A_log,
        dt_bias,
        state,
        state_indices,
        output_view,
        int(torch.cuda.current_stream(mixed_qkv.device).cuda_stream),
    )
    return output


__all__ = ["run_packed_kda_decode"]
