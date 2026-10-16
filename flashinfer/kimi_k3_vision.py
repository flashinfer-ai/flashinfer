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
"""

"""Experimental Kimi-K3 vision tower (MoonViT-3D encoder + PatchMergerV2) on SM100 / SM103."""

from typing import Any, Optional, Sequence

import torch

from .api_logging import flashinfer_experimental_api


@flashinfer_experimental_api(feature="Kimi-K3 vision tower")
def kimi_k3_vision_tower(
    pixel_values: torch.Tensor,
    grid_thws: Sequence[Sequence[int]],
    weights: Any,
    out: Optional[torch.Tensor] = None,
    *,
    plan: Any = None,
    pos_rows: Optional[torch.Tensor] = None,
    backend: str = "cake",
) -> torch.Tensor:
    r"""Complete Kimi-K3 vision tower: packed patch pixels to projected vision tokens.

    The experimental Cake backend runs the ``nvidia/Kimi-K3-NVFP4`` vision path
    (``MoonViT3dPretrainedModel`` + ``PatchMergerMLPV2``) as one sequence of
    generated tcgen05 programs: the 14x14 patch embedding with positional rows,
    27 encoder layers (RMSNorm fused into the QKV GEMM with the 2-D RoPE in
    its epilogue, packed-varlen noncausal BF16 attention per ``grid_thw``
    segment, out-projection and FC1 with fused residual adds that also emit
    the next norm's weighted activation and row statistics, RMSNorm-fused FC0
    with GELU-tanh), the final RMSNorm with the 2x2 spatial / temporal-mean
    merge, and the projector GEMMs with GELU-erf and the post RMSNorm
    (flashinfer-ai/flashinfer#4568).

    Parameters
    ----------
    pixel_values : torch.Tensor
        Contiguous BF16 ``[T, 3, 14, 14]`` normalized patches in packed
        ``grid_thws`` order (``t`` slow, then ``y``, then ``x``).
    grid_thws : Sequence[Sequence[int]]
        Host list of ``(t, h, w)`` per image (``t = 1``) or 4-frame video group
        (``t <= 4``); ``h``, ``w`` even and at most 512; ``T = sum t * h * w``.
    weights : dict or PreparedWeights
        The BF16 ``nn.Linear`` ``[out, in]`` parameters (see
        ``cake_backend.prepare_kimi_k3_vision_weights``) or their prepared form.
        Prepare once per model; the dict form is prepared on every call.
    out : Optional[torch.Tensor]
        Optional caller-owned BF16 ``[N, 7168]`` output, ``N = sum (h/2)*(w/2)``.
    plan, pos_rows
        Optional cached per-``grid_thws`` plan (``build_kimi_k3_vision_plan``)
        and positional rows (``pos_emb_rows``); derived here when omitted.
    backend : str
        Only ``"cake"`` is supported.

    Returns
    -------
    torch.Tensor
        The BF16 ``[N, 7168]`` output (``out`` when given).  For repeated
        launches or CUDA Graph capture use :func:`prepare_kimi_k3_vision_tower`,
        whose runner launches with no allocation or synchronization.  See
        ``flashinfer/experimental/kimi_k3_vision_tower/README.md``.
    """
    if backend != "cake":
        raise ValueError("Kimi-K3 vision tower currently supports backend='cake'")
    from .experimental.kimi_k3_vision_tower.cake_backend import (
        kimi_k3_vision_tower as run,
    )

    return run(
        pixel_values,
        grid_thws,
        weights,
        out,
        plan=plan,
        pos_rows=pos_rows,
        backend="cake",
    )


@flashinfer_experimental_api(feature="Prepared Kimi-K3 vision tower")
def prepare_kimi_k3_vision_tower(
    pixel_values: torch.Tensor,
    grid_thws: Sequence[Sequence[int]],
    weights: Any,
    out: Optional[torch.Tensor] = None,
    *,
    plan: Any = None,
    pos_rows: Optional[torch.Tensor] = None,
    backend: str = "cake",
) -> Any:
    r"""Plan and bind one Kimi-K3 vision tower call; returns a runner.

    Same arguments as :func:`kimi_k3_vision_tower`.  Every allocation (the
    optional output, the workspaces, the host plan tables) happens here; the
    returned ``KimiK3VisionTowerRunner.launch()`` writes ``out`` on the current
    stream with no allocation or host synchronization and may be captured into
    a CUDA graph by the caller.  Prepare a new runner when ``grid_thws`` or a
    tensor binding changes; values may change freely.
    """
    if backend != "cake":
        raise ValueError("Kimi-K3 vision tower currently supports backend='cake'")
    from .experimental.kimi_k3_vision_tower.cake_backend import (
        prepare_kimi_k3_vision_tower as prepare,
    )

    return prepare(
        pixel_values,
        grid_thws,
        weights,
        out,
        plan=plan,
        pos_rows=pos_rows,
        backend="cake",
    )


__all__ = ["kimi_k3_vision_tower", "prepare_kimi_k3_vision_tower"]
