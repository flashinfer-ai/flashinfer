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

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch

from ..api_logging import flashinfer_experimental_api

# Thin experimental entry points; the backend (weight preparation, measured
# dispatch, launch binding, JIT registration) lives in
# flashinfer.experimental.kimi_k3_fp8_projection.


@flashinfer_experimental_api(feature="Kimi-K3 FP8 projection")
def prepare_kimi_k3_fp8_projection_weights(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    n_valid: Optional[int] = None,
    *,
    splits: Optional[Sequence[int]] = None,
    backend: str = "cake",
) -> Any:
    r"""Prepare a serialized Kimi-K3 ``FP8_PB_WO`` projection weight for :func:`kimi_k3_fp8_projection`.

    The experimental generated-program backend runs the KDA / MLA projection GEMMs of
    ``nvidia/Kimi-K3-NVFP4`` whose weights are serialized as ``FP8_PB_WO`` (E4M3 with a
    128x128 FP32 block scale) on SM100 / SM103: BF16 activations are quantized per token
    in 1x128 blocks to E4M3 with power-of-two (UE8M0) scales, exactly like DeepGEMM's
    ``per_token_cast_to_fp8(use_ue8m0=True)``, and multiplied with the weight requantized
    to UE8M0 scales (vLLM ``requant_weight_ue8m0``) through block-scaled tcgen05 MMA with
    a single BF16 rounding.

    Parameters
    ----------
    weight : torch.Tensor
        Contiguous ``float8_e4m3fn`` ``[N_pad128, K]`` checkpoint tensor (128-row block
        padding as serialized; ``K`` a multiple of 128).
    weight_scale : torch.Tensor
        ModelOpt FP32 block scale ``[N_pad128 / 128, 1, K / 128, 1]`` or its 2-D view.
    n_valid : Optional[int]
        Stored output columns (even; defaults to ``N_pad128``).
    splits : Optional[Sequence[int]]
        Branch widths of a fused projection (they sum to ``n_valid``); ``output_views``
        of the prepared weight then slices a fused output at the split points.
    backend : str
        Only ``"cake"`` is supported.

    Returns
    -------
    PreparedProjectionWeight
        Requantized, tiled weight and scale tiles (device tensors) plus the shape record.
    """
    if backend != "cake":
        raise ValueError("the Kimi-K3 FP8 projection currently supports backend='cake'")
    from ..experimental.kimi_k3_fp8_projection.cake_backend import (
        prepare_kimi_k3_fp8_projection_weights as prepare,
    )

    return prepare(weight, weight_scale, n_valid, splits=splits)


def allocate_kimi_k3_fp8_projection_workspace(prepared: Any, M: int) -> Any:
    """Allocate the caller-owned workspaces for ``M`` activation rows (no launch)."""
    from ..experimental.kimi_k3_fp8_projection.cake_backend import (
        allocate_kimi_k3_fp8_projection_workspace as allocate,
    )

    return allocate(prepared, M)


@flashinfer_experimental_api(feature="Kimi-K3 FP8 projection")
def prepare_kimi_k3_fp8_projection(
    x: torch.Tensor,
    prepared: Any,
    out: torch.Tensor,
    workspace: Any,
    *,
    backend: str = "cake",
) -> Any:
    r"""Bind one ``(x, prepared, out, workspace)`` call of the Kimi-K3 FP8 projection for repeated launches.

    Parameters
    ----------
    x : torch.Tensor
        Contiguous BF16 ``[M, K]`` activations (read on device at every launch).
    prepared : PreparedProjectionWeight
        From :func:`prepare_kimi_k3_fp8_projection_weights`.
    out : torch.Tensor
        BF16 ``[M, n_valid]`` view with unit column stride and an even row stride (any
        view into a wider buffer; only the ``n_valid`` columns are written).
    workspace : ProjectionWorkspace
        From :func:`allocate_kimi_k3_fp8_projection_workspace` for this ``M``.
    backend : str
        Only ``"cake"`` is supported.

    Returns
    -------
    KimiK3Fp8ProjectionRunner
        Calling it launches the quantization (unless the selected decode instance
        quantizes in-CTA) and the GEMM / decode program on the current stream with no
        CUDA allocation or host synchronisation and returns ``out``.  CUDA Graph capture
        of the runner is supported; prepare outside capture.
    """
    if backend != "cake":
        raise ValueError("the Kimi-K3 FP8 projection currently supports backend='cake'")
    from ..experimental.kimi_k3_fp8_projection.cake_backend import (
        prepare_kimi_k3_fp8_projection as prepare,
    )

    return prepare(x, prepared, out, workspace)


@flashinfer_experimental_api(feature="Kimi-K3 FP8 projection")
def kimi_k3_fp8_projection(
    x: torch.Tensor,
    prepared: Any,
    out: Optional[torch.Tensor] = None,
    *,
    workspace: Any = None,
    backend: str = "cake",
) -> torch.Tensor:
    r"""``out = bf16(x @ dequant(weight).T)`` with the serialized ``FP8_PB_WO`` recipe in one call.

    Allocates ``out`` ``[M, n_valid]`` and the workspace when they are omitted; see
    :func:`prepare_kimi_k3_fp8_projection` for the allocation-free repeated form.
    """
    if backend != "cake":
        raise ValueError("the Kimi-K3 FP8 projection currently supports backend='cake'")
    from ..experimental.kimi_k3_fp8_projection.cake_backend import (
        kimi_k3_fp8_projection as run,
    )

    return run(x, prepared, out, workspace=workspace)
