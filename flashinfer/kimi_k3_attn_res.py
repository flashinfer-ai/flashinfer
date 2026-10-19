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

"""Experimental Kimi-K3 attention-residual mixing (AttnRes) on SM100 / SM103."""

from typing import Any, Optional

import torch

from .api_logging import flashinfer_experimental_api

# Thin experimental entry points; the host planner (route selection from
# architecture / SM count / M / K / PDL policy / semantic flags), launch binding
# and JIT registration live in flashinfer.experimental.cake_kimi_k3_attn_res.
_FEATURE = "Kimi-K3 AttnRes"


@flashinfer_experimental_api(feature=_FEATURE)
def prepare_kimi_k3_attn_res(
    prefix: torch.Tensor,
    delta: Optional[torch.Tensor],
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    output_norm_weight: Optional[torch.Tensor],
    out: torch.Tensor,
    *,
    num_blocks: int,
    block_write_idx: int = -1,
    eps: float = 1e-5,
    output_norm_eps: float = 1e-5,
    enable_pdl: bool = False,
    backend: str = "cake",
) -> Any:
    r"""Prepare one Kimi-K3 AttnRes call for repeated launches.

    The experimental generated-program backend computes, in one launch, the
    attention-residual mixing of ``Kimi-K3`` (``_apply_attn_res``) on
    caller-owned BF16 tensors with hidden size 7168 and ``K = num_blocks`` in
    ``[0, 8]``::

        prefix += delta                              (BF16 rounded; when delta is given)
        blocks[:, block_write_idx, :] = prefix       (exact snapshot; when block_write_idx >= 0)
        c_i = blocks[:, i, :] for i < K,  c_K = prefix
        s_i = sum_h rmsnorm(c_i, eps)[h] * norm_weight[h] * qk_weight[h]        (FP32)
        out = BF16(rmsnorm(sum_i softmax(s)_i * c_i, output_norm_eps) * output_norm_weight)

    (``out = BF16(sum_i softmax(s)_i * c_i)`` without ``output_norm_weight``).

    Parameters
    ----------
    prefix : torch.Tensor
        BF16 ``[M, 7168]`` residual, mutated in place when ``delta`` is given.
    delta : Optional[torch.Tensor]
        BF16 ``[M, 7168]`` residual increment, or ``None``.
    blocks : torch.Tensor
        BF16 ``[M, B, 7168]`` snapshot bank (``B <= 8``; the persistent path
        requires ``B == 8``); row ``block_write_idx`` is overwritten when it is
        not ``-1``; every other byte is preserved.
    norm_weight, qk_weight : torch.Tensor
        BF16 ``[7168]`` read-only score weights.
    output_norm_weight : Optional[torch.Tensor]
        BF16 ``[7168]`` output RMSNorm weight, or ``None`` to skip the output norm.
    out : torch.Tensor
        Caller-owned contiguous BF16 ``[M, 7168]`` output.
    num_blocks : int
        ``K`` in ``[0, 8]``: the leading snapshot rows mixed with the prefix.
    block_write_idx : int
        ``-1`` or the snapshot row receiving the updated prefix.
    eps, output_norm_eps : float
        RMSNorm epsilons of the candidates and of the output.
    enable_pdl : bool
        Launch the program with programmatic dependent launch.
    backend : str
        Only ``"cake"`` is supported.

    Returns
    -------
    KimiK3AttnResRunner
        ``launch()`` runs the program on the current stream without allocating
        and returns ``out``; it is CUDA-graph capturable (capture belongs to the
        caller).  ``plan`` records the selected program and launch geometry.

    Notes
    -----
    Dense token-major inputs with ``delta`` and ``output_norm_weight`` and no
    snapshot write take the persistent TMEM path; other variants take the
    one-CTA-per-token program.  The checkout registers the programs of the
    measured token counts ``M`` in {1, 2, 4, ..., 16384} and block counts of the
    Kimi-K3 evaluation; a call whose program is not registered raises
    ``NotImplementedError`` (see :func:`generated_program_available`).
    """
    if backend == "cake":
        from .experimental.cake_kimi_k3_attn_res.cake_backend import (
            prepare_kimi_k3_attn_res as cake_prepare_kimi_k3_attn_res,
        )

        return cake_prepare_kimi_k3_attn_res(
            prefix,
            delta,
            blocks,
            norm_weight,
            qk_weight,
            output_norm_weight,
            out,
            num_blocks=num_blocks,
            block_write_idx=block_write_idx,
            eps=eps,
            output_norm_eps=output_norm_eps,
            enable_pdl=enable_pdl,
        )
    raise ValueError("the Kimi-K3 AttnRes kernels currently support backend='cake'")


@flashinfer_experimental_api(feature=_FEATURE)
def kimi_k3_attn_res(
    prefix: torch.Tensor,
    delta: Optional[torch.Tensor],
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    output_norm_weight: Optional[torch.Tensor],
    out: torch.Tensor,
    *,
    num_blocks: int,
    block_write_idx: int = -1,
    eps: float = 1e-5,
    output_norm_eps: float = 1e-5,
    enable_pdl: bool = False,
    backend: str = "cake",
) -> torch.Tensor:
    r"""Run one Kimi-K3 AttnRes call (prepare + launch; see :func:`prepare_kimi_k3_attn_res`)."""
    if backend == "cake":
        from .experimental.cake_kimi_k3_attn_res.cake_backend import (
            kimi_k3_attn_res as cake_kimi_k3_attn_res,
        )

        return cake_kimi_k3_attn_res(
            prefix,
            delta,
            blocks,
            norm_weight,
            qk_weight,
            output_norm_weight,
            out,
            num_blocks=num_blocks,
            block_write_idx=block_write_idx,
            eps=eps,
            output_norm_eps=output_norm_eps,
            enable_pdl=enable_pdl,
        )
    raise ValueError("the Kimi-K3 AttnRes kernels currently support backend='cake'")


def generated_program_available(
    device: torch.device,
    M: Optional[int] = None,
    num_blocks: Optional[int] = None,
    **kwargs,
) -> bool:
    """True when this checkout registers the generated AttnRes program for ``device`` (and call)."""
    from .experimental.cake_kimi_k3_attn_res import cake_backend

    return cake_backend.generated_program_available(device, M, num_blocks, **kwargs)
