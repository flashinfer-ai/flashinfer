# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Torch-facing entry of the bf16 GEMM class: ``mm_bf16_batch_invariant`` is what :func:`flashinfer.mm_bf16` runs
under ``batch_invariant=True``. C = A @ B in bf16 with fp32 accumulation; A is (M, K) row-major, B is (K, N) either
row-major or the transpose of a row-major (N, K) tensor (FlashInfer's documented "column-major" weight), C is (M, N)
row-major bf16. Every unsupported input raises :class:`flashinfer.batch_invariant.BatchInvariantUnsupportedError`;
nothing falls back."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from ...batch_invariant.support import check_device, unsupported
from .generator import Knobs
from .schedule import chunks_for, default_config
from .spec import MMA_K

CLASS = "gemm.bf16"
#: The compute capabilities the class is stated and tested for: SM100 (B200), SM103 (B300), SM107 (Rubin).
MM_BF16_ARCHS = ((10, 0), (10, 3), (10, 7))
_ALIGN = 8  # bf16 elements per 16 bytes
_ALIGN_BYTES = 16  # the operand base alignment the kernels assume (TMA)


def _layout_of_b(b: torch.Tensor, K: int, N: int) -> str:
    if b.stride(1) == 1 and b.stride(0) == N:
        return "n"  # (K, N) row-major
    if b.stride(0) == 1 and b.stride(1) == K:
        return "k"  # the transpose of a row-major (N, K) weight
    raise unsupported(
        CLASS,
        f"B with strides {tuple(b.stride())}",
        "B (K, N) row-major or the transpose of a row-major (N, K) tensor",
    )


def sm_count_of(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def part_of(device: torch.device) -> Tuple[int, str]:
    """The part the dispatcher models for ``device``: its SM count and architecture family (`cost_model.family_of`)."""
    from ...batch_invariant.support import compute_capability
    from .cost_model import family_of

    return sm_count_of(device), family_of(compute_capability(device))


def prepare_operands(
    a: torch.Tensor, b: torch.Tensor, out: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, str, int, int, int]:
    """The class's typed checks and operand normalisation, shared by the direct entry and the autotuner entry: device
    and architecture, dtypes, 2-D shapes with matching K, 16-byte TMA alignment of K and N, A's row placement, B's layout
    (``"n"``: (K, N) row-major; ``"k"``: the transpose of a row-major (N, K) weight), the output's dtype / shape / layout
    (allocated when absent). Returns ``(a, out, b_major, M, N, K)``; raises :class:`BatchInvariantUnsupportedError`."""
    check_device(a.device, CLASS, MM_BF16_ARCHS)
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise unsupported(CLASS, f"dtypes ({a.dtype}, {b.dtype})", "bf16 A and B")
    if a.dim() != 2 or b.dim() != 2:
        raise unsupported(
            CLASS, f"{a.dim()}-D A and {b.dim()}-D B", "2-D operands (M, K) and (K, N)"
        )
    M, K = a.shape
    K2, N = b.shape
    if K != K2:
        raise unsupported(CLASS, f"A (M, {K}) with B ({K2}, N)", "matching K")
    if b.device != a.device:
        raise unsupported(
            CLASS, "operands on different devices", "A and B on one device"
        )
    if K == 0 or N == 0:
        raise unsupported(
            CLASS, f"K = {K}, N = {N}", "K > 0 and N > 0 (only M may be empty)"
        )
    if K % _ALIGN or N % _ALIGN:
        raise unsupported(
            CLASS,
            f"K = {K}, N = {N}",
            f"K and N multiples of {_ALIGN} (16-byte TMA rows)",
        )
    if a.stride(1) != 1 or a.stride(0) % _ALIGN or a.data_ptr() % _ALIGN_BYTES:
        # a placement, not numerics: the kernel reads 16-byte-aligned rows from a 16-byte-aligned base (a single
        # row's row stride is immaterial)
        a = (
            a.as_strided((1, K), (K, 1), a.storage_offset())
            if (M == 1 and a.stride(1) == 1 and a.data_ptr() % _ALIGN_BYTES == 0)
            else a.clone(memory_format=torch.contiguous_format)
        )
    b_major = _layout_of_b(b, K, N)
    if b.data_ptr() % _ALIGN_BYTES:
        raise unsupported(
            CLASS,
            "B at a base address that is not 16-byte aligned",
            "a 16-byte-aligned B",
        )
    if out is None:
        out = torch.empty(M, N, dtype=torch.bfloat16, device=a.device)
    else:
        if out.dtype != torch.bfloat16 or tuple(out.shape) != (M, N):
            raise unsupported(
                CLASS,
                f"out {tuple(out.shape)} {out.dtype}",
                f"a bf16 (M, N) = ({M}, {N}) output",
            )
        if out.device != a.device:
            raise unsupported(
                CLASS, "out on another device", "the output on A's device"
            )
        if out.data_ptr() % _ALIGN_BYTES:
            raise unsupported(
                CLASS,
                "out at a base address that is not 16-byte aligned",
                "a 16-byte-aligned output",
            )
        if out.stride(1) != 1 or (M > 1 and out.stride(0) != N):
            raise unsupported(
                CLASS,
                f"out with strides {tuple(out.stride())}",
                "a row-major contiguous output",
            )
        if M == 1 and out.stride(0) != N:
            out = out.as_strided(
                (1, N), (N, 1), out.storage_offset()
            )  # the same elements, an aligned row stride
    return a, out, b_major, M, N, K


def mm_bf16_batch_invariant(
    a: torch.Tensor,
    b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    config: Optional[Knobs] = None,
    chunks: Optional[int] = None,
    workspace: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``C = A @ B`` by the class ``gemm.bf16``: ``kfold`` when the weight shape's chunk rule gives 1, else
    ``kchunk[C]`` with ``C = chunks_for(N, K)``, a function of the weight shape only, so a row's bits never depend on
    the batch. ``config`` forces a configuration (a member of the enumerated space: the bits never depend on it);
    ``chunks`` forces the class's chunk count (tests); ``workspace`` (fp32, at least ``chunks * M * N`` elements) is the
    split class's partials buffer, otherwise allocated per call from the caching allocator (stream-ordered)."""
    a, out, b_major, M, N, K = prepare_operands(a, b, out)
    if M == 0:
        return out  # no rows: nothing to launch
    C = chunks if chunks is not None else chunks_for(N, K)
    if C > 1 and K % (MMA_K * C):
        raise unsupported(
            CLASS,
            f"chunks = {C} at K = {K}",
            "each K chunk a whole number of 16-wide MMA steps",
        )
    if config is None:
        config = default_config(M, N, K, C, b_major, *part_of(a.device))
    from .blocks.compile import get_compiled_gemm
    from .fixup import MAX_GRID_Y, get_fixup

    kernel = get_compiled_gemm(config, b_major, C, sm_count_of(a.device))
    # the kernel sees 3-D (m, k, l) / (n, k, l) / (m, n, l) tensors: l = the K chunk (1 for the unsplit class and the in-CTA
    # fold's output); every stride handed over is a multiple of the declared 16-byte alignment, the size-1 l stride included
    Kc = K // C
    a3 = a.as_strided((M, Kc, C), (a.stride(0), 1, Kc))
    bt = b.t()  # (N, K) view with strides (1, N) or (K, 1)
    b3 = bt.as_strided((N, Kc, C), (bt.stride(0), bt.stride(1), Kc * bt.stride(1)))
    if C == 1 or config.cta_chunks:
        kernel(a3, b3, out.as_strided((M, N, 1), (out.stride(0), 1, N)))
        return out
    # the fixed-chunk split with the partials in an fp32 workspace, folded in chunk order by the fixup kernel
    if workspace is None:
        ws = torch.empty(C, M, N, dtype=torch.float32, device=a.device)
    else:
        if (
            workspace.dtype != torch.float32
            or workspace.device != a.device
            or workspace.numel() < C * M * N
            or not workspace.is_contiguous()
            or workspace.data_ptr() % 16
        ):
            raise unsupported(
                CLASS,
                "the given workspace",
                f"a contiguous, 16-byte-aligned fp32 tensor on A's device of at least {C * M * N} elements",
            )
        ws = workspace.reshape(-1)[: C * M * N].view(C, M, N)
    ws = ws.permute(1, 2, 0)  # chunk-major storage (C, M, N) viewed as (M, N, C)
    kernel(a3, b3, ws)
    get_fixup(C, "bf16")(ws, out, M, N, min(M, MAX_GRID_Y))
    return out
