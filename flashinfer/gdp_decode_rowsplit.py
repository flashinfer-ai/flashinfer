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

Row-split Gated DeltaProduct (GDP) decode / MTP verification for small batch x
heads: a hand-written backend of
:func:`flashinfer.cake_gdp.cake_gated_delta_product_mtp` (``K = 128``,
``V = 64``, FP32 state pool ``[pool_size, HV, 64, 128]``).

The Cake decode programs run one CTA per (batch row, head).  When ``B x HV``
is small, most SMs idle and the call is latency bound.  The 64 state rows of a
head are independent, so this kernel splits them over ``64 / rows_per_cta``
CTAs (``csrc/gdp_decode_rowsplit.cu``, ``include/flashinfer/gdp/decode_rowsplit.cuh``).
Per state row it performs the Cake lane-owns-rows body op for op with the lane
width of the Cake program the same call selects, so its outputs and every state
it stores are bitwise equal to that program's.

:func:`flashinfer.cake_gdp.cake_gated_delta_product_mtp` hands a call to this
kernel only when :func:`route` returns a plan:

- ``FLASHINFER_DISABLE_GDP_DECODE_ROWSPLIT`` is not ``1``;
- the device is in :data:`ROUTED_COMPUTE_CAPABILITIES` (SM103, where the band
  below was measured; SM100 needs its own measurement and SM90 Cake programs
  use another summation order, so neither is routed);
- the call is in the kernel-ready form (BF16 activations with a dense last
  dimension and 8-byte aligned ``q`` / ``k``, FP32 ``A_log`` / ``dt_bias``,
  int32 slot indices, a BF16 or absent ``output``, ``use_qk_l2norm=True``);
  any other call keeps the Cake path, which converts or rejects it;
- ``(num_householder, T, B x HV)`` lies in :data:`ROUTED_BAND`, and the Cake
  program the call would launch is a lane-owns-rows program with the lane
  width the band entry was validated with (a re-exported Cake route that
  changes the program keeps the call on Cake).
"""

from __future__ import annotations

import functools
import os
from typing import Optional

import torch

from .cake_gdp_backend import cake_jit as _cake_plan
from .jit.gdp_decode_rowsplit import gen_gdp_decode_rowsplit_module
from .utils import get_compute_capability

HEAD_SIZE_K = 128
HEAD_SIZE_V = 64
DISABLE_ENV = "FLASHINFER_DISABLE_GDP_DECODE_ROWSPLIT"

# Devices whose calls may be routed: the band below was measured on SM103 (GB300).
ROUTED_COMPUTE_CAPABILITIES: frozenset[tuple[int, int]] = frozenset({(10, 3)})

# (num_householder, T) -> (lanes per state row, largest routed B x HV as a multiple of
# the SM count).  Measured on GB300 (152 SMs) under CUDA graphs for B x HV = 10 .. 2560
# against the Cake programs of the same calls, bitwise equal in every case
# (tests/gdp/test_gdp_decode_rowsplit.py, benchmarks/bench_gdp_decode_rowsplit.py):
#
# - n = 3, T = 2 .. 8 and 16, Cake's static lane-owns-rows programs (8 lanes): the
#   row-split kernel takes 2-44 % less time up to 8 x SM count (cold L2; 8-51 % with
#   the state in L2), and 1-8 % less at 16.8 x, which is not routed.
# - The Cake route's runtime-T program (4 lanes: n = 3 with T = 9 .. 15, and every T
#   for n = 1, 2, 4): 1.4-2.9x faster at every measured size; routed up to 16 x SM
#   count.
# - n = 3, T = 1 is not routed: below 160 state heads Cake runs its own row-partition
#   program (another summation order, and as fast).
_STATIC_BAND = (8, 8.0)
_RUNTIME_BAND = (4, 16.0)
ROUTED_BAND: dict[tuple[int, int], tuple[int, float]] = {
    **{(3, T): _STATIC_BAND for T in (2, 3, 4, 5, 6, 7, 8, 16)},
    **{(3, T): _RUNTIME_BAND for T in range(9, 16)},
    **{(n, T): _RUNTIME_BAND for n in (1, 2, 4) for T in range(1, 17)},
}

# CTA tiles (state rows per CTA, rows per lane) compiled into every module, per lane width.
TILES: dict[int, tuple[tuple[int, int], ...]] = {
    8: ((8, 1), (16, 1), (32, 1), (16, 2), (64, 4)),
    4: ((8, 1), (16, 1), (32, 2), (64, 2)),
}
_STATIC_SMEM_LIMIT = 48 * 1024
_MAX_QK_PIECES_PER_LANE = 40
_QK_ALIGNMENT_ELEMENTS = 4  # q / k are read in 8-byte pieces
_STATE_ALIGNMENT_BYTES = 16  # state rows are read and written in 16-byte pieces


def tile_supported(
    num_householder: int,
    num_tokens: int,
    lanes: int,
    rows_per_cta: int,
    rows_per_lane: int,
) -> bool:
    """Whether a module instantiates the tile (``KernelTraits::kSupported`` of the kernel)."""

    if (rows_per_cta, rows_per_lane) not in TILES.get(lanes, ()):
        return False
    vecs = num_tokens * num_householder
    threads = rows_per_cta // rows_per_lane * lanes
    warps = threads // 32
    smem = 4 * (
        (vecs + num_tokens) * HEAD_SIZE_K + vecs * rows_per_cta + vecs + 2 * num_tokens
    )
    return (
        threads % 32 == 0
        and threads >= num_tokens
        and -(-(vecs + num_tokens) // warps) <= _MAX_QK_PIECES_PER_LANE
        and smem <= _STATIC_SMEM_LIMIT
    )


def default_tile(
    num_householder: int,
    num_tokens: int,
    lanes: int,
    state_heads: int,
    sm_count: int,
) -> tuple[int, int]:
    """The CTA tile for ``state_heads = B x HV`` on a device of ``sm_count`` SMs: four
    CTAs of 16 rows per head while ``B x HV`` is small, then wider CTAs as the grid
    fills the device (the fastest tile of the GB300 sweep at each size, geometric mean
    over ``n`` and ``T``); the next tile in line when a module does not instantiate it."""

    order: tuple[tuple[int, int], ...]
    if lanes == 8:
        order = (
            ((16, 1), (16, 2), (8, 1), (32, 1), (64, 4))
            if state_heads <= 0.4 * sm_count
            else ((16, 2), (64, 4), (16, 1), (32, 1), (8, 1))
            if state_heads <= 6 * sm_count
            else ((64, 4), (16, 2), (16, 1), (32, 1), (8, 1))
        )
    else:
        order = (
            ((16, 1), (8, 1), (32, 2), (64, 2))
            if state_heads <= 1.5 * sm_count
            else ((32, 2), (64, 2), (16, 1), (8, 1))
            if state_heads <= 3 * sm_count
            else ((64, 2), (32, 2), (16, 1), (8, 1))
        )
    for rows_per_cta, rows_per_lane in order:
        if tile_supported(
            num_householder, num_tokens, lanes, rows_per_cta, rows_per_lane
        ):
            return rows_per_cta, rows_per_lane
    raise ValueError(
        f"no row-split tile for num_householder={num_householder}, T={num_tokens}, lanes={lanes}"
    )


@functools.cache
def get_gdp_decode_rowsplit_module(num_householder: int, num_tokens: int, lanes: int):
    return gen_gdp_decode_rowsplit_module(
        num_householder, num_tokens, lanes
    ).build_and_load()


def cake_lanes(
    device: torch.device, num_householder: int, num_tokens: int, state_heads: int
) -> Optional[int]:
    """Lanes per state row of the Cake program a call of ``state_heads = B x HV`` launches
    on ``device`` (``None`` when that program is not a lane-owns-rows program)."""

    arch = _cake_plan.arch_for_capability(get_compute_capability(device))
    if arch is None:
        return None
    return _cake_lanes(
        arch,
        num_householder,
        num_tokens,
        state_heads,
        _cake_plan.sm_count(device.index),
    )


@functools.lru_cache(maxsize=4096)
def _cake_lanes(
    arch: str, num_householder: int, num_tokens: int, state_heads: int, sm_count: int
) -> Optional[int]:
    try:
        name = _cake_plan.selected_module(
            arch, num_householder, num_tokens, state_heads, sm_count=sm_count
        )
    except ValueError:
        return None
    record = _cake_plan.MODULES[name]
    variant = str(record["build"].get("variant", ""))
    if record.get("body") != "lanerow" or variant[:2] not in ("l4", "l8"):
        return None
    return int(variant[1])


@functools.lru_cache(maxsize=4096)
def _plan(
    capability: tuple[int, int],
    num_householder: int,
    num_tokens: int,
    state_heads: int,
    sm_count: int,
) -> Optional[tuple[int, int, int]]:
    if capability not in ROUTED_COMPUTE_CAPABILITIES:
        return None
    band = ROUTED_BAND.get((num_householder, num_tokens))
    if band is None:
        return None
    lanes, sm_multiple = band
    if state_heads > sm_multiple * sm_count:
        return None
    arch = _cake_plan.arch_for_capability(capability)
    if (
        arch is None
        or _cake_lanes(arch, num_householder, num_tokens, state_heads, sm_count)
        != lanes
    ):
        return None
    rows_per_cta, rows_per_lane = default_tile(
        num_householder, num_tokens, lanes, state_heads, sm_count
    )
    return lanes, rows_per_cta, rows_per_lane


def _aligned(tensor: torch.Tensor, elements: int) -> bool:
    """Base address and the strides of the outer dimensions of size > 1 are multiples of
    ``elements`` elements."""

    if tensor.data_ptr() % (elements * tensor.element_size()):
        return False
    return all(
        stride % elements == 0
        for stride, size in zip(tensor.stride()[:-1], tensor.shape[:-1], strict=True)
        if size > 1
    )


def _unsupported_reason(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    output: Optional[torch.Tensor],
    ssm_state_indices: Optional[torch.Tensor],
    disable_state_update: bool,
    use_qk_l2norm: bool,
    output_state_indices: Optional[torch.Tensor],
) -> Optional[str]:
    """``None`` when the call is in the form the row-split kernel reads in place, else why not."""

    if not use_qk_l2norm:
        return "use_qk_l2norm=False"
    if not isinstance(q, torch.Tensor) or q.ndim != 4 or q.device.type != "cuda":
        return "q must be a CUDA [B, T, H, 128] tensor"
    B, T, H, K = q.shape
    if K != HEAD_SIZE_K or B < 1 or H < 1 or not 1 <= T <= 16:
        return f"unsupported q shape {list(q.shape)}"
    if not isinstance(k, torch.Tensor) or k.ndim != 5 or not 1 <= k.shape[2] <= 4:
        return "k must be a [B, T, n, H, 128] tensor with n in 1..4"
    n = k.shape[2]
    device, bf16 = q.device, torch.bfloat16
    shapes = (
        (q, "q", (B, T, H, HEAD_SIZE_K)),
        (k, "k", (B, T, n, H, HEAD_SIZE_K)),
        (v, "v", (B, T, n, H, HEAD_SIZE_V)),
        (a, "a", (B, T, H)),
        (b, "b", (B, T, n, H)),
    )
    for tensor, name, shape in shapes:
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.dtype != bf16
            or tensor.device != device
            or tuple(tensor.shape) != shape
            or tensor.stride(-1) != 1
        ):
            return f"{name} must be a dense-last-dimension bfloat16 {list(shape)} tensor on {device}"
    if not (
        _aligned(q, _QK_ALIGNMENT_ELEMENTS) and _aligned(k, _QK_ALIGNMENT_ELEMENTS)
    ):
        return "q and k must be 8-byte aligned"
    if (
        not isinstance(initial_state, torch.Tensor)
        or initial_state.dtype != torch.float32
        or initial_state.device != device
        or initial_state.ndim != 4
        or initial_state.shape[0] < 1
        or tuple(initial_state.shape[1:]) != (H, HEAD_SIZE_V, HEAD_SIZE_K)
        or initial_state.stride(3) != 1
        or initial_state.stride(2) < HEAD_SIZE_K
        or any(stride % 4 for stride in initial_state.stride()[:3])
        or initial_state.data_ptr() % _STATE_ALIGNMENT_BYTES
    ):
        return "initial_state must be a 16-byte aligned float32 [pool_size, H, 64, 128] pool with dense rows"
    for tensor, name in ((A_log, "A_log"), (dt_bias, "dt_bias")):
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.dtype != torch.float32
            or tensor.device != device
            or tuple(tensor.shape) != (H,)
            or not tensor.is_contiguous()
        ):
            return f"{name} must be a contiguous float32 [H] tensor"
    slot_tensors = [(initial_state_indices, "initial_state_indices")]
    if output_state_indices is not None:
        slot_tensors.append((output_state_indices, "output_state_indices"))
    for tensor, name in slot_tensors:
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.dtype != torch.int32
            or tensor.device != device
            or tuple(tensor.shape) != (B,)
            or not tensor.is_contiguous()
        ):
            return f"{name} must be a contiguous int32 [B] tensor"
    if ssm_state_indices is not None:
        if disable_state_update:
            return "ssm_state_indices requires disable_state_update=False"
        if (
            not isinstance(ssm_state_indices, torch.Tensor)
            or ssm_state_indices.dtype != torch.int32
            or ssm_state_indices.device != device
            or tuple(ssm_state_indices.shape) != (B, T)
            or ssm_state_indices.stride(-1) != 1
        ):
            return "ssm_state_indices must be an int32 [B, T] tensor with a dense last dimension"
    if output is not None and (
        not isinstance(output, torch.Tensor)
        or output.dtype != bf16
        or output.device != device
        or tuple(output.shape) != (B, T, H, HEAD_SIZE_V)
        or output.stride(-1) != 1
        or not _aligned(output, 4)
    ):
        return "output must be an 8-byte aligned bfloat16 [B, T, H, 64] tensor with a dense last dimension"
    return None


def route(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    output: Optional[torch.Tensor] = None,
    ssm_state_indices: Optional[torch.Tensor] = None,
    disable_state_update: Optional[bool] = None,
    use_qk_l2norm: bool = True,
    output_state_indices: Optional[torch.Tensor] = None,
) -> Optional[tuple[int, int, int]]:
    """``(lanes, rows_per_cta, rows_per_lane)`` when
    :func:`~flashinfer.cake_gdp.cake_gated_delta_product_mtp` runs this call on the
    row-split kernel (see the module docstring), else ``None``.  Reads no tensor
    contents."""

    if os.environ.get(DISABLE_ENV, "0") == "1":
        return None
    # The band checks first: they are cached and reject most calls before any tensor check.
    if (
        not isinstance(q, torch.Tensor)
        or not isinstance(k, torch.Tensor)
        or q.ndim != 4
        or k.ndim != 5
        or q.device.type != "cuda"
    ):
        return None
    B, T, H = q.shape[0], q.shape[1], q.shape[2]
    plan = _plan(
        get_compute_capability(q.device),
        k.shape[2],
        T,
        B * H,
        _cake_plan.sm_count(q.device.index),
    )
    if plan is None:
        return None
    if _unsupported_reason(
        q,
        k,
        v,
        initial_state,
        initial_state_indices,
        A_log,
        a,
        dt_bias,
        b,
        output,
        ssm_state_indices,
        bool(disable_state_update),
        bool(use_qk_l2norm),
        output_state_indices,
    ):
        return None
    return plan


def gdp_decode_rowsplit_mtp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    scale: Optional[float] = None,
    output: Optional[torch.Tensor] = None,
    ssm_state_indices: Optional[torch.Tensor] = None,
    disable_state_update: Optional[bool] = None,
    use_qk_l2norm: bool = True,
    output_state_indices: Optional[torch.Tensor] = None,
    *,
    lanes: Optional[int] = None,
    rows_per_cta: Optional[int] = None,
    rows_per_lane: Optional[int] = None,
    _checked: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The row-split kernel with the semantics of
    :func:`~flashinfer.cake_gdp.cake_gated_delta_product_mtp` for calls in the
    kernel-ready form (raises ``ValueError`` for any other call).

    ``lanes`` (4 or 8) fixes the summation order; ``None`` takes the lane width of the
    Cake program the same call selects, which makes the results bitwise equal to that
    program's (``ValueError`` when the call selects no lane-owns-rows program).
    ``rows_per_cta`` / ``rows_per_lane`` pick the CTA tile (:data:`TILES`);
    :func:`default_tile` when ``None``.  Requires an SM10x device; the results are
    claimed bitwise equal to Cake on SM100 / SM103 only.
    """

    disable = bool(disable_state_update)
    if not _checked:
        reason = _unsupported_reason(
            q,
            k,
            v,
            initial_state,
            initial_state_indices,
            A_log,
            a,
            dt_bias,
            b,
            output,
            ssm_state_indices,
            disable,
            bool(use_qk_l2norm),
            output_state_indices,
        )
        if reason is not None:
            raise ValueError(f"gdp_decode_rowsplit_mtp: {reason}")
        if get_compute_capability(q.device)[0] != 10:
            raise ValueError("gdp_decode_rowsplit_mtp requires an SM10x device")
    B, T, H = q.shape[0], q.shape[1], q.shape[2]
    n = k.shape[2]
    if lanes is None:
        lanes = cake_lanes(q.device, n, T, B * H)
        if lanes is None:
            raise ValueError(
                f"the Cake program of num_householder={n}, T={T}, B x HV={B * H} is not a "
                "lane-owns-rows program; pass lanes explicitly"
            )
    if rows_per_cta is None or rows_per_lane is None:
        rows_per_cta, rows_per_lane = default_tile(
            n, T, lanes, B * H, _cake_plan.sm_count(q.device.index)
        )
    if not tile_supported(n, T, lanes, rows_per_cta, rows_per_lane):
        raise ValueError(
            f"unsupported tile rows_per_cta={rows_per_cta}, rows_per_lane={rows_per_lane} "
            f"for num_householder={n}, T={T}, lanes={lanes}"
        )
    out = (
        torch.zeros((B, T, H, HEAD_SIZE_V), dtype=torch.bfloat16, device=q.device)
        if output is None
        else output
    )
    write = (
        initial_state_indices if output_state_indices is None else output_state_indices
    )
    scatter = (
        initial_state_indices.view(B, 1)
        if ssm_state_indices is None
        else ssm_state_indices
    )
    get_gdp_decode_rowsplit_module(n, T, lanes).gdp_decode_rowsplit(
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        initial_state,
        initial_state_indices,
        write,
        scatter,
        out,
        HEAD_SIZE_K**-0.5 if scale is None else float(scale),
        int(ssm_state_indices is not None),
        int(disable),
        rows_per_cta,
        rows_per_lane,
    )
    return out, initial_state


__all__ = [
    "DISABLE_ENV",
    "ROUTED_BAND",
    "ROUTED_COMPUTE_CAPABILITIES",
    "TILES",
    "cake_lanes",
    "default_tile",
    "gdp_decode_rowsplit_mtp",
    "get_gdp_decode_rowsplit_module",
    "route",
    "tile_supported",
]
