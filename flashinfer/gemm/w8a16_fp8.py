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

import functools
from typing import Optional

import torch

from ..api_logging import flashinfer_api
from ..jit.gemm.w8a16_fp8 import (
    gen_w8a16_fp8_gemm_module,
    gen_w8a16_fp8_gemm_sm100_module,
)
from ..utils import get_compute_capability, register_custom_op, register_fake_op


@functools.cache
def _get_module_for_major(major: int):
    if major == 9:
        return gen_w8a16_fp8_gemm_module().build_and_load()
    if major == 10:
        return gen_w8a16_fp8_gemm_sm100_module().build_and_load()
    raise RuntimeError(
        f"w8a16_fp8_gemm needs Hopper (sm_90a) or Blackwell (sm_100a), got SM {major}x"
    )


def get_w8a16_fp8_gemm_module(device: Optional[torch.device] = None):
    """Kernel module for ``device`` (current device when omitted): Hopper or Blackwell build."""
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return _get_module_for_major(get_compute_capability(device)[0])


@register_custom_op("flashinfer::w8a16_fp8_gemm", mutates_args=("out",))
def _w8a16_fp8_gemm_impl(
    out: torch.Tensor,
    act: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    workspace: Optional[torch.Tensor],
    algo: int,
) -> None:
    get_w8a16_fp8_gemm_module(act.device).w8a16_fp8_gemm(
        out, act, weight, weight_scale, bias, workspace, algo
    )


@register_fake_op("flashinfer::w8a16_fp8_gemm")
def _w8a16_fp8_gemm_impl_fake(
    out: torch.Tensor,
    act: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    workspace: Optional[torch.Tensor],
    algo: int,
) -> None:
    pass


@functools.cache
def _workspace_bytes(m: int, n: int, k: int, device_index: int, algo: int) -> int:
    """Bytes of workspace this problem needs on this device (0 = none)."""
    return int(
        get_w8a16_fp8_gemm_module(
            torch.device("cuda", device_index)
        ).w8a16_fp8_gemm_workspace_bytes(m, n, k, device_index, algo)
    )


# Blackwell kernel variants selectable through ``algo``.  The codes are the
# W8A16Sm100Tile enum values in include/flashinfer/gemm/w8a16_fp8_gemm_sm100.cuh.
W8A16_FP8_GEMM_ALGOS_SM100: dict[str, int] = {
    "auto": 0,
    "2sm256x128": 1,
    "1sm128x128": 2,
    "2sm256x256": 3,
    "2sm256x192": 4,
    "2sm256x128_splitk": 5,
    "2sm256x128_streamk": 6,
    "2sm256x160": 7,
}


# Hopper kernel variants selectable through ``algo``: the variants on CUTLASS's mixed-input
# mainloop (weight rows x activation rows per CTA tile, plus the stream-K and split-K
# decompositions).  The codes are the W8A16Sm90Algo enum values in
# include/flashinfer/gemm/w8a16_fp8_gemm_sm90_cutlass.cuh; code 1 was the retired
# hand-written kernel and stays reserved.
W8A16_FP8_GEMM_ALGOS_SM90: dict[str, int] = {
    "auto": 0,
    "cutlass_128x256": 2,
    "cutlass_128x192": 3,
    "cutlass_128x160": 4,
    "cutlass_128x128": 5,
    "cutlass_64x256": 6,
    "cutlass_128x256_streamk": 7,
    "cutlass_128x192_streamk": 8,
    "cutlass_64x128": 9,
    "cutlass_128x128_splitk": 10,
    "cutlass_128x256_splitk": 11,
}


def _algo_code(algo: Optional[str], device: torch.device) -> int:
    if algo is None or algo == "auto":
        return 0
    major = get_compute_capability(device)[0]
    if major == 10:
        table = W8A16_FP8_GEMM_ALGOS_SM100
    elif major == 9:
        table = W8A16_FP8_GEMM_ALGOS_SM90
    else:
        raise ValueError("algo is only selectable on Hopper / Blackwell; use algo=None")
    try:
        return table[algo]
    except KeyError:
        raise ValueError(f"unknown algo {algo!r}; one of {sorted(table)}") from None


@flashinfer_api
def w8a16_fp8_gemm(
    act: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    algo: Optional[str] = None,
) -> torch.Tensor:
    r"""bf16 / fp16 GEMM whose ``[N, K]`` operand is stored as FP8 e4m3 (W8A16).

    The math is a plain 16-bit GEMM: the tensor cores run bf16 x bf16 or fp16 x
    fp16 with fp32 accumulation, and the result is in ``act``'s dtype.  Nothing
    e4m3 reaches an MMA instruction.  "fp8" in the name says how the weight is
    *stored* (e4m3 plus one per-tensor scale, as opposed to the int8 of a plain
    W8A16 scheme); the kernel converts it to the activation dtype on the way from
    memory to the MMA.  Same reading as vLLM's ``CompressedTensorsW8A16Fp8``.

    ``out = act @ (weight.float() * weight_scale).T + bias``, computed with the
    weight dequantized inside the kernel; no 16-bit copy of the weight is
    materialised.  This is the fused form of TRT-LLM's
    ``W8A16FP8LinearMethod`` (``weight.to(dtype).mul_(scale)`` then
    ``F.linear``) and consumes the same checkpoint tensors: an FP8QDQ weight
    with one per-tensor scale.

    Parameters
    ----------
    act: torch.Tensor
        Activation, shape ``(..., K)``, ``float16`` or ``bfloat16``,
        contiguous in the last dimension.  Leading dimensions are flattened.
    weight: torch.Tensor
        Weight, shape ``(N, K)`` (``torch.nn.Linear`` layout),
        ``float8_e4m3fn``, contiguous.
    weight_scale: torch.Tensor
        One ``float32`` value (shape ``()`` or ``(1,)``) on ``act``'s device:
        the per-tensor dequantization scale.
    bias: Optional[torch.Tensor]
        Optional bias, shape ``(N,)``, same dtype as ``act``.
    out: Optional[torch.Tensor]
        Optional output, shape ``(..., N)``, same dtype as ``act``.  Allocated
        when omitted.

    Notes
    -----
    Hopper (``sm_90a``: wgmma + TMA) and Blackwell (``sm_100a``: tcgen05 + TMA,
    a FlashInfer fork of CUTLASS's mixed-input mainloop).  ``K`` must be a
    multiple of 16 and ``N`` a multiple of 8 (16-byte TMA rows), and ``act`` /
    ``weight`` / ``out`` 16-byte aligned; anything else is rejected with a
    ValueError before launch.  On Hopper the launcher picks a CUTLASS tile and
    decomposition (plain, stream-K or split-K) from a cost model; split-K runs
    go through an fp32 workspace that this function allocates.  On Blackwell it
    picks one of the 2SM tiles from the same kind of model.
    algo: Optional[str]
        Pins one kernel variant instead of the launcher's choice (benchmarking):
        a key of ``W8A16_FP8_GEMM_ALGOS_SM100`` on Blackwell or of
        ``W8A16_FP8_GEMM_ALGOS_SM90`` on Hopper
        (``cutlass_<weight rows>x<activation rows>[_streamk|_splitk]``).
        ``None`` / ``"auto"`` = the launcher decides.

    Returns
    -------
    out: torch.Tensor
        ``(..., N)`` in ``act``'s dtype.
    """
    if act.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"act must be float16 or bfloat16, got {act.dtype}")
    if weight.dtype != torch.float8_e4m3fn:
        raise ValueError(f"weight must be float8_e4m3fn, got {weight.dtype}")
    if weight.dim() != 2:
        raise ValueError(f"weight must be 2D [N, K], got shape {tuple(weight.shape)}")
    k = act.shape[-1]
    n = weight.shape[0]
    if weight.shape[1] != k:
        raise ValueError(f"K mismatch: act[..., {k}] vs weight[{n}, {weight.shape[1]}]")
    if weight_scale.numel() != 1:
        raise ValueError("weight_scale is one per-tensor value")
    if k % 16 != 0:
        raise ValueError(
            f"K must be a multiple of 16 (16-byte TMA rows of fp8), got K={k}"
        )
    if n % 8 != 0:
        raise ValueError(f"N must be a multiple of 8 (16-byte output rows), got N={n}")
    if weight_scale.dtype != torch.float32:
        weight_scale = weight_scale.to(torch.float32)
    weight_scale = weight_scale.reshape(1)
    algo_code = _algo_code(algo, act.device)

    lead_shape = act.shape[:-1]
    act2d = act.reshape(-1, k)
    if out is None:
        out = torch.empty(*lead_shape, n, dtype=act.dtype, device=act.device)
    elif out.dtype != act.dtype or out.shape[-1] != n:
        raise ValueError(
            f"out must be {tuple(lead_shape) + (n,)} in {act.dtype}, "
            f"got {tuple(out.shape)} in {out.dtype}"
        )
    out2d = out.reshape(-1, n)
    # Under-filled grids run split-K through an fp32 workspace; the kernel
    # decides, this only sizes the buffer (0 bytes = single pass).
    ws_bytes = _workspace_bytes(act2d.shape[0], n, k, act.device.index or 0, algo_code)
    workspace = (
        torch.empty(ws_bytes // 4, dtype=torch.float32, device=act.device)
        if ws_bytes > 0
        else None
    )
    _w8a16_fp8_gemm_impl(out2d, act2d, weight, weight_scale, bias, workspace, algo_code)
    return out
