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
def _get_module(major: int, act_dtype: torch.dtype):
    if major == 9:
        return gen_w8a16_fp8_gemm_module(act_dtype).build_and_load()
    if major == 10:
        return gen_w8a16_fp8_gemm_sm100_module(act_dtype).build_and_load()
    raise RuntimeError(
        f"w8a16_fp8_gemm needs Hopper (sm_90a) or Blackwell (sm_100a), got SM {major}x"
    )


def _resolve_device(device: Optional[torch.device]) -> torch.device:
    """``None`` and an index-less ``torch.device("cuda")`` both mean the current
    CUDA device; the result always carries an explicit index so the module
    choice, the algo table and the workspace cache key agree on one GPU."""
    if device is None:
        return torch.device("cuda", torch.cuda.current_device())
    if device.type != "cuda":
        raise ValueError(f"w8a16_fp8_gemm runs on CUDA devices only, got {device}")
    if device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


# TMA descriptors and the 16-byte vector loads / stores of the epilogues and the
# split-K reduce need 16-byte aligned base addresses.  A contiguous tensor does not
# guarantee that (a storage-offset view of a bigger buffer is contiguous too), and a
# misaligned pointer poisons the CUDA context instead of failing the call.
_ALIGN = 16


def _check_aligned(t: torch.Tensor, name: str) -> None:
    if t.data_ptr() % _ALIGN:
        raise ValueError(
            f"{name} must start at a {_ALIGN}-byte aligned address "
            f"(storage offset {t.storage_offset()} elements); pass a view that "
            "starts on a 16-byte boundary or a fresh tensor"
        )


def get_w8a16_fp8_gemm_module(
    device: Optional[torch.device] = None, act_dtype: torch.dtype = torch.bfloat16
):
    """Kernel module for ``device`` (current device when omitted) and activation dtype.

    One JIT module per (architecture, activation dtype): Hopper or Blackwell build,
    fp16 or bf16.  The first call for a pair compiles it; the other dtype is never
    compiled unless used."""
    device = _resolve_device(device)
    return _get_module(get_compute_capability(device)[0], act_dtype)


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
    get_w8a16_fp8_gemm_module(act.device, act.dtype).w8a16_fp8_gemm(
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
def _workspace_bytes(
    m: int, n: int, k: int, device_index: int, algo: int, act_dtype: torch.dtype
) -> int:
    """Bytes of workspace this problem needs on this device (0 = none).

    The size does not depend on the activation dtype; ``act_dtype`` only picks
    which module answers, so the GEMM call never loads a second one."""
    return int(
        get_w8a16_fp8_gemm_module(
            torch.device("cuda", device_index), act_dtype
        ).w8a16_fp8_gemm_workspace_bytes(m, n, k, device_index, algo)
    )


def w8a16_fp8_gemm_workspace_bytes(
    m: int,
    n: int,
    k: int,
    device: Optional[torch.device] = None,
    algo: Optional[str] = None,
    act_dtype: torch.dtype = torch.bfloat16,
) -> int:
    """Workspace bytes :func:`w8a16_fp8_gemm` needs for an ``[M, K] x [N, K]`` problem.

    ``0`` when the launcher's choice for this shape runs in a single pass;
    otherwise the fp32 partials plus CUTLASS scratch of the split-K (or
    stream-K) decomposition it would pick.  ``m`` is the flattened row count
    (the product of ``act``'s leading dimensions).  The answer depends on the
    device (SM count) and on ``algo`` exactly as the launch does, so size the
    buffer with the arguments you will call with; a caller serving several
    shapes from one buffer takes the maximum over them.  The size is the same
    for fp16 and bf16; ``act_dtype`` only selects which JIT module answers
    (pass the dtype you will call with to avoid compiling the other one).
    ``m == 0`` needs no workspace (the call returns the empty output without
    launching); ``device`` without an index means the current CUDA device.
    """
    device = _resolve_device(device)
    _check_dims(m, n, k)
    algo_code = _algo_code(algo, device)
    if m == 0:
        return 0
    return _workspace_bytes(m, n, k, device.index, algo_code, act_dtype)


_INT32_MAX = 2**31 - 1


def _check_dims(m: int, n: int, k: int) -> None:
    """Shape rules shared by the call and the workspace query (``m == 0`` is legal
    and handled by the caller; ``n`` / ``k`` of zero are not: the tile schedulers
    divide by the tile counts)."""
    if m < 0 or n <= 0 or k <= 0:
        raise ValueError(f"need M >= 0, N > 0, K > 0, got M={m} N={n} K={k}")
    if max(m, n, k) > _INT32_MAX:
        # The kernels index with 32-bit ints; the launchers check the same bound.
        raise ValueError(f"M, N, K must be <= {_INT32_MAX}, got M={m} N={n} K={k}")
    if k % 16 != 0:
        raise ValueError(
            f"K must be a multiple of 16 (16-byte TMA rows of fp8), got K={k}"
        )
    if n % 8 != 0:
        raise ValueError(f"N must be a multiple of 8 (16-byte output rows), got N={n}")


# Blackwell kernel variants selectable through ``algo``.  The codes are the
# W8A16Sm100Tile enum values in include/flashinfer/gemm/w8a16_sm100/dispatch.hpp.
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
# include/flashinfer/gemm/w8a16_sm90/dispatch.hpp; code 1 was the retired
# hand-written kernel and stays reserved.
W8A16_FP8_GEMM_ALGOS_SM90: dict[str, int] = {
    "auto": 0,
    "128x256": 2,
    "128x192": 3,
    "128x160": 4,
    "128x128": 5,
    "64x256": 6,
    "128x256_streamk": 7,
    "128x192_streamk": 8,
    "64x128": 9,
    "128x128_splitk": 10,
    "128x256_splitk": 11,
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
    workspace: Optional[torch.Tensor] = None,
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
    materialised.  It consumes the same checkpoint tensors as TRT-LLM's
    ``W8A16FP8LinearMethod`` (an FP8QDQ weight with one per-tensor scale) and
    replaces its ``weight.to(dtype).mul_(scale)`` followed by ``F.linear``.

    Numerics.  Each e4m3 weight value is converted to ``act``'s dtype exactly
    (every finite e4m3 value is representable in both bf16 and fp16, denormals
    included), the tensor cores accumulate the unscaled products in fp32, and
    ``weight_scale`` (fp32) and ``bias`` are applied to the fp32 accumulator
    before the single rounding to the output dtype.  The TRT-LLM path instead
    rounds ``weight * scale`` to the activation dtype first, so the two are
    the same mathematical operation but not bitwise equal; this one carries
    one rounding fewer.  The weight must be finite: the two e4m3fn NaN
    encodings (``0x7F``, ``0xFF``) are not checked for (that would cost a
    device synchronisation per call) and produce an unspecified finite value
    or NaN depending on the architecture and dtype.

    Parameters
    ----------
    act: torch.Tensor
        Activation, shape ``(..., K)``, ``float16`` or ``bfloat16``,
        contiguous.  Leading dimensions are flattened; a leading dimension of
        0 returns the empty output without launching.
    weight: torch.Tensor
        Weight, shape ``(N, K)`` (``torch.nn.Linear`` layout),
        ``float8_e4m3fn``, contiguous.
    weight_scale: torch.Tensor
        One ``float32`` value (shape ``()`` or ``(1,)``) on ``act``'s device:
        the per-tensor dequantization scale.
    bias: Optional[torch.Tensor]
        Optional bias, shape ``(N,)``, same dtype as ``act``.
    out: Optional[torch.Tensor]
        Optional output, shape ``(..., N)`` (``act``'s leading dimensions),
        same dtype as ``act``, contiguous.  Allocated when omitted.
    workspace: Optional[torch.Tensor]
        Optional scratch for the split-K / stream-K decompositions, on
        ``act``'s device, contiguous, 16-byte aligned, any dtype; at least
        :func:`w8a16_fp8_gemm_workspace_bytes` bytes for this call's shape and
        ``algo``.  Allocated per call when omitted (through the caching
        allocator, so the cost is a few microseconds of host time).  A caller
        that wants a fixed memory budget sizes one buffer once and passes it to
        every call.
    algo: Optional[str]
        Pins one kernel variant instead of the launcher's choice: a key of
        ``W8A16_FP8_GEMM_ALGOS_SM100`` on Blackwell or of
        ``W8A16_FP8_GEMM_ALGOS_SM90`` on Hopper
        (``<weight rows>x<activation rows>[_streamk|_splitk]``).
        ``None`` / ``"auto"`` = the launcher decides.  This is the only
        override; there are no environment variables.

    Notes
    -----
    Hopper (``sm_90a``: wgmma + TMA) and Blackwell (``sm_100a``: tcgen05 + TMA,
    a FlashInfer fork of CUTLASS's mixed-input mainloop); JIT-compiled on
    first use, not part of the AOT packages.  One module per architecture and
    activation dtype (the dtype is rendered through a Jinja config header), so
    the first bf16 call compiles only the bf16 variants and the first fp16 call
    only the fp16 ones.  ``K`` must be a multiple of 16
    and ``N`` a multiple of 8 (16-byte TMA rows), both non-zero, and every
    tensor (``act`` / ``weight`` / ``bias`` / ``out`` / ``workspace``) must
    start at a 16-byte aligned address (a view into a larger buffer at an odd
    storage offset does not; contiguity alone is not enough); anything else is
    rejected with a ValueError before launch, and the C++ launcher checks the
    same rules again.  Nothing is copied on the caller's behalf: a
    non-contiguous ``act`` or ``out`` is rejected (call ``.contiguous()``
    yourself if a copy is acceptable).  On Hopper the launcher picks a CUTLASS tile and
    decomposition (plain, stream-K or split-K) from a cost model; split-K runs
    go through an fp32 workspace (``workspace``, or one this function
    allocates).  On Blackwell it picks one of the 2SM tiles from the same kind
    of model.

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
    lead_shape = act.shape[:-1]
    m = act.numel() // k if k else 0
    _check_dims(m, n, k)
    if weight_scale.dtype != torch.float32:
        # No implicit conversions either: the caller casts if they want that.
        raise ValueError(f"weight_scale must be float32, got {weight_scale.dtype}")
    weight_scale = weight_scale.reshape(1)  # () or (1,): a view either way
    device = _resolve_device(act.device)
    algo_code = _algo_code(algo, device)

    out_shape = tuple(lead_shape) + (n,)
    if out is None:
        out = torch.empty(out_shape, dtype=act.dtype, device=act.device)
    elif out.dtype != act.dtype or tuple(out.shape) != out_shape:
        raise ValueError(
            f"out must be {out_shape} in {act.dtype}, got {tuple(out.shape)} in {out.dtype}"
        )
    elif not out.is_contiguous():
        # reshape() would hand the kernel a contiguous copy and the caller's
        # buffer would never see the result.
        raise ValueError("out must be contiguous")
    if m == 0:
        # Nothing to compute; the tile schedulers would divide by zero tiles.
        return out
    # No implicit copies: a non-contiguous act (row slice, transposed last dim,
    # permuted leading dims) is the caller's to .contiguous() if they want that.
    if not act.is_contiguous():
        raise ValueError(
            "act must be contiguous (this function does not copy; pass "
            "act.contiguous() if a copy is acceptable)"
        )
    act2d = act.view(-1, k)
    out2d = out.view(-1, n)
    _check_aligned(act2d, "act")
    _check_aligned(weight, "weight")
    _check_aligned(out2d, "out")
    if bias is not None:
        _check_aligned(bias, "bias")
    # Under-filled grids run split-K through an fp32 workspace; the kernel
    # decides, this only sizes the buffer (0 bytes = single pass).
    ws_bytes = _workspace_bytes(m, n, k, device.index, algo_code, act.dtype)
    if ws_bytes == 0:
        ws = None
    elif workspace is None:
        ws = torch.empty(ws_bytes // 4, dtype=torch.float32, device=act.device)
    else:
        have = workspace.numel() * workspace.element_size()
        if workspace.device != act.device or not workspace.is_contiguous():
            raise ValueError("workspace must be a contiguous tensor on act's device")
        _check_aligned(workspace, "workspace")
        if have < ws_bytes:
            raise ValueError(
                f"workspace too small: {have} bytes, this call needs {ws_bytes} "
                "(see w8a16_fp8_gemm_workspace_bytes)"
            )
        # The launchers take fp32; any dtype is accepted here and reinterpreted
        # (a whole number of 4-byte words, then exactly the bytes this call needs).
        ws = workspace.view(-1)
        if ws.dtype != torch.float32:
            words = have // 4
            ws = ws[: words * 4 // ws.element_size()].view(torch.float32)
        ws = ws[: ws_bytes // 4]
    _w8a16_fp8_gemm_impl(out2d, act2d, weight, weight_scale, bias, ws, algo_code)
    return out
