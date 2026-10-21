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


def _byte_span(t: torch.Tensor) -> Optional[tuple]:
    """``[begin, end)`` of the bytes a contiguous tensor covers; None when empty."""
    if t.numel() == 0:
        return None
    begin = t.data_ptr()
    return begin, begin + t.numel() * t.element_size()


def _check_disjoint(
    written_name: str, written: torch.Tensor, read_name: str, read: torch.Tensor
) -> None:
    """``written`` is overwritten while ``read`` is still being read, so the two
    must not share a byte (checked on the contiguous spans; nothing here is a
    strided view)."""
    w, r = _byte_span(written), _byte_span(read)
    if w is not None and r is not None and w[0] < r[1] and r[0] < w[1]:
        raise ValueError(
            f"{written_name} must not overlap {read_name} in memory ({written_name} "
            f"is written while {read_name} is still being read)"
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

    Parameters
    ----------
    m, n, k: int
        The problem: ``m`` is the flattened row count of ``act`` (the product
        of its leading dimensions), ``n`` the weight's row count, ``k`` the
        reduction depth.  Same bounds as the call: ``m >= 0``, ``n`` and ``k``
        positive, ``k`` a multiple of 16, ``n`` a multiple of 8, all at most
        ``2**31 - 1``.
    device: Optional[torch.device]
        The CUDA device the call will run on (current device when omitted or
        given without an index).  The answer depends on it.
    algo: Optional[str]
        The ``algo`` the call will be made with (``None`` / ``"auto"`` = the
        launcher's choice).  The answer depends on it exactly as the launch
        does, so query with the arguments you will call with; a caller serving
        several shapes from one buffer takes the maximum over them.
    act_dtype: torch.dtype
        The activation dtype the call will use.  The size is the same for
        fp16 and bf16; passing the dtype you will call with avoids compiling
        the other dtype's module.

    Returns
    -------
    int
        ``0`` when the call needs no workspace (every ``m == 0`` call, and
        every shape whose variant runs in a single pass), otherwise the byte
        count a ``workspace`` passed to :func:`w8a16_fp8_gemm` must reach.

    Raises
    ------
    ValueError
        A shape outside the bounds above, an unknown ``algo``, or a non-CUDA
        device.
    RuntimeError
        Raised by the native launcher (for ``m > 0``): a split-K ``algo`` on a
        ``k`` that cannot be sliced, or a byte count that leaves the signed
        64-bit range.
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
# W8A16Sm100Algo enum values in include/flashinfer/gemm/w8a16_sm100/dispatch.hpp.
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


# Hopper kernel variants selectable through ``algo`` (weight rows x activation rows per
# CTA tile, plus the stream-K and split-K decompositions).  The codes are the
# W8A16Sm90Algo enum values in include/flashinfer/gemm/w8a16_sm90/dispatch.hpp; code 1
# is reserved.
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

    ``out = act @ (weight.float() * weight_scale).T + bias`` with the weight
    dequantized inside the kernel; no 16-bit copy of the weight is allocated.
    "fp8" says how the weight is *stored* (e4m3 plus one per-tensor fp32
    scale); the arithmetic is a 16-bit GEMM in ``act``'s dtype.

    Numerics.  Each e4m3 weight value is converted to ``act``'s dtype exactly
    (every finite e4m3 value is representable in both bf16 and fp16, denormals
    included), the unscaled products are accumulated in fp32, and
    ``weight_scale`` and ``bias`` are applied to the fp32 accumulator before
    the single rounding to the output dtype.  The result is therefore not
    bitwise equal to a GEMM against a weight pre-scaled and rounded to the
    activation dtype (``(weight.to(dtype) * scale) @ ...``): that carries one
    rounding more.  The weight must be finite.  Values are not checked, and a
    NaN e4m3 encoding yields an unspecified result (a finite value or NaN).

    Parameters
    ----------
    act: torch.Tensor
        Activation, shape ``(..., K)``, ``float16`` or ``bfloat16``,
        contiguous, on a CUDA device (Hopper or Blackwell; every other tensor
        must be on the same device).  Leading dimensions are flattened into
        ``M``; a leading dimension of 0 returns the empty output without
        launching.
    weight: torch.Tensor
        Weight, shape ``(N, K)`` (``torch.nn.Linear`` layout),
        ``float8_e4m3fn``, contiguous, on ``act``'s device.
    weight_scale: torch.Tensor
        One ``float32`` value (shape ``()`` or ``(1,)``) on ``act``'s device:
        the per-tensor dequantization scale.
    bias: Optional[torch.Tensor]
        Optional bias, shape ``(N,)``, same dtype as ``act``, on ``act``'s
        device, contiguous.
    out: Optional[torch.Tensor]
        Optional output, shape ``(..., N)`` (``act``'s leading dimensions),
        same dtype as ``act``, on ``act``'s device, contiguous.  Allocated when
        omitted.
    workspace: Optional[torch.Tensor]
        Optional scratch buffer: contiguous, 16-byte aligned, of any dtype, on
        ``act``'s device, at least :func:`w8a16_fp8_gemm_workspace_bytes` bytes
        for this call's shape, device and ``algo`` (its byte size counts, not
        its dtype).  Allocated per call when omitted and needed.  A caller
        that wants a fixed memory budget sizes one buffer once and passes it to
        every call, under these rules: the call overwrites the buffer (nothing
        in it survives), it must not share memory with any other argument (the
        aliasing rule under Notes), and two calls may share one buffer only if
        they are ordered on one stream or synchronised, since a call still
        running on another stream would be reading what the next call
        overwrites.
    algo: Optional[str]
        Pins one kernel variant instead of the launcher's choice: a key of
        ``W8A16_FP8_GEMM_ALGOS_SM100`` on Blackwell or of
        ``W8A16_FP8_GEMM_ALGOS_SM90`` on Hopper
        (``<weight rows>x<activation rows>[_streamk|_splitk]``).
        ``None`` / ``"auto"`` = the launcher decides.  This is the only
        override; there are no environment variables.  A ``_splitk`` variant
        is rejected with a RuntimeError on a ``K`` it cannot slice (Hopper:
        ``K`` a multiple of 64 whose 64-deep tile count divides into 2..8
        equal slices of at least 8 tiles; Blackwell: ``K`` a multiple of 128
        whose 128-deep tile count has a divisor in 2..8).

    Notes
    -----
    Runs on Hopper (``sm_90a``) and Blackwell (``sm_100a``).  The kernels are
    JIT-compiled on first use (one module per architecture and activation
    dtype, so the first bf16 call compiles only the bf16 module and the first
    fp16 call only the fp16 one) and are not part of the AOT packages.

    Shapes and layout: ``K`` must be a multiple of 16 and ``N`` a multiple of
    8, both non-zero, and ``M`` (the product of ``act``'s leading dimensions,
    not each of them), ``N`` and ``K`` each at most ``2**31 - 1``.  Every tensor
    among ``act`` / ``weight`` / ``bias`` / ``out`` / ``workspace`` must start
    at a 16-byte aligned address (a view into a larger buffer at an odd
    storage offset does not; contiguity alone is not enough).  Nothing is
    copied on the caller's behalf: a non-contiguous ``act``, ``weight``,
    ``bias``, ``out`` or ``workspace`` is rejected (call ``.contiguous()``
    yourself if a copy is acceptable).  Anything else is rejected with a
    ValueError before launch.

    Aliasing: the two buffers the call writes, ``out`` and ``workspace``,
    must not share any bytes with ``act`` / ``weight`` / ``weight_scale`` /
    ``bias`` or with each other; a call whose buffers overlap is rejected with
    a ValueError.  The inputs may alias one another freely (they are only
    read).

    Empty calls: with ``M == 0`` the dtypes, shapes, devices, contiguity and
    aliasing of every argument and the alignment of ``weight``, ``bias`` and
    ``workspace`` are checked as for a non-empty call (the empty ``act`` and
    ``out`` have no address to align), then the empty output is returned
    without launching.  No kernel variant is resolved and no workspace is
    sized or checked for such a call, so a split-K ``algo`` on an unsliceable
    ``K`` or an undersized ``workspace`` is not reported there.

    Returns
    -------
    out: torch.Tensor
        ``(..., N)`` in ``act``'s dtype.

    Raises
    ------
    ValueError
        Any argument outside the contract above (dtype, shape, device,
        contiguity, alignment, aliasing, an unknown ``algo``, a ``workspace``
        shorter than the query's answer), before anything is launched.
    RuntimeError
        Raised by the native launcher, also before launch: a split-K ``algo``
        on a ``K`` it cannot slice, or a workspace byte count that leaves the
        signed 64-bit range.  A launch failure reported by the CUDA runtime
        surfaces the same way.
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
    if weight.device != act.device or weight_scale.device != act.device:
        raise ValueError(
            f"weight and weight_scale must be on act's device {act.device}, got "
            f"{weight.device} and {weight_scale.device}"
        )
    algo_code = _algo_code(algo, device)
    # No implicit copies: a non-contiguous act (row slice, transposed last dim,
    # permuted leading dims) is the caller's to .contiguous() if they want that.
    if not act.is_contiguous():
        raise ValueError(
            "act must be contiguous (this function does not copy; pass "
            "act.contiguous() if a copy is acceptable)"
        )
    if bias is not None:
        if bias.dtype != act.dtype or tuple(bias.shape) != (n,):
            raise ValueError(
                f"bias must be ({n},) in {act.dtype}, got {tuple(bias.shape)} in {bias.dtype}"
            )
        if bias.device != act.device:
            raise ValueError(f"bias must be on act's device, got {bias.device}")
        if not bias.is_contiguous():
            raise ValueError("bias must be contiguous")

    out_shape = tuple(lead_shape) + (n,)
    if out is None:
        out = torch.empty(out_shape, dtype=act.dtype, device=act.device)
    elif out.dtype != act.dtype or tuple(out.shape) != out_shape:
        raise ValueError(
            f"out must be {out_shape} in {act.dtype}, got {tuple(out.shape)} in {out.dtype}"
        )
    elif out.device != act.device:
        raise ValueError(f"out must be on act's device, got {out.device}")
    elif not out.is_contiguous():
        # reshape() would hand the kernel a contiguous copy and the caller's
        # buffer would never see the result.
        raise ValueError("out must be contiguous")
    if workspace is not None:
        if workspace.device != act.device or not workspace.is_contiguous():
            raise ValueError("workspace must be a contiguous tensor on act's device")
        _check_aligned(workspace, "workspace")
    if not weight.is_contiguous():
        raise ValueError("weight must be contiguous (this function does not copy)")
    _check_aligned(weight, "weight")
    if bias is not None:
        _check_aligned(bias, "bias")
    # Aliasing: the two buffers the call writes against everything it reads (and each
    # other).  Inputs may alias one another; the empty act / out of an M == 0 call
    # cover no bytes and so overlap nothing.
    reads = [("act", act), ("weight", weight), ("weight_scale", weight_scale)]
    if bias is not None:
        reads.append(("bias", bias))
    for name, t in reads:
        _check_disjoint("out", out, name, t)
    if workspace is not None:
        for name, t in reads + [("out", out)]:
            _check_disjoint("workspace", workspace, name, t)
    if m == 0:
        # Nothing to compute; the tile schedulers would divide by zero tiles.
        # Everything above applies to the empty call too; only act and out are
        # empty here, and an empty tensor has no address to align.
        return out
    act2d = act.view(-1, k)
    out2d = out.view(-1, n)
    _check_aligned(act2d, "act")
    _check_aligned(out2d, "out")
    # Split-K / stream-K variants need scratch; the launcher decides the variant,
    # this only sizes the buffer (0 bytes = single pass).  The launchers take any
    # buffer by its byte size.
    ws_bytes = _workspace_bytes(m, n, k, device.index, algo_code, act.dtype)
    if ws_bytes == 0:
        ws = None
    elif workspace is None:
        ws = torch.empty(ws_bytes, dtype=torch.uint8, device=act.device)
    else:
        have = workspace.numel() * workspace.element_size()
        if have < ws_bytes:
            raise ValueError(
                f"workspace too small: {have} bytes, this call needs {ws_bytes} "
                "(see w8a16_fp8_gemm_workspace_bytes)"
            )
        ws = workspace
    _w8a16_fp8_gemm_impl(out2d, act2d, weight, weight_scale, bias, ws, algo_code)
    return out
