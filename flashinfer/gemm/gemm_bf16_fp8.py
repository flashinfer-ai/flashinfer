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
from typing import Literal, Optional

import torch

from ..api_logging import flashinfer_api
from ..trace.templates.gemm import (
    mm_bf16_fp8_trace_dispatch,
    mm_fp16_fp8_trace_dispatch,
)
from ..utils import (
    BackendSupportedError,
    backend_requirement,
    get_compute_capability,
    register_custom_op,
    register_fake_op,
    supported_compute_capability,
)
from . import w8a16_fp8_dispatch as _dispatch

_Backend = Literal["auto", "cute-dsl"]


def _resolve_device(device: Optional[torch.device]) -> torch.device:
    """``None`` and an index-less ``torch.device("cuda")`` both mean the current
    CUDA device; the result always carries an explicit index so the module
    choice, the tactic table and the workspace cache key agree on one GPU."""
    if device is None:
        return torch.device("cuda", torch.cuda.current_device())
    if device.type != "cuda":
        raise ValueError(
            f"mm_bf16_fp8 / mm_fp16_fp8 run on CUDA devices only, got {device}"
        )
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
    written_name: str, w: Optional[tuple], read_name: str, r: Optional[tuple]
) -> None:
    """The byte span ``w`` is overwritten while ``r`` is still being read, so the
    two must not share a byte (spans of contiguous tensors from :func:`_byte_span`;
    nothing here is a strided view; None is an empty tensor, which overlaps
    nothing)."""
    if w is not None and r is not None and w[0] < r[1] and r[0] < w[1]:
        raise ValueError(
            f"{written_name} must not overlap {read_name} in memory ({written_name} "
            f"is written while {read_name} is still being read)"
        )


# The one definition of the supported devices: the calls' @backend_requirement
# (and ``is_compute_capability_supported``) and the workspace queries all use it.
_SUPPORTED_CCS = (90, 100, 103)


def _check_arch(device: torch.device) -> None:
    major, minor = get_compute_capability(device)
    cc = major * 10 + minor
    if cc not in _SUPPORTED_CCS:
        raise BackendSupportedError(
            f"mm_bf16_fp8 / mm_fp16_fp8 do not support compute capability {cc}; "
            f"supported compute capabilities: {list(_SUPPORTED_CCS)}"
        )


def _register(name: str):
    """One custom op per public function (so a trace / torch.compile graph names
    the function the caller used); both run the same launcher."""

    @register_custom_op(f"flashinfer::{name}", mutates_args=("out", "workspace"))
    def impl(
        out: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        alpha: torch.Tensor,
        bias: Optional[torch.Tensor],
        workspace: Optional[torch.Tensor],
        tactic: int,
    ) -> None:
        _dispatch.launch_w8a16_fp8_gemm(
            a, b, alpha, bias, out, tactic, workspace=workspace
        )

    @register_fake_op(f"flashinfer::{name}")
    def impl_fake(
        out: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        alpha: torch.Tensor,
        bias: Optional[torch.Tensor],
        workspace: Optional[torch.Tensor],
        tactic: int,
    ) -> None:
        pass

    return impl


_mm_bf16_fp8_impl = _register("mm_bf16_fp8")
_mm_fp16_fp8_impl = _register("mm_fp16_fp8")


@functools.cache
def _workspace_bytes(m: int, n: int, k: int, device_index: int, tactic: int) -> int:
    """Bytes of workspace this problem needs on this device (0 = none): the fp32
    partials of a split-K run (and, on Hopper, the split tiles' arrival
    counters), when the variant (forced or chosen) splits K."""
    device = torch.device("cuda", device_index)
    _check_arch(device)
    return _dispatch.w8a16_fp8_gemm_workspace_bytes(m, n, k, device, tactic)


def mm_bf16_fp8_workspace_bytes(
    m: int,
    n: int,
    k: int,
    device: Optional[torch.device] = None,
    tactic: Optional[str] = None,
) -> int:
    """Workspace bytes :func:`mm_bf16_fp8` needs for an ``[M, K] x [N, K]`` problem.

    Parameters
    ----------
    m, n, k: int
        The problem: ``m`` is the flattened row count of ``a`` (the product
        of its leading dimensions), ``n`` the weight's row count, ``k`` the
        reduction depth.  Same bounds as the call: ``m >= 0``, ``n`` and ``k``
        positive, ``k`` a multiple of 16, ``n`` a multiple of 8, each at most
        ``2**31 - 512``, and ``ceil(m / 64) * ceil(n / 64)`` at most
        ``2**28 - 1``.
    device: Optional[torch.device]
        The CUDA device the call will run on (current device when omitted or
        given without an index).  The answer depends on it.
    tactic: Optional[str]
        The ``tactic`` the call will be made with (``None`` = the launcher's
        choice).  The answer depends on it exactly as the launch does, so
        query with the arguments you will call with; a caller serving several
        shapes from one buffer takes the maximum over them.

    Returns
    -------
    int
        ``0`` when the call needs no workspace (every ``m == 0`` call, and
        every shape whose variant runs in a single pass), otherwise the byte
        count a ``workspace`` passed to :func:`mm_bf16_fp8` must reach.

    Raises
    ------
    ValueError
        A shape outside the bounds above, an unknown ``tactic``, a non-CUDA
        device, or a split-K ``tactic`` on a ``k`` that cannot be sliced (for
        ``m > 0``).
    BackendSupportedError
        A device the kernels do not support (anything but compute capability
        9.0, 10.0 or 10.3), as the call raises.
    """
    device = _resolve_device(device)
    _check_arch(device)
    _check_dims(m, n, k)
    code = _tactic_code(tactic, device)
    if m == 0:
        return 0
    return _workspace_bytes(m, n, k, device.index, code)


def mm_fp16_fp8_workspace_bytes(
    m: int,
    n: int,
    k: int,
    device: Optional[torch.device] = None,
    tactic: Optional[str] = None,
) -> int:
    """Workspace bytes :func:`mm_fp16_fp8` needs for an ``[M, K] x [N, K]`` problem.

    Same arguments, answer and errors as :func:`mm_bf16_fp8_workspace_bytes`:
    the two functions run the same kernels and the workspace does not depend
    on the activation dtype.
    """
    return mm_bf16_fp8_workspace_bytes(m, n, k, device, tactic)


# The kernels index with 32-bit ints: a dimension needs headroom for the largest
# tile (256) in the tile-count ceiling division, so `dim + 256` itself must fit; and
# the output tile count (times the split count, at most 8) must fit an int.
_MAX_DIM = 2**31 - 512
_MAX_TILES64 = (2**31 - 1) // 8  # 64x64 output tiles


def _check_dims(m: int, n: int, k: int) -> None:
    """Shape rules shared by the call and the workspace query (``m == 0`` is legal
    and handled by the caller; ``n`` / ``k`` of zero are not: the tile schedulers
    divide by the tile counts)."""
    if m < 0 or n <= 0 or k <= 0:
        raise ValueError(f"need M >= 0, N > 0, K > 0, got M={m} N={n} K={k}")
    if max(m, n, k) > _MAX_DIM:
        raise ValueError(f"M, N, K must be <= {_MAX_DIM}, got M={m} N={n} K={k}")
    if -(-m // 64) * -(-n // 64) > _MAX_TILES64:
        raise ValueError(
            f"M x N is too large: ceil(M / 64) x ceil(N / 64) must be <= {_MAX_TILES64}, "
            f"got M={m} N={n}"
        )
    if k % 16 != 0:
        raise ValueError(
            f"K must be a multiple of 16 (16-byte TMA rows of fp8), got K={k}"
        )
    if n % 8 != 0:
        raise ValueError(f"N must be a multiple of 8 (16-byte output rows), got N={n}")


# Blackwell variants selectable through ``tactic`` (``_SM100_TILES`` in
# w8a16_fp8_dispatch.py), for mm_bf16_fp8 and mm_fp16_fp8 alike.  The stream-K
# code is a policy: a wave-fill cost model over the plain 2SM tiles (256x128,
# 256x160, 256x192) and split-K on 256x128.  Code 3 is reserved.
MM_BF16_FP8_TACTICS_SM100: dict[str, int] = {
    "2sm256x128": 1,
    "1sm128x128": 2,
    "2sm256x192": 4,
    "2sm256x128_splitk": 5,
    "2sm256x128_streamk": 6,
    "2sm256x160": 7,
}


# Hopper kernel variants selectable through ``tactic`` (weight rows x activation
# rows per CTA tile, plus the stream-K and split-K decompositions; ``_SM90_TILES``
# in w8a16_fp8_dispatch.py), for mm_bf16_fp8 and mm_fp16_fp8 alike.  Codes 1 and 4
# are reserved.
MM_BF16_FP8_TACTICS_SM90: dict[str, int] = {
    "128x256": 2,
    "128x192": 3,
    "128x128": 5,
    "64x256": 6,
    "128x256_streamk": 7,
    "128x192_streamk": 8,
    "64x128": 9,
    "128x128_splitk": 10,
    "128x256_splitk": 11,
    "64x128_splitk": 12,
}


def _tactic_code(tactic: Optional[str], device: torch.device) -> int:
    if tactic is None:
        return 0
    major = get_compute_capability(device)[0]
    if major == 10:
        table = MM_BF16_FP8_TACTICS_SM100
    elif major == 9:
        table = MM_BF16_FP8_TACTICS_SM90
    else:
        raise ValueError(
            "tactic is only selectable on Hopper / Blackwell; use tactic=None"
        )
    try:
        return table[tactic]
    except KeyError:
        raise ValueError(f"unknown tactic {tactic!r}; one of {sorted(table)}") from None


@supported_compute_capability(_SUPPORTED_CCS)
def _cute_dsl_check(*args, **kwargs) -> bool:
    """Device support of the CuTe-DSL backend; the argument contract is checked
    in the function body, where the messages name the offending argument."""
    return True


def _heuristic(suitable_backends, *args, **kwargs):
    """``backend="auto"``: the one backend there is."""
    return suitable_backends


def _mm_16bit_fp8(
    fn_name: str,
    a_dtype: torch.dtype,
    impl,
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_dtype: Optional[torch.dtype],
    out: Optional[torch.Tensor],
    backend: str,
    tactic: Optional[str],
    workspace: Optional[torch.Tensor],
) -> torch.Tensor:
    """Shared body of :func:`mm_bf16_fp8` and :func:`mm_fp16_fp8`: the contract
    checks (each message names the argument), then one launch."""
    if a.dtype != a_dtype:
        other = "mm_fp16_fp8" if a_dtype == torch.bfloat16 else "mm_bf16_fp8"
        hint = f" (use {other})" if a.dtype in (torch.float16, torch.bfloat16) else ""
        raise ValueError(
            f"{fn_name} takes a {a_dtype} activation, got a in {a.dtype}{hint}"
        )
    if out_dtype is not None and out_dtype != a.dtype:
        raise ValueError(
            f"out_dtype must be a's dtype ({a.dtype}); the kernels round the fp32 "
            f"accumulator once into the activation dtype.  Got {out_dtype}"
        )
    if backend not in ("auto", "cute-dsl"):
        raise ValueError(f"backend must be 'auto' or 'cute-dsl', got {backend!r}")
    m, n, k, lead_shape = _check_operands(a, b, alpha)
    # alpha goes down as given: () or (1,), one fp32 value either way, and the
    # launcher reads only its address.
    device = _resolve_device(a.device)
    if b.device != a.device or alpha.device != a.device:
        raise ValueError(
            f"b and alpha must be on a's device {a.device}, got {b.device} and "
            f"{alpha.device}"
        )
    _check_arch(device)
    code = _tactic_code(tactic, device)
    # No implicit copies: a non-contiguous a (row slice, transposed last dim,
    # permuted leading dims) is the caller's to .contiguous() if they want that.
    if not a.is_contiguous():
        raise ValueError(
            "a must be contiguous (this function does not copy; pass "
            "a.contiguous() if a copy is acceptable)"
        )
    if bias is not None:
        _check_bias(bias, a, n)
    out = _prepare_out(out, a, tuple(lead_shape) + (n,))
    _check_buffers(workspace, b, bias, a.device)
    _check_aliasing(out, workspace, a, b, alpha, bias)
    if m == 0:
        # Nothing to compute; the tile schedulers would divide by zero tiles.
        # Everything above applies to the empty call too; only a and out are
        # empty here, and an empty tensor has no address to align.
        return out
    # The kernels take [M, K] / [M, N]: the tensors themselves when already 2-D,
    # otherwise metadata-only views of the same contiguous storage (no copy).
    a2d = a if a.dim() == 2 else a.view(-1, k)
    out2d = out if out.dim() == 2 else out.view(-1, n)
    _check_aligned(a2d, "a")
    _check_aligned(out2d, "out")
    ws_bytes = _workspace_bytes(m, n, k, device.index, code)
    ws = _select_workspace(workspace, ws_bytes, a.device, fn_name)
    impl(out2d, a2d, b, alpha, bias, ws, code)
    return out


@backend_requirement({"cute-dsl": _cute_dsl_check}, heuristic_func=_heuristic)
@flashinfer_api(trace=mm_bf16_fp8_trace_dispatch)
def mm_bf16_fp8(
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: torch.Tensor,
    *,
    bias: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
    backend: _Backend = "auto",
    tactic: Optional[str] = None,
    workspace: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""bfloat16 GEMM whose ``[N, K]`` operand is stored as FP8 e4m3 (W8A16).

    ``out = a @ (b.float() * alpha).T + bias`` with the weight dequantized
    inside the kernel; no 16-bit copy of the weight is allocated.  "fp8" says
    how the weight is *stored* (e4m3 plus one per-tensor fp32 scale); the
    arithmetic is a 16-bit GEMM in ``bfloat16``.  :func:`mm_fp16_fp8` is the same
    kernel for float16 activations.

    Numerics.  Each e4m3 weight value is converted to ``bfloat16`` exactly
    (every finite e4m3 value is representable in both bf16 and fp16, denormals
    included), the unscaled products are accumulated in fp32, and ``alpha``
    and ``bias`` are applied to the fp32 accumulator before the single
    rounding to the output dtype.  The result is therefore not bitwise equal
    to a GEMM against a weight pre-scaled and rounded to the activation dtype
    (``(b.to(dtype) * alpha) @ ...``): that carries one rounding more.  The
    weight must be finite.  Values are not checked, and a NaN e4m3 encoding
    yields an unspecified result (a finite value or NaN).

    Parameters
    ----------
    a: torch.Tensor
        Activation, shape ``(..., K)``, ``bfloat16``, contiguous, on a CUDA
        device (Hopper or Blackwell; every other tensor must be on the same
        device).  Leading dimensions are flattened into ``M``; a leading
        dimension of 0 returns the empty output without launching.
    b: torch.Tensor
        Weight, shape ``(N, K)`` (``torch.nn.Linear`` layout),
        ``float8_e4m3fn``, contiguous, on ``a``'s device.
    alpha: torch.Tensor
        One ``float32`` value (shape ``()`` or ``(1,)``) on ``a``'s device:
        the per-tensor dequantization scale of ``b``.
    bias: Optional[torch.Tensor]
        Optional bias, shape ``(N,)``, ``bfloat16``, on ``a``'s device,
        contiguous.
    out_dtype: Optional[torch.dtype]
        Output dtype.  Only ``bfloat16`` (``a``'s dtype, the default) is
        supported: the kernels round the fp32 accumulator once into the
        activation dtype.  The argument exists for uniformity with the other
        ``mm_*`` functions.
    out: Optional[torch.Tensor]
        Optional output, shape ``(..., N)`` (``a``'s leading dimensions),
        ``bfloat16``, on ``a``'s device, contiguous.  Allocated when omitted.
    backend: str
        ``"auto"`` (default) or ``"cute-dsl"``, the one implementation there
        is (CuTe-DSL kernels for Hopper and Blackwell).
    tactic: Optional[str]
        Selects a tile or a split policy instead of the launcher's choice: a
        key of ``MM_BF16_FP8_TACTICS_SM100`` on Blackwell or of
        ``MM_BF16_FP8_TACTICS_SM90`` on Hopper
        (``<weight rows>x<activation rows>[_streamk|_splitk]``; the tables
        serve both activation dtypes).  ``None`` = the launcher decides.  This
        is the only way to override the choice; no environment variable does
        (the JIT / cache variables under Notes do not affect selection).  A
        plain name runs every output tile whole on that tile shape.  A
        ``_splitk`` name cuts the tiles into K-slices (fp32 partials in the
        workspace); it is rejected with a ValueError on a ``K`` it cannot
        slice (Hopper: ``K`` a multiple of 64 whose 64-deep tile count divides
        into 2..8 equal slices of at least 8 tiles; Blackwell: ``K`` a
        multiple of 128 whose 128-deep tile count has a divisor in 2..8), and
        runs the tile plain (no workspace) where the split cannot run (Hopper:
        a tail whose slices would not fit one wave of SMs; Blackwell:
        ``splits x M`` at or above ``2**31``).  A ``_streamk`` name is a
        policy on the named tile, not a stream-K kernel: on Hopper it runs
        the whole waves of that tile plain and splits only the last partial
        wave over the idle SMs (plain when that tail is at least half a wave
        or ``K`` is short); on Blackwell it picks, by a wave-fill cost model,
        between the plain 2SM tiles and split-K on the 256x128 tile.
        :func:`mm_bf16_fp8_workspace_bytes` reports what the resolved policy needs.
    workspace: Optional[torch.Tensor]
        Optional scratch buffer: contiguous, 16-byte aligned, of any dtype, on
        ``a``'s device, at least :func:`mm_bf16_fp8_workspace_bytes` bytes for this call's shape,
        device and ``tactic`` (its byte size counts, not its dtype).
        Allocated per call when omitted and needed.  A caller that wants a
        fixed memory budget sizes one buffer once and passes it to every call,
        under these rules: the call overwrites the buffer (nothing in it
        survives), it must not share memory with any other argument (the
        aliasing rule under Notes), and two calls may share one buffer only if
        they are ordered on one stream or synchronised, since a call still
        running on another stream would be reading what the next call
        overwrites.

    Notes
    -----
    Runs on Hopper (``sm_90a``) and Blackwell (``sm_100a`` / ``sm_103a``);
    ``mm_bf16_fp8.is_compute_capability_supported(cc)`` answers for a device.  The
    kernels are CuTe-DSL, compiled on first use of each kernel variant a call
    resolves to (so the first call of a new configuration is slow), and are
    not part of the AOT packages.  Compiled kernels are cached on disk under
    the JIT workspace like every CuTe-DSL kernel, so a later process reloads
    instead of recompiling; ``FLASHINFER_DISABLE_JIT`` and
    ``FLASHINFER_CUTE_DSL_DISABLE_CACHE`` apply.

    Shapes and layout: ``K`` must be a multiple of 16 and ``N`` a multiple of
    8, both non-zero, and ``M`` (the product of ``a``'s leading dimensions,
    not each of them), ``N`` and ``K`` each at most ``2**31 - 512``, with
    ``ceil(M / 64) * ceil(N / 64)`` at most ``2**28 - 1`` (the kernels'
    tile-count limit).  Every tensor among ``a`` / ``b`` / ``bias`` / ``out``
    / ``workspace`` must start at a 16-byte aligned address (a view into a
    larger buffer at an odd storage offset does not; contiguity alone is not
    enough).  Nothing is copied on the caller's behalf: a non-contiguous
    ``a``, ``b``, ``bias``, ``out`` or ``workspace`` is rejected (call
    ``.contiguous()`` yourself if a copy is acceptable).  Anything else is
    rejected with a ValueError before launch.

    Aliasing: the two buffers the call writes, ``out`` and ``workspace``,
    must not share any bytes with ``a`` / ``b`` / ``alpha`` / ``bias`` or with
    each other; a call whose buffers overlap is rejected with a ValueError.
    The inputs may alias one another freely (they are only read).

    Empty calls: with ``M == 0`` the dtypes, shapes, devices, contiguity and
    aliasing of every argument and the alignment of ``b``, ``bias`` and
    ``workspace`` are checked as for a non-empty call (the empty ``a`` and
    ``out`` have no address to align), then the empty output is returned
    without launching.  No kernel variant is resolved and no workspace is
    sized or checked for such a call, so a split-K ``tactic`` on an
    unsliceable ``K`` or an undersized ``workspace`` is not reported there.

    Returns
    -------
    out: torch.Tensor
        ``(..., N)`` in ``bfloat16``.

    Raises
    ------
    ValueError
        Any argument outside the contract above (dtype, shape, device,
        contiguity, alignment, aliasing, an unknown ``tactic`` or
        ``backend``, a ``workspace`` shorter than the query's answer), before
        anything is launched.
    BackendSupportedError
        A device the kernels do not support (anything but compute capability
        9.0, 10.0 or 10.3).
    RuntimeError
        A launch failure reported by the CUDA runtime.
    """
    return _mm_16bit_fp8(
        "mm_bf16_fp8",
        torch.bfloat16,
        _mm_bf16_fp8_impl,
        a,
        b,
        alpha,
        bias,
        out_dtype,
        out,
        backend,
        tactic,
        workspace,
    )


@backend_requirement({"cute-dsl": _cute_dsl_check}, heuristic_func=_heuristic)
@flashinfer_api(trace=mm_fp16_fp8_trace_dispatch)
def mm_fp16_fp8(
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: torch.Tensor,
    *,
    bias: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
    backend: _Backend = "auto",
    tactic: Optional[str] = None,
    workspace: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""float16 GEMM whose ``[N, K]`` operand is stored as FP8 e4m3 (W8A16).

    ``out = a @ (b.float() * alpha).T + bias`` with the weight dequantized
    inside the kernel: :func:`mm_bf16_fp8` for ``float16`` activations.  Same
    kernels, same arguments, same contract, with ``float16`` wherever that
    function says ``bfloat16``: ``a`` is ``(..., K)`` ``float16``; ``bias`` and
    ``out`` are ``float16``; ``out_dtype`` accepts only ``torch.float16``; the
    result is ``(..., N)`` in ``float16``.  The e4m3 weight is converted to
    ``float16`` exactly, the products are accumulated in fp32, and ``alpha``
    and ``bias`` are applied before the single rounding into ``float16``.
    ``tactic`` takes the same keys (``MM_BF16_FP8_TACTICS_SM100`` /
    ``MM_BF16_FP8_TACTICS_SM90``); :func:`mm_fp16_fp8_workspace_bytes` sizes
    the ``workspace``.  See :func:`mm_bf16_fp8` for the full description of
    every argument, the shape, alignment and aliasing rules, and the errors.
    """
    return _mm_16bit_fp8(
        "mm_fp16_fp8",
        torch.float16,
        _mm_fp16_fp8_impl,
        a,
        b,
        alpha,
        bias,
        out_dtype,
        out,
        backend,
        tactic,
        workspace,
    )


def _check_operands(
    a: torch.Tensor, b: torch.Tensor, alpha: torch.Tensor
) -> tuple[int, int, int, tuple]:
    """dtype, rank and shape of the three required operands; the resolved
    ``(M, N, K, a's leading shape)``.  A 0-d ``a`` is rejected before its last
    dimension is read.  ``alpha`` is one fp32 value of shape ``()`` or ``(1,)``
    and goes to the launcher as given."""
    if b.dtype != torch.float8_e4m3fn:
        raise ValueError(f"b must be float8_e4m3fn, got {b.dtype}")
    if b.dim() != 2:
        raise ValueError(f"b must be 2D [N, K], got shape {tuple(b.shape)}")
    if a.dim() == 0:
        raise ValueError("a must have at least one dimension (..., K)")
    k = a.shape[-1]
    n = b.shape[0]
    if b.shape[1] != k:
        raise ValueError(f"K mismatch: a[..., {k}] vs b[{n}, {b.shape[1]}]")
    if tuple(alpha.shape) not in ((), (1,)):
        raise ValueError(
            f"alpha is one per-tensor value of shape () or (1,), got shape "
            f"{tuple(alpha.shape)}"
        )
    lead_shape = a.shape[:-1]
    m = a.numel() // k if k else 0
    _check_dims(m, n, k)
    if alpha.dtype != torch.float32:
        # No implicit conversions either: the caller casts if they want that.
        raise ValueError(f"alpha must be float32, got {alpha.dtype}")
    return m, n, k, lead_shape


def _check_bias(bias: torch.Tensor, a: torch.Tensor, n: int) -> None:
    """``bias`` is ``(N,)`` in ``a``'s dtype, on its device, contiguous."""
    if bias.dtype != a.dtype or tuple(bias.shape) != (n,):
        raise ValueError(
            f"bias must be ({n},) in {a.dtype}, got {tuple(bias.shape)} in {bias.dtype}"
        )
    if bias.device != a.device:
        raise ValueError(f"bias must be on a's device, got {bias.device}")
    if not bias.is_contiguous():
        raise ValueError("bias must be contiguous")


def _check_buffers(
    workspace: Optional[torch.Tensor],
    b: torch.Tensor,
    bias: Optional[torch.Tensor],
    device: torch.device,
) -> None:
    """Layout of the buffers the kernels address directly: a contiguous, 16-byte
    aligned ``workspace`` on ``device`` when given, a contiguous aligned ``b``,
    an aligned ``bias`` (its shape and contiguity were checked with its dtype)."""
    if workspace is not None:
        if workspace.device != device or not workspace.is_contiguous():
            raise ValueError("workspace must be a contiguous tensor on a's device")
        _check_aligned(workspace, "workspace")
    if not b.is_contiguous():
        raise ValueError("b must be contiguous (this function does not copy)")
    _check_aligned(b, "b")
    if bias is not None:
        _check_aligned(bias, "bias")


def _prepare_out(
    out: Optional[torch.Tensor], a: torch.Tensor, out_shape: tuple
) -> torch.Tensor:
    """The caller's ``out`` after validation, or a fresh one."""
    if out is None:
        return torch.empty(out_shape, dtype=a.dtype, device=a.device)
    if out.dtype != a.dtype or tuple(out.shape) != out_shape:
        raise ValueError(
            f"out must be {out_shape} in {a.dtype}, got {tuple(out.shape)} in {out.dtype}"
        )
    if out.device != a.device:
        raise ValueError(f"out must be on a's device, got {out.device}")
    if not out.is_contiguous():
        # reshape() would hand the kernel a contiguous copy and the caller's
        # buffer would never see the result.
        raise ValueError("out must be contiguous")
    return out


def _check_aliasing(
    out: torch.Tensor,
    workspace: Optional[torch.Tensor],
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> None:
    """The two buffers the call writes against everything it reads (and each
    other).  Inputs may alias one another; the empty a / out of an M == 0 call
    cover no bytes and so overlap nothing."""
    # Each span once per call (never cached across calls: the next call may pass
    # other storage), then the same comparisons in the same order.
    out_span = _byte_span(out)
    reads = [
        ("a", _byte_span(a)),
        ("b", _byte_span(b)),
        ("alpha", _byte_span(alpha)),
    ]
    if bias is not None:
        reads.append(("bias", _byte_span(bias)))
    for name, span in reads:
        _check_disjoint("out", out_span, name, span)
    if workspace is not None:
        ws_span = _byte_span(workspace)
        for name, span in reads + [("out", out_span)]:
            _check_disjoint("workspace", ws_span, name, span)


def _select_workspace(
    workspace: Optional[torch.Tensor],
    ws_bytes: int,
    device: torch.device,
    fn_name: str,
) -> Optional[torch.Tensor]:
    """The scratch buffer handed to the launcher: ``None`` when the variant
    needs no bytes; a fresh allocation when the caller passed none; otherwise
    the caller's buffer, or a ValueError when it is too small (it is never
    replaced).  The launchers take any buffer by its byte size."""
    if ws_bytes == 0:
        return None
    if workspace is None:
        return torch.empty(ws_bytes, dtype=torch.uint8, device=device)
    have = workspace.numel() * workspace.element_size()
    if have < ws_bytes:
        raise ValueError(
            f"workspace too small: {have} bytes, this call needs {ws_bytes} "
            f"(see {fn_name}_workspace_bytes)"
        )
    return workspace
