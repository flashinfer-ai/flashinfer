"""Offline MX-FP6 (W6A6) weight quantization + save/load for *dense* linears.

This is the dense counterpart of
:mod:`b12x.quantization.mxfp6.fp6_moe_weights`. It targets ordinary
``nn.Linear`` weights (MLP gate/up/down, attention q/k/v/o projections) and the
block-scaled :func:`b12x._lib.dense_gemm.dense_gemm` kernel, rather than
fused-MoE experts.

W6A6 split (same as MoE):

* **Weights** are static, so they are quantized once here into the packed
  MX-FP6 codes + swizzled UE8M0 block-scales that ``dense_gemm`` consumes.
* **Activations** are data-dependent, so :func:`dense_fp6_linear` quantizes them
  to FP6 on the fly each forward (via the GPU quantizer), then runs the matmul
  fully in 6-bit.

A linear ``y = x @ W.T`` with ``W`` shape ``(out_features, in_features)`` maps
directly onto ``dense_gemm``: the activation ``x`` ``(M, K)`` is the LHS and the
weight ``W`` ``(N, K)`` is the RHS (``dense_gemm`` computes ``a @ b.T``), so the
HF weight layout is used as-is with no transpose.

Scaling mirrors the validated ``test_dense_gemm_mxfp6_*`` recipe: each operand
uses a bf16 global-scale that maps its amax into FP6 range, and
``alpha = 1 / (a_gscale * b_gscale)`` undoes both at the epilogue.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import Optional

import torch

from b12x._lib.fp6 import (
    SF_VEC_SIZE_FP6,
    as_grouped_mxfp6_scale_view,
    mx_gs_numerator,
)
from b12x._lib.utils import mxfp6_packed_k_bytes
from b12x._lib.dense_gemm import dense_gemm, _DENSE_FUSED_QUANT

_TILE = 128  # MX-FP6 GPU quantizer tile (M and K must be multiples of this)

# Activations with at most this many rows use the small-M quantizer (only the
# real rows are computed; no padded copy). Mirrors bf16_to_fp6_small_m.SMALL_M_MAX.
_SMALL_M_QUANT_MAX = 16


# Persistent stream-local decode-quantization workspaces. The quantizer writes
# them before the GEMM consumes them on the same stream; GEMM outputs are not
# pooled because they escape the linear call.
_DENSE_PER_ROW_GS = os.getenv("B12X_DENSE_PER_ROW_GS", "1").lower() not in (
    "0",
    "false",
)
_PERSISTENT_SCRATCH = os.getenv(
    "B12X_DENSE_PERSISTENT_SCRATCH", "1"
).lower() not in ("0", "false")
# The fallback host chain and in-kernel per-row scaling are bit-identical.
_PER_ROW_IN_KERNEL = os.getenv(
    "B12X_DENSE_PER_ROW_IN_KERNEL", "1"
).lower() not in ("0", "false")
# Applies the per-row output correction inside the GEMM epilogue instead of as
# a trailing ``result.mul_(inv_gs)``. Bit-identical by construction: the
# epilogue reproduces both roundings to bf16 that the eager multiply performs
# (see the row_scale application site in _lib/dense_gemm.py).
_ROW_SCALE_EPILOGUE = True
_QUANT_SCRATCH: dict[tuple, tuple] = {}
# Captures claim prewarmed buckets with stable addresses. Claimed entries stay
# reserved for graph replay and are not returned to subsequent eager calls.
_CAPTURE_ASSIGNED: dict[tuple, tuple] = {}
_CAPTURE_CLAIMED: dict[tuple, list[int]] = {}


def _small_m_quant_scratch(m_pad: int, k: int, device: torch.device) -> tuple:
    """Return stream-local decode scratch with capture-stable ownership.

    Capture claims a prewarmed eager bucket and fails if none is available;
    allocation and zero-initialization never occur inside capture.
    """
    from b12x.quantization.mxfp6 import allocate_bf16_to_fp6_tma_outputs

    capturing = device.type == "cuda" and torch.cuda.is_current_stream_capturing()
    # Stream ownership permits independent main and side-stream execution.
    stream = (
        torch.cuda.current_stream(device).cuda_stream
        if device.type == "cuda"
        else 0
    )
    key = (device.type, device.index or 0, stream, m_pad, k)
    bucket_key = (device.type, device.index or 0, m_pad, k)
    if _PERSISTENT_SCRATCH and not capturing:
        entry = _QUANT_SCRATCH.get(key)
        claimed = _CAPTURE_CLAIMED.get(bucket_key, ())
        if entry is not None and id(entry) not in claimed:
            return entry
    if capturing:
        assigned = _CAPTURE_ASSIGNED.get(key)
        if assigned is not None:
            return assigned
        claimed = _CAPTURE_CLAIMED.setdefault(bucket_key, [])
        for (dt, di, _s, mp, kk), entry in _QUANT_SCRATCH.items():
            if (dt, di, mp, kk) == bucket_key and id(entry) not in claimed:
                _CAPTURE_ASSIGNED[key] = entry
                claimed.append(id(entry))
                return entry
        raise RuntimeError(
            "decode quantization requires a prewarmed scratch bucket during "
            "CUDA graph capture. Run an eager forward on each capture stream "
            "at the required shape before capture."
        )
    out = allocate_bf16_to_fp6_tma_outputs(m_pad, k, device=device, emit="bytes")
    alpha = torch.zeros(1, dtype=torch.float32, device=device)
    # Per-row output-correction buffer for the in-kernel per-row path: sized
    # at the small-M ceiling so every m <= 16 shares the (m_pad, k) bucket.
    inv_gs = torch.ones(_SMALL_M_QUANT_MAX, dtype=torch.bfloat16, device=device)
    entry = (out, alpha, inv_gs)
    if _PERSISTENT_SCRATCH and not capturing:
        _QUANT_SCRATCH[key] = entry
    return entry

# Minimum out_features for which the GEMM streams the 3:4-packed weight
# directly (``b_packed=True``) instead of a cached 1-byte/code expansion.
# Default 12288 enables shapes with enough N tiles for the packed path.
PACKED_GEMM_MIN_N = int(os.getenv("B12X_PACKED_B_MIN_N", "12288"))


def _weight_fmt_for_source(source_format: str) -> str:
    """FP6 sub-format string (``e3m2``/``e2m3``) for a source-format selector."""
    sf = source_format.lower()
    if sf in ("mxfp6_e3m2", "e3m2", "float6_e3m2fn"):
        return "e3m2"
    if sf in ("mxfp6_e2m3", "e2m3", "float6_e2m3fn"):
        return "e2m3"
    # mxfp6_default / mxfp6_mixed: E2M3 weights (more mantissa for static weights).
    return "e2m3"


def _bf16_global_scale(
    amax: float, device: torch.device | str, fmt: str = "e3m2"
) -> torch.Tensor:
    """Global scale mapping ``amax`` into the ``fmt`` range (fmt-aware numerator)."""
    return torch.tensor(
        [mx_gs_numerator(fmt) / max(amax, 1e-6)],
        dtype=torch.float32,
        device=device,
    )


def _quantize_matrix_fp6(
    mat_bf16: torch.Tensor, fmt: str, global_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``(M, K)`` bf16 to packed MX-FP6 codes + flat swizzled scales.

    Uses the GPU quantizer (``compile_bf16_to_fp6_tma``) — the kernel's own
    bit-identical quantizer. Requires CUDA and ``M % 128 == 0``, ``K % 128 == 0``.

    Returns ``(packed (M, 3K//4) uint8, scale_storage flat uint8)``.
    """
    m, k = mat_bf16.shape
    if m % _TILE != 0 or k % _TILE != 0:
        raise ValueError(
            f"GPU FP6 quantizer requires M%{_TILE}==0 and K%{_TILE}==0, got M={m} K={k}"
        )
    # Lazy import to avoid an import cycle with the package __init__.
    from b12x.quantization.mxfp6 import (
        allocate_bf16_to_fp6_tma_outputs,
        compile_bf16_to_fp6_tma,
    )

    launch = compile_bf16_to_fp6_tma(m, k, fmt=fmt)
    out = allocate_bf16_to_fp6_tma_outputs(m, k, device=mat_bf16.device)
    launch(mat_bf16.contiguous(), global_scale, out.packed_a_flat, out.scale_flat)
    # No host sync: the kernel writes packed/scale on the current stream and every
    # consumer (the dense GEMM at runtime, the .clone()/save at load time) is
    # ordered after it on that same stream. A torch.cuda.synchronize() here would
    # drain the device once per FP6 linear, which on a decode step (hundreds of
    # linears x MTP passes) collapses throughput; stream ordering keeps it correct.
    packed = out.packed_a_storage.view(m, mxfp6_packed_k_bytes(k)).clone()
    return packed, out.scale_storage.clone()


def _quantize_matrix_fp6_bytes(
    mat_bf16: torch.Tensor, fmt: str, global_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``(M, K)`` bf16 to byte-container MX-FP6 codes + flat scales.

    Identical quantization math to :func:`_quantize_matrix_fp6` but the kernel
    emits one code per byte (``emit="bytes"``) — the exact ``mxf8f6f4`` GEMM
    operand layout — so the per-call activation expansion disappears. Returns
    views over freshly allocated buffers (the caller owns them; no clones).

    Returns ``(codes (M, K) uint8, scale_storage flat uint8)``.
    """
    m, k = mat_bf16.shape
    if m % _TILE != 0 or k % _TILE != 0:
        raise ValueError(
            f"GPU FP6 quantizer requires M%{_TILE}==0 and K%{_TILE}==0, got M={m} K={k}"
        )
    from b12x.quantization.mxfp6 import (
        allocate_bf16_to_fp6_tma_outputs,
        compile_bf16_to_fp6_tma,
    )

    launch = compile_bf16_to_fp6_tma(m, k, fmt=fmt, emit="bytes")
    # m and k are 128-aligned by the guard above, so the buffers have no padding
    # row or column and the quantizer writes every byte: the zero-fill would be
    # pure extra HBM traffic on the prefill path (see allocate_...'s zero_init).
    out = allocate_bf16_to_fp6_tma_outputs(
        m, k, device=mat_bf16.device, emit="bytes", zero_init=False
    )
    launch(mat_bf16.contiguous(), global_scale, out.packed_a_flat, out.scale_flat)
    return out.packed_a_storage.view(m, k), out.scale_storage


def _quantize_matrix_fp6_bytes_per_row(
    mat_bf16: torch.Tensor,
    fmt: str,
    w_global_scale: torch.Tensor,
):
    """Quantize a padded ``(m, K)`` bf16 activation with FUSED per-row GS.

    Large-M counterpart of the ``per_row=True`` small-M path. The per-row
    global-scale recipe runs in two kernels:

    1. :func:`~b12x.quantization.mxfp6.fp6_row_gs.compile_fp6_row_gs`
       computes ``gs_r``/``inv_gs_r``/``alpha`` in one bandwidth-bound pass;
    2. the TMA quantizer (``per_row=True``) pre-scales each element through
       bf16 in-registers and quantizes with a unit gs.

    The result matches the host formulation bit-for-bit (same amax semantics,
    div.rn.f32 divisions, cvt.rn.bf16.f32 pre-scale — see the kernel
    docstrings). ``m`` must be 128-padded (TMA contract); zero padding rows
    quantize to the same bytes as the host chain's zero-padded input.

    Returns ``(codes (m, K) uint8, scale_storage flat uint8, alpha (1,) f32,
    inv_gs (m,) bf16)``; callers slice ``inv_gs`` to the true row count.
    """
    m, k = mat_bf16.shape
    if m % _TILE != 0 or k % _TILE != 0:
        raise ValueError(
            f"GPU FP6 quantizer requires M%{_TILE}==0 and K%{_TILE}==0, got M={m} K={k}"
        )
    from b12x.quantization.mxfp6 import (
        allocate_bf16_to_fp6_tma_outputs,
        compile_bf16_to_fp6_tma,
    )
    from b12x.quantization.mxfp6.fp6_row_gs import compile_fp6_row_gs

    device = mat_bf16.device
    x = mat_bf16.contiguous()
    gs_launch = compile_fp6_row_gs(m, k, fmt=fmt)
    gs_pr = torch.empty(m, dtype=torch.float32, device=device)
    inv_gs = torch.empty(m, dtype=torch.bfloat16, device=device)
    alpha = torch.empty(1, dtype=torch.float32, device=device)
    gs_launch(x, w_global_scale, gs_pr, inv_gs, alpha)
    launch = compile_bf16_to_fp6_tma(m, k, fmt=fmt, emit="bytes", per_row=True)
    # Padding-aligned inputs leave no unwritten rows or scale columns, so the
    # output workspace does not need initialization.
    out = allocate_bf16_to_fp6_tma_outputs(
        m, k, device=device, emit="bytes", zero_init=False
    )
    launch(x, gs_pr, out.packed_a_flat, out.scale_flat)
    return out.packed_a_storage.view(m, k), out.scale_storage, alpha, inv_gs


def _quantize_matrix_fp6_bytes_small_m(
    mat_bf16: torch.Tensor,
    fmt: str,
    w_global_scale: torch.Tensor,
    m_pad: int,
    per_row: bool = False,
):
    """Quantize a small-M ``(m, K)`` bf16 activation to byte-container MX-FP6.

    Decode-path counterpart of :func:`_quantize_matrix_fp6_bytes`: identical
    quantization math and output layouts, but only the ``m`` REAL rows are
    computed (the TMA kernel always processes 128, ~99% wasted at M=1) and no
    padded input copy is needed. Buffers are still allocated at ``m_pad`` rows
    so the GEMM's SFA tile reads stay in-bounds; rows ``m..m_pad-1`` are
    unwritten (never read at the true-``m`` GEMM launch).

    The WHOLE global-scale tail is fused into the kernel: it computes the
    activation amax itself (bit-identical to ``vector_norm(x, inf)``), derives
    ``a_gs`` in-registers and ``alpha = 1/(a_gs*w_gs)`` on thread 0 — no amax
    reduce or convert/clamp/div/mul/reciprocal launches on the hot path.

    With ``per_row=True`` the per-row global-scale recipe ALSO runs in-kernel
    (row amax, bf16 pre-scale, unit-gs quantization — see
    :class:`SmallMQuantKernel`), and a 4th value is returned: the ``(m, 1)``
    bf16 per-row output correction for the post-GEMM ``result.mul_``.

    Returns ``(codes (m_pad, K) uint8, scale_storage flat uint8, alpha (1,) f32)``
    plus ``inv_gs (m, 1) bf16`` when ``per_row=True``.
    """
    from b12x.quantization.mxfp6.bf16_to_fp6_small_m import (
        compile_bf16_to_fp6_small_m,
    )

    m, k = mat_bf16.shape
    launch = compile_bf16_to_fp6_small_m(m, k, fmt=fmt, per_row=per_row)
    # The persistent stream-local scratch is safe because the kernel overwrites
    # every real row and the GEMM consumes the result before return, so the
    # buffers are safely reused across layers/steps on the compute stream.
    out, alpha, inv_gs = _small_m_quant_scratch(m_pad, k, mat_bf16.device)
    launch(
        mat_bf16.contiguous(),
        w_global_scale,
        out.packed_a_flat,
        out.scale_flat,
        alpha,
        inv_gs,
    )
    codes = out.packed_a_storage.view(m_pad, k)
    if per_row:
        return codes, out.scale_storage, alpha, inv_gs[:m].view(m, 1)
    return codes, out.scale_storage, alpha


@dataclass
class FP6DenseWeight:
    """A single MX-FP6 quantized dense weight ready for :func:`dense_fp6_linear`.

    ``packed`` holds ``(out_features, 3*in_features//4)`` FP6 codes and
    ``scale_storage`` is the flat swizzled UE8M0 block-scale buffer; the 6D scale
    view ``dense_gemm`` wants is rebuilt on demand via :meth:`scale_view`.

    ``fmt`` is the *weight* sub-format (``e2m3`` / ``e3m2``). ``act_fmt`` is the
    runtime activation sub-format (``e2m3`` / ``e3m2`` / ``e4m3``); defaults to
    the weight format for W6A6, and is ``e4m3`` for W6A8 checkpoints.
    """

    packed: torch.Tensor
    scale_storage: torch.Tensor
    global_scale: torch.Tensor
    out_features: int
    in_features: int
    fmt: str
    act_fmt: str = ""
    # Full, unsharded output width used to keep packed-B routing consistent
    # across tensor-parallel shard counts. Zero uses the local output width.
    out_features_unsharded: int = 0

    def __post_init__(self) -> None:
        # Cached 1-byte-per-code expansion of ``packed`` (the dense_gemm RHS).
        # Not a dataclass field so it is excluded from ``fields()``/save and is
        # rebuilt lazily after ``to()`` (which constructs a fresh instance).
        self._packed_expanded: Optional[torch.Tensor] = None
        if not self.act_fmt:
            self.act_fmt = self.fmt

    @property
    def ab_dtype(self) -> str:
        return f"float6_{self.fmt}fn"

    def expanded_weight(self) -> torch.Tensor:
        """Weight unpacked once into FP8-container bytes for ``dense_gemm``.

        The static weight never changes, so the packed FP6 -> FP8-byte expansion
        is cached. The result is the ``(N, K, 1)`` K-major view that
        ``dense_gemm`` consumes directly when ``b_preexpanded=True``.
        """
        if self._packed_expanded is None:
            # Lazy import: avoids any import cycle with ``b12x._lib.dense_gemm``.
            from b12x._lib.dense_gemm import _expand_packed_mxfp6_ab

            self._packed_expanded = _expand_packed_mxfp6_ab(
                self.packed.unsqueeze(-1), self.in_features
            )
        return self._packed_expanded

    @property
    def use_packed_gemm(self) -> bool:
        """Whether the GEMM should stream the packed codes directly.

        Wide logical matrices use packed storage once their unsharded output
        width reaches ``PACKED_GEMM_MIN_N``. Using the logical width keeps the
        route stable when tensor parallelism changes the local shard width.
        """
        n = self.out_features_unsharded or self.out_features
        return n >= PACKED_GEMM_MIN_N

    def gemm_weight(self) -> torch.Tensor:
        """The B operand ``dense_fp6_linear`` should pass to the GEMM.

        Packed ``(N, 3K/4)`` codes for wide-N weights, the cached expanded
        ``(N, K, 1)`` view otherwise. ``dense_fp6_linear_expanded`` detects the
        format from the K extent.
        """
        if self.use_packed_gemm:
            return self.packed
        return self.expanded_weight()

    def scale_view(self) -> torch.Tensor:
        return as_grouped_mxfp6_scale_view(
            self.scale_storage.view(1, -1), self.out_features, self.in_features
        )

    def to(self, device: torch.device | str) -> "FP6DenseWeight":
        dev = torch.device(device)
        return FP6DenseWeight(
            packed=self.packed.to(dev),
            scale_storage=self.scale_storage.to(dev),
            global_scale=self.global_scale.to(dev),
            out_features=self.out_features,
            in_features=self.in_features,
            fmt=self.fmt,
            act_fmt=self.act_fmt,
            out_features_unsharded=self.out_features_unsharded,
        )


def quantize_dense_weight_to_fp6(
    weight_bf16: torch.Tensor,
    *,
    source_format: str = "mxfp6_w6a8",
) -> FP6DenseWeight:
    """Quantize a BF16 linear weight ``(out_features, in_features)`` to MX-FP6.

    The weight layout matches ``torch.nn.Linear.weight`` (rows = out_features,
    cols = in_features) and is used directly as the ``dense_gemm`` RHS — no
    transpose. ``in_features`` and ``out_features`` must both be multiples of 128.
    """
    if weight_bf16.ndim != 2:
        raise ValueError(
            f"weight must be rank-2 (out_features, in_features), got {tuple(weight_bf16.shape)}"
        )
    from b12x.quantization.mxfp6.fp6_checkpoint import (
        activation_format_for_source,
    )

    n, k = weight_bf16.shape
    fmt = _weight_fmt_for_source(source_format)
    act_fmt = activation_format_for_source(source_format)
    w = weight_bf16.to(torch.bfloat16)
    # Pure-MX contract, matching the safetensors/vLLM export
    # (quantize_linear_to_fp6): unit global scale, the UE8M0 block scales
    # carry the whole dynamic range. Alpha then reduces to 1/a_gs — exactly
    # what serving computes from the on-disk unit weight_scale_2 — so tests
    # and benches built on this path validate the served math.
    gs = torch.ones(1, dtype=torch.float32, device=w.device)
    packed, scale_storage = _quantize_matrix_fp6(w, fmt, gs)
    return FP6DenseWeight(
        packed=packed,
        scale_storage=scale_storage,
        global_scale=gs,
        out_features=n,
        in_features=k,
        fmt=fmt,
        act_fmt=act_fmt,
    )


def dense_fp6_linear_expanded(
    x_bf16: torch.Tensor,
    weight: torch.Tensor,
    scale_storage: torch.Tensor,
    global_scale: torch.Tensor,
    fmt: str,
    out_features: int,
    in_features: int,
    *,
    out: Optional[torch.Tensor] = None,
    act_fmt: Optional[str] = None,
) -> torch.Tensor:
    """Tensor-level FP6 dense linear over a raw weight tensor.

    Same computation as :func:`dense_fp6_linear` but takes the weight as raw
    tensors so it can be wrapped as an opaque ``torch.library.custom_op`` whose
    arguments are all tensors/ints/strings. ``weight`` is either the cached
    1-byte/code expansion (K extent ``in_features``) or the 3:4-packed codes
    (K extent ``3*in_features//4``, streamed natively via ``b_packed=True``);
    the format is detected from the shape.

    ``fmt`` is the *weight* sub-format. ``act_fmt`` is the runtime activation
    sub-format (defaults to ``fmt`` for W6A6; pass ``"e4m3"`` for W6A8).
    """
    if x_bf16.ndim != 2:
        raise ValueError(f"x must be rank-2 (M, K), got {tuple(x_bf16.shape)}")
    m, k = x_bf16.shape
    if k != in_features:
        raise ValueError(
            f"in_features mismatch: x K={k}, weight in_features={in_features}"
        )
    w_k = weight.shape[1]
    if w_k == in_features:
        b_packed = False
    elif w_k * 4 == in_features * 3:
        b_packed = True
    else:
        raise ValueError(
            f"weight K extent {w_k} matches neither expanded ({in_features}) "
            f"nor packed ({in_features * 3 // 4}) layout"
        )
    if weight.ndim == 2:
        weight = weight.unsqueeze(-1)
    n = out_features
    x = x_bf16.to(torch.bfloat16)
    a_fmt = act_fmt if act_fmt is not None else fmt

    m_pad = ((m + _TILE - 1) // _TILE) * _TILE
    # The fused prologue derives one global scale and is therefore valid only
    # for a single activation row.
    _fused_quant = _DENSE_FUSED_QUANT and m == 1
    device = x.device

    # ---- Per-row activation global scale (unfused paths) ----
    # FP6 per-tensor amax makes the output batch-composition-dependent:
    # if vLLM chunks a prefill differently between launches each chunk
    # sees a different amax → different KLD.  Per-row scaling makes each
    # row's quantization independent of the batch (same as BF16 cuBLAS).
    #
    # 1. Compute per-row amax and global scale   → (m, 1) float32
    # 2. Pre-scale x so each row's max ≈ mx_gs_numerator
    # 3. Quantize with a unit activation global scale (1.0)
    # 4. GEMM (alpha accounts only for the quantizer's internal gs and w_gs)
    # 5. Post-multiply by per-row correction to undo the pre-scaling
    #
    # The BF16 pre-scale and output correction deliberately preserve both
    # rounding points. Small-M performs the recipe in one quantizer; large-M
    # uses RowGsKernel followed by the per-row TMA quantizer.
    _per_row = _DENSE_PER_ROW_GS and m > 0
    _per_row_in_kernel = _per_row and not _fused_quant and _PER_ROW_IN_KERNEL
    if _per_row and not _per_row_in_kernel:
        _num = mx_gs_numerator(a_fmt)
        a_amax_pr = x.abs().amax(dim=1, keepdim=True).float()      # (m, 1)
        # f64 division followed by an f32 cast matches div.rn.f32 in the GPU
        # kernels and keeps the BF16 pre-scale bit-identical.
        a_gs_pr = (
            _num / a_amax_pr.clamp_min_(1e-6).double()
        ).float()                                                  # (m, 1)
        x = (x.float() * a_gs_pr).to(torch.bfloat16)               # pre-scaled
        _gs_unit = torch.ones(1, dtype=torch.float32, device=device)
    else:
        a_gs_pr = None
    inv_gs_pr = None

    if _fused_quant:
        # The producer warp quantizes directly in shared memory with an
        # in-kernel per-tensor scale. At m=1 with per-row scaling the
        # host pre-scale above already ran, so the kernel's amax scan sees
        # exactly mx_gs_numerator and its gs/alpha collapse to 1.0 / 1/w_gs
        # — the same arithmetic as the unfused per-row m=1 path.
        a_amax = x.float().abs().amax().reshape(1)
        a_gs = mx_gs_numerator(a_fmt) / a_amax.clamp_min_(1e-6)
        w_gs = global_scale.to(torch.float32).reshape(1)
        alpha = 1.0 / (a_gs * w_gs)
        if _PERSISTENT_SCRATCH:
            _scratch, _, _ = _small_m_quant_scratch(m_pad, k, device)
            a_codes = _scratch.packed_a_storage.view(m_pad, k)
            a_scale = _scratch.scale_storage
        else:
            a_codes = torch.zeros(
                m_pad, k, dtype=torch.uint8, device=device
            )
            a_scale = torch.zeros(
                m_pad * k // 32, dtype=torch.uint8, device=device
            )
    elif m <= _SMALL_M_QUANT_MAX:
        if _per_row_in_kernel:
            # The kernel computes row amax, applies the BF16 pre-scale,
            # quantizes with a unit scale, and emits the output correction.
            a_codes, a_scale, alpha, inv_gs_pr = _quantize_matrix_fp6_bytes_small_m(
                x, a_fmt, global_scale, m_pad, per_row=True
            )
        else:
            # For _per_row (m >= 1): x is already pre-scaled, so the kernel's
            # in-kernel gs is exactly 1.0 (the pre-scaled row amax is exactly
            # mx_gs_numerator) and its alpha = 1/(1 * w_gs) matches the padded
            # TMA branch's torch.reciprocal(gs_unit * global_scale) bit-for-bit.
            # Passing global_scale also keeps alpha correct when per-row
            # scaling is disabled.
            a_codes, a_scale, alpha = _quantize_matrix_fp6_bytes_small_m(
                x, a_fmt, global_scale, m_pad
            )
    else:
        if m_pad != m:
            x_pad = torch.zeros(m_pad, k, dtype=torch.bfloat16, device=device)
            x_pad[:m].copy_(x)
            x = x_pad
        if _per_row_in_kernel:
            # Large-M per-row scaling uses RowGsKernel followed by the TMA
            # quantizer's per-row mode.
            # Zero padding rows get a finite gs (amax clamped to 1e-6) and
            # pre-scale to exactly 0.0, matching the host chain's padded
            # zeros; their inv_gs entries are sliced away below.
            a_codes, a_scale, alpha, inv_gs_pr = _quantize_matrix_fp6_bytes_per_row(
                x, a_fmt, global_scale
            )
            inv_gs_pr = inv_gs_pr[:m].view(m, 1)
        elif _per_row:
            # x is already pre-scaled; pass unit activation global scale.
            a_codes, a_scale = _quantize_matrix_fp6_bytes(x, a_fmt, _gs_unit)
            alpha = torch.reciprocal(_gs_unit * global_scale)
        else:
            a_amax_raw = torch.linalg.vector_norm(
                x, ord=float("inf")
            ).reshape(1)
            # f64 divide + cast = correctly-rounded f32 division; keeps this
            # per-tensor fallback bit-consistent with the small-M kernel's
            # in-kernel div.rn gs (see the per-row comment above).
            a_gs = (
                mx_gs_numerator(a_fmt)
                / a_amax_raw.to(torch.float32).clamp_min_(1e-6).double()
            ).float()
            a_codes, a_scale = _quantize_matrix_fp6_bytes(x, a_fmt, a_gs)
            alpha = torch.reciprocal(a_gs * global_scale)
    a_sf = as_grouped_mxfp6_scale_view(a_scale.view(1, -1), m_pad, k)
    b_sf = as_grouped_mxfp6_scale_view(scale_storage.view(1, -1), n, k)
    # The GEMM runs at the TRUE row count: padding above exists only because the
    # TMA quantizer needs M%128. Slicing the activation codes to ``m`` rows
    # before the call lets dense_gemm's small-M decode tile fire (m==1 ->
    # (16,128) instead of a 128-row tile per weight column block). The kernel
    # takes ``m`` at launch; SFA stays the full padded block, which covers all
    # read tiles. Padded rows are never computed or written. The activation is
    # a byte-container (the quantizer emits it directly); the weight is either
    # pre-expanded at load or 3:4-packed and expanded in smem by the kernel.
    y = torch.empty((m, n, 1), device=x.device, dtype=torch.bfloat16)
    # ``inv_gs_pr`` is a contiguous bf16 (m,) buffer that both per-row quant
    # kernels write; the (m, 1) view exists only for the broadcast multiply.
    _row_scale = (
        inv_gs_pr.view(m)
        if _ROW_SCALE_EPILOGUE and inv_gs_pr is not None
        else None
    )
    dense_gemm(
        (a_codes[:m].unsqueeze(-1), a_sf),
        (weight, b_sf),
        alpha=alpha,
        ab_dtype=f"float6_{fmt}fn",
        sf_dtype="float8_e8m0fnu",
        sf_vec_size=SF_VEC_SIZE_FP6,
        c_dtype="bfloat16",
        out=y,
        a_preexpanded=True,
        b_preexpanded=not b_packed,
        b_packed=b_packed,
        a_fmt=a_fmt,
        b_fmt=fmt,
        x_bf16=x if _fused_quant else None,
        w_gscale=(
            global_scale.to(torch.float32).reshape(1)
            if _fused_quant
            else None
        ),
        row_scale=_row_scale,
    )
    result = y[:, :, 0]
    if inv_gs_pr is not None and _row_scale is None:
        # The quantizer emitted BF16(1/a_gs_per_row). The epilogue applies the
        # same multiply when ``_row_scale`` is present.
        result.mul_(inv_gs_pr)
    elif a_gs_pr is not None:
        # Undo per-row pre-scaling: multiply by 1 / a_gs_per_row.
        # The GEMM alpha already handled the quantizer's internal gs and w_gs;
        # this step undoes only the per-row activation scaling. f64 reciprocal
        # + f32 cast = correctly-rounded f32 division, matching the kernel's
        # div.rn-computed inv_gs bit-for-bit (see the a_gs_pr comment above).
        result.mul_(
            (1.0 / a_gs_pr[:m].double()).float().to(torch.bfloat16)
        )
    if out is not None:
        out.copy_(result)
        return out
    return result


def dense_fp6_linear(
    x_bf16: torch.Tensor,
    weight: FP6DenseWeight,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute ``y = x @ W.T`` in MX-FP6 for a pre-quantized dense weight.

    ``x_bf16`` is ``(M, in_features)`` bf16; the activation is quantized to
    ``weight.act_fmt`` on the fly (E4M3 for W6A8, else the weight format).
    ``M`` is padded to a multiple of 128 *for the quantizer only*; the GEMM
    itself runs at the true ``M`` (decode uses the small-M tile). Returns
    ``(M, out_features)`` bf16.
    """
    return dense_fp6_linear_expanded(
        x_bf16,
        weight.gemm_weight(),
        weight.scale_storage,
        weight.global_scale,
        weight.fmt,
        weight.out_features,
        weight.in_features,
        out=out,
        act_fmt=weight.act_fmt,
    )


def save_fp6_dense_weight(weight: FP6DenseWeight, path: str) -> None:
    """Serialize a :class:`FP6DenseWeight` to ``path`` as safetensors.

    Tensors are stored on CPU; non-tensor fields travel as JSON-encoded
    safetensors metadata. Pickle (.pt) persistence is intentionally not
    supported (project policy: safetensors only).
    """
    import json

    from safetensors.torch import save_file

    tensors: dict[str, torch.Tensor] = {}
    # historical schema id; do not rename (existing artifacts carry it)
    metadata = {"__format__": "b12x_fp6_dense_weight_v1"}
    # fields()+getattr instead of asdict(): asdict deep-copies every field,
    # cloning the (potentially on-GPU) packed tensors before the .cpu() move.
    for f in fields(weight):
        value = getattr(weight, f.name)
        if isinstance(value, torch.Tensor):
            tensors[f.name] = value.detach().cpu().contiguous()
        else:
            metadata[f.name] = json.dumps(value)
    save_file(tensors, path, metadata=metadata)


def load_fp6_dense_weight(
    path: str, *, device: torch.device | str = "cuda"
) -> FP6DenseWeight:
    """Load a weight saved by :func:`save_fp6_dense_weight` onto ``device``."""
    import json

    from safetensors.torch import safe_open

    fields: dict[str, object] = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
        for key in f.keys():
            fields[key] = f.get_tensor(key)
    for key, value in metadata.items():
        if key == "__format__":
            continue
        fields[key] = json.loads(value)
    return FP6DenseWeight(**fields).to(device)
