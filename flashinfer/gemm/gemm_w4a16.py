# SPDX-FileCopyrightText: Copyright (c) 2025 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""W4A16 FP4 GEMM with backend-tagged weight preparation.

Public API
==========

.. code-block:: python

    # 1) Prepare weights for a specific backend (one-time, at model load).
    b_p, sf_p, alpha_p = flashinfer.prepare_w4a16_fp4_weights(
        b, b_descale, alpha, backend="cute-dsl",
    )

    # 2) Run the GEMM, passing the *same* backend tag.
    out = flashinfer.mm_w4a16_fp4(
        a, b_p, sf_p, alpha_p, backend="cute-dsl",
    )

The contract: the ``backend`` string must match between prepare and
compute.  Each backend is free to choose any internal layout for ``b``
/ ``b_descale`` -- the shapes/dtypes returned by
``prepare_w4a16_fp4_weights`` are whatever that backend's compute step
consumes, and mixing prep from one backend with compute from another
is undefined.

Supported backends
------------------

* ``"cudnn"`` -- cuDNN FP4 GEMM graph (bf16 activation x FP4 weight).
  Reuses ``mm_fp4``'s cuDNN machinery and the autotuner; requires a
  cuDNN build with FP4 block-scale support on Blackwell-class GPUs.
  ``prepare_w4a16_fp4_weights`` unswizzles the SF into a linear layout;
  the weight bytes are passed through.
* ``"cute-dsl"`` -- a Blackwell-class (SM100/103/110/120/121) cute-DSL kernel
  (``BlackwellDenseGemmW4A16Kernel``).  ``prepare_w4a16_fp4_weights``
  Marlin-repacks the weight into ``(K//16, N*2)`` int32 and unswizzles
  the SF into a linear ``(K//16, N)`` byte tensor.  The kernel
  dequantizes the FP4 weight to bf16 for the tensor-core matmul (fp32
  accumulate), matching the cuDNN backend's numerics.

Adding a new backend
====================

To add a backend (e.g. ``"cudnn"``), implement two private functions
matching the contract below and wire them into the dispatchers.  All
touchpoints live in this file plus two callsites in the test / bench
harness:

1. Implement ``_prepare_<name>`` and ``_compute_<name>`` (see contract
   below).  Put them either in this file or in a sibling module that
   this file imports.
2. In :func:`prepare_w4a16_fp4_weights`:
     - extend the ``backend: Literal[...]`` annotation to include
       the new name;
     - add ``if backend == "<name>": return _prepare_<name>(...)`` above
       the trailing ``raise ValueError``;
     - update the ``"Supported: ..."`` string in the error.
3. Do the same three edits in :func:`mm_w4a16_fp4`.
4. Add a ``_<name>_w4a16_fp4_requirement`` function (same signature as
   :func:`_cudnn_w4a16_fp4_requirement`) and register it in the
   ``@backend_requirement({...})`` dict above :func:`mm_w4a16_fp4`.
   Decorate it with ``@supported_compute_capability([...])`` listing the
   SMs your backend supports -- this powers
   ``mm_w4a16_fp4.is_backend_supported("<name>", cc)``.
5. In ``tests/gemm/test_mm_w4a16_fp4.py`` append ``"<name>"`` to
   ``ALL_BACKENDS`` -- the full numeric + behaviour grid will run
   against the new backend automatically (tolerance is ``rtol=atol=
   5e-3``; tighten or loosen near the top of the file if needed).
6. In ``benchmarks/routines/gemm.py`` append ``"<name>"`` to the
   ``--backends`` ``choices=[...]`` list so the benchmark CLI accepts
   it.

Backend function contract
-------------------------

.. code-block:: python

    _prepare_<name>(
        b: torch.Tensor,                 # (N, K//2) uint8, two FP4 codes/byte
        b_descale: torch.Tensor,         # 128x4-swizzled FP8-E4M3 SF
        alpha: Optional[torch.Tensor],   # (1,) fp32, or None for alpha=1.0
        block_size: int,                 # always 16 for FP4
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # Returns (b_prepared, b_descale_prepared, alpha_prepared) in
        # whatever layout your compute kernel consumes.  Shapes / dtypes
        # are opaque to the caller -- they only round-trip back into
        # ``_compute_<name>``.  May fold ``alpha`` into ``b_descale``
        # and return alpha_prepared=None.

    _compute_<name>(
        a: torch.Tensor,                 # (M, K) bfloat16
        b: torch.Tensor,                 # exactly what _prepare_<name> returned
        b_descale: torch.Tensor,         # ditto
        alpha: Optional[torch.Tensor],   # ditto
        out_dtype: torch.dtype,          # bfloat16 or float16
        out: Optional[torch.Tensor],     # preallocated (M, N) of out_dtype, or None
        block_size: int,                 # always 16 for FP4
    ) -> torch.Tensor:
        # Returns (M, N) tensor of out_dtype.  Semantics:
        #   out = (a @ dequant(b).T) * alpha
        # using an fp32 accumulator.  If ``out`` is provided, write into
        # it and return it.  Basic input validation (a.dim, a.dtype) has
        # already happened in the dispatcher; backend-specific shape /
        # alignment checks are the backend's responsibility.

See :func:`_prepare_cudnn` / :func:`_compute_cudnn` (or the cute-DSL
pair) below for working implementations of these two functions.
"""

import functools
from typing import List, Literal, Optional, Tuple, cast

import torch

from ..api_logging import flashinfer_api
from ..autotuner import (
    AutoTuner,
    ConstraintSpec,
    DynamicTensorSpec,
    OptimizationProfile,
    TunableRunner,
    TuningConfig,
)
from ..fused_moe.utils import (
    get_hybrid_num_tokens_buckets,
    map_to_hybrid_bucket_uncapped,
)
from ..utils import (
    _get_cache_buf,
    backend_requirement,
    get_native_fp4_dtype,
    supported_compute_capability,
)

# cuDNN backend plumbing is shared with ``mm_fp4`` -- reuse the helpers
# defined in ``gemm_base`` rather than re-implementing the graph machinery.
# ``gemm_base`` is fully imported before this module (see gemm/__init__.py),
# so this sibling import is safe and non-circular.
from .gemm_base import (
    CUDNN_AVAILABLE,
    DEFAULT_WORKSPACE_SIZE,
    UIDs,
    _TORCH_TO_CUTLASS_DTYPE_ATTR,
    _check_cudnn_fp4_availability,
    _check_cute_dsl_availability,
    _get_cudnn_handle,
    _get_cudnn_override_shape_workspace_size,
    _get_cudnn_workspace_size,
    _torch_data_type_to_cudnn_data_type,
    is_cudnn_override_shape_available,
)

if CUDNN_AVAILABLE:
    import cudnn


# Earliest cuDNN backend version supporting W4A16 (bf16 activation x FP4
# weight) GEMM.  Encoded as MMmmpp (e.g. 9.23.1 -> 92301), matching
# ``cudnn.backend_version()``.
_CUDNN_W4A16_MIN_BACKEND_VERSION = 92301


# =============================================================================
# Backend requirement checks (used by @backend_requirement on mm_w4a16_fp4)
# =============================================================================


def _check_mm_w4a16_fp4_problem_size(
    a: torch.Tensor,
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    *,
    backend: Literal["cudnn", "cute-dsl"],
    out_dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
    block_size: int = 16,
    is_sf_swizzled: bool = True,
    enable_pdl: bool = True,
):
    """Common problem-size / dtype checks applied to every backend.

    Mirrors the inline validation that used to live in ``mm_w4a16_fp4``'s
    body -- moved here so :func:`backend_requirement` can run it before
    dispatch (and expose ``mm_w4a16_fp4.is_backend_supported(...)`` etc.).
    """
    if a.dim() != 2:
        raise ValueError(f"a must be 2-D (M, K); got shape {tuple(a.shape)}")
    if a.dtype != torch.bfloat16:
        raise TypeError(
            f"a must be bfloat16; got {a.dtype}.  fp16 support is not implemented yet."
        )
    if out_dtype is not None and out_dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"out_dtype must be bfloat16 or float16; got {out_dtype}")
    if block_size != 16:
        raise ValueError(f"block_size must be 16 for FP4; got {block_size}")
    return True


@supported_compute_capability([100, 103, 110, 120, 121])
def _cudnn_w4a16_fp4_requirement(
    a: torch.Tensor,
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    *,
    backend: Literal["cudnn", "cute-dsl"],
    out_dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
    block_size: int = 16,
    is_sf_swizzled: bool = True,
    enable_pdl: bool = True,
):
    """cuDNN backend: requires a cuDNN build with FP4 block-scale support.

    The W4A16 GEMM reuses ``mm_fp4``'s cuDNN FP4 graph machinery (the
    only difference is that ``a`` stays bf16 instead of being an FP4
    operand), so it inherits the same FP4 minimum-version requirements.
    On top of that, W4A16 (bf16 activation x FP4 weight) was first
    supported by the cuDNN backend in 9.11, so we gate on that.

    cuDNN does not support the 128x4-swizzled SF layout, so it requires a
    non-swizzled (linear) SF (``is_sf_swizzled=False``).
    """
    if is_sf_swizzled:
        raise ValueError(
            "cudnn backend does not support the 128x4-swizzled SF layout; "
            "pass a non-swizzled (linear) SF with is_sf_swizzled=False "
            "(prepare_w4a16_fp4_weights(..., backend='cudnn') produces it)."
        )
    _check_cudnn_fp4_availability()

    backend_version = cudnn.backend_version()
    if backend_version < _CUDNN_W4A16_MIN_BACKEND_VERSION:
        raise RuntimeError(
            f"cuDNN W4A16 FP4 GEMM requires backend version >= "
            f"{_CUDNN_W4A16_MIN_BACKEND_VERSION} (9.11.0), found {backend_version}. "
            f"Please upgrade cuDNN: pip install --upgrade nvidia-cudnn-cu12 "
            f"nvidia-cudnn-frontend"
        )
    return True


@supported_compute_capability([100, 103, 110, 120, 121])
def _cute_dsl_w4a16_fp4_requirement(
    a: torch.Tensor,
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    *,
    backend: Literal["cudnn", "cute-dsl"],
    out_dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
    block_size: int = 16,
    is_sf_swizzled: bool = True,
    enable_pdl: bool = True,
):
    """cute-DSL backend: a Blackwell-class (SM100/103/110/120/121) cute-DSL kernel.

    The kernel consumes a fully backend-specific weight layout produced by
    ``prepare_w4a16_fp4_weights(..., backend='cute-dsl')``: a Marlin-packed
    ``(K // 16, N * 2)`` int32 weight and a linear ``(K // 16, N)``
    FP8-E4M3 scale factor.  Because the prepared SF is neither the
    canonical 128x4-swizzled layout nor the cuDNN linear ``(K_sf, N)``
    layout, the ``is_sf_swizzled`` flag does not apply here and is
    ignored -- the prepared tensors are opaque and only round-trip back
    into the cute-DSL compute path.
    """
    _check_cute_dsl_availability()
    return True


# =============================================================================
# Public dispatchers
# =============================================================================


def prepare_w4a16_fp4_weights(
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    *,
    backend: Literal["cudnn", "cute-dsl"],
    block_size: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Prepare FP4 W4A16 weights for a specific backend.

    The caller is expected to start with weights in the canonical format
    that :func:`flashinfer.nvfp4_quantize` produces with
    ``sfLayout=layout_128x4``:

    * ``b`` is ``(N, K // 2)`` ``uint8`` with two FP4 codes packed per
      byte (low nibble = K=2i, high nibble = K=2i+1).
    * ``b_descale`` is the 128x4-swizzled FP8-E4M3 per-block scales,
      either as a 1-D byte buffer or a 2-D tensor.

    Each backend transforms these into whatever layout its compute kernel
    expects.  The returned ``(b, b_descale, alpha)`` tuple must be passed
    back to :func:`mm_w4a16_fp4` with the *same* ``backend`` -- the
    shapes / dtypes may not match other backends' expectations.

    Args:
        b: ``(N, K // 2)`` ``uint8`` packed FP4 weight.
        b_descale: 128x4-swizzled FP8-E4M3 scale factors from
            ``nvfp4_quantize``.  Either 1-D byte buffer or 2-D tensor.
        alpha: Optional ``(1,) float32`` global scalar.  Pass ``None``
            (default) for implicit ``alpha=1.0``.  Some backends fold
            ``alpha`` into ``b_descale`` at prep time and return
            ``alpha=None``; callers should forward whatever this function
            returns unchanged.
        backend: Identifier of a supported backend (``"cudnn"`` or
            ``"cute-dsl"``).
        block_size: SF block size.  Always 16 for FP4.

    Returns:
        ``(b_prepared, b_descale_prepared, alpha_prepared)`` -- pass all
        three to :func:`mm_w4a16_fp4` with the same ``backend``.

    Raises:
        ValueError: ``backend`` is unknown, or the inputs fail basic
            shape / dtype checks.
    """
    if b.dim() != 2:
        raise ValueError(f"b must be 2-D (N, K/2); got shape {tuple(b.shape)}")
    if b.dtype != torch.uint8:
        raise TypeError(f"b must be uint8; got {b.dtype}")
    k = int(b.shape[1]) * 2
    if k % block_size != 0:
        raise ValueError(f"K={k} must be a multiple of block_size={block_size}")
    if alpha is not None:
        if alpha.dim() != 1 or alpha.shape[0] != 1:
            raise ValueError(f"alpha must be shape (1,); got {tuple(alpha.shape)}")
        if alpha.dtype != torch.float32:
            raise TypeError(f"alpha must be float32; got {alpha.dtype}")
    if backend == "cudnn":
        return _prepare_cudnn(b, b_descale, alpha, block_size)
    if backend == "cute-dsl":
        return _prepare_cute_dsl(b, b_descale, alpha, block_size)
    raise ValueError(f"Unknown backend {backend!r}.  Supported: 'cudnn', 'cute-dsl'.")


@backend_requirement(
    {
        "cudnn": _cudnn_w4a16_fp4_requirement,
        "cute-dsl": _cute_dsl_w4a16_fp4_requirement,
    },
    common_check=_check_mm_w4a16_fp4_problem_size,
)
@flashinfer_api
def mm_w4a16_fp4(
    a: torch.Tensor,
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    *,
    backend: Literal["cudnn", "cute-dsl"],
    out_dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
    block_size: int = 16,
    is_sf_swizzled: bool = True,
    enable_pdl: bool = True,
) -> torch.Tensor:
    """W4A16 FP4 GEMM: ``out = (a @ dequant(b).T) * alpha``.

    ``b``, ``b_descale``, and ``alpha`` must be the tensors returned by
    :func:`prepare_w4a16_fp4_weights` with the same ``backend``.  Mixing
    backends (preparing with one, computing with another) is undefined.

    Args:
        a: ``(M, K)`` activation matrix in ``torch.bfloat16``.  This is
            the only currently supported activation dtype; fp16 support
            can be added when needed.
        b: Prepared weight tensor (backend-specific layout).
        b_descale: Prepared scale-factor tensor (backend-specific layout).
        alpha: Optional ``(1,) float32`` global scalar.  Pass through
            whatever ``prepare_w4a16_fp4_weights`` returned -- it may be
            ``None`` if the backend folded it into ``b_descale``.
        backend: Same identifier passed to ``prepare_w4a16_fp4_weights``.
        out_dtype: Output dtype.  Defaults to ``a.dtype`` (``bfloat16``).
        out: Optional preallocated ``(M, N)`` output tensor.
        block_size: SF block size.  Always 16 for FP4.
        is_sf_swizzled: Whether ``b_descale`` is in the canonical
            128x4-swizzled SF layout.  Used by the swizzle-sensitive
            backends: ``True`` (default) requires the ``torch`` backend
            (cuDNN does not support swizzled SF), while ``False`` (a
            non-swizzled / linear SF, as produced by
            ``prepare_w4a16_fp4_weights(..., backend="cudnn")``) requires
            the ``cudnn`` backend.  The ``cute-dsl`` backend consumes a
            fully bespoke prepared layout and ignores this flag.
        enable_pdl: Enable `Programmatic Dependent Launch
            <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization>`_.
            Only the ``cute-dsl`` backend uses it (the kernel launches with
            ``use_pdl`` and emits griddepcontrol bookends so a producer kernel's
            tail can overlap this kernel's prologue in a back-to-back / CUDA-graph
            sequence).  No effect for the ``torch`` / ``cudnn`` backends.

    Returns:
        ``(M, N)`` tensor of ``out_dtype``.

    Notes:
        ``@backend_requirement`` adds two introspection helpers to this
        function: ``mm_w4a16_fp4.is_backend_supported("cute-dsl")`` and
        ``mm_w4a16_fp4.is_compute_capability_supported(cc)``.  It also
        accepts a ``skip_check=True`` kwarg to bypass validation in
        latency-sensitive code paths.
    """
    out_dtype = out_dtype or a.dtype
    # enable_pdl is consumed only by the cute-dsl backend (Programmatic
    # Dependent Launch); cudnn (own graph) ignores it.
    if backend == "cudnn":
        return _compute_cudnn(a, b, b_descale, alpha, out_dtype, out, block_size)
    if backend == "cute-dsl":
        return _compute_cute_dsl(
            a, b, b_descale, alpha, out_dtype, out, block_size, enable_pdl=enable_pdl
        )
    raise ValueError(f"Unknown backend {backend!r}.  Supported: 'cudnn', 'cute-dsl'.")


# =============================================================================
# Reference / shared SF utilities
# =============================================================================
#
# These pure-PyTorch helpers are NOT a compute backend -- they are shared
# infrastructure:
#   * _unswizzle_sf_128x4 is REQUIRED by the cudnn and cute-dsl prepare
#     paths (both consume a non-swizzled SF derived from the canonical
#     128x4-swizzled input).
#   * _E2M1_CODEBOOK_FP32 + _dequantize_w4a16_fp4_torch implement an fp32
#     dequantize used as the ground-truth reference by the unit tests
#     (tests/gemm/test_mm_w4a16_fp4.py) and the benchmark refcheck
#     (benchmarks/routines/gemm.py).  (Named ``_torch`` for the precision
#     it computes in, not a user-facing backend.)


# E2M1 (FP4) codebook, signed.  Codes 0-7 = positives, 8-15 = negatives.
# Matches the layout produced by ``flashinfer.nvfp4_quantize``.
_E2M1_CODEBOOK_FP32 = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def _unswizzle_sf_128x4(sf_swizzled: torch.Tensor, n: int, k_sf: int) -> torch.Tensor:
    """Reverse the 128x4 SF swizzle into a flat ``(N, K_sf)`` byte tensor.

    The swizzle stores SF in 512-byte blocks each holding 128 N-rows x 4
    K_sf-cols.  The byte address of logical ``(n, k_sf)`` is::

        offset = ((n // 128) * sf_pad_blocks + k_sf // 4) * 512
               + (n % 32) * 16
               + ((n % 128) // 32) * 4
               + (k_sf % 4)

    where ``sf_pad_blocks = ceil(k_sf, 4) // 4`` accounts for K_sf
    padding inside each 128-row N block.
    """
    device = sf_swizzled.device
    sf_flat = sf_swizzled.contiguous().view(torch.uint8).view(-1)
    sf_pad_blocks = (k_sf + 3) // 4  # ceil_div(k_sf, 4)
    n_idx = torch.arange(n, device=device, dtype=torch.int64)
    k_idx = torch.arange(k_sf, device=device, dtype=torch.int64)
    n_grid, k_grid = torch.meshgrid(n_idx, k_idx, indexing="ij")
    offsets = (
        ((n_grid // 128) * sf_pad_blocks + (k_grid // 4)) * 512
        + (n_grid % 32) * 16
        + ((n_grid % 128) // 32) * 4
        + (k_grid % 4)
    )
    return sf_flat[offsets]


def _dequantize_w4a16_fp4_torch(
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor],
    n: int,
    k: int,
    block_size: int,
) -> torch.Tensor:
    """Pure-torch FP4 dequantize -> ``(N, K)`` fp32 weight matrix.

    Steps: unpack FP4 nibbles -> codebook lookup -> per-block SF scale ->
    optional global ``alpha``.
    """
    device = b.device
    k_sf = k // block_size
    lut = torch.tensor(_E2M1_CODEBOOK_FP32, dtype=torch.float32, device=device)

    # Unpack: (N, K/2) uint8 -> (N, K/2, 2) low/high nibbles -> (N, K).
    b_int = b.to(torch.int64)
    code_lo = b_int & 0xF
    code_hi = (b_int >> 4) & 0xF
    codes = torch.stack([code_lo, code_hi], dim=-1).reshape(n, k)
    values = lut[codes]  # (N, K), fp32

    # SF: unswizzle, reinterpret bytes as FP8 E4M3, cast to fp32, broadcast.
    sf_bytes = _unswizzle_sf_128x4(b_descale, n, k_sf)
    sf_fp32 = sf_bytes.view(torch.float8_e4m3fn).to(torch.float32)
    sf_expanded = sf_fp32.repeat_interleave(block_size, dim=1)  # (N, K)

    weight = values * sf_expanded
    if alpha is not None:
        weight = weight * alpha.to(torch.float32)
    return weight  # (N, K) fp32


# =============================================================================
# cuDNN backend
# =============================================================================
#
# W4A16 = bf16 activation (A) x FP4 weight (B).  This is structurally the
# ``mm_fp4`` cuDNN graph with a single change: ``A`` is a plain bf16
# operand instead of an FP4 operand that gets block-scale dequantized.
# Only ``B`` carries an FP4 block-scale dequantize.
#
# Weight buffer layout (no transpose needed):
#   The weight ``b`` is ``(N, K//2)`` uint8 (row-major, 2 FP4 codes/byte),
#   i.e. logically ``(N, K)`` FP4 stored row-major.  cuDNN's matmul
#   computes ``A @ B`` so the second operand must be presented as
#   ``(K, N)``.  A row-major ``(N, K)`` buffer is bit-identical to a
#   column-major ``(K, N)`` buffer, so we hand cuDNN the same bytes with a
#   ``(batch, K, N)`` column-major stride -- exactly what ``mm_fp4`` does
#   when its caller passes ``mat2_fp4.T``.
#
# Scale-factor layout (NON-swizzled / linear):
#   The cuDNN W4A16 path does NOT support the 128x4-swizzled SF layout that
#   ``mm_fp4`` consumes -- it requires the per-block SF in a plain linear
#   layout.  ``_prepare_cudnn`` unswizzles the canonical 128x4 SF into a
#   linear ``(N, K//block_size)`` FP8 tensor, and the graph declares the SF
#   tensor with ``tensor_reordering.NONE``.  (cuDNN requires
#   ``is_sf_swizzled=False`` on :func:`mm_w4a16_fp4`; the cute-dsl backend
#   consumes its own prepared layout and ignores the flag.)


# Sentinel "cache M" for override-shape graphs (any value works; this one
# covers typical LLM inference shapes).  Kept local to avoid importing a
# private constant from gemm_base.
_OVERRIDE_SHAPE_CACHE_M = 8192


def _w4a16_b_descale_layout(batch, n, k, block_size):
    """Return ``(dim, stride, reordering_type)`` for the B scale-factor tensor.

    The B operand is presented to cuDNN as ``(batch, K, N)``, so its SF is
    presented as ``(batch, K // block_size, N)``.  cuDNN W4A16 only supports
    a non-swizzled (linear) SF: the ``(N, K_sf)`` buffer is presented
    column-major as ``(K_sf, N)`` with ``NONE`` reordering and exact
    (unpadded) dims.
    """
    k_sf = k // block_size
    dim = (batch, k_sf, n)
    stride = (k_sf * n, 1, k_sf)
    return dim, stride, cudnn.tensor_reordering.NONE


def _build_w4a16_fp4_graph_common(
    graph,
    a_cudnn_tensor,
    b_cudnn_tensor,
    block_descale_b_cudnn_tensor,
    a_type,
    o_type,
    block_size,
    alpha_is_not_none,
):
    """Shared graph body: dequant(B) -> A @ dequant(B) -> optional alpha.

    The dequantized B is emitted in ``a_type`` (the activation dtype, e.g.
    bf16) rather than FLOAT: cuDNN's matmul requires both operands to share
    the same input type (``is_input_compute_type_match``), and matching the
    bf16 activation also selects the bf16 tensor-core path instead of the
    slower float path.  Accumulation still happens in FLOAT.
    """
    dequant_b_tensor = graph.block_scale_dequantize(
        b_cudnn_tensor,
        block_descale_b_cudnn_tensor,
        block_size=[block_size, 1],
        name="dequant_b",
    )
    dequant_b_tensor.set_data_type(a_type)

    c_tensor = graph.matmul(
        a_cudnn_tensor,
        dequant_b_tensor,
        compute_data_type=cudnn.data_type.FLOAT,
        name="gemm",
    )
    c_tensor.set_data_type(cudnn.data_type.FLOAT)

    c_final_cudnn_tensor = c_tensor
    if alpha_is_not_none:
        global_scale_cudnn_tensor = graph.tensor(
            name="global_scale",
            dim=(1, 1, 1),
            stride=(1, 1, 1),
            data_type=cudnn.data_type.FLOAT,
        )
        c_final_cudnn_tensor = graph.mul(
            name="scale_mul",
            a=c_tensor,
            b=global_scale_cudnn_tensor,
            compute_data_type=cudnn.data_type.FLOAT,
        )
        global_scale_cudnn_tensor.set_uid(UIDs.ALPHA_UID.value)

    c_final_cudnn_tensor.set_name("c_final").set_output(True).set_data_type(o_type)

    a_cudnn_tensor.set_uid(UIDs.A_UID.value)
    b_cudnn_tensor.set_uid(UIDs.B_UID.value)
    block_descale_b_cudnn_tensor.set_uid(UIDs.BLOCK_DESCALE_B_UID.value)
    c_final_cudnn_tensor.set_uid(UIDs.O_UID.value)
    return c_final_cudnn_tensor


@functools.lru_cache(maxsize=1024)
def build_cudnn_w4a16_fp4_graph(
    batch,
    m,
    n,
    k,
    a_type,
    o_type,
    block_size,
    device,
    alpha_is_not_none,
    use_nvfp4,
    policy=None,
):
    """Build a fixed-shape cuDNN W4A16 FP4 GEMM graph (no override-shape).

    Block-scale dimensions are derived from ``(m, n, k)`` internally (the
    FP8 SF buffer is a flat byte tensor, so its logical shape must come from
    the problem size, not from the tensor's own ``.shape``).  The SF is
    always non-swizzled (see :func:`_w4a16_b_descale_layout`).
    """
    _check_cudnn_fp4_availability()
    if policy is None:
        policy = cudnn.build_plan_policy.HEURISTICS_CHOICE

    scale_type = cudnn.data_type.FP8_E4M3 if use_nvfp4 else cudnn.data_type.FP8_E8M0

    a_shape = (batch, m, k)
    a_stride = (m * k, k, 1)
    # b weight bytes are row-major (N, K); present as column-major (K, N).
    b_shape = (batch, k, n)
    b_stride = (k * n, 1, k)
    b_descale_shape, b_descale_stride, b_descale_reordering = _w4a16_b_descale_layout(
        batch, n, k, block_size
    )

    stream = torch.cuda.current_stream(device)
    with cudnn.graph(_get_cudnn_handle(device, stream)) as (graph, _):
        a_cudnn_tensor = graph.tensor(
            name="a", dim=a_shape, stride=a_stride, data_type=a_type
        )
        b_cudnn_tensor = graph.tensor(
            name="b", dim=b_shape, stride=b_stride, data_type=cudnn.data_type.FP4_E2M1
        )
        block_descale_b_cudnn_tensor = graph.tensor(
            name="block_descale_b",
            dim=b_descale_shape,
            stride=b_descale_stride,
            data_type=scale_type,
            reordering_type=b_descale_reordering,
        )

        _build_w4a16_fp4_graph_common(
            graph,
            a_cudnn_tensor,
            b_cudnn_tensor,
            block_descale_b_cudnn_tensor,
            a_type,
            o_type,
            block_size,
            alpha_is_not_none,
        )

        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph.check_support()
        graph.build_plans(policy)
        return graph


@functools.lru_cache(maxsize=1024)
def build_cudnn_w4a16_fp4_graph_override_shape(
    batch,
    n,
    k,
    a_type,
    o_type,
    block_size,
    device,
    alpha_is_not_none,
    use_nvfp4,
    cache_m: int = _OVERRIDE_SHAPE_CACHE_M,
    policy=None,
):
    """Build a cuDNN W4A16 FP4 GEMM graph with override-shape support.

    The graph is compiled once using ``cache_m`` for the M dimension; the
    real M (and the corresponding A / output shapes) is supplied at execute
    time via ``override_shapes`` / ``override_strides``.  The cache key
    omits M, so a single compiled graph is reused across token counts that
    map to the same tuning bucket.  The SF is always non-swizzled (see
    :func:`_w4a16_b_descale_layout`).
    """
    _check_cudnn_fp4_availability()
    if policy is None:
        policy = cudnn.build_plan_policy.HEURISTICS_CHOICE

    scale_type = cudnn.data_type.FP8_E4M3 if use_nvfp4 else cudnn.data_type.FP8_E8M0

    a_shape = [batch, cache_m, k]
    a_stride = [cache_m * k, k, 1]
    b_shape = [batch, k, n]
    b_stride = [k * n, 1, k]
    b_descale_shape, b_descale_stride, b_descale_reordering = _w4a16_b_descale_layout(
        batch, n, k, block_size
    )

    stream = torch.cuda.current_stream(device)
    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.FLOAT,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=_get_cudnn_handle(device, stream),
        is_override_shape_enabled=True,
    )

    a_cudnn_tensor = graph.tensor(
        name="a", dim=a_shape, stride=a_stride, data_type=a_type
    )
    b_cudnn_tensor = graph.tensor(
        name="b", dim=b_shape, stride=b_stride, data_type=cudnn.data_type.FP4_E2M1
    )
    block_descale_b_cudnn_tensor = graph.tensor(
        name="block_descale_b",
        dim=b_descale_shape,
        stride=b_descale_stride,
        data_type=scale_type,
        reordering_type=b_descale_reordering,
    )

    _build_w4a16_fp4_graph_common(
        graph,
        a_cudnn_tensor,
        b_cudnn_tensor,
        block_descale_b_cudnn_tensor,
        a_type,
        o_type,
        block_size,
        alpha_is_not_none,
    )

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans(policy)
    return graph


def _w4a16_variant_pack(a, b, b_descale, alpha, out):
    """Build the {uid: tensor} variant pack shared by both execute paths."""
    variant_pack = {
        UIDs.A_UID.value: a,
        UIDs.B_UID.value: b.view(get_native_fp4_dtype()),
        UIDs.BLOCK_DESCALE_B_UID.value: b_descale,
        UIDs.O_UID.value: out,
    }
    if alpha is not None:
        variant_pack[UIDs.ALPHA_UID.value] = alpha.view(torch.float)
    return variant_pack


def execute_cudnn_w4a16_fp4_graph(
    graph,
    a,
    b,
    b_descale,
    alpha,
    out,
    workspace_buffer,
    tactic: int = -1,
):
    variant_pack = _w4a16_variant_pack(a, b, b_descale, alpha, out)

    workspace_size = _get_cudnn_workspace_size(graph, tactic)
    if workspace_buffer.numel() < workspace_size:
        workspace_buffer.resize_(workspace_size)

    stream = torch.cuda.current_stream(a.device)
    handle = _get_cudnn_handle(a.device, stream)
    if tactic == -1:
        graph.execute(variant_pack, workspace_buffer, handle=handle)
    else:
        graph.execute_plan_at_index(
            variant_pack, workspace_buffer, tactic, handle=handle
        )


def execute_cudnn_w4a16_fp4_graph_override_shape(
    graph,
    a,
    b,
    b_descale,
    alpha,
    out,
    workspace_buffer,
    block_size: int = 16,
    tactic: int = 0,
):
    """Execute the W4A16 graph, overriding A / output to the real M."""
    m, k = int(a.shape[0]), int(a.shape[1])
    n = int(b.shape[0])
    batch = 1

    a_shape = (batch, m, k)
    a_stride = (m * k, k, 1)
    b_shape = (batch, k, n)
    b_stride = (k * n, 1, k)
    b_descale_shape, b_descale_stride, _ = _w4a16_b_descale_layout(
        batch, n, k, block_size
    )
    out_shape = (batch, m, n)
    out_stride = (m * n, n, 1)

    variant_pack = _w4a16_variant_pack(a, b, b_descale, alpha, out)

    override_uids = [
        UIDs.A_UID.value,
        UIDs.B_UID.value,
        UIDs.BLOCK_DESCALE_B_UID.value,
        UIDs.O_UID.value,
    ]
    override_shapes = [a_shape, b_shape, b_descale_shape, out_shape]
    override_strides = [a_stride, b_stride, b_descale_stride, out_stride]

    stream = torch.cuda.current_stream(a.device)
    cudnn_handle = _get_cudnn_handle(a.device, stream)

    workspace_size = _get_cudnn_override_shape_workspace_size(
        graph, tactic, cudnn_handle, override_uids, override_shapes, override_strides
    )
    if workspace_buffer.numel() < workspace_size:
        workspace_buffer.resize_(workspace_size)

    graph.execute_plan_at_index(
        variant_pack,
        workspace_buffer,
        tactic,
        handle=cudnn_handle,
        override_uids=override_uids,
        override_shapes=override_shapes,
        override_strides=override_strides,
    )


# Autotuner sweeps M (token count) of the bf16 activation ``a`` and keeps
# the output ``out`` in lockstep.  The FP4 weight ``b`` and its scale are
# M-independent and stay fixed during profiling.
#
# inputs layout (see _compute_cudnn):
#   [0]=a  [1]=b  [2]=b_descale  [3]=alpha  [4]=out_dtype
#   [5]=out  [6]=block_size  [7]=use_nvfp4  [8]=workspace_buffer
_W4A16_FP4_TUNING_CONFIG = TuningConfig(
    dynamic_tensor_specs=(
        DynamicTensorSpec(
            (0,),  # a_tensor_index
            (0,),  # M dimension
            get_hybrid_num_tokens_buckets,
            map_to_hybrid_bucket_uncapped,
        ),
    ),
    constraint_specs=(
        ConstraintSpec(
            5,  # out_tensor_index follows M
            0,
            lambda shapes: shapes[0][0],
        ),
    ),
)


def _cudnn_w4a16_fp4_runner(tuning_config):
    """Build a ``CudnnW4a16Fp4Runner`` bound to the active tuning config.

    Mirrors ``_cudnn_gemm_fp4_runner``: the runner derives its ``cache_m``
    mapper from ``AutoTuner.get_effective_map_to_tuning_buckets`` so the
    graph's ``cache_m`` and the autotuner's per-bucket tactic key stay in
    lockstep (otherwise a tactic profiled on one graph would be silently
    applied to a differently-keyed graph at runtime).
    """
    m_bucket_mapper = AutoTuner.get().get_effective_map_to_tuning_buckets(
        tuning_config, spec_idx=0
    )

    class CudnnW4a16Fp4Runner(TunableRunner):
        def __init__(self):
            super().__init__()
            self._m_bucket_mapper = m_bucket_mapper
            self._use_override_shape = is_cudnn_override_shape_available()

        def get_cache_key_extras(self, inputs: List[torch.Tensor]) -> tuple:
            _, _, _, alpha, out_dtype, _, block_size, use_nvfp4, _ = inputs
            return (out_dtype, block_size, use_nvfp4, alpha is not None)

        def _get_override_graph(self, a, b, alpha, out_dtype, block_size, use_nvfp4):
            actual_m, k = int(a.shape[0]), int(a.shape[1])
            n = int(b.shape[0])
            cache_m = self._m_bucket_mapper(actual_m)
            return build_cudnn_w4a16_fp4_graph_override_shape(
                batch=1,
                n=n,
                k=k,
                a_type=_torch_data_type_to_cudnn_data_type(a.dtype),
                o_type=_torch_data_type_to_cudnn_data_type(out_dtype),
                block_size=block_size,
                device=a.device,
                alpha_is_not_none=alpha is not None,
                use_nvfp4=use_nvfp4,
                cache_m=cache_m,
                policy=cudnn.build_plan_policy.ALL,
            )

        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
        ) -> List[int]:
            (
                a,
                b,
                b_descale,
                alpha,
                out_dtype,
                out,
                block_size,
                use_nvfp4,
                workspace_buffer,
            ) = inputs
            if self._use_override_shape:
                graph = self._get_override_graph(
                    a, b, alpha, out_dtype, block_size, use_nvfp4
                )
            else:
                graph = build_cudnn_w4a16_fp4_graph(
                    batch=1,
                    m=int(a.shape[0]),
                    n=int(b.shape[0]),
                    k=int(a.shape[1]),
                    a_type=_torch_data_type_to_cudnn_data_type(a.dtype),
                    o_type=_torch_data_type_to_cudnn_data_type(out_dtype),
                    block_size=block_size,
                    device=a.device,
                    alpha_is_not_none=alpha is not None,
                    use_nvfp4=use_nvfp4,
                    policy=cudnn.build_plan_policy.HEURISTICS_CHOICE,
                )
            return list(range(graph.get_execution_plan_count()))

        def forward(
            self,
            inputs: List[torch.Tensor],
            tactic: int = -1,
            do_preparation: bool = False,
            **kwargs,
        ) -> torch.Tensor:
            (
                a,
                b,
                b_descale,
                alpha,
                out_dtype,
                out,
                block_size,
                use_nvfp4,
                workspace_buffer,
            ) = inputs
            if self._use_override_shape:
                graph = self._get_override_graph(
                    a, b, alpha, out_dtype, block_size, use_nvfp4
                )
                execute_cudnn_w4a16_fp4_graph_override_shape(
                    graph,
                    a,
                    b,
                    b_descale,
                    alpha,
                    out,
                    workspace_buffer,
                    block_size=block_size,
                    tactic=max(tactic, 0),
                )
            else:
                graph = build_cudnn_w4a16_fp4_graph(
                    batch=1,
                    m=int(a.shape[0]),
                    n=int(b.shape[0]),
                    k=int(a.shape[1]),
                    a_type=_torch_data_type_to_cudnn_data_type(a.dtype),
                    o_type=_torch_data_type_to_cudnn_data_type(out_dtype),
                    block_size=block_size,
                    device=a.device,
                    alpha_is_not_none=alpha is not None,
                    use_nvfp4=use_nvfp4,
                    policy=cudnn.build_plan_policy.HEURISTICS_CHOICE,
                )
                execute_cudnn_w4a16_fp4_graph(
                    graph,
                    a,
                    b,
                    b_descale,
                    alpha,
                    out,
                    workspace_buffer,
                    tactic=-1,
                )
            return out

    return CudnnW4a16Fp4Runner()


def _prepare_cudnn(
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor],
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """cuDNN-backend prep.

    The weight bytes ``(N, K//2)`` are already in the layout the cuDNN graph
    consumes (see the module banner).  The scale factor, however, must be
    *non-swizzled* for the cuDNN W4A16 path (cuDNN does not support the
    128x4-swizzled SF layout), so we unswizzle the canonical 128x4 SF into a
    linear ``(N, K // block_size)`` FP8-E4M3 tensor.  Pair the prepared
    tensors with ``mm_w4a16_fp4(..., is_sf_swizzled=False)``.
    """
    n = int(b.shape[0])
    k = int(b.shape[1]) * 2
    k_sf = k // block_size
    # (N, K_sf) uint8 bytes, each byte an FP8-E4M3 per-block scale.
    linear_sf = _unswizzle_sf_128x4(b_descale, n, k_sf).contiguous()
    return b, linear_sf.view(torch.float8_e4m3fn), alpha


def _compute_cudnn(
    a: torch.Tensor,
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out: Optional[torch.Tensor],
    block_size: int,
) -> torch.Tensor:
    """cuDNN-backend compute with autotuning over the M (token) dimension."""
    n = int(b.shape[0])
    k = int(b.shape[1]) * 2
    if a.shape[1] != k:
        raise ValueError(
            f"a.shape[1]={a.shape[1]} but k inferred from b.shape={tuple(b.shape)} "
            f"is {k}"
        )

    if out is None:
        out = torch.empty((a.shape[0], n), device=a.device, dtype=out_dtype)
    else:
        if tuple(out.shape) != (a.shape[0], n):
            raise ValueError(
                f"out shape {tuple(out.shape)} != expected {(a.shape[0], n)}"
            )
        if out.dtype != out_dtype:
            raise TypeError(f"out dtype {out.dtype} != requested out_dtype {out_dtype}")

    workspace_buffer = _get_cache_buf(
        "mm_w4a16_fp4_workspace", DEFAULT_WORKSPACE_SIZE, a.device
    )

    tuning_config = _W4A16_FP4_TUNING_CONFIG
    tuner = AutoTuner.get()
    runner = _cudnn_w4a16_fp4_runner(tuning_config)

    # W4A16 FP4 is always NVFP4 with block_size=16 (validated upstream).
    use_nvfp4 = True
    inputs = [
        a,
        b,
        b_descale,
        alpha,
        out_dtype,
        out,
        block_size,
        use_nvfp4,
        workspace_buffer,
    ]
    chosen_runner, tactic = tuner.choose_one(
        "w4a16_fp4_gemm",
        [runner],
        tuning_config,
        inputs,
    )
    chosen_runner(inputs=inputs, tactic=tactic)
    return out


# =============================================================================
# cute-DSL backend
# =============================================================================
#
# W4A16 = bf16 activation (A) x FP4 weight (B), computed by a Blackwell
# cute-DSL kernel (``BlackwellDenseGemmW4A16Kernel``).  Unlike the cuDNN
# path, the kernel consumes a fully bespoke weight layout, so
# ``_prepare_cute_dsl`` does real work at model-load time:
#
#   * ``b`` -- the canonical ``(N, K//2)`` uint8 packed FP4 weight is
#     transposed to ``(K//2, N)`` (byte-for-byte; the low/high nibble
#     K-semantics are preserved) and Marlin-repacked into a
#     ``(K//16, N*2)`` int32 tensor (128 int32 per 16Kx64N MMA block).
#   * ``b_descale`` -- the canonical 128x4-swizzled SF is unswizzled into
#     a linear ``(N, K//block_size)`` byte tensor and transposed to
#     ``(K//block_size, N)`` -- the per-group scale layout the kernel
#     loads alongside each B tile.
#
# The kernel dequantizes the FP4 weight to the activation dtype (bf16)
# before the tensor-core matmul (accumulation stays fp32), so its numeric
# behaviour matches the cuDNN backend (~1 bf16 ULP vs an fp32 reference).
#
# The kernel targets Blackwell-class GPUs (SM100/103/110/120/121) and is compiled once per
# (tile_shape, a_dtype, c_dtype, atom_layout) via ``cute.compile`` and
# cached in ``_CUTE_DSL_MM_FP4_W4A16_KERNEL_CACHE``.


_W4A16_ALPHA_ONE_CACHE: dict = {}


def _prepare_w4a16_alpha(
    alpha: Optional[torch.Tensor], device: torch.device
) -> torch.Tensor:
    """Normalize ``alpha`` to a ``(1,) float32`` tensor for the kernel.

    ``alpha=None`` means implicit ``1.0``; we cache the per-device unit
    scalar so the hot path doesn't allocate.
    """
    if alpha is None:
        cached = _W4A16_ALPHA_ONE_CACHE.get(device)
        if cached is None:
            cached = torch.tensor([1.0], dtype=torch.float32, device=device)
            _W4A16_ALPHA_ONE_CACHE[device] = cached
        return cached
    if alpha.dim() == 0:
        return alpha.to(torch.float32).unsqueeze(0)
    return alpha.to(torch.float32).reshape(1)


def _select_w4a16_tile_shape(
    m: int, n: int, k: int
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Pick a CTA tile shape AND MMA atom_layout for the cute-DSL W4A16 kernel.

    Returns ``(tile_shape_mnk, atom_layout)``.

    Tile shape selection:
      tile_M choice
        * M <= 16 (and tile_K=128 path): use tile_M=16 with atom_layout
          (1,2,1).  Halves wasted M-rows vs tile_M=32, and a 1-M-warp
          layout removes the duplicate dequant that (2,2,1) suffers from.
        * 16 < M <= 32: use tile_M=32 with atom_layout (2,2,1).  Smaller
          MMA + epilogue waste than tile_M=64.
        * M > 32: use tile_M=64 with atom_layout (2,2,1) -- standard tile,
          more rows to amortize across.

      tile_K choice
        * K % 128 == 0: tile_K=128 (halves K-tile count and barrier
          overhead).
        * Otherwise: tile_K=64.

    Why atom_layout differs:
      * (2,2,1) (default for tile_M >= 32): 4 MMA warps as 2 M x 2 N --
        well-tested cute layout, but the 2 M-warps redundantly dequant
        the same B values into their own register files (~50% waste in
        dequant compute).
      * (1,2,1) (used for tile_M=16): 2 MMA warps as 1 M x 2 N -- no
        M-warp duplication.  Permutation_m = 16, so tile_M must be 16.
    """
    tile_k = 128 if k % 128 == 0 else 64
    if m <= 16 and tile_k == 128:
        return ((16, 64, 128), (1, 2, 1))
    if m <= 32:
        return ((32, 64, tile_k), (2, 2, 1))
    return ((64, 64, tile_k), (2, 2, 1))


_CUTE_DSL_MM_FP4_W4A16_KERNEL_CACHE: dict = {}


def _get_cute_dsl_w4a16_gemm(
    tile_shape_mnk: Tuple[int, int, int],
    a_dtype: torch.dtype,
    c_dtype: torch.dtype,
    atom_layout: Tuple[int, int, int] = (2, 2, 1),
    pipeline_depth: int = 1,
    use_fp16_mma: int = 1,
    enable_pdl: bool = True,
    tile_swizzle: int = 1,
):
    """Compile (or fetch from cache) the cute-DSL W4A16 GEMM kernel.

    The compiled kernel takes (in order) at call time:
      mA        : (m, k)              a_dtype (bf16 / fp16)
      mB        : (k // 16, n * 2)    int32 -- Marlin-packed FP4
      mB_sf     : (k // 16, n)        uint8 -- FP8-E4M3 scales
      mC        : (m, n)              c_dtype
      mAlpha    : (1,)                float32 scalar

    ``pipeline_depth`` (K-block dequant prefetch depth, 0/1) and
    ``use_fp16_mma`` (1 = fp16 MMA path, 0 = bf16 MMA path) are
    autotuner-selectable perf knobs; see ``BlackwellDenseGemmW4A16Kernel``.
    """
    # Normalize to a tuple (callers may pass a list) so the cache key is hashable.
    atom_layout = cast(Tuple[int, int, int], tuple(atom_layout))
    pipeline_depth = int(pipeline_depth)
    use_fp16_mma = int(use_fp16_mma)
    enable_pdl = bool(enable_pdl)
    tile_swizzle = int(tile_swizzle)
    # The cute-DSL backend's prep (``_prepare_cute_dsl``) emits S0E5M3 scales when
    # ``_CUTE_DSL_SCALE_S0E5M3``, so the kernel must decode that format -- this is
    # not a free knob, it is locked to what prep produced.
    s0e5m3_scale = 1 if _CUTE_DSL_SCALE_S0E5M3 else 0
    # enable_pdl, tile_swizzle, s0e5m3_scale are baked into the compiled kernel
    # (launch / tile-scheduler / dequant params), so they are part of the cache key.
    cache_key = (
        tile_shape_mnk,
        a_dtype,
        c_dtype,
        atom_layout,
        pipeline_depth,
        use_fp16_mma,
        enable_pdl,
        tile_swizzle,
        s0e5m3_scale,
    )
    cached = _CUTE_DSL_MM_FP4_W4A16_KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    _check_cute_dsl_availability()

    import cutlass
    import cutlass.cute as cute
    from flashinfer.cute_dsl.utils import get_max_active_clusters

    from .kernels.dense_gemm_w4a16_blackwell import (
        BlackwellDenseGemmW4A16Kernel,
    )

    a_cutlass_dtype = getattr(cutlass, _TORCH_TO_CUTLASS_DTYPE_ATTR[a_dtype])
    c_cutlass_dtype = getattr(cutlass, _TORCH_TO_CUTLASS_DTYPE_ATTR[c_dtype])

    sym_m = cute.sym_int()
    sym_k = cute.sym_int()
    sym_n = cute.sym_int()
    sym_k_tiles = cute.sym_int()
    sym_n_packed = cute.sym_int()

    a_fake = cute.runtime.make_fake_compact_tensor(
        a_cutlass_dtype, (sym_m, sym_k), stride_order=(1, 0), assumed_align=16
    )
    b_marlin_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (sym_k_tiles, sym_n_packed),
        stride_order=(1, 0),
        assumed_align=16,
    )
    b_sf_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (sym_k_tiles, sym_n), stride_order=(1, 0), assumed_align=16
    )
    c_fake = cute.runtime.make_fake_compact_tensor(
        c_cutlass_dtype, (sym_m, sym_n), stride_order=(1, 0), assumed_align=16
    )
    alpha_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (1,), assumed_align=4
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    gemm = BlackwellDenseGemmW4A16Kernel(
        acc_dtype=cutlass.Float32,
        tile_shape_mnk=tile_shape_mnk,
        atom_layout=atom_layout,
        pipeline_depth=pipeline_depth,
        use_fp16_mma=use_fp16_mma,
        enable_pdl=enable_pdl,
        tile_swizzle=tile_swizzle,
        s0e5m3_scale=s0e5m3_scale,
    )
    max_active_clusters = get_max_active_clusters(1)

    compiled = cute.compile(
        gemm.wrapper,
        a_fake,
        b_marlin_fake,
        b_sf_fake,
        c_fake,
        alpha_fake,
        1,  # l (batch)
        max_active_clusters,
        stream_fake,
        options="--opt-level 2 --enable-tvm-ffi",
    )

    _CUTE_DSL_MM_FP4_W4A16_KERNEL_CACHE[cache_key] = compiled
    return compiled


# The cute-DSL backend stores its per-block scales as S0E5M3 (a host reformat of
# the canonical E4M3 nvfp4 scale).  This makes the in-kernel scale decode a single
# ``mul.lo.u32`` (byte<<7 = fp16 bits; see ``cvt_s0e5m3_to_f16x2_broadcast``)
# instead of the 3-op E4M3 cvt+broadcast -- a measured ~2-4% prefill win, decode-
# neutral, and memory-neutral (still 1 byte/scale).  The SF format is fixed at
# prep time and shared by every (prefill AND decode) launch on that weight, so
# this is a single global choice for the backend, NOT a per-tactic autotuner knob.
# ``prepare_w4a16_fp4_weights(backend='cute-dsl')`` and the cute-DSL GEMM are a
# matched pair: prep emits S0E5M3, the kernel is always built to decode it.
_CUTE_DSL_SCALE_S0E5M3 = True


def _e4m3_to_s0e5m3(sf_u8: torch.Tensor) -> torch.Tensor:
    """Reformat a uint8 tensor of E4M3 scale bytes to S0E5M3 bytes.

    E4M3 (4-exp bias 7, 3-mant) -> S0E5M3 (5-exp bias 15, 3-mant).  Round-trip via
    fp16: every E4M3 value -- including subnormals (down to 2^-9) and zero -- is a
    *normal* fp16 (min normal 2^-14) carrying only the top 3 mantissa bits, so
    ``fp16_bits >> 7`` is exact and yields ``[exp5 | mant3]``.  Scales are
    non-negative, so the dropped sign bit is always 0.  This is the host side of
    ``cvt_s0e5m3_to_f16x2_broadcast`` (which reconstructs fp16 via ``byte<<7``).
    """
    f16 = sf_u8.contiguous().view(torch.float8_e4m3fn).to(torch.float16)
    bits = f16.view(torch.int16).to(torch.int32) & 0xFFFF
    return ((bits >> 7) & 0xFF).to(torch.uint8)


def _prepare_cute_dsl(
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor],
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """cute-DSL-backend prep: Marlin-repack the weight + unswizzle the SF.

    Produces the bespoke layout the cute-DSL kernel consumes:
      * weight: ``(K // 16, N * 2)`` int32 (Marlin-packed FP4).
      * SF:     ``(K // block_size, N)`` uint8 -- per-block scales in the format
        the cute-DSL kernel decodes (S0E5M3 when ``_CUTE_DSL_SCALE_S0E5M3``, else
        the raw E4M3).
    ``alpha`` is passed through unchanged (the compute step normalizes it
    to a ``(1,) float32`` scalar).  Pair the returned tensors with
    ``mm_w4a16_fp4(..., backend='cute-dsl')``.
    """
    from .marlin_repack import prepare_fp4_w4a16_weight

    n = int(b.shape[0])
    k = int(b.shape[1]) * 2
    k_sf = k // block_size

    # (N, K//2) -> (K//2, N) uint8.  A transpose preserves each byte's
    # low/high-nibble K-semantics, which is exactly what the repack expects.
    b_kn = b.t().contiguous()
    b_marlin = prepare_fp4_w4a16_weight(b_kn)  # (K//16, N*2) int32

    # Canonical 128x4-swizzled SF -> linear (N, K_sf) -> (K_sf, N) uint8.
    linear_sf = _unswizzle_sf_128x4(b_descale, n, k_sf)  # (N, K_sf) uint8
    sf_ksf_n = linear_sf.t().contiguous()  # (K_sf, N) uint8 (E4M3)
    if _CUTE_DSL_SCALE_S0E5M3:
        sf_ksf_n = _e4m3_to_s0e5m3(sf_ksf_n)  # -> S0E5M3, matched to the kernel decode
    return b_marlin, sf_ksf_n, alpha


# =============================================================================
# cute-DSL autotuner integration
# =============================================================================
#
# The cute-DSL kernel is compiled shape-generically (M/N/K are runtime sym_int
# args), so a single compiled kernel handles any problem size -- the only thing
# to "tune" is the *config*: the CTA tile shape, the MMA atom layout, and two
# perf knobs (``pipeline_depth`` = K-block dequant prefetch, ``use_fp16_mma`` =
# fp16 vs bf16 MMA datapath).  ``_w4a16_cute_dsl_tactic_configs`` enumerates a
# small, shape-dependent candidate set; the AutoTuner profiles each and caches
# the fastest per (M-bucket, N, K).  Tactic ``-1`` (the autotuner fallback used
# when tuning is off / on a cache miss) routes to the M-aware
# ``_select_w4a16_tile_shape`` heuristic, so behaviour without ``autotune(True)``
# is byte-for-byte the pre-autotuner path.
#
# Notes on the candidate set:
#   * The MMA atom layout is coupled to tile_M: (1,2,1) requires tile_M == 16
#     (Permutation_m = 16); tile_M >= 32 uses (2,2,1).  See
#     ``_select_w4a16_tile_shape``.
#   * tile_K defaults to 128 (K % 128 == 0) else 64.  At large N (>=8192) a
#     tile_K=64 variant (deeper K-pipeline -> more ab_stages fit -> better
#     TMA-latency hiding) is also offered for the (1,2,1)/tile_M=16 tile; it
#     measured +2-4% on large-N moderate-K cells, neutral/negative elsewhere.
#   * tile_N is 64 by default; a tile_N=128 variant (2 Marlin 64-N blocks per
#     tile, looped in ``_dequant_b_to_register``) is offered only at very large
#     N (>=12288), where halving the (m,n)-tile count cuts per-tile cold-start
#     overhead without hurting the wave count (~+2-6%; neutral/negative below,
#     so gated out).  Both tile_N values are numerically verified correct.
#   * CRITICAL: the AutoTuner ranks tactics by *latency only* -- it never checks
#     correctness.  A config that is fast but wrong would be silently selected.
#     Every config enumerated here must therefore be numerically verified
#     against the torch reference before being added.  The set below is exactly
#     the tile shapes ``_select_w4a16_tile_shape`` already emits (the kernel's
#     validated production tiles) varied only over ``pipeline_depth`` (K-block
#     dequant prefetch), which changes the dequant schedule but not the result.
#   * ``use_fp16_mma`` is NOT swept: it is a compute-datapath knob (fp16 vs bf16
#     MMA) and per-config isolation measured it inert (<=1.5%, never a winner)
#     for the memory-bound small-M W4A16 regime, so a fp16=0 tactic would only
#     add profiling cost.  It stays at the kernel default (1).
#   * Infeasible configs (e.g. a tile that overflows SMEM -> ab_stage == 0)
#     raise during profiling; the AutoTuner records inf time and never selects
#     them, so an occasionally-infeasible config is harmless (but a
#     silently-wrong one is not -- see above).


# GB10 (DGX Spark) L2 size in bytes; used to gate the tile_M=128 prefill tactic
# to weights that overflow L2 (so per-M-tile weight re-reads hit HBM).  A few MB
# off does not matter -- the AutoTuner picks tile_M=64 vs 128 per M-bucket.
_GB10_L2_BYTES = 25 * 1024 * 1024


def _w4a16_cute_dsl_tactic_configs(
    n: int, k: int
) -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int], int, int, int]]:
    """Enumerate cute-DSL tactic configs for a given ``(N, K)``.

    Returns a list of ``(tile_shape_mnk, atom_layout, pipeline_depth,
    use_fp16_mma)`` tuples.  The list is intentionally independent of ``M`` so
    that a tactic index profiled for one M-bucket stays valid when the chosen
    index is replayed for a runtime ``M`` in the same bucket.  The AutoTuner
    selects the best (tile_M, perf-knob) combo for each M-bucket from the
    candidates here.  ``n`` gates the tile_N=128 variant (offered only for very
    large N); the rest of the set uses tile_N=64.
    """
    tile_k = 128 if k % 128 == 0 else 64

    # (tile_M, atom_layout) shapes the kernel is designed/validated for, at the
    # default tile_N=64; a tile_N=128 variant is added below for very large N.
    tile_m_atoms: List[Tuple[int, Tuple[int, int, int]]] = []
    if tile_k == 128:
        tile_m_atoms.append((16, (1, 2, 1)))
    tile_m_atoms.append((32, (2, 2, 1)))
    tile_m_atoms.append((64, (2, 2, 1)))

    configs: List[Tuple[Tuple[int, int, int], Tuple[int, int, int], int, int, int]] = []
    seen = set()

    def add(tile_m, atom, pdepth, fp16, tile_n=64, tk=None, swz=1):
        cfg = (
            (tile_m, tile_n, tile_k if tk is None else tk),
            atom,
            pdepth,
            fp16,
            swz,
        )
        key = (cfg[0], cfg[1], pdepth, fp16, swz)
        if key not in seen:
            seen.add(key)
            configs.append(cfg)

    # Index 0 is the default config (matches the small-M heuristic at
    # tile_K == 128) so the autotuner fallback / first candidate is the proven
    # baseline.  pipeline_depth=0 is swept on the smallest tile, where the
    # short-K cold-start it targets dominates; larger tile_M shapes (which the
    # heuristic uses at bigger M) are offered with default knobs.
    base_tile_m, base_atom = tile_m_atoms[0]
    add(base_tile_m, base_atom, 1, 1)  # 0: baseline
    add(base_tile_m, base_atom, 0, 1)  # no dequant prefetch (helps short-K)
    for tile_m, atom in tile_m_atoms[1:]:
        add(tile_m, atom, 1, 1)

    # tile_N=128 halves the (m,n)-tile count -> fewer per-tile cold-starts.
    # Measured to help only at very large N (>=12288), where the halved tile
    # count still maps to a healthy wave count (~+2-6%); below that the halved
    # count lands on a worse wave quantization (e.g. n=8192: 128->64 tiles,
    # wave_eff 0.89->0.67) and it is neutral/negative, so it is gated out.
    # Both tile_N values are numerically correct, so a mis-gate only costs a
    # few % via a wrong autotuner pick, never correctness.  Requires
    # tile_k==128 (the (1,2,1) atom) and n % 128 == 0 (exact N tiling).
    if tile_k == 128 and n >= 12288 and n % 128 == 0:
        add(base_tile_m, base_atom, 1, 1, tile_n=128)

    # tile_K=64 (deeper K-pipeline: more, smaller K-tiles let more ab_stages
    # fit in SMEM -> better TMA-latency hiding).  Offered as a tile_K=128
    # alternative only at large N (>=8192), where it measured +2-4% on
    # moderate-K cells (e.g. n=8192 k=2048 +4.2%); it is neutral/negative for
    # small N or long K, where the default tile_K=128 wins.  Same (1,2,1) /
    # tile_M=16 atom as the baseline, just a finer K split -- numerically
    # verified correct, so a wrong autotuner pick only costs a few %.  (Only
    # meaningful when the default tile_k is 128; for k%128!=0 tile_K is already
    # 64.)
    if tile_k == 128 and n >= 8192:
        add(base_tile_m, base_atom, 1, 1, tile_n=64, tk=64)

    # tile_M=128 (taller M tile, atom (2,2,1)) -- the large-M *prefill* lever.
    # The persistent kernel computes one (m,n) output tile per CTA pass and
    # independently TMA-loads + re-dequantizes the full B weight for each
    # M-tile, so at large M the weight is re-read M/tile_M times.  When the
    # weight overflows L2 (GB10 L2 = 25 MB) those re-reads hit HBM and the
    # kernel goes bandwidth-bound on redundant weight traffic -- the measured
    # large-M throughput "cliff" (e.g. 8192x8192 M=4096: ~14 TFLOP/s vs
    # Marlin's ~70).  Doubling tile_M halves the M-tile count -> halves the
    # redundant weight HBM traffic: measured +48-80% at K=8192 (B=32 MB > L2),
    # neutral when the weight fits L2 (4096^2, B=8 MB).  Gated to weights that
    # overflow L2 (B = n*k/2 bytes >= L2); the AutoTuner then picks tile_M=64
    # vs 128 per M-bucket (small-M decode keeps tile_M<=64).  IMPORTANT: only
    # tile_M in {64, 128} are numerically correct -- tile_M >= 160 corrupts the
    # output (the epilogue/MMA path breaks beyond 4 MMA M-steps), so 128 is the
    # ceiling.  Both safe values are verified correct, so a mis-pick costs only
    # a few %, never correctness.
    if tile_k == 128 and (n * k) // 2 >= _GB10_L2_BYTES:
        add(128, (2, 2, 1), 1, 1)

    # Threadblock swizzle (tile_swizzle=8) -- the large-M prefill *cliff* fix.
    # The cliff is L2 thrashing: at large M the per-M-tile weight re-reads plus
    # the huge A-tile reads exceed L2 (25 MB), so the weight isn't resident
    # across its re-reads -> HBM-bound.  A swizzled (super-tile) traversal order
    # restores A+B L2 locality.  Measured (8192x8192 M=4096): tile_M=128 + swz8
    # = 8.2 ms (~67 TFLOP/s) vs swz1 36 ms -- from 4.98x behind Marlin to ~1.05x.
    # Swizzle only reorders tile traversal (numerically identical, cos=1.0) and
    # is a no-op when there are few tiles (decode-neutral), so the AutoTuner can
    # pick it freely per M-bucket: it wins at large M, costs nothing at small M.
    # Offered (swz=8, the measured sweet spot; 16/32 slightly worse) for the
    # tile_M=64 and tile_M=128 prefill tiles.  swz=1 versions remain available so
    # the autotuner picks swz=8 only for the large-M buckets where it wins.
    #
    # tile_M=64 + swz8: gated broadly (n*k >= 16 MiB) -- it helps wherever a large
    # M produces many tiles, INCLUDING 4096x4096 (weight fits L2, but the M=4096
    # A-tile traffic still thrashes it; measured 1.83x).  The gate only excludes
    # tiny decode shapes, where swizzle is a no-op and the autotuner picks swz=1.
    if tile_k == 128 and n * k >= 16 * 1024 * 1024:
        add(64, (2, 2, 1), 1, 1, swz=8)
    # tile_M=128 + swz8: pairs with the weight>L2 cliff fix (where tile_M=128 is
    # offered); it was the best config at the worst cliff (8192^2 M=4096).
    if tile_k == 128 and (n * k) // 2 >= _GB10_L2_BYTES:
        add(128, (2, 2, 1), 1, 1, swz=8)

    # tile_N=128 (with tile_M=64, atom (2,2,1)) -- the fits-L2 / compute-bound
    # "cluster A" fix (4096^2, 5120^2, 6144x4096: weight in L2, so not the re-read
    # cliff -- MMA/per-tile-overhead bound, where Marlin's wider tiles win).  A
    # wider-N tile amortizes the per-tile dequant/epilogue overhead and feeds the
    # MMA better: measured (+ swz8) it lifted those shapes from ~73% to ~92-99% of
    # Marlin at M>=2048.  Offered with swz8 (compute + any L2 reuse) and swz1; the
    # autotuner picks per M-bucket.  CORRECTNESS: only (tile_M=64, tile_N=128) is
    # valid -- tile_M=128 x tile_N=128 and tile_N=256 corrupt the output (the
    # hand-coded dequant/epilogue ceiling, same as tile_M>128), so they are NOT
    # offered.  Requires n % 128 == 0 (exact N tiling).
    if tile_k == 128 and n % 128 == 0 and n >= 4096:
        add(64, (2, 2, 1), 1, 1, tile_n=128, swz=8)
        add(64, (2, 2, 1), 1, 1, tile_n=128, swz=1)

    return configs


# Autotuner sweeps M (token count) of the bf16 activation ``a`` and keeps the
# output ``out`` in lockstep.  The Marlin weight ``b``, its scale, and ``alpha``
# are M-independent and stay fixed during profiling.
#
# inputs layout (see _compute_cute_dsl):
#   [0]=a  [1]=b (marlin int32)  [2]=b_sf (uint8)  [3]=alpha
#   [4]=out_dtype  [5]=out  [6]=block_size
_W4A16_CUTE_DSL_TUNING_CONFIG = TuningConfig(
    dynamic_tensor_specs=(
        DynamicTensorSpec(
            (0,),  # a_tensor_index
            (0,),  # M dimension
            get_hybrid_num_tokens_buckets,
            map_to_hybrid_bucket_uncapped,
        ),
    ),
    constraint_specs=(
        ConstraintSpec(
            5,  # out_tensor_index follows M
            0,
            lambda shapes: shapes[0][0],
        ),
    ),
)


def _cute_dsl_w4a16_fp4_runner(enable_pdl: bool = True) -> TunableRunner:
    """Build a ``CuteDslW4a16Fp4Runner`` for the cute-DSL W4A16 GEMM.

    Each tactic is an index into :func:`_w4a16_cute_dsl_tactic_configs` for the
    runtime ``(N, K)``.  Tactic ``-1`` is the fallback: it routes to the M-aware
    ``_select_w4a16_tile_shape`` heuristic with default knobs (i.e. the
    pre-autotuner behaviour), so an un-tuned call is unchanged.  ``enable_pdl``
    is a launch-time flag (captured here, baked into the compiled kernel); it is
    orthogonal to the tactic choice so it does not enter the tactic key.
    """

    class CuteDslW4a16Fp4Runner(TunableRunner):
        def get_cache_key_extras(self, inputs: List[torch.Tensor]) -> tuple:
            a, b, _, _, out_dtype, _, block_size = inputs
            n = int(b.shape[1]) // 2
            k = int(b.shape[0]) * int(block_size)
            return (out_dtype, n, k)

        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
        ) -> List[int]:
            _, b, _, _, _, _, block_size = inputs
            n = int(b.shape[1]) // 2
            k = int(b.shape[0]) * int(block_size)
            return list(range(len(_w4a16_cute_dsl_tactic_configs(n, k))))

        def forward(
            self,
            inputs: List[torch.Tensor],
            tactic: int = -1,
            do_preparation: bool = False,
            **kwargs,
        ) -> torch.Tensor:
            a, b, b_sf_u8, alpha_for_launch, out_dtype, out, block_size = inputs
            n = int(b.shape[1]) // 2
            k = int(b.shape[0]) * int(block_size)
            m = int(a.shape[0])
            if tactic < 0:
                # Fallback == pre-autotuner heuristic (M-aware), default knobs.
                tile_shape_mnk, atom_layout = _select_w4a16_tile_shape(m, n, k)
                pipeline_depth, use_fp16_mma, tile_swizzle = 1, 1, 1
            else:
                (
                    tile_shape_mnk,
                    atom_layout,
                    pipeline_depth,
                    use_fp16_mma,
                    tile_swizzle,
                ) = _w4a16_cute_dsl_tactic_configs(n, k)[tactic]
            compiled = _get_cute_dsl_w4a16_gemm(
                tile_shape_mnk,
                a.dtype,
                out_dtype,
                atom_layout,
                pipeline_depth,
                use_fp16_mma,
                enable_pdl=enable_pdl,
                tile_swizzle=tile_swizzle,
            )
            compiled(a, b, b_sf_u8, out, alpha_for_launch)
            return out

    return CuteDslW4a16Fp4Runner()


def _compute_cute_dsl(
    a: torch.Tensor,
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out: Optional[torch.Tensor],
    block_size: int,
    enable_pdl: bool = True,
) -> torch.Tensor:
    """cute-DSL-backend compute: dispatch to the compiled Blackwell kernel.

    ``b`` is the Marlin-packed ``(K // 16, N * 2)`` int32 weight and
    ``b_descale`` the ``(K // 16, N)`` FP8-E4M3 SF returned by
    :func:`_prepare_cute_dsl`.
    """
    if b.dtype != torch.int32:
        raise TypeError(
            f"cute-dsl backend expects the Marlin-packed int32 weight from "
            f"prepare_w4a16_fp4_weights(..., backend='cute-dsl'); got {b.dtype}."
        )
    # The kernel's MMA path requires the output dtype to match the
    # activation dtype (it asserts a_dtype == c_dtype internally).  With a
    # bf16 activation, an fp16 output would need an fp16 activation, which
    # this API does not support yet -- reject it explicitly rather than
    # surfacing the kernel's lower-level TypeError.  (Use the cudnn backend
    # if you need a mismatched output dtype.)
    if out_dtype != a.dtype:
        raise NotImplementedError(
            f"cute-dsl backend requires out_dtype == a.dtype (got "
            f"out_dtype={out_dtype}, a.dtype={a.dtype}).  Use the cudnn "
            f"backend for a mismatched output dtype."
        )
    k_tiles = int(b.shape[0])
    n = int(b.shape[1]) // 2
    k = k_tiles * block_size
    m = int(a.shape[0])
    if a.shape[1] != k:
        raise ValueError(
            f"a.shape[1]={a.shape[1]} but k inferred from prepared b.shape="
            f"{tuple(b.shape)} is {k}"
        )

    if out is None:
        out = torch.empty((m, n), device=a.device, dtype=out_dtype)
    else:
        if tuple(out.shape) != (m, n):
            raise ValueError(f"out shape {tuple(out.shape)} != expected {(m, n)}")
        if out.dtype != out_dtype:
            raise TypeError(f"out dtype {out.dtype} != requested out_dtype {out_dtype}")

    b_sf_u8 = b_descale.view(torch.uint8).contiguous()
    alpha_for_launch = _prepare_w4a16_alpha(alpha, a.device)

    # Config selection (tile shape + perf knobs) goes through the AutoTuner.
    # Outside an ``autotune(True)`` context (or on a cache miss) ``choose_one``
    # returns tactic -1, which the runner maps to the M-aware
    # ``_select_w4a16_tile_shape`` heuristic -- i.e. the pre-autotuner path.
    tuner = AutoTuner.get()
    runner = _cute_dsl_w4a16_fp4_runner(enable_pdl=enable_pdl)
    inputs = [a, b, b_sf_u8, alpha_for_launch, out_dtype, out, block_size]
    chosen_runner, tactic = tuner.choose_one(
        "w4a16_fp4_cute_dsl_gemm",
        [runner],
        _W4A16_CUTE_DSL_TUNING_CONFIG,
        inputs,
    )
    chosen_runner(inputs=inputs, tactic=tactic)
    return out


__all__ = [
    "prepare_w4a16_fp4_weights",
    "mm_w4a16_fp4",
]
