# SPDX-FileCopyrightText: Copyright (c) 2025 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""W4A16 FP4 GEMM with backend-tagged weight preparation.

Public API
==========

.. code-block:: python

    # 1) Prepare weights for a specific backend (one-time, at model load).
    b_p, sf_p, alpha_p = flashinfer.prepare_w4a16_fp4_weights(
        b, b_descale, alpha, backend="torch",
    )

    # 2) Run the GEMM, passing the *same* backend tag.
    out = flashinfer.mm_w4a16_fp4(
        a, b_p, sf_p, alpha_p, backend="torch",
    )

The contract: the ``backend`` string must match between prepare and
compute.  Each backend is free to choose any internal layout for ``b``
/ ``b_descale`` -- the shapes/dtypes returned by
``prepare_w4a16_fp4_weights`` are whatever that backend's compute step
consumes, and mixing prep from one backend with compute from another
is undefined.

Supported backends
------------------

* ``"torch"`` -- pure-PyTorch dequantize + ``torch.matmul`` reference.
  Slow but works on any device with PyTorch.  Useful as a refcheck
  baseline.  No prep transformation: it returns its inputs unchanged.
* ``"cudnn"`` -- cuDNN FP4 GEMM graph (bf16 activation x FP4 weight).
  Reuses ``mm_fp4``'s cuDNN machinery and the autotuner; requires a
  cuDNN build with FP4 block-scale support on Blackwell-class GPUs.
  No prep transformation: weights are already in the layout the graph
  consumes.

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
     - extend the ``backend: Literal["torch"]`` annotation to include
       the new name;
     - add ``if backend == "<name>": return _prepare_<name>(...)`` above
       the trailing ``raise ValueError``;
     - update the ``"Supported: 'torch'."`` string in the error.
3. Do the same three edits in :func:`mm_w4a16_fp4`.
4. Add a ``_<name>_w4a16_fp4_requirement`` function (same signature as
   :func:`_torch_w4a16_fp4_requirement`) and register it in the
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

See :func:`_prepare_torch` / :func:`_compute_torch` below for a
minimal working implementation of these two functions.
"""

import functools
from typing import List, Literal, Optional, Tuple

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
    _check_cudnn_fp4_availability,
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
    backend: Literal["torch", "cudnn"],
    out_dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
    block_size: int = 16,
    is_sf_swizzled: bool = True,
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
def _torch_w4a16_fp4_requirement(
    a: torch.Tensor,
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    *,
    backend: Literal["torch", "cudnn"],
    out_dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
    block_size: int = 16,
    is_sf_swizzled: bool = True,
):
    """Torch backend gated to Blackwell-class SMs (100, 103, 110, 120, 121).

    The implementation is pure PyTorch and would technically run on older
    GPUs, but the API exists to support W4A16 on Blackwell GPUs.  Gating
    here keeps the supported-capability surface honest.

    The torch reference unswizzles the SF internally, so it requires the
    canonical 128x4-swizzled layout (``is_sf_swizzled=True``); a
    non-swizzled SF is not supported.
    """
    if not is_sf_swizzled:
        raise ValueError(
            "torch backend requires the 128x4-swizzled SF layout "
            "(is_sf_swizzled=True); use the cudnn backend for a "
            "non-swizzled (linear) SF."
        )
    return True


@supported_compute_capability([100, 103, 110, 120, 121])
def _cudnn_w4a16_fp4_requirement(
    a: torch.Tensor,
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    *,
    backend: Literal["torch", "cudnn"],
    out_dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
    block_size: int = 16,
    is_sf_swizzled: bool = True,
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


# =============================================================================
# Public dispatchers
# =============================================================================


def prepare_w4a16_fp4_weights(
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    *,
    backend: Literal["torch", "cudnn"],
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
        backend: Identifier of a supported backend (currently
            ``"torch"``).
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
    if backend == "torch":
        return _prepare_torch(b, b_descale, alpha, block_size)
    if backend == "cudnn":
        return _prepare_cudnn(b, b_descale, alpha, block_size)
    raise ValueError(f"Unknown backend {backend!r}.  Supported: 'torch', 'cudnn'.")


@backend_requirement(
    {
        "torch": _torch_w4a16_fp4_requirement,
        "cudnn": _cudnn_w4a16_fp4_requirement,
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
    backend: Literal["torch", "cudnn"],
    out_dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
    block_size: int = 16,
    is_sf_swizzled: bool = True,
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
            128x4-swizzled SF layout.  This selects which backend is
            supported: ``True`` (default) requires the ``torch`` backend
            (cuDNN does not support swizzled SF), while ``False`` (a
            non-swizzled / linear SF, as produced by
            ``prepare_w4a16_fp4_weights(..., backend="cudnn")``) requires
            the ``cudnn`` backend.

    Returns:
        ``(M, N)`` tensor of ``out_dtype``.

    Notes:
        ``@backend_requirement`` adds two introspection helpers to this
        function: ``mm_w4a16_fp4.is_backend_supported("torch")`` and
        ``mm_w4a16_fp4.is_compute_capability_supported(cc)``.  It also
        accepts a ``skip_check=True`` kwarg to bypass validation in
        latency-sensitive code paths.
    """
    out_dtype = out_dtype or a.dtype
    if backend == "torch":
        return _compute_torch(a, b, b_descale, alpha, out_dtype, out, block_size)
    if backend == "cudnn":
        return _compute_cudnn(a, b, b_descale, alpha, out_dtype, out, block_size)
    raise ValueError(f"Unknown backend {backend!r}.  Supported: 'torch', 'cudnn'.")


# =============================================================================
# Torch backend (reference implementation)
# =============================================================================
#
# NOTE: Everything below this banner is a reference implementation kept
# only to validate the API surface and serve as a refcheck baseline.
# Once a real backend (cute-DSL, cuDNN, ...) lands AND a refcheck test
# compares it directly against a ground-truth path (e.g. round-tripping
# through ``nvfp4_quantize``), this whole section can be deleted:
#   * _E2M1_CODEBOOK_FP32, _unswizzle_sf_128x4, _dequantize_w4a16_fp4_torch
#   * _prepare_torch, _compute_torch
# along with the ``"torch"`` branches in the two dispatchers above, the
# ``"torch"`` entry in their ``backend: Literal[...]`` annotations, and
# the torch-backend tests in tests/gemm/test_mm_w4a16_fp4.py.


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


def _prepare_torch(
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor],
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Torch-backend prep: pass-through.  All work happens at compute time."""
    return b, b_descale, alpha


def _compute_torch(
    a: torch.Tensor,
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out: Optional[torch.Tensor],
    block_size: int,
) -> torch.Tensor:
    """Torch-backend compute: dequant + ``torch.matmul`` in fp32."""
    n = int(b.shape[0])
    k = int(b.shape[1]) * 2
    if a.shape[1] != k:
        raise ValueError(
            f"a.shape[1]={a.shape[1]} but k inferred from b.shape={tuple(b.shape)} "
            f"is {k}"
        )
    weight_fp32 = _dequantize_w4a16_fp4_torch(b, b_descale, alpha, n, k, block_size)
    # Match the kernel's fp32-accumulator semantics: cast a -> fp32 for
    # the matmul, then cast the result to out_dtype.
    result_fp32 = a.to(torch.float32) @ weight_fp32.T  # (M, N) fp32
    result = result_fp32.to(out_dtype)
    if out is not None:
        if tuple(out.shape) != (a.shape[0], n):
            raise ValueError(
                f"out shape {tuple(out.shape)} != expected {(a.shape[0], n)}"
            )
        if out.dtype != out_dtype:
            raise TypeError(f"out dtype {out.dtype} != requested out_dtype {out_dtype}")
        out.copy_(result)
        return out
    return result


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
#   tensor with ``tensor_reordering.NONE``.  (The ``is_sf_swizzled``
#   attribute on :func:`mm_w4a16_fp4` gates which backend is allowed: a
#   swizzled SF forces ``torch``; a non-swizzled SF forces ``cudnn``.)


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


__all__ = [
    "prepare_w4a16_fp4_weights",
    "mm_w4a16_fp4",
]
