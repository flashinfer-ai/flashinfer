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

from typing import Literal, Optional, Tuple

import torch

from ..api_logging import flashinfer_api
from ..utils import backend_requirement, supported_compute_capability


# =============================================================================
# Backend requirement checks (used by @backend_requirement on mm_w4a16_fp4)
# =============================================================================


def _check_mm_w4a16_fp4_problem_size(
    a: torch.Tensor,
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    *,
    backend: Literal["torch"],
    out_dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
    block_size: int = 16,
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
    backend: Literal["torch"],
    out_dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
    block_size: int = 16,
):
    """Torch backend gated to Blackwell-class SMs (100, 103, 110, 120, 121).

    The implementation is pure PyTorch and would technically run on older
    GPUs, but the API exists to support W4A16 on Blackwell GPUs.  Gating
    here keeps the supported-capability surface honest.
    """
    return True


# =============================================================================
# Public dispatchers
# =============================================================================


def prepare_w4a16_fp4_weights(
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    *,
    backend: Literal["torch"],
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
    raise ValueError(f"Unknown backend {backend!r}.  Supported: 'torch'.")


@backend_requirement(
    {"torch": _torch_w4a16_fp4_requirement},
    common_check=_check_mm_w4a16_fp4_problem_size,
)
@flashinfer_api
def mm_w4a16_fp4(
    a: torch.Tensor,
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    *,
    backend: Literal["torch"],
    out_dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
    block_size: int = 16,
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
    raise ValueError(f"Unknown backend {backend!r}.  Supported: 'torch'.")


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


__all__ = [
    "prepare_w4a16_fp4_weights",
    "mm_w4a16_fp4",
]
