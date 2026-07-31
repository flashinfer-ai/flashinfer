"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Single-kernel ("mono") top-K Mixture-of-Experts for the Qwen3.5-35B
block-FP8 shape on Hopper (SM90a).  The full pipeline — routing,
up-projection, SiLU, down-projection and reduction — runs inside one
`__global__` launch.  See csrc/fused_moe/monomoe/README.md for the design.

This path is hard-specialized to a single shape:
  E (experts)  = 256
  N (== N_half) = 512   (up-projection produces 2*N = 1024 rows: gate || up)
  K (hidden)   = 2048
  BS (token cap) = 8
Inputs that do not match are rejected up front rather than silently
corrupting memory.
"""

import contextlib
import functools
from typing import Optional

import torch

from ..api_logging import flashinfer_api
from ..trace.templates.moe import mono_moe_trace
from ..utils import backend_requirement, supported_compute_capability

# Hard-coded geometry of the only compiled variant
# (Dims_BS8_E256_Qwen3_5_35B_BlockFP8_WGMMA_TMA in
# csrc/fused_moe/monomoe/src/moe_interface.h).
_MONOMOE_E = 256
_MONOMOE_N = 512  # N_half: gate and up each have N rows
_MONOMOE_K = 2048
_MONOMOE_BS = 8
_BLOCK = 128  # block-wise FP8 quantization tile (128 x 128)

_SCORING_SIGMOID = 0
_SCORING_SOFTMAX = 1


@functools.cache
def _get_monomoe_module():
    """Lazily build and load the monomoe CUDA extension via FlashInfer's JIT."""
    try:
        from ..jit.monomoe import load_monomoe_module

        return load_monomoe_module()
    except (ImportError, FileNotFoundError, RuntimeError) as e:
        raise ImportError(
            f"Failed to load the MonoMoe kernel CUDA extension via JIT. "
            f"Ensure a Hopper (SM90a) GPU and the CUDA toolkit are available and "
            f"that csrc/fused_moe/monomoe/ sources exist.\nError: {e}"
        ) from e


@functools.cache
@flashinfer_api
def has_monomoe() -> bool:
    """Return True if the monomoe CUDA extension can be built and loaded."""
    try:
        _get_monomoe_module()
        return True
    except ImportError:
        return False


@functools.cache
@flashinfer_api
def get_scratchpad_size_bytes() -> int:
    """Return the global scratchpad size (bytes) required by the kernel.

    Sourced from the C++ `sizeof(MoEGemmSpec<Dims>)` (exported as
    `monomoe_scratchpad_size`) so the buffer can never desync from the
    kernel's struct layout — the software-grid-barrier counters live at the
    tail of that struct and must be backed by allocated memory.
    """
    mod = _get_monomoe_module()
    return int(mod.monomoe_scratchpad_size())


@flashinfer_api
def alloc_scratchpad(device: torch.device) -> torch.Tensor:
    """Allocate a zero-initialized scratchpad on ``device`` for the kernel.

    Returns a 1-D ``uint8`` tensor sized to ``get_scratchpad_size_bytes()``.
    The buffer is reusable across calls (the kernel self-maintains its barrier
    counters via the ping-pong reset discipline), so callers should allocate
    once and pass the same tensor to every :func:`mono_moe` invocation to avoid
    per-call allocation and the one-time counter zero-init on the C++ side.

    Parameters
    ----------
    device : torch.device
        The CUDA device on which to allocate the scratchpad buffer.

    Returns
    -------
    torch.Tensor
        A 1-D uint8 tensor of get_scratchpad_size_bytes() bytes,
        zero-initialised and placed on *device*.
    """
    nbytes = get_scratchpad_size_bytes()
    return torch.zeros(nbytes, dtype=torch.uint8, device=device)


@flashinfer_api
def interleave_for_tma_wgmma_up(w_fp8: torch.Tensor) -> torch.Tensor:
    """Repack fp8 up-projection weights so one ``boxDim=(128, 128)``
    SWIZZLE_128B TMA issue fetches a full 128-row x 128-K WGMMA A-tile.

    Input layout: ``[E, 2*N, K]`` row-major fp8 — the first ``N`` rows per
    expert are gate weights, the last ``N`` are up weights.  ``N`` must be a
    multiple of 64.

    Output layout (still ``[E, 2*N, K]``, identical byte footprint): for every
    expert ``e`` and every 64-gate-row block ``k`` in ``[0, N/64)``::

        new[e, 128k +  0:128k+ 32] = gate[e, 64k    :64k+32]
        new[e, 128k + 32:128k+ 64] =   up[e, 64k    :64k+32]
        new[e, 128k + 64:128k+ 96] = gate[e, 64k+32 :64k+64]
        new[e, 128k + 96:128k+128] =   up[e, 64k+32 :64k+64]

    Under SWZ128 the TMA applies the 8-row x 128-byte core-matrix XOR swizzle
    automatically, so this only rearranges GM rows (no byte-level permutation).
    The result is cached on the input tensor as ``_tma_interleaved_up``.

    The down-projection weights need no preparation — the raw ``[E, K, N]``
    row-major fp8 tensor is passed straight through.

    Parameters
    ----------
    w_fp8 : torch.Tensor
        FP8 up/gate weight tensor with shape [E, 2*N, K] (row-major).
        The first N rows per expert are gate weights; the last N are
        up weights.  N must be a multiple of 64.

    Returns
    -------
    torch.Tensor
        Repacked weight tensor with the same shape and dtype as *w_fp8*,
        laid out so that a single TMA boxDim=(128, 128) issue covers a
        complete 128-row x 128-K WGMMA A-tile.
    """
    cached = getattr(w_fp8, "_tma_interleaved_up", None)
    if cached is not None:
        return cached

    E, rows, K = w_fp8.shape
    if rows % 2 != 0:
        raise ValueError(f"expected rows = 2*N, got rows={rows}")
    n_half = rows // 2
    if n_half % 64 != 0:
        raise ValueError(f"N (half of rows) must be a multiple of 64; got N={n_half}")

    gate = w_fp8[:, :n_half, :]
    up = w_fp8[:, n_half:, :]

    blocks = n_half // 64
    gate_r = gate.reshape(E, blocks, 64, K)
    up_r = up.reshape(E, blocks, 64, K)
    gate_lo = gate_r[:, :, :32, :]
    gate_hi = gate_r[:, :, 32:, :]
    up_lo = up_r[:, :, :32, :]
    up_hi = up_r[:, :, 32:, :]

    # gate_lo, up_lo, gate_hi, up_hi along a new stripe axis -> [E, blocks*128, K]
    stripes = torch.stack([gate_lo, up_lo, gate_hi, up_hi], dim=2)
    result = stripes.reshape(E, blocks * 128, K).contiguous()

    with contextlib.suppress(AttributeError, RuntimeError):
        w_fp8._tma_interleaved_up = result
    return result


def _check_shapes(
    activations_in: torch.Tensor,
    router_logits: torch.Tensor,
    expert_weights_up: torch.Tensor,
    expert_scales_up: torch.Tensor,
    expert_weights_down: torch.Tensor,
    expert_scales_down: torch.Tensor,
) -> int:
    """Validate the fixed-shape contract and return the active token count."""
    E, N, K, BS = _MONOMOE_E, _MONOMOE_N, _MONOMOE_K, _MONOMOE_BS

    # Explicit raises (not assert): these validate user-provided tensor
    # shapes that, if wrong, let the fixed-shape CUDA kernel read/write out
    # of bounds.  `assert` would be stripped under `python -O`, so the
    # checks must always run.
    if activations_in.dim() != 2:
        raise ValueError(f"activations_in must be [M, K], got {activations_in.dim()}D")
    m = activations_in.size(0)
    if m > BS:
        raise ValueError(f"this kernel caps tokens at BS={BS}; got M={m}")
    if activations_in.size(1) != K:
        raise ValueError(f"activations K must be {K}, got {activations_in.size(1)}")
    if tuple(router_logits.shape) != (m, E):
        raise ValueError(
            f"router_logits must be [{m}, {E}], got {tuple(router_logits.shape)}"
        )

    # Up weights: interleaved [E, 2*N, K]; up scales block-wise [E, 2N/128, K/128].
    if tuple(expert_weights_up.shape) != (E, 2 * N, K):
        raise ValueError(
            f"expert_weights_up must be [{E}, {2 * N}, {K}], got {tuple(expert_weights_up.shape)}"
        )
    if tuple(expert_scales_up.shape) != (E, (2 * N) // _BLOCK, K // _BLOCK):
        raise ValueError(
            f"expert_scales_up must be [{E}, {(2 * N) // _BLOCK}, {K // _BLOCK}], "
            f"got {tuple(expert_scales_up.shape)}"
        )
    # Down weights: raw [E, K, N]; down scales block-wise [E, K/128, N/128].
    if tuple(expert_weights_down.shape) != (E, K, N):
        raise ValueError(
            f"expert_weights_down must be [{E}, {K}, {N}], got {tuple(expert_weights_down.shape)}"
        )
    if tuple(expert_scales_down.shape) != (E, K // _BLOCK, N // _BLOCK):
        raise ValueError(
            f"expert_scales_down must be [{E}, {K // _BLOCK}, {N // _BLOCK}], "
            f"got {tuple(expert_scales_down.shape)}"
        )
    return m


@supported_compute_capability([90])
def _check_mono_moe_supported(
    activations_in: torch.Tensor,
    router_logits: torch.Tensor,
    expert_weights_up: torch.Tensor,
    expert_scales_up: torch.Tensor,
    expert_weights_down: torch.Tensor,
    expert_scales_down: torch.Tensor,
    top_k: int,
    scoring_func: str = "softmax",
    renormalize: bool = True,
    out: Optional[torch.Tensor] = None,
    scratchpad: Optional[torch.Tensor] = None,
    interleave_up: bool = True,
) -> bool:
    """Backend-requirement check for :func:`mono_moe`.

    Carries the ``@supported_compute_capability([90])`` annotation so the
    ``@backend_requirement`` wrapper rejects any non-Hopper device with a
    clear ``BackendSupportedError`` *before* the SM90a-only kernel is JIT
    compiled — otherwise a call on, say, SM80 would fail deep inside nvcc
    with a cryptic ``wgmma`` / TMA build error.  Also runs the fixed-shape
    contract validation (delegated to :func:`_check_shapes`) so shape and
    architecture support are decided in one place.

    Accepts ``mono_moe``'s full signature because ``@backend_requirement``
    forwards every (default-applied) argument to the checker.  Returns
    ``True`` when the call is supported; raises ``ValueError`` on a shape
    mismatch.
    """
    _check_shapes(
        activations_in,
        router_logits,
        expert_weights_up,
        expert_scales_up,
        expert_weights_down,
        expert_scales_down,
    )
    return True


@backend_requirement({}, common_check=_check_mono_moe_supported)
@flashinfer_api(trace=mono_moe_trace)
def mono_moe(
    activations_in: torch.Tensor,
    router_logits: torch.Tensor,
    expert_weights_up: torch.Tensor,
    expert_scales_up: torch.Tensor,
    expert_weights_down: torch.Tensor,
    expert_scales_down: torch.Tensor,
    top_k: int,
    scoring_func: str = "softmax",
    renormalize: bool = True,
    out: Optional[torch.Tensor] = None,
    scratchpad: Optional[torch.Tensor] = None,
    interleave_up: bool = True,
) -> torch.Tensor:
    """Single-kernel block-FP8 top-K MoE (Qwen3.5-35B shape, SM90a only).

    Fixed shape: E=256 experts, N=512, K=2048 hidden, up to BS=8 tokens.

    Args:
        activations_in: bf16 input activations ``[M, K]`` (``M <= 8``).
        router_logits: bf16 router logits ``[M, E]``.
        expert_weights_up: fp8_e4m3 up/gate weights ``[E, 2*N, K]``. By default
            this function applies :func:`interleave_for_tma_wgmma_up`; pass
            ``interleave_up=False`` if the tensor is already interleaved.
        expert_scales_up: fp32 block-wise scales ``[E, 2N/128, K/128]``.
        expert_weights_down: fp8_e4m3 down weights ``[E, K, N]`` (raw row-major).
        expert_scales_down: fp32 block-wise scales ``[E, K/128, N/128]``.
        top_k: experts selected per token (1..8).
        scoring_func: ``"sigmoid"`` or ``"softmax"``.
        renormalize: renormalize the top-K weights to sum to 1.
        out: optional bf16 output buffer ``[M, K]``; allocated if omitted.
        scratchpad: optional reusable uint8 scratchpad from
            :func:`alloc_scratchpad`; allocated per-call if omitted.
        interleave_up: apply the gate/up TMA repack to ``expert_weights_up``
            (default True).

    Returns:
        bf16 MoE output ``[M, K]``.
    """
    if not (1 <= top_k <= 8):
        raise ValueError(f"top_k must be in [1, 8], got {top_k}")
    sf_map = {"sigmoid": _SCORING_SIGMOID, "softmax": _SCORING_SOFTMAX}
    if scoring_func not in sf_map:
        raise ValueError(
            f"scoring_func must be 'sigmoid' or 'softmax', got {scoring_func!r}"
        )

    m = _check_shapes(
        activations_in,
        router_logits,
        expert_weights_up,
        expert_scales_up,
        expert_weights_down,
        expert_scales_down,
    )

    if interleave_up:
        expert_weights_up = interleave_for_tma_wgmma_up(expert_weights_up)

    if out is None:
        out = torch.empty(
            m, _MONOMOE_K, dtype=torch.bfloat16, device=activations_in.device
        )
    if scratchpad is None:
        scratchpad = alloc_scratchpad(activations_in.device)

    mod = _get_monomoe_module()
    mod.monomoe_topk(
        activations_in,
        router_logits,
        expert_weights_up,
        expert_scales_up,
        expert_weights_down,
        expert_scales_down,
        out,
        scratchpad,
        int(top_k),
        int(sf_map[scoring_func]),
        bool(renormalize),
    )
    return out
