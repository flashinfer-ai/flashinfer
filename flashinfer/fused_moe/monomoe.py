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

Single-kernel ("mono") top-K Mixture-of-Experts, block-FP8, Hopper (SM90a).
The full pipeline — routing, up-projection, SiLU, down-projection and
reduction — runs inside one `__global__` launch.  See
docs/design_docs/monomoe_kernel.md for the design.

The kernel is hard-specialized to a single fixed shape: E=256 experts,
N=512 (intermediate half), K=2048 (hidden).  Token count M <= 8 (the BS8
kernel).  A tensor whose ``(E, N, K)`` is not this shape is rejected up front.
"""

import contextlib
import functools
from typing import Optional

import torch

from ..api_logging import flashinfer_api
from ..trace.templates.moe import mono_moe_trace
from ..utils import backend_requirement, supported_compute_capability

_BLOCK = 128  # block-wise FP8 quantization tile (128 x 128)

# The single hard-specialized shape (must match
# csrc/fused_moe/monomoe/src/moe_interface.h).
_MONOMOE_E = 256
_MONOMOE_N = 512
_MONOMOE_K = 2048
_MONOMOE_BS = 8  # BS8 kernel serves M <= 8

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
            "Failed to load the MonoMoe kernel CUDA extension via JIT. Ensure "
            "a Hopper (SM90a) GPU and the CUDA toolkit are available and that "
            "csrc/fused_moe/monomoe/ sources exist.\n"
            f"Error: {e}"
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

    Sourced from the C++ ``sizeof(MoEGemmSpec<Dims>)`` so the buffer can never
    desync from the kernel's struct layout.
    """
    mod = _get_monomoe_module()
    return int(mod.monomoe_scratchpad_size())


@flashinfer_api
def alloc_scratchpad(device: torch.device) -> torch.Tensor:
    """Allocate a zero-initialized scratchpad on ``device`` for the kernel.

    Returns a 1-D ``uint8`` tensor sized to :func:`get_scratchpad_size_bytes`.
    The zero fill establishes the kernel's handoff invariants (the 0.0f
    activation-scale sentinel, launch parity counters, and readiness flags —
    see docs/design_docs/monomoe_kernel.md §2/§4); afterwards the kernel
    self-maintains them, so allocate once and reuse the same tensor for every
    :func:`mono_moe` invocation.
    """
    nbytes = get_scratchpad_size_bytes()
    return torch.zeros(nbytes, dtype=torch.uint8, device=device)


@flashinfer_api
def interleave_for_tma_wgmma_up(w_fp8: torch.Tensor) -> torch.Tensor:
    """Repack fp8 up-projection weights for the Pair_Layout WGMMA A-tile.

    Under the pair layout, each warp's 16-row SHM stripe holds 8 gate rows
    and 8 up rows, so ``silu(gate) * up`` becomes a per-lane register
    operation after the WGMMA (no cross-warp exchange).

    Input layout: ``[E, 2*N, K]`` row-major fp8 — the first ``N`` rows per
    expert are gate weights, the last ``N`` are up weights.  ``N`` must be a
    multiple of 64.

    Output layout (still ``[E, 2*N, K]``, identical byte footprint): for
    every expert ``e`` and every 64-gate-row block ``b``, the 128-row slab
    at ``128*b`` packs, per warpgroup ``wg`` and warp ``w``::

        rows [wg*64 + w*16     .. +8) = gate[e, 64b + wg*32 + w*8 .. +8, :]
        rows [wg*64 + w*16 + 8 .. +8) =   up[e, 64b + wg*32 + w*8 .. +8, :]

    Under SWZ128 the TMA applies the 8-row x 128-byte core-matrix XOR swizzle
    automatically, so this only rearranges GM rows (no byte-level
    permutation).  The result is cached on the input tensor as
    ``_tma_interleaved_up``.

    The down-projection weights need no preparation — the raw ``[E, K, N]``
    row-major fp8 tensor is passed straight through.

    Parameters
    ----------
    w_fp8 : torch.Tensor
        FP8 up/gate weight tensor with shape [E, 2*N, K] (row-major).

    Returns
    -------
    torch.Tensor
        Repacked weight tensor with the same shape and dtype as *w_fp8*.
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

    # Per warpgroup (32 rows) and warp (8 rows): gate(8) then up(8) = one
    # 16-row warp stripe; 4 warps per WG, 2 WGs per 128-row slab.
    gate_wg0 = gate_r[:, :, :32, :].reshape(E, blocks, 4, 8, K)
    gate_wg1 = gate_r[:, :, 32:, :].reshape(E, blocks, 4, 8, K)
    up_wg0 = up_r[:, :, :32, :].reshape(E, blocks, 4, 8, K)
    up_wg1 = up_r[:, :, 32:, :].reshape(E, blocks, 4, 8, K)

    wg0 = torch.stack([gate_wg0, up_wg0], dim=3).reshape(E, blocks, 64, K)
    wg1 = torch.stack([gate_wg1, up_wg1], dim=3).reshape(E, blocks, 64, K)
    result = torch.cat([wg0, wg1], dim=2).reshape(E, blocks * 128, K).contiguous()

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
    """Validate the input tensors against the fixed E=256/N=512/K=2048 shape.

    Confirms every operand's extents match the hard-specialized shape and
    returns the token count ``m``.  A different shape is refused.
    """
    E, N, K = _MONOMOE_E, _MONOMOE_N, _MONOMOE_K

    # Explicit raises (not assert): these validate user-provided tensor
    # shapes that, if wrong, let the fixed-shape CUDA kernel read/write out
    # of bounds.  `assert` would be stripped under `python -O`.
    if activations_in.dim() != 2:
        raise ValueError(f"activations_in must be [M, K], got {activations_in.dim()}D")
    if expert_weights_down.dim() != 3:
        raise ValueError(
            f"expert_weights_down must be [E, K, N], got {expert_weights_down.dim()}D"
        )
    if expert_weights_up.dim() != 3:
        raise ValueError(
            f"expert_weights_up must be [E, 2*N, K], got {expert_weights_up.dim()}D"
        )

    # (E, N, K) come from the down weights [E, K, N]; must equal the fixed
    # shape the kernel is specialized for.
    de, dk, dn = (int(x) for x in expert_weights_down.shape)
    if (de, dn, dk) != (E, N, K):
        raise ValueError(
            f"monomoe: unsupported shape (E={de}, N={dn}, K={dk}). This kernel "
            f"is hard-specialized to E={E}, N={N}, K={K} only."
        )

    m = activations_in.size(0)
    if m > _MONOMOE_BS:
        raise ValueError(f"this kernel caps tokens at {_MONOMOE_BS}; got M={m}")
    if activations_in.size(1) != K:
        raise ValueError(f"activations K must be {K}, got {activations_in.size(1)}")
    if tuple(router_logits.shape) != (m, E):
        raise ValueError(
            f"router_logits must be [{m}, {E}], got {tuple(router_logits.shape)}"
        )

    # Up weights: [E, 2*N, K]; up scales block-wise [E, 2N/128, K/128].
    if tuple(expert_weights_up.shape) != (E, 2 * N, K):
        raise ValueError(
            f"expert_weights_up must be [{E}, {2 * N}, {K}], got {tuple(expert_weights_up.shape)}"
        )
    if tuple(expert_scales_up.shape) != (E, (2 * N) // _BLOCK, K // _BLOCK):
        raise ValueError(
            f"expert_scales_up must be [{E}, {(2 * N) // _BLOCK}, {K // _BLOCK}], "
            f"got {tuple(expert_scales_up.shape)}"
        )
    # Down scales block-wise [E, K/128, N/128].
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
    expert_bias: Optional[torch.Tensor] = None,
    routed_scaling_factor: float = 1.0,
    out: Optional[torch.Tensor] = None,
    scratchpad: Optional[torch.Tensor] = None,
    interleave_up: bool = True,
) -> bool:
    """Backend-requirement check for :func:`mono_moe`.

    Carries the ``@supported_compute_capability([90])`` annotation so the
    ``@backend_requirement`` wrapper rejects any non-Hopper device with a
    clear ``BackendSupportedError`` *before* the SM90a-only kernel is JIT
    compiled.  Also runs the fixed-shape contract validation so shape and
    architecture support are decided in one place.  The signature mirrors
    :func:`mono_moe` because ``@backend_requirement`` forwards all kwargs.
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
    expert_bias: Optional[torch.Tensor] = None,
    routed_scaling_factor: float = 1.0,
    out: Optional[torch.Tensor] = None,
    scratchpad: Optional[torch.Tensor] = None,
    interleave_up: bool = True,
) -> torch.Tensor:
    """Single-kernel block-FP8 top-K MoE (fixed E256/N512/K2048 shape, SM90a).

    The ``(E, N, K)`` shape is derived from the weight tensors and must be the
    fixed E=256/N=512/K=2048 shape the kernel is specialized for; up to 8
    tokens (the BS8 kernel).

    Args:
        activations_in: bf16 input activations ``[M, K]`` (``M <= 8``).
        router_logits: bf16 router logits ``[M, E]``.
        expert_weights_up: fp8_e4m3 up/gate weights ``[E, 2*N, K]``. This
            function applies :func:`interleave_for_tma_wgmma_up` by default;
            pass ``interleave_up=False`` if the tensor is already interleaved.
        expert_scales_up: fp32 block-wise scales ``[E, 2N/128, K/128]``.
        expert_weights_down: fp8_e4m3 down weights ``[E, K, N]`` (raw row-major).
        expert_scales_down: fp32 block-wise scales ``[E, K/128, N/128]``.
        top_k: experts selected per token (1..8).
        scoring_func: ``"sigmoid"`` or ``"softmax"``.
        renormalize: renormalize the top-K weights to sum to 1.
        expert_bias: optional fp32 per-expert selection bias ``[E]``
            (GLM-style noaux_tc routing; sigmoid scoring only).  Winners are
            ranked by ``sigmoid(logit) + bias`` while the routing weight
            stays the unbiased sigmoid.
        routed_scaling_factor: scalar folded into every routing weight.
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
    E, K = _MONOMOE_E, _MONOMOE_K

    if expert_bias is not None:
        if scoring_func != "sigmoid":
            raise ValueError("expert_bias requires scoring_func='sigmoid'")
        if tuple(expert_bias.shape) != (E,):
            raise ValueError(
                f"expert_bias must be [{E}], got {tuple(expert_bias.shape)}"
            )

    # The kernel reads the up weights in the interleaved Pair_Layout.
    if interleave_up:
        expert_weights_up = interleave_for_tma_wgmma_up(expert_weights_up)

    if out is None:
        out = torch.empty(m, K, dtype=torch.bfloat16, device=activations_in.device)
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
        expert_bias,
        float(routed_scaling_factor),
    )
    return out
