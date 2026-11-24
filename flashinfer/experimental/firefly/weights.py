# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight preparation for ``CuteMoE``.

Copies of the vLLM functions used when an NVFP4 MoE layer is loaded, byte-compatible with them (``interleave_gate_up``
with this package's row order):

- ``quantize_expert_weights``: BF16 experts -> NVFP4 (vLLM's ``_quantize_moe_weight_to_nvfp4``; FlashInfer's
  ``nvfp4_quantize`` with linear scales), as a ModelOpt checkpoint stores them.
- ``swizzle_blockscale``: the 128x4 tcgen05 scale-factor layout the kernels read.
- ``interleave_gate_up``: gate/up rows interleaved per 8 rows, so that a 128-row tile holds the gate and up rows of
  64 outputs and a SwiGLU epilogue thread reads each pair from TMEM lanes l and l + 8.

Added here: ``prepare_weights`` (BF16 experts -> the six tensors ``CuteMoE`` takes), also from FlashInfer's canonical
source layout (``w1_order="up_gate"``: ``MoEWeightPack``'s ``w1`` rows ``[up, gate]``, as ``XxxConfig.prepare_weights``
take them). The global scale stays one per expert from its amax (ModelOpt's; FlashInfer 0.7.5's TRT-LLM / prims_ts
``prepare_weights`` quantize BF16 with a unit global scale instead), so the values differ from FlashInfer-prepared
weights; checkpoint NVFP4 tensors go through ``kernel_weights`` (also with ``w1_order``). For FlashInfer's unified API:
``prepare_view`` (the canonical BF16 experts and ``CuteDslConfig.prepare_weights``' keyword arguments ->
our weight view, built once at model load), ``weight_view`` / ``view_weights`` (``KernelWeights`` <-> view) and
``check_weights``; ``CuteMoE`` takes the view as its weights.

Conventions: gate/up weights ``[E, 2I, H]`` with the gate rows (w1) first and the up rows (w3) second; down
weights ``[E, H, I]``. Needs FlashInfer (quantization); the kernels do not.
"""

from contextlib import nullcontext
from typing import NamedTuple

import torch

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = 448.0


def _weight_amax(weight: torch.Tensor, *, dim: int) -> torch.Tensor:
    lo, hi = weight.aminmax(dim=dim)
    return torch.maximum(lo.abs(), hi.abs())


def quantize_expert_weights(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize stacked MoE expert weights ``(E, N, K)`` to NVFP4.

    One FP32 global scale per expert plus per-block (group-16) FP8 scales, matching the ModelOpt NVFP4
    checkpoint layout. Returns packed FP4 weights ``(E, N, K // 2)``, block scales ``(E, N, K // 16)`` and the
    per-expert global scale ``(E,)`` stored as ``1 / (amax / (fp4_max * fp8_max))``, which is the convention the
    kernels multiply by.
    """
    from flashinfer import SfLayout, nvfp4_quantize

    if weight.dim() != 3:
        raise ValueError(f"expected 3D expert weights, got {tuple(weight.shape)}")
    k = weight.shape[-1]
    if k % 16:
        raise ValueError(f"last dim must be a multiple of 16, got {k}")

    amax = _weight_amax(weight.flatten(1), dim=-1).to(torch.float32).clamp_min(1e-8)
    global_scale = (FLOAT4_E2M1_MAX * FLOAT8_E4M3_MAX) / amax
    weight_scale_2 = (1.0 / global_scale).to(torch.float32)

    weight = weight.contiguous()
    # on the weights' GPU (the quantizer allocates on the current one); plain NVFP4 whatever the
    # FLASHINFER_NVFP4_4OVER6 environment (None: off)
    with torch.cuda.device(weight.device) if weight.is_cuda else nullcontext():
        pairs = [
            nvfp4_quantize(
                expert, scale, sfLayout=SfLayout.layout_linear, nvfp4_4over6=None
            )
            for expert, scale in zip(weight, global_scale, strict=True)
        ]
    return (
        torch.stack([q for q, _ in pairs]),
        torch.stack([sf.view(torch.float8_e4m3fn) for _, sf in pairs]),
        weight_scale_2,
    )


def round_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


def swizzle_blockscale(scale: torch.Tensor) -> torch.Tensor:
    """Pad and block-interleave FP4 block scales into the 128x4 MMA layout."""
    if scale.dtype != torch.float8_e4m3fn:
        raise TypeError(
            f"swizzle_blockscale expects torch.float8_e4m3fn, got {scale.dtype}"
        )
    ndim = scale.ndim
    if ndim == 2:
        scale = scale.unsqueeze(0)
    if scale.ndim != 3:
        raise ValueError("Expected a 2-D or 3-D tensor for block scales.")
    b, m, k = scale.shape
    m_padded, k_padded = round_up(m, 128), round_up(k, 4)
    padded = torch.zeros(
        (b, m_padded, k_padded), dtype=scale.dtype, device=scale.device
    )
    padded[:, :m, :k] = scale
    padded = padded.reshape(b, m_padded // 128, 4, 32, k_padded // 4, 4)
    swizzled = padded.permute(0, 1, 4, 3, 2, 5).contiguous()
    if ndim == 2:
        return swizzled.reshape(m_padded, k_padded)
    return swizzled.reshape(b, m_padded, k_padded)


def interleave_gate_up(weight: torch.Tensor) -> torch.Tensor:
    """[gate, up] rows -> interleaved per 8: rows 16 q .. 16 q + 7 the gate rows of outputs 8 q .. 8 q + 7, then their up
    rows (a 128-row tile: the 64 outputs 64 t ..; the SwiGLU epilogues read gate and up of an output from TMEM lanes
    l and l + 8, ``tcgen05.ld.16x256b``)."""
    e, n, k = weight.shape
    if n % 128:
        raise ValueError(f"gate/up row count must be a multiple of 128, got {n}")
    return weight.reshape(e, 2, n // 16, 8, k).transpose(1, 2).reshape(e, n, k)


def swap_halves(t: torch.Tensor) -> torch.Tensor:
    """[a, b] rows (dim 1) -> [b, a]: FlashInfer's canonical [up, gate] gate/up rows <-> this package's [gate, up]."""
    half = t.shape[1] // 2
    return torch.cat((t[:, half:], t[:, :half]), dim=1)


def kernel_weights(
    w1: torch.Tensor,
    w1_sf: torch.Tensor,
    w2_sf: torch.Tensor,
    w1_order: str = "gate_up",
    gated: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gate/up FP4 weights with interleaved rows, their swizzled block scales, and the down weights' swizzled block
    scales (inputs: ``quantize_expert_weights`` outputs or a checkpoint's NVFP4 tensors with linear block scales; down
    weights and global scales are used as quantized). ``w1_order``: the gate/up rows' order, "gate_up" or "up_gate"
    (FlashInfer's canonical order). ``gated=False`` (a non-gated activation, ``activation.ReLU2``): w1 has the I up rows
    only, kept in order (``w1_order`` does not apply)."""
    if w1_order not in ("gate_up", "up_gate"):
        raise ValueError(f"w1_order={w1_order!r}: 'gate_up' or 'up_gate'")
    if not gated:
        return w1, swizzle_blockscale(w1_sf), swizzle_blockscale(w2_sf)
    if w1_order == "up_gate":
        w1, w1_sf = swap_halves(w1), swap_halves(w1_sf)
    return (
        interleave_gate_up(w1),
        swizzle_blockscale(interleave_gate_up(w1_sf)),
        swizzle_blockscale(w2_sf),
    )


class KernelWeights(NamedTuple):
    """The weight arguments of ``CuteMoE``, in its order."""

    w1: torch.Tensor  # [E, 2I, H / 2] FP4 (uint8), gate/up rows interleaved per 8 (non-gated: [E, I, H / 2])
    w1_sf: torch.Tensor  # swizzled E4M3 block scales of w1
    w1_gs: torch.Tensor  # [E] FP32 global scales (1 / global_scale)
    w2: torch.Tensor  # [E, H, I / 2] FP4 (uint8)
    w2_sf: torch.Tensor  # swizzled E4M3 block scales of w2
    w2_gs: torch.Tensor  # [E] FP32


def prepare_weights(
    w1: torch.Tensor,
    w2: torch.Tensor,
    chunk: int = 16,
    w1_order: str = "gate_up",
    gated: bool = True,
    device=None,
) -> KernelWeights:
    """BF16 experts (gate/up ``[E, 2I, H]``, down ``[E, H, I]``) -> ``CuteMoE``'s weights. ``gated=False`` (``ReLU2``):
    w1 ``[E, I, H]``, the up rows only. ``w1_order``: "gate_up" (the
    gate rows first, this package's and the checkpoints' order) or "up_gate" (FlashInfer's canonical order,
    ``act(x @ w1[I:].T) * (x @ w1[:I].T)``); the halves are swapped per chunk, and the global scale (per expert over
    both halves) does not depend on the order: the same tensors either way.

    Quantizes ``chunk`` experts at a time to bound temporary memory (``device``: each chunk is moved there first, so
    the BF16 experts can stay on the host)."""
    if w1_order not in ("gate_up", "up_gate"):
        raise ValueError(f"w1_order={w1_order!r}: 'gate_up' or 'up_gate'")
    q1: list[torch.Tensor] = []
    s1: list[torch.Tensor] = []
    g1: list[torch.Tensor] = []
    q2: list[torch.Tensor] = []
    s2: list[torch.Tensor] = []
    g2: list[torch.Tensor] = []
    for e0 in range(0, w1.shape[0], chunk):
        w1c, w2c = w1[e0 : e0 + chunk], w2[e0 : e0 + chunk]
        if device is not None:
            w1c, w2c = w1c.to(device), w2c.to(device)
        if w1_order == "up_gate" and gated:
            w1c = swap_halves(w1c)
        for (q, s, g), w in (((q1, s1, g1), w1c), ((q2, s2, g2), w2c)):
            a, b, c = quantize_expert_weights(w)
            q.append(a)
            s.append(b)
            g.append(c)
    w1q, w1s, w1g = torch.cat(q1), torch.cat(s1), torch.cat(g1)
    w2q, w2s, w2g = torch.cat(q2), torch.cat(s2), torch.cat(g2)
    k_w1, k_sf1, k_sf2 = kernel_weights(w1q, w1s, w2s, gated=gated)
    return KernelWeights(k_w1, k_sf1, w1g, w2q, k_sf2, w2g)


# ----------------------------------------------------------------------------------- FlashInfer's weight view


def check_weights(
    w: KernelWeights,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    gated: bool = True,
    device=None,
) -> None:
    """Raise ValueError unless ``w`` holds the weights of ``num_experts`` experts of (H, I) in the kernels' layout:
    the shapes, one-byte FP4 / E4M3 elements and FP32 global scales, contiguous at 16-byte aligned addresses (the
    kernels read them through the TMA and with 16-byte loads), on ``device`` if given. Checked, never copied: the
    weights are prepared once (``prepare_view``) and a call only passes their addresses."""
    E, H, I = num_experts, hidden_size, intermediate_size
    n1 = 2 * I if gated else I
    want = dict(
        w1=((E, n1, H // 2), 1),
        w1_sf=(E * round_up(n1, 128) * round_up(H // 16, 4), 1),
        w1_gs=((E,), 4),
        w2=((E, H, I // 2), 1),
        w2_sf=(E * round_up(H, 128) * round_up(I // 16, 4), 1),
        w2_gs=((E,), 4),
    )
    for name, t in zip(KernelWeights._fields, w, strict=True):
        shape, size = want[name]
        if isinstance(shape, tuple):
            bad = (
                tuple(t.shape) != shape and f"shape {tuple(t.shape)}, expected {shape}"
            )
        else:  # swizzled block scales: [E, rows padded to 128, K / 16 padded to 4], any view of them
            bad = t.numel() != shape and f"{t.numel()} elements, expected {shape}"
        bad = (
            bad
            or (
                t.element_size() != size
                and f"{t.element_size()}-byte elements, expected {size}"
            )
            or (
                size == 4
                and t.dtype != torch.float32
                and f"dtype {t.dtype}, expected torch.float32"
            )
            or (not t.is_contiguous() and "not contiguous")
            or (t.data_ptr() % 16 and "not 16-byte aligned")
            or (
                device is not None
                and t.device != torch.device(device)
                and f"on {t.device}, expected {device}"
            )
        )
        if bad:
            raise ValueError(
                f"weights.{name}: {bad} (E={E}, H={H}, I={I}, {'gated' if gated else 'non-gated'})"
            )


def weight_view(w: KernelWeights, gated: bool = True) -> dict:
    """``CuteMoE``'s weights as a FlashInfer weight view (``MoEWeightPack.prepare_for(<backend key>, view)``): the six
    tensors under ``KernelWeights``' field names, and ``gated`` (checked against the layer's activation). From
    checkpoint NVFP4 tensors (linear block scales): ``k_w1, k_sf1, k_sf2 = kernel_weights(w1, w1_sf, w2_sf, w1_order=,
    gated=)``, then ``weight_view(KernelWeights(k_w1, k_sf1, w1_gs, w2, k_sf2, w2_gs), gated)``; from BF16 experts:
    ``prepare_view``."""
    check_weights(w, w.w1.shape[0], w.w1.shape[2] * 2, w.w2.shape[2] * 2, gated)
    return {**w._asdict(), "gated": bool(gated)}


def view_weights(view) -> KernelWeights:
    """The ``KernelWeights`` of a weight view (``weight_view`` / ``prepare_view``)."""
    return KernelWeights(
        view["w1"],
        view["w1_sf"],
        view["w1_gs"],
        view["w2"],
        view["w2_sf"],
        view["w2_gs"],
    )


def check_nvfp4_4over6(setting):
    """Reject FlashInfer's NVFP4 4over6 recipe (``QuantConfig.nvfp4_4over6``, ``prepare_weights(nvfp4_4over6=)``):
    raise NotImplementedError when ``setting`` resolves to a recipe (``flashinfer.resolve_nvfp4_4over6``: an unset
    field reads the ``FLASHINFER_NVFP4_4OVER6*`` environment), return for None (4over6 off) and for an unset field with
    the environment off. 4over6 picks each block's E4M3 scale between amax / 6 and amax / 4 by the quantization error;
    in cute_dsl it applies only to its per-token GEMM2-input quantization, which this backend does not have (its
    intermediate's scales are described in ``moe``). The MoE input arrives quantized: whatever recipe made it, its format is the
    same. For the runner's ``_check_support`` (MoELayer then picks another backend); the runner declares
    ``supports_nvfp4_4over6 = True``, since with False FlashInfer's base class also rejects an explicit None (it
    assumes a quantizer that reads the environment)."""
    if setting is None:
        return
    from flashinfer import resolve_nvfp4_4over6

    recipe = resolve_nvfp4_4over6(setting)
    if recipe is not None:
        raise NotImplementedError(
            f"NVFP4 4over6 ({recipe!r}) is not supported: the intermediate is quantized with "
            "the standard NVFP4 recipe only"
        )


def prepare_view(
    w1_bf16: torch.Tensor,
    w2_bf16: torch.Tensor,
    *,
    quant=None,
    num_local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    activation=None,
    device=None,
    nvfp4_4over6=None,
) -> dict:
    """This package's FlashInfer weight view from the canonical BF16 experts every ``XxxConfig.prepare_weights`` takes
    (``w1`` [E, 2I, H] with the rows [up, gate] for a gated activation, [E, I, H] for ``ReLU2``; ``w2`` [E, H, I]),
    with the keyword arguments of ``CuteDslConfig.prepare_weights`` (its ``swap_ab`` has no counterpart here): what
    our config's ``prepare_weights`` returns, registered once at model load. A call passes the view's tensors to the
    kernels as they are (``CuteMoE`` takes the view as its ``weights``): nothing is converted per call.

    Not the cute_dsl view: that one interleaves the gate/up rows per 64 (16 with swap_ab), keeps its block scales in
    ``convert_sf_to_mma_layout``'s layout and quantizes with a unit global scale; ours interleaves per 8, swizzles the
    scales 128x4 and has one global scale per expert from its amax (``prepare_weights``, ModelOpt's convention).

    ``quant``: NVFP4 weights and activations only. ``nvfp4_4over6`` (cute_dsl: selects the view's ``fc2_input_scale``;
    it must equal the layer's ``QuantConfig.nvfp4_4over6``) must resolve to None (``check_nvfp4_4over6``): 4over6 is
    not supported. So the view has no ``fc2_input_scale`` (it would matter only for cute_dsl's per-token GEMM2 input)."""
    from .activation import as_activation, is_gated

    if quant is not None:
        pair = tuple(getattr(f, "name", f) for f in (quant.weight, quant.activation))
        if pair != ("NVFP4", "NVFP4"):
            raise ValueError(f"NVFP4 weights and activations only, got {pair}")
    check_nvfp4_4over6(nvfp4_4over6)
    gated = is_gated(as_activation(activation))
    E, H, I = num_local_experts, hidden_size, intermediate_size
    n1 = 2 * I if gated else I
    if tuple(w1_bf16.shape) != (E, n1, H) or tuple(w2_bf16.shape) != (E, H, I):
        raise ValueError(
            f"w1 {tuple(w1_bf16.shape)} / w2 {tuple(w2_bf16.shape)}, expected {(E, n1, H)} / {(E, H, I)} "
            f"({'gated' if gated else 'non-gated'})"
        )
    if device is None:
        device = (
            w1_bf16.device
            if w1_bf16.is_cuda
            else torch.device("cuda", torch.cuda.current_device())
        )
    w = prepare_weights(
        w1_bf16, w2_bf16, w1_order="up_gate", gated=gated, device=device
    )
    return weight_view(w, gated)
