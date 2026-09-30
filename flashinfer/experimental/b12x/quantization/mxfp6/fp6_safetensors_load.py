"""Load a b12x MX-FP6 (W6A6) safetensors checkpoint into kernel-ready weights.

Read side of :mod:`b12x.quantization.mxfp6.fp6_safetensors_export`. Given
the per-expert ModelOpt-mirror keys (``.weight`` / ``.weight_scale`` /
``.weight_scale_2`` / ``.input_scale``), this reconstructs the exact
:class:`~b12x.quantization.mxfp6.fp6_moe_weights.FP6MoEWeights` layout
that the fused FP6 MoE path consumes:

* FC1 codes are row-stacked ``[up; gate]`` (the kernel's gated-silu convention).
* Block scales are stored **unswizzled** on disk and swizzled here via the same
  :func:`swizzle_block_scale` used by the offline quantizer, so the result is
  byte-identical to the validated ``.pt`` path.

Because the per-row FP6 packing and per-block UE8M0 scale are computed
independently, quantizing ``gate``/``up`` separately on disk and re-stacking here
reproduces the single-matrix offline path bit-for-bit.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch

from b12x._lib.intrinsics import swizzle_block_scale
from .fp6_checkpoint import (
    INPUT_SCALE_SUFFIX,
    WEIGHT_SCALE_2_SUFFIX,
    WEIGHT_SCALE_SUFFIX,
    WEIGHT_SUFFIX,
    activation_format_for_source,
    weight_format_for_source,
)
from .fp6_dense_weights import FP6DenseWeight
from .fp6_moe_weights import FP6MoEWeights

_UNIT_TOL = 1e-3


def _swizzle_stacked(unswizzled: torch.Tensor) -> torch.Tensor:
    """Swizzle stacked ``(E, rows, blocks)`` UE8M0 bytes into the cutlass layout.

    Identical call to the offline ``_swizzled_block_scales`` (minus the block-max
    step, since the bytes already hold the UE8M0 exponents), so the output matches
    the ``.pt`` artifact byte-for-byte.
    """
    return swizzle_block_scale(unswizzled.view(torch.float8_e8m0fnu)).view(torch.uint8)


def _source_format_from_config(quant_config: dict) -> str:
    """Reconstruct a ``source_format`` selector from a ``quantization_config`` block."""
    wf = str(quant_config.get("weight_format", "e2m3")).lower()
    af = str(quant_config.get("activation_format", "e3m2")).lower()
    if wf == "e2m3" and af == "e3m2":
        return "mxfp6_default"
    if wf == "e2m3" and af == "e2m3":
        return "mxfp6_e2m3"
    if wf == "e3m2" and af == "e3m2":
        return "mxfp6_e3m2"
    if wf == "e2m3" and af == "e4m3":
        # W6A8: unchanged E2M3 weight bytes, FP8 E4M3 runtime activations.
        # Enabled on an existing E2M3-weight checkpoint by editing config.json.
        return "mxfp6_w6a8"
    return "mxfp6_default"


def load_fp6_moe_weights_from_safetensors(
    get_tensor: Callable[[str], torch.Tensor],
    prefix: str,
    num_experts: int,
    *,
    activation: str = "silu",
    source_format: str = "mxfp6_default",
    device: torch.device | str = "cuda",
    gate_name: str = "gate_proj",
    up_name: str = "up_proj",
    down_name: str = "down_proj",
) -> FP6MoEWeights:
    """Reconstruct :class:`FP6MoEWeights` for one MoE layer from per-expert keys.

    ``get_tensor(key)`` returns the on-disk tensor for a full key. ``prefix`` is
    the per-layer module path (e.g. ``model...layers.3.mlp``); expert keys are
    read as ``{prefix}.experts.{e}.{proj}.{weight|weight_scale|...}``.
    """
    is_gated = activation == "silu"
    weight_fmt = weight_format_for_source(source_format)

    w1_codes: list[torch.Tensor] = []
    w2_codes: list[torch.Tensor] = []
    w1_scales: list[torch.Tensor] = []
    w2_scales: list[torch.Tensor] = []
    w1_gs: list[torch.Tensor] = []
    w2_gs: list[torch.Tensor] = []
    a1_is: list[torch.Tensor] = []
    a2_is: list[torch.Tensor] = []

    for e in range(num_experts):
        base = f"{prefix}.experts.{e}"
        gate_w = get_tensor(f"{base}.{gate_name}{WEIGHT_SUFFIX}")  # (N, 3K/4) uint8
        gate_s = get_tensor(f"{base}.{gate_name}{WEIGHT_SCALE_SUFFIX}")  # (N, K/32)
        down_w = get_tensor(f"{base}.{down_name}{WEIGHT_SUFFIX}")  # (K, 3N/4)
        down_s = get_tensor(f"{base}.{down_name}{WEIGHT_SCALE_SUFFIX}")  # (K, N/32)
        if is_gated:
            up_w = get_tensor(f"{base}.{up_name}{WEIGHT_SUFFIX}")
            up_s = get_tensor(f"{base}.{up_name}{WEIGHT_SCALE_SUFFIX}")
            fc1_w = torch.cat([up_w, gate_w], dim=0)  # [up; gate] (2N, 3K/4)
            fc1_s = torch.cat([up_s, gate_s], dim=0)  # (2N, K/32) unswizzled
        else:
            fc1_w = gate_w
            fc1_s = gate_s
        w1_codes.append(fc1_w)
        w2_codes.append(down_w)
        w1_scales.append(fc1_s)
        w2_scales.append(down_s)
        w1_gs.append(get_tensor(f"{base}.{gate_name}{WEIGHT_SCALE_2_SUFFIX}").reshape(()))
        w2_gs.append(get_tensor(f"{base}.{down_name}{WEIGHT_SCALE_2_SUFFIX}").reshape(()))
        a1_is.append(get_tensor(f"{base}.{gate_name}{INPUT_SCALE_SUFFIX}").reshape(()))
        a2_is.append(get_tensor(f"{base}.{down_name}{INPUT_SCALE_SUFFIX}").reshape(()))

    dev = torch.device(device)
    w1_fp6 = torch.stack(w1_codes, dim=0).to(dev).contiguous()
    w2_fp6 = torch.stack(w2_codes, dim=0).to(dev).contiguous()
    w1_blockscale = _swizzle_stacked(torch.stack(w1_scales, dim=0)).to(dev).contiguous()
    w2_blockscale = _swizzle_stacked(torch.stack(w2_scales, dim=0)).to(dev).contiguous()

    w1_gs_t = torch.stack(w1_gs).float()
    w2_gs_t = torch.stack(w2_gs).float()
    a1_is_t = torch.stack(a1_is).float()
    a2_is_t = torch.stack(a2_is).float()
    if not (
        torch.allclose(w1_gs_t, torch.ones_like(w1_gs_t), atol=_UNIT_TOL)
        and torch.allclose(w2_gs_t, torch.ones_like(w2_gs_t), atol=_UNIT_TOL)
        and torch.allclose(a1_is_t, torch.ones_like(a1_is_t), atol=_UNIT_TOL)
        and torch.allclose(a2_is_t, torch.ones_like(a2_is_t), atol=_UNIT_TOL)
    ):
        raise NotImplementedError(
            "Non-unit weight_scale_2/input_scale found; this loader only supports the "
            "pure-MX W6A6 contract (unit global scales). Per-expert alpha wiring is TODO."
        )

    # Pure-MX contract: per-block UE8M0 carries the range, so dequant alphas and
    # activation global scales are all 1.0 (matches the validated FP6MoEWeights).
    experts = int(num_experts)
    n = int(w2_fp6.shape[2] * 4 // 3)
    k = int(w2_fp6.shape[1])
    ones_e = torch.ones(experts, dtype=torch.float32, device=dev)
    ones_1 = torch.ones(1, dtype=torch.float32, device=dev)
    return FP6MoEWeights(
        w1_fp6=w1_fp6,
        w1_blockscale=w1_blockscale,
        w1_alphas=ones_e.clone(),
        w2_fp6=w2_fp6,
        w2_blockscale=w2_blockscale,
        w2_alphas=ones_e.clone(),
        a1_gscale=ones_1.clone(),
        a2_gscale=ones_1.clone(),
        num_experts=experts,
        k=k,
        n=n,
        weight_fmt=weight_fmt,
        source_format=source_format,
        activation=activation,
    )


def load_fp6_moe_checkpoint(
    model_path: str,
    *,
    activation: str = "silu",
    device: torch.device | str = "cuda",
    limit_layers: Optional[int] = None,
) -> dict[int, FP6MoEWeights]:
    """Load every routed-MoE layer of an FP6 safetensors checkpoint.

    Returns ``{layer_index: FP6MoEWeights}``. ``source_format`` is recovered from
    ``config.json``'s ``quantization_config``; ``activation`` defaults to silu.
    """
    from .model_fp6 import SafetensorsModel, discover_moe_experts

    model = SafetensorsModel(model_path)
    quant_config = model.config.get("quantization_config", {})
    source_format = _source_format_from_config(quant_config)
    scheme = discover_moe_experts(model)
    if scheme is None:
        raise ValueError(f"no FP6 MoE experts discovered under {model_path}")

    layers = scheme.layers if limit_layers is None else scheme.layers[:limit_layers]
    out: dict[int, FP6MoEWeights] = {}
    for layer in layers:
        prefix = scheme.prefix_template.format(L=layer)
        out[layer] = load_fp6_moe_weights_from_safetensors(
            model.get_tensor,
            prefix,
            scheme.num_experts,
            activation=activation,
            source_format=source_format,
            device=device,
            gate_name=scheme.gate_name,
            up_name=scheme.up_name,
            down_name=scheme.down_name,
        )
    return out


def load_fp6_dense_weight_from_safetensors(
    get_tensor: Callable[[str], torch.Tensor],
    name: str,
    *,
    source_format: str = "mxfp6_default",
    device: torch.device | str = "cuda",
) -> FP6DenseWeight:
    """Reconstruct an :class:`FP6DenseWeight` for one linear from its FP6 keys.

    Reads ``{name}.weight`` / ``.weight_scale`` / ``.weight_scale_2``, swizzles the
    unswizzled on-disk UE8M0 scales into the cutlass layout
    :meth:`FP6DenseWeight.scale_view` expects, and carries the (unit) weight global
    scale. The result drops straight into :func:`dense_fp6_linear`, whose
    ``alpha = 1/(a_gscale * weight_global_scale)`` is correct for the unit scale.
    """
    packed = get_tensor(name + WEIGHT_SUFFIX)  # (out, 3*in/4) uint8
    wscale = get_tensor(name + WEIGHT_SCALE_SUFFIX)  # (out, in/32) uint8, unswizzled
    wgs = get_tensor(name + WEIGHT_SCALE_2_SUFFIX).reshape(1).float()
    if not torch.allclose(wgs, torch.ones_like(wgs), atol=_UNIT_TOL):
        raise NotImplementedError(
            "Non-unit weight_scale_2 found; the dense FP6 loader only supports the "
            "pure-MX W6A6 contract (unit weight global scale)."
        )
    out_f, packed_in = int(packed.shape[0]), int(packed.shape[1])
    in_f = packed_in * 4 // 3
    fmt = weight_format_for_source(source_format)
    act_fmt = activation_format_for_source(source_format)
    scale_storage = (
        swizzle_block_scale(wscale.view(torch.float8_e8m0fnu)).reshape(-1).view(torch.uint8)
    )
    dev = torch.device(device)
    return FP6DenseWeight(
        packed=packed.to(dev).contiguous(),
        scale_storage=scale_storage.to(dev).contiguous(),
        global_scale=wgs.to(dev),
        out_features=out_f,
        in_features=in_f,
        fmt=fmt,
        act_fmt=act_fmt,
    )


def load_fp6_dense_checkpoint(
    model_path: str,
    *,
    device: torch.device | str = "cuda",
) -> dict[str, FP6DenseWeight]:
    """Load every FP6-quantized dense linear from a safetensors checkpoint.

    Keys are detected by the presence of a ``.weight_scale`` sibling next to a
    ``.weight`` tensor, so this is format-driven (works regardless of module
    naming). Returns ``{module_name: FP6DenseWeight}``.
    """
    from .model_fp6 import SafetensorsModel

    model = SafetensorsModel(model_path)
    quant_config = model.config.get("quantization_config", {})
    source_format = _source_format_from_config(quant_config)
    out: dict[str, FP6DenseWeight] = {}
    for key in model.keys():
        if not key.endswith(WEIGHT_SUFFIX):
            continue
        name = key[: -len(WEIGHT_SUFFIX)]
        if not model.has(name + WEIGHT_SCALE_SUFFIX):
            continue
        out[name] = load_fp6_dense_weight_from_safetensors(
            model.get_tensor, name, source_format=source_format, device=device
        )
    return out
