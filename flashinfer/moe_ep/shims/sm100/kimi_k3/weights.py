"""Canonical FP4 weights to the isolated Kimi layout."""

from dataclasses import dataclass

import torch

from flashinfer.moe_ep.weights import PrequantizedMoEWeights


@dataclass(frozen=True)
class KimiWeights:
    quant_kind: str
    w13: torch.Tensor
    w2: torch.Tensor
    w13_scale: torch.Tensor
    w2_scale: torch.Tensor
    fc1_alpha: torch.Tensor
    fc2_alpha: torch.Tensor
    fc1_norm_const: torch.Tensor


def interleave_gate_up(x):
    experts, rows, width = x.shape
    gate, up = x.chunk(2, dim=1)
    return (
        torch.stack(
            (
                gate.reshape(experts, rows // 32, 16, width),
                up.reshape(experts, rows // 32, 16, width),
            ),
            dim=2,
        )
        .reshape(experts, rows, width)
        .contiguous()
    )


def swizzle_scales(x):
    experts, rows, cols = x.shape
    x = torch.nn.functional.pad(x.view(torch.uint8), (0, -cols % 4, 0, -rows % 128))
    return (
        x.view(experts, x.shape[1] // 128, 4, 32, x.shape[2] // 4, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .reshape(experts, -1)
        .contiguous()
    )


def _tensor(tensor, shape, dtypes, device, name):
    if (
        not isinstance(tensor, torch.Tensor)
        or tuple(tensor.shape) != shape
        or tensor.dtype not in dtypes
        or tensor.device != device
        or not tensor.is_contiguous()
    ):
        raise ValueError(
            f"{name} must be contiguous {shape}, dtype {dtypes}, on {device}"
        )


def _scalar(value, count, device, name):
    if value is None:
        return torch.ones(count, dtype=torch.float32, device=device)
    _tensor(value, (count,), (torch.float32,), device, name)
    _positive(value, name)
    return value


def _positive(value, name):
    if not bool(torch.isfinite(value).all()) or not bool((value > 0).all()):
        raise ValueError(f"{name} must be positive and finite")


def _pack_fp4(value):
    mids = torch.tensor((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0), device=value.device)
    absolute = value.abs().contiguous()
    code = torch.bucketize(absolute, mids)
    tie = (code < 7) & (absolute == mids[code.clamp(max=6)])
    code = (code + (tie & ((code & 1) != 0))).to(torch.uint8)
    code |= torch.signbit(value).to(torch.uint8) << 3
    return code[..., ::2] | (code[..., 1::2] << 4)


def _quantize_expert(weight, kind, global_scale):
    block = 16 if kind == "nvfp4" else 32
    rows, width = weight.shape
    values = weight.float().reshape(rows, width // block, block)
    maximum = values.abs().amax(-1)
    if not bool(torch.isfinite(maximum).all()):
        raise ValueError("Unquantized Kimi weights must be finite")
    if kind == "nvfp4":
        scale = (maximum * (global_scale / 6.0)).clamp(max=448).to(torch.float8_e4m3fn)
        divisor = scale.float()
        inverse = torch.where(divisor > 0, global_scale / divisor, 0.0)
        normalized = values * inverse[..., None]
        scale = scale.view(torch.uint8)
    else:
        exponent = torch.ceil(torch.log2((maximum / 6.0).double())).clamp(-127, 127)
        scale = (exponent + 127).to(torch.uint8)
        divisor = torch.exp2(exponent).float()
        normalized = values / divisor[..., None]
    return _pack_fp4(normalized.reshape(rows, width)), scale


def preprocess_weights(weights, config, fleet_params, *, world_size):
    count = fleet_params.num_experts // world_size
    hidden, intermediate = fleet_params.token_hidden_size, config.intermediate_size
    device = weights.w13.device
    if device.type != "cuda" or device.index != torch.cuda.current_device():
        raise ValueError("Kimi weights must be on the current CUDA device")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Preprocess Kimi weights before CUDA graph capture")
    kind, block = config.quant_kind, 16 if config.quant_kind == "nvfp4" else 32
    alpha1 = _scalar(config.fc1_alpha, count, device, "fc1_alpha")
    alpha2 = _scalar(config.fc2_alpha, count, device, "fc2_alpha")
    norm = _scalar(config.fc1_norm_const, count, device, "fc1_norm_const")
    packed = isinstance(weights, PrequantizedMoEWeights)
    width_divisor = 2 if packed else 1
    dtypes = (torch.uint8,) if packed else (torch.bfloat16, torch.float32)
    _tensor(
        weights.w13,
        (count, 2 * intermediate, hidden // width_divisor),
        dtypes,
        device,
        "w13",
    )
    _tensor(
        weights.w2, (count, hidden, intermediate // width_divisor), dtypes, device, "w2"
    )
    if packed:
        sf_type = (
            torch.float8_e4m3fn
            if kind == "nvfp4"
            else getattr(torch, "float8_e8m0fnu", torch.uint8)
        )
        _tensor(
            weights.w13_scale,
            (count, 2 * intermediate, hidden // block),
            (torch.uint8, sf_type),
            device,
            "w13_scale",
        )
        _tensor(
            weights.w2_scale,
            (count, hidden, intermediate // block),
            (torch.uint8, sf_type),
            device,
            "w2_scale",
        )
        w13 = weights.w13
        w2 = weights.w2 if weights.w2.data_ptr() % 16 == 0 else weights.w2.clone()
        sf13, sf2 = (
            weights.w13_scale.view(torch.uint8),
            weights.w2_scale.view(torch.uint8),
        )
    else:
        w13 = torch.empty(
            (count, 2 * intermediate, hidden // 2), dtype=torch.uint8, device=device
        )
        w2 = torch.empty(
            (count, hidden, intermediate // 2), dtype=torch.uint8, device=device
        )
        sf13 = torch.empty(
            (count, 2 * intermediate, hidden // block), dtype=torch.uint8, device=device
        )
        sf2 = torch.empty(
            (count, hidden, intermediate // block), dtype=torch.uint8, device=device
        )
        gs1 = torch.ones(count, dtype=torch.float32, device=device)
        gs2 = torch.ones_like(gs1)
        if kind == "nvfp4":
            if config.fc1_alpha is not None:
                gs1 = 1 / (config.input_norm_const * alpha1)
            if config.fc2_alpha is not None:
                gs2 = 1 / (norm * alpha2)
            _positive(gs1, "FC1 weight global scale")
            _positive(gs2, "FC2 weight global scale")
        for expert in range(count):
            if kind == "nvfp4":
                if config.fc1_alpha is None:
                    maximum = weights.w13[expert].abs().amax().float().clamp_min(1e-12)
                    gs1[expert] = (448 * 6) / maximum
                if config.fc2_alpha is None:
                    maximum = weights.w2[expert].abs().amax().float().clamp_min(1e-12)
                    gs2[expert] = (448 * 6) / maximum
            w13[expert], sf13[expert] = _quantize_expert(
                weights.w13[expert], kind, gs1[expert]
            )
            w2[expert], sf2[expert] = _quantize_expert(
                weights.w2[expert], kind, gs2[expert]
            )
        if kind == "nvfp4":
            if config.fc1_alpha is None:
                alpha1 = 1 / (config.input_norm_const * gs1)
            if config.fc2_alpha is None:
                alpha2 = 1 / (norm * gs2)
            _positive(alpha1, "fc1_alpha")
            _positive(alpha2, "fc2_alpha")
    return KimiWeights(
        kind,
        interleave_gate_up(w13),
        w2,
        swizzle_scales(interleave_gate_up(sf13)),
        swizzle_scales(sf2),
        alpha1,
        alpha2,
        norm,
    )


def validate_transformed_weights(weights, config, fleet_params, *, world_size):
    if not isinstance(weights, KimiWeights) or weights.quant_kind != config.quant_kind:
        raise ValueError(
            "transformed_weights must be Kimi weights of the selected quant kind"
        )
    count = fleet_params.num_experts // world_size
    hidden, intermediate = fleet_params.token_hidden_size, config.intermediate_size
    block = 16 if config.quant_kind == "nvfp4" else 32
    device = torch.device("cuda", torch.cuda.current_device())
    for name, shape in (
        ("w13", (count, 2 * intermediate, hidden // 2)),
        ("w2", (count, hidden, intermediate // 2)),
        ("w13_scale", (count, 2 * intermediate * hidden // block)),
        ("w2_scale", (count, hidden * intermediate // block)),
    ):
        value = getattr(weights, name)
        _tensor(value, shape, (torch.uint8,), device, name)
        if value.data_ptr() % 16:
            raise ValueError(f"{name} must be 16-byte aligned")
    for name in ("fc1_alpha", "fc2_alpha", "fc1_norm_const"):
        value = getattr(weights, name)
        _tensor(value, (count,), (torch.float32,), device, name)
        if value.data_ptr() % 4:
            raise ValueError(f"{name} must be 4-byte aligned")
        _positive(value, name)
        if config.quant_kind == "mxfp4_mxfp8" and not bool((value == 1).all()):
            raise ValueError(f"MXFP4/MXFP8 requires {name}=1")
