from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn.functional as F


FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
NVFP4_BLOCK_SIZE = 16

E2M1_TO_FLOAT32 = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def dequantize_nvfp4_mla_nope(
    records: torch.Tensor,
    *,
    nope_bytes: int = 256,
    group_scales_offset: int = 256,
    group_scales_end: int = 288,
    latent_scale_offset: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode the NoPE lane of compact NVFP4 MLA records.

    ``latent_scale_offset=None`` models the implicit/static outer-scale
    record. Passing an offset models a record carrying one inline fp32 outer
    scale per token.
    """
    packed = records[:, :nope_bytes]
    codes = torch.stack((packed & 0xF, (packed >> 4) & 0xF), dim=-1).reshape(
        records.shape[0], nope_bytes * 2
    )
    e2m1 = torch.tensor(E2M1_TO_FLOAT32, dtype=torch.float32, device=records.device)
    group_scales = (
        records[:, group_scales_offset:group_scales_end]
        .contiguous()
        .view(torch.float8_e4m3fn)
        .float()
    )
    values = (
        e2m1[codes.long()].reshape(records.shape[0], group_scales.shape[-1], -1)
        * group_scales.unsqueeze(-1)
    ).reshape(records.shape[0], nope_bytes * 2)
    if latent_scale_offset is not None:
        latent_scales = (
            records[:, latent_scale_offset : latent_scale_offset + 4]
            .contiguous()
            .view(torch.float32)
            .reshape(-1)
        )
        values = values * latent_scales.unsqueeze(-1)
    return values, group_scales


def prepare_tp_moe_fp4_experts(
    *,
    a: torch.Tensor,
    a1_gscale: torch.Tensor,
    w1_fp4: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w1_alphas: torch.Tensor,
    a2_gscale: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_alphas: torch.Tensor,
    activation: str = "silu",
    quant_mode: str = "nvfp4",
    source_format: str = "modelopt_nvfp4",
    w13_layout: str = "w13",
    swiglu_limit: float | None = None,
    swiglu_alpha: float | None = None,
    swiglu_beta: float | None = None,
):
    """Prepare canonical expert weights while retaining the source-scale recipe."""
    from b12x.moe import fused_moe

    normalized_mode = quant_mode.lower()
    activation_mode = {
        "nvfp4": fused_moe.ActivationMode.A4,
        "w4a16": fused_moe.ActivationMode.A16,
        "w4a8_nvfp4": fused_moe.ActivationMode.A8,
        "w4a8_mx": fused_moe.ActivationMode.A8,
    }.get(normalized_mode)
    if activation_mode is None:
        raise ValueError(f"unsupported TP MoE fixture recipe {quant_mode!r}")
    if normalized_mode == "w4a8_mx":
        source_format = "fp4_e8m0_k32"
    elif normalized_mode in {"nvfp4", "w4a16", "w4a8_nvfp4"}:
        source_format = "modelopt_nvfp4"

    weight_plan = fused_moe.plan_weights(
        source=fused_moe.PackedSource(
            format=fused_moe.PackedSourceFormat(source_format),
            w13_layout=fused_moe.W13Layout(w13_layout),
        ),
        activation=fused_moe.ActivationSpec(
            mode=activation_mode,
            nonlinearity=activation,
            io_dtype=a.dtype,
            swiglu_limit=swiglu_limit,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
        ),
        geometry=fused_moe.MoEGeometry(
            num_experts=int(w1_fp4.shape[0]),
            hidden_size=int(a.shape[1]),
            intermediate_size=int(w2_fp4.shape[2]) * 2,
        ),
    )
    if activation_mode is fused_moe.ActivationMode.A16:
        w1_global_scale, w2_global_scale = w1_alphas, w2_alphas
        input_scale = intermediate_scale = None
    else:
        w1_global_scale = (w1_alphas.float() * a1_gscale.float()).contiguous()
        w2_global_scale = (w2_alphas.float() * a2_gscale.float()).contiguous()
        input_scale, intermediate_scale = a1_gscale, a2_gscale
    return fused_moe.prepare_weights(
        plan=weight_plan,
        weights=fused_moe.PackedWeights(
            w13=w1_fp4,
            w2=w2_fp4,
            w13_block_scales=w1_blockscale,
            w2_block_scales=w2_blockscale,
            w13_global_scales=w1_global_scale,
            w2_global_scales=w2_global_scale,
            input_scale=input_scale,
            intermediate_scale=intermediate_scale,
        ),
    )


@contextmanager
def make_tp_moe_fp4_binding(
    *,
    a: torch.Tensor,
    experts: object,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    apply_router_weight_on_input: bool = False,
    output: torch.Tensor | None = None,
    input_scales_static: bool = False,
    fast_math: bool | None = None,
    quant_mode: str | None = None,
    swiglu_limit: float | None = None,
    swiglu_alpha: float | None = None,
    swiglu_beta: float | None = None,
    deterministic_output: bool | None = None,
) -> Iterator[object]:
    """Prepare one exact-M execution and yield its real caller-owned binding."""
    from b12x.moe import fused_moe
    from b12x.preparation import FrozenMapping, PreparationSession, PreparedCall
    from b12x.preparation.types import require_prepared

    if not isinstance(experts, fused_moe.PreparedExperts):
        raise TypeError("experts must be canonical fused_moe.PreparedExperts")
    activation = experts.plan.activation
    requested_mode = quant_mode.lower() if quant_mode is not None else None
    expected_mode = {
        fused_moe.ActivationMode.A4: "nvfp4",
        fused_moe.ActivationMode.A16: "w4a16",
        fused_moe.ActivationMode.A8: {
            fused_moe.PackedSourceFormat.MXFP4_E8M0_K32: "w4a8_mx",
            fused_moe.PackedSourceFormat.MXFP6_E8M0_K32: "w6a8_mx",
            fused_moe.PackedSourceFormat.MXFP8_E8M0_K32: "w8a8_mx",
        }.get(experts.plan.source.format, "w4a8_nvfp4"),
    }.get(activation.mode)
    if requested_mode is not None and requested_mode != expected_mode:
        raise ValueError(
            f"requested test recipe {requested_mode!r} does not match "
            f"canonical prepared recipe {expected_mode!r}"
        )
    for name, requested, prepared in (
        ("swiglu_limit", swiglu_limit, activation.swiglu_limit),
        ("swiglu_alpha", swiglu_alpha, activation.swiglu_alpha),
        ("swiglu_beta", swiglu_beta, activation.swiglu_beta),
    ):
        if requested is not None and requested != prepared:
            raise ValueError(f"{name} must be declared while preparing expert weights")

    declaration = fused_moe.plan_execution(
        experts=experts,
        capacity=fused_moe.ExecutionCapacity(
            max_tokens=int(a.shape[0]),
            top_k=int(topk_ids.shape[1]),
        ),
        routing=fused_moe.RoutingSpec(
            apply_router_weight_on_input=apply_router_weight_on_input,
            deterministic_output=deterministic_output,
        ),
        invocation=FrozenMapping(
            {
                "fast_math": True if fast_math is None else bool(fast_math),
            }
        ),
    )

    def bind_with_state(state, scratch):
        return state.bind(
            scratch=scratch,
            a=a,
            experts=experts,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            output=output,
            input_scales_static=input_scales_static,
            fast_math=fast_math,
        )

    def prepare_call(state):
        scratch = tuple(
            torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
            for spec in state.scratch.scratch_specs()
        )
        binding = bind_with_state(state, scratch)
        return PreparedCall(run=lambda: state.run(binding), owners=scratch)

    request = declaration.request(
        name="tp-moe-fixture",
        prepare_call=prepare_call,
    )
    session = PreparationSession(device=a.device, autotune=False, compile_workers=2)
    result = session.prepare((request,))
    scratch = tuple(
        torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        for spec in require_prepared(request.plan, "moe.decode").scratch.scratch_specs()
    )
    try:
        yield fused_moe.bind(
            request.plan,
            scratch=scratch,
            a=a,
            experts=experts,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            output=output,
            input_scales_static=input_scales_static,
            fast_math=fast_math,
        )
    finally:
        torch.cuda.synchronize(a.device)
        result.close()
        session.close()


def run_tp_moe_fp4(**kwargs) -> torch.Tensor:
    from b12x.moe import fused_moe

    with make_tp_moe_fp4_binding(**kwargs) as binding:
        return fused_moe.run(binding=binding)


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def cast_from_fp4(x: torch.Tensor) -> torch.Tensor:
    v_lo = x.to(torch.uint8) & 0xF
    v_hi = (x.to(torch.uint8) >> 4) & 0xF
    combined = torch.stack((v_lo, v_hi), dim=-1)
    new_shape = combined.shape[:-2] + (combined.shape[-2] * combined.shape[-1],)
    lookup = torch.tensor(E2M1_TO_FLOAT32, dtype=torch.float32, device=x.device)
    return lookup[combined.to(torch.long)].reshape(new_shape)


def cast_to_fp4(x: torch.Tensor) -> torch.Tensor:
    sign = torch.sign(x)
    x = torch.abs(x.clone())
    x[(x >= 0.0) & (x <= 0.25)] = 0.0
    x[(x > 0.25) & (x < 0.75)] = 0.5
    x[(x >= 0.75) & (x <= 1.25)] = 1.0
    x[(x > 1.25) & (x < 1.75)] = 1.5
    x[(x >= 1.75) & (x <= 2.5)] = 2.0
    x[(x > 2.5) & (x < 3.5)] = 3.0
    x[(x >= 3.5) & (x <= 5.0)] = 4.0
    x[x > 5.0] = 6.0
    return x * sign


def _reciprocal(x: torch.Tensor | float) -> torch.Tensor | float:
    if isinstance(x, torch.Tensor):
        return torch.where(x == 0, torch.zeros_like(x), 1.0 / x)
    if x == 0:
        return 0.0
    return 1.0 / x


def ref_fp4_quant(
    x: torch.Tensor,
    global_scale: torch.Tensor | float,
    block_size: int = NVFP4_BLOCK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    sliced_shape = x.shape[:-1] + (x.shape[-1] // block_size, block_size)
    sliced_x = x.reshape(sliced_shape)
    vec_max = torch.max(torch.abs(sliced_x), dim=-1, keepdim=True)[0].to(torch.float32)
    scale = global_scale * (vec_max * _reciprocal(FLOAT4_E2M1_MAX))
    scale = scale.to(torch.float8_e4m3fn).to(torch.float32)
    output_scale = _reciprocal(scale * _reciprocal(global_scale))
    scaled_x = sliced_x.to(torch.float32) * output_scale
    clipped_x = torch.clamp(scaled_x, -FLOAT4_E2M1_MAX, FLOAT4_E2M1_MAX).reshape(
        x.shape
    )
    return cast_to_fp4(clipped_x), scale.squeeze(-1)


def ref_grouped_fp4_quantize(
    input_tensor: torch.Tensor,
    row_counts: torch.Tensor,
    global_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_groups, rows, cols = input_tensor.shape
    quantized = torch.zeros(
        (num_groups, rows, cols), dtype=torch.float32, device=input_tensor.device
    )
    scales = torch.zeros(
        (num_groups, rows, cols // NVFP4_BLOCK_SIZE),
        dtype=torch.float32,
        device=input_tensor.device,
    )
    for group_idx in range(num_groups):
        valid_rows = int(row_counts[group_idx].item())
        if valid_rows == 0:
            continue
        quantized[group_idx, :valid_rows], scales[group_idx, :valid_rows] = (
            ref_fp4_quant(
                input_tensor[group_idx, :valid_rows].float(),
                global_scale[group_idx],
                NVFP4_BLOCK_SIZE,
            )
        )
    return quantized, scales


def ref_grouped_silu_mul_quantize(
    input_tensor: torch.Tensor,
    row_counts: torch.Tensor,
    global_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cols = input_tensor.shape[-1] // 2
    left = input_tensor[..., :cols].float()
    right = input_tensor[..., cols:].float()
    activated = (F.silu(left) * right).to(input_tensor.dtype).to(torch.float32)
    return ref_grouped_fp4_quantize(activated, row_counts, global_scale)


def swizzle_block_scale_reference(scale: torch.Tensor) -> torch.Tensor:
    if scale.ndim == 2:
        scale = scale.unsqueeze(0)
        squeeze_batch = True
    else:
        squeeze_batch = False
    batch, rows, cols = scale.shape
    rows_padded = _align_up(rows, 128)
    cols_padded = _align_up(cols, 4)
    padded = torch.zeros(
        (batch, rows_padded, cols_padded), dtype=scale.dtype, device=scale.device
    )
    padded[:, :rows, :cols] = scale
    swizzled = padded.reshape(batch, rows_padded // 128, 4, 32, cols_padded // 4, 4)
    swizzled = swizzled.permute(0, 1, 4, 3, 2, 5).contiguous()
    swizzled = swizzled.reshape(batch, rows_padded, cols_padded)
    return swizzled[0] if squeeze_batch else swizzled


def recover_grouped_e4m3_scales(
    scale_view: torch.Tensor,
    rows: int,
    cols: int,
) -> torch.Tensor:
    num_groups = scale_view.shape[-1]
    rows_padded = _align_up(rows, 128)
    cols_padded = _align_up(cols // NVFP4_BLOCK_SIZE, 4)
    swizzled = scale_view.permute(5, 2, 4, 0, 1, 3).contiguous()
    swizzled = swizzled.reshape(num_groups, rows_padded, cols_padded)
    unswizzled = swizzled.view(
        num_groups,
        rows_padded // 128,
        cols_padded // 4,
        32,
        4,
        4,
    )
    unswizzled = unswizzled.permute(0, 1, 4, 3, 2, 5).contiguous()
    unswizzled = unswizzled.reshape(num_groups, rows_padded, cols_padded)
    return unswizzled[:, :rows, : cols // NVFP4_BLOCK_SIZE].to(torch.float32)


def dequantize_grouped_nvfp4(
    packed: torch.Tensor,
    scale_view: torch.Tensor,
    cols: int,
    global_scale: torch.Tensor,
) -> torch.Tensor:
    if global_scale.numel() == 1:
        global_scale = global_scale.expand(packed.shape[0]).contiguous()
    packed_fp32 = cast_from_fp4(packed.view(torch.uint8)).view(
        packed.shape[0], packed.shape[1], cols
    )
    scales = recover_grouped_e4m3_scales(scale_view, packed.shape[1], cols)
    values = packed_fp32.view(
        packed.shape[0], packed.shape[1], cols // NVFP4_BLOCK_SIZE, NVFP4_BLOCK_SIZE
    )
    return (values * scales.unsqueeze(-1) / global_scale.view(-1, 1, 1, 1)).reshape(
        packed.shape[0], packed.shape[1], cols
    )


def dequantize_token_major_nvfp4(
    x_fp4: torch.Tensor,
    x_sf: torch.Tensor,
    *,
    hidden_size: int,
    global_scale: torch.Tensor,
) -> torch.Tensor:
    x_fp4_float = cast_from_fp4(x_fp4.view(torch.uint8))
    num_tokens = x_fp4_float.shape[0]
    x_fp4_float = x_fp4_float.view(
        num_tokens, hidden_size // NVFP4_BLOCK_SIZE, NVFP4_BLOCK_SIZE
    )
    scales = x_sf.float().view(num_tokens, hidden_size // NVFP4_BLOCK_SIZE, 1)
    return (x_fp4_float * scales).view(num_tokens, hidden_size) / global_scale.item()


def compute_global_scale(x: torch.Tensor) -> torch.Tensor:
    amax = x.abs().max().to(torch.float32)
    value = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / amax
    return torch.tensor([value], dtype=torch.float32, device=x.device)


def compute_per_group_global_scale(x: torch.Tensor) -> torch.Tensor:
    amax = x.abs().amax(dim=(1, 2)).to(torch.float32)
    numerator = torch.full_like(amax, FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX)
    return torch.where(amax > 0, numerator / amax, torch.ones_like(amax))


def llama_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    x_fp32 = x.float()
    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    return (x_fp32 * torch.rsqrt(variance + eps) * weight.float()).to(x.dtype)


def require_b12x(*, allow_module_level=False) -> torch.device:
    """Skip unless a real SM120/SM121 device is present (migrated-test gate)."""
    import pytest

    if not torch.cuda.is_available():
        pytest.skip(
            "CUDA is required for b12x tests", allow_module_level=allow_module_level
        )
    major, minor = torch.cuda.get_device_capability()
    if major != 12 or minor not in (0, 1):
        pytest.skip(
            f"SM12x (SM120/SM121) GPU required, found sm_{major}{minor}",
            allow_module_level=allow_module_level,
        )
    return torch.device("cuda")
