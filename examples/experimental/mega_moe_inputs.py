"""Synthetic packed inputs and independent PyTorch references for SM100a/SM103a MoE.

FP4 is low-nibble-first E2M1. All scales have granularity 32 and are powers of
2; every int32 scale word contains four consecutive UE8M0 exponent bytes.
These are small API examples, not a model-weight conversion interface.
"""

import torch


def scale_words(scales):
    return (
        ((scales.contiguous().view(torch.int32) >> 23) & 255)
        .to(torch.uint8)
        .contiguous()
        .view(torch.int32)
    )


def unpack(values, scales, precision):
    if precision == "fp4":
        table = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
            device=values.device,
        )
        byte = values.view(torch.uint8)
        decoded = torch.stack(
            (table[(byte & 15).long()], table[(byte >> 4).long()]), dim=-1
        ).flatten(-2)
    else:
        decoded = values.view(torch.float8_e4m3fn).float()
    return decoded * scales.repeat_interleave(32, dim=-1)


def operand(shape, precision, generator, exponent_low=-5):
    exponents = torch.randint(
        exponent_low,
        exponent_low + 3,
        (*shape[:-1], shape[-1] // 32),
        device="cuda",
        generator=generator,
    )
    scales = torch.pow(2.0, exponents.float())
    if precision == "fp4":
        codes = torch.randint(
            0, 16, shape, device="cuda", dtype=torch.uint8, generator=generator
        )
        values = codes[..., 0::2] | (codes[..., 1::2] << 4)
    else:
        values = (
            torch.randint(-16, 17, shape, device="cuda", generator=generator).float()
            / 2
        ).to(torch.float8_e4m3fn)
    return values, scales


def _expert(x_rows, first, second, route_weight, width):
    """BF16 L1, clamped SwiGLU, FP32 activation quantization, BF16 L2 for one expert."""
    projected = (x_rows @ first.T).bfloat16().float()
    gate = projected[:, :width].clamp(max=10.0)
    up = projected[:, width:].clamp(min=-10.0, max=10.0)
    intermediate = gate * torch.sigmoid(gate) * up * route_weight[:, None]
    groups = intermediate.reshape(-1, width // 32, 32)
    scales = torch.pow(
        2.0, torch.ceil(torch.log2(groups.abs().amax(-1).clamp_min(1e-4) / 448.0))
    )
    quantized = (groups / scales[..., None]).to(torch.float8_e4m3fn)
    dequantized = (quantized.float() * scales[..., None]).reshape(-1, width)
    return (dequantized @ second.T).bfloat16().float()


def model_inputs(precision, num_tokens=1, seed=0, l2_scale_shift=0):
    """Synthetic packed inputs at the catalogued model geometry: 384 experts,
    top-6, hidden 5120, intermediate 2304, clamp 10. Every token routes to six
    distinct experts. The packed weights take about 7 GiB of device memory.

    ``l2_scale_shift`` lowers the exponents of the routed down-projection
    weight scales by that many powers of two. The long token counts use 5 so
    that every bf16-rounded weighted expert output stays below 128 in
    magnitude (one rounding-order flip is then at most 0.5, inside the
    elementwise tolerance); the alternating +0.5 / -0.25 route weights, the L1
    projection and the activation clamp are unchanged.
    """
    generator = torch.Generator(device="cuda").manual_seed(seed)
    experts, top_k, hidden, intermediate = 384, 6, 5120, 2304
    x, xs = operand((num_tokens, hidden), "fp8", generator, exponent_low=-1)
    w1, s1 = operand((experts, 2 * intermediate, hidden), precision, generator)
    w2, s2 = operand(
        (experts, hidden, intermediate),
        precision,
        generator,
        exponent_low=-5 - l2_scale_shift,
    )
    token = torch.arange(num_tokens, device="cuda")
    routes = torch.stack(
        [(token * top_k + slot + seed) % experts for slot in range(top_k)], dim=1
    ).long()
    weights = torch.stack(
        [
            torch.full_like(token, 0.5 if slot % 2 == 0 else -0.25, dtype=torch.float32)
            for slot in range(top_k)
        ],
        dim=1,
    )
    inputs = dict(
        num_tokens=num_tokens,
        num_experts=experts,
        top_k=top_k,
        hidden=hidden,
        intermediate=intermediate,
        routed_weight_dtype=precision,
        activation_clamp=10.0,
        x_fp8_packed=x,
        x_sf_packed=scale_words(xs),
        topk_idx=routes,
        topk_weights=weights,
        w1_sf=s1,
        w2_sf=s2,
    )
    inputs[f"w1_{precision}"] = w1
    inputs[f"w2_{precision}"] = w2
    return inputs, xs


def model_reference(inputs, x_scales):
    """BF16 L1, clamped SwiGLU, FP32 activation quantization, BF16 L2 of the
    routed experts (only the selected ones are dequantized; all 384 at once
    would need tens of GiB), summed in FP32 and rounded to BF16 once, like the
    kernel."""
    precision = inputs["routed_weight_dtype"]
    x = unpack(inputs["x_fp8_packed"], x_scales, "fp8")
    width = inputs["intermediate"]
    output = torch.zeros_like(x)
    for index in inputs["topk_idx"].unique().tolist():
        first = unpack(
            inputs[f"w1_{precision}"][index], inputs["w1_sf"][index], precision
        )
        second = unpack(
            inputs[f"w2_{precision}"][index], inputs["w2_sf"][index], precision
        )
        for slot in range(inputs["top_k"]):
            selected = inputs["topk_idx"][:, slot] == index
            if selected.any():
                output[selected] += _expert(
                    x[selected],
                    first,
                    second,
                    inputs["topk_weights"][selected, slot],
                    width,
                )
    return output.bfloat16()


def make_model(precision, num_tokens=16, seed=0, l2_scale_shift=0):
    """Prepared pipeline plan on a catalogued model route (both routed
    precisions exist at 16 tokens; FP4 also at 1, 128, 512, 1024 and 4096
    tokens). Returns (plan, inputs, x_scales)."""
    from flashinfer.mega_moe_v3 import prepare_pipeline

    inputs, xs = model_inputs(
        precision, num_tokens=num_tokens, seed=seed, l2_scale_shift=l2_scale_shift
    )
    return prepare_pipeline(inputs), inputs, xs


def make_grouped_l2(seed=0):
    from flashinfer.mega_moe_v3 import prepare_grouped_l2

    generator = torch.Generator(device="cuda").manual_seed(seed)
    a, sa = operand((4096, 3072), "fp8", generator)
    b, sb = operand((4, 7168, 3072), "fp4", generator)
    words_a, words_b = scale_words(sa), scale_words(sb)
    plan = prepare_grouped_l2(a, b, words_a, words_b, [1024] * 4)
    return plan, a, b, sa, sb, words_a, words_b


def grouped_reference(a, b, sa, sb):
    output = torch.empty((4096, 7168), dtype=torch.bfloat16, device=a.device)
    for expert in range(4):
        rows = slice(expert * 1024, (expert + 1) * 1024)
        output[rows] = (
            unpack(a[rows], sa[rows], "fp8") @ unpack(b[expert], sb[expert], "fp4").T
        ).bfloat16()
    return output
