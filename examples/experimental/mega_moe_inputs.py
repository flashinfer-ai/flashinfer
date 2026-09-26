"""Synthetic packed inputs and independent PyTorch references for SM103a MoE.

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


def smoke_inputs(family, precision, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    tokens, hidden = (17, 256) if family == "source" else (16, 512)
    experts, top_k, intermediate = 8, 2, 256
    x, xs = operand((tokens, hidden), "fp8", generator, exponent_low=-1)
    w1, s1 = operand((experts, 2 * intermediate, hidden), precision, generator)
    w2, s2 = operand((experts, hidden, intermediate), precision, generator)
    token = torch.arange(tokens, device="cuda")
    routes = torch.stack(
        ((token + seed) % experts, (token + seed + 3) % experts), dim=1
    ).long()
    weights = torch.stack(
        (torch.ones_like(token).float(), torch.where(token % 2 == 0, -0.5, 1.0)), dim=1
    )
    inputs = dict(
        num_tokens=tokens,
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
    shared = None
    if family == "source":
        a, sa = operand((1, 2 * intermediate, hidden), "fp8", generator)
        b, sb = operand((1, hidden, intermediate), "fp8", generator)
        shared = (a, sa, b, sb)
    return inputs, xs, shared


def interleave_rows(tensor):
    experts, rows = tensor.shape[:2]
    tail = tensor.shape[2:]
    gate = tensor[:, : rows // 2].reshape(experts, rows // 16, 8, *tail)
    up = tensor[:, rows // 2 :].reshape(experts, rows // 16, 8, *tail)
    return torch.stack((gate, up), dim=2).reshape(experts, rows, *tail).contiguous()


def source_weights(inputs, shared):
    def encode(values, sf, first_layer):
        if first_layer:
            values, sf = interleave_rows(values), interleave_rows(sf)
        experts, rows, _ = values.shape
        words = scale_words(sf)
        # The scale row permutation is distinct from gate/up interleaving.
        words = (
            words.reshape(experts, rows // 128, 4, 32, -1)
            .transpose(2, 3)
            .contiguous()
            .reshape(experts, rows, -1)
        )
        words = words.permute(0, 2, 1).contiguous().reshape(-1, rows).view(torch.uint32)
        return values.view(torch.uint8).reshape(experts * rows, -1), words

    precision = inputs["routed_weight_dtype"]
    b1, s1 = encode(inputs[f"w1_{precision}"], inputs["w1_sf"], True)
    b2, s2 = encode(inputs[f"w2_{precision}"], inputs["w2_sf"], False)
    sb1, ss1 = encode(shared[0], shared[1], True)
    sb2, ss2 = encode(shared[2], shared[3], False)
    return dict(B1=b1, SFB1=s1, B2=b2, SFB2=s2, SB1=sb1, SSFB1=ss1, SB2=sb2, SSFB2=ss2)


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


def reference(inputs, x_scales, shared=None):
    """BF16 L1, clamped SwiGLU, FP32 activation quantization, BF16 L2/sum."""
    precision = inputs["routed_weight_dtype"]
    x = unpack(inputs["x_fp8_packed"], x_scales, "fp8")
    w1 = unpack(inputs[f"w1_{precision}"], inputs["w1_sf"], precision)
    w2 = unpack(inputs[f"w2_{precision}"], inputs["w2_sf"], precision)
    width = inputs["intermediate"]

    output = torch.zeros_like(x)
    for slot in range(inputs["top_k"]):
        for index in range(inputs["num_experts"]):
            selected = inputs["topk_idx"][:, slot] == index
            output[selected] += _expert(
                x[selected],
                w1[index],
                w2[index],
                inputs["topk_weights"][selected, slot],
                width,
            )
    if shared is not None:
        output += _expert(
            x,
            unpack(shared[0], shared[1], "fp8")[0],
            unpack(shared[2], shared[3], "fp8")[0],
            torch.ones(x.shape[0], device=x.device),
            width,
        )
    return output.bfloat16()


def model_inputs(precision, num_tokens=1, seed=0):
    """Synthetic packed inputs at the catalogued model geometry: 384 experts,
    top-6, hidden 5120, intermediate 2304, clamp 10. Every token routes to six
    distinct experts. The packed weights take about 7 GiB of device memory."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    experts, top_k, hidden, intermediate = 384, 6, 5120, 2304
    x, xs = operand((num_tokens, hidden), "fp8", generator, exponent_low=-1)
    w1, s1 = operand((experts, 2 * intermediate, hidden), precision, generator)
    w2, s2 = operand((experts, hidden, intermediate), precision, generator)
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
    """``reference`` restricted to the routed experts; dequantizing all 384
    experts at once would need tens of GiB."""
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


def make_smoke(family, precision, seed=0):
    inputs, xs, shared = smoke_inputs(family, precision, seed)
    if family == "source":
        from flashinfer.source_mega_moe import prepare_mega_moe

        plan = prepare_mega_moe(
            inputs["x_fp8_packed"],
            inputs["x_sf_packed"],
            inputs["topk_idx"],
            inputs["topk_weights"],
            weights=source_weights(inputs, shared),
            num_experts=8,
            intermediate=256,
            routed_weight_dtype=precision,
            num_shared_experts=1,
            activation_clamp=10.0,
            fast_math=True,
            num_sms=2,
        )
    else:
        from flashinfer.mega_moe_v3 import prepare_pipeline

        plan = prepare_pipeline(inputs)
    return plan, inputs, xs, shared


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
