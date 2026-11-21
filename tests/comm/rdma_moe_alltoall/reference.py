# Copyright (c) 2025 DeepSeek.
# Adapted reference helpers are MIT licensed; see LICENSE.deepep.txt.

"""Small mathematical and route-aware oracles adapted from the DeepEP port.

Transport compares against the standalone device quantizer because BF16/FP32
arithmetic can differ at quantization thresholds. The pure-Torch quantizers
remain separate references, and dispatch verifies exact payload/scale pairs.
Original DeepEP code is MIT licensed; see LICENSE.deepep.txt.
"""

import os
from contextlib import contextmanager

import torch
import torch.distributed as dist

DISPATCH_GLOBAL_SCALE = 256.0
EXPERTS = 288
WORLD = int(os.environ.get("WORLD_SIZE", "8"))


@contextmanager
def managed_buffer(buffer):
    try:
        yield buffer
    finally:
        torch.cuda.synchronize()
        buffer.close()


@contextmanager
def captured_graph(run, device, *, capture_calls=1):
    """Warm up twice and retire the graph before releasing its live buffers.

    Dispatch-only captures use two calls to preserve the arena slot phase;
    dispatch/combine callbacks already contain an even number of collectives.
    Replay mutations and numerical checks run on the capture stream.
    """
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            run()
    torch.cuda.synchronize(device)
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(capture_calls):
                result = run()
        with torch.cuda.stream(stream):
            yield graph, result, stream
    finally:
        torch.cuda.synchronize(device)
        graph.reset()


def set_source_lengths(case, lengths):
    """Mask unsent global rows and bind this rank's views to its actual length."""
    for source, length in enumerate(lengths):
        case["routes"][source, length:] = -1
    rank, tokens = case["rank"], lengths[case["rank"]]
    case["tokens"] = tokens
    for local, global_name in (
        ("input", "packed"),
        ("input_scales", "scales"),
        ("topk", "routes"),
        ("topk_weights", "weights"),
    ):
        case[local] = case[global_name][rank, :tokens]


def assert_all(ok, message, device):
    flag = torch.tensor(int(ok), dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    assert flag.item(), message if not ok else "Another rank failed: " + message


def global_scales(x):
    amax = x.float().abs().amax(dim=-1, keepdim=True)
    return torch.where(amax > 0, 2688.0 / amax.clamp_min(1e-20), 1.0)


def quantize_nvfp4_reference(x, global_scale):
    blocks = x.float().reshape(*x.shape[:-1], x.shape[-1] // 16, 16)
    scale = (blocks.abs().amax(dim=-1) * (global_scale / 6.0)).to(torch.float8_e4m3fn)
    effective = scale.float() / global_scale
    values = torch.where(effective[..., None] > 0, blocks / effective[..., None], 0.0)
    thresholds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device)
    magnitude = values.abs().clamp_max(6.0).contiguous()
    codes = torch.bucketize(magnitude, thresholds)
    ties = (codes < 7) & (magnitude == thresholds[codes.clamp_max(6)])
    codes += (ties & ((codes & 1) != 0)).to(codes.dtype)
    codes = (
        (codes | (torch.signbit(values).to(codes.dtype) << 3))
        .to(torch.uint8)
        .reshape(x.shape)
    )
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous(), scale.view(
        torch.uint8
    ).contiguous()


def quantize_fp8_reference(x):
    blocks = x.float().reshape(*x.shape[:-1], x.shape[-1] // 128, 128)
    amax = blocks.abs().amax(dim=-1).clamp_min(1e-4)
    scales = amax * (1.0 / 448.0)
    packed = (blocks * (448.0 / amax)[..., None]).to(torch.float8_e4m3fn)
    return packed.reshape(x.shape).view(torch.uint8).contiguous(), scales.contiguous()


def decode(
    packed, scales, dtype, global_scale=DISPATCH_GLOBAL_SCALE, kernel_arithmetic=False
):
    if dtype == "bf16":
        return packed.float()
    if dtype == "fp8":
        return packed.view(torch.float8_e4m3fn).float() * scales.repeat_interleave(
            128, dim=-1
        )
    codebook = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=packed.device
    )
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    values = codebook[(codes & 7).long()] * torch.where((codes & 8) != 0, -1.0, 1.0)
    global_scale = torch.as_tensor(
        global_scale, dtype=torch.float32, device=packed.device
    )
    effective = scales.view(torch.float8_e4m3fn).float()
    effective = (
        effective * global_scale.reciprocal()
        if kernel_arithmetic
        else effective / global_scale
    )
    return values * effective.repeat_interleave(16, dim=-1)


def expert_output(x, expert_id):
    expert = torch.as_tensor(expert_id, dtype=torch.float32, device=x.device)
    gain = 0.75 + (expert % 13) / 32.0
    bias = ((expert % 17) - 8.0) / 64.0
    return (x.float() * gain[..., None] + bias[..., None]).to(torch.bfloat16)


def make_case(
    tokens, hidden, topk, rank, device, dtype, pattern="uniform", seed=20260917
):
    generator = torch.Generator(device=device).manual_seed(seed)
    source = torch.randn(
        (WORLD, tokens, hidden),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    scores = torch.rand((WORLD, tokens, EXPERTS), device=device, generator=generator)
    routes = scores.topk(topk, dim=-1).indices.to(torch.int32)
    if pattern == "skew":
        routes.copy_(
            torch.arange(topk, dtype=torch.int32, device=device).expand_as(routes)
        )
    elif pattern == "empty_rank":
        routes[0].fill_(-1)
    elif pattern == "invalid":
        routes.fill_(-1)
    weights = torch.rand(
        (WORLD, tokens, topk), device=device, generator=generator
    ).softmax(dim=-1)
    if dtype == "bf16":
        packed, scales = (
            source,
            torch.empty((WORLD, tokens, 0), dtype=torch.uint8, device=device),
        )
    elif dtype == "fp8":
        packed, scales = quantize_fp8_reference(source)
    else:
        packed, scales = quantize_nvfp4_reference(source, DISPATCH_GLOBAL_SCALE)
    local_tokens = 0 if pattern == "empty_rank" and rank == 0 else tokens
    return {
        "packed": packed,
        "scales": scales,
        "routes": routes,
        "weights": weights,
        "rank": rank,
        "dtype": dtype,
        "tokens": local_tokens,
        "input": packed[rank, :local_tokens],
        "input_scales": scales[rank, :local_tokens],
        "topk": routes[rank, :local_tokens],
        "topk_weights": weights[rank, :local_tokens],
    }


def mutate_payload(case):
    if case["dtype"] == "bf16":
        case["packed"].neg_()
    else:
        case["packed"].bitwise_xor_(0x80 if case["dtype"] == "fp8" else 0x88)


def dispatch(buffer, case, hooks=False, out=None):
    result = buffer.dispatch(
        case["input"],
        case["topk"],
        return_recv_hook=hooks,
        out=out,
        input_dtype="prequantized",
        x_scales=None if case["dtype"] == "bf16" else case["input_scales"],
    )
    if hooks:
        result[-1]()
    return result


def combine(buffer, case, result, x, scales, hooks=False, out=None):
    combined = buffer.combine(
        x,
        result[3],
        return_recv_hook=hooks,
        out=out,
        topk_idx=case["topk"],
        topk_weights=case["topk_weights"],
        global_scale=scales,
    )
    if hooks:
        combined[-1]()
    return combined[0]


def allocate_dispatch_out(case, buffer):
    rows, experts = buffer.num_max_dispatch_tokens_per_rank * WORLD, EXPERTS // WORLD
    device = case["packed"].device
    output = (
        torch.empty(
            (experts, rows, case["packed"].shape[-1]),
            dtype=case["packed"].dtype,
            device=device,
        ),
        torch.empty(
            (experts, rows, case["scales"].shape[-1]),
            dtype=case["scales"].dtype,
            device=device,
        ),
        torch.empty(experts, dtype=torch.int32, device=device),
        torch.empty((experts, rows), dtype=torch.int32, device=device),
        torch.empty((experts, WORLD), dtype=torch.int64, device=device),
    )
    return output


def check_dispatch(result, case):
    received, scales, counts, handle, _ = result
    source_ids, ranges = handle[:2]
    counts_cpu, ranges_cpu = counts.cpu(), ranges.cpu()
    errors = []
    for expert in range(EXPERTS // WORLD):
        global_expert = case["rank"] * (EXPERTS // WORLD) + expert
        expected_count = int((case["routes"] == global_expert).sum().item())
        if int(counts_cpu[expert]) != expected_count:
            errors.append(f"expert {global_expert}: count mismatch")
        occupied = []
        for source in range(WORLD):
            encoded = int(ranges_cpu[expert, source])
            count, offset = encoded & 0xFFFFFFFF, encoded >> 32
            wanted = torch.where((case["routes"][source] == global_expert).any(dim=-1))[
                0
            ]
            if (
                count != wanted.numel()
                or offset < 0
                or offset + count > received.shape[1]
            ):
                errors.append(f"expert {global_expert}, source {source}: invalid range")
                continue
            if count:
                ids = source_ids[expert, offset : offset + count].long()
                if not torch.equal(ids.sort().values, wanted):
                    errors.append(
                        f"expert {global_expert}, source {source}: source IDs mismatch"
                    )
                    continue
                if not torch.equal(
                    received[expert, offset : offset + count],
                    case["packed"][source, ids],
                ):
                    errors.append(
                        f"expert {global_expert}, source {source}: payload mismatch"
                    )
                if not torch.equal(
                    scales[expert, offset : offset + count], case["scales"][source, ids]
                ):
                    errors.append(
                        f"expert {global_expert}, source {source}: paired scales mismatch"
                    )
            occupied.extend(range(offset, offset + count))
        if sorted(occupied) != list(range(expected_count)):
            errors.append(f"expert {global_expert}: ranges do not partition valid rows")
    assert_all(
        not errors, "; ".join(errors[:4]) or "dispatch correctness", received.device
    )
    return counts_cpu


def prepare_expert_outputs(result, case):
    packed, blocks = result[:2]
    counts = check_dispatch(result, case)
    x = torch.empty(
        (
            *packed.shape[:2],
            case["packed"].shape[-1] * (2 if case["dtype"] == "nvfp4" else 1),
        ),
        dtype=torch.bfloat16,
        device=packed.device,
    )
    scales = (
        torch.empty((*packed.shape[:2], 1), dtype=torch.float32, device=packed.device)
        if case["dtype"] == "nvfp4"
        else None
    )
    for expert, count in enumerate(counts.tolist()):
        decoded = decode(packed[expert, :count], blocks[expert, :count], case["dtype"])
        x[expert, :count] = expert_output(
            decoded, case["rank"] * (EXPERTS // WORLD) + expert
        )
        if scales is not None:
            scales[expert, :count] = global_scales(x[expert, :count])
    return x, scales


def expected_combine(buffer, case, profiles=None):
    rank, tokens, dtype = case["rank"], case["tokens"], case["dtype"]
    decoded = decode(
        case["packed"][rank, :tokens], case["scales"][rank, :tokens], dtype
    )
    expected = torch.zeros_like(decoded)
    magnitude = torch.zeros_like(decoded)
    if not tokens:
        return expected, magnitude
    for selection in range(case["topk"].shape[-1]):
        ids = case["topk"][:, selection]
        x = (
            expert_output(decoded, ids.clamp_min(0))
            if profiles is None
            else profiles[ids.clamp_min(0).long()].contiguous()
        )
        if dtype == "bf16":
            restored = x.float()
        elif dtype == "fp8":
            payload, blocks = buffer.quantize_bf16_to_fp8(x)
            restored = decode(payload, blocks, dtype)
        else:
            scales = global_scales(x)
            payload, blocks = buffer.quantize_bf16_to_nvfp4(x, scales)
            restored = decode(payload, blocks, dtype, scales, kernel_arithmetic=True)
        weight = case["topk_weights"][:, selection] * (ids >= 0)
        expected = torch.addcmul(expected, restored, weight[:, None])
        magnitude += (restored * weight[:, None]).abs()
    return expected, magnitude


def check_combine(actual, buffer, case, profiles=None):
    expected, magnitude = expected_combine(buffer, case, profiles)
    rounded = expected.to(torch.bfloat16)
    rounding_error = (rounded.float() - expected).abs()
    envelope = 32 * torch.finfo(torch.float32).eps * magnitude + 1e-7
    bad = (actual.float() - expected).abs() > rounding_error + envelope
    ok = (
        actual.shape == expected.shape
        and bool(torch.isfinite(actual).all().item())
        and not bool(bad.any().item())
    )
    assert_all(
        ok,
        f"combine outside BF16/FP32 rounding envelope: {int(bad.sum().item())} elements",
        actual.device,
    )


def _same_peer_routes(case, phase):
    """Force followers, then move the first valid route past a negative slot."""
    local_experts = EXPERTS // WORLD
    topk = case["routes"].shape[-1]
    for source in range(WORLD):
        peer = (source + phase + 1) % WORLD
        experts = (
            torch.arange(topk, dtype=torch.int32, device=case["routes"].device)
            + peer * local_experts
        )
        case["routes"][source].copy_(experts.expand_as(case["routes"][source]))
    if phase:
        case["routes"][..., 0] = -1
