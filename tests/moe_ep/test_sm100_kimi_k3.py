"""K3 public API checks with a sparse, independent FP64 numerical oracle.

Run on one GPU or with torchrun on 8/16 GPUs. Sparse weights test expert,
gate/up and scale layouts; they do not cover dense MMA accumulation error.
"""

from dataclasses import replace
import os

import pytest
import torch
import torch.distributed as dist

from flashinfer import moe_ep

pytestmark = [pytest.mark.arch_blackwell, pytest.mark.node_exclusive]

EXPERTS, HIDDEN, INTERMEDIATE, TOPK = 896, 3584, 3072, 16
FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _runtime():
    if not torch.cuda.is_available():
        pytest.skip("requires SM100/SM103")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("requires SM100/SM103")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world not in (1, 8, 16):
        pytest.skip("requires EP1, EP8 or EP16")
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("cpu:gloo,cuda:nccl")
    return dist.get_rank() if dist.is_initialized() else 0, world


def _sparse_weight(rows, columns, experts, first_expert, kind):
    device = torch.device("cuda")
    e = torch.arange(experts, device=device)
    global_e = e + first_expert
    r = torch.arange(rows, device=device)
    col = (13 * r[None, :] + 7 * global_e[:, None]) % columns
    code = (r[None, :] + global_e[:, None]) % 7 + 1
    code |= (((r[None, :] + global_e[:, None]) % 3) == 0).long() << 3
    packed = torch.zeros(experts, rows, columns // 2, dtype=torch.uint8, device=device)
    packed[e[:, None], r[None, :], col // 2] = (code << (4 * (col % 2))).byte()
    block = 16 if kind == "nvfp4" else 32
    c = torch.arange(columns // block, device=device)
    exponents = (
        (global_e % 4).byte()[:, None, None]
        + (r % 4).byte()[None, :, None]
        + (c % 4).byte()[None, None, :]
    ) % 4
    scales = (exponents + 5) * 8 if kind == "nvfp4" else exponents + 125
    return packed, scales


def _weights(rank, world, kind):
    count = EXPERTS // world
    w13, s13 = _sparse_weight(2 * INTERMEDIATE, HIDDEN, count, rank * count, kind)
    w2, s2 = _sparse_weight(HIDDEN, INTERMEDIATE, count, rank * count, kind)
    return moe_ep.MoEWeightPack(w13, w2, s13, s2)


def _alpha(expert):
    norm = 2.0 ** (expert % 4)
    return (expert % 3 + 1) * 0.5, (expert % 5 + 1) * 0.25 / norm, norm


def _round_fp4(values):
    midpoints = torch.tensor(
        (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0), device=values.device
    )
    index = torch.bucketize(values.abs().contiguous(), midpoints)
    tie = (index < 7) & (values.abs() == midpoints[index.clamp(max=6)])
    index += (tie & (index % 2 == 1)).long()
    grid = torch.tensor(FP4_VALUES, dtype=torch.float64, device=values.device)
    return grid[index.clamp(max=7)] * values.sign()


def _round_blocks(values, kind, norm=1.0):
    """Return the dequantized payload, before the NVFP4 global alpha."""
    shape = values.shape
    block = 16 if kind == "nvfp4" else 32
    blocks = values.float().reshape(shape[0], -1, block)
    maximum = blocks.abs().amax(-1, keepdim=True)
    if kind == "nvfp4":
        norm = torch.as_tensor(norm, dtype=torch.float32, device=values.device)
        if norm.ndim:
            norm = norm[:, None, None]
        scale = ((maximum * (1.0 / 6.0)) * norm).clamp(max=448)
        scale = scale.to(torch.float8_e4m3fn).float()
        inverse = torch.where(scale > 0, norm / scale, 0.0)
        quantized = _round_fp4((blocks * inverse).double())
    else:
        candidate = maximum * (1.0 / 448.0)
        exponent = torch.ceil(torch.log2(candidate.double())).clamp(-127, 127)
        scale = 2.0**exponent
        quantized = (blocks.double() / scale).float().to(torch.float8_e4m3fn).double()
    return (quantized * scale).reshape(shape)


def _row_weight(row, expert, columns, kind):
    col = (13 * row[None, :] + 7 * expert[:, None]) % columns
    number = row[None, :] + expert[:, None]
    grid = torch.tensor(FP4_VALUES, dtype=torch.float64, device=row.device)
    value = grid[number % 7 + 1] * torch.where(number % 3 == 0, -1.0, 1.0)
    exponent = (number + col // (16 if kind == "nvfp4" else 32)) % 4 - 2
    return col, value * 2.0**exponent


def _reference(tensors, kind, wire):
    tokens = tensors.num_tokens
    if tokens == 0:
        return torch.empty(0, HIDDEN, dtype=torch.bfloat16, device="cuda")
    x = _round_blocks(tensors.hidden_states, kind)
    result = torch.zeros(tokens, HIDDEN, dtype=torch.float64, device=x.device)
    rows = torch.arange(2 * INTERMEDIATE, device=x.device)
    output_rows = torch.arange(HIDDEN, device=x.device)
    for slot in range(TOPK):
        ids = tensors.topk_ids[:, slot].long()
        valid = (ids >= 0) & (ids < EXPERTS)
        expert = ids.masked_fill(~valid, 0)
        columns, weight = _row_weight(rows, expert, HIDDEN, kind)
        fc1 = x.gather(1, columns) * weight
        alpha1, alpha2, norm = _alpha(expert)
        if kind == "nvfp4":
            fc1 *= alpha1[:, None]
        gate, up = fc1.chunk(2, dim=1)
        activation = 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)
        activation *= 25.0 * torch.tanh(up / 25.0)
        activation = _round_blocks(activation, kind, norm if kind == "nvfp4" else 1.0)
        columns, weight = _row_weight(output_rows, expert, INTERMEDIATE, kind)
        partial = activation.gather(1, columns) * weight
        if kind == "nvfp4":
            partial *= alpha2[:, None]
        partial = partial.to(torch.bfloat16).double()
        if wire == "mxfp8":
            partial = _round_blocks(partial, "mxfp4_mxfp8")
        score = torch.where(valid, tensors.topk_weights[:, slot], 0.0)
        result += partial * score.double()[:, None]
    return result.to(torch.bfloat16)


def _inputs(tokens, rank, world, step, *, masked=False, mask_routes=True):
    row = torch.arange(tokens, device="cuda")[:, None]
    col = torch.arange(HIDDEN, device="cuda")[None, :]
    x = (((row * 5 + col * 17 + rank * 3 + step * 13) % 31 - 15) / 8).bfloat16()
    slot = torch.arange(TOPK, device="cuda")[None, :]
    local_experts = EXPERTS // world
    ids = (
        (slot % world) * local_experts
        + (row + slot // world + step * 11) % local_experts
    ).int()
    scores = ((slot + row + step) % 7 + 1).float() / 64
    invalid = (
        torch.ones_like(ids, dtype=torch.bool)
        if masked
        else (slot + row + step) % 11 == 0
    )
    if mask_routes:
        ids.masked_fill_(invalid, -1)
        scores.masked_fill_(invalid, float("nan"))
    return moe_ep.MoEEpTensors(x, ids, scores)


def _check(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()
    if actual.numel() == 0:
        return
    ref = expected.double()
    error = (actual.double() - ref).abs()
    ulp = 2.0 ** (torch.floor(torch.log2(ref.abs().clamp(min=2.0**-126))) - 7)
    tolerance = 2 * ulp + 0.001 * ref.square().mean(-1, keepdim=True).sqrt()
    assert (error <= tolerance).double().mean().item() >= 0.99
    assert (error.norm() / ref.norm().clamp(min=1e-20)).item() <= 0.01


@pytest.mark.parametrize("kind", ["nvfp4", "mxfp4_mxfp8"])
@pytest.mark.parametrize("wire", ["bf16", "mxfp8"])
def test_kimi_k3_public_eager_graph_and_masks(kind, wire):
    rank, world = _runtime()
    from flashinfer.moe_ep.kernel_src.sm100 import _kimi_k3_megamoe as native

    calibration = {}
    if kind == "nvfp4":
        expert = torch.arange(EXPERTS // world, device="cuda") + rank * (
            EXPERTS // world
        )
        alpha1, alpha2, norm = _alpha(expert)
        calibration = dict(
            fc1_alpha=alpha1.float(),
            fc2_alpha=alpha2.float(),
            fc1_norm_const=norm.float(),
        )
    bootstrap = moe_ep.BootstrapConfig(
        world_size=world,
        rank=rank,
        auto_bootstrap=False,
        process_group=dist.group.WORLD if world > 1 else None,
        device=torch.cuda.current_device(),
    )
    fleet = moe_ep.FleetParams(
        num_experts=EXPERTS, max_tokens_per_rank=256, token_hidden_size=HIDDEN
    )
    config = moe_ep.Sm100_KimiK3_MegaMoeConfig(
        quant_kind=kind,
        combine_dtype=wire,
        situ_beta=4.0,
        situ_linear_beta=25.0,
        **calibration,
    )
    transformed = native.preprocess_weights(
        _weights(rank, world, kind), config, fleet, world_size=world
    )

    def make_layer(*, quantize_input=True, fast_situ=False, knobs=None):
        return moe_ep.MoEEpMegaLayer(
            bootstrap=bootstrap,
            fleet_params=fleet,
            weights=moe_ep.MoEWeightPack(torch.empty(0), torch.empty(0)),
            backend=moe_ep.MegaConfig(
                megakernel=replace(config, fast_situ=fast_situ, knobs=knobs),
                quantize_input=quantize_input,
                preprocess_weights=False,
                transformed_weights=transformed,
            ),
        )

    id_dtype = torch.int64 if kind == "nvfp4" else torch.int32

    def make_inputs(tokens, step, *, masked=False):
        tensors = _inputs(tokens, rank, world, step, masked=masked)
        tensors.topk_ids = tensors.topk_ids.to(id_dtype)
        return tensors

    layer = make_layer()
    graph = None
    try:
        workspace = layer.create_workspace(256)
        for step, (tokens, masked) in enumerate(
            ((17, False), (17, True), (0 if rank == 0 else 7, False), (33, False))
        ):
            tensors = make_inputs(tokens, step, masked=masked)
            if step == 3:
                invalid = (2**32 + 1, EXPERTS, -2) if kind == "nvfp4" else (EXPERTS, -2)
                tensors.topk_ids[:, : len(invalid)] = torch.tensor(
                    invalid, dtype=id_dtype, device="cuda"
                )
                tensors.topk_weights[:, : len(invalid)] = float("nan")
            output = torch.empty_like(tensors.hidden_states)
            result = layer(replace(tensors, output=output), workspace=workspace)
            assert result.data_ptr() == output.data_ptr()
            expected = _reference(tensors, kind, wire)
            _check(result, expected)
            layer.stage_inputs(tensors, workspace=workspace)
            if step == 0:
                for invalid_output in (
                    torch.empty(tokens, 8, dtype=torch.bfloat16, device="cuda"),
                    torch.empty(
                        tokens - 1, HIDDEN, dtype=torch.bfloat16, device="cuda"
                    ),
                    torch.empty((), dtype=torch.bfloat16, device="cuda"),
                    torch.empty(tokens, HIDDEN, dtype=torch.float32, device="cuda"),
                    torch.empty(tokens, HIDDEN, dtype=torch.bfloat16),
                    torch.empty(
                        tokens, 2 * HIDDEN, dtype=torch.bfloat16, device="cuda"
                    )[:, ::2],
                    torch.empty(
                        tokens * HIDDEN + 1, dtype=torch.bfloat16, device="cuda"
                    )[1:].reshape(tokens, HIDDEN),
                ):
                    with pytest.raises(ValueError, match="Kimi output"):
                        layer.compute_staged(output=invalid_output)
                    with pytest.raises(ValueError):
                        layer(
                            replace(tensors, output=invalid_output), workspace=workspace
                        )
            staged_output = torch.full(
                (tokens + 2, HIDDEN), -123.0, dtype=torch.bfloat16, device="cuda"
            )
            result = layer.compute_staged(output=staged_output)
            if tokens:
                assert result.data_ptr() == staged_output.data_ptr()
            _check(result, expected)
            assert (staged_output[tokens:] == -123).all()
            if step == 0:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = layer.compute_staged(output=staged_output)
                graph.replay()
                torch.cuda.synchronize()
                _check(captured, expected)
                assert (staged_output[tokens:] == -123).all()
                graph = None
            if step == 0:
                for name in ("hidden_states", "topk_ids", "topk_weights", "output"):
                    original = output if name == "output" else getattr(tensors, name)
                    storage = torch.empty(
                        original.numel() + 1, dtype=original.dtype, device="cuda"
                    )
                    unaligned = storage[1:].reshape(original.shape).copy_(original)
                    with pytest.raises(ValueError, match="align"):
                        layer(
                            replace(tensors, **{name: unaligned}), workspace=workspace
                        )
                x = tensors.hidden_states
                storage = torch.empty(x.numel() + 8, dtype=x.dtype, device="cuda")
                unaligned = storage[8:].reshape(x.shape).copy_(x)
                with pytest.raises(ValueError, match="32-byte aligned"):
                    layer(
                        replace(tensors, hidden_states=unaligned), workspace=workspace
                    )

        tensors = make_inputs(17, 4)
        layer.warmup(tensors, workspace=workspace)
        eager = layer(tensors, workspace=workspace)
        output = torch.empty_like(eager)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = layer(replace(tensors, output=output), workspace=workspace)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, eager, rtol=0, atol=0)
        for step in (5, 6):
            changed = make_inputs(17, step, masked=step == 6)
            tensors.hidden_states.copy_(changed.hidden_states)
            tensors.topk_ids.copy_(changed.topk_ids)
            tensors.topk_weights.copy_(changed.topk_weights)
            graph.replay()
            torch.cuda.synchronize()
            _check(captured, _reference(tensors, kind, wire))
        graph = None
        torch.cuda.synchronize()
        tensors = make_inputs(17, 7)
        tensors.hidden_states.fill_(1.5)
        expected = layer(tensors, workspace=workspace)
        _check(expected, _reference(tensors, kind, wire))
        if kind == "nvfp4":
            data = torch.full((17, HIDDEN // 2), 0x77, dtype=torch.uint8, device="cuda")
            scales = torch.full(
                (17, HIDDEN // 16), 0x28, dtype=torch.uint8, device="cuda"
            )
        else:
            data = torch.full(
                (17, HIDDEN), 384, dtype=torch.float8_e4m3fn, device="cuda"
            )
            scales = torch.full(
                (17, HIDDEN // 32), 119, dtype=torch.uint8, device="cuda"
            )
        prequantized = replace(tensors, hidden_states=data, scales=scales)
        batch_limit = 32 if kind == "nvfp4" else 4
        variants = [
            (False, False, prequantized, None),
            (
                True,
                False,
                tensors,
                {
                    "epi_flag_batches": (batch_limit, batch_limit),
                    "schedule_policy": ("grouped", 32),
                },
            ),
        ]
        if (kind, wire) == ("nvfp4", "bf16"):
            variants.append((True, True, tensors, None))
        for quantize, fast, inputs, knobs in variants:
            variant = make_layer(quantize_input=quantize, fast_situ=fast, knobs=knobs)
            try:
                handle = variant.create_workspace(256)
                variant.warmup(inputs, workspace=handle)
                result = variant(inputs, workspace=handle)
                if fast:
                    _check(result, expected)
                else:
                    torch.testing.assert_close(result, expected, rtol=0, atol=0)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = variant(inputs, workspace=handle)
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(captured, result, rtol=0, atol=0)
            finally:
                graph = None
                torch.cuda.synchronize()
                variant.destroy()
        workspace.destroy()
        assert workspace.is_destroyed
    finally:
        graph = None
        torch.cuda.synchronize()
        layer.destroy()


@pytest.mark.parametrize("wire", ["bf16", "mxfp8"])
def test_kimi_rejects_unsafe_async_batches(wire):
    rank, world = _runtime()
    from flashinfer.moe_ep.kernel_src.sm100 import _kimi_k3_megamoe as native

    fleet = moe_ep.FleetParams(EXPERTS, 8192, HIDDEN)
    for batches in ((5, 1), (1, 5), (32, 1)):
        config = moe_ep.Sm100_KimiK3_MegaMoeConfig(
            quant_kind="mxfp4_mxfp8",
            combine_dtype=wire,
            situ_beta=4.0,
            situ_linear_beta=25.0,
            knobs={"epi_flag_batches": batches, "schedule_policy": ("grouped", 32)},
        )
        with pytest.raises(ValueError, match="async epilogue flag batches.*at most 4"):
            native.create_workspace(
                config,
                fleet,
                rank=rank,
                world_size=world,
                process_group=dist.group.WORLD if world > 1 else None,
            )
    torch.cuda.synchronize()


@pytest.mark.parametrize("wire", ["bf16", "mxfp8"])
def test_kimi_reducer_skips_poisoned_masked_slots(wire):
    _runtime()
    import cuda.bindings.driver as cuda
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    from flashinfer.moe_ep.kernel_src.sm100 import _kimi_k3_megamoe as native

    dtype = torch.bfloat16 if wire == "bf16" else torch.float8_e4m3fn
    codes = torch.full((3, TOPK, HIDDEN), float("nan"), device="cuda").to(dtype)
    scores = torch.zeros(3, TOPK, device="cuda")
    codes[0, 0].fill_(2)
    codes[0, 15].fill_(4)
    scores[0, 0], scores[0, 15] = 0.25, 0.125
    codes[2, 0].fill_(1)
    codes[2, 15].fill_(-1)
    scores[2, 0], scores[2, 15] = 1.0, 1.0
    scales = None
    if wire == "mxfp8":
        scale_bytes = torch.full(
            (3, TOPK, HIDDEN // 32), 255, dtype=torch.uint8, device="cuda"
        )
        scale_bytes[0, (0, 15)] = 127
        scale_bytes[2, (0, 15)] = 127
        scales = scale_bytes.view(torch.float8_e8m0fnu)
    output = torch.empty(3, HIDDEN, dtype=torch.bfloat16, device="cuda")
    reducer = native.make_reducer(HIDDEN, TOPK, native.get_combine_format(wire))
    arguments = (
        from_dlpack(codes, assumed_align=16),
        None if scales is None else from_dlpack(scales, assumed_align=16),
        from_dlpack(output, assumed_align=16),
        from_dlpack(scores, assumed_align=16),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    compiled = cute.compile(reducer, *arguments)
    compiled(*arguments)
    expected = torch.zeros_like(output)
    expected[0].fill_(1)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    assert (output[1:].view(torch.int16) == 0).all()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled(
            *arguments[:-1], cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        )
    scores.zero_()
    codes.view(torch.uint8).fill_(0x7F if wire == "mxfp8" else 0xFF)
    if scales is not None:
        scales.view(torch.uint8).fill_(255)
    graph.replay()
    torch.cuda.synchronize()
    assert (output.view(torch.int16) == 0).all()
