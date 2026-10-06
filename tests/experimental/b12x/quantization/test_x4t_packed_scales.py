from __future__ import annotations

import numpy as np
import pytest
import torch

from b12x._lib.quant.x4t_scales import make_x4t_scale_batch
from b12x._lib.quant.x4t_packed_scales import (
    decode_x4t_packed_scales,
    _compiled_packed_scale,
    decode_x4t_packed_scale_pair,
    _compiled_packed_scale_pair,
)
from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.moe._shared.kernels.w4a16.prepare import _pack_e8m0_k32_scales
from ..conftest import require_b12x


def _batch(rows, columns, rotation, task_rows=64, *, finite=False):
    device = require_b12x()
    fixed, exceptions, logical = [], [], []
    for expert in range(4):
        bits = (np.arange(rows * columns).reshape(rows, columns) + expert) % 2
        selectors = np.packbits(bits.astype(np.uint8), axis=1, bitorder="little")
        base = 120 + expert
        values = (bits + base).astype(np.uint8)
        positions = np.unique(
            [
                0,
                63 * columns,
                64 * columns - 1,
                64 * columns,
                (rows // 2 + 1) * columns,
                rows * columns - 1,
            ]
        )
        overrides = np.array([0, 246, 247, 248, 254, 255], dtype=np.uint32)[
            -len(positions) :
        ]
        if finite:
            overrides = np.full(len(positions), 122 + expert, dtype=np.uint32)
        values.flat[positions] = overrides
        words = positions.astype(np.uint32) | (overrides << 24)
        bases = np.full((rows // 16, 16), base, dtype=np.uint8)
        stream = np.concatenate((bases, selectors.reshape(rows // 16, -1)), axis=1)
        fixed.append(torch.from_numpy(stream))
        exceptions.append(torch.from_numpy(words))
        logical.append(torch.from_numpy(values))
    batch = make_x4t_scale_batch(
        fixed,
        exceptions,
        rows=rows,
        columns=columns,
        device=device,
        exception_task_rows=task_rows,
        exception_row_rotation=rotation,
    )
    return batch, torch.stack(logical).to(device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "rows,columns,rotation",
    [
        (128, 1, 0),
        (128, 18, 64),
        (1152, 160, 0),
        (5120, 18, 0),
        (2304, 160, 1152),
        (5120, 36, 0),
        (4608, 160, 0),
        (5120, 72, 0),
    ],
)
def test_packed_scale_exact_boundaries_duplicates_and_dynamic_graph(
    rows, columns, rotation, ids_dtype
):
    batch, logical = _batch(rows, columns, rotation)
    output = torch.full(
        (4, columns, rows), 0xD6, dtype=torch.uint8, device=logical.device
    )
    invalid = 2**32 + 3 if ids_dtype == torch.int64 else 4
    ids = torch.tensor([3, 1, 3, -1, invalid], dtype=ids_dtype, device=logical.device)
    decode_x4t_packed_scales(batch, ids, output)
    reference = _pack_e8m0_k32_scales(
        logical, size_k=columns * 32, size_n=rows, row_rotation=rotation
    ).view(torch.uint8)
    assert torch.equal(output[[3, 1]], reference[[3, 1]])
    assert bool((output[[0, 2]] == 0xD6).all())
    misses = _compiled_packed_scale.cache_info().misses
    with kernel_resolution_guard("packed-scale graph qualification"):
        for count in (1, 3, 5):
            decode_x4t_packed_scales(batch, ids[:count], output)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            decode_x4t_packed_scales(batch, ids, output)
        ids.copy_(
            torch.tensor([2, 0, 2, -1, invalid], dtype=ids_dtype, device=ids.device)
        )
        output.fill_(0xD6)
        allocated = torch.cuda.memory_allocated()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated
    assert _compiled_packed_scale.cache_info().misses == misses
    assert torch.equal(output[[2, 0]], reference[[2, 0]])
    assert bool((output[[1, 3]] == 0xD6).all())
    graph.reset()
    # The exact-byte mode preserves the full UE8M0 alphabet. The serving clamp
    # is a distinct operation shared with native W4A16 preparation.
    ids_unique = torch.arange(4, dtype=torch.int32, device=ids.device)
    decode_x4t_packed_scales(
        batch, ids_unique, output, clamp_e8m0_bf16=False, expert_ids_unique=True
    )
    from b12x.moe._shared.kernels.w4a16.prepare import _scale_perms

    permutation = _scale_perms()[int(columns == 1)]
    rotated = torch.roll(logical, -rotation, dims=1).transpose(1, 2).contiguous()
    exact = rotated.reshape(-1, len(permutation))[:, permutation]
    exact = exact.reshape(-1, 4)[:, [0, 2, 1, 3]].reshape_as(output)
    assert torch.equal(output, exact)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_packed_scale_rejects_misaligned_exception_partition():
    batch, logical = _batch(128, 18, 0, task_rows=32)
    ids = torch.arange(4, dtype=torch.int32, device=logical.device)
    output = torch.empty((4, 18, 128), dtype=torch.uint8, device=logical.device)
    with pytest.raises(ValueError, match="64 rows"):
        decode_x4t_packed_scales(batch, ids, output)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_packed_counts_retained_program_and_poisoned_graph():
    batch, logical = _batch(1152, 160, 0)
    output = torch.full((4, 160, 1152), 0xD6, dtype=torch.uint8, device=logical.device)
    counts = torch.tensor([3, 0, 8, 0], dtype=torch.int32, device=logical.device)
    program = _compiled_packed_scale(1152, 160, 64, 0, True, False, True)
    reference = _pack_e8m0_k32_scales(logical, size_k=5120, size_n=1152).view(
        torch.uint8
    )
    decode_x4t_packed_scales(batch, counts, output, expert_counts=True, program=program)
    assert torch.equal(output[[0, 2]], reference[[0, 2]])
    _compiled_packed_scale.cache_clear()
    with kernel_resolution_guard("retained X4T counts decoder"):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            decode_x4t_packed_scales(
                batch, counts, output, expert_counts=True, program=program
            )
        for active in ([0, 4, 0, 10], [0, 0, 0, 0], [1, 0, 4, 2]):
            counts.copy_(torch.tensor(active, dtype=torch.int32, device=counts.device))
            output.fill_(0xD6)
            graph.replay()
            torch.cuda.synchronize()
            for expert, count in enumerate(active):
                if count:
                    assert torch.equal(output[expert], reference[expert])
                else:
                    assert bool((output[expert] == 0xD6).all())
    graph.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_native_x4t_keeps_nibbles_and_shares_micro_and_prefill_scales():
    from b12x.moe import fused_moe

    fc1, _ = _batch(1152, 160, 0)
    fc2, _ = _batch(5120, 18, 0)
    device = fc1.fixed.device
    w13 = torch.zeros((4, 1152, 2560), dtype=torch.uint8, device=device)
    w2 = torch.zeros((4, 5120, 288), dtype=torch.uint8, device=device)
    scales13 = torch.empty((4, 160, 1152), dtype=torch.uint8, device=device)
    scales2 = torch.empty((4, 18, 5120), dtype=torch.uint8, device=device)
    plan = fused_moe.plan_weights(
        source=fused_moe.PackedSource(format="fp4_e8m0_k32", w13_layout="w31"),
        activation=fused_moe.ActivationSpec(
            mode="a16",
            nonlinearity="silu",
            io_dtype=torch.bfloat16,
            swiglu_limit=10.0,
        ),
        geometry=fused_moe.MoEGeometry(
            num_experts=4,
            hidden_size=5120,
            intermediate_size=576,
        ),
    )
    experts = fused_moe.prepare_weights(
        plan=plan,
        weights=fused_moe.X4TWeights(
            w13=w13,
            w2=w2,
            w13_scales=fc1,
            w2_scales=fc2,
            w13_scale_scratch=scales13,
            w2_scale_scratch=scales2,
        ),
    )
    assert experts.plan.prepared_format.packing.value == "source_native"
    payload = experts._impl.representation.value
    assert payload.w13 is w13 and payload.w2 is w2
    assert (
        payload.w13_scale.data_ptr()
        == payload.micro_w13_scale.data_ptr()
        == scales13.data_ptr()
    )
    assert (
        payload.w2_scale.data_ptr()
        == payload.micro_w2_scale.data_ptr()
        == scales2.data_ptr()
    )
    assert len(payload.x4t_packed_pair_programs) == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("tp", [1, 2, 4, 8])
@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("mode", ["direct", "sorted", "counts"])
def test_packed_pair_retained_program_exact_poisoned_graph(tp, ids_dtype, mode):
    local = 2304 // tp
    planes_and_logical = (_batch(2 * local, 160, 0), _batch(5120, local // 32, 0))
    planes = tuple(item[0] for item in planes_and_logical)
    device = planes[0].fixed.device
    outputs = tuple(
        torch.full((4, p.columns, p.rows), 0xD6, dtype=torch.uint8, device=device)
        for p in planes
    )
    references = tuple(
        _pack_e8m0_k32_scales(logical, size_k=p.columns * 32, size_n=p.rows).view(
            torch.uint8
        )
        for p, logical in planes_and_logical
    )
    invalid = 2**32 + 3 if ids_dtype == torch.int64 else 4
    if mode == "counts":
        mutations = ([0, 3, 0, 8], [5, 0, 9, 0], [0, 0, 0, 0])
    elif mode == "sorted":
        mutations = ([1, 1, 3, 3, -1, invalid], [0, 0, 2, 2, -1, invalid], [-1] * 6)
    else:
        mutations = ([3, 1, 3, 1, -1, invalid], [2, 0, 2, 0, -1, invalid], [-1] * 6)
    ids = torch.tensor(mutations[0], dtype=ids_dtype, device=device)
    keys = tuple(
        (
            p.rows,
            p.columns,
            64,
            0,
            True,
            False,
            mode == "counts",
            ids_dtype == torch.int64,
            mode == "sorted",
        )
        for p in planes
    )
    program = _compiled_packed_scale_pair(*keys)

    def run(active_ids):
        decode_x4t_packed_scale_pair(
            *planes,
            active_ids,
            *outputs,
            expert_counts=mode == "counts",
            expert_ids_sorted=mode == "sorted",
            program=program,
        )

    run(ids)
    _compiled_packed_scale_pair.cache_clear()
    with kernel_resolution_guard("retained paired X4T decoder"):
        if mode != "counts":
            for count in (0, 1, 3, 6):
                run(ids[:count])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(ids)
        addresses = tuple(t.data_ptr() for t in outputs)
        for mutation in mutations * 4:
            ids.copy_(torch.tensor(mutation, dtype=ids_dtype, device=device))
            for output in outputs:
                output.fill_(0xD6)
            allocated = torch.cuda.memory_allocated()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == allocated
            assert tuple(t.data_ptr() for t in outputs) == addresses
            active = (
                set(i for i, count in enumerate(mutation) if count > 0)
                if mode == "counts"
                else set(i for i in mutation if 0 <= i < 4)
            )
            for output, reference in zip(outputs, references, strict=True):
                for expert in range(4):
                    if expert in active:
                        assert torch.equal(output[expert], reference[expert])
                    else:
                        assert bool((output[expert] == 0xD6).all())
    graph.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_packed_pair_rejects_conflicting_modes_and_aliases():
    batch, logical = _batch(128, 18, 64)
    ids = torch.arange(4, dtype=torch.int32, device=logical.device)
    output = torch.empty((4, 18, 128), dtype=torch.uint8, device=logical.device)
    with pytest.raises(ValueError, match="must not overlap"):
        decode_x4t_packed_scale_pair(batch, batch, ids, output, output)
    with pytest.raises(ValueError, match="cannot also"):
        decode_x4t_packed_scales(
            batch, ids, output, expert_counts=True, expert_ids_sorted=True
        )


@pytest.mark.parametrize("tokens", [1, 129])
@pytest.mark.parametrize("packing", ["packed", "modelopt"])
def test_paired_x4t_moe_without_caller_counts_matches_dense_scales(tokens, packing, monkeypatch):
    from b12x.moe._shared.kernels.w4a16.kernel import run_w4a16_moe
    from b12x.moe._shared.kernels.w4a16.prepare import (
        prepare_w4a16_fp4_e8m0_k32_weights,
        prepare_w4a16_e8m0_native_weights,
        prepare_w4a16_x4t_weights,
        make_w4a16_packed_buffers,
    )
    from b12x._lib.quant.x4t_scales import X4TScaleBatch

    hidden, intermediate, experts, topk = 3584, 192, 4, 2
    first, logical_first = _batch(2 * intermediate, hidden // 32, 0, finite=True)
    second, logical_second = _batch(hidden, intermediate // 32, 0, finite=True)
    device = first.fixed.device
    w13 = torch.randint(256, (experts, 2 * intermediate, hidden // 2), dtype=torch.uint8, device=device)
    w2 = torch.randint(256, (experts, hidden, intermediate // 2), dtype=torch.uint8, device=device)
    unit = torch.ones(experts, dtype=torch.float32, device=device)
    scales = (
        torch.empty((experts, hidden // 32, 2 * intermediate), dtype=torch.uint8, device=device),
        torch.empty((experts, intermediate // 32, hidden), dtype=torch.uint8, device=device),
    )
    prepare_dense = (prepare_w4a16_e8m0_native_weights if packing == "modelopt"
                     else prepare_w4a16_fp4_e8m0_k32_weights)
    dense = prepare_dense(
        w13, logical_first, unit, w2, logical_second, unit,
        activation="situ", w13_layout="w31",
    )
    with pytest.raises(ValueError, match="must not overlap"):
        prepare_w4a16_x4t_weights(
            w13, first, unit, w2, second, unit, scales[0],
            scales[0].reshape(-1)[:scales[1].numel()].reshape_as(scales[1]),
            activation="situ", w13_layout="w31", weight_layout=packing,
        )
    nonpaired, _ = _batch(2 * intermediate, hidden // 32, 0, task_rows=32, finite=True)
    with pytest.raises(ValueError, match="paired 64-row tasks"):
        prepare_w4a16_x4t_weights(
            w13, nonpaired, unit, w2, second, unit, *scales,
            activation="situ", w13_layout="w31", weight_layout=packing,
        )
    compressed = prepare_w4a16_x4t_weights(
        w13, first, unit, w2, second, unit, *scales,
        activation="situ", w13_layout="w31", weight_layout=packing,
    )
    x = torch.randn((tokens, hidden), dtype=torch.bfloat16, device=device) * 0.125
    ids = torch.randint(experts, (tokens, topk), dtype=torch.int32, device=device)
    route_weights = torch.softmax(torch.randn(tokens, topk, device=device), -1)
    buffers = [make_w4a16_packed_buffers(p, m=tokens, topk=topk, dtype=x.dtype, device=device)
               for p in (dense, compressed)]

    def run(prepared, scratch):
        return run_w4a16_moe(
            x, prepared, route_weights, ids, activation="situ",
            intermediate_cache13=scratch.intermediate_cache13,
            intermediate_cache2=scratch.intermediate_cache2,
            output=scratch.output, fc1_c_tmp=scratch.fc1_c_tmp, fc2_c_tmp=scratch.fc2_c_tmp,
            packed_route_indices=scratch.packed_route_indices,
            block_expert_ids=scratch.block_expert_ids,
            packed_route_count=scratch.packed_route_count,
            expert_offsets=scratch.expert_offsets,
        )

    monkeypatch.setattr(X4TScaleBatch, "validate", lambda *_: pytest.fail("scale validation during execution"))
    expected = run(dense, buffers[0]).clone()
    actual = run(compressed, buffers[1]).clone()
    assert torch.isfinite(actual).all() and actual.abs().any()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with kernel_resolution_guard("paired X4T MoE graph"):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(compressed, buffers[1])
        for _ in range(3):
            ids.random_(0, experts)
            ids[0, 0] = -1
            x.mul_(-0.9)
            expected = run(dense, buffers[0]).clone()
            for scale in scales:
                scale.fill_(0xD6)
            buffers[1].output.fill_(float("nan"))
            allocated = torch.cuda.memory_allocated()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == allocated
            torch.testing.assert_close(buffers[1].output, expected, rtol=0, atol=0)
        graph.reset()
