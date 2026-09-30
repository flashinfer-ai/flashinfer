from __future__ import annotations

import pytest
import torch

from b12x._lib.quant.x4t_scales import (
    X4TScaleBatch,
    decode_x4t_scales,
    decode_x4t_tp12_w4a16_scales,
    make_x4t_scale_batch,
)
from b12x.moe._shared.kernels.w4a16.kernel import run_w4a16_moe
from b12x.moe._shared.kernels.w4a16.prepare import (
    _pack_e8m0_k32_scales,
    make_w4a16_packed_buffers,
    prepare_w4a16_fp4_e8m0_k32_weights,
    prepare_w4a16_x4t_weights,
)

from ..conftest import require_b12x


def _make_plane(
    rows: int,
    columns: int,
    expert: int,
    *,
    base_start: int = 72,
    clamp_edge: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a canonical X4T component pair and its exact logical scale plane."""

    assert rows % 16 == 0
    selector_bytes = (columns + 7) // 8
    tile_bytes = 16 * (1 + selector_bytes)
    fixed = torch.zeros((rows // 16, tile_bytes), dtype=torch.uint8)
    logical = torch.empty((rows, columns), dtype=torch.uint8)

    for row in range(rows):
        tile = row // 16
        local_row = row % 16
        base = base_start + ((row * 3 + expert * 5) % 4)
        fixed[tile, local_row] = base
        for column in range(columns):
            bit = (row + column * 3 + expert) & 1
            logical[row, column] = base + bit
            selector_offset = 16 + local_row * selector_bytes + column // 8
            fixed[tile, selector_offset] |= bit << (column & 7)

    # Exercise sparse overrides, including the BF16 kernel's E8M0 clamp edge.
    positions = sorted(
        {
            (expert * 17 + 3) % (rows * columns),
            (expert * 29 + rows + 5) % (rows * columns),
            rows * columns - 1,
        }
    )
    entries: list[int] = []
    for index, position in enumerate(positions):
        value = (
            251
            if clamp_edge and index == len(positions) - 1
            else 125 + ((expert + index) % 3)
        )
        logical.view(-1)[position] = value
        entries.append(position | (value << 24))
    exceptions = torch.tensor(entries, dtype=torch.uint32)
    return fixed.contiguous(), exceptions, logical


def _make_batch(
    *,
    rows: int,
    columns: int,
    experts: int,
    device: torch.device,
    base_start: int = 72,
    clamp_edge: bool = True,
) -> tuple[X4TScaleBatch, torch.Tensor]:
    components = [
        _make_plane(
            rows,
            columns,
            expert,
            base_start=base_start,
            clamp_edge=clamp_edge,
        )
        for expert in range(experts)
    ]
    task_kwargs = {}
    if (rows, columns) == (512, 112):
        task_kwargs = {"exception_task_rows": 128, "exception_row_rotation": 256}
    elif (rows, columns) == (3584, 8):
        task_kwargs = {"exception_task_rows": 896}
    batch = make_x4t_scale_batch(
        [item[0] for item in components],
        [item[1] for item in components],
        rows=rows,
        columns=columns,
        device=device,
        **task_kwargs,
    )
    logical = torch.stack([item[2] for item in components]).to(device)
    return batch, logical


def test_x4t_scale_batch_rejects_mismatched_component_counts() -> None:
    fixed = torch.empty((1,), dtype=torch.uint8)
    with pytest.raises(ValueError, match="nonempty and equally sized"):
        make_x4t_scale_batch(
            [fixed], [], rows=16, columns=8, device="cuda"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_x4t_scale_batch_accepts_target_device_components() -> None:
    device = require_b12x()
    components = [_make_plane(128, 8, expert) for expert in range(3)]
    batch = make_x4t_scale_batch(
        [item[0].to(device) for item in components],
        [item[1].to(device) for item in components],
        rows=128,
        columns=8,
        device=device,
        exception_task_rows=32,
    )

    expected_device = torch.device("cuda", torch.cuda.current_device())
    assert batch.fixed.device == expected_device
    assert batch.exceptions.device == expected_device
    assert torch.equal(
        batch.fixed.cpu(), torch.stack([item[0] for item in components])
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("columns", [8, 112])
def test_x4t_cuda_decode_is_exact_and_leaves_inactive_experts(columns: int) -> None:
    device = require_b12x()
    batch, logical = _make_batch(rows=128, columns=columns, experts=4, device=device)
    expert_ids = torch.tensor([3, 1, -1], dtype=torch.int32, device=device)
    sentinel = 0xD6
    output = torch.full(
        (batch.num_experts, batch.rows, batch.columns),
        sentinel,
        dtype=torch.uint8,
        device=device,
    )

    decode_x4t_scales(batch, expert_ids, output)
    torch.cuda.synchronize()

    assert torch.equal(output[3], logical[3])
    assert torch.equal(output[1], logical[1])
    assert bool((output[0] == sentinel).all().item())
    assert bool((output[2] == sentinel).all().item())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_x4t_cuda_decode_matches_w4a16_packed_layout_and_graph_replay() -> None:
    device = require_b12x()
    rows, columns, experts = 128, 8, 4
    batch, logical = _make_batch(
        rows=rows, columns=columns, experts=experts, device=device
    )
    expert_ids = torch.tensor([3, 1, -1], dtype=torch.int32, device=device)
    output = torch.full(
        (experts, columns, rows), 0xD6, dtype=torch.uint8, device=device
    )
    row_rotation = 64

    # Compile before capture, as serving warmup does.
    decode_x4t_scales(
        batch,
        expert_ids,
        output,
        layout="w4a16_packed",
        row_rotation=row_rotation,
        clamp_e8m0_bf16=True,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        decode_x4t_scales(
            batch,
            expert_ids,
            output,
            layout="w4a16_packed",
            row_rotation=row_rotation,
            clamp_e8m0_bf16=True,
        )

    output.fill_(0xD6)
    expert_ids.copy_(torch.tensor([2, 0, -1], dtype=torch.int32, device=device))
    graph.replay()
    torch.cuda.synchronize()

    reference = _pack_e8m0_k32_scales(
        logical,
        size_k=columns * 32,
        size_n=rows,
        row_rotation=row_rotation,
    ).view(torch.uint8)
    assert torch.equal(output[2], reference[2])
    assert torch.equal(output[0], reference[0])
    assert bool((output[1] == 0xD6).all().item())
    assert bool((output[3] == 0xD6).all().item())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_x4t_tp12_single_launch_matches_both_w4a16_scale_planes() -> None:
    device = require_b12x()
    experts = 3
    w13, logical_w13 = _make_batch(
        rows=512, columns=112, experts=experts, device=device
    )
    w2, logical_w2 = _make_batch(
        rows=3584, columns=8, experts=experts, device=device
    )
    expert_ids = torch.tensor([0, 2, 0, -1], dtype=torch.int32, device=device)
    expert_map = torch.tensor([2, -1, 0, 1], dtype=torch.int32, device=device)
    output_w13 = torch.full(
        (experts, 112, 512), 0xD6, dtype=torch.uint8, device=device
    )
    output_w2 = torch.full(
        (experts, 8, 3584), 0xD6, dtype=torch.uint8, device=device
    )

    decode_x4t_tp12_w4a16_scales(
        w13,
        w2,
        expert_ids,
        output_w13,
        output_w2,
        expert_map=expert_map,
    )
    torch.cuda.synchronize()

    reference_w13 = _pack_e8m0_k32_scales(
        logical_w13,
        size_k=112 * 32,
        size_n=512,
        row_rotation=256,
    ).view(torch.uint8)
    reference_w2 = _pack_e8m0_k32_scales(
        logical_w2,
        size_k=8 * 32,
        size_n=3584,
    ).view(torch.uint8)
    for expert in (2, 0):
        assert torch.equal(output_w13[expert], reference_w13[expert])
        assert torch.equal(output_w2[expert], reference_w2[expert])
    assert bool((output_w13[1] == 0xD6).all().item())
    assert bool((output_w2[1] == 0xD6).all().item())

    # Capture the mapped/deduplicated production variant, then change the
    # fixed-capacity route list before replay. Only mapped expert 1 is live.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        decode_x4t_tp12_w4a16_scales(
            w13,
            w2,
            expert_ids,
            output_w13,
            output_w2,
            expert_map=expert_map,
        )
    output_w13.fill_(0xD6)
    output_w2.fill_(0xD6)
    expert_ids.copy_(
        torch.tensor([3, 1, 3, -1], dtype=torch.int32, device=device)
    )
    graph.replay()
    torch.cuda.synchronize()

    assert torch.equal(output_w13[1], reference_w13[1])
    assert torch.equal(output_w2[1], reference_w2[1])
    for expert in (0, 2):
        assert bool((output_w13[expert] == 0xD6).all().item())
        assert bool((output_w2[expert] == 0xD6).all().item())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_x4t_tp12_w4a16_moe_matches_dense_scales_and_graph_replay() -> None:
    """Exercise X4T inside the actual routed TP12 W4A16 execution path."""

    device = require_b12x()
    torch.manual_seed(20260803)
    experts, hidden_size, intermediate_size = 3, 3584, 256
    w13_rows = 2 * intermediate_size
    w13_x4t, logical_w13 = _make_batch(
        rows=w13_rows,
        columns=hidden_size // 32,
        experts=experts,
        device=device,
        base_start=119,
        clamp_edge=False,
    )
    w2_x4t, logical_w2 = _make_batch(
        rows=hidden_size,
        columns=intermediate_size // 32,
        experts=experts,
        device=device,
        base_start=119,
        clamp_edge=False,
    )
    w13 = torch.randint(
        0,
        256,
        (experts, w13_rows, hidden_size // 2),
        dtype=torch.uint8,
        device=device,
    )
    w2 = torch.randint(
        0,
        256,
        (experts, hidden_size, intermediate_size // 2),
        dtype=torch.uint8,
        device=device,
    )
    w13_global = torch.ones(experts, dtype=torch.float32, device=device)
    w2_global = torch.ones(experts, dtype=torch.float32, device=device)
    dense = prepare_w4a16_fp4_e8m0_k32_weights(
        w13,
        logical_w13,
        w13_global,
        w2,
        logical_w2,
        w2_global,
        activation="situ",
        params_dtype=torch.bfloat16,
        w13_layout="w13",
    )
    sentinel = 0xD6
    w13_scratch = torch.full(
        (experts, hidden_size // 32, w13_rows),
        sentinel,
        dtype=torch.uint8,
        device=device,
    )
    w2_scratch = torch.full(
        (experts, intermediate_size // 32, hidden_size),
        sentinel,
        dtype=torch.uint8,
        device=device,
    )
    compressed = prepare_w4a16_x4t_weights(
        w13,
        w13_x4t,
        w13_global,
        w2,
        w2_x4t,
        w2_global,
        w13_scratch,
        w2_scratch,
        activation="situ",
        params_dtype=torch.bfloat16,
        w13_layout="w13",
    )
    assert torch.equal(compressed.w13, dense.w13)
    assert torch.equal(compressed.w2, dense.w2)

    m, topk = 1, 2
    x = (torch.randn(m, hidden_size, device=device) * 0.125).to(torch.bfloat16)
    topk_ids = torch.tensor([[3, 0]], dtype=torch.int32, device=device)
    topk_weights = torch.tensor([[0.625, 0.375]], dtype=torch.float32, device=device)
    expert_map = torch.tensor([2, -1, 0, 1], dtype=torch.int32, device=device)
    dense_buffers = make_w4a16_packed_buffers(
        dense,
        m=m,
        topk=topk,
        dtype=torch.bfloat16,
        device=device,
        route_num_experts=int(expert_map.numel()),
    )
    compressed_buffers = make_w4a16_packed_buffers(
        compressed,
        m=m,
        topk=topk,
        dtype=torch.bfloat16,
        device=device,
        route_num_experts=int(expert_map.numel()),
    )

    def launch(prepared, buffers) -> torch.Tensor:
        return run_w4a16_moe(
            x,
            prepared,
            topk_weights,
            topk_ids,
            activation="situ",
            expert_map=expert_map,
            intermediate_cache13=buffers.intermediate_cache13,
            intermediate_cache2=buffers.intermediate_cache2,
            output=buffers.output,
            fc1_c_tmp=buffers.fc1_c_tmp,
            fc2_c_tmp=buffers.fc2_c_tmp,
            packed_route_indices=buffers.packed_route_indices,
            block_expert_ids=buffers.block_expert_ids,
            packed_route_count=buffers.packed_route_count,
            expert_offsets=buffers.expert_offsets,
        )

    dense_output = launch(dense, dense_buffers).clone()
    compressed_output = launch(compressed, compressed_buffers).clone()
    torch.cuda.synchronize(device)
    assert bool((dense_output != 0).any().item())
    torch.testing.assert_close(compressed_output, dense_output, rtol=0, atol=0)
    for expert in (1, 2):
        assert torch.equal(
            compressed.w13_scale[expert], dense.w13_scale[expert].view(torch.uint8)
        )
        assert torch.equal(
            compressed.w2_scale[expert], dense.w2_scale[expert].view(torch.uint8)
        )
    assert bool((compressed.w13_scale[0] == sentinel).all().item())
    assert bool((compressed.w2_scale[0] == sentinel).all().item())

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(compressed, compressed_buffers)
    compressed.w13_scale.fill_(sentinel)
    compressed.w2_scale.fill_(sentinel)
    compressed_buffers.output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize(device)
    torch.testing.assert_close(
        compressed_buffers.output, dense_output, rtol=0, atol=0
    )
