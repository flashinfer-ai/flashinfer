from __future__ import annotations

import pytest
import torch

from flashinfer.experimental.sm12x.comm.pcie.pcie_dcp_a2a import (
    PCIeDCPA2A,
    _staging_layout,
    lse_reduce_scatter_reference,
)


class _FakeExt:
    def __init__(self) -> None:
        self.init_calls = []
        self.run_calls = []
        self.dispose_calls = []

    def init_dcp_a2a(
        self,
        signal_ptrs,
        staging0_ptrs,
        staging1_ptrs,
        output_capacity_elems,
        lse_offset,
        lse_capacity,
        rank,
    ):
        self.init_calls.append(
            (
                tuple(signal_ptrs),
                tuple(staging0_ptrs),
                tuple(staging1_ptrs),
                output_capacity_elems,
                lse_offset,
                lse_capacity,
                rank,
            )
        )
        return 1234

    def lse_reduce_scatter(
        self,
        pointer,
        partial_output,
        partial_lse,
        out,
        natural_log,
        threads,
        block_limit,
    ):
        self.run_calls.append(
            (pointer, natural_log, threads, block_limit, tuple(partial_output.shape))
        )
        heads_per_rank = partial_output.shape[1] // 2
        out.copy_(partial_output[:, :heads_per_rank])

    def all_gather_heads(
        self,
        pointer,
        local_input,
        out,
        threads,
        block_limit,
    ):
        self.run_calls.append(
            (
                pointer,
                "all_gather_heads",
                threads,
                block_limit,
                tuple(local_input.shape),
            )
        )
        out.copy_(torch.cat((local_input, local_input), dim=1))

    def dispose(self, pointer):
        self.dispose_calls.append(pointer)


def _make_runtime(ext: _FakeExt | None = None) -> PCIeDCPA2A:
    return PCIeDCPA2A(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        signal_ptrs=(100, 200),
        staging0_ptrs=(300, 400),
        staging1_ptrs=(500, 600),
        max_batch_size=4,
        total_heads=32,
        head_dim=64,
        output_capacity_elems=4 * 32 * 64,
        lse_offset=4 * 32 * 64 * 2,
        lse_capacity=4 * 32,
        ext_module=ext or _FakeExt(),
    )


def test_staging_layout_has_aligned_disjoint_slots():
    layout = _staging_layout(
        signal_bytes=12345,
        world_size=2,
        max_batch_size=4,
        total_heads=32,
        head_dim=512,
    )

    assert layout.staging0_offset % 256 == 0
    assert layout.staging1_offset == layout.staging0_offset + layout.slot_bytes
    assert layout.lse_offset % 256 == 0
    assert layout.slab_bytes == layout.staging1_offset + layout.slot_bytes
    assert layout.output_capacity_elems >= 4 * 32 * 512
    assert layout.lse_capacity >= 4 * 32

    wider_query_layout = _staging_layout(
        signal_bytes=12345,
        world_size=2,
        max_batch_size=4,
        total_heads=32,
        head_dim=512,
        query_head_dim=576,
    )
    assert wider_query_layout.output_capacity_elems >= 4 * 32 * 576
    assert wider_query_layout.lse_offset > layout.lse_offset


def test_reference_selects_destination_heads_and_combines_lse_weights():
    outputs = torch.tensor(
        [
            [[[[1.0], [2.0], [10.0], [20.0]]]],
            [[[[3.0], [4.0], [30.0], [40.0]]]],
        ],
        dtype=torch.float32,
    ).reshape(2, 1, 4, 1)
    lses = torch.tensor(
        [
            [[0.0, 0.0, 0.0, -torch.inf]],
            [[0.0, -torch.inf, 0.0, -torch.inf]],
        ]
    )

    rank0 = lse_reduce_scatter_reference(outputs, lses, 0)
    rank1 = lse_reduce_scatter_reference(outputs, lses, 1)

    torch.testing.assert_close(rank0, torch.tensor([[[2.0], [2.0]]]))
    torch.testing.assert_close(rank1, torch.tensor([[[20.0], [0.0]]]))


def test_runtime_validates_and_dispatches_to_extension():
    ext = _FakeExt()
    runtime = _make_runtime(ext)
    partial_output = torch.arange(2 * 32 * 64, dtype=torch.bfloat16).reshape(2, 32, 64)
    partial_lse = torch.zeros(2, 32, dtype=torch.float32)

    out = runtime.lse_reduce_scatter(
        partial_output,
        partial_lse,
        is_lse_base_on_e=False,
        threads=256,
        block_limit=32,
    )

    assert out.shape == (2, 16, 64)
    assert torch.equal(out, partial_output[:, :16])
    assert ext.run_calls == [(1234, False, 256, 32, (2, 32, 64))]

    local_input = partial_output[:, :16].contiguous()
    gathered = runtime.all_gather_heads(
        local_input,
        threads=64,
        block_limit=16,
    )
    assert gathered.shape == partial_output.shape
    assert torch.equal(gathered, torch.cat((local_input, local_input), dim=1))
    assert ext.run_calls[-1] == (
        1234,
        "all_gather_heads",
        64,
        16,
        (2, 16, 64),
    )
    runtime.close()
    assert ext.dispose_calls == [1234]


def test_runtime_rejects_shape_dtype_and_capacity_mismatches():
    runtime = _make_runtime()
    good_output = torch.zeros(1, 32, 64, dtype=torch.bfloat16)
    good_lse = torch.zeros(1, 32, dtype=torch.float32)

    with pytest.raises(ValueError, match="float32"):
        runtime.lse_reduce_scatter(good_output, good_lse.bfloat16())
    with pytest.raises(ValueError, match="configured heads/head_dim"):
        runtime.lse_reduce_scatter(good_output[:, :, :32], good_lse)
    with pytest.raises(ValueError, match="exceeds configured capacity"):
        runtime.lse_reduce_scatter(
            torch.zeros(5, 32, 64, dtype=torch.bfloat16),
            torch.zeros(5, 32, dtype=torch.float32),
        )

    with pytest.raises(ValueError, match="configured local heads/head_dim"):
        runtime.all_gather_heads(good_output[:, :8])
    with pytest.raises(ValueError, match="exceeds configured capacity"):
        runtime.all_gather_heads(torch.zeros(5, 16, 64, dtype=torch.bfloat16))
