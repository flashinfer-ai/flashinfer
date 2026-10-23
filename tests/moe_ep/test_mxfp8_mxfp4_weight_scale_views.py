# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused layout tests for transformed MXFP4 weight scale views."""

import pytest
import torch

from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.weight_scale_views import (
    SF_ATOM_BYTES,
    SwizzledWeightScaleViews,
    atom_swizzled_offset,
    from_atom_swizzled,
    swizzled_scale_numel,
    to_atom_swizzled,
)


def test_atom_swizzle_round_trip_with_padding_and_nonreshape_order() -> None:
    rows, k_blocks = 133, 7
    logical = torch.arange(rows * k_blocks, dtype=torch.int64).reshape(rows, k_blocks)
    swizzled = to_atom_swizzled(logical)

    assert swizzled.numel() == 2 * 2 * SF_ATOM_BYTES
    assert torch.equal(
        from_atom_swizzled(swizzled, rows=rows, k_elements=k_blocks * 32),
        logical,
    )
    assert not torch.equal(swizzled[: logical.numel()], logical.reshape(-1))
    for row, k_block in ((0, 0), (1, 0), (31, 3), (32, 0), (127, 6), (132, 6)):
        offset = atom_swizzled_offset(row, k_block, rows=rows, k_elements=k_blocks * 32)
        assert swizzled[offset] == logical[row, k_block]


def test_fc1_gate_up_interleave_is_preserved_by_atom_layout() -> None:
    intermediate, hidden = 128, 128
    canonical = torch.arange(2 * intermediate, dtype=torch.int64)[:, None].expand(
        -1, hidden // 32
    )
    interleaved = torch.empty_like(canonical).view(-1, 2, 8, hidden // 32)
    interleaved[:, 0].copy_(canonical[:intermediate].view(-1, 8, hidden // 32))
    interleaved[:, 1].copy_(canonical[intermediate:].view(-1, 8, hidden // 32))
    physical = to_atom_swizzled(interleaved.reshape(2 * intermediate, -1))

    restored = from_atom_swizzled(physical, rows=2 * intermediate, k_elements=hidden)
    for block in range(intermediate // 8):
        assert torch.equal(
            restored[16 * block : 16 * block + 8], canonical[8 * block : 8 * block + 8]
        )
        assert torch.equal(
            restored[16 * block + 8 : 16 * block + 16],
            canonical[intermediate + 8 * block : intermediate + 8 * block + 8],
        )


def test_adapter_freezes_swap_ab_shapes_and_flat_extents() -> None:
    experts, hidden, intermediate = 3, 128, 256
    fc1 = torch.zeros(
        experts,
        swizzled_scale_numel(2 * intermediate, hidden),
        dtype=torch.uint8,
    )
    fc2 = torch.zeros(
        experts,
        swizzled_scale_numel(hidden, intermediate),
        dtype=torch.uint8,
    )
    views = SwizzledWeightScaleViews(
        fc1, fc2, hidden_size=hidden, intermediate_size=intermediate
    )

    assert views.fc1_logical_shape == (2 * intermediate, hidden, experts)
    assert views.fc2_logical_shape == (hidden, intermediate, experts)
    with pytest.raises(ValueError, match="CUDA"):
        views.to_cute()


def test_adapter_rejects_a_row_major_scale_matrix_disguised_as_flat() -> None:
    hidden = intermediate = 128
    correct_fc1 = swizzled_scale_numel(2 * intermediate, hidden)
    correct_fc2 = swizzled_scale_numel(hidden, intermediate)
    with pytest.raises(ValueError, match="bytes per expert"):
        SwizzledWeightScaleViews(
            torch.zeros((1, correct_fc1 - 1), dtype=torch.uint8),
            torch.zeros((1, correct_fc2), dtype=torch.uint8),
            hidden_size=hidden,
            intermediate_size=intermediate,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_b200_traces_and_reads_atom_tiled_logical_view() -> None:
    pytest.importorskip("cutlass")
    from flashinfer.utils import is_sm100a_supported

    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("requires SM100/SM103 and CUDA 12.8+")

    cutlass = pytest.importorskip("cutlass")
    import cutlass.cute as cute
    from cuda.bindings import driver as cuda
    from cutlass.cute.runtime import from_dlpack

    hidden, intermediate = 128, 256
    fc1_logical = torch.zeros(
        (2 * intermediate, hidden // 32), dtype=torch.uint8, device="cuda"
    )
    # Raw E8M0 0x7f encodes 1.0.  Row 32 is deliberately non-contiguous in
    # the atom byte order, so this coordinate exercises more than a reshape.
    fc1_logical[32, 0] = 0x7F
    fc2_logical = torch.zeros(
        (hidden, intermediate // 32), dtype=torch.uint8, device="cuda"
    )
    fc1 = to_atom_swizzled(fc1_logical).unsqueeze(0)
    fc2 = to_atom_swizzled(fc2_logical).unsqueeze(0)
    adapter = SwizzledWeightScaleViews(
        fc1, fc2, hidden_size=hidden, intermediate_size=intermediate
    )
    arguments = adapter.to_cute()
    output = torch.zeros(1, dtype=torch.float32, device="cuda")
    output_cute = from_dlpack(output, assumed_align=4).mark_layout_dynamic()
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    class _Harness:
        @cute.jit
        def __call__(self, fc1_flat, fc2_flat, destination, stream):
            views = adapter.make_logical_views(fc1_flat, fc2_flat)
            self._kernel(views.fc1_weight_sfa_logical, destination).launch(
                grid=(1, 1, 1), block=(32, 1, 1), stream=stream
            )

        @cute.kernel
        def _kernel(self, source, destination):
            if cute.arch.thread_idx()[0] == 0:
                destination[0] = cutlass.Float32(source[32, 0, 0])

    compiled = cute.compile(
        _Harness(),
        arguments.fc1_weight_scales,
        arguments.fc2_weight_scales,
        output_cute,
        stream,
    )
    compiled(
        arguments.fc1_weight_scales,
        arguments.fc2_weight_scales,
        output_cute,
        stream,
    )
    torch.cuda.synchronize()
    assert output.item() == 1.0
