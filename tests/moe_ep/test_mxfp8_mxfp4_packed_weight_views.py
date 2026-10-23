# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Executable B200 check for pointer-backed packed MXFP4 weight views."""

import pytest


@pytest.mark.skipif(
    pytest.importorskip("torch").cuda.is_available() is False,
    reason="requires CUDA",
)
def test_b200_reads_fc1_fc2_nibbles_through_explicit_logical_layout() -> None:
    import torch

    pytest.importorskip("cutlass")
    from flashinfer.utils import is_sm100a_supported

    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("requires SM100/SM103 and CUDA 12.8+")
    if not hasattr(torch, "float4_e2m1fn_x2"):
        pytest.skip("requires torch.float4_e2m1fn_x2")

    cutlass = pytest.importorskip("cutlass")
    import cutlass.cute as cute
    from cuda.bindings import driver as cuda
    from cutlass.cute.runtime import from_dlpack

    from flashinfer.moe_ep.backends.mega.kernel.sm100.mxfp8_mxfp4_bf16_cutedsl.weights import (
        _interleave_gate_up_8,
    )

    from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.packed_weight_views import (
        PackedWeightViews,
    )

    experts, hidden, intermediate = 2, 128, 128

    # Physical preprocessing ABI before the transpose is (E,M,K/2).  Use a
    # constant E2M1 code per row/expert so each sampled logical coordinate has
    # an unambiguous expected value.  FC1 rows 0..7 and 8..15 model the first
    # gate8/up8 pair produced by preprocessing.
    canonical_fc1_emk = torch.zeros(
        (experts, 2 * intermediate, hidden // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    canonical_fc1_emk[0, 0].fill_(0x22)  # gate row 0: code 2 -> 1.0
    canonical_fc1_emk[0, intermediate].fill_(0x44)  # up row 0: code 4 -> 2.0
    canonical_fc1_emk[1, intermediate].fill_(0x66)  # expert 1: code 6 -> 4.0
    fc1_emk = _interleave_gate_up_8(canonical_fc1_emk, intermediate_size=intermediate)
    fc2_emk = torch.zeros(
        (experts, hidden, intermediate // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    fc2_emk[1, 7].fill_(0x55)  # E2M1 code 5 -> 3.0

    fc1 = fc1_emk.view(torch.float4_e2m1fn_x2).transpose(1, 2)
    fc2 = fc2_emk.view(torch.float4_e2m1fn_x2).transpose(1, 2)
    adapter = PackedWeightViews(
        fc1,
        fc2,
        hidden_size=hidden,
        intermediate_size=intermediate,
    )
    arguments = adapter.to_cute()
    output = torch.zeros(4, dtype=torch.float32, device="cuda")
    output_cute = from_dlpack(output, assumed_align=16).mark_layout_dynamic()
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    class _Harness:
        @cute.jit
        def __call__(self, fc1_ptr, fc2_ptr, destination, stream):
            views = adapter.make_logical_views(fc1_ptr, fc2_ptr)
            self._kernel(
                views.fc1_weight_a_logical,
                views.fc2_weight_a_logical,
                destination,
            ).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

        @cute.kernel
        def _kernel(self, fc1_logical, fc2_logical, destination):
            if cute.arch.thread_idx()[0] == 0:
                # CuTe forbids scalar dereference of a sub-byte tensor.  A
                # Uint8 recast pairs logical K=(2j,2j+1), so reading these
                # bytes still executes and verifies the nibble layout.  The
                # row and expert checks prove gate/up-M and L strides.
                fc1_bytes = cute.recast_tensor(fc1_logical, cutlass.Uint8)
                fc2_bytes = cute.recast_tensor(fc2_logical, cutlass.Uint8)
                destination[0] = cutlass.Float32(fc1_bytes[0, 0, 0])
                destination[1] = cutlass.Float32(fc1_bytes[8, 0, 0])
                destination[2] = cutlass.Float32(fc1_bytes[8, 0, 1])
                destination[3] = cutlass.Float32(fc2_bytes[7, 0, 1])

    compiled = cute.compile(
        _Harness(),
        arguments.fc1_weight,
        arguments.fc2_weight,
        output_cute,
        stream,
    )
    compiled(
        arguments.fc1_weight,
        arguments.fc2_weight,
        output_cute,
        stream,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(
        output.cpu(), torch.tensor([0x22, 0x44, 0x66, 0x55], dtype=torch.float32)
    )
