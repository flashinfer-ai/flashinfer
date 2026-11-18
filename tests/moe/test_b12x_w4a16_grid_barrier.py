# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the production barrier with stores from every warp and CTA."""

import pytest
import torch

from tests.moe.test_b12x_fused_moe import cute_dsl_available


def _compile_barrier_probe(arch, blocks, threads, rounds):
    import cutlass
    from cutlass import cute
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_kernel import (
        W4A16FusedMoeKernel,
    )

    # The production method needs only these two scratch offsets. Avoid
    # constructing unrelated architecture-specific GEMM machinery in this test.
    barrier = object.__new__(W4A16FusedMoeKernel)
    barrier.barrier_count_off = 0
    barrier.barrier_sense_off = 1

    @cute.kernel
    def kernel(data, observed, locks):
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        index = cta * threads + tid
        peer = ((cta + 1) % blocks) * threads + tid
        for epoch in range(rounds):
            data[index] = epoch * blocks * threads + index + 1
            barrier._grid_barrier(locks, tid, cutlass.Int32(blocks))
            observed[epoch * blocks * threads + index] = data[peer]
            # Every reader must finish before the next generation overwrites data.
            barrier._grid_barrier(locks, tid, cutlass.Int32(blocks))

    @cute.jit
    def launch(data, observed, locks, stream):
        kernel(data, observed, locks).launch(
            grid=(blocks, 1, 1),
            block=(threads, 1, 1),
            cooperative=True,
            stream=stream,
        )

    tensors = [
        cute.runtime.make_fake_tensor(cutlass.Int32, (n,), (1,), assumed_align=16)
        for n in (blocks * threads, rounds * blocks * threads, 2)
    ]
    return cute.compile(
        launch,
        *tensors,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options=f"--gpu-arch {arch} --enable-tvm-ffi --keep-ptx",
    )


@cute_dsl_available
def test_grid_barrier_publishes_before_first_arrival():
    compiled = _compile_barrier_probe("sm_120a", 2, 128, 2)
    # Inspect generated code, not Python source: a probabilistic stress test
    # cannot establish release ordering. The old barrier first fenced *after*
    # incrementing the arrival counter, and fails this check.
    ptx = compiled.__ptx__
    assert ptx is not None
    arrival = ptx.index("atom.global.add.s32")
    assert "membar.gl" in ptx[:arrival]


@cute_dsl_available
@pytest.mark.parametrize("blocks,threads", [(2, 128), (4, 256)])
def test_grid_barrier_cross_cta_reuse_and_live_graph(blocks, threads):
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    major, minor = torch.cuda.get_device_capability()
    if major < 8:
        pytest.skip("requires SM80 or newer")
    if torch.cuda.get_device_properties(0).multi_processor_count < blocks:
        pytest.skip("requires one resident CTA per SM")
    arch = f"sm_{major}{minor}" + ("a" if major >= 9 else "")
    rounds = 64
    compiled = _compile_barrier_probe(arch, blocks, threads, rounds)
    size = blocks * threads
    data = torch.empty(size, device="cuda", dtype=torch.int32)
    observed = torch.empty(rounds * size, device="cuda", dtype=torch.int32)
    locks = torch.zeros(2, device="cuda", dtype=torch.int32)
    indices = torch.arange(size, device="cuda", dtype=torch.int32)
    peers = ((indices // threads + 1) % blocks) * threads + indices % threads
    expected = (
        torch.arange(rounds, device="cuda", dtype=torch.int32)[:, None] * size
        + peers[None, :]
        + 1
    ).flatten()

    def check():
        torch.testing.assert_close(observed, expected, atol=0, rtol=0)
        assert locks[0].item() == 0

    # Reuse the epoch/count scratch without resetting it between invocations.
    for _ in range(3):
        observed.fill_(-1)
        compiled(data, observed, locks)
        check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled(data, observed, locks)
    for _ in range(8):
        observed.fill_(-1)
        graph.replay()
        check()
