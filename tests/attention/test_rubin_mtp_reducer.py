# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Numerical contract of the Rubin MTP scalar and vector reducer candidates."""

import math

import pytest
import torch

from flashinfer.cute_dsl import is_cute_dsl_available


@pytest.mark.parametrize(
    "splits,capacity", [(3, 4), (6, 8), (13, 16), (26, 32), (32, 32), (65, 256)]
)
@pytest.mark.parametrize("d_tiles", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.bfloat16])
def test_mtp_reducer_empty_rows_and_live_graph(splits, capacity, d_tiles, dtype):
    if not torch.cuda.is_available() or not is_cute_dsl_available():
        pytest.skip("requires CUDA and CuTe DSL")
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("reducer validation runs on SM100, SM103 or SM107")
    pytest.importorskip("cutlass.utils.rubin_helpers")

    import cutlass
    import cutlass.cute as cute
    import tvm_ffi
    from cutlass import Float32, Int32
    from cutlass.cute.runtime import from_dlpack

    from flashinfer.cute_dsl.attention.monolithic.mla_reducer import MLAReducer
    from flashinfer.cute_dsl.attention.rubin_mtp.kernel import (
        RubinMultiHeadLatentAttentionForwardFP8TwoPlusTwo,
    )

    class StandaloneReducer:
        def __init__(self):
            self.kernel = RubinMultiHeadLatentAttentionForwardFP8TwoPlusTwo(
                Float32,
                Float32,
                (256, 128),
                (256, 256),
                1,
                64,
                0.0,
                False,
                True,
                False,
                reducer_max_splits=capacity,
            )
            self.vector = d_tiles != 0
            if self.vector:
                self.kernel.reducer = MLAReducer(
                    acc_dtype=Float32,
                    lse_dtype=Float32,
                    qk_tile_shape=(256, 128),
                    num_heads=256,
                    seq_len_q=2,
                    max_splits=capacity,
                    d_tiles=d_tiles,
                    empty_lse_negative_inf=True,
                    max_prefetch_splits=8,
                )

        @cute.jit
        def __call__(
            self, out, lse, partials, partial_lse, split, lengths, scale, stream
        ):
            if cutlass.const_expr(self.vector):
                self.kernel.reducer(
                    out,
                    lse,
                    partials,
                    partial_lse,
                    split,
                    lengths,
                    None,
                    None,
                    scale,
                    stream,
                )
            else:
                self.kernel.o_dtype = out.element_type
                self.kernel.reduction_kernel(
                    out,
                    lse,
                    partials,
                    partial_lse,
                    split,
                    lengths,
                    None,
                    scale,
                ).launch(
                    grid=(out.shape[0], out.shape[2], out.shape[3]),
                    block=(128, 1, 1),
                    smem=self.kernel.reducer_max_splits * 4,
                    stream=stream,
                )

    torch.manual_seed(421 + splits)
    batch, query_tiles, rows, dim = 4, 2, 256, 512
    host_lengths = [0, 1, 129, 8191]
    lengths = torch.tensor(host_lengths, dtype=torch.int32, device="cuda")
    partials = torch.randn((batch, query_tiles, rows, splits, dim), device="cuda")
    partial_lse = torch.randn((batch, query_tiles, rows, splits), device="cuda") * 20

    def active_count(length):
        tiles = (length + 127) // 128
        per_split = max(1, (tiles + splits - 1) // splits)
        return (tiles + per_split - 1) // per_split

    for b, length in enumerate(host_lengths):
        active = active_count(length)
        # Invalid trailing splits must not be read, even by vector prefetches.
        partials[b, ..., active:, :] = math.nan
        partial_lse[b, ..., active:] = math.nan
        partials[b, :, ::17, :active] = 0
        partial_lse[b, :, ::17, :active] = -math.inf
        if active > 1:
            partials[b, :, ::7, 0] = 0
            partial_lse[b, :, ::7, 0] = -math.inf

    def reference():
        out = torch.zeros((batch, query_tiles, rows, dim), device="cuda")
        lse = torch.full((batch, query_tiles, rows), -math.inf, device="cuda")
        for b, length in enumerate(host_lengths):
            count = active_count(length)
            if count:
                scores = partial_lse[b, ..., :count].double() * math.log(2)
                weights = torch.nan_to_num(scores.softmax(-1))
                out[b] = torch.einsum(
                    "qhs,qhsd->qhd", weights, partials[b, ..., :count, :].double()
                ).float()
                lse[b] = (scores.logsumexp(-1) / math.log(2)).float()
        return out, lse

    expected, expected_lse = reference()
    out = torch.empty_like(expected, dtype=dtype)
    lse = torch.empty_like(expected_lse)
    tensors = (
        out.permute(2, 3, 1, 0),
        lse.permute(2, 1, 0),
        partials.permute(2, 3, 4, 1, 0),
        partial_lse.permute(2, 3, 1, 0),
    )
    cc = torch.cuda.get_device_capability()
    compiled = cute.compile(
        StandaloneReducer(),
        *(from_dlpack(t, assumed_align=16) for t in tensors),
        Int32(splits),
        from_dlpack(lengths, assumed_align=4),
        Float32(1),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options=f"--enable-tvm-ffi --gpu-arch=sm_{cc[0]}{cc[1]}a",
    )

    def launch(scale):
        with tvm_ffi.use_torch_stream():
            compiled(*tensors, Int32(splits), lengths, Float32(scale))

    def check(scale):
        torch.cuda.synchronize()
        torch.testing.assert_close(
            out.float(), expected.to(dtype).float(), atol=0.002, rtol=0.125
        )
        torch.testing.assert_close(lse, expected_lse * scale, atol=4e-5, rtol=4e-6)

    for scale in (1.0, math.log(2)):
        launch(scale)
        check(scale)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(1.0)
    # Replay reads live lengths and fresh partial values, not stale eager results.
    host_lengths = [0, 0, 1, 129]
    lengths.copy_(torch.tensor(host_lengths, dtype=torch.int32, device="cuda"))
    partials.mul_(2)
    partial_lse.add_(1)
    expected, expected_lse = reference()
    out.fill_(math.nan)
    lse.fill_(math.nan)
    graph.replay()
    check(1.0)
