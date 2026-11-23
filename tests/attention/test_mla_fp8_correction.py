# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

"""Check correction skipping against an independent causal attention reference."""

import math
from contextlib import nullcontext

import pytest
import torch


@pytest.mark.parametrize(
    "heads,q_len,splits,page", [(96, 4, 1, 64), (96, 8, 13, 128), (128, 2, 3, 64)]
)
@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize("threshold", [0.0, 8.0])
def test_fp8_correction_skip_live_graph(
    heads, q_len, splits, page, persistent, threshold
):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 7),
    ):
        pytest.skip("requires SM100 or SM107")
    pytest.importorskip("cutlass")
    import tvm_ffi
    from cutlass import Float32, Int32
    from flashinfer.cute_dsl.attention.monolithic import mla_decode as mono

    torch.manual_seed(6177 + q_len)
    batch, capacity = 2, 4097
    pages = (capacity + page - 1) // page
    q = torch.randn(batch, q_len, heads, 576, device="cuda").to(torch.float8_e4m3fn)
    kv = torch.randn(batch * pages, page, 576, device="cuda").to(q.dtype)
    table = (
        torch.randperm(batch * pages, device="cuda").to(torch.int32).view(batch, pages)
    )
    lengths = torch.tensor([q_len, capacity], dtype=torch.int32, device="cuda")
    out = torch.empty(batch, q_len, heads, 512, device="cuda", dtype=q.dtype)
    lse = torch.empty(batch, q_len, heads, device="cuda")
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    actual, nbytes = mono._get_split_kv_and_workspace_size(
        batch, q_len, heads, 512, sms, capacity, num_kv_splits=splits
    )
    assert actual == splits
    workspace = torch.empty(nbytes, dtype=torch.int8, device="cuda") if nbytes else None
    cc = torch.cuda.get_device_capability()
    run = mono._get_compiled_mla_kernel(
        f"sm_{cc[0]}{cc[1]}a",
        q.dtype,
        out.dtype,
        page,
        512,
        64,
        heads,
        q_len,
        persistent,
        True,
        False,
        False,
        reducer_max_splits=mono._get_reducer_max_splits(splits),
        skip_correction_threshold=threshold,
        is_workspace_size_zero=splits == 1,
    )
    scale, output_scale = 192**-0.5, 0.75
    lse_scale = 1.0 if persistent else math.log(2)

    def launch():
        with tvm_ffi.use_torch_stream():
            run(
                q[..., :512],
                q[..., 512:],
                kv[..., :512],
                kv[..., 512:],
                table,
                out,
                lse,
                workspace,
                Int32(splits),
                lengths,
                None,
                None,
                Int32(0),
                None,
                Float32(scale),
                Float32(output_scale),
                Float32(lse_scale),
            )

    def validate():
        previous = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            expected, expected_lse = [], []
            tokens = torch.arange(q_len, device="cuda").repeat_interleave(heads)
            for b in range(batch):
                length = int(lengths[b])
                cache = kv[table[b].long()].reshape(-1, 576)[:length].float()
                logits = q[b].reshape(-1, 576).float() @ cache.T * scale
                visible = (
                    torch.arange(length, device="cuda")[None, :]
                    < length - q_len + tokens[:, None] + 1
                )
                logits.masked_fill_(~visible, -math.inf)
                expected.append(
                    (logits.softmax(-1) @ cache[:, :512] * output_scale).reshape(
                        q_len, heads, 512
                    )
                )
                expected_lse.append(
                    (logits.logsumexp(-1) * (lse_scale / math.log(2))).reshape(
                        q_len, heads
                    )
                )
            ref, ref_lse = torch.stack(expected), torch.stack(expected_lse)
            assert torch.isfinite(out.float()).all()
            nrmse = ((out.float() - ref).square().mean() / ref.square().mean()).sqrt()
            assert nrmse.item() < 0.04
            torch.testing.assert_close(lse, ref_lse, rtol=0, atol=0.03)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous

    launch()
    validate()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    q.copy_((-q.float()).to(q.dtype))
    lengths[1] = 257
    table.copy_(table.flip(1))
    out.fill_(math.nan)
    lse.fill_(math.nan)
    graph.replay()
    torch.cuda.synchronize()
    validate()


@pytest.mark.parametrize(
    "threshold,valid",
    [
        (0.0, True),
        (8.0, True),
        (math.log2(448.0), True),
        (math.nextafter(math.log2(448.0), math.inf), False),
        (9.0, False),
        (16.0, False),
        (math.inf, False),
        (-math.inf, False),
        (math.nan, False),
    ],
)
def test_fp8_correction_threshold_bounds(threshold, valid):
    pytest.importorskip("cutlass")
    from cutlass import Float32
    from flashinfer.cute_dsl.attention.monolithic.mla_decode_fp8 import (
        BlackwellMultiHeadLatentAttentionForwardFP8,
    )

    # Constructor validation needs no CUDA device or kernel compilation.
    check = (
        nullcontext()
        if valid
        else pytest.raises(ValueError, match="skip_correction_threshold")
    )
    with check:
        BlackwellMultiHeadLatentAttentionForwardFP8(
            acc_dtype=Float32,
            lse_dtype=Float32,
            mma_qk_tiler_mn=(128, 128),
            mma_pv_tiler_mn=(128, 256),
            max_active_clusters=1,
            page_size=64,
            skip_correction_threshold=threshold,
            is_persistent=False,
            is_var_seq=True,
            is_var_split_kv=False,
            enable_pdl=False,
        )
