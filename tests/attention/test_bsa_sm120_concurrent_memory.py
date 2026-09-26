# SPDX-License-Identifier: Apache-2.0
"""Regression for K/V stage reuse across the generic and TMA async proxies."""

import pytest
import torch


@pytest.mark.parametrize("backend", ["bf16", "fp16", "sage"])
def test_bsa_sm120_concurrent_copy_repeatability(backend):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")

    from flashinfer.cute_dsl.sparse.bsa_attn_sm120 import (
        bsa_attn_sm120_blk64_fwd,
        bsa_attn_sm120_blk64_sage_fwd,
    )
    from flashinfer.cute_dsl.sparse.bsa_utils.sage_quant_sm120 import (
        quantize_sage_qkv_sm120,
    )

    # A long KV loop is intentional: short single-stream tests did not expose
    # the stage-reuse race. This needs neither NCCL nor a multi-GPU machine.
    torch.manual_seed(772)
    sequence, heads, dim = 37807, 1, 128
    dtype = torch.float16 if backend == "fp16" else torch.bfloat16
    qkv = [
        torch.randn(1, sequence, heads, dim, device="cuda", dtype=dtype)
        for _ in range(3)
    ]
    blocks = (sequence + 63) // 64
    indices = (
        torch.arange(blocks, device="cuda", dtype=torch.int32)
        .view(1, 1, 1, blocks)
        .expand(1, heads, blocks, blocks)
        .contiguous()
    )
    sizes = torch.full((blocks,), 64, device="cuda", dtype=torch.int32)
    sizes[-1] = sequence - (blocks - 1) * 64
    if backend == "sage":
        quantized = quantize_sage_qkv_sm120(
            *(x.transpose(1, 2).contiguous() for x in qkv)
        )

        def run():
            return bsa_attn_sm120_blk64_sage_fwd(
                *quantized, indices, blocks, block_sizes=sizes, backend="cute_dsl"
            )

    else:

        def run():
            return bsa_attn_sm120_blk64_fwd(*qkv, indices, blocks, block_sizes=sizes)[0]

    expected = run().clone()
    caller = torch.cuda.current_stream()
    traffic = torch.cuda.Stream()
    source = torch.ones(32 << 20, dtype=torch.bfloat16, device="cuda")
    destination = torch.empty_like(source)
    torch.cuda.synchronize()  # includes preparation/JIT, outside the test overlap
    traffic.wait_stream(caller)
    try:
        for _ in range(20):
            with torch.cuda.stream(traffic):
                for _ in range(100):
                    destination.copy_(source)
            actual = run()
            caller.wait_stream(traffic)
            # Compare the same native kernel, not a lossy Sage-vs-BF16 oracle.
            # Unrelated memory traffic must not change even one output bit.
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    finally:
        # Keep both traffic buffers alive even if the assertion fails.
        traffic.synchronize()
