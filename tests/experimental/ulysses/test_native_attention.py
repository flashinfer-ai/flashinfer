# SPDX-License-Identifier: Apache-2.0
"""Example integration checks against actual in-tree attention kernels."""

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from types import SimpleNamespace

from benchmarks.comm.ulysses_native_attention import (
    BACKENDS,
    NativeAttention,
    validate_backend,
)


@pytest.mark.parametrize("backend,cc", list(BACKENDS.items()))
def test_native_backend_admission(backend, cc):
    validate_backend(backend, cc, 256, 193, 1, 128, 1 / 32)
    with pytest.raises(ValueError):
        validate_backend(backend, (0, 0), 256, 193, 1, 128, 1 / 32)
    with pytest.raises(ValueError):
        validate_backend(backend, cc, 256, 257, 1, 128, 1 / 32)
    with pytest.raises(ValueError):
        validate_backend(backend, cc, 256, 193, 1, 128, float("nan"))


@pytest.mark.parametrize("backend,cc", list(BACKENDS.items()))
def test_native_kernel_prefix_stream_and_reuse(backend, cc):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != cc:
        pytest.skip(f"requires SM{cc}")
    torch.manual_seed(903)
    sequence, used, heads = 256, 193, 4
    op = NativeAttention(backend, sequence=sequence, used=used, heads=heads)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    pointer = op.output.data_ptr()
    with torch.cuda.stream(stream):
        for _ in range(3):
            qkv = [
                torch.randn(
                    1, sequence, heads, 128, device="cuda", dtype=torch.bfloat16
                )
                for _ in range(3)
            ]
            reference = F.scaled_dot_product_attention(
                *(t[:, :used].transpose(1, 2).float() for t in qkv)
            ).transpose(1, 2)
            actual = op(*qkv).clone()
            tolerance = 0.01 if backend.endswith("bf16") else 0.06
            torch.testing.assert_close(
                actual[:, :used].float(), reference, atol=tolerance, rtol=tolerance
            )
            assert torch.count_nonzero(actual[:, used:]).item() == 0
            for t in qkv:
                t[:, used:].fill_(1000)
            torch.testing.assert_close(op(*qkv), actual, atol=0, rtol=0)
    torch.cuda.current_stream().wait_stream(stream)
    assert pointer == op.output.data_ptr()


@pytest.mark.parametrize("backend,cc", list(BACKENDS.items()))
def test_native_chunk_equivalence(backend, cc):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != cc:
        pytest.skip(f"requires SM{cc}")
    torch.manual_seed(904)
    shape = (1, 128, 3, 128)
    qkv = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    whole = NativeAttention(backend, sequence=128, used=128, heads=3)
    expected = whole(*qkv).clone()
    outputs = []
    offset = 0
    for count in (1, 2):
        chunk = NativeAttention(backend, sequence=128, used=128, heads=count)
        outputs.append(chunk(*(t[:, :, offset : offset + count] for t in qkv)).clone())
        offset += count
    torch.testing.assert_close(torch.cat(outputs, 2), expected, atol=0.002, rtol=0.01)


@pytest.mark.parametrize("backend,cc", list(BACKENDS.items()))
def test_single_rank_pipeline_reuse(backend, cc, tmp_path):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != cc:
        pytest.skip(f"requires SM{cc}")
    if dist.is_initialized():
        pytest.skip("standalone one-rank test owns its process group")
    from benchmarks.comm.bench_ulysses_native_attention import NativePipeline
    from flashinfer.comm import UlyssesCommunicator

    dist.init_process_group(
        "nccl", init_method=(tmp_path / "init").as_uri(), rank=0, world_size=1
    )
    gin, gout = dist.new_group(), dist.new_group()
    cin = UlyssesCommunicator(
        group=gin, backend="nccl", max_elems=3 * 128 * 3 * 128, dtype=torch.bfloat16
    )
    cout = UlyssesCommunicator(
        group=gout, backend="nccl", max_elems=128 * 3 * 128, dtype=torch.bfloat16
    )
    try:
        args = SimpleNamespace(
            attention=backend,
            sequence=128,
            used=99,
            heads=3,
            schedule="1,2",
            fp8_scale=1 / 32,
        )
        pipeline = NativePipeline(args, cin, cout)
        for _ in range(3):
            qkv = [
                torch.randn(1, 128, 3, 128, device="cuda", dtype=torch.bfloat16)
                for _ in range(3)
            ]
            expected = pipeline.run("ordinary", *qkv).clone()
            for mode in ("whole_fused", "chunk_serial", "chunk_overlap"):
                actual = pipeline.run(mode, *qkv)
                torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.01)
    finally:
        torch.cuda.synchronize()
        cin.close()
        cout.close()
        dist.destroy_process_group(gin)
        dist.destroy_process_group(gout)
        dist.destroy_process_group()
