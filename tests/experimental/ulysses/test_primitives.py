# SPDX-License-Identifier: Apache-2.0
"""Single-device checks: destination simulation is NOT a multi-GPU test."""

import pytest
import torch

from flashinfer.comm.ulysses_experimental import (
    pack_ulysses_qkv_fp8,
    prepare_ulysses_k_mean,
    prepare_ulysses_producer,
)
from flashinfer.experimental.ulysses.producer import validate_schedule
from flashinfer.experimental.ulysses.sm100 import validate_geometry


@pytest.mark.parametrize("schedule", [(), (0, 7), (True, 6), (3, 3), (8,)])
def test_schedule_rejects_invalid(schedule):
    with pytest.raises(ValueError):
        validate_schedule(8, 56, 128, 32, schedule)


def test_schedule_uneven_small_heads():
    assert validate_schedule(8, 56, 128, 32, (3, 4)) == (3, 4)
    assert validate_schedule(4, 4, 128, 32, (1,)) == (1,)
    with pytest.raises(ValueError):
        validate_schedule(4, 2, 128, 32, (1,))


@pytest.mark.parametrize("world,physical", [(2, 37888), (4, 37888), (8, 38912)])
def test_sm100_historical_allowlist(world, physical):
    extent = validate_geometry(world, physical // world, 37807, True, True)
    assert extent == (37888 if world == 8 else None)
    with pytest.raises(ValueError):
        validate_geometry(world, physical // world, 37807, True, False)
    with pytest.raises(ValueError):
        validate_geometry(world, physical // world, 37807, False, True)


@pytest.mark.parametrize(
    "world,s,used",
    [(8, 4864, 37889), (4, 9456, 37807), (6, 6400, 37807), (2, 18944, True)],
)
def test_sm100_rejects_unsupported_geometry(world, s, used):
    with pytest.raises(ValueError):
        validate_geometry(world, s, used, True, True)


gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@gpu
@pytest.mark.parametrize(
    "world,heads,dim", [(1, 4, 32), (2, 8, 128), (4, 56, 128), (8, 56, 32)]
)
def test_fp8_pack_strided_and_current_stream(world, heads, dim):
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("test requires SM90 or newer")
    torch.manual_seed(113)
    raw = torch.randn(17, heads, 3, dim, device="cuda", dtype=torch.bfloat16)
    qkv = raw.unbind(2)
    scales = tuple(t.float().abs().amax((0, 2)).clamp_min(1e-12) / 448 for t in qkv)
    out = torch.empty(
        world, 17, heads // world, 3 * dim, device="cuda", dtype=torch.float8_e4m3fn
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            out.view(torch.uint8).fill_(0xFF)
            assert pack_ulysses_qkv_fp8(*qkv, scales, world_size=world, out=out) is out
    torch.cuda.current_stream().wait_stream(stream)
    refs = []
    for t, scale in zip(qkv, scales, strict=True):
        quant = (
            (t.float() / scale[None, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
        )
        refs.append(quant.view(17, world, heads // world, dim).permute(1, 0, 2, 3))
    expected = torch.cat(refs, -1).contiguous()
    assert torch.equal(out.view(torch.uint8), expected.view(torch.uint8))
    # Stable caller-owned output is graph-capturable after compilation.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        pack_ulysses_qkv_fp8(*qkv, scales, world_size=world, out=out)
    graph.replay()
    assert torch.equal(out.view(torch.uint8), expected.view(torch.uint8))


@gpu
def test_fp8_pack_rounding_regression():
    # Long-shape reproduction: approximate Triton division crossed E4M3
    # rounding midpoints which short random fixtures did not exercise.
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("test requires SM90 or newer")
    torch.manual_seed(173)
    rows, heads, dim, world = 18944, 56, 128, 2
    qkv = tuple(
        torch.randn(rows, heads, dim, device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    )
    scales = tuple(t.float().abs().amax((0, 2)).clamp_min(1e-12) / 448 for t in qkv)
    out = torch.empty(
        world, rows, heads // world, 3 * dim, device="cuda", dtype=torch.float8_e4m3fn
    )
    pack_ulysses_qkv_fp8(*qkv, scales, world_size=world, out=out)
    for plane, (value, scale) in enumerate(zip(qkv, scales, strict=True)):
        quant = (value.float() / scale[None, :, None]).clamp(-448, 448).to(out.dtype)
        expected = quant.view(rows, world, heads // world, dim).permute(1, 0, 2, 3)
        actual = out[..., plane * dim : (plane + 1) * dim]
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@gpu
def test_fp8_pack_invalid_layout_and_alias():
    q = torch.zeros(8, 8, 32, device="cuda", dtype=torch.bfloat16)
    scales = tuple(torch.ones(8, device="cuda") for _ in range(3))
    out = torch.empty(2, 8, 4, 96, device="cuda", dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError):
        pack_ulysses_qkv_fp8(q, q, q, scales, world_size=3, out=out)
    with pytest.raises(ValueError):
        pack_ulysses_qkv_fp8(
            q[..., ::2], q[..., ::2], q[..., ::2], scales, world_size=2, out=out
        )
    with pytest.raises(ValueError):
        pack_ulysses_qkv_fp8(
            q, q, q, tuple(s.to(torch.bfloat16) for s in scales), world_size=2, out=out
        )
    # A larger common allocation can expose differently typed overlapping views.
    backing = torch.empty(out.numel() + q.numel() * 2, device="cuda", dtype=torch.uint8)
    aliased_q = backing[: q.numel() * 2].view(torch.bfloat16).view_as(q)
    aliased_out = backing[: out.numel()].view(torch.float8_e4m3fn).view_as(out)
    with pytest.raises(ValueError, match="overlap"):
        pack_ulysses_qkv_fp8(aliased_q, q, q, scales, world_size=2, out=aliased_out)


@gpu
@pytest.mark.parametrize(
    "world,local_heads,schedule",
    [(2, 4, (2, 2)), (4, 7, (3, 4)), (8, 7, (7,)), (4, 1, (1,))],
)
def test_grouped_producer_spans_all_destinations(world, local_heads, schedule):
    heads, dim, hidden, rows = world * local_heads, 8, 48, 19
    weight = (
        torch.randn(3 * heads * dim, hidden, device="cuda", dtype=torch.bfloat16) * 0.1
    )
    producer = prepare_ulysses_producer(
        weight,
        world_size=world,
        heads=heads,
        head_dim=dim,
        local_seq=rows,
        schedule=schedule,
    )
    pointers = [t.data_ptr() for t in producer.raw]
    for _ in range(10):
        x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16) * 0.1
        whole = torch.nn.functional.linear(x, weight).view(
            rows, 3, world, local_heads, dim
        )
        offset = 0
        for i, count in enumerate(schedule):
            qkv = producer.produce(x, i)
            for plane, actual in enumerate(qkv):
                ref = whole[:, plane, :, offset : offset + count].reshape(
                    1, rows, world * count, dim
                )
                torch.testing.assert_close(actual, ref, atol=2e-3, rtol=1e-2)
            offset += count
    assert pointers == [t.data_ptr() for t in producer.raw]
    with pytest.raises(ValueError):
        producer.produce(x, -1)
    with (
        torch.cuda.stream(torch.cuda.Stream()),
        pytest.raises(RuntimeError, match="bound"),
    ):
        producer.produce(x, 0)


@gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_shape_stable_mean_reuse(dtype):
    op = prepare_ulysses_k_mean(
        sequence=513, heads=14, head_dim=128, device="cuda", dtype=dtype
    )
    # Same full-H tensor geometry is the numerical reference. Results must be
    # consumed before the next update, because mean storage is reused.
    for _ in range(10):
        whole = torch.randn(1, 513, 14, 128, device="cuda", dtype=dtype)
        reference = whole.mean(1, keepdim=True)
        offset = 0
        for count in (5, 5, 4):
            actual = op.update(whole[:, :, offset : offset + count], head_offset=offset)
            assert torch.equal(actual, reference[:, :, offset : offset + count])
            offset += count
    with pytest.raises(ValueError):
        op.update(whole[:, :, :5], head_offset=12)
    with pytest.raises(ValueError, match="alias"):
        op.update(op.scratch[:, :, :5], head_offset=0)
