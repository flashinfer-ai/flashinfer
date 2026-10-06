"""Unquantized row identity, frozen dynamic counts, and Int64 pool addressing."""
from contextlib import contextmanager

import subprocess
import sys
import textwrap

import pytest
import torch

from b12x.preparation import PreparationSession, PreparedCall


@contextmanager
def _prepared(weight, ids, out, *, num_rows=None):
    from b12x.sequence import embedding
    declaration = embedding.plan(
        embedding.query_from_call(weight, ids, out=out, num_rows=num_rows), device=weight.device,
    )
    with PreparationSession(device=weight.device, autotune=False, compile_workers=2) as session:
        session.prepare((declaration.request(
            name="embedding", prepare_call=lambda state: PreparedCall(
                run=lambda: state.run(weight, ids, out=out, num_rows=num_rows),
            ),
        ),))
        session.freeze()
        yield declaration


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
def test_exact_rows_preserve_shape_dtype_and_strided_table(dtype, id_dtype):
    from b12x.sequence import embedding

    # Odd width covers the final partial CTA; signed zero and FP32 fractions
    # expose accidental arithmetic or a BF16 intermediate in an FP32 gather.
    storage = torch.arange(19 * 134, device="cuda", dtype=torch.float32)
    weight = (storage.reshape(19, 134) * 0.00012345 - 0.7).to(dtype)[:, :129]
    weight[0, 0] = -0.0
    for shape, values in [((), [0]), ((2, 3), [18, 0, 7, 7, 1, 18]), ((0,), [])]:
        ids = torch.tensor(values, device="cuda", dtype=id_dtype).reshape(shape)
        out = torch.empty((*shape, 129), device="cuda", dtype=dtype)
        with _prepared(weight, ids, out) as plan:
            assert embedding.run(weight, ids, out=out, plan=plan) is out
        expected = weight[ids.long()]
        assert out.shape == expected.shape and out.dtype == dtype
        bits = torch.int16 if dtype == torch.bfloat16 else torch.int32
        torch.testing.assert_close(out.view(bits), expected.view(bits), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
def test_frozen_lookup_changes_counts_and_replays_mutated_ids(dtype, id_dtype):
    from b12x.sequence import embedding

    weight = torch.arange(23 * 129, device="cuda", dtype=torch.float32).reshape(23, 129).to(dtype)
    ids = torch.zeros(11, device="cuda", dtype=id_dtype)
    out = torch.full((11, 129), -7, device="cuda", dtype=dtype)
    count = torch.zeros((), device="cuda", dtype=torch.int32)
    torch.cuda.synchronize()
    with _prepared(weight, ids, out) as host_plan, _prepared(weight, ids, out, num_rows=count) as device_plan:
        for table_rows, rows in [(3, 1), (23, 11), (7, 0), (11, 5)]:
            ids.fill_(table_rows - 1)
            out.fill_(-7)
            embedding.run(weight[:table_rows], ids[:rows], out=out[:rows], plan=host_plan)
            torch.testing.assert_close(out[:rows], weight[ids[:rows].long()], rtol=0, atol=0)
            torch.testing.assert_close(out[rows:], torch.full_like(out[rows:], -7), rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = embedding.run(weight, ids, out=out, num_rows=count, plan=device_plan)
        address = out.data_ptr()
        for rows in (11, 2, 0, 7):
            # Invalid inactive IDs must not be read even after a larger batch.
            ids.fill_(-1)
            ids[:rows].copy_((torch.arange(rows, device="cuda") * 3 % 23).to(id_dtype))
            count.fill_(rows)
            out.fill_(-7)
            graph.replay()
            torch.cuda.synchronize()
            assert captured.data_ptr() == address
            torch.testing.assert_close(out[:rows], weight[ids[:rows].long()], rtol=0, atol=0)
            torch.testing.assert_close(out[rows:], torch.full_like(out[rows:], -7), rtol=0, atol=0)
        graph.reset()


@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
def test_high_row_offset_is_widened_before_multiplication(id_dtype):
    from b12x.sequence import embedding

    width = 128
    high_row = 2**31 // width + 3
    required = (high_row + 1) * width * 2
    if torch.cuda.mem_get_info()[0] < required + 512 * 1024**2:
        pytest.skip("requires just over 4 GiB free for the Int32-offset regression")
    weight = torch.empty((high_row + 1, width), device="cuda", dtype=torch.bfloat16)
    weight[0].fill_(-2)
    weight[high_row].copy_(torch.arange(width, device="cuda", dtype=torch.bfloat16))
    ids = torch.zeros(1, device="cuda", dtype=id_dtype)
    out = torch.empty((1, width), device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    with _prepared(weight, ids, out) as plan:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            embedding.run(weight, ids, out=out, plan=plan)
        ids.fill_(high_row)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out[0], weight[high_row], rtol=0, atol=0)
        graph.reset()


@pytest.mark.parametrize("bad_id", [-1, 3, 2**32])
def test_invalid_ids_raise_device_error_instead_of_zero_or_oob(bad_id):
    # trap poisons the CUDA context, so each invalid replay owns a subprocess.
    code = textwrap.dedent(f"""
        import torch
        from b12x.sequence import embedding
        weight = torch.ones((3, 129), device='cuda', dtype=torch.float32)
        ids = torch.zeros(1, device='cuda', dtype=torch.int64)
        out = torch.empty((1, 129), device='cuda', dtype=weight.dtype)
        from b12x.preparation import PreparationSession, PreparedCall
        plan = embedding.plan(embedding.query_from_call(weight, ids, out=out), device=weight.device)
        session = PreparationSession(device=weight.device, autotune=False, compile_workers=2)
        session.prepare((plan.request(name='invalid-id', prepare_call=lambda state:
            PreparedCall(run=lambda: state.run(weight, ids, out=out))),))
        session.freeze()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            embedding.run(weight, ids, out=out, plan=plan)
        ids.fill_({bad_id})
        try:
            graph.replay()
            torch.cuda.synchronize()
        except RuntimeError:
            pass
        else:
            raise AssertionError('invalid embedding ID did not raise a CUDA error')
    """)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
def test_q8_blocks_decode_exactly_with_frozen_counts_and_graph_replay(id_dtype):
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.sequence import embedding
    from b12x.testing.q8_0_reference import dequantize_blocks

    raw = torch.arange(19 * 3 * 34, dtype=torch.int32).to(torch.uint8).reshape(19, 3, 34)
    bases = torch.tensor([0., -0., 2**-24, -2**-14, 0.125, -0.25, 65504.], dtype=torch.float16)
    raw[..., :2] = bases[torch.arange(19 * 3).reshape(19, 3) % len(bases)][..., None].view(torch.uint8)
    expected = dequantize_blocks(raw).bfloat16().cuda()
    weight = raw.cuda()
    ids = torch.zeros(11, device="cuda", dtype=id_dtype)
    out = torch.empty((11, 96), device="cuda", dtype=torch.bfloat16)
    with _prepared(weight, ids, out) as plan, kernel_resolution_guard("Q8 embedding"):
        for rows in (1, 11, 3):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                embedding.run(weight, ids[:rows], out=out[:rows], plan=plan)
            ids[:rows].copy_((torch.arange(rows, device="cuda") * 3 % 19).to(id_dtype))
            out.fill_(float("nan"))
            allocated = torch.cuda.memory_allocated()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == allocated
            torch.testing.assert_close(out[:rows].view(torch.int16), expected[ids[:rows].long()].view(torch.int16), rtol=0, atol=0)
            assert torch.isnan(out[rows:]).all()


@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
def test_q8_high_row_uses_64_bit_byte_addressing(id_dtype):
    from b12x.sequence import embedding
    from b12x.testing.q8_0_reference import dequantize_blocks

    width, stride = 128, 136
    high_row = 2**31 // stride + 3
    required = (high_row + 1) * stride
    if torch.cuda.mem_get_info()[0] < required + 512 * 1024**2:
        pytest.skip("requires just over 2 GiB for high Q8 row offsets")
    weight = torch.empty((high_row + 1, width // 32, 34), device="cuda", dtype=torch.uint8)
    tail = torch.arange(width // 32 * 34, dtype=torch.int32).to(torch.uint8).reshape(width // 32, 34)
    tail[:, :2] = torch.tensor([0.125], dtype=torch.float16).view(torch.uint8)
    weight[high_row].copy_(tail)
    ids = torch.full((1,), high_row, device="cuda", dtype=id_dtype)
    out = torch.empty((1, width), device="cuda", dtype=torch.bfloat16)
    with _prepared(weight, ids, out) as plan:
        embedding.run(weight, ids, out=out, plan=plan)
    torch.testing.assert_close(out[0], dequantize_blocks(tail).bfloat16().cuda(), rtol=0, atol=0)
