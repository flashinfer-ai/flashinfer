# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Exact BF16 input staging on one Blackwell GPU.

The unchanged Torch stager is the output oracle. Special values and ID
extrema exercise the copy contract without launching the distributed MegaMoE.
"""

from unittest.mock import Mock

import pytest
import torch

from flashinfer.moe_ep.backends.mega.kernel.sm100.bf16_bf16_bf16_cutedsl.staging import (
    stage_mega_moe_inputs as torch_stage,
)
from flashinfer.moe_ep.backends.mega.kernel.sm100.bf16_nvfp4_bf16_cutedsl import (
    staging,
)

HIDDEN, TOPK = 288, 8


@pytest.fixture(autouse=True)
def blackwell():
    if not torch.cuda.is_available():
        pytest.skip("requires one CUDA GPU")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100 or SM103")


def _pattern(shape, values, dtype, offset):
    choices = torch.tensor(values, dtype=dtype, device="cuda")
    positions = torch.arange(shape[0] * shape[1], device="cuda")
    return choices[(positions + offset) % len(values)].reshape(shape)


def _batch(n, id_dtype, *, topk=TOPK, offset=0, hidden_size=HIDDEN):
    hidden = _pattern(
        (n, hidden_size),
        [0, -32768, 32640, -128, 32705, -63, 16256, -16512, 1],
        torch.int16,
        offset,
    ).view(torch.bfloat16)
    weights = _pattern(
        (n, topk),
        [
            0,
            -2147483648,
            1065353216,
            -1082130432,
            2139095040,
            -8388608,
            2143363909,
            -4119739,
            1,
        ],
        torch.int32,
        offset,
    ).view(torch.float32)
    ids = [-2147483648, -1, 0, 1, 31, 2147483647]
    if id_dtype == torch.int64:
        ids += [2**40, -(2**40), 2**63 - 1, -(2**63)]
    return hidden, weights, _pattern((n, topk), ids, id_dtype, offset)


def _outputs(capacity, *, topk=TOPK, hidden_size=HIDDEN):
    return (
        torch.full((capacity, hidden_size), -2.0, dtype=torch.bfloat16, device="cuda"),
        torch.full((capacity, topk), 99, dtype=torch.int64, device="cuda"),
        torch.full((capacity, topk), -3.0, dtype=torch.float32, device="cuda"),
    )


def _exact(got, reference, n, capacity):
    # Flatten also handles contiguous singleton columns with non-unit stride.
    for actual, expected in zip(got, reference, strict=False):
        assert torch.equal(
            actual.flatten().view(torch.uint8), expected.flatten().view(torch.uint8)
        )
    assert torch.all(got[1][n:capacity] == -1)


def _key(id_dtype, *, hidden_size=HIDDEN):
    return torch.cuda.current_device(), hidden_size, TOPK, id_dtype


@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
def test_exact_bits_h7168_vector_loop_and_tail(monkeypatch, id_dtype):
    # 7168 BF16 values / eight per vector / 128 threads = seven iterations.
    hidden_size, n, capacity = 7168, 1, 8
    fallback = Mock(wraps=torch_stage)
    monkeypatch.setattr(staging, "_torch_stage_mega_moe_inputs", fallback)
    batch = _batch(n, id_dtype, hidden_size=hidden_size)
    got = _outputs(capacity, hidden_size=hidden_size)
    reference = _outputs(capacity, hidden_size=hidden_size)
    compiled = None
    for offset in (0, 3, 7):
        for target, fresh in zip(
            batch,
            _batch(n, id_dtype, offset=offset, hidden_size=hidden_size),
            strict=False,
        ):
            target.copy_(fresh)
        staging.stage_mega_moe_inputs(*batch, *got)
        torch_stage(*batch, *reference)
        # Covers every vector and the untouched x/weight tails, not only IDs.
        _exact(got, reference, n, capacity)
        compiled_now = staging._STAGERS[
            _key(id_dtype, hidden_size=hidden_size)
        ].compiled
        if compiled is None:
            compiled = compiled_now
        assert compiled_now is compiled
    assert fallback.call_count == 0


@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
def test_exact_bits_dynamic_n_capacity_and_empty_refill(monkeypatch, id_dtype):
    fallback = Mock(wraps=torch_stage)
    monkeypatch.setattr(staging, "_torch_stage_mega_moe_inputs", fallback)
    got, reference = _outputs(128), _outputs(128)
    storage = _batch(128, id_dtype)
    compiled = None
    for step, (n, capacity) in enumerate(
        [
            (1, 64),
            (7, 64),
            (64, 64),
            (7, 64),
            (7, 128),
            (7, 64),
            (64, 64),
            (0, 64),
            (1, 64),
            (63, 64),
            (64, 64),
            (64, 64),
        ]
    ):
        # The same pointers change contents; the repeated final shape hits the
        # launch-argument cache. The n/capacity changes must reuse compiled code.
        for target, fresh in zip(
            storage, _batch(128, id_dtype, offset=step), strict=False
        ):
            target.copy_(fresh)
        batch = tuple(t[:n] for t in storage)
        actual = tuple(t[:capacity] for t in got)
        expected = tuple(t[:capacity] for t in reference)
        before = fallback.call_count
        staging.stage_mega_moe_inputs(*batch, *actual)
        torch_stage(*batch, *expected)
        assert fallback.call_count == before + (n == 0)
        compiled_now = staging._STAGERS[_key(id_dtype)].compiled
        if compiled is None:
            compiled = compiled_now
        assert compiled_now is compiled
        # Check the whole backing buffer, including outside the current view.
        _exact(got, reference, n, capacity)


@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
def test_singleton_inner_stride_preserves_torch_fallback(monkeypatch, id_dtype):
    fallback = Mock(wraps=torch_stage)
    monkeypatch.setattr(staging, "_torch_stage_mega_moe_inputs", fallback)
    # Independently exercise each routing input/output gate with a legal
    # contiguous (N,1) view whose inner stride is four, not one.
    for argument in (1, 2, 4, 5):
        batch = _batch(7, id_dtype, topk=1)
        got, reference = _outputs(64, topk=1), _outputs(64, topk=1)
        args = list(batch + got)
        tensor = args[argument]
        args[argument] = tensor.as_strided(tensor.shape, (1, 4))
        assert args[argument].is_contiguous() and args[argument].stride(1) == 4
        before = fallback.call_count
        staging.stage_mega_moe_inputs(*args)
        torch_stage(*batch, *reference)
        assert fallback.call_count == before + 1
        _exact(got, reference, 7, 64)


@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
def test_two_graphs_keep_their_shapes_pointers_and_full_tail(monkeypatch, id_dtype):
    fallback = Mock(wraps=torch_stage)
    monkeypatch.setattr(staging, "_torch_stage_mega_moe_inputs", fallback)
    batches = [_batch(7, id_dtype), _batch(64, id_dtype, offset=3)]
    capacities = [64, 128]
    got, reference = _outputs(128), _outputs(128)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for batch, capacity in zip(batches, capacities, strict=False):
            staging.stage_mega_moe_inputs(*batch, *(t[:capacity] for t in got))
    torch.cuda.current_stream().wait_stream(stream)
    compiled = staging._STAGERS[_key(id_dtype)].compiled
    graphs = [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]
    for graph, batch, capacity in zip(graphs, batches, capacities, strict=False):
        with torch.cuda.graph(graph, stream=stream):
            staging.stage_mega_moe_inputs(*batch, *(t[:capacity] for t in got))
    torch.cuda.synchronize()
    for actual, expected in zip(got, reference, strict=False):
        actual.copy_(expected)
    for step, which in enumerate([0, 1, 0, 1, 0]):
        batch, capacity = batches[which], capacities[which]
        for target, fresh in zip(
            batch, _batch(len(batch[0]), id_dtype, offset=step + 7), strict=False
        ):
            target.copy_(fresh)
        graphs[which].replay()
        torch_stage(*batch, *(t[:capacity] for t in reference))
        _exact(got, reference, len(batch[0]), capacity)
        assert staging._STAGERS[_key(id_dtype)].compiled is compiled
    assert fallback.call_count == 0


def test_default_int64_warmup_allows_first_int32_capture(monkeypatch):
    # Stager-level reproduction of MegaLayer.warmup()'s default dummy inputs;
    # no Mega layer, NVSHMEM or distributed warmup is executed by this test.
    monkeypatch.setattr(staging, "_STAGERS", {})
    fallback = Mock(wraps=torch_stage)
    monkeypatch.setattr(staging, "_torch_stage_mega_moe_inputs", fallback)
    capacity = 64
    warm = (
        torch.zeros(capacity, HIDDEN, dtype=torch.bfloat16, device="cuda"),
        torch.full((capacity, TOPK), 1.0 / TOPK, dtype=torch.float32, device="cuda"),
        (torch.arange(capacity * TOPK, device="cuda") % 32).view(capacity, TOPK),
    )
    assert warm[2].dtype == torch.int64
    batch = _batch(7, torch.int32)
    got, reference = _outputs(capacity), _outputs(capacity)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        staging.stage_mega_moe_inputs(*warm, *got)
    torch.cuda.current_stream().wait_stream(stream)
    torch_stage(*warm, *reference)
    compiled = staging._STAGERS[_key(torch.int64)].compiled
    assert _key(torch.int32) not in staging._STAGERS
    assert fallback.call_count == 0

    import cutlass.cute as cute

    real_compile = cute.compile
    compile_guard = Mock(side_effect=AssertionError("CuTe compile during capture"))
    monkeypatch.setattr(cute, "compile", compile_guard)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        staging.stage_mega_moe_inputs(*batch, *got)
    torch.cuda.synchronize()
    for actual, expected in zip(got, reference, strict=False):
        actual.copy_(expected)
    assert fallback.call_count == 1
    for step in range(3):
        for target, fresh in zip(
            batch, _batch(7, torch.int32, offset=step + 11), strict=False
        ):
            target.copy_(fresh)
        graph.replay()
        torch_stage(*batch, *reference)
        _exact(got, reference, 7, capacity)
    compile_guard.assert_not_called()
    assert _key(torch.int32) not in staging._STAGERS
    assert staging._STAGERS[_key(torch.int64)].compiled is compiled

    # A later eager call can compile int32 without changing the previously
    # captured Torch graph or redirecting it to these new buffers/pointers.
    monkeypatch.setattr(cute, "compile", real_compile)
    eager_batch = _batch(63, torch.int32, offset=23)
    eager_got, eager_reference = _outputs(128), _outputs(128)
    assert all(
        a.data_ptr() != b.data_ptr() for a, b in zip(eager_batch, batch, strict=False)
    )
    assert all(
        a.data_ptr() != b.data_ptr() for a, b in zip(eager_got, got, strict=False)
    )
    before = fallback.call_count
    staging.stage_mega_moe_inputs(*eager_batch, *eager_got)
    torch_stage(*eager_batch, *eager_reference)
    _exact(eager_got, eager_reference, 63, 128)
    assert callable(staging._STAGERS[_key(torch.int32)].compiled)
    assert fallback.call_count == before
    for target, fresh in zip(batch, _batch(7, torch.int32, offset=29), strict=False):
        target.copy_(fresh)
    graph.replay()
    torch_stage(*batch, *reference)
    _exact(got, reference, 7, capacity)
    _exact(eager_got, eager_reference, 63, 128)
    assert fallback.call_count == before
