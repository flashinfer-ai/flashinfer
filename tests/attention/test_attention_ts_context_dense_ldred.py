# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Packed dense full-tile reduction, masked tails and live-length regressions."""

from contextlib import contextmanager, nullcontext
import inspect
import math

import pytest
import torch

pytest.importorskip("cutlass", minversion="4.7.0a0")

import flashinfer.attention.prims_ts.context as context_module
from flashinfer.attention.prims_ts import (
    BatchPrefillPagedTSWrapper,
    BatchPrefillTSWrapper,
)
from flashinfer.utils import is_sm100a_supported


_FP8 = torch.float8_e4m3fn
_Q_LENGTHS = (1, 127, 128, 257)
_HEADS = 6
_REQUIRES_GPU = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm100a_supported(torch.device("cuda")),
    reason="PrimTS context attention requires a supported SM10x GPU",
)


@pytest.fixture(autouse=True)
def _fp32_matmul_reference():
    original = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original


def _indptr(lengths: tuple[int, ...]) -> torch.Tensor:
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    return torch.tensor(offsets, device="cuda", dtype=torch.int32)


def _reference(
    q: torch.Tensor,
    keys: list[torch.Tensor],
    values: list[torch.Tensor],
    sm_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the FP32 output and natural-log max/sum reference from context tests."""
    outputs, statistics = [], []
    offset = 0
    for nq, k, v in zip(_Q_LENGTHS, keys, values, strict=True):
        qr = q[offset : offset + nq].float()
        if k.shape[0] == 0:
            outputs.append(torch.zeros((nq, _HEADS, v.shape[-1]), device=q.device))
            stats = torch.zeros((nq, _HEADS, 2), device=q.device)
            stats[..., 0] = -torch.inf
        else:
            scores = torch.einsum("qhd,khd->qhk", qr, k.float()) * sm_scale
            maximum = scores.amax(-1)
            denominator = (scores - maximum[..., None]).exp().sum(-1)
            stats = torch.stack((maximum, denominator), -1)
            outputs.append(
                torch.einsum("qhk,khd->qhd", scores.softmax(-1), v.float()) * 0.625
            )
        statistics.append(stats)
        offset += nq
    return torch.cat(outputs), torch.cat(statistics)


def _check(
    output: torch.Tensor,
    stats: torch.Tensor | None,
    expected: torch.Tensor,
    expected_stats: torch.Tensor,
    dtype: torch.dtype,
) -> None:
    # Match test_attention_ts_context_mla_prefill, including its stricter MLA L2.
    fp8 = dtype == _FP8
    atol, rtol, l2_limit = (0.13, 0.05, 0.05) if fp8 else (0.01, 0.02, 0.01)
    assert torch.isfinite(output.float()).all()
    torch.testing.assert_close(output.float(), expected, atol=atol, rtol=rtol)
    offset = 0
    for nq in _Q_LENGTHS:
        actual, reference = (
            output[offset : offset + nq].float(),
            expected[offset : offset + nq],
        )
        error = torch.linalg.vector_norm(actual - reference)
        denominator = torch.linalg.vector_norm(reference)
        if denominator == 0:
            assert error == 0
        else:
            assert error / denominator < l2_limit
        offset += nq
    if stats is not None:
        # Same canonical max/sum contract as test_attention_ts_context_softmax_stats.
        torch.testing.assert_close(stats, expected_stats, atol=2e-3, rtol=2e-3)
        empty = expected_stats[..., 1] == 0
        assert torch.isneginf(stats[..., 0][empty]).all()
        assert (stats[..., 1][empty] == 0).all()


def _bytes_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


@contextmanager
def _plan_mode(monkeypatch: pytest.MonkeyPatch, *, software: bool, paged: bool):
    """Observe the real policy; give forced software code a separate disk key."""
    if not software and not context_module._default_uses_ldtm_stat(
        torch.cuda.current_device(), 8, 8
    ):
        pytest.skip("This regression requires the normal LDTM.STAT-enabled policy")
    observed = []
    make_kernel = context_module._make_context_kernel
    build = context_module.build_and_load_attention_kernel
    cache_name = "_get_compiled_paged_context" if paged else "_get_compiled_context"
    compile_cached = getattr(context_module, cache_name)

    def observe(*args, **kwargs):
        kernel = make_kernel(*args, **kwargs)
        observed.append(kernel.cfg.uses_ldtm_stat)
        return kernel

    def isolated_build(*args, **kwargs):
        kwargs["extra"] = (*kwargs.get("extra", ()), "dense_ldred_test_software")
        return build(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(context_module, "_make_context_kernel", observe)
        # Bypass only the memo lookup, so observation also works after another test.
        patch.setattr(context_module, cache_name, compile_cached.__wrapped__)
        if software:
            patch.setattr(
                context_module, "_default_uses_ldtm_stat", lambda *_args: False
            )
            patch.setattr(
                context_module, "build_and_load_attention_kernel", isolated_build
            )
        yield
    # Planning constructs a scheduler probe as well as the compiled kernel.
    assert observed and all(value is (not software) for value in observed), observed


def _exercise(run, output, stats, inputs, references) -> None:
    """One captured plan sees new live lengths and freshly poisoned outputs."""
    graph = torch.cuda.CUDAGraph()
    graph_ready = False
    for expected, expected_stats in references:
        before = [value.clone() for value in inputs]
        if not graph_ready:
            output.fill_(torch.nan)
            if stats is not None:
                stats.fill_(torch.nan)
            assert run(True) is output
            _check(output, stats, expected, expected_stats, inputs[0].dtype)
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                assert run(False) is output
            graph_ready = True
        # Later live metadata changes must reach the graph without an eager
        # wrapper call first refreshing any host-derived internal buffers.
        output.fill_(torch.nan)
        if stats is not None:
            stats.fill_(torch.nan)
        graph.replay()
        torch.cuda.synchronize()
        _check(output, stats, expected, expected_stats, inputs[0].dtype)
        replay_output = output.clone()
        replay_stats = None if stats is None else stats.clone()
        output.fill_(torch.nan)
        if stats is not None:
            stats.fill_(torch.nan)
        assert run(True) is output
        _check(output, stats, expected, expected_stats, inputs[0].dtype)
        _bytes_equal(output, replay_output)
        if stats is not None:
            _bytes_equal(stats, replay_stats)
        for actual, original in zip(inputs, before, strict=True):
            _bytes_equal(actual, original)


def test_dense_live_metadata_replay_precedes_eager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A changed state reaches the existing graph before any host wrapper call."""
    value = torch.ones(1)
    output = torch.empty_like(value)
    calls = []

    class Graph:
        def replay(self) -> None:
            calls.append(("graph", int(value.item())))
            output.copy_(value)

    def run(validate: bool) -> torch.Tensor:
        calls.append(("eager" if validate else "capture", int(value.item())))
        output.copy_(value)
        return output

    def check(actual, _stats, expected, _expected_stats, _dtype) -> None:
        assert torch.equal(actual, expected)

    def references():
        for state in (1, 2):
            value.fill_(state)
            yield value.clone(), None

    monkeypatch.setattr(torch.cuda, "CUDAGraph", Graph)
    monkeypatch.setattr(torch.cuda, "graph", lambda _graph: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setitem(_exercise.__globals__, "_check", check)
    _exercise(run, output, None, (value,), references())
    assert calls == [
        ("eager", 1),
        ("capture", 1),
        ("graph", 1),
        ("eager", 1),
        ("graph", 2),
        ("eager", 2),
    ]


@pytest.mark.parametrize(
    "dtype,store_stats,software",
    [
        pytest.param(_FP8, True, False, id="fp8-stats"),
        pytest.param(_FP8, False, False, id="fp8-output-only"),
        pytest.param(torch.bfloat16, True, True, id="bf16-software"),
    ],
)
@_REQUIRES_GPU
def test_packed_dense_ldred_full_tail_and_live_offsets(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    store_stats: bool,
    software: bool,
) -> None:
    """D192/V128 mixes full/tail/empty tiles and excludes positive-score peers."""
    torch.manual_seed(2026103101)
    q = torch.randn(
        sum(_Q_LENGTHS), _HEADS, 192, device="cuda", dtype=torch.bfloat16
    ).to(dtype)
    k = torch.randn(514, _HEADS, 192, device="cuda", dtype=torch.bfloat16).to(dtype)
    v = torch.randn(514, _HEADS, 128, device="cuda", dtype=torch.bfloat16).to(dtype)
    qo, ko = _indptr(_Q_LENGTHS), _indptr((129, 128, 128, 129))
    output = torch.empty(
        sum(_Q_LENGTHS), _HEADS, 128, device="cuda", dtype=torch.bfloat16
    )
    stats = torch.empty((*output.shape[:-1], 2), device="cuda") if store_stats else None
    wrapper = BatchPrefillTSWrapper()
    scale = 0.75 / math.sqrt(192)
    with _plan_mode(monkeypatch, software=software, paged=False):
        wrapper.plan(
            device=q.device,
            batch_size=4,
            max_q_len=257,
            max_kv_len=257,
            num_qo_heads=_HEADS,
            num_kv_heads=_HEADS,
            head_dim=192,
            head_dim_vo=128,
            q_dtype=dtype,
            k_dtype=dtype,
            v_dtype=dtype,
            out_dtype=torch.bfloat16,
            packed_query=True,
            mask_type="dense",
            store_softmax_stats=store_stats,
            softmax_scale=scale,
            output_scale=0.625,
        )

    def run(validate: bool) -> torch.Tensor:
        return wrapper.run(
            q, k, v, qo, ko, out=output, softmax_stats=stats, validate=validate
        )

    lengths = (
        ((129, 128, 0, 257), (128, 129, 257, 0), (257, 1, 128, 128))
        if store_stats
        else ((129, 128, 128, 129), (128, 129, 1, 256), (257, 1, 128, 128))
    )

    def references():
        for pattern in ("random", "negative-peers", "zero-q"):
            for live_k in lengths:
                ko.copy_(_indptr(live_k))
                if pattern == "negative-peers":
                    qr, kr, vr = (
                        torch.zeros_like(q, dtype=torch.float32),
                        torch.zeros_like(k, dtype=torch.float32),
                        torch.zeros_like(v, dtype=torch.float32),
                    )
                    qr[..., 160] = 1
                    offset = 0
                    for request, nk in enumerate(live_k):
                        kr[offset : offset + nk, :, 160] = (
                            -128 if request % 2 == 0 else 128
                        )
                        vr[offset : offset + nk] = request + 1
                        if nk:
                            # A lower tail maximum must not replace the previous tile's max.
                            kr[offset + nk - 1, :, 160] = -256
                        offset += nk
                    q.copy_(qr.to(dtype))
                    k.copy_(kr.to(dtype))
                    v.copy_(vr.to(dtype))
                elif pattern == "zero-q":
                    q.zero_()
                yield _reference(q, list(k.split(live_k)), list(v.split(live_k)), scale)

    _exercise(run, output, stats, (q, k, v, qo, ko), references())
    qo[2] = 0
    with pytest.raises(ValueError, match="nondecreasing"):
        run(True)
    qo.copy_(_indptr(_Q_LENGTHS))
    if not store_stats:
        ko[2] = 0
        with pytest.raises(ValueError, match="nondecreasing"):
            run(True)


@pytest.mark.parametrize("head_dim", (128, 256), ids=("d128", "d256"))
@_REQUIRES_GPU
def test_paged_dense_ldred_full_tail_and_empty_admission(
    monkeypatch: pytest.MonkeyPatch, head_dim: int
) -> None:
    """Paged full/tail tiles honor live bounds and reject unsupported empty rows."""
    torch.manual_seed(2026103102)
    q = torch.randn(
        sum(_Q_LENGTHS), _HEADS, head_dim, device="cuda", dtype=torch.bfloat16
    ).to(_FP8)
    k = torch.randn(12, _HEADS, 128, head_dim, device="cuda", dtype=torch.bfloat16).to(
        _FP8
    )
    v = torch.randn_like(k, dtype=torch.bfloat16).to(_FP8)
    qo = _indptr(_Q_LENGTHS)
    table = torch.arange(11, -1, -1, device="cuda", dtype=torch.int32).reshape(4, 3)
    seq = torch.tensor([129, 128, 257, 1], device="cuda", dtype=torch.int32)
    output = torch.empty_like(q, dtype=torch.bfloat16)
    stats = torch.empty((*q.shape[:-1], 2), device="cuda")
    wrapper = BatchPrefillPagedTSWrapper()
    scale = 0.75 / math.sqrt(head_dim)
    with _plan_mode(monkeypatch, software=False, paged=True):
        wrapper.plan(
            device=q.device,
            batch_size=4,
            max_q_len=257,
            max_kv_len=257,
            num_qo_heads=_HEADS,
            num_kv_heads=_HEADS,
            head_dim=head_dim,
            q_dtype=_FP8,
            k_dtype=_FP8,
            v_dtype=_FP8,
            out_dtype=torch.bfloat16,
            page_size=128,
            mask_type="dense",
            store_softmax_stats=True,
            softmax_scale=scale,
            output_scale=0.625,
        )

    def run(validate: bool) -> torch.Tensor:
        return wrapper.run(
            q, k, v, qo, table, seq, out=output, softmax_stats=stats, validate=validate
        )

    def references():
        for live_k in ((129, 128, 257, 1), (128, 257, 1, 129), (257, 1, 128, 129)):
            seq.copy_(torch.tensor(live_k, device="cuda", dtype=torch.int32))
            keys, values = [], []
            for request, nk in enumerate(live_k):
                indices = table[request].long()
                keys.append(
                    k[indices].permute(0, 2, 1, 3).reshape(-1, _HEADS, head_dim)[:nk]
                )
                values.append(
                    v[indices].permute(0, 2, 1, 3).reshape(-1, _HEADS, head_dim)[:nk]
                )
            yield _reference(q, keys, values, scale)

    _exercise(run, output, stats, (q, k, v, qo, table, seq), references())
    seq[0] = 0
    with pytest.raises(ValueError, match="require positive KV lengths"):
        run(True)


def test_dense_ldred_uses_current_plan_api():
    """Bind the directed test's public keywords without requiring a GPU."""
    common = dict(
        device="cuda",
        batch_size=4,
        max_q_len=257,
        max_kv_len=257,
        num_qo_heads=_HEADS,
        num_kv_heads=_HEADS,
        head_dim=192,
        q_dtype=_FP8,
        k_dtype=_FP8,
        v_dtype=_FP8,
        out_dtype=torch.bfloat16,
        mask_type="dense",
        store_softmax_stats=True,
        softmax_scale=0.75 / math.sqrt(192),
        output_scale=0.625,
    )
    inspect.signature(BatchPrefillTSWrapper.plan).bind(
        None, **common, head_dim_vo=128, packed_query=True
    )
    inspect.signature(BatchPrefillPagedTSWrapper.plan).bind(
        None, **common, page_size=128
    )
