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

"""Runtime packed-dense N/N-1 domains under a fixed 4096-token plan."""

import math

import pytest
import torch


_Q_LENGTHS = (257, 129)
_K_BOUNDARIES = (0, 1, 127, 128, 129, 255, 256, 257, 4096)
_HEADS = 6
_REQUIRES_CONTEXT_GPU = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
    reason="Runtime packed-dense domains require SM100, SM103, or SM107",
)


def _offsets(lengths: tuple[int, ...]) -> list[int]:
    return [0, lengths[0], sum(lengths)]


def _reference(q, k, v, k_lengths, scale):
    outputs, statistics = [], []
    q_begin = k_begin = 0
    for nq, nk in zip(_Q_LENGTHS, k_lengths, strict=True):
        if nk == 0:
            output = torch.zeros(nq, _HEADS, 128, device=q.device)
            stats = torch.zeros(nq, _HEADS, 2, device=q.device)
            stats[..., 0] = -torch.inf
        else:
            scores = (
                torch.einsum(
                    "qhd,khd->qhk",
                    q[q_begin : q_begin + nq].float(),
                    k[k_begin : k_begin + nk].float(),
                )
                * scale
            )
            maximum = scores.amax(-1)
            denominator = (scores - maximum[..., None]).exp().sum(-1)
            stats = torch.stack((maximum, denominator), -1)
            output = (
                torch.einsum(
                    "qhk,khd->qhd",
                    scores.softmax(-1),
                    v[k_begin : k_begin + nk].float(),
                )
                * 0.625
            )
        outputs.append(output)
        statistics.append(stats)
        q_begin += nq
        k_begin += nk
    return torch.cat(outputs), torch.cat(statistics)


def _check(output, stats, expected, expected_stats, k_lengths) -> None:
    assert torch.isfinite(output.float()).all()
    torch.testing.assert_close(output.float(), expected, atol=0.01, rtol=0.02)
    begin = 0
    for nq, nk in zip(_Q_LENGTHS, k_lengths, strict=True):
        actual, target = (
            output[begin : begin + nq].float(),
            expected[begin : begin + nq],
        )
        error, norm = (
            torch.linalg.vector_norm(actual - target),
            torch.linalg.vector_norm(target),
        )
        if norm == 0:
            assert torch.equal(actual, target)
        else:
            assert error / norm < 0.02
        if nk == 0:
            assert (actual == 0).all()
            assert torch.isneginf(stats[begin : begin + nq, :, 0]).all()
            assert (stats[begin : begin + nq, :, 1] == 0).all()
        else:
            state, reference = (
                stats[begin : begin + nq],
                expected_stats[begin : begin + nq],
            )
            assert torch.isfinite(state).all() and (state[..., 1] > 0).all()
            torch.testing.assert_close(state, reference, atol=0.002, rtol=0.002)
            torch.testing.assert_close(
                state[..., 0] + state[..., 1].log(),
                reference[..., 0] + reference[..., 1].log(),
                atol=0.002,
                rtol=0.002,
            )
        begin += nq


def _same_bytes(actual, expected) -> None:
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


def _states() -> tuple[int, ...]:
    alternating = tuple(value for length in _K_BOUNDARIES for value in (4096, length))
    return alternating + alternating[::-1]


@_REQUIRES_CONTEXT_GPU
def test_runtime_dense_kv_domain_graph_replay() -> None:
    """One cap4096 graph alternates live N=1 and N=32 with partial Q peers."""
    pytest.importorskip("cutlass", minversion="4.8.0")
    from cutlass.experimental import primitives
    from flashinfer.attention.prims_ts import BatchPrefillTSWrapper

    if torch.cuda.get_device_capability() == (10, 7):
        assert hasattr(primitives, "tcgen05_ld_red"), (
            "Native SM107 DSL reduction support required"
        )
    dtype, head_dim = torch.bfloat16, 192
    torch.manual_seed(2026110101)
    total_k = 4096
    q = torch.randn(
        sum(_Q_LENGTHS), _HEADS, head_dim, device="cuda", dtype=torch.bfloat16
    ).to(dtype)
    k = torch.randn(total_k, _HEADS, head_dim, device="cuda", dtype=torch.bfloat16).to(
        dtype
    )
    v = torch.randn(total_k, _HEADS, 128, device="cuda", dtype=torch.bfloat16).to(dtype)
    qo = torch.tensor(_offsets(_Q_LENGTHS), device="cuda", dtype=torch.int32)
    ko = torch.tensor([0, 4096, total_k], device="cuda", dtype=torch.int32)
    output = torch.empty(
        sum(_Q_LENGTHS),
        _HEADS,
        128,
        device="cuda",
        dtype=dtype,
    )
    stats = torch.empty(sum(_Q_LENGTHS), _HEADS, 2, device="cuda")
    scale = 0.75 / math.sqrt(head_dim)
    wrapper = BatchPrefillTSWrapper()
    wrapper.plan(
        device=q.device,
        batch_size=2,
        max_q_len=512,
        max_kv_len=4096,
        num_qo_heads=_HEADS,
        num_kv_heads=_HEADS,
        head_dim=head_dim,
        head_dim_vo=128,
        q_dtype=q.dtype,
        k_dtype=k.dtype,
        v_dtype=v.dtype,
        out_dtype=output.dtype,
        packed_query=True,
        mask_type="dense",
        store_softmax_stats=True,
        softmax_scale=scale,
        output_scale=0.625,
    )
    from cutlass.cutlass_dsl import BaseDSL

    arch = BaseDSL._get_dsl().get_arch_enum()
    assert (arch.major, arch.minor) == torch.cuda.get_device_capability()
    assert dict(wrapper._plan_state.policy)["scheduler"] == "clc_dynamic_persistent"

    def run(validate=False):
        return wrapper.run(
            q, k, v, qo, ko, out=output, softmax_stats=stats, validate=validate
        )

    run(validate=True)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        assert run() is output
    original_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for pattern in ("random", "negative-peer", "zero-query"):
            for length in _states():
                live = (length, total_k - length)
                ko.copy_(torch.tensor(_offsets(live), device="cuda", dtype=torch.int32))
                if pattern == "negative-peer":
                    qr, kr, vr = (
                        torch.zeros_like(q, dtype=torch.float32),
                        torch.zeros_like(k, dtype=torch.float32),
                        torch.zeros_like(v, dtype=torch.float32),
                    )
                    feature = head_dim - 32
                    qr[..., feature] = 1
                    kr[:length, :, feature] = -128
                    kr[length:, :, feature] = 128
                    vr[:length] = 1
                    vr[length:] = 2
                    if length:
                        kr[length - 1, :, feature] = -256
                    q.copy_(qr.to(q.dtype))
                    k.copy_(kr.to(k.dtype))
                    v.copy_(vr.to(v.dtype))
                elif pattern == "zero-query":
                    q.zero_()
                expected, expected_stats = _reference(q, k, v, live, scale)
                inputs = [tensor.clone() for tensor in (q, k, v, qo, ko)]
                output.fill_(torch.nan)
                stats.fill_(torch.nan)
                # No eager call may refresh metadata between its mutation and replay.
                graph.replay()
                torch.cuda.synchronize()
                _check(output, stats, expected, expected_stats, live)
                graph_output = output.clone()
                graph_stats = stats.clone()
                output.fill_(torch.nan)
                stats.fill_(torch.nan)
                run(validate=True)
                _check(output, stats, expected, expected_stats, live)
                _same_bytes(output, graph_output)
                _same_bytes(stats, graph_stats)
                for value, before in zip((q, k, v, qo, ko), inputs, strict=True):
                    _same_bytes(value, before)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_tf32
        graph.reset()
