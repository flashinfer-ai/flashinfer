# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dense packed scheduling preserves partial tiles and empty-K/V identities."""

import ast
import itertools
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


_REQUIRES_CONTEXT_GPU = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
    reason="Prims-TS context requires SM100, SM103, or SM107",
)


def _indptr(lengths):
    return torch.tensor(
        [0, *itertools.accumulate(lengths)], device="cuda", dtype=torch.int32
    )


def _reference(q, k, v, q_lengths, k_lengths, mask_type="dense"):
    head_ratio = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(head_ratio, dim=1)
    v = v.repeat_interleave(head_ratio, dim=1)
    outputs, statistics = [], []
    qi = ki = 0
    for nq, nk in zip(q_lengths, k_lengths, strict=True):
        if nk:
            scores = torch.einsum(
                "qhd,khd->qhk", q[qi : qi + nq].double(), k[ki : ki + nk].double()
            ) / math.sqrt(q.shape[-1])
            if mask_type == "causal":
                visible = torch.arange(nk, device=q.device)[None, :] <= (
                    torch.arange(nq, device=q.device)[:, None] + nk - nq
                )
                scores.masked_fill_(~visible[:, None, :], -torch.inf)
            maximum = scores.amax(-1)
            probability = (scores - maximum[..., None]).exp()
            denominator = probability.sum(-1)
            outputs.append(
                torch.einsum(
                    "qhk,khd->qhd",
                    probability / denominator[..., None],
                    v[ki : ki + nk].double(),
                )
            )
            statistics.append(torch.stack((maximum, denominator), -1))
        else:
            outputs.append(
                torch.zeros(
                    nq, q.shape[1], v.shape[-1], device="cuda", dtype=torch.float64
                )
            )
            empty = torch.zeros(nq, q.shape[1], 2, device="cuda", dtype=torch.float64)
            empty[..., 0] = -torch.inf
            statistics.append(empty)
        qi += nq
        ki += nk
    return torch.cat(outputs), torch.cat(statistics)


@pytest.mark.parametrize("dtype_pair", ("bf16", "fp8", "mixed"))
@pytest.mark.parametrize("store_softmax_stats", (False, True))
@pytest.mark.parametrize("kv_heads", (2, 6))
@_REQUIRES_CONTEXT_GPU
def test_attention_ts_dense_packed_q_pruning_replay(
    dtype_pair, store_softmax_stats, kv_heads
):
    """Replay moves partial/absent Q tiles while valid empty-K rows still store."""
    pytest.importorskip(
        "cutlass",
        minversion="4.8.0"
        if torch.cuda.get_device_capability() == (10, 7)
        else "4.7.0",
    )
    from flashinfer.attention.prims_ts import BatchPrefillTSWrapper

    torch.manual_seed(91231)
    q_lengths = (1, 127, 129, 257)
    k_lengths = (0 if store_softmax_stats else 1, 129, 257, 513)
    qk_dtype = torch.float8_e4m3fn if dtype_pair == "fp8" else torch.bfloat16
    v_dtype = torch.bfloat16 if dtype_pair == "bf16" else torch.float8_e4m3fn
    heads = 6
    q = (0.2 * torch.randn(sum(q_lengths), heads, 192, device="cuda")).to(qk_dtype)
    k = (0.2 * torch.randn(sum(k_lengths), kv_heads, 192, device="cuda")).to(qk_dtype)
    v = (0.2 * torch.randn(sum(k_lengths), kv_heads, 128, device="cuda")).to(v_dtype)
    qo, ko = _indptr(q_lengths), _indptr(k_lengths)
    output_storage = torch.full(
        (sum(q_lengths) + 2, heads, 128), 71, device="cuda", dtype=torch.bfloat16
    )
    output = output_storage[1:-1]
    stats_storage = torch.full(
        (sum(q_lengths) + 2, heads, 2), 71, device="cuda", dtype=torch.float32
    )
    stats = stats_storage[1:-1] if store_softmax_stats else None
    wrapper = BatchPrefillTSWrapper()
    wrapper.plan(
        device="cuda",
        batch_size=len(q_lengths),
        max_q_len=1024,
        max_kv_len=1024,
        num_qo_heads=heads,
        num_kv_heads=kv_heads,
        head_dim=192,
        head_dim_vo=128,
        q_dtype=qk_dtype,
        k_dtype=qk_dtype,
        v_dtype=v_dtype,
        out_dtype=torch.bfloat16,
        packed_query=True,
        mask_type="dense",
        store_softmax_stats=store_softmax_stats,
    )
    assert dict(wrapper._plan_state.policy)["scheduler"] == "clc_dynamic_persistent"

    def check(q_lens, k_lens):
        expected_output, expected_stats = _reference(q, k, v, q_lens, k_lens)
        torch.testing.assert_close(
            output.double(), expected_output, atol=0.01, rtol=0.02
        )
        relative_l2 = (
            output.double() - expected_output
        ).norm() / expected_output.norm().clamp_min(1e-6)
        assert relative_l2.item() < (0.01 if dtype_pair == "bf16" else 0.05)
        if stats is not None:
            torch.testing.assert_close(
                stats.double(), expected_stats, atol=2e-3, rtol=2e-3
            )
            qi = 0
            for nq, nk in zip(q_lens, k_lens, strict=True):
                if not nk:
                    assert (output[qi : qi + nq] == 0).all()
                    assert torch.isneginf(stats[qi : qi + nq, :, 0]).all()
                    assert (stats[qi : qi + nq, :, 1] == 0).all()
                qi += nq
        for storage in (output_storage, stats_storage):
            assert (storage[0] == 71).all()
            assert (storage[-1] == 71).all()

    output.fill_(torch.nan)
    if stats is not None:
        stats.fill_(torch.nan)
    wrapper.run(q, k, v, qo, ko, out=output, softmax_stats=stats)
    check(q_lengths, k_lengths)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(q, k, v, qo, ko, out=output, softmax_stats=stats, validate=False)
    for q_lens, k_lens in (
        (q_lengths[::-1], k_lengths[::-1]),
        (q_lengths, k_lengths),
    ):
        qo.copy_(_indptr(q_lens))
        ko.copy_(_indptr(k_lens))
        output.fill_(torch.nan)
        if stats is not None:
            stats.fill_(torch.nan)
        graph.replay()
        check(q_lens, k_lens)


_GRAPH_OWNERS = []


@pytest.fixture(scope="module")
def context_source():
    pytest.importorskip("cutlass", minversion="4.7.0")
    from flashinfer.attention.prims_ts import context

    return Path(context.__file__).parent / "kernels/fmha_context"


@pytest.fixture(scope="module")
def queue_selection_predicates(context_source):
    """Inspect both real queue branches without constructing a CUDA task graph."""
    tree = ast.parse((context_source / "fmha_kernel.py").read_text())
    predicates = [
        node.test
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(statement, ast.Assign)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Name)
            and statement.value.func.id == "PackedContextWorkQueue"
            for statement in node.body
        )
    ]
    assert len(predicates) == 2
    return [
        compile(ast.Expression(test), str(context_source), "eval")
        for test in predicates
    ]


@pytest.mark.parametrize("selection_index", (0, 1), ids=("clc", "static"))
def test_attention_ts_dense_q_pruning_queue_scope(
    queue_selection_predicates, selection_index
):
    """Both current schedulers prune nonuniform packed and paged requests."""
    for varlen, uniform, paged_lengths in itertools.product((False, True), repeat=3):
        cfg = SimpleNamespace(has_varlen=varlen, has_uniform_varlen=uniform)
        actual = eval(
            queue_selection_predicates[selection_index],
            {},
            {"cfg": cfg, "g_seq_lens_kv": object() if paged_lengths else None},
        )
        assert actual is ((varlen and not uniform) or paged_lengths)


@pytest.fixture(scope="module")
def skip_work_tile(context_source):
    """Evaluate the existing integer predicate, not the generated GPU program."""
    tree = ast.parse((context_source / "fmha_tasks.py").read_text())
    queue = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PackedContextWorkQueue"
    )
    method = next(
        node
        for node in queue.body
        if isinstance(node, ast.FunctionDef) and node.name == "skip_work_tile_if"
    )
    method.decorator_list = []
    method.returns = None
    for arg in method.args.args:
        arg.annotation = None
    namespace = {
        "Int32": int,
        "cutlass": SimpleNamespace(const_expr=lambda value: value, Boolean=bool),
    }
    exec(
        compile(
            ast.Module(body=[method], type_ignores=[]), str(context_source), "exec"
        ),
        namespace,
    )
    return namespace["skip_work_tile_if"]


@pytest.mark.parametrize("tile_size", (128, 256))
@pytest.mark.parametrize("indices", ((0, 1, 2), (2, 0, 1)))
@pytest.mark.parametrize("reverse", (False, True))
def test_attention_ts_dense_q_pruning_query_boundaries(
    skip_work_tile, tile_size, indices, reverse
):
    """An empty private Q domain skips all tiles; partial pairs stay executable."""
    tiles = 1024 // tile_size
    cfg = SimpleNamespace(
        work_tile_coord_indices=indices,
        uses_causal_reversed_head_batch_seq_tile_order=reverse,
        num_seq_tiles=tiles,
        cta_tiler=(tile_size, 128, 192),
        has_uniform_varlen=False,
    )
    for q_length in (0, 1, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024):
        queue = SimpleNamespace(
            cfg=cfg,
            cum_seqlen_q=(7, 7 + q_length),
            cum_seqlen_k=(0, 0),
            seq_lens_kv=None,
            keep_empty_kv=True,
        )
        for tile in range(tiles):
            coordinate = [0, 0, 0]
            coordinate[indices[0]] = tile
            actual = skip_work_tile(queue, SimpleNamespace(tile_idx=coordinate))
            logical_tile = tiles - tile - 1 if reverse else tile
            assert actual == (logical_tile * tile_size >= q_length)


def test_attention_ts_dense_q_pruning_preserves_nonnegative_offset_contract(
    context_source,
):
    """The public metadata reader accepts empty requests and rejects decreasing offsets."""
    pytest.importorskip("cutlass", minversion="4.7.0")
    from flashinfer.attention.prims_ts.context import _read_indptr

    offsets = torch.tensor([0, 0, 1], dtype=torch.int32)
    for name in ("qo_indptr", "kv_indptr"):
        assert _read_indptr(offsets, name, expected_total=1)[1] == (0, 1)
        with pytest.raises(ValueError, match="nondecreasing"):
            _read_indptr(
                torch.tensor([0, 2, 1], dtype=torch.int32), name, expected_total=1
            )


def _exercise_expanded_mla_replay(
    *, q_lengths, k_lengths, dtype_pair, store_softmax_stats, packed, mask_type
):
    pytest.importorskip(
        "cutlass",
        minversion="4.8.0"
        if torch.cuda.get_device_capability() == (10, 7)
        else "4.7.0",
    )
    from flashinfer.attention.prims_ts import BatchPrefillTSWrapper

    torch.manual_seed(91296)
    dtype = torch.float8_e4m3fn if dtype_pair == "fp8" else torch.bfloat16
    heads = 96
    q_flat = (0.2 * torch.randn(sum(q_lengths), heads, 192, device="cuda")).to(dtype)
    k_flat = (0.2 * torch.randn(sum(k_lengths), heads, 192, device="cuda")).to(dtype)
    v_flat = (0.2 * torch.randn(sum(k_lengths), heads, 128, device="cuda")).to(dtype)
    batch_size = len(q_lengths)
    q = q_flat if packed else q_flat.view(batch_size, q_lengths[0], heads, 192)
    k = k_flat if packed else k_flat.view(batch_size, k_lengths[0], heads, 192)
    v = v_flat if packed else v_flat.view(batch_size, k_lengths[0], heads, 128)
    qo, ko = (_indptr(q_lengths), _indptr(k_lengths)) if packed else (None, None)
    output_storage = torch.full(
        (sum(q_lengths) + 2, heads, 128), 71, device="cuda", dtype=torch.bfloat16
    )
    stats_storage = torch.full(
        (sum(q_lengths) + 2, heads, 2), 71, device="cuda", dtype=torch.float32
    )
    output = output_storage[1:-1].view(*q.shape[:-1], 128)
    stats = stats_storage[1:-1].view(*q.shape[:-1], 2) if store_softmax_stats else None
    wrapper = BatchPrefillTSWrapper()
    wrapper.plan(
        device="cuda",
        batch_size=batch_size,
        max_q_len=1024 if packed else q_lengths[0],
        max_kv_len=1024 if packed else k_lengths[0],
        num_qo_heads=heads,
        num_kv_heads=heads,
        head_dim=192,
        head_dim_vo=128,
        q_dtype=dtype,
        k_dtype=dtype,
        v_dtype=dtype,
        out_dtype=torch.bfloat16,
        packed_query=packed,
        mask_type=mask_type,
        store_softmax_stats=store_softmax_stats,
    )
    state = wrapper._plan_state
    assert state is not None
    if packed:
        assert dict(state.policy)["scheduler"] == "clc_dynamic_persistent"

    def check(q_lens, k_lens):
        expected, expected_stats = _reference(
            q_flat, k_flat, v_flat, q_lens, k_lens, mask_type
        )
        actual = output.reshape(-1, heads, 128).double()
        torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.02)
        relative_l2 = (actual - expected).norm() / expected.norm().clamp_min(1e-6)
        assert relative_l2.item() < (0.01 if dtype_pair == "bf16" else 0.05)
        if stats is not None:
            actual_stats = stats.reshape(-1, heads, 2)
            torch.testing.assert_close(
                actual_stats.double(), expected_stats, atol=2e-3, rtol=2e-3
            )
            qi = 0
            for nq, nk in zip(q_lens, k_lens, strict=True):
                if not nk:
                    assert (actual[qi : qi + nq] == 0).all()
                    assert torch.isneginf(actual_stats[qi : qi + nq, :, 0]).all()
                    assert (actual_stats[qi : qi + nq, :, 1] == 0).all()
                qi += nq
        for storage in (output_storage, stats_storage):
            assert (storage[0] == 71).all()
            assert (storage[-1] == 71).all()

    wrapper.run(q, k, v, qo, ko, out=output, softmax_stats=stats)
    check(q_lengths, k_lengths)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(q, k, v, qo, ko, out=output, softmax_stats=stats, validate=False)
    for q_lens, k_lens in ((q_lengths[::-1], k_lengths[::-1]), (q_lengths, k_lengths)):
        if packed:
            qo.copy_(_indptr(q_lens))
            ko.copy_(_indptr(k_lens))
        wrapper.run(q, k, v, qo, ko, out=output, softmax_stats=stats)
        eager_output = output.clone()
        eager_stats = stats.clone() if stats is not None else None
        output.fill_(torch.nan)
        if stats is not None:
            stats.fill_(torch.nan)
        graph.replay()
        check(q_lens, k_lens)
        torch.testing.assert_close(output, eager_output, rtol=0, atol=0)
        if stats is not None:
            torch.testing.assert_close(stats, eager_stats, rtol=0, atol=0)
        assert wrapper._plan_state is state
    _GRAPH_OWNERS.append(
        (graph, wrapper, q, k, v, qo, ko, output_storage, stats_storage)
    )


@pytest.mark.parametrize("dtype_pair", ("bf16", "fp8"))
@pytest.mark.parametrize("store_softmax_stats", (False, True))
@_REQUIRES_CONTEXT_GPU
def test_attention_ts_dense_q_pruning_expanded_mla_cached_prefix(
    dtype_pair, store_softmax_stats
):
    """H96 expanded MLA crosses each half/pair boundary within one cached plan."""
    _exercise_expanded_mla_replay(
        q_lengths=(1, 127, 128, 129, 255, 256, 257, 511, 512, 513),
        k_lengths=(
            0 if store_softmax_stats else 257,
            385,
            512,
            513,
            641,
            768,
            769,
            1024,
            1024,
            1024,
        ),
        dtype_pair=dtype_pair,
        store_softmax_stats=store_softmax_stats,
        packed=True,
        mask_type="dense",
    )


@pytest.mark.parametrize(
    "packed,mask_type", ((True, "causal"), (False, "dense"), (False, "causal"))
)
@pytest.mark.parametrize("store_softmax_stats", (False, True))
@_REQUIRES_CONTEXT_GPU
def test_attention_ts_dense_q_pruning_unaffected_controls(
    packed, mask_type, store_softmax_stats
):
    """Causal queues and nonpacked layouts retain their existing contracts."""
    _exercise_expanded_mla_replay(
        q_lengths=(1, 127, 129, 257) if packed else (129, 129),
        k_lengths=(129, 257, 513, 769) if packed else (513, 513),
        dtype_pair="fp8",
        store_softmax_stats=store_softmax_stats,
        packed=packed,
        mask_type=mask_type,
    )
