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

"""Runtime packed-dense N/N-1 domains under a fixed 4096-token plan."""

import ast
from contextlib import contextmanager
import inspect
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch


_Q_LENGTHS = (257, 129)
_K_BOUNDARIES = (0, 1, 127, 128, 129, 255, 256, 257, 4096)
_HEADS = 6
_REQUIRES_SM107 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="Runtime packed-dense MLA domains require SM107",
)


def _kernel_source() -> Path:
    suffix = "kernels.fmha_context.fmha_kernel"
    for prefix in (
        "flashinfer.attention.prims_ts",
        "tensorrt_llm._torch.attention.backends.prims_ts",
    ):
        module = sys.modules.get(prefix + "." + suffix)
        if module is not None:
            return Path(module.__file__)
    path = (
        Path(__file__).resolve().parents[2]
        / "flashinfer/attention/prims_ts/kernels/fmha_context/fmha_kernel.py"
    )
    if not path.is_file():
        raise RuntimeError("The actual native kernel module or source tree is required")
    return path


@contextmanager
def _actual_cpu_policy(arch=(10, 7)):
    """Execute actual policy/domain source with integer stand-ins, not DSL lowering."""
    source = _kernel_source()
    tree = ast.parse(source.read_text())
    names = {
        "_as_i32",
        "_domain_min",
        "_domain_max",
        "_init_task_with_domain",
        "PackedDenseDomainTask",
        "_uses_sm107_dense_mla_runtime_kv_bounds",
        "_select_fmha_domain_policy",
    }
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    assert {node.name for node in nodes} == names

    class Task:
        def __init__(self, **kwargs):
            self.schedule = kwargs["schedule"]

    cutlass = ModuleType("cutlass")
    cutlass.Float8E4M3FN, cutlass.BFloat16, cutlass.Float16 = (
        object(),
        object(),
        object(),
    )
    dsl = ModuleType("cutlass.cutlass_dsl")
    dsl.BaseDSL = SimpleNamespace(
        _get_dsl=lambda: SimpleNamespace(
            get_arch_enum=lambda: SimpleNamespace(major=arch[0], minor=arch[1])
        )
    )
    namespace = dict(
        __name__=__name__,
        Task=Task,
        Int32=int,
        cutlass=cutlass,
        cute=SimpleNamespace(
            ceil_div=lambda a, b: (a + b - 1) // b,
            math=SimpleNamespace(
                min=lambda a, b: min(int(a), int(b)),
                max=lambda a, b: max(int(a), int(b)),
            ),
        ),
        FmhaDomainPolicy=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module_ast = ast.fix_missing_locations(
        ast.Module(body=[future, *nodes], type_ignores=[])
    )
    with patch.dict(sys.modules, {"cutlass": cutlass, "cutlass.cutlass_dsl": dsl}):
        exec(compile(module_ast, str(source), "exec"), namespace)
        cfg = SimpleNamespace(
            is_causal=False,
            has_variable_window=False,
            window_size_left=0,
            head_paired=False,
            use_paged_kv=False,
            has_varlen=True,
            has_uniform_varlen=False,
            num_qkv_instances=2,
            two_cta_umma=False,
            logical_head_dim_qk=192,
            pv_mma_tiler=(128, 128, 32),
            q_dtype=cutlass.Float8E4M3FN,
            k_dtype=cutlass.Float8E4M3FN,
            v_dtype=cutlass.Float8E4M3FN,
            o_dtype=cutlass.BFloat16,
            work_tile_coord_indices=(1, 0, 2),
            uses_causal_reversed_head_batch_seq_tile_order=False,
            kv_tile_n=128,
            single_qkv_instance=False,
            fixed_dense_k_tail=0,
        )
        yield namespace, cfg


@pytest.mark.parametrize("offset", (0, 1), ids=("N", "N-minus-1"))
def test_runtime_dense_domain_formula_and_static_schedule_bounds(offset: int) -> None:
    with _actual_cpu_policy() as (namespace, _cfg):
        task_type = namespace["PackedDenseDomainTask"]
        for planned in (1, 2, 32):
            offsets = [0, 0, 4096]
            schedule = SimpleNamespace(loop_end=-99)
            task = task_type(
                num_kv_tiles=planned,
                tile_size_kv=128,
                batch_idx=2,
                cum_seqlen_k=offsets,
                offset=offset,
                schedule=schedule,
            )
            assert schedule.loop_end == planned - offset
            for length in (*_K_BOUNDARIES, 4097, *_K_BOUNDARIES[::-1]):
                offsets[1] = min(length, 4096)
                offsets[2] = max(length, 4096)
                for batch in (0, 1):
                    live = offsets[batch + 1] - offsets[batch]
                    expected = min(planned, max((live + 127) // 128, 1)) - offset
                    assert task.get_domain((5, 3, batch)) == expected
                    assert schedule.loop_end == planned - offset
            # Runtime metadata is read again, while validation retains the plan bound.
            offsets[:] = [0, 0, 4096]
            assert task.get_domain((0, 0, 0)) == 1 - offset
            assert task.get_domain((0, 0, 1)) == planned - offset


def test_runtime_dense_domain_requires_captured_schedule() -> None:
    with (
        _actual_cpu_policy() as (namespace, _cfg),
        pytest.raises(ValueError, match="captured schedule"),
    ):
        namespace["PackedDenseDomainTask"](32, 128, 2, [0, 128])


@pytest.mark.parametrize("offset", (0, 1))
def test_runtime_dense_domain_preserves_dynamic_launch_bound(offset):
    """Current wrapper kernels pass a runtime bound, not a compile-time capacity."""

    class DynamicBound:
        value = 32

        def __int__(self):
            return self.value

    with _actual_cpu_policy() as (namespace, _cfg):
        bound = DynamicBound()
        schedule = SimpleNamespace(loop_end=71)
        indptr = [0, 0, 4096]
        task = namespace["PackedDenseDomainTask"](
            bound, 128, 2, indptr, offset=offset, schedule=schedule
        )
        assert schedule.loop_end == 71
        for launch_tiles in (1, 2, 32):
            bound.value = launch_tiles
            for length in _K_BOUNDARIES:
                indptr[1] = length
                expected = min(launch_tiles, max((length + 127) // 128, 1)) - offset
                assert task.get_domain((0, 0, 0)) == expected
                assert schedule.loop_end == 71


def test_runtime_dense_domain_plan_uses_current_api():
    pytest.importorskip("cutlass", minversion="4.7.0")
    from flashinfer.attention.prims_ts import BatchPrefillTSWrapper

    inspect.signature(BatchPrefillTSWrapper.plan).bind(
        None,
        device="cuda",
        batch_size=2,
        max_q_len=512,
        max_kv_len=4096,
        num_qo_heads=_HEADS,
        num_kv_heads=_HEADS,
        head_dim=192,
        head_dim_vo=128,
        q_dtype=torch.float8_e4m3fn,
        k_dtype=torch.float8_e4m3fn,
        v_dtype=torch.float8_e4m3fn,
        out_dtype=torch.bfloat16,
        packed_query=True,
        mask_type="dense",
        store_softmax_stats=True,
        softmax_scale=0.75 / math.sqrt(192),
        output_scale=0.625,
    )


def test_runtime_dense_policy_domains_share_the_live_pointer() -> None:
    with _actual_cpu_policy() as (namespace, cfg):
        pointer = [0, 0, 4096]
        select = namespace["_select_fmha_domain_policy"]
        policy = select(
            cfg,
            num_kv_tiles=32,
            q_offset=0,
            cum_seqlen_q=[0, 257, 386],
            cum_seqlen_k=pointer,
        )
        for values, offset in (
            (policy.domain_n_kwargs, 0),
            (policy.domain_n_minus_1_kwargs, 1),
            (policy.softmax0_domain_kwargs, 0),
            (policy.softmax1_domain_kwargs, 0),
        ):
            assert values["task_class"] is namespace["PackedDenseDomainTask"]
            assert values["cum_seqlen_k"] is pointer
            assert values["num_kv_tiles"] == 32 and values["tile_size_kv"] == 128
            assert values["batch_idx"] == 2 and values["offset"] == offset
        with pytest.raises(ValueError, match="runtime KV offsets"):
            select(
                cfg,
                num_kv_tiles=32,
                q_offset=0,
                cum_seqlen_q=[0, 257, 386],
                cum_seqlen_k=None,
            )


@pytest.mark.parametrize("arch", ((10, 0), (10, 3), (10, 7), (11, 0)))
def test_runtime_dense_policy_is_sm107_only(arch: tuple[int, int]) -> None:
    with _actual_cpu_policy(arch) as (namespace, cfg):
        assert namespace["_uses_sm107_dense_mla_runtime_kv_bounds"](cfg) is (
            arch == (10, 7)
        )
        if arch != (10, 7):
            policy = namespace["_select_fmha_domain_policy"](
                cfg, num_kv_tiles=32, q_offset=0, cum_seqlen_q=None, cum_seqlen_k=None
            )
            assert policy.domain_n_kwargs == {"domain": 32}
            assert policy.domain_n_minus_1_kwargs == {"domain": 31}


def test_runtime_dense_policy_excludes_other_layouts_and_precisions() -> None:
    with _actual_cpu_policy() as (namespace, cfg):
        predicate = namespace["_uses_sm107_dense_mla_runtime_kv_bounds"]
        assert predicate(cfg)
        changes = dict(
            is_causal=True,
            has_variable_window=True,
            window_size_left=128,
            head_paired=True,
            use_paged_kv=True,
            has_varlen=False,
            has_uniform_varlen=True,
            num_qkv_instances=1,
            two_cta_umma=True,
            logical_head_dim_qk=128,
            pv_mma_tiler=(128, 256, 32),
            q_dtype=object(),
            k_dtype=object(),
            v_dtype=object(),
            o_dtype=object(),
        )
        for name, value in changes.items():
            previous = getattr(cfg, name)
            setattr(cfg, name, value)
            assert not predicate(cfg), name
            setattr(cfg, name, previous)


def test_runtime_dense_task_routing_matches_n_and_n_minus_one() -> None:
    tree = ast.parse(_kernel_source().read_text())
    builder = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "build_context_task_manager"
    )
    routes = {}
    for node in ast.walk(builder):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            routes.setdefault(node.func.id, []).extend(
                keyword.value.id
                for keyword in node.keywords
                if keyword.arg is None and isinstance(keyword.value, ast.Name)
            )
    assert routes["create_load_task"] == ["domain_n_kwargs"]
    assert routes["create_mma_task"] == ["mma_domain_kwargs"]
    assert routes["create_softmax_task"] == [
        "softmax0_domain_kwargs",
        "softmax1_domain_kwargs",
    ]
    assert routes["create_correction_task"] == ["domain_n_minus_1_kwargs"]
    assert routes["create_epilogue_task"] == ["domain_n_kwargs"]
    assignment = next(
        node
        for node in builder.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "mma_domain_kwargs"
            for target in node.targets
        )
    )
    assert (
        isinstance(assignment.value, ast.Name)
        and assignment.value.id == "domain_n_minus_1_kwargs"
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
    torch.testing.assert_close(output.float(), expected, atol=0.13, rtol=0.05)
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
            assert error / norm < 0.05
        if nk == 0:
            assert (actual == 0).all()
            assert stats is not None
            assert torch.isneginf(stats[begin : begin + nq, :, 0]).all()
            assert (stats[begin : begin + nq, :, 1] == 0).all()
        elif stats is not None:
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


def _states(store_stats: bool) -> tuple[int, ...]:
    lengths = _K_BOUNDARIES if store_stats else _K_BOUNDARIES[1:]
    alternating = tuple(value for length in lengths for value in (4096, length))
    return alternating + alternating[::-1]


@pytest.mark.parametrize("store_stats", (True, False), ids=("stats", "output-only"))
@pytest.mark.arch_rubin
@_REQUIRES_SM107
def test_runtime_dense_kv_domain_graph_replay(store_stats: bool) -> None:
    """One cap4096 graph alternates live N=1 and N=32 with partial Q peers."""
    pytest.importorskip("cutlass", minversion="4.8.0")
    from cutlass.experimental import primitives
    from flashinfer.attention.prims_ts import BatchPrefillTSWrapper

    assert hasattr(primitives, "tcgen05_ld_red"), (
        "Native SM107 DSL reduction support required"
    )
    torch.manual_seed(2026110101)
    total_k = 4096 if store_stats else 4097
    q = torch.randn(
        sum(_Q_LENGTHS), _HEADS, 192, device="cuda", dtype=torch.bfloat16
    ).to(torch.float8_e4m3fn)
    k = torch.randn(total_k, _HEADS, 192, device="cuda", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    v = torch.randn(total_k, _HEADS, 128, device="cuda", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    qo = torch.tensor(_offsets(_Q_LENGTHS), device="cuda", dtype=torch.int32)
    ko = torch.tensor([0, 4096, total_k], device="cuda", dtype=torch.int32)
    output = torch.empty(
        sum(_Q_LENGTHS), _HEADS, 128, device="cuda", dtype=torch.bfloat16
    )
    stats = (
        torch.empty(sum(_Q_LENGTHS), _HEADS, 2, device="cuda") if store_stats else None
    )
    scale = 0.75 / math.sqrt(192)
    wrapper = BatchPrefillTSWrapper()
    wrapper.plan(
        device=q.device,
        batch_size=2,
        max_q_len=512,
        max_kv_len=4096,
        num_qo_heads=_HEADS,
        num_kv_heads=_HEADS,
        head_dim=192,
        head_dim_vo=128,
        q_dtype=q.dtype,
        k_dtype=k.dtype,
        v_dtype=v.dtype,
        out_dtype=output.dtype,
        packed_query=True,
        mask_type="dense",
        store_softmax_stats=store_stats,
        softmax_scale=scale,
        output_scale=0.625,
    )
    from cutlass.cutlass_dsl import BaseDSL

    arch = BaseDSL._get_dsl().get_arch_enum()
    assert (arch.major, arch.minor) == (10, 7)

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
            for length in _states(store_stats):
                live = (length, total_k - length)
                ko.copy_(torch.tensor(_offsets(live), device="cuda", dtype=torch.int32))
                if pattern == "negative-peer":
                    qr, kr, vr = (
                        torch.zeros_like(q, dtype=torch.float32),
                        torch.zeros_like(k, dtype=torch.float32),
                        torch.zeros_like(v, dtype=torch.float32),
                    )
                    qr[..., 160] = 1
                    kr[:length, :, 160] = -128
                    kr[length:, :, 160] = 128
                    vr[:length] = 1
                    vr[length:] = 2
                    if length:
                        kr[length - 1, :, 160] = -256
                    q.copy_(qr.to(q.dtype))
                    k.copy_(kr.to(k.dtype))
                    v.copy_(vr.to(v.dtype))
                elif pattern == "zero-query":
                    q.zero_()
                expected, expected_stats = _reference(q, k, v, live, scale)
                inputs = [tensor.clone() for tensor in (q, k, v, qo, ko)]
                output.fill_(torch.nan)
                if stats is not None:
                    stats.fill_(torch.nan)
                # No eager call may refresh metadata between its mutation and replay.
                graph.replay()
                torch.cuda.synchronize()
                _check(output, stats, expected, expected_stats, live)
                graph_output = output.clone()
                graph_stats = None if stats is None else stats.clone()
                output.fill_(torch.nan)
                if stats is not None:
                    stats.fill_(torch.nan)
                run(validate=True)
                _check(output, stats, expected, expected_stats, live)
                _same_bytes(output, graph_output)
                if stats is not None:
                    _same_bytes(stats, graph_stats)
                for value, before in zip((q, k, v, qo, ko), inputs, strict=True):
                    _same_bytes(value, before)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_tf32
        graph.reset()
