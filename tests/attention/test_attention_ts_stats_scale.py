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

"""Original-scale statistics at finite logits where base conversions lose bits."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


_ROOT = Path(__file__).resolve().parents[2]
_PRIMS = _ROOT / "flashinfer/attention/prims_ts"
_REQUIRES_PRIMTS_GPU = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
    reason="PrimTS attention requires SM100, SM103, or SM107",
)
_GRAPH_OWNERS = []


@pytest.mark.parametrize("family", ("context", "decode"))
@pytest.mark.parametrize("scale", (0.37, 0.1, 0.7))
def test_attention_ts_statistics_use_original_fp32_scale(family, scale):
    """Evaluate the actual export expression with explicit FP32 arithmetic."""
    relative = (
        "kernels/fmha_context/fmha_resources.py"
        if family == "context"
        else "kernels/fmha_decode/fmha_decode_resources/tmem_corr.py"
    )
    tree = ast.parse((_PRIMS / relative).read_text())
    name = "TmemStatsResource" if family == "context" else "TmemCorrResource"
    resource = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name
    )
    raw_name = "row_max" if family == "context" else "max_val"
    destination = "maximum" if family == "context" else "scaled_max"
    expressions = [
        n.value
        for n in ast.walk(resource)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == destination for t in n.targets)
        and isinstance(n.value, ast.BinOp)
        and raw_name in {x.id for x in ast.walk(n.value) if isinstance(x, ast.Name)}
    ]
    assert len(expressions) == 1
    natural_scale = np.float32(scale)
    log2_scale = natural_scale * np.float32(math.log2(math.e))
    resource_values = SimpleNamespace(
        stats_scale=natural_scale,
        scale_softmax_log2=log2_scale,
    )
    expression = compile(ast.Expression(expressions[0]), str(_PRIMS / relative), "eval")
    actual = []
    expected = []
    for raw in (np.float32(2**24), np.float32(2**24 + 2)):
        actual.append(
            eval(
                expression,
                {"Float32": np.float32, "math": math},
                {
                    "self": resource_values,
                    raw_name: raw,
                    "scale_softmax_log2": log2_scale,
                },
            )
        )
        expected.append(raw * natural_scale)
    np.testing.assert_array_equal(actual, expected)
    # Strict maxima matter: small absolute errors change both merge weights
    # and the relative mass of a sink at the original maximum.
    np.testing.assert_allclose(
        np.exp(np.asarray(actual) - actual[1]),
        np.exp(np.asarray(expected) - expected[1]),
        rtol=0,
        atol=0,
    )
    assert np.float32(1) + np.exp(np.float32(expected[0] - actual[0])) == np.float32(2)


def test_attention_ts_statistics_scale_callable_and_cache_contract():
    """The new context operand is appended; source hashes invalidate old artifacts."""
    context_tree = ast.parse((_PRIMS / "context.py").read_text())
    adapters = [
        n
        for n in ast.walk(context_tree)
        if isinstance(n, ast.FunctionDef) and n.name == "tensor_adapter"
    ]
    assert len(adapters) == 2
    assert all(n.args.args[-1].arg == "stats_scale" for n in adapters)
    compiled_calls = [
        n
        for n in ast.walk(context_tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "state"
        and n.func.attr == "compiled"
    ]
    assert len(compiled_calls) == 2
    assert all(
        ast.unparse(n.args[-1])
        == "softmax_scale if geometry.store_softmax_stats else None"
        for n in compiled_calls
    )
    compile_calls = [
        n
        for n in ast.walk(context_tree)
        if isinstance(n, ast.Call)
        and n.args
        and isinstance(n.args[0], ast.Name)
        and n.args[0].id == "tensor_adapter"
    ]
    assert len(compile_calls) == 2
    assert all(
        ast.unparse(n.args[-1])
        == "scale_fake if compile_spec.store_softmax_stats else None"
        for n in compile_calls
    )
    decode_tree = ast.parse(
        (_PRIMS / "kernels/fmha_decode/fmha_decode_kernel.py").read_text()
    )
    for name in (
        "_build_decode_gen_schedule",
        "_run_decode_gen_active",
        "_run_decode_gen_runtime_prefix",
        "decode_gen_kernel",
    ):
        function = next(
            n
            for n in decode_tree.body
            if isinstance(n, ast.FunctionDef) and n.name == name
        )
        assert function.args.args[-1].arg == "stats_scale"
    cache_source = (_PRIMS / "_compile_cache.py").read_text()
    assert 'root.rglob("*.py")' in cache_source
    assert "extra_key_files=_kernel_source_files()" in cache_source


def test_attention_ts_statistics_scale_reaches_every_decode_branch():
    tree = ast.parse((_PRIMS / "kernels/fmha_decode/fmha_decode_kernel.py").read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in ("_run_decode_gen_active", "_run_decode_gen_runtime_prefix")
    ]
    assert len(calls) == 5
    for call in calls:
        keywords = {
            keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords
        }
        assert keywords.get("g_softmax_stats") == "g_softmax_stats", call.lineno
        assert keywords.get("stats_scale") == "stats_scale", call.lineno


@_REQUIRES_PRIMTS_GPU
@pytest.mark.parametrize("paged", (False, True), ids=("contiguous", "paged"))
@pytest.mark.parametrize(
    "runtime_override", (False, True), ids=("plan_scale", "runtime_scale")
)
def test_attention_ts_context_mla_partition_original_scale(paged, runtime_override):
    """Merge context and dense MLA whose large maxima differ by half an FP32 unit."""
    from flashinfer.attention.prims_ts import (
        BatchMLADecodePagedTSWrapper,
        BatchPrefillPagedTSWrapper,
        BatchPrefillTSWrapper,
    )

    scale, heads, length = 0.37, 8, 128
    q = torch.zeros((1, 1, heads, 128), device="cuda", dtype=torch.bfloat16)
    q[..., 0], q[..., 1] = 4096, 1
    key = torch.zeros((1, length, heads, 128), device="cuda", dtype=q.dtype)
    key[..., 0] = 4096
    value = torch.zeros_like(key)
    value[..., 0] = 4096
    offsets = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    pages = torch.zeros((1, 1), device="cuda", dtype=torch.int32)
    lengths = torch.full((1,), length, device="cuda", dtype=torch.int32)
    if paged:
        q = q.squeeze(0)
        key = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()
    stats = torch.empty((*q.shape[:-1], 2), device="cuda", dtype=torch.float32)
    wrapper = BatchPrefillPagedTSWrapper() if paged else BatchPrefillTSWrapper()
    wrapper.plan(
        device="cuda",
        batch_size=1,
        max_q_len=1,
        max_kv_len=length,
        num_qo_heads=heads,
        num_kv_heads=heads,
        head_dim=128,
        q_dtype=q.dtype,
        k_dtype=key.dtype,
        mask_type="dense",
        softmax_scale=0.19 if runtime_override else scale,
        store_softmax_stats=True,
        **({"page_size": length} if paged else {"packed_query": False}),
    )
    args = (q, key, value, offsets, pages, lengths) if paged else (q, key, value)
    runtime_kwargs = {}
    if runtime_override:
        runtime_kwargs = {"softmax_scale": scale}
    output = wrapper.run(*args, softmax_stats=stats, **runtime_kwargs)
    mla_q = torch.zeros((1, 1, heads, 576), device="cuda", dtype=q.dtype)
    mla_q[..., :128] = q.reshape(1, 1, heads, 128)
    cache = torch.zeros((1, length, 576), device="cuda", dtype=q.dtype)
    cache[..., 0], cache[..., 1], cache[..., 2] = 4096, 2, 1
    mla = BatchMLADecodePagedTSWrapper()
    mla.plan(
        device="cuda",
        batch_size=1,
        max_q_len=1,
        max_kv_len=length,
        page_size=length,
        num_qo_heads=heads,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        packed_query=False,
        q_dtype=q.dtype,
        kv_dtype=cache.dtype,
        out_dtype=q.dtype,
        mask_type="dense",
        store_softmax_stats=True,
    )
    mla_stats = torch.empty((1, 1, heads, 2), device="cuda", dtype=torch.float32)
    mla_output = mla.run(
        mla_q, cache, pages, lengths, softmax_scale=scale, softmax_stats=mla_stats
    )
    maxima = (
        torch.tensor([2**24, 2**24 + 2], device="cuda", dtype=torch.float32) * scale
    )
    for result, maximum in ((stats, maxima[0]), (mla_stats, maxima[1])):
        torch.testing.assert_close(
            result[..., 0], maximum.expand_as(result[..., 0]), rtol=0, atol=0
        )
        torch.testing.assert_close(
            result[..., 1],
            torch.full_like(result[..., 1], length),
            rtol=1e-6,
            atol=1e-4,
        )
    weight = (stats[..., 0].reshape(1, 1, heads) - mla_stats[..., 0]).exp()
    merged = (
        output.reshape(1, 1, heads, 128).float() * weight[..., None]
        + mla_output[..., :128].float()
    ) / (weight[..., None] + 1)
    expected = 1 / (1 + (maxima[0] - maxima[1]).exp())
    torch.testing.assert_close(
        merged[..., 2], expected.expand_as(merged[..., 2]), rtol=0, atol=1e-6
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(
            *args, softmax_stats=stats, out=output, validate=False, **runtime_kwargs
        )
    stats.fill_(torch.nan)
    graph.replay()
    torch.testing.assert_close(
        stats[..., 0], maxima[0].expand_as(stats[..., 0]), rtol=0, atol=0
    )
    torch.testing.assert_close(
        stats[..., 1], torch.full_like(stats[..., 1], length), rtol=1e-6, atol=1e-4
    )
    _GRAPH_OWNERS.append((graph, wrapper, args, output, stats, runtime_kwargs))


@_REQUIRES_PRIMTS_GPU
@pytest.mark.parametrize(
    "reduction",
    (
        "disabled",
        "gmem_reduction",
        "cluster_smem_reduction",
        "gmem_reduction_with_separate_kernel",
    ),
)
@pytest.mark.parametrize("raw_extra", (0, 2))
def test_attention_ts_decode_sink_original_scale(
    monkeypatch, record_property, reduction, raw_extra
):
    """Each final owner agrees with the natural-scale standalone reducer contract."""
    import cutlass
    from flashinfer.attention.prims_ts import BatchDecodePagedTSWrapper, decode
    from flashinfer.attention.prims_ts.kernels.fmha_decode.fmha_decode_config import (
        make_decode_config,
    )

    split = reduction != "disabled"
    config = make_decode_config(
        headdim=128,
        args={
            "use_keeps_mma_ab": False,
            "tile_size_q": 16,
            "store_softmax_stats": True,
        },
        seq_len_q=1,
        seq_len_kv=1024,
        batch_size=1,
        num_heads_q=16,
        num_heads_kv=1,
        q_dtype=cutlass.BFloat16,
        k_dtype=cutlass.BFloat16,
        v_dtype=cutlass.BFloat16,
        o_dtype=cutlass.BFloat16,
        qkv_layout="pagedKv",
        num_tokens_per_page=32,
        split_kv_mode=reduction,
        splits_kv=2 if split else 1,
        max_splits_kv=2 if split else 1,
        use_attention_sinks=True,
        mask_type="dense",
        auto_tuner=False,
    )
    assert config.use_split_kv == split
    assert config.use_cluster_smem_reduction == (reduction == "cluster_smem_reduction")
    assert config.use_separate_reduction_kernel == (
        reduction == "gmem_reduction_with_separate_kernel"
    )
    spec = decode._decode_launch_spec_from_config(
        config,
        batch_size=1,
        num_qo_heads=16,
        num_kv_heads=1,
        head_dim=128,
        seq_len_q=1,
        max_active_clusters=1,
    )
    monkeypatch.setattr(
        decode, "_resolve_decode_launch_spec", lambda *_args, **_kwargs: spec
    )
    query = torch.zeros((1, 16, 128), device="cuda", dtype=torch.bfloat16)
    key = torch.zeros((32, 1, 32, 128), device="cuda", dtype=query.dtype)
    query[..., 0], key[..., 0] = 4096, 4096
    query[..., 1], key[..., 1] = 1, raw_extra
    value = torch.full_like(key, 0.25)
    tables = torch.arange(32, device="cuda", dtype=torch.int32)[None]
    lengths = torch.full((1,), 1024, device="cuda", dtype=torch.int32)
    scale = 0.37
    maximum = (
        torch.tensor(2**24 + raw_extra, device="cuda", dtype=torch.float32) * scale
    )
    sinks = (maximum + math.log(1024)).expand(16).contiguous()
    expected_sum = 1024 + (sinks - maximum).exp()
    wrapper = BatchDecodePagedTSWrapper()
    wrapper.plan(
        "cuda",
        1,
        16,
        1,
        128,
        32,
        1024,
        q_dtype=query.dtype,
        store_softmax_stats=True,
        use_attention_sinks=True,
    )
    stats = torch.empty((1, 16, 2), device="cuda", dtype=torch.float32)
    output = wrapper.run(
        query,
        (key, value),
        lengths,
        tables,
        softmax_scale=scale,
        attention_sinks=sinks,
        softmax_stats=stats,
    )

    expected_output = (256 / expected_sum)[:, None].expand(16, 128)
    # Main's existing sink O/LSE path is intentionally unchanged. Its absolute
    # LSE arithmetic can lose low bits at these extremes independently of m/s.
    record_property(
        "unchanged_sink_output_max_abs_error",
        (output[0].float() - expected_output).abs().max().item(),
    )

    def check():
        torch.testing.assert_close(stats[0, :, 0], maximum.expand(16), rtol=0, atol=0)
        torch.testing.assert_close(stats[0, :, 1], expected_sum, rtol=1e-5, atol=0.01)
        assert torch.isfinite(output).all()

    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(
            query,
            (key, value),
            lengths,
            tables,
            softmax_scale=scale,
            attention_sinks=sinks,
            softmax_stats=stats,
            out=output,
            validate=False,
        )
    stats.fill_(torch.nan)
    graph.replay()
    check()
    _GRAPH_OWNERS.append(
        (graph, wrapper, query, key, value, lengths, tables, sinks, stats, output)
    )
