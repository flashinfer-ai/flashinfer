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

"""FMHA statistics preserve sums and raw maxima before exponent scaling."""

import ast
import inspect
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from flashinfer.cute_dsl.availability import is_cute_dsl_arch_supported


_REQUIRES_PRIMTS_GPU = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7))
    or not is_cute_dsl_arch_supported(
        *torch.cuda.get_device_capability(), native_only=True
    ),
    reason="PrimTS attention requires native DSL support for SM100, SM103, or SM107",
)


def test_attention_ts_fmha_large_logits_lose_denominator_in_lse():
    """The regression input is finite BF16, but absolute FP32 LSE loses bits."""
    query = torch.zeros(128, dtype=torch.bfloat16)
    query[0] = 4096.0
    scores = query.float().repeat(512, 1) @ query.float()
    maximum = scores.max()
    assert maximum.item() == 2**24
    max_log2 = maximum * math.log2(math.e)
    partial_sums = torch.tensor([256.0, 256.0])
    partial_maxima = max_log2.repeat(2)
    partial_lse = partial_maxima + partial_sums.log2()
    # Match the existing LSE/O merge's rounded absolute addition.
    merged_lse = (
        partial_lse.max() + torch.exp2(partial_lse - partial_lse.max()).sum().log2()
    )
    assert torch.exp2(merged_lse - max_log2).item() == 1024.0
    actual_sum = (
        partial_sums * torch.exp2(partial_maxima - partial_maxima.max())
    ).sum()
    assert actual_sum.item() == 512.0


def test_attention_ts_fmha_large_unequal_logits_lose_gap_when_prescaled():
    query = torch.tensor([4096.0, 1.0], dtype=torch.bfloat16)
    keys = torch.tensor([[4096.0, 0.0], [4096.0, 2.0]], dtype=torch.bfloat16)
    raw = keys.float() @ query.float()
    torch.testing.assert_close(
        raw, torch.tensor([2**24, 2**24 + 2]).float(), rtol=0, atol=0
    )
    partial_sums = torch.tensor([256.0, 256.0])
    prescaled_maxima = raw * math.log2(math.e)
    incorrect = (
        partial_sums * torch.exp2(prescaled_maxima - prescaled_maxima.max())
    ).sum()
    assert incorrect.item() == 320.0
    actual = (partial_sums * torch.exp2((raw - raw.max()) * math.log2(math.e))).sum()
    expected = (partial_sums * torch.exp(raw - raw.max())).sum()
    torch.testing.assert_close(actual, expected)
    assert abs(actual.item() - incorrect.item()) > 29.0


@pytest.mark.parametrize("probability_scale", (1.0, 448.0))
def test_attention_ts_fmha_max_probability_uses_raw_difference(probability_scale):
    score = torch.tensor(float(2**24 + 2), dtype=torch.float32)
    scale = torch.tensor(0.5 * math.log2(math.e), dtype=torch.float32)
    bias = torch.tensor(math.log2(probability_scale), dtype=torch.float32)
    # FP64 intermediates emulate fused FP32 multiply-add with one final rounding.
    offset = (-score.double() * scale.double() + bias.double()).float()
    exponent = (score.double() * scale.double() + offset.double()).float()
    assert abs(torch.exp2(exponent).item() / probability_scale - 1.0) > 0.1
    stable = torch.exp2((score - score) * scale + bias)
    torch.testing.assert_close(stable, torch.tensor(probability_scale))


@pytest.mark.parametrize(
    "tile_q,keeps", ((8, False), (16, False), (32, False), (64, True), (128, True))
)
@pytest.mark.parametrize("enabled", (False, True))
def test_attention_ts_fmha_stats_centering_uses_actual_row_groups(
    tile_q, keeps, enabled
):
    """Execute the source's scalar indexing on CPU, not a DSL-lowering proof."""
    pytest.importorskip("cutlass", minversion="4.8.0")
    from flashinfer.attention.prims_ts.kernels.fmha_decode.fmha_decode_resources.smem_p import (
        SmemPResource,
    )

    module = ast.parse(Path(inspect.getfile(SmemPResource)).read_text())
    cls = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "SmemPResource"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_center_stats_scores"
    )
    method.decorator_list = []
    method.returns = None
    for arg in method.args.args:
        arg.annotation = None
    namespace = {
        "cutlass": SimpleNamespace(const_expr=bool, range_constexpr=range),
        "Float32": float,
        "_neg_max_f32": lambda: -torch.finfo(torch.float32).max,
    }
    exec(
        compile(
            ast.Module(body=[method], type_ignores=[]), "<centering source>", "exec"
        ),
        namespace,
    )

    class Values(list):
        @property
        def shape(self):
            return (len(self),)

    groups = 1 if keeps else tile_q // 4
    maxima = Values(float(32 + 4 * group) for group in range(groups))
    if not keeps:
        maxima[-1] = -torch.finfo(torch.float32).max
    scores = Values(float(index) for index in range(tile_q if not keeps else 128))
    before = scores.copy()
    namespace["_center_stats_scores"](
        SimpleNamespace(
            cfg=SimpleNamespace(
                store_softmax_stats=enabled, use_keeps_mma_ab=keeps, tile_size_q=tile_q
            )
        ),
        scores,
        maxima,
    )
    for index, actual in enumerate(scores):
        group = 0 if keeps else (index % (tile_q // 2)) // 4 * 2 + index % 2
        anchor = (
            maxima[group]
            if enabled and maxima[group] != -torch.finfo(torch.float32).max
            else 0.0
        )
        assert actual == before[index] - anchor


@pytest.mark.parametrize(
    "family,reduction,splits,use_sink",
    [
        ("swaps", "direct", 1, False),
        ("swaps", "fused", 2, False),
        ("swaps", "cluster", 2, False),
        ("swaps", "serial", 2, False),
        ("swaps", "parallel", 2, False),
        ("swaps", "parallel", 8, False),
        ("keeps", "serial", 2, False),
        ("keeps", "parallel", 2, False),
        ("swaps", "serial", 2, True),
        ("swaps", "parallel", 8, True),
    ],
)
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_fmha_stats_large_identical_logits(
    monkeypatch, family, reduction, splits, use_sink
):
    _check_large_logits(monkeypatch, family, reduction, splits, use_sink)


@pytest.mark.parametrize(
    "family,reduction,splits,use_sink",
    [
        ("swaps", "serial", 2, False),
        ("swaps", "parallel", 2, False),
        ("swaps", "parallel", 8, False),
        ("keeps", "serial", 2, False),
        ("keeps", "parallel", 2, False),
        ("swaps", "parallel", 8, True),
    ],
)
@pytest.mark.parametrize("scale", (0.5, 1.0))
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_fmha_stats_large_unequal_logits(
    monkeypatch, family, reduction, splits, use_sink, scale
):
    _check_large_logits(
        monkeypatch, family, reduction, splits, use_sink, logit_gap=2.0, scale=scale
    )


@pytest.mark.parametrize("family", ("swaps", "keeps"))
@pytest.mark.parametrize(
    "qk_component,store_softmax_stats", ((4096.0, True), (4.0, False))
)
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_fmha_direct_output_and_stats_precision(
    monkeypatch, family, qk_component, store_softmax_stats
):
    _check_large_logits(
        monkeypatch,
        family,
        "direct",
        1,
        False,
        logit_gap=2.0,
        scale=0.5,
        qk_component=qk_component,
        store_softmax_stats=store_softmax_stats,
        check_output=True,
    )


def _check_large_logits(
    monkeypatch,
    family,
    reduction,
    splits,
    use_sink,
    *,
    logit_gap=0.0,
    scale=1.0,
    qk_component=4096.0,
    store_softmax_stats=True,
    check_output=False,
):
    cutlass = pytest.importorskip("cutlass", minversion="4.7.0")
    from flashinfer.attention.prims_ts import BatchDecodePagedTSWrapper, decode
    from flashinfer.attention.prims_ts.kernels.fmha_decode import fmha_decode_config

    build = decode.build_and_load_attention_kernel

    def build_for_topology(kind_family, kind, spec, options, compile_fn, *, extra=()):
        return build(
            kind_family,
            kind,
            spec,
            options,
            compile_fn,
            extra=(
                *extra,
                "stats-precision-test",
                family,
                reduction,
                splits,
                store_softmax_stats,
            ),
        )

    monkeypatch.setattr(decode, "build_and_load_attention_kernel", build_for_topology)

    if reduction == "serial":
        for name in (
            "use_parallel_separate_reduction",
            "use_parallel_separate_reduction_pdl",
        ):
            monkeypatch.setattr(
                fmha_decode_config.FmhaDecodeConfig, name, property(lambda _self: False)
            )
    keeps = family == "keeps"
    heads = 64 if keeps else 16
    kv_len = max(512, splits * 256)
    page_size = 32
    modes = {
        "direct": "disabled",
        "fused": "gmem_reduction",
        "cluster": "cluster_smem_reduction",
        "serial": "gmem_reduction_with_separate_kernel",
        "parallel": "gmem_reduction_with_separate_kernel",
    }
    config = fmha_decode_config.make_decode_config(
        headdim=128,
        args={
            "use_keeps_mma_ab": keeps,
            "groups_tokens_heads_q": not keeps,
            "tile_size_q": 64 if keeps else 16,
            "tile_size_kv": 128,
            "use_persistent_scheduler": False,
            "store_softmax_stats": store_softmax_stats,
        },
        seq_len_q=1,
        seq_len_kv=kv_len,
        batch_size=1,
        num_heads_q=heads,
        num_heads_kv=1,
        q_dtype=cutlass.BFloat16,
        k_dtype=cutlass.BFloat16,
        v_dtype=cutlass.BFloat16,
        o_dtype=cutlass.BFloat16,
        qkv_layout="pagedKv",
        num_tokens_per_page=page_size,
        split_kv_mode=modes[reduction],
        splits_kv=splits,
        max_splits_kv=splits,
        min_loop_iters_per_split=1,
        use_attention_sinks=use_sink,
        mask_type="dense",
        auto_tuner=False,
    )
    assert config.splits_kv == splits
    assert config.use_separate_reduction_kernel == (reduction in ("serial", "parallel"))
    if config.use_separate_reduction_kernel:
        assert config.use_parallel_separate_reduction == (reduction == "parallel")
    spec = decode._decode_launch_spec_from_config(
        config,
        batch_size=1,
        num_qo_heads=heads,
        num_kv_heads=1,
        head_dim=128,
        seq_len_q=1,
        max_active_clusters=1,
    )
    monkeypatch.setattr(
        decode, "_resolve_decode_launch_spec", lambda *_args, **_kwargs: spec
    )
    decode._get_compiled_decode.cache_clear()
    try:
        query = torch.zeros((1, heads, 128), dtype=torch.bfloat16, device="cuda")
        key = torch.zeros(
            (kv_len // page_size, 1, page_size, 128),
            dtype=torch.bfloat16,
            device="cuda",
        )
        query[..., 0] = qk_component
        key[..., 0] = qk_component
        query[..., 1] = 1.0
        key[kv_len // page_size // 2 :, ..., 1] = logit_gap
        value = torch.ones_like(key)
        value[kv_len // page_size // 2 :] = 3.0
        block_tables = torch.arange(
            kv_len // page_size, dtype=torch.int32, device="cuda"
        )[None]
        seq_lens = torch.tensor([kv_len], dtype=torch.int32, device="cuda")
        wrapper = BatchDecodePagedTSWrapper()
        wrapper.plan(
            "cuda",
            1,
            heads,
            1,
            128,
            page_size,
            kv_len,
            q_dtype=torch.bfloat16,
            out_dtype=torch.bfloat16,
            store_softmax_stats=store_softmax_stats,
            use_attention_sinks=use_sink,
        )
        sinks = None
        if use_sink:
            sinks = torch.full((heads,), qk_component**2 * scale, device="cuda")
        stats = (
            torch.full((1, heads, 2), torch.nan, device="cuda")
            if store_softmax_stats
            else None
        )
        output = wrapper.run(
            query,
            (key, value),
            seq_lens,
            block_tables,
            softmax_scale=scale,
            softmax_stats=stats,
            attention_sinks=sinks,
        )
        low_weight = math.exp(-logit_gap * scale)
        denominator = kv_len // 2 * (1.0 + low_weight)
        # The sink equals the lower token logit and contributes only once.
        if use_sink:
            denominator += low_weight
        if stats is not None:
            torch.testing.assert_close(
                stats[..., 0],
                torch.full_like(stats[..., 0], (qk_component**2 + logit_gap) * scale),
                rtol=0,
                atol=2,
            )
            torch.testing.assert_close(
                stats[..., 1],
                torch.full_like(stats[..., 1], denominator),
                rtol=1e-5,
                atol=1e-4,
            )
        if check_output:
            expected_output = kv_len // 2 * (low_weight + 3.0) / denominator
            torch.testing.assert_close(
                output.float(),
                torch.full_like(output.float(), expected_output),
                rtol=0.01,
                atol=0.01,
            )
    finally:
        decode._get_compiled_decode.cache_clear()
