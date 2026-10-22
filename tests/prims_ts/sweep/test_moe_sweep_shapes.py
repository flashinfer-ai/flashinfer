# Copyright (c) 2026 by FlashInfer team.
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

from dataclasses import replace
from types import SimpleNamespace

import pytest

from flashinfer.prims_ts.moe.tools.sweep.engine import context_for_shapes
from flashinfer.prims_ts.moe.tools.sweep.measure import _activation_value_kwargs
from flashinfer.prims_ts.moe.tools.sweep.shapes import (
    ActivationParams,
    assert_modeled_shape_consistent,
    build_sweep_shapes,
    merge_duplicate_shapes,
    model_moe_shape,
    modeled_shape_consistency_errors,
    resolve_activation_params,
)


def test_activation_params_resolve_presets_and_explicit_overrides():
    assert resolve_activation_params("kimi-k3-nvfp4") == ActivationParams(
        gemm1_alpha=4.0,
        gemm1_beta=25.0,
    )
    assert resolve_activation_params(
        "gpt-oss-120b",
        has_bias=False,
        gemm1_alpha="none",
        gemm1_beta=2.0,
        gemm1_clamp_limit="none",
    ) == ActivationParams(has_bias=False, gemm1_beta=2.0)
    assert resolve_activation_params(
        None,
        has_bias=True,
        gemm1_alpha=1.25,
    ) == ActivationParams(has_bias=True, gemm1_alpha=1.25)


def test_context_carries_activation_params_only_for_fc1():
    shapes = [
        model_moe_shape(
            model="gpt-oss-120b",
            tp=4,
            ep=4,
            global_num_tokens=32,
        )
    ]
    kwargs = {
        "has_bias": True,
        "gemm1_alpha": 1.702,
        "gemm1_beta": 1.0,
        "gemm1_clamp_limit": 7.0,
    }

    fc1 = context_for_shapes(shapes, fc="fc1", sm_arch=107, **kwargs)
    assert fc1.has_bias
    assert (
        fc1.has_gemm1_alpha,
        fc1.has_gemm1_beta,
        fc1.has_gemm1_clamp_limit,
    ) == (True, True, True)
    assert _activation_value_kwargs(fc1) == {
        "gemm1_alpha_value": 1.702,
        "gemm1_beta_value": 1.0,
        "gemm1_clamp_limit_value": 7.0,
    }

    fc2 = context_for_shapes(shapes, fc="fc2", sm_arch=107, **kwargs)
    assert fc2.has_bias
    assert not any(
        (
            fc2.has_gemm1_alpha,
            fc2.has_gemm1_beta,
            fc2.has_gemm1_clamp_limit,
        )
    )
    assert _activation_value_kwargs(fc2) == {}


def test_context_rejects_activation_params_for_unsupported_activation():
    shapes = [
        model_moe_shape(
            dtype="nvfp4",
            hidden_size=2048,
            intermediate_size=768,
            num_experts=128,
            top_k=8,
            activation="Geglu",
            tp=2,
            ep=2,
            global_num_tokens=32,
        )
    ]

    with pytest.raises(
        ValueError, match="alpha/beta/clamp_limit require Swiglu or Situ"
    ):
        context_for_shapes(
            shapes,
            fc="fc1",
            sm_arch=107,
            gemm1_alpha=1.0,
        )


@pytest.mark.parametrize(
    ("ep", "timing_tokens", "expected"),
    [
        (4, 128, (8, 16, 32)),
        (4, 640, (32, 64, 128)),
        (8, 160, (16, 32, 64, 128)),
    ],
)
def test_dsv4_tile_windows_follow_actual_rows_per_expert(ep, timing_tokens, expected):
    shape = model_moe_shape(
        model="deepseek-v4-pro-nvfp4",
        tp=ep,
        ep=ep,
        global_num_tokens=timing_tokens * ep,
    )

    assert shape.timing_num_tokens == timing_tokens
    assert shape.selected_tile_ns == expected


def test_exact_tp_ep_pairs_do_not_expand_to_cross_product():
    shapes = build_sweep_shapes(
        model="deepseek-v4-pro-nvfp4",
        global_num_tokens=(32,),
        tp_ep=((2, 2), (4, 4), (8, 8)),
        merge=False,
    )

    assert [(shape.tp, shape.ep) for shape in shapes] == [
        (2, 2),
        (4, 4),
        (8, 8),
    ]


def test_deepseek_flash_tp4_rejects_only_the_unaligned_fc2_k():
    fc1_shapes = build_sweep_shapes(
        model="deepseek-v4.1-flash",
        global_num_tokens=(32,),
        tp_ep=((4, 4),),
        fc="fc1",
        merge=False,
    )
    assert len(fc1_shapes) == 1
    assert fc1_shapes[0].fc1_k == 5120

    with pytest.raises(
        ValueError,
        match=r"deepseek-v4\.1-flash TP=4/EP=4 fc2 K=576.*multiple of 128",
    ):
        build_sweep_shapes(
            model="deepseek-v4.1-flash",
            global_num_tokens=(32,),
            tp_ep=((4, 4),),
            fc="fc2",
            merge=False,
        )


def test_deepseek_flash_tp2_has_aligned_fc1_and_fc2_k():
    shapes = build_sweep_shapes(
        model="deepseek-v4.1-flash",
        global_num_tokens=(32,),
        tp_ep=((2, 2),),
        merge=False,
    )
    assert len(shapes) == 1
    assert (shapes[0].fc1_k, shapes[0].fc2_k) == (5120, 1152)


@pytest.mark.parametrize(
    ("model", "tp_ep"),
    (
        ("deepseek-v3-nvfp4", (8, 1)),
        ("deepseek-v4-pro-nvfp4", (2, 2)),
        ("deepseek-v4.1-flash-mxfp8", (2, 2)),
        ("gpt-oss-120b", (8, 1)),
        ("gpt-oss-120b-mxfp8", (8, 1)),
        ("kimi-k3-nvfp4", (8, 8)),
        ("minimax-m3-mxfp8", (8, 8)),
        ("nemotron-3.5-lightning-nvfp4", (1, 8)),
        ("nemotron-3-ultra-nvfp4", (4, 1)),
        ("qwen3-30b-a3b", (2, 2)),
    ),
)
def test_block_scaled_presets_have_aligned_fc_k(model, tp_ep):
    shapes = build_sweep_shapes(
        model=model,
        global_num_tokens=(32,),
        tp_ep=(tp_ep,),
        merge=False,
    )
    assert len(shapes) == 1


def test_fc_shapes_match_unpartitioned_runner_geometry():
    shape = model_moe_shape(
        model="deepseek-v4-pro-nvfp4",
        tp=4,
        ep=4,
        global_num_tokens=17,
    )

    assert (shape.fc1_n, shape.fc1_k) == (1536, 7168)
    assert (shape.fc2_n, shape.fc2_k) == (7168, 768)
    assert shape.activation == "Swiglu"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"tp": 5, "ep": 1}, "divisible by TP"),
        ({"tp": 1, "ep": 5}, "divisible by EP"),
        ({"tp": 0, "ep": 1}, "tp must be positive"),
        ({"tp": 1, "ep": 0}, "ep must be positive"),
    ),
)
def test_tp_ep_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        model_moe_shape(
            model="deepseek-v4-pro-nvfp4",
            global_num_tokens=32,
            **kwargs,
        )


def test_launch_equivalent_duplicates_merge_with_exact_weight_sum():
    first = model_moe_shape(
        model="deepseek-v4-pro-nvfp4",
        tp=4,
        ep=4,
        global_num_tokens=7,
        weight=1.25,
    )
    second = model_moe_shape(
        model="deepseek-v4-pro-nvfp4",
        tp=4,
        ep=4,
        global_num_tokens=8,
        weight=2.75,
    )
    assert first.launch_key() == second.launch_key()

    merged = merge_duplicate_shapes((first, second))

    assert len(merged) == 1
    assert merged[0].weight == 4.0
    assert [origin.global_num_tokens for origin in merged[0].origins] == [7, 8]
    assert [origin.weight for origin in merged[0].origins] == [1.25, 2.75]
    assert_modeled_shape_consistent(merged[0])


def test_merge_key_keeps_tile_selection_and_local_launch_shape():
    small = model_moe_shape(
        model="deepseek-v4-pro-nvfp4",
        tp=4,
        ep=4,
        global_num_tokens=1,
    )
    large = model_moe_shape(
        model="deepseek-v4-pro-nvfp4",
        tp=4,
        ep=4,
        global_num_tokens=8192,
    )

    assert small.launch_key() != large.launch_key()
    assert len(merge_duplicate_shapes((small, large))) == 2


def test_routing_and_seed_are_part_of_launch_identity():
    balanced = model_moe_shape(
        model="deepseek-v4-pro-nvfp4",
        tp=4,
        ep=4,
        global_num_tokens=1024,
    )
    uniform = model_moe_shape(
        model="deepseek-v4-pro-nvfp4",
        tp=4,
        ep=4,
        global_num_tokens=1024,
        routing="uniform",
        routing_seed=7,
    )
    other_seed = replace(uniform, routing_seed=8)

    assert balanced.launch_key("fc1") != uniform.launch_key("fc1")
    assert uniform.launch_key("fc1") != other_seed.launch_key("fc1")
    assert len(merge_duplicate_shapes((balanced, uniform, other_seed), fc="fc1")) == 3


def test_default_search_and_final_token_sets_are_distinct():
    from flashinfer.prims_ts.moe.tools.sweep.__main__ import (
        _final_tokens,
        _tokens,
    )

    args = SimpleNamespace(tokens=None, final_tokens=None)
    search_tokens = _tokens(args)
    final_tokens = _final_tokens(args)

    assert len(search_tokens) == 16
    assert search_tokens[-2:] == (12288, 16384)
    assert final_tokens[-2:] == (12288, 16384)
    assert search_tokens != final_tokens


def test_host_consistency_check_reports_corrupt_derived_field():
    shape = model_moe_shape(
        model="kimi-k3",
        tp=8,
        ep=16,
        global_num_tokens=1025,
    )
    assert modeled_shape_consistency_errors(shape) == ()

    corrupt = replace(shape, timing_num_tokens=shape.timing_num_tokens + 1)
    errors = modeled_shape_consistency_errors(corrupt)
    assert len(errors) == 1
    assert errors[0].startswith("timing_num_tokens=")
    with pytest.raises(ValueError, match="timing_num_tokens"):
        assert_modeled_shape_consistent(corrupt)
