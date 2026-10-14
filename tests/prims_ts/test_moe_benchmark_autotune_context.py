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

"""Regressions for the PrimsTS/TRT-LLM Gen qualification driver."""

import ast
from pathlib import Path


_BENCHMARK = (
    Path(__file__).parents[2] / "benchmarks" / "bench_trtllm_gen_fused_moe_autotuner.py"
)


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_autotune_lookup_preserves_profile_replay_override():
    tree = ast.parse(_BENCHMARK.read_text())
    run_benchmark = _function(tree, "_run_benchmark")
    lookup_contexts = [
        item.context_expr
        for node in ast.walk(run_benchmark)
        if isinstance(node, ast.With)
        for item in node.items
        if isinstance(item.context_expr, ast.Call)
        and isinstance(item.context_expr.func, ast.Name)
        and item.context_expr.func.id == "autotune"
        and item.context_expr.args
        and isinstance(item.context_expr.args[0], ast.Constant)
        and item.context_expr.args[0].value is False
    ]

    assert len(lookup_contexts) == 1
    overrides = {keyword.arg for keyword in lookup_contexts[0].keywords}
    assert overrides >= {"tuning_buckets", "cuda_graph_profile_replays"}


def test_bf16_benchmark_forwards_profile_replays_and_uses_situ_gating():
    tree = ast.parse(_BENCHMARK.read_text())
    benchmark = _function(tree, "bench_trtllm_gen_fused_moe_autotuner_bf16")
    run_call = next(
        node
        for node in ast.walk(benchmark)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_run_benchmark"
    )
    assert "cuda_graph_profile_replays" in {
        keyword.arg for keyword in run_call.keywords
    }

    is_gated = next(
        node
        for node in ast.walk(benchmark)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "is_gated"
            for target in node.targets
        )
    )
    assert ast.unparse(is_gated.value) == "ActivationType(activation_type).is_gated"


def test_fp4_benchmark_cli_clamp_takes_precedence_over_model_default():
    tree = ast.parse(_BENCHMARK.read_text())
    benchmark = _function(tree, "bench_trtllm_gen_fused_moe_autotuner_fp4")
    effective_clamp = next(
        node
        for node in ast.walk(benchmark)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "effective_clamp_limit"
            for target in node.targets
        )
    )
    assert ast.unparse(effective_clamp.value) == (
        "gemm1_clamp_limit if gemm1_clamp_limit is not None else swiglu_limit"
    )

    clamp_keyword = next(
        keyword
        for node in ast.walk(benchmark)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "gemm1_clamp_limit"
        and isinstance(keyword.value, ast.Name)
        and keyword.value.id == "gemm1_clamp_limit_tensor"
    )
    assert clamp_keyword.value.id == "gemm1_clamp_limit_tensor"
