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
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest


_BENCHMARK = (
    Path(__file__).parents[3] / "benchmarks" / "bench_trtllm_gen_fused_moe_autotuner.py"
)


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


@pytest.fixture
def bench(monkeypatch):
    monkeypatch.syspath_prepend(str(_BENCHMARK.parent))
    return importlib.import_module(_BENCHMARK.stem)


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
def test_localized_benchmark_preserves_enable_pdl():
    tree = ast.parse(_BENCHMARK.read_text())
    localized_setup = _function(tree, "_localized_setup")
    popped_names = {
        element.value
        for node in ast.walk(localized_setup)
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Tuple)
        for element in node.iter.elts
        if isinstance(element, ast.Constant) and isinstance(element.value, str)
    }

    assert "enable_pdl" not in popped_names
    for name in (
        "bench_trtllm_gen_fused_moe_autotuner_bf16",
        "bench_trtllm_gen_fused_moe_autotuner_fp8",
        "bench_trtllm_gen_fused_moe_autotuner_fp4",
    ):
        parameters = {argument.arg for argument in _function(tree, name).args.args}
        assert "enable_pdl" in parameters


@pytest.mark.cpu_only
def test_localization_support_accepts_mixed_backends(bench, monkeypatch):
    monkeypatch.setattr(
        bench, "_require_cupti_for_localization", lambda localizations: None
    )

    bench._validate_localization_support(
        backends=["trtllm", "prims_ts"],
        quant_mode="BF16",
        activation_type=bench.ActivationType.Swiglu,
        localizations=["none", "tp_n_tp_n"],
    )


@pytest.mark.cpu_only
def test_localization_support_rejects_deepseek_fp8_block(bench):
    assert bench._op_family("Fp8-Block") == "fp8_deepseek"

    with pytest.raises(ValueError, match="DeepSeek FP8 block-scale layout"):
        bench._validate_localization_support(
            backends=["trtllm", "prims_ts"],
            quant_mode="Fp8-Block",
            activation_type=bench.ActivationType.Swiglu,
            localizations=["tp_n_tp_n"],
        )


@pytest.mark.cpu_only
def test_localization_support_rejects_non_gated_activation_first(bench):
    with pytest.raises(ValueError, match="only gated activations; got Relu2"):
        bench._validate_localization_support(
            backends=["prims_ts"],
            quant_mode="Fp8-Block",
            activation_type=bench.ActivationType.Relu2,
            localizations=["tp_n_tp_n"],
        )

    bench._validate_localization_support(
        backends=["prims_ts"],
        quant_mode="BF16",
        activation_type=bench.ActivationType.Relu2,
        localizations=["none"],
    )


@pytest.mark.cpu_only
def test_localized_setup_forwards_fp4_activation_parameters(bench, monkeypatch):
    monkeypatch.setitem(
        bench._LOCALIZED_OPS, ("prims_ts", "fp4"), lambda **kwargs: kwargs
    )
    monkeypatch.setattr(
        bench,
        "_localized_partition",
        lambda device, sm_split: SimpleNamespace(resources=object()),
    )
    activation_parameter = object()

    setup = bench._localized_setup(
        batch_size=1,
        backend="prims_ts",
        family="fp4",
        strategy="tp_n_tp_n",
        sm_split="backfill",
        common_fn_kwargs={
            "gemm1_alpha": activation_parameter,
            "gemm1_beta": activation_parameter,
            "gemm1_clamp_limit": activation_parameter,
        },
        input_kwargs={"hidden_states": SimpleNamespace(device="cuda:0")},
        shards=SimpleNamespace(
            fc1=(object(), object()),
            fc2=(object(), object()),
            fc1_scale=None,
            fc2_scale=None,
        ),
    )

    for name in ("gemm1_alpha", "gemm1_beta", "gemm1_clamp_limit"):
        assert setup.fn.keywords[name] is activation_parameter
