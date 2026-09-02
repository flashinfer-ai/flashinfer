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


def test_autotune_lookup_preserves_profile_replay_override():
    benchmark = (
        Path(__file__).parents[2]
        / "benchmarks"
        / "bench_trtllm_gen_fused_moe_autotuner.py"
    )
    tree = ast.parse(benchmark.read_text())
    run_benchmark = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_run_benchmark"
    )
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
