"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""cuDNN structured tactics ``(engine_id, knob_items)`` must survive plans that
carry no knobs.

cudnn-frontend's unified plan list mixes backend plans (knobs: a dict) with
python engines (FROST, opt-in via ``CUDNN_FRONTEND_ENABLE_FROST_ENGINES``),
whose ``get_engine_and_knobs_at_index`` answered ``(engine_id, None)`` before
cudnn-frontend PR #1024 normalized it to ``{}``. The enumeration must accept
both, or every cuDNN GEMM path and the autotuner crash the moment frost is
opted in. No GPU needed: the graph is a stub."""

import pytest

pytest.importorskip("cudnn")

from flashinfer.gemm.gemm_base import (  # noqa: E402
    _cudnn_graph_engine_knob_tactics,
    _get_cudnn_plan_index_for_tactic,
)


class _StubGraph:
    """Just the two calls the tactic code makes."""

    def __init__(self, plans):
        self._plans = plans

    def get_execution_plan_count(self):
        return len(self._plans)

    def get_engine_and_knobs_at_index(self, i):
        return self._plans[i]


FROST_GEMM_ID = 20400  # python-engine id block; no cuDNN knob dict


def test_enumeration_accepts_plans_without_knobs():
    import cudnn

    kt = cudnn.knob_type
    graph = _StubGraph(
        [
            (FROST_GEMM_ID, None),  # pre-#1024 python plan
            (FROST_GEMM_ID, {}),  # post-#1024 python plan
            (0, {kt.TILE_M: 128, kt.TILE_N: 256}),  # backend plan
        ]
    )
    tactics = _cudnn_graph_engine_knob_tactics(graph)
    assert tactics == [
        (FROST_GEMM_ID, ()),
        (FROST_GEMM_ID, ()),
        (0, ((int(kt.TILE_M), 128), (int(kt.TILE_N), 256))),
    ]
    # knob items are sorted by knob id so the same plan hashes the same
    graph2 = _StubGraph([(0, {kt.TILE_N: 256, kt.TILE_M: 128})])
    assert _cudnn_graph_engine_knob_tactics(graph2) == [tactics[2]]


def test_plan_lookup_matches_knobless_tactic():
    import cudnn

    kt = cudnn.knob_type
    graph = _StubGraph(
        [(0, {kt.TILE_M: 128}), (FROST_GEMM_ID, None), (FROST_GEMM_ID, {})]
    )
    assert _get_cudnn_plan_index_for_tactic(graph, (FROST_GEMM_ID, ())) == 1
    assert _get_cudnn_plan_index_for_tactic(graph, (0, ((int(kt.TILE_M), 128),))) == 0
