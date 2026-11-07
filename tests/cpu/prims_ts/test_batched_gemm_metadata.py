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

"""Host-side metadata tests for BatchedGemm TS runners."""

from types import SimpleNamespace

import pytest


def _experts_by_token(token_layout, num_tokens):
    experts = [[] for _ in range(num_tokens)]
    for expert_idx, token_idx in zip(
        token_layout.expanded_to_expert,
        token_layout.expanded_to_token,
        strict=True,
    ):
        if expert_idx >= 0:
            experts[token_idx].append(expert_idx)
    return experts


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "routing",
    ("balanced_round_robin", "uniform", "zipf:0.8"),
)
def test_token_layout_routing_is_deterministic_distinct_and_complete(routing):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        _make_token_layout,
    )

    kwargs = dict(
        num_tokens=37,
        num_experts=11,
        top_k=4,
        tile_size=8,
        cluster_dim_in_token=2,
        routing=routing,
        routing_seed=1234,
    )
    first = _make_token_layout(**kwargs)
    second = _make_token_layout(**kwargs)

    assert first == second
    experts_by_token = _experts_by_token(first, kwargs["num_tokens"])
    assert all(len(experts) == kwargs["top_k"] for experts in experts_by_token)
    assert all(len(set(experts)) == kwargs["top_k"] for experts in experts_by_token)
    assert sum(first.tokens_per_expert) == kwargs["num_tokens"] * kwargs["top_k"]


@pytest.mark.cpu_only
@pytest.mark.parametrize("routing", ("uniform", "zipf:0.8"))
def test_stochastic_token_layout_routing_seed_changes_assignments(routing):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        _make_token_layout,
    )

    common = dict(
        num_tokens=37,
        num_experts=11,
        top_k=4,
        tile_size=8,
        cluster_dim_in_token=2,
        routing=routing,
    )
    first = _make_token_layout(**common, routing_seed=1)
    second = _make_token_layout(**common, routing_seed=2)

    assert _experts_by_token(first, common["num_tokens"]) != _experts_by_token(
        second, common["num_tokens"]
    )


@pytest.mark.cpu_only
def test_cached_token_layout_does_not_share_mutable_lists():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        _make_token_layout,
    )

    kwargs = dict(
        num_tokens=7,
        num_experts=4,
        top_k=2,
        tile_size=8,
        cluster_dim_in_token=1,
        routing="uniform",
        routing_seed=9,
    )
    first = _make_token_layout(**kwargs)
    expected = list(first.tokens_per_expert)
    first.tokens_per_expert[0] = -1

    assert _make_token_layout(**kwargs).tokens_per_expert == expected


@pytest.mark.cpu_only
def test_reference_check_diagnostics_preserve_bool_compatibility():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        ReferenceCheckDiagnostics,
    )

    passed = ReferenceCheckDiagnostics(
        passed=True,
        max_abs=0.25,
        mean_abs=0.05,
        max_rel=0.1,
        threshold=0.5,
    )
    failed = ReferenceCheckDiagnostics(
        passed=False,
        nan_count=2,
        inf_count=1,
        reason="non-finite output",
    )

    assert passed
    assert not failed
    assert failed.nan_count == 2
    assert failed.inf_count == 1
    assert failed.reason == "non-finite output"


@pytest.mark.cpu_only
def test_cli_accepts_stochastic_routing_options():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import build_arg_parser

    args = build_arg_parser().parse_args(
        [
            "--routing",
            "zipf:0.8",
            "--routing-seed",
            "123",
        ]
    )

    assert args.routing == "zipf:0.8"
    assert args.routing_seed == 123


@pytest.mark.cpu_only
def test_partial_route_map_padding_keeps_trt_absolute_limits():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        _launch_metadata_lists,
        _make_token_layout,
    )

    token_layout = _make_token_layout(
        num_tokens=128,
        num_experts=2,
        top_k=1,
        tile_size=128,
        cluster_dim_in_token=1,
    )
    cfg = SimpleNamespace(
        use_early_exit=0,
        is_persistent=False,
    )

    _, mn_limit, route_map = _launch_metadata_lists(
        cfg,
        token_layout,
        early_exit_max_token_ctas=0,
    )
    padded_routes = [
        route_map[idx]
        for idx, expert_idx in enumerate(token_layout.expanded_to_expert)
        if expert_idx < 0
    ]

    assert mn_limit == [64, 192]
    assert padded_routes
    assert set(padded_routes) == {0}


@pytest.mark.cpu_only
def test_early_exit_route_map_padding_is_not_semantic_metadata():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        _launch_metadata_lists,
        _make_token_layout,
    )

    token_layout = _make_token_layout(
        num_tokens=128,
        num_experts=2,
        top_k=1,
        tile_size=128,
        cluster_dim_in_token=1,
    )
    cfg = SimpleNamespace(
        use_early_exit=1,
        is_persistent=True,
    )

    _, _, route_map = _launch_metadata_lists(
        cfg,
        token_layout,
        early_exit_max_token_ctas=4,
    )
    extra_routes = route_map[token_layout.total_padded_tokens :]

    assert len(route_map) == 4 * token_layout.tile_size
    assert extra_routes
    assert set(extra_routes) == {0}


@pytest.mark.cpu_only
def test_mixed_cga_routing_metadata_uses_cga_scoped_tile_token():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        _make_token_layout,
    )

    token_layout = _make_token_layout(
        num_tokens=2,
        num_experts=2,
        top_k=1,
        tile_size=128,
        cluster_dim_in_token=4,
    )

    assert token_layout.cga_tile_size == 128 * 4
    assert token_layout.num_token_tiles == 8
    assert token_layout.num_token_cgas == 2
    assert token_layout.tile_idx == [0, 1]
    assert token_layout.mn_limit == [1, 513]


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("num_tokens", "requested_extent", "expected_extent"),
    (
        (1, 8, 8),
        (512, 1, 4),
    ),
)
def test_early_exit_launch_extent_covers_active_token_tiles(
    num_tokens, requested_extent, expected_extent
):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        _make_token_layout,
        _normalize_early_exit_launch_extent,
    )

    cfg = SimpleNamespace(use_early_exit=1)
    token_layout = _make_token_layout(
        num_tokens=num_tokens,
        num_experts=2,
        top_k=1,
        tile_size=128,
        cluster_dim_in_token=1,
    )

    extent = _normalize_early_exit_launch_extent(
        cfg,
        token_layout,
        early_exit_max_token_ctas=requested_extent,
    )

    assert extent == expected_extent


@pytest.mark.cpu_only
def test_disabled_early_exit_keeps_requested_launch_extent():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        _make_token_layout,
        _normalize_early_exit_launch_extent,
    )

    cfg = SimpleNamespace(use_early_exit=0)
    token_layout = _make_token_layout(
        num_tokens=512,
        num_experts=2,
        top_k=1,
        tile_size=128,
        cluster_dim_in_token=1,
    )

    extent = _normalize_early_exit_launch_extent(
        cfg,
        token_layout,
        early_exit_max_token_ctas=3,
    )

    assert extent == 3


@pytest.mark.cpu_only
def test_persistent_multistage_workid_disables_c_scratch_ab_alias():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        TileScheduler,
        make_config,
    )

    cfg = make_config(
        tile_scheduler=int(TileScheduler.PERSISTENT),
        num_stages_workid=3,
    )

    assert not cfg.aliases_c_scratch_with_ab


@pytest.mark.cpu_only
def test_single_stage_workid_keeps_c_scratch_disjoint_from_ab():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        TileScheduler,
        make_config,
    )

    cfg = make_config(
        tile_scheduler=int(TileScheduler.PERSISTENT),
        num_stages_workid=1,
    )

    assert not cfg.aliases_c_scratch_with_ab
