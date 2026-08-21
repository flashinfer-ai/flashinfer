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


def test_clustered_persistent_normalization_keeps_invalid_static_grid():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        TileScheduler,
        make_config,
    )
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        _make_token_layout,
        _normalize_runtime_scheduler,
    )

    cfg = make_config(
        tile_scheduler=int(TileScheduler.PERSISTENT),
        cluster_m=2,
        tile_m=128,
        tile_n=64,
        use_early_exit=0,
    )
    token_layout = _make_token_layout(
        num_tokens=256,
        num_experts=2,
        top_k=1,
        tile_size=cfg.tile_n,
        cluster_dim_in_token=1,
    )

    normalized = _normalize_runtime_scheduler(
        cfg,
        token_layout,
        out_hidden=128,
        early_exit_max_token_ctas=0,
    )

    assert normalized.is_persistent


def test_clustered_persistent_normalization_keeps_max_tmem_overlap():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        TileScheduler,
        make_config,
    )
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        _make_token_layout,
        _normalize_runtime_scheduler,
    )

    cfg = make_config(
        tile_scheduler=int(TileScheduler.PERSISTENT),
        cluster_m=2,
        tile_m=128,
        tile_n=256,
        use_early_exit=0,
        use_max_tmem_overlap=1,
    )
    token_layout = _make_token_layout(
        num_tokens=512,
        num_experts=2,
        top_k=1,
        tile_size=cfg.tile_n,
        cluster_dim_in_token=1,
    )

    normalized = _normalize_runtime_scheduler(
        cfg,
        token_layout,
        out_hidden=8192,
        early_exit_max_token_ctas=0,
    )

    assert normalized.is_persistent


def test_clustered_persistent_normalization_keeps_persistent_grid():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        TileScheduler,
        make_config,
    )
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        _make_token_layout,
        _normalize_runtime_scheduler,
    )

    cfg = make_config(
        tile_scheduler=int(TileScheduler.PERSISTENT),
        cluster_m=2,
        tile_m=128,
        tile_n=128,
        use_early_exit=0,
    )
    token_layout = _make_token_layout(
        num_tokens=1,
        num_experts=1,
        top_k=1,
        tile_size=cfg.tile_n,
        cluster_dim_in_token=1,
    )

    normalized = _normalize_runtime_scheduler(
        cfg,
        token_layout,
        out_hidden=8192,
        early_exit_max_token_ctas=0,
    )

    assert normalized.tile_scheduler == int(TileScheduler.PERSISTENT)


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


def test_single_stage_workid_allows_c_scratch_ab_alias():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        TileScheduler,
        make_config,
    )

    cfg = make_config(
        tile_scheduler=int(TileScheduler.PERSISTENT),
        num_stages_workid=1,
    )

    assert cfg.aliases_c_scratch_with_ab
