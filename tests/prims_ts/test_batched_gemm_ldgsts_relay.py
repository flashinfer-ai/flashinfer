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

"""Small relay regressions for separate SF copies and swap-AB loaders."""

import pytest

from flashinfer.prims_ts.batched_gemm import batched_gemm_run as runner
from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
    BatchMode,
    RouteImpl,
    SfLayout,
    make_config,
)
from tests.prims_ts.test_batched_gemm_rubin import RUBIN_ROUTE_ACT_FP4

pytestmark = [pytest.mark.arch_rubin, pytest.mark.timeout(180)]


@pytest.mark.parametrize(
    "swap,route_a,fused_copy,fused_loads,tile_n,problem_k",
    [
        pytest.param(
            False, RouteImpl.LDGSTS, 0, 0, 128, 1792, id="a-ldgsts-separate-sf"
        ),
        pytest.param(False, RouteImpl.TMA, 0, 0, 128, 1792, id="a-tma-separate-sf"),
        pytest.param(
            True, RouteImpl.LDGSTS, 0, 0, 128, 1792, id="b-ldgsts-combined-sf"
        ),
        pytest.param(True, RouteImpl.TMA, 0, 0, 128, 1792, id="b-tma-combined-sf"),
        pytest.param(True, RouteImpl.LDGSTS, 1, 0, 128, 1792, id="b-ldgsts-fused-copy"),
        pytest.param(True, RouteImpl.TMA, 1, 0, 128, 1792, id="b-tma-fused-copy"),
        pytest.param(
            True, RouteImpl.LDGSTS, 1, 1, 128, 1792, id="b-ldgsts-fused-loads"
        ),
        pytest.param(
            True, RouteImpl.LDGSTS, 1, 1, 128, 256, id="b-ldgsts-fused-short-k"
        ),
        pytest.param(True, RouteImpl.TMA, 0, 0, 16, 1792, id="b-tma-local-sttm"),
        pytest.param(True, RouteImpl.LDGSTS, 0, 0, 16, 1792, id="b-ldgsts-local-sttm"),
    ],
)
def test_ldgsts_relay_consumers(
    monkeypatch, swap, route_a, fused_copy, fused_loads, tile_n, problem_k
):
    check_relay_consumers(
        monkeypatch, swap, route_a, fused_copy, fused_loads, tile_n, problem_k
    )


@pytest.mark.parametrize("route_a", [RouteImpl.TMA, RouteImpl.LDGSTS])
def test_compact_relay_single_sf_stage(monkeypatch, route_a):
    check_relay_consumers(
        monkeypatch, True, route_a, 0, 0, 16, 1792, num_stages_smem_sfb=1
    )


def check_relay_consumers(
    monkeypatch, swap, route_a, fused_copy, fused_loads, tile_n, problem_k, **overrides
):
    monkeypatch.setattr(
        runner,
        "_random_sf_bytes",
        lambda shape, *, cfg, device, layout_kind: runner._random_uint8(
            shape, low=0x28, high=0x41, device=device
        ),
    )
    options = dict(RUBIN_ROUTE_ACT_FP4)
    options.update(
        batch_mode=int(BatchMode.BATCH_N if swap else BatchMode.BATCH_M),
        transpose_mma_output=int(swap),
        route_act=int(route_a),
        route_sfs_act=int(RouteImpl.LDGSTS),
        fuse_sf_copy_to_mma=fused_copy,
        fuse_operand_sf_loads=fused_loads,
        tile_n=tile_n,
        mma_n=tile_n,
        tile_k=256,
        epi_tile_n=min(tile_n, 64),
        sf_layout_b=int(SfLayout.R8c4 if tile_n < 128 else SfLayout.R128c4),
        num_load_sfb_warps=4 if fused_loads else 0,
        use_early_exit=1,
    )
    options.update(overrides)
    cfg = make_config(**options)
    assert cfg.uses_ldgsts_sf_relay
    assert cfg.warp_layout.num_sync_warps == 1
    result = runner.reference_check(
        num_experts=2,
        num_tokens=259,
        top_k=1,
        problem_n=512,
        problem_k=problem_k,
        repeat_launches=3,
        output_guard_elements=256,
        **options,
    )
    assert result
    assert result.max_abs == 0
