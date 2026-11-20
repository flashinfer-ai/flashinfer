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

"""Small, guarded counterparts of the E32/E256/E384 performance matrix."""

import pytest

from flashinfer.prims_ts.batched_gemm import batched_gemm_run as runner
from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
    make_config,
    validate_config,
)
from tests.prims_ts.routing_matrix_utils import (
    CASES,
    CGAS,
    install_matched_inputs,
    matrix_options,
    matrix_shape,
)


@pytest.mark.parametrize("tile_m", [256, 512])
@pytest.mark.parametrize("cga", CGAS, ids=lambda c: f"{c[0]}x{c[1]}")
@pytest.mark.parametrize("routed", [False, True], ids=["dense", "routed"])
def test_routing_matrix_host_cga_policy(tile_m, cga, routed):
    cfg = make_config(**matrix_options(tile_m, cga, routed=routed))
    validate_config(cfg, problem_mnk=(32768, 1280, 2304))
    flexible = cga != (2, 1)
    assert cfg.fallback_cluster_shape == ((2, 1, 1) if flexible else None)
    assert cfg.dense_specialize_tma_cga == cfg.dense_local_barrier == int(flexible)


@pytest.mark.parametrize("tile_m", [256, 512])
@pytest.mark.parametrize("cluster_x", [0, 2], ids=["implicit-x", "explicit-x"])
@pytest.mark.parametrize("routed", [False, True], ids=["dense", "routed"])
def test_routing_matrix_host_rejects_2x1_fallback(tile_m, cluster_x, routed):
    options = matrix_options(tile_m, (2, 1), routed=routed)
    options.update(cluster_dim_x=cluster_x, fallback_cluster_dim_x=2)
    with pytest.raises(ValueError, match="preferred CGA 2x1 requires a fixed launch"):
        validate_config(make_config(**options))


@pytest.mark.arch_rubin
@pytest.mark.timeout(180)
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("tile_m", [256, 512])
@pytest.mark.parametrize("cga", CGAS, ids=lambda c: f"{c[0]}x{c[1]}")
@pytest.mark.parametrize("routed", [False, True], ids=["dense", "routed"])
def test_routing_matrix(monkeypatch, case, tile_m, cga, routed):
    import torch

    options = matrix_options(tile_m, cga, routed=routed)
    cfg = make_config(**options)
    assert cfg.has_routed_act == cfg.has_routed_sfs == routed
    flexible = cga != (2, 1)
    assert cfg.fallback_cluster_shape == ((2, 1, 1) if flexible else None)
    assert cfg.dense_specialize_tma_cga == cfg.dense_local_barrier == int(flexible)
    if routed:
        assert cfg.warp_layout.num_load_a_warps == 8
        assert cfg.warp_layout.num_load_sfa_warps == 4
    else:
        warps = cfg.warp_layout
        assert warps.num_load_a_warps == warps.num_load_b_warps == 1
        assert warps.load_a_warp_idx == warps.load_sfa_warp_idx
        assert warps.load_b_warp_idx == warps.load_sfb_warp_idx
        assert warps.load_a_warp_idx != warps.load_b_warp_idx
    inputs = install_matched_inputs(monkeypatch, case, zero=False)
    shape = matrix_shape(case, small=True)
    if routed and tile_m == 512 and case == "e256":
        # The reduced eight-expert fixture needs more N tiles to exceed one
        # resident wave for both the preferred and fallback physical CGAs.
        shape["problem_n"] = 2304
    result = runner.reference_check(
        **shape,
        data_distribution="default",
        repeat_launches=3,
        output_guard_elements=256,
        require_mixed_cga_runtime=routed and cga != (2, 1),
        **options,
    )
    assert result and result.max_abs == 0
    for name in ("a_compact", "b", "sfa_compact", "sfb"):
        assert torch.count_nonzero(inputs[name]), name


@pytest.mark.arch_rubin
@pytest.mark.parametrize("case", CASES)
def test_routing_matrix_input_equivalence(monkeypatch, case):
    """Audit actual launch pointers, including dense SFA's different layout."""
    import torch

    shape = matrix_shape(case, small=True)
    shape.update(problem_n=256, problem_k=256)
    prepared = []
    for routed in (False, True):
        with monkeypatch.context() as patch:
            state = install_matched_inputs(patch, case, zero=False)
            options = matrix_options(512, (4, 4), routed=routed)
            io = runner._build_launch_io(
                **shape,
                data_distribution="default",
                **options,
            )
            tensors = {t.data_ptr(): t for t in io["_keepalive"]}
            operands = {
                name: tensors[io[name]._pointer]
                for name in ("a_dp", "b_dp", "sfa_dp", "sfb_dp")
            }
            for name, tensor in operands.items():
                assert torch.count_nonzero(tensor), name
            prepared.append((state, operands, io))
    dense, routed = prepared
    for name in ("a_compact", "b", "sfa_compact", "sfb"):
        assert torch.equal(dense[0][name], routed[0][name]), name
    for name in ("b_dp", "sfb_dp"):
        assert torch.equal(dense[1][name], routed[1][name]), name
    layout = dense[0]["layout"]
    assert layout.expanded_to_token == routed[0]["layout"].expanded_to_token
    rows = (
        torch.tensor(layout.expanded_to_expert, device="cuda").ge(0).nonzero().flatten()
    )
    gathers = torch.tensor(layout.expanded_to_token, device="cuda")[rows]
    assert torch.equal(dense[1]["a_dp"][rows], routed[1]["a_dp"][gathers])
    offsets = runner._sf_offsets(
        row_idx=rows,
        sf_cols=16,
        layout=make_config(**matrix_options(512, (4, 4), routed=False)).sf_layout_a,
        device="cuda",
    )
    assert torch.equal(
        dense[1]["sfa_dp"][offsets], routed[1]["sfa_dp"].view(-1, 16)[gathers]
    )
