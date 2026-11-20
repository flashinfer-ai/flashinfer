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

"""NVFP4/SwiGLU FC1 on the native routed and fused dense Rubin mainloops."""

import pytest

from flashinfer.prims_ts.batched_gemm import batched_gemm_run as runner
from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
    ActKind,
    DType,
    SfLayout,
    make_config,
    validate_config,
)
from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
    _build_batched_gemm_task_manager,
)
from tests.prims_ts.routing_matrix_utils import CGAS, matrix_options
from flashinfer.prims_ts.batched_gemm.routed_config import (
    routed_tile256_fc1_options,
)
from flashinfer.prims_ts.batched_gemm.routed_smem import RoutedTile256SmemAllocator


def fc1_options(tile_m, cga, *, routed):
    options = matrix_options(tile_m, cga, routed=routed)
    options.update(
        dtype_c=int(DType.E2M1),
        act_kind=int(ActKind.SWIGLU),
        sf_layout_c=int(SfLayout.R128c4),
        use_global_scales=1,
        # Keep this matrix on the unrolled gated baseline; the rolled option
        # has separate coverage. Preloading remains a plain 16-bit path.
        dense_rolled_epilogue=0,
        dense_preload_epilogue=0,
        dense_streaming_c=0,
        dense_dkg_store_order=0,
        dense_epilogue_warps=4,
        epi_tile_n=64,
        tmem_ldst_max_num_regs=32,
        epilogue_regs=192,
    )
    return options


@pytest.mark.parametrize("tile_m", [256, 512])
@pytest.mark.parametrize("cga", CGAS, ids=lambda c: f"{c[0]}x{c[1]}")
@pytest.mark.parametrize("routed", [False, True], ids=["dense", "routed"])
def test_native_fp4_fc1_host_schedule(tile_m, cga, routed):
    options = fc1_options(tile_m, cga, routed=routed)
    cfg = make_config(**options)
    validate_config(cfg, problem_mnk=(8192, 1280, 2304))
    assert cfg.has_gated_epilogue and cfg.uses_fp4_output_quant
    assert cfg.non_swap_epilogue_subtile_cnt == tile_m // 64
    assert cfg.non_swap_gated_output_cols_per_call == 32
    assert cfg.fallback_cluster_shape == (None if cga == (2, 1) else (2, 1, 1))
    if routed:
        assert cfg.warp_layout.num_load_a_warps == 8
        assert cfg.warp_layout.num_load_sfa_warps == 4
    else:
        assert cfg.warp_layout.num_load_a_warps == 1
        assert cfg.warp_layout.num_load_b_warps == 1
    _build_batched_gemm_task_manager(verbose=False, **options)


@pytest.mark.parametrize(
    "dtype_c,act_kind",
    [(DType.BF16, ActKind.SWIGLU), (DType.E2M1, ActKind.NONE)],
)
def test_native_fp4_fc1_host_rejects_other_epilogues(dtype_c, act_kind):
    options = fc1_options(256, (2, 1), routed=True)
    options.update(dtype_c=int(dtype_c), act_kind=int(act_kind))
    with pytest.raises(ValueError, match="routed_native_sfa requires"):
        validate_config(make_config(**options))


def test_native_fp4_fc1_host_legacy_overlap_sf_relay():
    options = fc1_options(256, (2, 1), routed=True)
    options.update(
        routed_native_sfa=0,
        routed_sfa_first=0,
        use_max_tmem_overlap=1,
        mma_k=64,
        sf_layout_a=int(SfLayout.LINEAR),
        epilogue_regs=96,
    )
    cfg = make_config(**options)
    warps = cfg.warp_layout
    assert warps.num_epilogue_warps == 8
    assert warps.num_sync_warps == 1
    for role in (
        "epilogue",
        "load_a",
        "load_b",
        "load_sfa",
        "load_sfb",
        "mma",
        "workid",
    ):
        start = getattr(warps, f"{role}_warp_idx")
        count = getattr(warps, f"num_{role}_warps")
        assert warps.sync_warp_idx not in range(start, start + count)
    _build_batched_gemm_task_manager(verbose=False, **options)


@pytest.mark.arch_rubin
@pytest.mark.timeout(240)
@pytest.mark.parametrize("tile_m", [256, 512])
@pytest.mark.parametrize("cga", CGAS, ids=lambda c: f"{c[0]}x{c[1]}")
@pytest.mark.parametrize("routed", [False, True], ids=["dense", "routed"])
def test_native_fp4_fc1_gpu(tile_m, cga, routed):
    # Nonzero quantized Kaiming inputs, multiple experts, partial token tiles,
    # a partial CGA-N tile, K-ring wraparound and persistent C/SF-C reuse.
    result = runner.reference_check(
        num_experts=8,
        num_tokens=1031,
        top_k=8,
        problem_n=1280,
        problem_k=2304,
        routing="uniform",
        repeat_launches=3,
        output_guard_elements=256,
        require_mixed_cga_runtime=cga != (2, 1),
        **fc1_options(tile_m, cga, routed=routed),
    )
    assert result


@pytest.mark.arch_rubin
@pytest.mark.timeout(240)
def test_native_fp4_fc1_ragged_routing_gpu():
    # Top-k < expert count makes both token indices and per-expert counts
    # irregular, unlike the all-expert routing in the geometry matrix above.
    result = runner.reference_check(
        num_experts=8,
        num_tokens=1543,
        top_k=4,
        problem_n=1792,
        problem_k=2304,
        routing="uniform",
        repeat_launches=3,
        output_guard_elements=256,
        require_mixed_cga_runtime=True,
        **fc1_options(512, (4, 4), routed=True),
    )
    assert result


@pytest.mark.arch_rubin
@pytest.mark.timeout(240)
@pytest.mark.parametrize(
    "tile_m,cga,load_warps,automatic_width",
    [
        (256, (2, 1), 4, False),
        (256, (4, 2), 8, False),
        (512, (2, 4), 8, False),
        (512, (4, 4), 4, False),
        pytest.param(256, (2, 1), 4, True, id="auto-2x1"),
        pytest.param(512, (2, 4), 8, True, id="auto-2x4"),
    ],
)
def test_native_fp4_fc1_padding_graph_replay(
    monkeypatch, tile_m, cga, load_warps, automatic_width
):
    """Change full/empty/partial prefixes in captured graphs with reused SMEM.

    Compare against real zero-fill gathers, including SF-C, to catch omitted
    byte credits and leakage from stale padding into valid quantized rows.
    """
    import torch
    from cuda.bindings import driver as cuda

    from flashinfer.prims_ts.batched_gemm.smem_ab_resources import (
        SmemTmaGatherResource,
    )

    options = fc1_options(tile_m, cga, routed=True)
    options["num_load_a_warps"] = load_warps
    if automatic_width:
        options["cluster_dim_x"] = 0
    rows_per_expert = make_config(**options).tile_token
    experts = 3
    io = runner._build_launch_io(
        num_experts=experts,
        num_tokens=rows_per_expert,
        top_k=experts,
        problem_n=1280,  # Also exercises the 2x1 fallback for wide CGAs.
        problem_k=768,
        **options,
    )
    tensors = {tensor.data_ptr(): tensor for tensor in io["_keepalive"]}
    limits = tensors[io["mn_limit_dp"]._pointer]
    output = tensors[io["c0_dp"]._pointer]
    scales = tensors[io["sf_c_dp"]._pointer]
    offsets = torch.arange(experts, device="cuda", dtype=limits.dtype) * rows_per_expert
    torch.testing.assert_close(limits, offsets + rows_per_expert, rtol=0, atol=0)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    optimized = runner._compile_for_launch(io, stream)
    with monkeypatch.context() as baseline:
        baseline.setattr(
            SmemTmaGatherResource,
            "_load_tma_gather_tile_impl",
            SmemTmaGatherResource._load_tma_gather_full_tile,
        )
        zero_fill = runner._compile_for_launch(io, stream)
    graphs = []
    for fn in (zero_fill, optimized):
        fn(*runner._launch_arg_tuple(io, stream))
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            capture_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
            for _ in range(3):
                fn(*runner._launch_arg_tuple(io, capture_stream))
        graphs.append(graph)

    def decoded_output():
        return runner._dequantize_fp4_output(
            output,
            scales,
            output_m=io["logical_output_m"],
            n=io["logical_output_n"],
            is_swap_ab=False,
            sf_layout_c=io["cfg"].sf_layout_c,
            uses_mxfp4=False,
        ).view(experts, rows_per_expert, -1)

    rows = torch.arange(rows_per_expert, device="cuda")
    full = rows_per_expert
    for counts in ([full] * experts, [0, 1, 3], [4, 5, full - 1], [full // 2, 0, full]):
        counts = torch.tensor(counts, device="cuda", dtype=limits.dtype)
        limits.copy_(offsets + counts)
        valid = rows[None, :] < counts[:, None]
        graphs[0].replay()
        expected = decoded_output()[valid].clone()
        output.fill_(0xFF)
        scales.fill_(0x7F)
        graphs[1].replay()
        torch.testing.assert_close(decoded_output()[valid], expected, rtol=0, atol=0)


@pytest.mark.parametrize("c_stages", [1, 2, 3, 4])
def test_native_fp4_fc1_optimized_schedule(c_stages):
    options = routed_tile256_fc1_options(max_active_clusters=2)
    options["num_stages_c_smem"] = c_stages
    cfg = make_config(**options)
    validate_config(cfg)
    assert cfg.num_stages_a == cfg.num_stages_b == 8
    assert cfg.num_stages_tmem_acc == 2
    assert cfg.non_swap_gated_output_cols_per_call == 64
    assert cfg.gated_weight_group_size == 64
    assert cfg.non_swap_epilogue_subtile_cnt == 2
    assert cfg.num_bytes_c_smem_scratch == c_stages * 4096
    assert cfg.uses_non_swap_c_tma_ring == (c_stages > 1)
    assert cfg.uses_routed_weight_pipeline
    barriers = RoutedTile256SmemAllocator(cfg).barrier_regions()
    assert barriers["SmemTmaGather"] == (40, 8)
    assert barriers["SmemSfA"] == (624, 8)
    assert barriers["TmaWeights"] == (168, 8)
    assert "LdgstsActivations" not in barriers
    _build_batched_gemm_task_manager(verbose=False, **options)


def test_native_fp4_fc1_optimized_rejects_scratch_overflow():
    options = routed_tile256_fc1_options(max_active_clusters=2)
    options["num_stages_c_smem"] = 5
    with pytest.raises(ValueError, match="routed_low_smem"):
        validate_config(make_config(**options))


@pytest.mark.parametrize("group_size", [1, 64])
def test_native_fp4_fc1_grouped_weight_and_scale_rows(group_size):
    import torch

    # Unique row IDs verify both groups and their boundary across 128 rows.
    rows = torch.arange(512).reshape(1, 512, 1)
    packed = runner.reorder_rows_for_gated_act(rows, group_size)
    expected = torch.cat(
        [
            torch.cat(
                (
                    rows[0, start : start + group_size, 0],
                    rows[0, start + 256 : start + 256 + group_size, 0],
                )
            )
            for start in range(0, 256, group_size)
        ]
    )
    assert torch.equal(packed.flatten(), expected)
    sf_sources = runner._kernel_weight_row_source_indices(
        512,
        is_gated=True,
        epilogue_tile_m=128,
        shuffle_weight_rows=False,
        device=torch.device("cpu"),
        group_size=group_size,
    )
    assert torch.equal(sf_sources, expected)


@pytest.mark.parametrize(
    "override",
    [
        {"epi_tile_n": 64},
        {"gated_weight_group_size": 32},
    ],
)
def test_native_fp4_fc1_grouped_weights_reject_incompatible_epilogue(override):
    options = routed_tile256_fc1_options(max_active_clusters=2)
    options.update(override)
    with pytest.raises(ValueError, match="gated_weight_group_size"):
        validate_config(make_config(**options))


@pytest.mark.arch_rubin
def test_native_fp4_fc1_weight_cache_distinguishes_grouping_gpu():
    import torch

    cache = {}
    restored = []
    for group_size in (1, 64):
        options = routed_tile256_fc1_options(max_active_clusters=2)
        options["gated_weight_group_size"] = group_size
        io = runner._build_launch_io(
            num_experts=3,
            num_tokens=257,
            top_k=2,
            problem_n=512,
            problem_k=768,
            seed=123,
            weight_cache=cache,
            **options,
        )
        tensors = {tensor.data_ptr(): tensor for tensor in io["_keepalive"]}
        source_rows = runner._kernel_weight_row_source_indices(
            512,
            is_gated=True,
            epilogue_tile_m=128,
            shuffle_weight_rows=False,
            device=torch.device("cuda"),
            group_size=group_size,
        )
        inverse = torch.argsort(source_rows)
        weights = tensors[io["b_dp"]._pointer].index_select(1, inverse)
        scales = runner._permute_mma_sf_rows(
            tensors[io["sfb_dp"]._pointer],
            row_source_indices=inverse,
            logical_cols=768,
            layout=int(SfLayout.R128c4),
            num_experts=3,
            sf_vec_size=16,
        )
        restored.append((weights, scales))
    assert len(cache) == 2
    assert torch.equal(restored[0][0], restored[1][0])
    assert torch.equal(restored[0][1], restored[1][1])


@pytest.mark.arch_rubin
@pytest.mark.timeout(240)
@pytest.mark.parametrize("c_stages", [1, 2, 3])
def test_native_fp4_fc1_optimized_persistent_ring_gpu(c_stages):
    options = routed_tile256_fc1_options(max_active_clusters=2)
    options["num_stages_c_smem"] = c_stages
    # Two resident clusters force many work-tile handoffs. Three C slots do
    # not divide the two epilogue calls, exposing tile-local counter resets.
    # Nine K tiles also wrap the eight-stage operand rings.
    result = runner.reference_check(
        num_experts=3,
        num_tokens=1031,
        top_k=2,
        problem_n=512,
        problem_k=2304,
        routing="uniform",
        repeat_launches=3,
        output_guard_elements=256,
        **options,
    )
    assert result


@pytest.mark.parametrize("n", [1344, 1408])
def test_grouped_weights_require_complete_gate_up_groups(n):
    from tests.prims_ts.test_batched_gemm_eight_warp_fp4_fc1 import (
        eight_warp_fc1_options,
    )

    options = eight_warp_fc1_options(512, (2, 1), routed=False)
    options["dense_single_expert"] = 1
    cfg = make_config(**options)
    assert cfg.gated_weight_group_size == 64
    if n == 1344:
        with pytest.raises(ValueError, match="complete gate/up groups"):
            validate_config(cfg, problem_mnk=(512, n, 768))
    else:
        validate_config(cfg, problem_mnk=(512, n, 768))
