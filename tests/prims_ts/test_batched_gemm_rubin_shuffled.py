# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shuffled weights in the retained Rubin Batch-M epilogues."""

from types import MethodType, SimpleNamespace

import pytest
import torch

from benchmarks.bench_rubin_gemm import configuration
from flashinfer.prims_ts.batched_gemm import batched_gemm_run as runner
from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
    DType,
    make_config,
    validate_config,
)
from flashinfer.prims_ts.batched_gemm.gmem_c_resources import GmemCResource
from flashinfer.tllm_enums import WeightLayout
from tests.prims_ts.python_body import python_body
from tests.prims_ts.routing_matrix_utils import CGAS
from tests.prims_ts.test_batched_gemm_eight_warp_fp4_fc1 import eight_warp_fc1_options
from tests.prims_ts.test_batched_gemm_rolled_fp4_fc1 import rolled_fc1_options


def shuffled_profile(mode, cga, weight_layout, group_size=1):
    _, options = configuration("e32", mode)
    flexible = cga != (2, 1)
    options.update(
        cluster_dim_x=cga[0],
        cluster_dim_y=cga[1],
        fallback_cluster_dim_x=2 if flexible else 0,
        fallback_cluster_dim_y=1,
        dense_specialize_tma_cga=int(flexible),
        dense_local_barrier=int(flexible),
        use_shuffled_matrix=1,
        weight_layout=int(weight_layout),
        gated_weight_group_size=group_size,
    )
    return options


@pytest.mark.parametrize("group_size", [1, 64])
@pytest.mark.parametrize("shuffled", [0, 1])
def test_grouped_shuffle_accumulator_matches_host_weight_and_scale_rows(
    monkeypatch, group_size, shuffled
):
    _, options = configuration("e32", "fc1")
    options.update(gated_weight_group_size=group_size, use_shuffled_matrix=shuffled)
    cfg = make_config(**options)
    validate_config(cfg)
    original = torch.arange(512).reshape(1, 512, 1)
    packed = runner.reorder_rows_for_gated_act(original, group_size)
    if shuffled:
        packed = runner.shuffle_matrix(packed, cfg.epi_tile_m)
    sf_rows = runner._kernel_weight_row_source_indices(
        512,
        is_gated=True,
        epilogue_tile_m=cfg.epi_tile_m,
        shuffle_weight_rows=bool(shuffled),
        device=torch.device("cpu"),
        group_size=group_size,
    )
    assert torch.equal(packed.flatten(), sf_rows)
    resource = SimpleNamespace(cfg=cfg)
    resource._non_swap_shuffled_col_idx = MethodType(
        GmemCResource._non_swap_shuffled_col_idx, resource
    )
    load = python_body(GmemCResource._load_non_swap_accumulator)
    import cutlass

    monkeypatch.setattr(cutlass, "const_expr", lambda condition: condition)
    for start in range(0, 256, 64):
        fragments = packed.flatten()[2 * start : 2 * start + 128].tolist()
        actual = [
            load(resource, fragments[:64], fragments[64:], col, 64)
            for col in range(128)
        ]
        expected = [start + col // 2 + (256 if col % 2 else 0) for col in range(128)]
        assert actual == expected


@pytest.mark.parametrize(
    "mode,group_size",
    [("fc1", 1), ("fc1", 64), ("routed-bf16", 1), ("dense-bf16", 1)],
    ids=["fc1_group1", "fc1_group64", "routed-bf16", "dense-bf16"],
)
@pytest.mark.parametrize(
    "layout", [WeightLayout.MajorK, WeightLayout.BlockMajorK], ids=lambda w: w.name
)
@pytest.mark.parametrize("cga", CGAS, ids=lambda c: f"{c[0]}x{c[1]}")
def test_shuffled_best_profile_contract(mode, group_size, layout, cga):
    cfg = make_config(**shuffled_profile(mode, cga, layout, group_size))
    validate_config(cfg, problem_mnk=(12288, 1280, 2304))
    assert cfg.dense_m_repeats == 2
    assert cfg.warp_layout.num_epilogue_warps == 8
    assert cfg.dense_split_tma_pipeline and cfg.dense_fuse_tma_loads
    assert cfg.dense_partition_tma_loads
    assert cfg.uses_shuffled_matrix
    assert cfg.uses_block_major_k_weight == (layout == WeightLayout.BlockMajorK)
    assert cfg.gated_weight_group_size == group_size


@pytest.mark.arch_rubin
@pytest.mark.timeout(240)
@pytest.mark.parametrize(
    "mode,group_size",
    [("fc1", 1), ("fc1", 64), ("routed-bf16", 1), ("dense-bf16", 1)],
    ids=["fc1_group1", "fc1_group64", "routed-bf16", "dense-bf16"],
)
@pytest.mark.parametrize(
    "layout", [WeightLayout.MajorK, WeightLayout.BlockMajorK], ids=lambda w: w.name
)
@pytest.mark.parametrize("cga", CGAS, ids=lambda c: f"{c[0]}x{c[1]}")
def test_shuffled_best_profile_gpu(mode, group_size, layout, cga):
    assert runner.reference_check(
        num_experts=8,
        num_tokens=1031,
        top_k=8,
        problem_n=1280,
        problem_k=2304,
        routing="uniform",
        repeat_launches=3,
        output_guard_elements=256,
        require_mixed_cga_runtime=cga != (2, 1),
        **shuffled_profile(mode, cga, layout, group_size),
    )


@pytest.mark.arch_rubin
@pytest.mark.timeout(240)
@pytest.mark.parametrize("tile_m,warps", [(256, 4), (256, 8), (512, 4)])
@pytest.mark.parametrize("group_size", [1, 64], ids=lambda g: f"group{g}")
@pytest.mark.parametrize(
    "layout", [WeightLayout.MajorK, WeightLayout.BlockMajorK], ids=lambda w: w.name
)
def test_shuffled_fc1_other_epilogues_gpu(tile_m, warps, layout, group_size):
    factory = eight_warp_fc1_options if warps == 8 else rolled_fc1_options
    options = factory(tile_m, (2, 2), routed=True)
    options.update(
        use_shuffled_matrix=1,
        weight_layout=int(layout),
        gated_weight_group_size=group_size,
    )
    assert runner.reference_check(
        num_experts=8,
        num_tokens=1031,
        top_k=8,
        problem_n=1280,
        problem_k=2304,
        routing="uniform",
        repeat_launches=3,
        output_guard_elements=256,
        require_mixed_cga_runtime=True,
        **options,
    )


@pytest.mark.arch_rubin
@pytest.mark.timeout(240)
@pytest.mark.parametrize("tile_m", [256, 512])
@pytest.mark.parametrize("warps", [4, 8])
@pytest.mark.parametrize(
    "layout", [WeightLayout.MajorK, WeightLayout.BlockMajorK], ids=lambda w: w.name
)
def test_shuffled_group64_dense_fc1_gpu(tile_m, warps, layout):
    # Dense FC1 uses fused TMA A/SFA loads and can take wider TMEM reads than
    # the routed profile. Check the new group64+shuffle composition there too.
    factory = eight_warp_fc1_options if warps == 8 else rolled_fc1_options
    # The eight-stage tile256 operand rings only leave room for two C slots
    # with eight epilogue warps. Exercise a non-dividing ring elsewhere.
    c_stages = 2 if tile_m == 256 and warps == 8 else 3
    options = factory(tile_m, (2, 2), routed=False, c_stages=c_stages)
    options.update(
        use_shuffled_matrix=1,
        weight_layout=int(layout),
        gated_weight_group_size=64,
    )
    # With top_k < experts, these sizes keep ragged expert counts while
    # forcing more than one persistent wave for both physical CGA shapes.
    assert runner.reference_check(
        num_experts=8,
        num_tokens=2579 if tile_m == 512 else 1543,
        top_k=4,
        problem_n=1280,
        problem_k=2304,
        routing="uniform",
        repeat_launches=3,
        output_guard_elements=256,
        require_mixed_cga_runtime=True,
        **options,
    )


@pytest.mark.arch_rubin
@pytest.mark.timeout(240)
@pytest.mark.parametrize("group_size", [1, 64])
@pytest.mark.parametrize(
    "layout", [WeightLayout.MajorK, WeightLayout.BlockMajorK], ids=lambda w: w.name
)
def test_unshuffled_fc1_layout_group_gpu(group_size, layout):
    # Together with the shuffled cases above, covers all eight layout/group/
    # shuffle combinations using the same small problem and mixed CGAs.
    options = shuffled_profile("fc1", (2, 2), layout, group_size)
    options["use_shuffled_matrix"] = 0
    assert runner.reference_check(
        num_experts=8,
        num_tokens=1031,
        top_k=8,
        problem_n=1280,
        problem_k=2304,
        routing="uniform",
        repeat_launches=3,
        output_guard_elements=256,
        require_mixed_cga_runtime=True,
        **options,
    )


@pytest.mark.arch_rubin
@pytest.mark.timeout(240)
@pytest.mark.parametrize("mode", ["routed-bf16", "dense-bf16"])
@pytest.mark.parametrize(
    "layout", [WeightLayout.MajorK, WeightLayout.BlockMajorK], ids=lambda w: w.name
)
def test_shuffled_four_warp_preload_gpu(mode, layout):
    # The four-warp preload uses both x64 reads and an x32 tail, unlike the
    # cooperative eight-warp path. Both must restore logical column order.
    options = shuffled_profile(mode, (2, 2), layout)
    # Its two C stages need the older five-stage weight pipeline to fit SMEM.
    options.update(
        dense_epilogue_warps=4,
        epilogue_regs=248,
        num_stages_c_smem=2,
        num_stages_b=5,
        num_stages_smem_sfb=5,
    )
    assert runner.reference_check(
        num_experts=8,
        num_tokens=1031,
        top_k=8,
        problem_n=1280,
        problem_k=2304,
        routing="uniform",
        repeat_launches=3,
        output_guard_elements=256,
        require_mixed_cga_runtime=True,
        **options,
    )


@pytest.mark.arch_rubin
@pytest.mark.timeout(240)
@pytest.mark.parametrize(
    "mode,warps,dtype",
    [
        ("dense-bf16", 4, DType.BF16),
        ("dense-bf16", 8, DType.BF16),
        ("routed-bf16", 4, DType.BF16),
        ("routed-bf16", 8, DType.BF16),
        ("dense-bf16", 4, DType.FP16),
        ("dense-bf16", 8, DType.FP16),
    ],
    ids=[
        "dense4-bf16",
        "dense8-bf16",
        "routed4-bf16",
        "routed8-bf16",
        "dense4-fp16",
        "dense8-fp16",
    ],
)
@pytest.mark.parametrize("shuffled", [0, 1])
@pytest.mark.parametrize(
    "layout", [WeightLayout.MajorK, WeightLayout.BlockMajorK], ids=lambda w: w.name
)
def test_preloaded_epilogue_expert_scales_gpu(
    monkeypatch, mode, warps, dtype, shuffled, layout
):
    """Scale FP32 accumulators before packing, including after expert changes."""
    options = shuffled_profile(mode, (4, 2) if warps == 8 else (2, 2), layout)
    options.update(
        use_shuffled_matrix=shuffled,
        use_global_scales=1,
        dtype_c=int(dtype),
        dense_epilogue_warps=warps,
        dense_preload_store_prefix_fragments=2 if warps == 8 else 0,
    )
    if warps == 4:
        options.update(
            epilogue_regs=248,
            num_stages_c_smem=2,
            num_stages_b=5,
            num_stages_smem_sfb=5,
        )

    if dtype == DType.FP16:
        # The default NVFP4 scale distribution can overflow plain FP16.
        # Bound input block scales, as in the existing FP16 preload tests,
        # while keeping the output multipliers below distinct and non-unit.
        def bounded_sf(shape, *, cfg, device, layout_kind):
            return runner._random_uint8(shape, low=0x28, high=0x41, device=device)

        monkeypatch.setattr(runner, "_random_sf_bytes", bounded_sf)

    # The reference runner normally fills every expert's scale with one scalar.
    # Replace only that allocation, so the independent reference and kernel
    # both receive distinct scales without changing operand generation.
    scale_sentinel = 0.375
    original_full = torch.full
    scale_allocations = []

    def full_with_expert_scales(size, fill_value, **kwargs):
        result = original_full(size, fill_value, **kwargs)
        if (
            size == (8,)
            and fill_value == scale_sentinel
            and kwargs.get("dtype") == torch.float32
        ):
            result.copy_(
                result.new_tensor([0.0, 0.125, -0.75, 0.3, 1.25, -1.5, 0.7, 2.0])
            )
            scale_allocations.append(result)
        return result

    monkeypatch.setattr(torch, "full", full_with_expert_scales)
    assert runner.reference_check(
        num_experts=8,
        num_tokens=2579,
        top_k=4,
        problem_n=1280,
        problem_k=2304,
        routing="uniform",
        scale_c_value=scale_sentinel,
        repeat_launches=3,
        output_guard_elements=256,
        require_mixed_cga_runtime=True,
        **options,
    )
    assert len(scale_allocations) == 1


@pytest.mark.arch_rubin
@pytest.mark.timeout(240)
@pytest.mark.parametrize(
    "layout", [WeightLayout.MajorK, WeightLayout.BlockMajorK], ids=lambda w: w.name
)
@pytest.mark.parametrize("c_stages", [1, 2])
@pytest.mark.parametrize("store_prefix", [0, 2])
@pytest.mark.parametrize("release", [1, 2])
def test_preloaded_epilogue_scale_graph_replay_gpu(
    layout, c_stages, store_prefix, release
):
    """Captured FC2 reads updated expert scales without recompilation."""
    import cuda.bindings.driver as cuda

    options = shuffled_profile("dense-bf16", (2, 2), layout)
    options["use_global_scales"] = 1
    options["dense_preload_store_prefix_fragments"] = store_prefix
    options["epilogue_tmem_release"] = release
    options["num_stages_c_smem"] = c_stages
    if c_stages == 2:
        # Make room for both C slots without changing the paired B/SFB depths.
        options.update(num_stages_b=5, num_stages_smem_sfb=5)
    io = runner._build_launch_io(
        num_experts=8,
        num_tokens=1031,
        top_k=4,
        problem_n=1280,
        problem_k=2304,
        routing="uniform",
        scale_c_value=1.0,
        **options,
    )
    owners = {tensor.data_ptr(): tensor for tensor in io["_keepalive"]}
    scales = owners[io["scale_c_dp"]._pointer]
    output = owners[io["c0_dp"]._pointer]
    experts = torch.tensor(io["token_layout"].expanded_to_expert, device="cuda")
    active = experts >= 0
    expert_ids = experts[active]
    torch_stream = torch.cuda.Stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)
    torch_stream.wait_stream(torch.cuda.current_stream())
    kernel = runner._compile_for_launch(io, stream)
    args = runner._launch_arg_tuple(io, stream)
    with torch.cuda.stream(torch_stream):
        kernel(*args)
    torch.cuda.current_stream().wait_stream(torch_stream)
    baseline = output[active].clone()
    assert torch.isfinite(baseline).all()
    assert torch.count_nonzero(baseline) > 0
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=torch_stream):
        kernel(*args)
    # Powers of two commute with BF16 rounding for these finite normal values,
    # so the unit-scale output is an exact oracle for scale-update visibility.
    # The numerical-reference tests above also cover non-power-of-two scales.
    for values in (
        [0.5, 2.0, 0.0, -1.0, 4.0, 0.25, -2.0, 1.0],
        [2.0, -1.0, 0.5, 4.0, 0.25, 0.0, 1.0, -2.0],
    ):
        scales.copy_(scales.new_tensor(values))
        expected = (baseline.float() * scales[expert_ids, None]).to(output.dtype)
        for _ in range(3):
            graph.replay()
            torch.testing.assert_close(output[active], expected, rtol=0, atol=0)


@pytest.mark.arch_rubin
@pytest.mark.timeout(240)
def test_preloaded_store_prefix_without_global_scales_gpu():
    options = shuffled_profile("dense-bf16", (4, 2), WeightLayout.BlockMajorK)
    assert options["use_global_scales"] == 0
    options["dense_preload_store_prefix_fragments"] = 2
    assert runner.reference_check(
        num_experts=8,
        num_tokens=1031,
        top_k=8,
        problem_n=1280,
        problem_k=2304,
        routing="uniform",
        repeat_launches=3,
        output_guard_elements=256,
        require_mixed_cga_runtime=True,
        **options,
    )


@pytest.mark.arch_rubin
@pytest.mark.timeout(240)
@pytest.mark.parametrize(
    "mode,cga",
    [
        ("routed-bf16", (2, 4)),
        ("routed-bf16", (4, 2)),
        ("dense-bf16", (2, 2)),
        ("dense-bf16", (4, 2)),
    ],
    ids=["routed-2x4", "routed-4x2", "dense-2x2", "dense-4x2"],
)
@pytest.mark.parametrize("shuffled", [0, 1], ids=["unshuffled", "shuffled"])
@pytest.mark.parametrize(
    "layout", [WeightLayout.MajorK, WeightLayout.BlockMajorK], ids=lambda w: w.name
)
def test_plain_preload_register_allocation_gpu(
    monkeypatch, mode, cga, shuffled, layout
):
    import cuda.bindings.driver as cuda

    # Cover the opt-in spill-free allocations separately from the retained
    # timing presets. More epilogue registers require smaller loader budgets
    # in the 24-warp routed CTA; the 12-warp dense CTA has enough headroom.
    _, options = configuration(
        "e32" if cga == (4, 2) else "e256",
        mode,
        use_shuffled_matrix=shuffled,
        weight_layout="major_k" if layout == WeightLayout.MajorK else "block_major_k",
        epilogue_regs=176 if mode == "routed-bf16" else 160,
        non_epilogue_regs=32 if mode == "routed-bf16" else 80,
    )
    compiled = []
    original_compile = runner._compile_for_launch

    def capture_compile(*args, **kwargs):
        kernel = original_compile(*args, **kwargs)
        compiled.append(kernel)
        return kernel

    monkeypatch.setattr(runner, "_compile_for_launch", capture_compile)
    assert runner.reference_check(
        num_experts=8,
        num_tokens=1031,
        top_k=8,
        problem_n=1280,
        problem_k=2304,
        routing="uniform",
        repeat_launches=3,
        output_guard_elements=256,
        # Do not add CGA recorder instructions to the kernel being inspected.
        # The geometry tests above check mixed preferred/fallback execution.
        **options,
    )
    assert compiled
    for kernel in compiled:
        libraries = kernel.jit_module.cuda_library
        assert libraries
        functions = []
        for library in libraries:
            handle = cuda.CUlibrary(int(library))
            error, count = cuda.cuLibraryGetKernelCount(handle)
            assert error == cuda.CUresult.CUDA_SUCCESS
            assert count > 0
            error, kernels = cuda.cuLibraryEnumerateKernels(count, handle)
            assert error == cuda.CUresult.CUDA_SUCCESS
            functions.extend(kernels)
        for function_kernel in functions:
            error, function = cuda.cuKernelGetFunction(function_kernel)
            assert error == cuda.CUresult.CUDA_SUCCESS
            error, local_bytes = cuda.cuFuncGetAttribute(
                cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, function
            )
            assert error == cuda.CUresult.CUDA_SUCCESS
            assert local_bytes == 0, f"{local_bytes} bytes of kernel local memory"
