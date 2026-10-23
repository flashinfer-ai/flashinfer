# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused storage/layout trace for persistent kernel pipelines."""

from types import SimpleNamespace

import pytest
import torch

cutlass = pytest.importorskip("cutlass")
import cutlass.cute as cute  # noqa: E402
from cuda.bindings import driver as cuda  # noqa: E402

from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.device_epilogue import (  # noqa: E402
    DeviceEpilogueConfig,
    PersistentMxfp8Mxfp4Epilogue,
)
from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.dispatch import (
    Mxfp8DispatchPhase,
    SingleRankMxfp8DispatchPhase,
)  # noqa: E402
from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.kernel_pipelines import (  # noqa: E402
    KernelPipelines,
    KernelWarpIds,
    fit_operand_stages,
    mma_operand_stage_bytes,
)
from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.scale_tma_staging import (
    Mxfp8Mxfp4TmaStaging,
)  # noqa: E402
from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.workspace import WorkspaceConfig  # noqa: E402


def _make_pipelines(*, mma_m: int = 128, mma_n: int = 128, dispatch_start: int = 8):
    cluster_shape_mn = (2, 1) if mma_m == 256 else (1, 1)
    staging = Mxfp8Mxfp4TmaStaging(
        mma_tiler_mn=(mma_m, mma_n),
        cluster_shape_mn=cluster_shape_mn,
        num_stages=2,
    )
    workspace = WorkspaceConfig(
        world_size=1,
        num_topk=2,
        num_experts_per_rank=4,
        max_tokens_per_rank=64,
        hidden=128,
        intermediate=256,
        token_padding_block=64,
        sf_padding_block=128,
        cluster_tile_tokens=128,
    )
    dispatch = SingleRankMxfp8DispatchPhase(
        workspace,
        cluster_shape_mn=cluster_shape_mn,
        dispatch_warp_start=dispatch_start,
        num_other_warps=8,
    )
    epilogue = PersistentMxfp8Mxfp4Epilogue(
        DeviceEpilogueConfig(cta_tile_features=128, cta_tile_tokens=mma_n)
    )
    # Storage shape depends on these scheduler fields.  Device-valued
    # scheduler parameters are supplied by the final driver before it calls
    # create_scheduler.
    scheduler_shape = SimpleNamespace(
        num_stages=2, load_balance_mode="static", expert_count=32
    )
    return KernelPipelines(
        staging=staging,
        scheduler_params=scheduler_shape,
        dispatch_phase=dispatch,
        epilogue=epilogue,
    )


def test_pipeline_contract_is_m128_or_m256_block32() -> None:
    KernelPipelines.validate_mma_tiler_mn((128, 128))
    KernelPipelines.validate_mma_tiler_mn((256, 128))
    assert KernelWarpIds().dispatch == (8, 9, 10, 11)
    assert KernelWarpIds().threads_per_cta == 12 * 32


@pytest.mark.parametrize("warps", [4, 8])
@pytest.mark.parametrize("n", [64, 128, 192])
def test_fc2_staging_stage_fit(warps, n):
    kwargs = dict(
        mma_tiler_mnk=(256, n, 128),
        scheduler_params=SimpleNamespace(
            num_stages=2, load_balance_mode="static", expert_count=32
        ),
        dispatch_phase=_ep16_dispatch(4096),
        epilogue_warps=warps,
        smem_capacity_bytes=SM100_SMEM_CAPACITY,
        maximum_stages=12,
    )
    fit = fit_operand_stages(**kwargs)
    storage = KernelPipelines.build_shared_storage_type(
        num_ab_stages=fit.stages,
        num_acc_stages=2,
        scheduler_params=kwargs["scheduler_params"],
        dispatch_phase=kwargs["dispatch_phase"],
        epilogue_warps=warps,
    )
    assert fit.fixed_bytes == (storage.__sizeof__() + 127) // 128 * 128
    assert fit.used_bytes <= SM100_SMEM_CAPACITY
    # 16 x 32 BF16 staging values per warp alias the FC1 exchange.
    assert fit.fixed_bytes >= warps * 1024


@pytest.mark.parametrize("mega", [False, True])
@pytest.mark.parametrize("warps", [4, 8])
def test_epilogue_warpgroup_roles(mega, warps) -> None:
    ids = KernelWarpIds.for_epilogue(warps, mega=mega)
    assert len(ids.epilogue) == warps
    assert ids.threads_per_cta == (warps + 4 + 4 * mega) * 32
    assert ids.epilogue[:4] == (0, 1, 2, 3)
    if warps == 8:
        start = 12 if mega else 8
        assert ids.epilogue[4:] == tuple(range(start, start + 4))
    assert ids.dispatch == ((8, 9, 10, 11) if mega else ())


def test_epilogue_rejects_mixed_role_warpgroups() -> None:
    with pytest.raises(ValueError, match="role-pure"):
        KernelWarpIds(epilogue=(0, 1, 2, 12, 3, 13, 14, 15))


@pytest.mark.parametrize("mega", [False, True])
@pytest.mark.parametrize("n", [64, 128, 192])
def test_eight_warp_stage_fit_sizes_exchange_storage(mega, n) -> None:
    kwargs = dict(
        mma_tiler_mnk=(256, n, 128),
        scheduler_params=SimpleNamespace(
            num_stages=2, load_balance_mode="static", expert_count=32
        ),
        dispatch_phase=_ep16_dispatch(6400) if mega else None,
        smem_capacity_bytes=SM100_SMEM_CAPACITY,
        maximum_stages=12,
    )
    four = fit_operand_stages(**kwargs, epilogue_warps=4)
    eight = fit_operand_stages(**kwargs, epilogue_warps=8)
    # Four more warps each stage 16 x 32 BF16 values for FC2.
    assert eight.fixed_bytes - four.fixed_bytes == 4096
    assert eight.stages <= four.stages
    assert eight.used_bytes <= SM100_SMEM_CAPACITY


def test_pipeline_rejects_other_mma_m_and_shifted_dispatch_warps() -> None:
    with pytest.raises(ValueError, match="M128"):
        KernelPipelines.validate_mma_tiler_mn((64, 128))
    with pytest.raises(ValueError, match="distinct"):
        KernelWarpIds(dispatch=(7, 9, 10, 11))


def test_two_cta_stage_bytes_are_per_cta() -> None:
    # Each CTA of a 2-CTA pair stages its own 128 weight rows and half of the
    # token tile; SFB stays whole (1-CTA SFB view).
    for n in (64, 128, 192):
        one = mma_operand_stage_bytes((128, n, 128))
        two = mma_operand_stage_bytes((256, n, 128))
        assert two.weight_a == one.weight_a
        assert two.weight_sfa == one.weight_sfa
        assert two.activation_b * 2 == one.activation_b
        assert two.activation_sfb == one.activation_sfb


class _PipelineTraceHarness:
    """Trace allocation of the combined struct, operands, and both pipes."""

    @cute.jit
    def __call__(self, stream):
        self._kernel().launch(
            grid=(1, 1, 1),
            block=(12 * 32, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def _kernel(self):
        # Layouts contain MLIR values and therefore must be created and
        # consumed wholly within this kernel region.  Storing ``pipelines`` on
        # the harness would leak composed layouts across CuTe region isolation.
        pipelines = _make_pipelines()
        stage_bytes = pipelines.staging.operand_stage_bytes()
        assert stage_bytes.weight_a == 128 * 128
        # The host-side stage fit uses this formula; keep it tied to layouts.
        assert stage_bytes == mma_operand_stage_bytes((128, 128, 128))
        assert pipelines.staging.tma_transaction_bytes_a() == 128 * 128 // 2 + 512
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(pipelines.shared_storage_type())
        pipelines.allocate_operand_smem(smem)
        pipelines.create_mainloop_pipelines(storage)


class _TwoCtaPipelineTraceHarness:
    """Trace the 2-CTA pipelines and tie the host stage bytes to the layouts."""

    def __init__(self, mma_n: int):
        self.mma_n = mma_n

    @cute.jit
    def __call__(self, stream):
        self._kernel().launch(
            grid=(2, 1, 1),
            block=(12 * 32, 1, 1),
            cluster=(2, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def _kernel(self):
        pipelines = _make_pipelines(mma_m=256, mma_n=self.mma_n)
        stage_bytes = pipelines.staging.operand_stage_bytes()
        assert stage_bytes == mma_operand_stage_bytes((256, self.mma_n, 128))
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(pipelines.shared_storage_type())
        pipelines.allocate_operand_smem(smem)
        pipelines.create_mainloop_pipelines(storage)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("mma_n", [64, 128, 192])
def test_b200_traces_two_cta_pipelines_and_stage_bytes(mma_n: int) -> None:
    pytest.importorskip("cutlass")
    from flashinfer.utils import is_sm100a_supported

    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("requires SM100/SM103 and CUDA 12.8+")
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    cute.compile(_TwoCtaPipelineTraceHarness(mma_n), stream)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_b200_traces_shared_storage_and_pipeline_layout() -> None:
    pytest.importorskip("cutlass")
    from flashinfer.utils import is_sm100a_supported

    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("requires SM100/SM103 and CUDA 12.8+")

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    # Compilation traces every allocation and both pipeline constructors.  No
    # launch is required: execution behavior belongs to the final driver test.
    cute.compile(_PipelineTraceHarness(), stream)


# Shared-memory stage fit.  GB200/B200 allow 232448 dynamic bytes per CTA.
SM100_SMEM_CAPACITY = 232448


def _ep16_dispatch(hidden: int) -> Mxfp8DispatchPhase:
    """Dispatch phase of the EP16 multirank gate (2 experts/rank, top-k 2)."""

    warps = KernelWarpIds()
    workspace = WorkspaceConfig(
        world_size=16,
        num_topk=2,
        num_experts_per_rank=2,
        max_tokens_per_rank=7,
        hidden=hidden,
        intermediate=4096,
        token_padding_block=64,
        sf_padding_block=128,
        cluster_tile_tokens=128,
        load_balance_mode="static",
    )
    return Mxfp8DispatchPhase(
        workspace,
        rank=0,
        cluster_shape_mn=(1, 1),
        dispatch_warp_start=warps.dispatch[0],
        num_other_warps=len(warps.epilogue) + 4,
        flag_batch=2,
    )


def _fit(hidden: int, capacity: int, maximum_stages: int = 8) -> int:
    return fit_operand_stages(
        mma_tiler_mnk=(128, 128, 128),
        scheduler_params=SimpleNamespace(
            num_stages=2, load_balance_mode="static", expert_count=32
        ),
        dispatch_phase=_ep16_dispatch(hidden),
        smem_capacity_bytes=capacity,
        maximum_stages=maximum_stages,
    ).stages


def test_stage_fit_boundary_at_h6400() -> None:
    # M128/N128, four epilogue warps: 33792 bytes per operand stage on top of
    # the control struct, whose 4 KB FC2 staging buffer aliases the FC1
    # amax exchange, plus four 32-entry scheduler tables (512 bytes).
    # The GPU launch-boundary test below checks this model.
    assert _fit(6400, 233472, maximum_stages=6) == 6
    assert _fit(6400, 233471, maximum_stages=6) == 5
    assert _fit(6400, SM100_SMEM_CAPACITY) == 5
    assert _fit(256, SM100_SMEM_CAPACITY) == 6


class _StageFitLaunchHarness:
    """Allocate exactly the persistent kernel's shared memory, then exit."""

    def __init__(self, stages: int, hidden: int):
        self.stages = stages
        self.hidden = hidden

    @cute.jit
    def __call__(self, stream):
        self._kernel().launch(
            grid=(1, 1, 1),
            block=(KernelWarpIds().threads_per_cta, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def _kernel(self):
        pipelines = KernelPipelines(
            staging=Mxfp8Mxfp4TmaStaging(
                mma_tiler_mn=(128, 128),
                cluster_shape_mn=(1, 1),
                num_stages=self.stages,
            ),
            scheduler_params=SimpleNamespace(
                num_stages=2, load_balance_mode="static", expert_count=32
            ),
            dispatch_phase=_ep16_dispatch(self.hidden),
            epilogue=PersistentMxfp8Mxfp4Epilogue(
                DeviceEpilogueConfig(cta_tile_features=128, cta_tile_tokens=128)
            ),
        )
        smem = cutlass.utils.SmemAllocator()
        smem.allocate(pipelines.shared_storage_type())
        pipelines.allocate_operand_smem(smem)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_b200_stage_fit_boundary_matches_launch_limit() -> None:
    pytest.importorskip("cutlass")
    from flashinfer.utils import is_sm100a_supported

    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("requires SM100/SM103 and CUDA 12.8+")
    fitted = _fit(6400, SM100_SMEM_CAPACITY)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    # The fitted depth launches; one more stage exceeds the per-CTA limit.
    cute.compile(_StageFitLaunchHarness(fitted, 6400), stream)(stream)
    torch.cuda.synchronize()
    # CuTe 4.6.2 prints the shared-memory diagnostic to stderr but raises
    # CUDA_ERROR_INVALID_VALUE from the launch wrapper.
    with pytest.raises(Exception, match="(?i)shared memory|CUDA_ERROR_INVALID_VALUE"):
        cute.compile(_StageFitLaunchHarness(fitted + 1, 6400), stream)(stream)
        torch.cuda.synchronize()
