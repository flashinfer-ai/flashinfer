# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts and B200 trace test for the persistent device driver."""

import inspect
from types import SimpleNamespace

import pytest
import torch

cutlass = pytest.importorskip("cutlass")
import cutlass.cute as cute  # noqa: E402
from cuda.bindings import driver as cuda  # noqa: E402
from cutlass.cute.nvgpu import tcgen05  # noqa: E402
from cutlass.cute.runtime import from_dlpack  # noqa: E402

from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.kernel_driver import (
    PersistentM128DeviceDriver,
)  # noqa: E402
from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.kernel_pipelines import (
    KernelWarpIds,
)  # noqa: E402


def _fake_pipelines(**overrides):
    staging = SimpleNamespace(
        mma_tiler_mn=(128, 128),
        cta_group=tcgen05.CtaGroup.ONE,
        cta_tile_shape_mnk=(128, 128, 128),
    )
    dispatch = SimpleNamespace(config=SimpleNamespace(world_size=1))
    epilogue = SimpleNamespace(
        config=SimpleNamespace(cta_tile_features=128, cta_tile_tokens=128)
    )
    values = dict(
        staging=staging,
        dispatch_phase=dispatch,
        epilogue=epilogue,
        warp_ids=KernelWarpIds(),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_driver_accepts_m128_one_cta_and_m256_two_cta() -> None:
    driver = PersistentM128DeviceDriver(kernel_pipelines=_fake_pipelines())
    assert driver.warp_ids.threads_per_cta == 384
    assert not driver.two_cta

    pair = SimpleNamespace(
        mma_tiler_mn=(256, 128),
        cta_group=tcgen05.CtaGroup.TWO,
        cta_tile_shape_mnk=(128, 128, 128),
    )
    assert PersistentM128DeviceDriver(
        kernel_pipelines=_fake_pipelines(staging=pair)
    ).two_cta

    one_cta_m256 = SimpleNamespace(
        mma_tiler_mn=(256, 128),
        cta_group=tcgen05.CtaGroup.ONE,
        cta_tile_shape_mnk=(256, 128, 128),
    )
    with pytest.raises(ValueError, match="MMA M must be"):
        PersistentM128DeviceDriver(
            kernel_pipelines=_fake_pipelines(staging=one_cta_m256)
        )

    distributed = SimpleNamespace(config=SimpleNamespace(world_size=2))
    driver = PersistentM128DeviceDriver(
        kernel_pipelines=_fake_pipelines(dispatch_phase=distributed)
    )
    assert driver.dispatch_phase is distributed


def test_driver_owns_every_persistent_role_and_tail() -> None:
    source = inspect.getsource(PersistentM128DeviceDriver.__call__)
    for role in ("scheduler", "tma_a", "tma_b", "mma", "epilogue"):
        assert role in source
    assert "dispatch_phase(" in source
    assert "kernel_tail(" in source
    assert "pipeline_init_arrive(" in source
    assert "pipeline_init_wait(" in source


@pytest.mark.parametrize("mega", [False, True])
def test_eight_warp_register_budget(mega) -> None:
    pipes = _fake_pipelines(warp_ids=KernelWarpIds.for_epilogue(8, mega=mega))
    driver = PersistentM128DeviceDriver(kernel_pipelines=pipes)
    assert driver.epilogue_registers == 184
    assert driver.task_registers == 72
    with pytest.raises(ValueError, match="register file"):
        PersistentM128DeviceDriver(kernel_pipelines=pipes, epilogue_registers=256)


def test_driver_wires_native_mma_and_readiness_protocols() -> None:
    source = inspect.getsource(PersistentM128DeviceDriver)
    # Dynamic-N issue is the only MMA path.
    assert "issue_dynamic_block_scaled_mma_tile(" in source
    assert "cute.gemm(" not in source
    assert "work.peek_ready" in source
    assert "fc1_ready_counter_ptr" in source
    assert "fc1_done_counter_ptr" in source
    assert "producer_tail(" in source
    assert "run_epilogue(" in source


class _DriverCounterTraceHarness:
    """Trace the acquire-spin primitive used at both FC phase boundaries."""

    @cute.jit
    def __call__(self, counter: cute.Tensor, observed: cute.Tensor, stream):
        self._kernel(counter, observed).launch(
            grid=(1, 1, 1), block=(32, 1, 1), stream=stream
        )

    @cute.kernel
    def _kernel(self, counter: cute.Tensor, observed: cute.Tensor):
        if cute.arch.thread_idx()[0] == 0:
            PersistentM128DeviceDriver._wait_for_counter(
                counter.iterator, cutlass.Int32(1)
            )
            observed[0] = counter[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_b200_trace_and_execute_driver_counter_wait() -> None:
    pytest.importorskip("cutlass")
    from flashinfer.utils import is_sm100a_supported

    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("requires SM100/SM103 and CUDA 12.8+")

    counter = torch.ones(1, dtype=torch.int32, device="cuda")
    observed = torch.zeros_like(counter)
    counter_cute = from_dlpack(counter, assumed_align=4).mark_layout_dynamic()
    observed_cute = from_dlpack(observed, assumed_align=4).mark_layout_dynamic()
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    compiled = cute.compile(
        _DriverCounterTraceHarness(), counter_cute, observed_cute, stream
    )
    compiled(counter_cute, observed_cute, stream)
    torch.cuda.synchronize()
    assert observed.item() == 1
