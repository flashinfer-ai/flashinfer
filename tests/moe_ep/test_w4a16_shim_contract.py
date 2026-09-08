"""Host launch contracts for the W4A16-only MegaMoE shim."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from flashinfer.moe_ep import Sm100_Bf16_Nvfp4_Bf16_Cutedsl_MegaMoeConfig


@pytest.fixture
def shim():
    pytest.importorskip("cutlass")
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import w4a16_mega_moe

    return w4a16_mega_moe


@pytest.fixture
def buffer():
    frontend = mock.Mock()
    frontend.config.in_kernel_fc2_reduce = False
    frontend._reduce = object()
    combined = torch.empty(4, 2, 64, dtype=torch.bfloat16)
    frontend.run.return_value = combined[:1]
    return SimpleNamespace(
        _destroyed=False,
        num_max_tokens=4,
        hidden=64,
        x=torch.empty(4, 64, dtype=torch.bfloat16),
        topk_idx=torch.empty(4, 2, dtype=torch.int64),
        topk_weights=torch.empty(4, 2, dtype=torch.float32),
        combine_output=combined,
        _frontend=frontend,
    )


def _output(num_tokens=1, *, contiguous=True):
    # Only CUDA ownership/launch ordering is under test. No device allocation
    # or kernel execution is needed to exercise the host wrapper.
    output = mock.Mock(spec=torch.Tensor)
    output.shape = (num_tokens, 64)
    output.dtype = torch.bfloat16
    output.is_cuda = True
    output.is_contiguous.return_value = contiguous
    return output


def _call(shim, buffer, output, **kwargs):
    shim(
        output,
        (None, None, None),
        (None, None, None),
        buffer,
        num_tokens=output.shape[0],
        **kwargs,
    )


def test_config_rejects_ignored_fast_math_option():
    with pytest.raises(TypeError, match="fast_math"):
        Sm100_Bf16_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
            intermediate_size=64, top_k=2, fast_math=False
        )


def test_shim_rejects_ignored_fast_math_option(shim, buffer):
    with pytest.raises(TypeError, match="fast_math"):
        _call(shim, buffer, _output(), fast_math=False)
    buffer._frontend.run.assert_not_called()


@pytest.mark.parametrize("kind", ("cpu", "noncontiguous"))
def test_output_rejected_before_collective_launch(shim, buffer, kind):
    output = (
        torch.empty(1, 64, dtype=torch.bfloat16)
        if kind == "cpu"
        else _output(contiguous=False)
    )
    with pytest.raises(ValueError, match="contiguous CUDA"):
        _call(shim, buffer, output)
    buffer._frontend.run.assert_not_called()


@pytest.mark.parametrize("sync", (False, True))
def test_sync_covers_final_reduction(shim, buffer, sync):
    events = []

    def fused(*args, **kwargs):
        events.append("fused")
        return buffer.combine_output[:1]

    buffer._frontend.run.side_effect = fused
    buffer._frontend.reduce_topk.side_effect = lambda *args: events.append("reduce")
    with mock.patch(
        "torch.cuda.synchronize", side_effect=lambda: events.append("sync")
    ):
        _call(shim, buffer, _output(), sync=sync)
    assert events == (["fused", "reduce", "sync"] if sync else ["fused", "reduce"])


def test_nonempty_capture_after_empty_warmup_fails_before_collective(shim, buffer):
    buffer._frontend._reduce = None
    with (
        mock.patch("torch.cuda.is_current_stream_capturing", return_value=True),
        pytest.raises(RuntimeError, match="warmup.*default batch"),
    ):
        _call(shim, buffer, _output())
    buffer._frontend.run.assert_not_called()
    buffer._frontend.reduce_topk.assert_not_called()


def test_empty_capture_does_not_require_a_compiled_reducer(shim, buffer):
    buffer._frontend._reduce = None
    buffer._frontend.run.return_value = buffer.combine_output[:0]
    with mock.patch("torch.cuda.is_current_stream_capturing", return_value=True):
        _call(shim, buffer, _output(0))
    buffer._frontend.run.assert_called_once()


@pytest.fixture
def symm_factory():
    pytest.importorskip("cutlass")
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import (
        get_symm_buffer_for_w4a16_mega_moe,
    )

    factory = get_symm_buffer_for_w4a16_mega_moe
    with (
        mock.patch(
            f"{factory.__module__}.sym_zeros",
            side_effect=lambda shape, dtype: torch.zeros(shape, dtype=dtype),
        ),
        mock.patch(f"{factory.__module__}.free_sym_tensor"),
    ):
        yield factory


@pytest.mark.parametrize("default_reduce", (False, True))
def test_buffer_knobs_override_optional_defaults(symm_factory, default_reduce):
    workspace = symm_factory(
        4,
        4,
        2,
        64,
        64,
        0,
        1,
        gate_up_clamp=2.0,
        in_kernel_fc2_reduce=default_reduce,
        token_back_mode="epi_warps",
        knobs={
            "gate_up_clamp": 1.5,
            "in_kernel_fc2_reduce": False,
            "token_back_mode": "reuse_dispatch_warps",
        },
    )
    try:
        config = workspace._frontend.config
        assert config.gate_up_clamp == 1.5
        assert config.token_back_mode == "reuse_dispatch_warps"
        assert not config.in_kernel_fc2_reduce
        assert workspace.combine_output.shape == (4, 2, 64)
    finally:
        workspace.destroy()


def test_buffer_knobs_reject_routing_reduction(symm_factory):
    with pytest.raises(ValueError, match="routing scores are applied after FC2"):
        symm_factory(4, 4, 2, 64, 64, 0, 1, knobs={"in_kernel_fc2_reduce": True})


@pytest.mark.parametrize("field", ("unknown_knob", "rank", "world_size", "hidden"))
def test_buffer_knobs_cannot_replace_required_geometry(symm_factory, field):
    message = "unexpected keyword" if field == "unknown_knob" else "multiple values"
    with pytest.raises(TypeError, match=message):
        symm_factory(4, 4, 2, 64, 64, 0, 1, knobs={field: 1})


@pytest.mark.parametrize("hidden,intermediate", [(64, 64), (192, 320), (7168, 2048)])
def test_tmem_config_preserves_public_geometry_and_swapped_knobs(hidden, intermediate):
    pytest.importorskip("cutlass")
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim.w4a16.frontend import (
        MegaMoEW4A16Config,
        MegaMoEW4A16Frontend,
    )

    config = MegaMoEW4A16Config(
        rank=0,
        world_size=1,
        num_tokens_per_rank=4,
        num_topk=2,
        num_total_experts=4,
        hidden=hidden,
        intermediate=intermediate,
    )
    assert config.mma_tiler_mnk == (256, 128, 256)
    frontend = MegaMoEW4A16Frontend(config)
    frontend.apply_knobs({"mma_tiler_mnk": (256, 128, 256), "group_hint": 512})
    assert frontend.config.group_hint == 512
    with pytest.raises(ValueError, match="mma_tiler_mnk"):
        frontend.apply_knobs({"mma_tiler_mnk": (256, 256, 64)})


def test_frontend_validates_native_flat_scale_storage_before_compile():
    pytest.importorskip("cutlass")
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim.w4a16.frontend import (
        MegaMoEW4A16Config,
        MegaMoEW4A16Frontend,
        MegaMoEW4A16Inputs,
    )

    config = MegaMoEW4A16Config(
        rank=0,
        world_size=1,
        num_tokens_per_rank=4,
        num_topk=2,
        num_total_experts=4,
        hidden=64,
        intermediate=64,
    )

    def tensor(shape, dtype):
        result = mock.Mock(spec=torch.Tensor)
        result.shape = shape
        result.dtype = dtype
        result.is_cuda = True
        result.is_contiguous.return_value = True
        return result

    inputs = MegaMoEW4A16Inputs(
        tensor((4, 64), torch.bfloat16),
        tensor((4, 2), torch.int64),
        tensor((4, 2), torch.float32),
        tensor((4, 128, 32), torch.uint8),
        tensor((2048,), torch.float8_e4m3fn),
        tensor((4,), torch.float32),
        tensor((4, 64, 32), torch.uint8),
        tensor((2048,), torch.float8_e4m3fn),
        tensor((4,), torch.float32),
        tensor((4, 2, 64), torch.bfloat16),
    )
    frontend = MegaMoEW4A16Frontend(config)
    frontend._validate(inputs, 1)
    inputs.fc2_weight_sf.shape = (4, 64, 4)
    with pytest.raises(ValueError, match="native flat E4M3"):
        frontend._validate(inputs, 1)


@pytest.mark.parametrize("hidden,intermediate", ((32, 64), (288, 448)))
@pytest.mark.parametrize("mode", ("epi_warps", "reuse_dispatch_warps"))
@pytest.mark.parametrize(
    "clamp,epi_flags", ((None, (1, 1)), (1.5, (2, 4)), (2.0, (32, 32)))
)
def test_tmem_kernel_preserves_public_knob_contract(
    symm_factory, hidden, intermediate, mode, clamp, epi_flags
):
    import cutlass
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim.w4a16.kernel import (
        Sm100W4A16MegaMoEKernel,
    )
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim.w4a16.epilogue import (
        W4A16Epilogue,
    )

    workspace = symm_factory(
        4,
        257,
        2,
        hidden,
        intermediate,
        0,
        1,
        knobs={
            "gate_up_clamp": clamp,
            "token_back_mode": mode,
            "epi_flag_batch": epi_flags,
            "load_balance_mode": "atomic_counter",
        },
    )
    try:
        config = workspace._frontend.config
        kernel = Sm100W4A16MegaMoEKernel(
            local_rank=config.rank,
            mma_tiler_mnk=config.mma_tiler_mnk,
            cluster_shape_mnk=config.cluster_shape_mnk,
            use_2cta_instrs=config.use_2cta_instrs,
            group_hint=512,
            token_padding_block=128,
            load_balance_mode=config.load_balance_mode,
            static_expert_shape=(4, 2 * intermediate, hidden),
            force_static_sched=True,
            world_size=1,
            num_topk=2,
            max_tokens_per_rank=257,
            hidden=hidden,
            token_back_mode=config.token_back_mode,
            token_back_by_dispatch=mode == "reuse_dispatch_warps",
            gate_up_clamp=config.gate_up_clamp,
            epi_flag_batch=config.epi_flag_batch,
        )
        epi = W4A16Epilogue(
            mma_tiler_mnk=config.mma_tiler_mnk,
            cluster_shape_mn=(2, 1),
            use_2cta_instrs=True,
            fc1_output_dtype=cutlass.BFloat16,
            combine_format=kernel.combine_format,
            static_expert_shape=(4, 2 * intermediate, hidden),
            token_back_by_dispatch=kernel.token_back_by_dispatch,
            gate_up_clamp=kernel.gate_up_clamp,
            epi_flag_batch=kernel.epi_flag_batch,
        )
        assert (epi.fc1_epi_flag_batch, epi.fc2_epi_flag_batch) == epi_flags
        assert epi.gate_up_clamp == clamp
        assert epi.epi_smem_bytes == 0 and epi.acc_sf_cols == 0
        assert not epi.reduce_topk_in_kernel
        assert kernel.token_comm.num_total_threads == 512
        assert kernel.token_comm.sf_uint32_per_token == 0
        by_dispatch = mode == "reuse_dispatch_warps"
        assert epi.token_back_by_dispatch == by_dispatch
        regions = kernel._local_region_by_name
        assert ("fc2_output_workspace" in regions) == by_dispatch
        assert ("fc2_done_counter" in regions) == by_dispatch
        if by_dispatch:
            assert regions["fc2_output_workspace"].cute_dtype is cutlass.BFloat16
            assert kernel.token_comm.fc2_publishes_per_token_cluster_tile == (
                2 * ((hidden + 255) // 256)
            )
            assert kernel.token_comm.token_back_schedule_mode == "atomic_counter"
            assert kernel._local_offsets["fc2_done_counter"] + 16 <= (
                kernel.local_zero_i32_count * 4
            )
    finally:
        workspace.destroy()
