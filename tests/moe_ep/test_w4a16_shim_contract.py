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


def test_clc_config_and_bundle_have_distinct_compile_keys():
    import dataclasses

    pytest.importorskip("cutlass")
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import (
        MegaMoEW4A16Config,
        MegaMoEW4A16Frontend,
    )

    base = MegaMoEW4A16Config(
        rank=0,
        world_size=1,
        num_tokens_per_rank=4,
        num_topk=2,
        num_total_experts=4,
        hidden=64,
        intermediate=64,
    )
    assert base.load_balance_mode == "static" and base.clc_bundle_size is None
    configs = [base, dataclasses.replace(base, load_balance_mode="atomic_counter")]
    configs += [
        dataclasses.replace(base, load_balance_mode="clc", clc_bundle_size=b)
        for b in (1, 3)
    ]
    keys = [MegaMoEW4A16Frontend(config)._compile_key() for config in configs]
    assert len(set(keys)) == len(configs)
    assert all(config.mma_tiler_mnk == base.mma_tiler_mnk for config in configs)
    assert all(config.epi_flag_batch == base.epi_flag_batch for config in configs)


@pytest.mark.parametrize("bundle", (0, -1, 1.5, True))
def test_clc_rejects_invalid_bundle_before_compile(bundle):
    pytest.importorskip("cutlass")
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import (
        MegaMoEW4A16Config,
    )

    with pytest.raises(ValueError, match="positive integer"):
        MegaMoEW4A16Config(
            rank=0,
            world_size=1,
            num_tokens_per_rank=4,
            num_topk=2,
            num_total_experts=4,
            hidden=64,
            intermediate=64,
            load_balance_mode="clc",
            clc_bundle_size=bundle,
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
def symm_factory(monkeypatch):
    # These tests assert built-in defaults; cache lookup has its own isolated tests.
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "0")
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


@pytest.mark.parametrize("num_tokens", (1, 4096))
@pytest.mark.parametrize(
    ("knobs", "token_back_mode", "expected"),
    (
        (None, "epi_warps", (512, 4, (2, 4), "atomic_counter", "epi_warps")),
        (
            None,
            "reuse_dispatch_warps",
            (512, 4, (2, 4), "atomic_counter", "reuse_dispatch_warps"),
        ),
        ({}, "epi_warps", (None, 1, (1, 1), "static", "epi_warps")),
        (
            {
                "group_hint": 64,
                "flag_batch": 8,
                "token_back_mode": "reuse_dispatch_warps",
            },
            "epi_warps",
            (64, 8, (1, 1), "static", "reuse_dispatch_warps"),
        ),
    ),
    ids=("default", "named_return", "empty", "explicit"),
)
def test_buffer_default_profile_and_explicit_overrides(
    symm_factory, num_tokens, knobs, token_back_mode, expected
):
    original = None if knobs is None else dict(knobs)
    workspace = symm_factory(
        4,
        num_tokens,
        2,
        64,
        64,
        0,
        1,
        gate_up_clamp=1.5,
        token_back_mode=token_back_mode,
        knobs=knobs,
    )
    try:
        config = workspace._frontend.config
        assert (
            config.group_hint,
            config.flag_batch,
            config.epi_flag_batch,
            config.load_balance_mode,
            config.token_back_mode,
        ) == expected
        assert config.mma_tiler_mnk == (256, 128, 256)
        assert config.cluster_shape_mnk == (2, 1, 1)
        assert config.gate_up_clamp == 1.5
        assert not config.apply_topk_in_fc1 and not config.in_kernel_fc2_reduce
        assert knobs == original
    finally:
        workspace.destroy()


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
@pytest.mark.parametrize(
    "tile", ((128, 64, 256), (128, 128, 256), (256, 64, 256), (256, 128, 256))
)
def test_tmem_config_preserves_public_geometry_and_swapped_knobs(
    hidden, intermediate, tile
):
    pytest.importorskip("cutlass")
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import (
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
    # Benchmark JSON arrays must canonicalize before configuration/cache use.
    frontend.apply_knobs(
        {
            "mma_tiler_mnk": list(tile),
            "cluster_shape_mnk": [2, 1, 1],
            "use_2cta_instrs": tile[0] == 256,
            "group_hint": 512,
        }
    )
    assert frontend.config.mma_tiler_mnk == tile
    assert frontend.config.cluster_shape_mnk == (2, 1, 1)
    assert frontend.config.use_2cta_instrs == (tile[0] == 256)
    assert frontend.config.group_hint == 512
    hash(frontend._compile_key())
    with pytest.raises(ValueError, match="mma_tiler_mnk"):
        frontend.apply_knobs({"mma_tiler_mnk": (256, 256, 64)})


@pytest.mark.parametrize("tile", ((128, 64, 256), (256, 64, 256)))
def test_buffer_geometry_rejects_mismatched_mma_instruction_group(symm_factory, tile):
    with pytest.raises(ValueError, match="M128/M256 requires one/two-CTA"):
        symm_factory(
            4,
            4,
            2,
            64,
            64,
            0,
            1,
            knobs={"mma_tiler_mnk": list(tile), "use_2cta_instrs": tile[0] != 256},
        )


@pytest.mark.parametrize("scale_dtype", (torch.float8_e4m3fn, torch.uint8))
@pytest.mark.parametrize(
    "weight_dtype",
    (torch.uint8, getattr(torch, "float4_e2m1fn_x2", torch.uint8)),
)
def test_frontend_validates_shared_nvfp4_layout_before_compile(
    weight_dtype, scale_dtype
):
    pytest.importorskip("cutlass")
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import (
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
        result.transpose.return_value.is_contiguous.return_value = True
        return result

    inputs = MegaMoEW4A16Inputs(
        tensor((4, 64), torch.bfloat16),
        tensor((4, 2), torch.int64),
        tensor((4, 2), torch.float32),
        tensor((4, 32, 128), weight_dtype),
        tensor((4, 512), scale_dtype),
        tensor((4,), torch.float32),
        tensor((4, 32, 64), weight_dtype),
        tensor((4, 512), scale_dtype),
        tensor((4,), torch.float32),
        tensor((4, 2, 64), torch.bfloat16),
    )
    frontend = MegaMoEW4A16Frontend(config)
    frontend._validate(inputs, 1)
    # The old flattened layout and unswizzled scale planes are not prepared SF.
    for invalid_shape in ((2048,), (4, 64, 4)):
        inputs.fc2_weight_sf.shape = invalid_shape
        with pytest.raises(ValueError, match="native per-expert E4M3"):
            frontend._validate(inputs, 1)
    inputs.fc2_weight_sf.shape = (4, 512)
    inputs.fc2_weight.transpose.return_value.is_contiguous.return_value = False
    with pytest.raises(ValueError, match="K-major backing"):
        frontend._validate(inputs, 1)
    inputs.fc2_weight.transpose.return_value.is_contiguous.return_value = True
    inputs.fc2_weight_sf.is_contiguous.return_value = False
    with pytest.raises(ValueError, match="contiguous CUDA"):
        frontend._validate(inputs, 1)


@pytest.mark.parametrize("hidden,intermediate", ((32, 64), (288, 448)))
@pytest.mark.parametrize(
    "tile,cluster", (((256, 128, 256), (2, 1, 1)), ((128, 64, 256), (1, 1, 1)))
)
@pytest.mark.parametrize("mode", ("epi_warps", "reuse_dispatch_warps"))
@pytest.mark.parametrize(
    "clamp,epi_flags", ((None, (1, 1)), (1.5, (2, 4)), (2.0, (32, 32)))
)
def test_tmem_kernel_preserves_public_knob_contract(
    symm_factory, hidden, intermediate, tile, cluster, mode, clamp, epi_flags
):
    import cutlass
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.src.moe_nvfp4_w4a16.megamoe_kernel import (
        Sm100W4A16MegaMoEKernel,
    )
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.src.moe_nvfp4_w4a16.epilogue import (
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
            "mma_tiler_mnk": tile,
            "cluster_shape_mnk": cluster,
            "use_2cta_instrs": tile[0] == 256,
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
            token_padding_block=tile[1],
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
            cluster_shape_mn=cluster[:2],
            use_2cta_instrs=config.use_2cta_instrs,
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
        assert kernel.num_transform_warpgroups == 2
        assert kernel.token_comm.num_total_threads == 640
        assert kernel.token_comm.sf_uint32_per_token == 0
        by_dispatch = mode == "reuse_dispatch_warps"
        assert epi.token_back_by_dispatch == by_dispatch
        regions = kernel._local_region_by_name
        assert ("fc2_output_workspace" in regions) == by_dispatch
        assert ("fc2_done_counter" in regions) == by_dispatch
        if by_dispatch:
            assert regions["fc2_output_workspace"].cute_dtype is cutlass.BFloat16
            expected_publishes = {(32, 2): 2, (288, 2): 4, (32, 1): 1, (288, 1): 3}
            assert (
                kernel.token_comm.fc2_publishes_per_token_cluster_tile
                == (expected_publishes[hidden, cluster[0]])
            )
            assert kernel.token_comm.token_back_schedule_mode == "atomic_counter"
            assert kernel._local_offsets["fc2_done_counter"] + 16 <= (
                kernel.local_zero_i32_count * 4
            )
    finally:
        workspace.destroy()


@pytest.mark.parametrize("tile", ((128, 128, 256), (256, 64, 256), (256, 128, 256)))
def test_cluster_one_rejects_geometries_outside_curated_support(symm_factory, tile):
    with pytest.raises(ValueError, match="cluster"):
        symm_factory(
            4,
            4,
            2,
            64,
            64,
            0,
            1,
            knobs={
                "mma_tiler_mnk": tile,
                "cluster_shape_mnk": (1, 1, 1),
                "use_2cta_instrs": tile[0] == 256,
            },
        )
