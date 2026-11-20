"""Construction and staged-copy bounds for the exported communication component."""

import pytest

from tests.moe_ep import test_sm107_shared_expert as shared

ep_runtime = shared.ep_runtime


def _component(
    *,
    kind="nvfp4",
    hidden=2048,
    depths=(1, 2),
    group=1,
    mode="reuse_dispatch_warps",
    split=False,
):
    pytest.importorskip("cutlass")
    from flashinfer.moe_ep.kernels.cutedsl_megamoe.sources.api import (
        ImplDesc,
        ProblemDesc,
    )
    from flashinfer.moe_ep.kernels.cutedsl_megamoe.sources.communication.nvlink_domain.token_comm import (
        TokenCommNonDeterministic,
    )
    from flashinfer.moe_ep.kernels.cutedsl_megamoe.sources.quant_def import (
        CombineFormat,
    )

    return TokenCommNonDeterministic(
        ProblemDesc(
            dict(
                world_size=4,
                expert_count=32,
                topk=4,
                max_tokens_per_rank=32,
                hidden_size=hidden,
                quant_kind=kind,
                combine_format=CombineFormat.parse("bf16"),
                apply_topk_at_fc1=False,
            )
        ),
        ImplDesc(
            dict(
                token_padding_block=128,
                sf_padding_block=128,
                tokens_per_fc1_ready_slot=128,
                fc2_done_signals_per_token_tile=1,
                promised_launchable_sm_count=32,
                token_in_flag_batch=2,
                token_back_mode=mode,
                token_back_schedule_mode="static",
                reduce_topk_in_kernel=False,
                comm_stage_depths=depths,
                dispatch_sf_group=group,
                comm_cta_count=4 if split else 0,
                comm_warps=(4, 4),
            )
        ),
    )


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("depth,group", [(2, 1), (4, 1), (2, 4), (6, 4), (8, 8)])
def test_multistage_dispatch_rejects_incomplete_scale_groups(split, depth, group):
    with pytest.raises(ValueError, match="dispatch_sf_group=4"):
        _component(depths=(depth, 1), group=group, split=split)


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("depth", [4, 8])
def test_multistage_dispatch_accepts_complete_scale_groups(split, depth):
    component = _component(depths=(depth, 1), group=4, split=split)
    assert component.dispatch_stage_depth == depth


@pytest.mark.parametrize("kind", ["nvfp4", "mxfp8_e4m3"])
@pytest.mark.parametrize("hidden", [512, 2048, 3072])
@pytest.mark.parametrize("mode", ["reuse_dispatch_warps", "standalone_warps"])
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_token_back_chunks_fit_registered_stages(kind, hidden, mode, depth):
    component = _component(kind=kind, hidden=hidden, depths=(1, depth), mode=mode)
    from flashinfer.moe_ep.kernels.cutedsl_megamoe.sources.helpers.smem_workspace import (
        SmemWorkspace,
    )

    workspace = SmemWorkspace()
    component._register_token_back_staging(workspace)
    stage = workspace.region(component.token_back_activation_smem_region)
    chunk = component.token_back_staging_bytes
    assert stage.nbytes == component.combine_warp_count * depth * chunk
    assert chunk == (
        2048
        if mode == "standalone_warps"
        else hidden // 2
        if kind == "nvfp4"
        else hidden
    )


@pytest.mark.arch_rubin
@pytest.mark.gpu_2
@pytest.mark.parametrize(
    "kind,hidden,depths,mode",
    [
        ("nvfp4", 512, (1, 2), "reuse_dispatch_warps"),
        ("nvfp4", 2048, (1, 3), "reuse_dispatch_warps"),
        ("mxfp8_e4m3", 3072, (1, 2), "reuse_dispatch_warps"),
        ("nvfp4", 512, (1, 2), "standalone_warps"),
        ("nvfp4", 2048, (4, 2), "reuse_dispatch_warps"),
    ],
)
def test_native_multistage_copies_graph(
    ep_runtime, monkeypatch, kind, hidden, depths, mode
):
    from flashinfer.moe_ep.kernels import cutedsl_megamoe as src
    from flashinfer.moe_ep.kernels.cutedsl_megamoe.sources.api import ImplDesc

    construct = src.RubinInferenceMegaMoE

    def staged(problem, impl):
        kernel = construct(
            problem,
            ImplDesc(
                dict(impl)
                | dict(
                    comm_stage_depths=depths,
                    dispatch_sf_group=4 if depths[0] > 1 else 1,
                )
            ),
        )
        assert kernel.token_comm.comm_stage_depths == depths
        return kernel

    monkeypatch.setattr(src, "RubinInferenceMegaMoE", staged)
    shared.test_public_shared_and_dedup_graph(
        ep_runtime,
        kind,
        "inference",
        False,
        False,
        False,
        False,
        "mxfp8_e4m3",
        "swiglu",
        geometry=(hidden, 256, 32, 32, 4, 128),
        return_mode=mode,
    )
