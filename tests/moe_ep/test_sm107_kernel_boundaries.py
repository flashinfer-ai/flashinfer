"""Rubin correctness at routing vector tails, partial tiles, and knob boundaries."""

import pytest
import torch

from flashinfer.moe_ep.kernel_src.sm107 import next_cutedsl_megamoe as pkg

KINDS = ("nvfp4", "mxfp8_e4m3", "mxfp8_e5m2")
pytestmark = pytest.mark.arch_rubin


def _run_case(kind, capacity, topk, knobs):
    from flashinfer.moe_ep.backends.mega.kernel.sm107.mxfp8_mxfp8_bf16_cutedsl.staging import (
        stage_mega_moe_inputs as stage_mx,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm107.nvfp4_nvfp4_bf16_cutedsl.staging import (
        stage_mega_moe_inputs as stage_nv,
    )

    hidden, intermediate = (192, 64) if kind == "nvfp4" else (384, 192)
    if kind == "mxfp4_mxfp8":
        intermediate = 256
    torch.manual_seed(1973)
    w13 = (
        torch.randn(8, 2 * intermediate, hidden, device="cuda") / hidden**0.5
    ).bfloat16()
    w2 = (
        torch.randn(8, hidden, intermediate, device="cuda") / intermediate**0.5
    ).bfloat16()
    weights = pkg.preprocess_block_scaled_weights(
        w13, w2, quant_kind=kind, intermediate_size=intermediate
    )
    ws = pkg.get_symm_buffer_for_sm107_block_scaled_mega_moe(
        8,
        capacity,
        topk,
        hidden,
        intermediate,
        0,
        1,
        quant_kind=kind,
        **knobs,
    )
    try:
        x = torch.randn(capacity, hidden, device="cuda").bfloat16()
        for iteration in range(3):
            ids = (
                torch.arange(capacity * topk, device="cuda", dtype=torch.int32).view(
                    capacity, topk
                )
                % 8
            )
            if iteration:
                ids[::3, 0] = -1
            scores = torch.full((capacity, topk), 1 / topk, device="cuda")
            stage = stage_nv if kind == "nvfp4" else stage_mx
            extra = {} if kind == "nvfp4" else {"kind": kind}
            count = stage(
                torch.zeros_like(x) if iteration == 2 else x * (1 + iteration * 0.25),
                scores,
                ids,
                ws.x,
                ws.x_sf,
                ws.topk_idx,
                ws.topk_weights,
                **extra,
            )
            ws.note_staged_tokens(count)
            y = pkg.sm107_block_scaled_mega_moe(None, *weights, ws, num_tokens=count)
            indices, expected = pkg.sampled_reference(
                ws, *weights, count, sample_size=capacity
            )
            error = pkg.output_error(y, indices, expected)
            assert error < (0.06 if kind == "nvfp4" else 0.02), (
                kind,
                knobs,
                capacity,
                topk,
                error,
            )
    finally:
        ws.destroy()


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    "capacity, topk", [(1, 1), (1, 3), (3, 2), (5, 3), (127, 6), (129, 1)]
)
def test_router_vector_tails(monkeypatch, kind, capacity, topk):
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    _run_case(kind, capacity, topk, {})


@pytest.mark.parametrize("kind", (*KINDS, "mxfp4_mxfp8"))
@pytest.mark.parametrize(
    "variant", ["one_cta", "deep_k", "mixed_ikr", "late_weight_clamp"]
)
def test_supported_knob_variants(monkeypatch, kind, variant):
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    ik = 128 if kind == "nvfp4" else 64
    variants = {
        "one_cta": dict(
            mma_tiler_mnk=(128, 64, 2 * ik),
            cluster_shape_mn=(1, 1),
            token_back_mode="standalone_warps",
            epi_flag_batches=(1, 1),
            token_in_flag_batch=4,
        ),
        "deep_k": dict(
            mma_tiler_mnk=(256, 128, 4 * ik),
            token_back_mode="reuse_dispatch_warps",
            fc2_use_bulk=True,
            fc2_tma_stages=1,
            epi_flag_batches=(4, 4),
            token_in_flag_batch=32,
        ),
        "mixed_ikr": dict(
            mma_tiler_mnk=(256, 256, 2 * ik),
            cluster_shape_mn=(4, 1),
            fallback_cluster_shape_mn=(2, 1),
            schedule_policy=("phase_interleave", None),
            work_id_mode="atomic_counter",
            fc2_use_bulk=True,
            fc2_tma_stages=4,
            reduce_topk_in_kernel=True,
        ),
        "late_weight_clamp": dict(
            apply_routing_weights_before_fc2=False,
            gate_up_clamp=0.75,
        ),
    }
    _run_case(kind, 65, 3, variants[variant])


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("raw_scale_bytes", [False, True])
def test_layer_prequantized_input_and_output_view(monkeypatch, kind, raw_scale_bytes):
    """Check prequantized parity, borrowed output views, and shared-buffer lifetime."""
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEEpTensors,
        MoEWeightPack,
        Sm107_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig,
        Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
    )

    monkeypatch.setenv("MEGA_NO_DIST", "1")
    cfg = (
        Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(intermediate_size=64, top_k=3)
        if kind == "nvfp4"
        else Sm107_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(
            intermediate_size=64, top_k=3, kind=kind
        )
    )
    fp = FleetParams(num_experts=8, max_tokens_per_rank=9, token_hidden_size=128)
    bootstrap = BootstrapConfig(rank=0, world_size=1, auto_bootstrap=False)
    torch.manual_seed(417)
    pack = MoEWeightPack(
        w13=(torch.randn(8, 128, 128, device="cuda") / 128**0.5).bfloat16(),
        w2=(torch.randn(8, 128, 64, device="cuda") / 64**0.5).bfloat16(),
    )
    eager = MoEEpLayer(
        bootstrap=bootstrap,
        fleet_params=fp,
        weights=pack,
        backend=MegaConfig(megakernel=cfg),
    )
    prestaged = None
    try:
        t = MoEEpTensors(
            torch.randn(8, 128, device="cuda").bfloat16(),
            torch.arange(24, device="cuda", dtype=torch.int32).view(8, 3) % 8,
            torch.full((8, 3), 1 / 3, device="cuda"),
        )
        expected = eager.forward(t)
        xq = eager._workspace.x[:8].clone()
        # Public inputs contain logical scale columns; x_sf also includes
        # private communication padding that is not part of that contract.
        sf_cols = fp.token_hidden_size // (16 if kind == "nvfp4" else 32)
        sf = eager._workspace.x_sf[:8, :sf_cols].clone()
        prestaged = MoEEpLayer(
            bootstrap=bootstrap,
            fleet_params=fp,
            weights=None,
            backend=MegaConfig(
                megakernel=cfg,
                quantize_input=False,
                transformed_weights=eager._transformed,
            ),
        )
        if raw_scale_bytes:
            sf = sf.view(torch.uint8)
            if kind == "nvfp4":
                xq = xq.view(torch.uint8)
        actual = prestaged.forward(
            MoEEpTensors(xq, t.topk_ids, t.topk_weights, scales=sf),
            return_workspace_view=True,
        )
        assert eager._workspace is prestaged._workspace
        assert actual.data_ptr() == prestaged._workspace.output_activation.data_ptr()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # Owned results survive the next launch, which invalidates borrowed views.
        snapshot = expected.clone()
        t.hidden_states.zero_()
        assert torch.count_nonzero(eager.forward(t)) == 0
        torch.testing.assert_close(expected, snapshot, rtol=0, atol=0)
        eager.destroy()
        # The other layer's reference keeps the shared allocation alive.
        actual = prestaged.forward(
            MoEEpTensors(xq, t.topk_ids, t.topk_weights, scales=sf)
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        eager.destroy()
        if prestaged is not None:
            prestaged.destroy()


@pytest.mark.parametrize("kind", (*KINDS, "mxfp4_mxfp8"))
@pytest.mark.parametrize(
    "family,combine_dtype",
    [
        ("inference", "bf16"),
        ("genphase", "bf16"),
        ("inference", "nvfp4"),
        ("inference", "mxfp8"),
    ],
)
def test_caller_output_rebinds_live_rows_in_graph(
    monkeypatch, kind, family, combine_dtype
):
    """Reuse one compiled kernel across output addresses, row counts and graphs."""
    import cutlass.cute as cute

    from flashinfer.moe_ep.backends.mega.kernel.sm107.mxfp8_mxfp8_bf16_cutedsl.staging import (
        stage_mega_moe_inputs as stage_mx,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm107.nvfp4_nvfp4_bf16_cutedsl.staging import (
        stage_mega_moe_inputs as stage_nv,
    )

    monkeypatch.setenv("MEGA_NO_DIST", "1")
    capacity, hidden, intermediate, topk, experts = 9, 512, 256, 3, 8
    torch.manual_seed(1597)
    weights = pkg.preprocess_block_scaled_weights(
        (
            torch.randn(experts, 2 * intermediate, hidden, device="cuda") / hidden**0.5
        ).bfloat16(),
        (
            torch.randn(experts, hidden, intermediate, device="cuda")
            / intermediate**0.5
        ).bfloat16(),
        quant_kind=kind,
        intermediate_size=intermediate,
    )
    ws = pkg.get_symm_buffer_for_sm107_block_scaled_mega_moe(
        experts,
        capacity,
        topk,
        hidden,
        intermediate,
        0,
        1,
        quant_kind=kind,
        kernel_variant=family,
        fc2_use_bulk=combine_dtype != "nvfp4",
        combine_dtype=combine_dtype,
        **({"cluster_shape_mn": (4, 1)} if family == "genphase" else {}),
    )
    captures = []
    stage = stage_nv if kind == "nvfp4" else stage_mx
    extra = {} if kind == "nvfp4" else {"kind": kind, "num_experts": experts}

    def run(x, ids, scores, y):
        stage(x, scores, ids, ws.x, ws.x_sf, ws.topk_idx, ws.topk_weights, **extra)
        return pkg.sm107_block_scaled_mega_moe(y, *weights, ws, num_tokens=x.shape[0])

    try:
        x = torch.randn(capacity, hidden, device="cuda").bfloat16()
        ids = torch.arange(topk, device="cuda", dtype=torch.int64).repeat(capacity, 1)
        scores = torch.full((capacity, topk), 1 / topk, device="cuda")
        ids[::2, 0] = -1
        scores[::2, 0] = float("nan")
        # Warm the borrowed-output binding, then capture different caller outputs.
        run(x, ids, scores, None)
        torch.cuda.synchronize()

        def unexpected_compile(*args, **kwargs):
            pytest.fail("Changing the output must reuse the warmed kernel")

        monkeypatch.setattr(cute, "compile", unexpected_compile)
        for live, layout in (
            (1, "contiguous"),
            (5, "contiguous"),
            (9, "contiguous"),
            (3, "strided"),
            (2, "unaligned"),
            (0, "contiguous"),
        ):
            if layout == "strided":
                y = torch.full(
                    (live + 2, hidden * 2), 7, device="cuda", dtype=torch.bfloat16
                )[:, ::2]
            elif layout == "unaligned":
                y = torch.full(
                    ((live + 2) * hidden + 1,), 7, device="cuda", dtype=torch.bfloat16
                )[1:].view(live + 2, hidden)
            else:
                y = torch.full(
                    (live + 2, hidden), 7, device="cuda", dtype=torch.bfloat16
                )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run(x[:live], ids[:live], scores[:live], y)
            captures.append((graph, live, y))

        for graph, live, y in reversed(captures):
            graph.replay()
            torch.cuda.synchronize()
            assert (y[live:] == 7).all()
            if live:
                indices, expected = pkg.sampled_reference(
                    ws, *weights, live, sample_size=live
                )
                assert pkg.output_error(y[:live], indices, expected) < (
                    0.06 if kind == "nvfp4" else 0.02
                )
    finally:
        for graph, _, _ in captures:
            graph.reset()
        ws.destroy()
