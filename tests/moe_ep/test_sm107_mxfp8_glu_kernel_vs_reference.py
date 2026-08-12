"""Single-GPU torch-oracle correctness for the SM107 (Rubin) mxfp8 GLU fprop kernel.

Drives the vendored ``next_cutedsl_megamoe`` drop's fused mega kernel through
the shim allocator + compute entry on ONE Rubin GPU (``MEGA_NO_DIST=1``,
world_size 1: every "peer" resolves to the local buffer, no NVSHMEM), and
compares against the shim's pure-torch reference over the SAME staged mxfp8
payloads.

Process isolation: the drop is imported only inside test bodies, this file is
excluded from ``run_unit``, and runs via ``run_tests.sh oracle_sm107``.
Direct invocation::

    MEGA_NO_DIST=1 CUDA_VISIBLE_DEVICES=0 python -m pytest \
        tests/moe_ep/test_sm107_mxfp8_glu_kernel_vs_reference.py -v -m arch_rubin

The torch reference emulates the in-kernel FC2-input requantization but not
the instruction-exact rcp/ex2 sequences, so comparisons use tolerance bands
(rel_l2), never bitwise.
"""

from __future__ import annotations

import pytest
import torch


def _sm107_tree():
    """Import the drop package, skipping if something shadows ``sources``."""
    try:
        import flashinfer.moe_ep.kernel_src.next_cutedsl_megamoe as pkg
    except RuntimeError as exc:
        pytest.skip(f"next_cutedsl_megamoe tree unavailable in this process: {exc}")
    return pkg


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")


def _single_rank_problem():
    """Small geometry scaled so |y| ~ O(1) (the tolerance bands are relative,
    but keeping activations in range avoids fp8 saturation noise)."""
    torch.manual_seed(1234)
    hidden, intermediate = 512, 256
    num_experts, top_k = 4, 4
    num_tokens, max_tokens = 96, 128
    x = torch.randn(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
    w13 = (
        torch.randn(num_experts, 2 * intermediate, hidden, device="cuda", dtype=torch.float32)
        * hidden**-0.5
    ).to(torch.bfloat16)
    w2 = (
        torch.randn(num_experts, hidden, intermediate, device="cuda", dtype=torch.float32)
        * intermediate**-0.5
    ).to(torch.bfloat16)
    topk_ids = torch.stack(
        [torch.randperm(num_experts, device="cuda")[:top_k] for _ in range(num_tokens)]
    ).to(torch.int32)
    topk_weights = torch.softmax(
        torch.randn(num_tokens, top_k, device="cuda"), dim=-1
    )
    return dict(
        hidden=hidden,
        intermediate=intermediate,
        num_experts=num_experts,
        top_k=top_k,
        num_tokens=num_tokens,
        max_tokens=max_tokens,
        x=x,
        w13=w13,
        w2=w2,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
    )


@pytest.mark.arch_rubin
def test_shim_config_validation():
    """Host-side ``Sm107MegaMoEMxfp8GluConfig`` invariants (no compile)."""
    pkg = _sm107_tree()
    cfg_cls = pkg.Sm107MegaMoEMxfp8GluConfig

    kwargs = dict(
        num_total_experts=8,
        max_tokens_per_rank=128,
        num_topk=4,
        hidden=512,
        intermediate=256,
        rank=0,
        world_size=1,
    )
    cfg = cfg_cls(**kwargs)
    assert cfg.experts_per_rank == 8
    assert cfg.torch_data_dtype == torch.float8_e4m3fn

    with pytest.raises(ValueError, match="kind"):
        cfg_cls(**{**kwargs, "kind": "nvfp4"})
    with pytest.raises(ValueError, match="divide"):
        cfg_cls(**{**kwargs, "num_total_experts": 7, "world_size": 2})
    with pytest.raises(ValueError, match="multiple of 32"):
        cfg_cls(**{**kwargs, "hidden": 500})
    with pytest.raises(ValueError, match="apply_topk_in_fc1"):
        cfg_cls(**{**kwargs, "in_kernel_fc2_reduce": True, "apply_topk_in_fc1": False})
    with pytest.raises(ValueError, match="K must be one"):
        cfg_cls(**{**kwargs, "cluster_shape_mnk": (2, 1, 2)})


@pytest.mark.arch_rubin
@pytest.mark.parametrize("apply_topk_in_fc1", [True, False])
def test_sm107_glu_kernel_matches_torch_reference(monkeypatch, apply_topk_in_fc1):
    """Fused kernel vs pure-torch oracle over identical staged payloads."""
    pkg = _sm107_tree()
    _require_cuda()
    # monkeypatch (not os.environ): restored after the test, so it cannot
    # silently downgrade later nvshmem-path tests in the same process.
    monkeypatch.setenv("MEGA_NO_DIST", "1")

    from flashinfer.moe_ep.backends.mega.kernel.sm107.mxfp8_mxfp8_bf16_cutedsl.staging import (
        stage_mega_moe_inputs,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm107.mxfp8_mxfp8_bf16_cutedsl.weights import (
        preprocess_mega_weights,
    )
    from flashinfer.moe_ep.weights import MoEWeightPack

    p = _single_rank_problem()

    transformed = preprocess_mega_weights(
        MoEWeightPack(w13=p["w13"], w2=p["w2"]),
        intermediate_size=p["intermediate"],
        hidden_size=p["hidden"],
    )

    symm_buffer = pkg.get_symm_buffer_for_sm107_mxfp8_glu_mega_moe(
        p["num_experts"],
        p["max_tokens"],
        p["top_k"],
        p["hidden"],
        p["intermediate"],
        0,
        1,
        apply_topk_in_fc1=apply_topk_in_fc1,
    )
    try:
        staged = stage_mega_moe_inputs(
            p["x"],
            p["topk_weights"],
            p["topk_ids"],
            symm_buffer.x,
            symm_buffer.x_sf,
            symm_buffer.topk_idx,
            symm_buffer.topk_weights,
        )
        symm_buffer.note_staged_tokens(staged)

        y = torch.empty(
            p["num_tokens"], p["hidden"], device="cuda", dtype=torch.bfloat16
        )
        pkg.sm107_mxfp8_glu_mega_moe(
            y, transformed[0], transformed[1], symm_buffer, num_tokens=p["num_tokens"]
        )

        # Reference over the SAME staged fp8 payloads and RAW (pre-swizzle)
        # weight scales, rebuilt from the bf16 originals with the identical
        # quantization path the preprocess used.
        from flashinfer.moe_ep.kernel_src.next_cutedsl_megamoe import (
            interleave_gate_up_32,
            quantize_mxfp8_block32,
        )

        w13_q, w13_sf = quantize_mxfp8_block32(
            interleave_gate_up_32(
                p["w13"].to(torch.float32), intermediate_size=p["intermediate"]
            ).contiguous(),
            torch.float8_e4m3fn,
        )
        w2_q, w2_sf = quantize_mxfp8_block32(
            p["w2"].to(torch.float32).contiguous(), torch.float8_e4m3fn
        )
        y_ref = pkg.compute_megamoe_reference_sm107_glu(
            symm_buffer.x,
            symm_buffer.x_sf,
            symm_buffer.topk_idx,
            symm_buffer.topk_weights,
            w13_q.permute(0, 2, 1),
            w13_sf,
            w2_q.permute(0, 2, 1),
            w2_sf,
            local_expert_offset=0,
            gate_up_clamp=None,
            apply_topk_in_fc1=apply_topk_in_fc1,
            num_tokens=p["num_tokens"],
        )[: p["num_tokens"]]

        yk = y.to(torch.float32)
        yr = y_ref.to(torch.float32)
        rel_l2 = (yk - yr).norm() / yr.norm().clamp_min(1e-6)
        max_abs = (yk - yr).abs().max().item()
        print(
            f"[sm107 oracle] apply_topk_in_fc1={apply_topk_in_fc1} "
            f"rel_l2={rel_l2.item():.5f} max|d|={max_abs:.5f} "
            f"amax={yr.abs().max().item():.3f}"
        )
        assert rel_l2.item() < 0.02, f"rel_l2 {rel_l2.item()} out of band"
        torch.testing.assert_close(yk, yr, atol=1e-1, rtol=5e-2)

        # Second forward on the same session: regression guard for stale
        # workspace / launch-kwargs reuse.
        pkg.sm107_mxfp8_glu_mega_moe(
            y, transformed[0], transformed[1], symm_buffer, num_tokens=p["num_tokens"]
        )
        rel_l2_2 = (y.to(torch.float32) - yr).norm() / yr.norm().clamp_min(1e-6)
        assert rel_l2_2.item() < 0.02, f"second forward rel_l2 {rel_l2_2.item()}"
    finally:
        symm_buffer.destroy()


@pytest.mark.arch_rubin
def test_sm107_glu_preprocess_weight_shapes():
    """Kernel-layout invariants of the weight transform (no compile)."""
    pkg = _sm107_tree()
    _require_cuda()

    from flashinfer.moe_ep.backends.mega.kernel.sm107.mxfp8_mxfp8_bf16_cutedsl.weights import (
        preprocess_mega_weights,
        validate_transformed_mega_weights,
    )
    from flashinfer.moe_ep.weights import MoEWeightPack

    E, hidden, intermediate = 4, 256, 128
    w13 = torch.randn(E, 2 * intermediate, hidden, device="cuda", dtype=torch.bfloat16)
    w2 = torch.randn(E, hidden, intermediate, device="cuda", dtype=torch.bfloat16)
    (fc1_w, fc1_sf), (fc2_w, fc2_sf) = preprocess_mega_weights(
        MoEWeightPack(w13=w13, w2=w2),
        intermediate_size=intermediate,
        hidden_size=hidden,
    )
    assert fc1_w.shape == (E, hidden, 2 * intermediate)
    assert fc1_w.stride(1) == 1  # K (hidden) innermost
    assert fc2_w.shape == (E, intermediate, hidden)
    assert fc2_w.stride(1) == 1  # K (intermediate) innermost
    assert fc1_sf.shape == (
        E,
        pkg.swizzled_flat_sf_size(2 * intermediate, hidden // pkg.Mxfp8BlockSize),
    )
    assert fc2_sf.shape == (
        E,
        pkg.swizzled_flat_sf_size(hidden, intermediate // pkg.Mxfp8BlockSize),
    )
    validate_transformed_mega_weights(
        ((fc1_w, fc1_sf), (fc2_w, fc2_sf)),
        intermediate_size=intermediate,
        hidden_size=hidden,
        kind="mxfp8_e4m3",
        world_size=1,
        num_experts=E,
    )
