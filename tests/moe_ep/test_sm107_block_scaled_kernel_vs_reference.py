"""Single-GPU torch-oracle correctness for the SM107 (Rubin) block-scaled kernel.

Drives the vendored ``next_cutedsl_megamoe`` drop's fused inference mega
kernel (``BlockScaledSwapAbMegaMoeKernel``) through the shim allocator +
compute entry on ONE Rubin GPU (``MEGA_NO_DIST=1``, world_size 1: every
"peer" resolves to the local buffer, no NVSHMEM), for BOTH wired quant kinds
(mxfp8_e4m3 and nvfp4), and compares against the shim's pure-torch reference
over the SAME staged quantized payloads.

Process isolation: the drop is imported only inside test bodies, this file is
excluded from ``run_unit``, and runs via ``run_tests.sh oracle_sm107``.
Direct invocation::

    MEGA_NO_DIST=1 CUDA_VISIBLE_DEVICES=0 python -m pytest \
        tests/moe_ep/test_sm107_block_scaled_kernel_vs_reference.py -v -m arch_rubin

The torch reference emulates the in-kernel FC2-input requantization but not
the instruction-exact rcp / E2M1-tie sequences, so comparisons use tolerance
bands (rel_l2), never bitwise.
"""

from __future__ import annotations

import pytest
import torch

QUANT_KINDS = ("mxfp8_e4m3", "nvfp4")


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
    but keeping activations in range avoids fp8/fp4 saturation noise)."""
    torch.manual_seed(1234)
    hidden, intermediate = 512, 256
    num_experts, top_k = 4, 4
    num_tokens, max_tokens = 96, 128
    x = torch.randn(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
    w13 = (
        torch.randn(
            num_experts, 2 * intermediate, hidden, device="cuda", dtype=torch.float32
        )
        * hidden**-0.5
    ).to(torch.bfloat16)
    w2 = (
        torch.randn(
            num_experts, hidden, intermediate, device="cuda", dtype=torch.float32
        )
        * intermediate**-0.5
    ).to(torch.bfloat16)
    topk_ids = torch.stack(
        [torch.randperm(num_experts, device="cuda")[:top_k] for _ in range(num_tokens)]
    ).to(torch.int32)
    topk_weights = torch.softmax(torch.randn(num_tokens, top_k, device="cuda"), dim=-1)
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


# The nvfp4 wire is much coarser (4-bit data, per-16 fp8 scales through TWO
# GEMMs); the mxfp8 band matches the previous GLU-kernel test.
_REL_L2_BAND = {"mxfp8_e4m3": 0.02, "nvfp4": 0.06}


def _backend_modules(quant_kind: str):
    if quant_kind == "nvfp4":
        from flashinfer.moe_ep.backends.mega.kernel.sm107.nvfp4_nvfp4_bf16_cutedsl import (
            staging,
            weights,
        )
    else:
        from flashinfer.moe_ep.backends.mega.kernel.sm107.mxfp8_mxfp8_bf16_cutedsl import (
            staging,
            weights,
        )
    return staging, weights


def _quantize_reference_weights(pkg, p, quant_kind: str):
    """Rebuild the K-major kernel payloads + RAW SF planes the transforms made."""
    w13_interleaved = pkg.interleave_gate_up_16(
        p["w13"].to(torch.float32), intermediate_size=p["intermediate"]
    ).contiguous()
    w2_f32 = p["w2"].to(torch.float32).contiguous()
    if quant_kind == "nvfp4":
        w13_q, w13_sf = pkg.quantize_nvfp4_block16(w13_interleaved)
        w2_q, w2_sf = pkg.quantize_nvfp4_block16(w2_f32)
    else:
        w13_q, w13_sf = pkg.quantize_mxfp8_block32(w13_interleaved, torch.float8_e4m3fn)
        w2_q, w2_sf = pkg.quantize_mxfp8_block32(w2_f32, torch.float8_e4m3fn)
    return w13_q, w13_sf, w2_q, w2_sf


@pytest.mark.arch_rubin
def test_shim_config_validation():
    """Host-side ``Sm107BlockScaledMoeConfig`` invariants (no compile)."""
    pkg = _sm107_tree()
    cfg_cls = pkg.Sm107BlockScaledMoeConfig

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
    assert cfg.sf_vec_size == 32
    assert cfg.instruction_k == 64
    assert cfg.resolved_mma_tiler_mnk == (256, 128, 128)

    nvfp4 = cfg_cls(**{**kwargs, "quant_kind": "nvfp4"})
    assert nvfp4.sf_vec_size == 16
    assert nvfp4.instruction_k == 128
    assert nvfp4.resolved_mma_tiler_mnk == (256, 128, 256)
    assert nvfp4.torch_act_sf_dtype == torch.float8_e4m3fn

    with pytest.raises(ValueError, match="quant_kind"):
        cfg_cls(**{**kwargs, "quant_kind": "int8"})
    with pytest.raises(ValueError, match="divide"):
        cfg_cls(**{**kwargs, "num_total_experts": 7, "world_size": 2})
    with pytest.raises(ValueError, match="multiple of"):
        cfg_cls(**{**kwargs, "hidden": 500})
    with pytest.raises(ValueError, match="apply_topk_at_fc1"):
        cfg_cls(**{**kwargs, "reduce_topk_in_kernel": True, "apply_topk_at_fc1": False})
    with pytest.raises(ValueError, match="tile K"):
        cfg_cls(**{**kwargs, "mma_tiler_mnk": (256, 128, 512)})
    with pytest.raises(ValueError, match="N=256"):
        cfg_cls(**{**kwargs, "quant_kind": "nvfp4", "mma_tiler_mnk": (256, 256, 512)})
    with pytest.raises(ValueError, match="even cluster M"):
        cfg_cls(**{**kwargs, "cluster_shape_mn": (1, 1)})
    with pytest.raises(ValueError, match="token"):
        cfg_cls(
            **{**kwargs, "mma_tiler_mnk": (128, 64, 128), "token_padding_block": 128}
        )
    with pytest.raises(ValueError, match="atomic_counter"):
        cfg_cls(**{**kwargs, "schedule_policy": ("phase_interleave", 2)})
    with pytest.raises(ValueError, match="epi_flag_batches"):
        cfg_cls(**{**kwargs, "epi_flag_batches": (8, 2)})

    # Mixed-CGA (fallback cluster) rules, added with upstream a5b4d33.
    mixed = cfg_cls(
        **{
            **kwargs,
            "cluster_shape_mn": (4, 1),
            "fallback_cluster_shape_mn": (2, 1),
        }
    )
    assert mixed.fallback_cluster_shape_mn == (2, 1)
    # fallback == preferred collapses to a uniform launch (kernel does the same).
    assert (
        cfg_cls(
            **{**kwargs, "fallback_cluster_shape_mn": (2, 1)}
        ).fallback_cluster_shape_mn
        is None
    )
    with pytest.raises(ValueError, match="divisible"):
        cfg_cls(
            **{
                **kwargs,
                "cluster_shape_mn": (4, 1),
                "fallback_cluster_shape_mn": (3, 1),
            }
        )
    with pytest.raises(ValueError, match="even fallback cluster M"):
        cfg_cls(
            **{
                **kwargs,
                "cluster_shape_mn": (4, 1),
                "fallback_cluster_shape_mn": (1, 1),
            }
        )
    with pytest.raises(ValueError, match="cluster N=1"):
        cfg_cls(
            **{
                **kwargs,
                "cluster_shape_mn": (4, 2),
                "fallback_cluster_shape_mn": (2, 2),
            }
        )
    with pytest.raises(ValueError, match="max_sm_count"):
        cfg_cls(
            **{
                **kwargs,
                "cluster_shape_mn": (4, 1),
                "fallback_cluster_shape_mn": (2, 1),
                "max_sm_count": 64,
            }
        )


@pytest.mark.arch_rubin
@pytest.mark.parametrize("quant_kind", QUANT_KINDS)
@pytest.mark.parametrize("apply_topk_at_fc1", [True, False])
def test_sm107_block_scaled_kernel_matches_torch_reference(
    monkeypatch, quant_kind, apply_topk_at_fc1
):
    """Fused kernel vs pure-torch oracle over identical staged payloads."""
    pkg = _sm107_tree()
    _require_cuda()
    # monkeypatch (not os.environ): restored after the test, so it cannot
    # silently downgrade later nvshmem-path tests in the same process.
    monkeypatch.setenv("MEGA_NO_DIST", "1")

    staging_mod, weights_mod = _backend_modules(quant_kind)
    from flashinfer.moe_ep.weights import MoEWeightPack

    p = _single_rank_problem()

    transform_kwargs = {} if quant_kind == "nvfp4" else {"kind": quant_kind}
    transformed = weights_mod.preprocess_mega_weights(
        MoEWeightPack(w13=p["w13"], w2=p["w2"]),
        intermediate_size=p["intermediate"],
        hidden_size=p["hidden"],
        **transform_kwargs,
    )

    symm_buffer = pkg.get_symm_buffer_for_sm107_block_scaled_mega_moe(
        p["num_experts"],
        p["max_tokens"],
        p["top_k"],
        p["hidden"],
        p["intermediate"],
        0,
        1,
        quant_kind=quant_kind,
        apply_topk_at_fc1=apply_topk_at_fc1,
    )
    try:
        stage_kwargs = {} if quant_kind == "nvfp4" else {"kind": quant_kind}
        staged = staging_mod.stage_mega_moe_inputs(
            p["x"],
            p["topk_weights"],
            p["topk_ids"],
            symm_buffer.x,
            symm_buffer.x_sf,
            symm_buffer.topk_idx,
            symm_buffer.topk_weights,
            **stage_kwargs,
        )
        symm_buffer.note_staged_tokens(staged)

        y = torch.empty(
            p["num_tokens"], p["hidden"], device="cuda", dtype=torch.bfloat16
        )
        pkg.sm107_block_scaled_mega_moe(
            y, transformed[0], transformed[1], symm_buffer, num_tokens=p["num_tokens"]
        )

        # Reference over the SAME staged payloads and RAW (pre-swizzle) weight
        # scales, rebuilt from the bf16 originals with the identical
        # quantization path the preprocess used.
        w13_q, w13_sf, w2_q, w2_sf = _quantize_reference_weights(pkg, p, quant_kind)
        y_ref = pkg.compute_megamoe_reference_sm107_block_scaled(
            symm_buffer.x,
            symm_buffer.x_sf,
            symm_buffer.topk_idx,
            symm_buffer.topk_weights,
            w13_q,
            w13_sf,
            w2_q,
            w2_sf,
            quant_kind=quant_kind,
            local_expert_offset=0,
            gate_up_clamp=None,
            apply_topk_at_fc1=apply_topk_at_fc1,
            num_tokens=p["num_tokens"],
        )[: p["num_tokens"]]

        band = _REL_L2_BAND[quant_kind]
        yk = y.to(torch.float32)
        yr = y_ref.to(torch.float32)
        rel_l2 = (yk - yr).norm() / yr.norm().clamp_min(1e-6)
        max_abs = (yk - yr).abs().max().item()
        print(
            f"[sm107 oracle] kind={quant_kind} apply_topk_at_fc1={apply_topk_at_fc1} "
            f"rel_l2={rel_l2.item():.5f} max|d|={max_abs:.5f} "
            f"amax={yr.abs().max().item():.3f}"
        )
        assert rel_l2.item() < band, f"rel_l2 {rel_l2.item()} out of band {band}"

        # Second forward on the same session: regression guard for stale
        # workspace / launch-kwargs reuse.
        pkg.sm107_block_scaled_mega_moe(
            y, transformed[0], transformed[1], symm_buffer, num_tokens=p["num_tokens"]
        )
        rel_l2_2 = (y.to(torch.float32) - yr).norm() / yr.norm().clamp_min(1e-6)
        assert rel_l2_2.item() < band, f"second forward rel_l2 {rel_l2_2.item()}"
    finally:
        symm_buffer.destroy()


@pytest.mark.arch_rubin
@pytest.mark.parametrize("quant_kind", QUANT_KINDS)
def test_sm107_block_scaled_kernel_perf_winner_config(monkeypatch, quant_kind):
    """Oracle check for the upstream perf-report winner knobs (a5b4d33):
    mixed CGA (preferred 4x1, fallback 2x1), phase-interleave scheduling,
    atomic work IDs, FC2 bulk TMA with 2 stages, epi-warp token back."""
    pkg = _sm107_tree()
    _require_cuda()
    monkeypatch.setenv("MEGA_NO_DIST", "1")

    staging_mod, weights_mod = _backend_modules(quant_kind)
    from flashinfer.moe_ep.weights import MoEWeightPack

    p = _single_rank_problem()

    transform_kwargs = {} if quant_kind == "nvfp4" else {"kind": quant_kind}
    transformed = weights_mod.preprocess_mega_weights(
        MoEWeightPack(w13=p["w13"], w2=p["w2"]),
        intermediate_size=p["intermediate"],
        hidden_size=p["hidden"],
        **transform_kwargs,
    )

    symm_buffer = pkg.get_symm_buffer_for_sm107_block_scaled_mega_moe(
        p["num_experts"],
        p["max_tokens"],
        p["top_k"],
        p["hidden"],
        p["intermediate"],
        0,
        1,
        quant_kind=quant_kind,
        cluster_shape_mn=(4, 1),
        fallback_cluster_shape_mn=(2, 1),
        schedule_policy=("phase_interleave", None),  # None -> minimum safe hint
        work_id_mode="atomic_counter",
        fc2_use_bulk=True,
        fc2_tma_stages=2,
        epi_flag_batches=(1, 4),
    )
    try:
        stage_kwargs = {} if quant_kind == "nvfp4" else {"kind": quant_kind}
        staged = staging_mod.stage_mega_moe_inputs(
            p["x"],
            p["topk_weights"],
            p["topk_ids"],
            symm_buffer.x,
            symm_buffer.x_sf,
            symm_buffer.topk_idx,
            symm_buffer.topk_weights,
            **stage_kwargs,
        )
        symm_buffer.note_staged_tokens(staged)

        y = torch.empty(
            p["num_tokens"], p["hidden"], device="cuda", dtype=torch.bfloat16
        )
        pkg.sm107_block_scaled_mega_moe(
            y, transformed[0], transformed[1], symm_buffer, num_tokens=p["num_tokens"]
        )

        w13_q, w13_sf, w2_q, w2_sf = _quantize_reference_weights(pkg, p, quant_kind)
        y_ref = pkg.compute_megamoe_reference_sm107_block_scaled(
            symm_buffer.x,
            symm_buffer.x_sf,
            symm_buffer.topk_idx,
            symm_buffer.topk_weights,
            w13_q,
            w13_sf,
            w2_q,
            w2_sf,
            quant_kind=quant_kind,
            local_expert_offset=0,
            gate_up_clamp=None,
            apply_topk_at_fc1=True,
            num_tokens=p["num_tokens"],
        )[: p["num_tokens"]]

        band = _REL_L2_BAND[quant_kind]
        rel_l2 = (y.to(torch.float32) - y_ref.to(torch.float32)).norm() / y_ref.to(
            torch.float32
        ).norm().clamp_min(1e-6)
        print(f"[sm107 oracle mixed-cga] kind={quant_kind} rel_l2={rel_l2.item():.5f}")
        assert rel_l2.item() < band, f"rel_l2 {rel_l2.item()} out of band {band}"
    finally:
        symm_buffer.destroy()


@pytest.mark.arch_rubin
@pytest.mark.parametrize("quant_kind", QUANT_KINDS)
def test_sm107_block_scaled_preprocess_weight_shapes(quant_kind):
    """Kernel-layout invariants of the weight transforms (no compile)."""
    pkg = _sm107_tree()
    _require_cuda()

    staging_mod, weights_mod = _backend_modules(quant_kind)
    from flashinfer.moe_ep.weights import MoEWeightPack

    E, hidden, intermediate = 4, 256, 128
    w13 = torch.randn(E, 2 * intermediate, hidden, device="cuda", dtype=torch.bfloat16)
    w2 = torch.randn(E, hidden, intermediate, device="cuda", dtype=torch.bfloat16)
    transform_kwargs = {} if quant_kind == "nvfp4" else {"kind": quant_kind}
    (fc1_w, fc1_sf), (fc2_w, fc2_sf) = weights_mod.preprocess_mega_weights(
        MoEWeightPack(w13=w13, w2=w2),
        intermediate_size=intermediate,
        hidden_size=hidden,
        **transform_kwargs,
    )
    if quant_kind == "nvfp4":
        vec = pkg.Nvfp4BlockSize
        assert fc1_w.shape == (E, hidden // 2, 2 * intermediate)
        assert fc2_w.shape == (E, intermediate // 2, hidden)
        assert fc1_sf.dtype == torch.float8_e4m3fn
    else:
        vec = pkg.Mxfp8BlockSize
        assert fc1_w.shape == (E, hidden, 2 * intermediate)
        assert fc2_w.shape == (E, intermediate, hidden)
        assert fc1_sf.dtype == torch.float8_e8m0fnu
    assert fc1_w.stride(1) == 1  # K (hidden, packed for nvfp4) innermost
    assert fc2_w.stride(1) == 1  # K (intermediate, packed for nvfp4) innermost
    assert fc1_sf.shape == (
        E,
        pkg.swizzled_flat_sf_size(2 * intermediate, hidden // vec),
    )
    assert fc2_sf.shape == (
        E,
        pkg.swizzled_flat_sf_size(hidden, intermediate // vec),
    )
    validate_kwargs = {} if quant_kind == "nvfp4" else {"kind": quant_kind}
    weights_mod.validate_transformed_mega_weights(
        ((fc1_w, fc1_sf), (fc2_w, fc2_sf)),
        intermediate_size=intermediate,
        hidden_size=hidden,
        world_size=1,
        num_experts=E,
        **validate_kwargs,
    )
