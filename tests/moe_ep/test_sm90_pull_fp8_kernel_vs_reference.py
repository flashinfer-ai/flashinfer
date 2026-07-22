"""Single-GPU checks: SM90 ``hopper_fp8_mega_moe`` vs the drop's torch reference.

Hopper counterpart of ``test_nvfp4_cutedsl_kernel_vs_reference.py``: validates
that a single-rank ``hopper_fp8_mega_moe`` launch matches the kernel drop's own
pure-torch ground truth ``compute_megamoe_reference_fp8`` (imported through the
package/shim boundary, never ``src/`` directly) on the SAME staged fp8
payloads, in both ``per_tensor`` and ``blockwise`` scale modes and both native
and swap-AB geometries.  Tolerances follow the drop's ``mega_runner``
validation (atol/rtol 1e-2 with inputs conditioned to O(1) outputs).

Process isolation: the SM90 and SM100 kernel trees share top-level module
names (``common``, ``src``, ``moe_nvfp4_swapab``) and are mutually exclusive
per process.  This file therefore imports the SM90 tree only inside test
bodies (guarded), is EXCLUDED from run_tests.sh's ``unit`` target, and runs in
its own pytest process via the ``oracle_sm90`` target::

    bash tests/moe_ep/run_tests.sh oracle_sm90

or directly on one Hopper GPU from the FlashInfer repo root::

    cd /path/to/flashinfer
    export PYTHONPATH="${PWD}:${PYTHONPATH}"
    MEGA_NO_DIST=1 CUDA_VISIBLE_DEVICES=0 pytest \\
        tests/moe_ep/test_sm90_pull_fp8_kernel_vs_reference.py -v -m arch_hopper
"""

from __future__ import annotations

import pytest

E4M3_MAX = 448.0
FP8_BLOCK_K = 128  # Fp8BlockScaleK (re-asserted via the package boundary below)


def _sm90_tree():
    """Import the SM90 kernel package, skipping if the SM100 tree owns us."""
    try:
        import flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel as pkg
    except RuntimeError as exc:
        # shim/_paths sibling-tree exclusivity guard: another test already
        # loaded the SM100 kernel modules into this process.
        pytest.skip(f"SM90 kernel tree unavailable in this process: {exc}")
    return pkg


def _require_cuda():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")


def _single_rank_problem(hidden=1024, intermediate=512):
    """O(1)-output problem: randn activations, 1/sqrt(K)-scaled weights.

    The drop's validation tolerances (atol=1e-2) are absolute, so weights are
    normalized to keep |y| ~ O(1) (unnormalized randn weights would put |y| in
    the hundreds and turn bf16 rounding into false failures).
    """
    import torch

    num_tokens = 32
    max_tokens = 64
    num_experts = 4
    topk = 4
    gate_up_clamp = 10.0

    g = torch.Generator(device="cuda").manual_seed(7)
    hidden_states = torch.randn(
        num_tokens, hidden, dtype=torch.bfloat16, device="cuda", generator=g
    )
    scores = torch.randn(
        num_tokens, num_experts, dtype=torch.float32, device="cuda", generator=g
    )
    topk_weights, topk_ids = torch.topk(
        scores, topk, dim=-1, largest=True, sorted=False
    )
    topk_weights = torch.softmax(topk_weights, dim=-1)

    g = torch.Generator(device="cuda").manual_seed(13)
    w13 = torch.randn(
        num_experts,
        2 * intermediate,
        hidden,
        dtype=torch.bfloat16,
        device="cuda",
        generator=g,
    ) * (hidden**-0.5)
    w2 = torch.randn(
        num_experts,
        hidden,
        intermediate,
        dtype=torch.bfloat16,
        device="cuda",
        generator=g,
    ) * (intermediate**-0.5)

    return dict(
        hidden=hidden,
        intermediate=intermediate,
        num_tokens=num_tokens,
        max_tokens=max_tokens,
        num_experts=num_experts,
        topk=topk,
        gate_up_clamp=gate_up_clamp,
        hidden_states=hidden_states,
        topk_weights=topk_weights.to(torch.float32),
        topk_ids=topk_ids.to(torch.int64),
        w13=w13,
        w2=w2,
    )


@pytest.mark.arch_hopper
def test_shim_config_validation():
    """Host-side ``MegaMoEHopperFp8Config`` invariants (no compile needed)."""
    import dataclasses

    pkg = _sm90_tree()
    base = dict(
        rank=0,
        world_size=1,
        num_tokens_per_rank=64,
        num_topk=4,
        num_total_experts=4,
        hidden=1024,
        intermediate=512,
    )
    cfg = pkg.MegaMoEHopperFp8Config(**base)
    assert cfg.fc1_out == 1024
    assert cfg.num_experts_per_rank == 4

    with pytest.raises(ValueError, match="kind"):
        pkg.MegaMoEHopperFp8Config(**{**base, "kind": "mxfp8_e4m3"})
    with pytest.raises(ValueError, match="1-CTA-only"):
        pkg.MegaMoEHopperFp8Config(**{**base, "cluster_shape_mnk": (2, 1, 1)})
    with pytest.raises(ValueError, match="native"):
        # M=128 is a swap-AB geometry; native requires M=64.
        pkg.MegaMoEHopperFp8Config(**{**base, "mma_tiler_mnk": (128, 128, 128)})
    with pytest.raises(ValueError, match="swap-AB"):
        pkg.MegaMoEHopperFp8Config(
            **{**base, "swap_ab": True, "mma_tiler_mnk": (64, 128, 128)}
        )
    with pytest.raises(ValueError, match="cannot both"):
        pkg.MegaMoEHopperFp8Config(
            **{**base, "in_kernel_fc2_reduce": True, "token_back_by_dispatch": True}
        )
    with pytest.raises(ValueError, match="blockwise"):
        pkg.MegaMoEHopperFp8Config(
            **{**base, "fp8_scale_mode": "blockwise", "hidden": 1088}
        )
    # swap-AB default geometry is valid.
    swab = dataclasses.replace(cfg, swap_ab=True, mma_tiler_mnk=(256, 32, 128))
    assert swab.swap_ab


def _reference_reduced(pkg, *, problem, symm_buffer, l1, l2, fp8_scale_mode, s1):
    """Drop ground truth on the staged fp8 payloads, top-k reduced.

    Returns ``(y_ref, fc2_activation_dequant_scale_or_None)``; per_tensor mode
    derives the fc2 activation requant scale exactly the way the drop's tester
    does (from the reference swiglu absmax) so the kernel launch can share it.
    """
    import torch

    n = problem["num_tokens"]
    hidden = problem["hidden"]

    common_kwargs = dict(
        input_activation=symm_buffer.x[:n].unsqueeze(0),
        input_topk_idx=symm_buffer.topk_idx[:n].unsqueeze(0),
        input_topk_weights=symm_buffer.topk_weights[:n].unsqueeze(0),
        fc1_weight=l1[0].unsqueeze(0),
        fc2_weight=l2[0].unsqueeze(0),
        ab_dtype=torch.float8_e4m3fn,
        ref_compute_graph="deepgemm",  # matches the shim's apply_topk_in_fc1=True
        fp8_accum_mode="1xacc",
        mma_tiler_k=128,
        fc2_output_dtype=torch.bfloat16,
        gate_up_clamp=problem["gate_up_clamp"],
        fp8_scale_mode=fp8_scale_mode,
        return_fc2_activation_dequant_scale=True,
    )
    if fp8_scale_mode == "blockwise":
        sf_cols = hidden // FP8_BLOCK_K
        combine_ref, _fc2_scale = pkg.compute_megamoe_reference_fp8(
            input_activation_sf=symm_buffer.x_sf[:n, :sf_cols].unsqueeze(0),
            fc1_weight_sf=l1[1].unsqueeze(0),
            fc2_weight_sf=l2[1].unsqueeze(0),
            fc1_activation_block_scale=symm_buffer.x_sf[:n, :sf_cols].unsqueeze(0),
            fc1_weight_block_scale=l1[1].unsqueeze(0),
            fc2_weight_block_scale=l2[1].unsqueeze(0),
            fc2_activation_block_scale=None,  # reference derives per-token, like the kernel
            **common_kwargs,
        )
        fc2_scale_out = None
    else:
        combine_ref, fc2_scale_out = pkg.compute_megamoe_reference_fp8(
            input_activation_sf=symm_buffer.x_sf[:n].unsqueeze(0),  # unused (ABI)
            fc1_weight_sf=l1[1].unsqueeze(0),  # unused (ABI)
            fc2_weight_sf=l2[1].unsqueeze(0),  # unused (ABI)
            fc1_activation_dequant_scale=torch.tensor(
                (s1,), dtype=torch.float32, device="cuda"
            ),
            fc1_weight_dequant_scale=l1[3].unsqueeze(0),
            fc2_activation_dequant_scale=None,  # derive from swiglu absmax (tester recipe)
            fc2_weight_dequant_scale=l2[3].unsqueeze(0),
            **common_kwargs,
        )
    # deepgemm graph folds topk weights before fc1-out quantization, so the
    # per-topk terms reduce with a plain sum.
    y_ref = combine_ref[0].to(torch.float32).sum(dim=1)
    return y_ref, fc2_scale_out


@pytest.mark.arch_hopper
@pytest.mark.parametrize(
    "fp8_scale_mode,swap_ab",
    [
        ("per_tensor", False),
        ("per_tensor", True),
        ("blockwise", False),
        ("blockwise", True),
    ],
)
def test_sm90_fp8_kernel_matches_drop_reference(monkeypatch, fp8_scale_mode, swap_ab):
    """Single-rank ``hopper_fp8_mega_moe`` matches ``compute_megamoe_reference_fp8``."""
    _require_cuda()

    import torch

    pkg = _sm90_tree()
    # Constant mirrored above; fail loudly if a drop moves it.
    assert pkg.Fp8BlockScaleK == FP8_BLOCK_K

    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.mega.kernel.sm90_pull_fp8.staging import (
        stage_mega_moe_inputs,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm90_pull_fp8.weights import (
        preprocess_mega_weights,
    )

    # monkeypatch (not os.environ): restored after the test, so it cannot
    # silently downgrade later nvshmem-path tests in the same process.
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    problem = _single_rank_problem()
    rank, world_size = 0, 1
    n = problem["num_tokens"]

    # per_tensor static activation calibration (identical on all EP ranks by
    # contract; here derived from the known test inputs with the reference's
    # 0.95 headroom margin).
    if fp8_scale_mode == "per_tensor":
        s1 = float(
            problem["hidden_states"].to(torch.float32).abs().amax().item()
        ) / (0.95 * E4M3_MAX)
    else:
        s1 = 1.0

    pack = MoEWeightPack(w13=problem["w13"], w2=problem["w2"])
    l1, l2 = preprocess_mega_weights(
        pack,
        intermediate_size=problem["intermediate"],
        hidden_size=problem["hidden"],
        kind="fp8_e4m3",
        fp8_scale_mode=fp8_scale_mode,
        fc1_activation_dequant_scale=s1,
        # placeholder; per_tensor replaces it below with the reference-derived
        # requant scale (the drop tester's compute_reference -> run_kernel flow).
        fc2_activation_dequant_scale=1.0,
    )

    symm_buffer = pkg.get_symm_buffer_for_hopper_fp8_mega_moe(
        problem["num_experts"],
        problem["max_tokens"],
        problem["topk"],
        problem["hidden"],
        problem["intermediate"],
        rank,
        world_size,
        kind="fp8_e4m3",
        fp8_scale_mode=fp8_scale_mode,
        swap_ab=swap_ab,
        gate_up_clamp=problem["gate_up_clamp"],
    )
    try:
        stage_mega_moe_inputs(
            problem["hidden_states"],
            problem["topk_weights"],
            problem["topk_ids"],
            symm_buffer.x,
            symm_buffer.x_sf,
            symm_buffer.topk_idx,
            symm_buffer.topk_weights,
            kind="fp8_e4m3",
            fp8_scale_mode=fp8_scale_mode,
            fc1_activation_dequant_scale=s1,
        )

        y_ref, fc2_scale = _reference_reduced(
            pkg,
            problem=problem,
            symm_buffer=symm_buffer,
            l1=l1,
            l2=l2,
            fp8_scale_mode=fp8_scale_mode,
            s1=s1,
        )
        if fp8_scale_mode == "per_tensor":
            assert fc2_scale is not None
            l2 = (l2[0], l2[1], fc2_scale.to("cuda"), l2[3])

        y_kernel = torch.empty(
            n, problem["hidden"], dtype=torch.bfloat16, device="cuda"
        )
        pkg.hopper_fp8_mega_moe(
            y_kernel,
            l1,
            l2,
            symm_buffer,
            num_tokens=n,
            gate_up_clamp=problem["gate_up_clamp"],
            sync=True,
        )

        assert torch.isfinite(y_kernel).all()
        yk = y_kernel.to(torch.float32)
        yr = y_ref
        rel_l2 = (yk - yr).norm() / yr.norm().clamp_min(1e-6)
        print(
            f"[sm90 fp8 oracle {fp8_scale_mode} swap_ab={swap_ab}] "
            f"rel_l2={rel_l2.item():.4g} "
            f"max|d|={(yk - yr).abs().max().item():.4g} "
            f"amax(ref)={yr.abs().max().item():.4g}"
        )
        # Drop-tester tolerances (mega_runner.validate: atol=rtol=1e-2), valid
        # here because the problem is conditioned to O(1) outputs and kernel +
        # reference share the same staged fp8 operands.  Blockwise carries the
        # drop's caveat that reduced-only drift can be bf16 cancellation from
        # WGMMA/torch ordering differences — the rel_l2 gate below catches a
        # real numerical break either way.
        torch.testing.assert_close(yk, yr, atol=1e-2, rtol=1e-2)
        assert rel_l2.item() < 0.02
    finally:
        symm_buffer.destroy()
