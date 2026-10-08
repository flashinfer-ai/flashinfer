"""Single-GPU checks: SM90 ``hopper_bf16_mega_moe`` vs the drop's torch reference.

BF16 twin of ``test_sm90_pull_fp8_kernel_vs_reference.py``: validates that a
single-rank ``hopper_bf16_mega_moe`` launch matches the kernel drop's own
pure-torch ground truth ``compute_megamoe_reference_bf16`` (imported through
the package/shim boundary, never ``src/`` directly) on the SAME staged bf16
payload, for native, swap-AB, and swap-AB ping-pong geometries.  Tolerances
follow the drop's ``mega_runner`` validation (atol/rtol 1e-2 with inputs
conditioned to O(1) outputs).

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
        tests/moe_ep/test_sm90_pull_bf16_kernel_vs_reference.py -v -m arch_hopper
"""

from __future__ import annotations

import pytest


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
    the hundreds and turn bf16 rounding into false failures).  ``hidden`` must
    be a multiple of 256 and ``intermediate`` of 64 (BF16 shim contract).
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
    """Host-side ``MegaMoEHopperBf16Config`` invariants (no compile needed)."""
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
    cfg = pkg.MegaMoEHopperBf16Config(**base)
    assert cfg.fc1_out == 1024
    assert cfg.num_experts_per_rank == 4
    assert cfg.mma_tiler_mnk == (64, 128, 64)

    with pytest.raises(ValueError, match="cluster_shape_mnk"):
        # CGA k must stay 1; (m, n) beyond 2x2 is rejected.
        pkg.MegaMoEHopperBf16Config(**{**base, "cluster_shape_mnk": (1, 1, 2)})
    with pytest.raises(ValueError, match="cluster_shape_mnk"):
        pkg.MegaMoEHopperBf16Config(**{**base, "cluster_shape_mnk": (4, 1, 1)})
    cga = pkg.MegaMoEHopperBf16Config(**{**base, "cluster_shape_mnk": (2, 2, 1)})
    assert cga.cluster_shape_mnk == (2, 2, 1)
    with pytest.raises(ValueError, match="native"):
        # M=128 is a swap-AB geometry; native requires M=64.
        pkg.MegaMoEHopperBf16Config(**{**base, "mma_tiler_mnk": (128, 128, 64)})
    with pytest.raises(ValueError, match="swap-AB"):
        pkg.MegaMoEHopperBf16Config(
            **{**base, "swap_ab": True, "mma_tiler_mnk": (64, 128, 64)}
        )
    # Tile K must be a positive multiple of the BF16 swizzle atom (64).
    with pytest.raises(ValueError, match="mma_tiler K"):
        pkg.MegaMoEHopperBf16Config(**{**base, "mma_tiler_mnk": (64, 128, 100)})
    k128 = pkg.MegaMoEHopperBf16Config(**{**base, "mma_tiler_mnk": (64, 128, 128)})
    assert k128.mma_tiler_mnk[2] == 128
    # Shape contract: hidden % 256 == 0, intermediate % 64 == 0.
    with pytest.raises(ValueError, match="hidden"):
        pkg.MegaMoEHopperBf16Config(**{**base, "hidden": 1088})
    with pytest.raises(ValueError, match="intermediate"):
        pkg.MegaMoEHopperBf16Config(**{**base, "intermediate": 544})
    # All six token-back x reduce mode combinations are supported.
    six = pkg.MegaMoEHopperBf16Config(
        **{**base, "in_kernel_fc2_reduce": True, "token_back_by_dispatch": True}
    )
    assert six.resolved_token_back_mode == "reuse_dispatch_warps"
    with pytest.raises(ValueError, match="token_back_mode"):
        pkg.MegaMoEHopperBf16Config(**{**base, "token_back_mode": "bogus"})
    # Ping-pong geometry constraints.
    with pytest.raises(ValueError, match="ping-pong"):
        pkg.MegaMoEHopperBf16Config(
            **{**base, "pingpong": True, "mma_tiler_mnk": (64, 256, 64)}
        )
    pp = pkg.MegaMoEHopperBf16Config(
        **{**base, "pingpong": True, "mma_tiler_mnk": (64, 128, 64)}
    )
    assert pp.pingpong
    # Tail-split pair tasks need a 2-CTA token cluster.
    with pytest.raises(ValueError, match="tail_split_pairs"):
        pkg.MegaMoEHopperBf16Config(
            **{**base, "tail_split_pairs": True, "cluster_shape_mnk": (1, 2, 1)}
        )
    # swap-AB default geometry is valid.
    swab = dataclasses.replace(cfg, swap_ab=True, mma_tiler_mnk=(256, 32, 64))
    assert swab.swap_ab


def _reference_reduced(pkg, *, problem, symm_buffer, l1, l2):
    """Drop ground truth on the staged bf16 payload, top-k reduced."""
    import torch

    n = problem["num_tokens"]
    combine_ref = pkg.compute_megamoe_reference_bf16(
        input_activation=symm_buffer.x[:n].unsqueeze(0),
        input_topk_idx=symm_buffer.topk_idx[:n].unsqueeze(0),
        input_topk_weights=symm_buffer.topk_weights[:n].unsqueeze(0),
        fc1_weight=l1.unsqueeze(0),  # (1, E, H, 2I) K-major
        fc2_weight=l2.unsqueeze(0),  # (1, E, I, H) K-major
        ref_compute_graph="deepgemm",  # matches the shim's apply_topk_in_fc1=True
        fc2_output_dtype=torch.bfloat16,
        gate_up_clamp=problem["gate_up_clamp"],
    )
    # deepgemm graph folds topk weights before the bf16 fc1-out store, so the
    # per-topk terms reduce with a plain sum.
    return combine_ref[0].to(torch.float32).sum(dim=1)


@pytest.mark.arch_hopper
@pytest.mark.parametrize(
    "swap_ab,pingpong",
    [(False, None), (True, None), (True, True)],
    ids=["native", "swap_ab", "swap_ab_pingpong"],
)
def test_sm90_bf16_kernel_matches_drop_reference(monkeypatch, swap_ab, pingpong):
    """Single-rank ``hopper_bf16_mega_moe`` matches ``compute_megamoe_reference_bf16``.

    ``pingpong=True`` with swap-AB resolves to the manual-mode default tile
    (128, 32, 64).
    """
    _require_cuda()

    import torch

    pkg = _sm90_tree()

    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.mega.kernel.sm90.bf16_bf16_bf16_pull_cutedsl.staging import (
        stage_mega_moe_inputs,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm90.bf16_bf16_bf16_pull_cutedsl.weights import (
        preprocess_mega_weights,
    )

    # monkeypatch (not os.environ): restored after the test, so it cannot
    # silently downgrade later nvshmem-path tests in the same process.
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    problem = _single_rank_problem()
    rank, world_size = 0, 1
    n = problem["num_tokens"]

    pack = MoEWeightPack(w13=problem["w13"], w2=problem["w2"])
    l1, l2 = preprocess_mega_weights(
        pack,
        intermediate_size=problem["intermediate"],
        hidden_size=problem["hidden"],
    )

    symm_buffer = pkg.get_symm_buffer_for_hopper_bf16_mega_moe(
        problem["num_experts"],
        problem["max_tokens"],
        problem["topk"],
        problem["hidden"],
        problem["intermediate"],
        rank,
        world_size,
        swap_ab=swap_ab,
        pingpong=pingpong,
        gate_up_clamp=problem["gate_up_clamp"],
    )
    try:
        if pingpong:
            assert symm_buffer._frontend.config.mma_tiler_mnk == (128, 32, 64)
        stage_mega_moe_inputs(
            problem["hidden_states"],
            problem["topk_weights"],
            problem["topk_ids"],
            symm_buffer.x,
            symm_buffer.topk_idx,
            symm_buffer.topk_weights,
        )

        y_ref = _reference_reduced(
            pkg, problem=problem, symm_buffer=symm_buffer, l1=l1, l2=l2
        )

        y_kernel = torch.empty(
            n, problem["hidden"], dtype=torch.bfloat16, device="cuda"
        )
        pkg.hopper_bf16_mega_moe(
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
            f"[sm90 bf16 oracle swap_ab={swap_ab} pingpong={pingpong}] "
            f"rel_l2={rel_l2.item():.4g} "
            f"max|d|={(yk - yr).abs().max().item():.4g} "
            f"amax(ref)={yr.abs().max().item():.4g}"
        )
        # Drop-tester tolerances (mega_runner.validate: atol=rtol=1e-2), valid
        # here because the problem is conditioned to O(1) outputs and kernel +
        # reference share the same staged bf16 operands; the rel_l2 gate
        # catches a real numerical break independent of per-element noise.
        torch.testing.assert_close(yk, yr, atol=1e-2, rtol=1e-2)
        assert rel_l2.item() < 0.02
    finally:
        symm_buffer.destroy()
