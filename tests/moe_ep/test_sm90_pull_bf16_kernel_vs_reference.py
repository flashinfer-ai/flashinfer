"""Single-GPU checks: ``sm90_bf16_bf16_bf16_pull_cutedsl`` vs a pure-torch BF16 oracle.

The SM90 pull-style megakernel compiled with BF16 operands, driven through
the public ``MoEEpMegaLayer`` at EP1 (``MEGA_NO_DIST=1``) and compared with
``_sm90_pull_bf16_reference.sm90_bf16_moe_reference`` -- an independent FP32
oracle that rounds where the kernel does.  Each case replays routing rounds
(full / masked / short / hot / empty / full) on ONE layer, so a round that
leaks state into the next fails.

Process isolation: imports the SM90 kernel tree (mutually exclusive with the
SM100 tree per process); excluded from run_tests.sh ``unit`` and run by the
``oracle_sm90`` target.  Directly, on one Hopper GPU::

    MEGA_NO_DIST=1 CUDA_VISIBLE_DEVICES=0 pytest \\
        tests/moe_ep/test_sm90_pull_bf16_kernel_vs_reference.py -v -m arch_hopper
"""

from __future__ import annotations

import pytest

from ._sm90_pull_bf16_reference import assert_bf16_close, sm90_bf16_moe_reference

GATE_UP_CLAMP = 10.0


def _require_sm90_tree():
    import torch

    from flashinfer.utils import is_sm90a_supported

    if not torch.cuda.is_available() or not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("Requires SM90a")
    try:
        import flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel  # noqa: F401
    except RuntimeError as exc:
        pytest.skip(f"SM90 kernel tree unavailable in this process: {exc}")


def _weights(num_experts, hidden, intermediate, *, seed=13):
    import torch

    g = torch.Generator(device="cuda").manual_seed(seed)
    w13 = torch.randn(
        num_experts, 2 * intermediate, hidden, device="cuda", generator=g
    ) * (hidden**-0.5)
    w2 = torch.randn(num_experts, hidden, intermediate, device="cuda", generator=g) * (
        intermediate**-0.5
    )
    return w13.to(torch.bfloat16), w2.to(torch.bfloat16)


def _layer(*, hidden, intermediate, num_experts, topk, max_tokens, w13, w2, **cfg):
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpMegaLayer,
        MoEWeightPack,
        Sm90_Bf16_Bf16_Bf16_PullCutedsl_MegaMoeConfig,
    )

    return MoEEpMegaLayer(
        bootstrap=BootstrapConfig(world_size=1, rank=0, auto_bootstrap=False),
        fleet_params=FleetParams(
            num_experts=num_experts,
            max_tokens_per_rank=max_tokens,
            token_hidden_size=hidden,
        ),
        weights=MoEWeightPack(w13=w13, w2=w2),
        backend=MegaConfig(
            megakernel=Sm90_Bf16_Bf16_Bf16_PullCutedsl_MegaMoeConfig(
                intermediate_size=intermediate,
                top_k=topk,
                gate_up_clamp=GATE_UP_CLAMP,
                **cfg,
            ),
        ),
    )


def _rounds(num_tokens, num_experts, topk, hidden, *, seed=7):
    """(name, MoEEpTensors) routing rounds; masked slots and a fully masked token."""
    import torch

    from flashinfer.moe_ep import MoEEpTensors

    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(
        num_tokens, hidden, dtype=torch.bfloat16, device="cuda", generator=g
    )
    scores = torch.randn(num_tokens, num_experts, device="cuda", generator=g)
    topk_w, ids = torch.topk(scores, topk, dim=-1, sorted=False)
    topk_w = torch.softmax(topk_w, dim=-1)
    masked = ids.clone()
    masked[::3, 0] = -1
    masked[1::5, -1] = -1
    masked[5] = -1
    hot = torch.arange(topk, device="cuda").expand_as(ids).contiguous()

    def t(n, i):
        return MoEEpTensors(
            hidden_states=x[:n], topk_ids=i[:n], topk_weights=topk_w[:n]
        )

    return [
        ("full", t(num_tokens, ids)),
        ("masked", t(num_tokens, masked)),
        ("short", t(7, ids)),
        ("hot", t(num_tokens, hot)),
        ("empty", t(0, ids)),
        ("full_again", t(num_tokens, ids)),
    ]


def _run_rounds(layer, w13, w2, rounds, *, check=assert_bf16_close):
    import torch

    for name, t in rounds:
        y = layer.forward(t)
        torch.cuda.synchronize()
        n = t.hidden_states.shape[0]
        assert y.shape == (n, w2.shape[1]) and y.dtype == torch.bfloat16, name
        if n == 0:
            continue
        ref = sm90_bf16_moe_reference(
            t.hidden_states,
            t.topk_ids,
            t.topk_weights,
            w13,
            w2,
            gate_up_clamp=GATE_UP_CLAMP,
        )
        check(y, ref, label=name)
        if name == "masked":
            assert torch.all(y[5] == 0), "fully masked token must reduce to zero"


# Geometry / combine variants (H=1024, I=512, E=8, top-k 4, 64 tokens).
_CASES = {
    "heuristic": {},
    "native_m64n128": dict(swap_ab=False, mma_tiler_mnk=(64, 128, 128)),
    "native_coop_m64n256": dict(swap_ab=False, mma_tiler_mnk=(64, 256, 128)),
    "native_pingpong": dict(swap_ab=False, pingpong=True, mma_tiler_mnk=(64, 128, 128)),
    "native_cga2x1": dict(
        swap_ab=False, mma_tiler_mnk=(64, 128, 128), cluster_shape_mnk=(2, 1, 1)
    ),
    "swap_m256n32": dict(swap_ab=True, mma_tiler_mnk=(256, 32, 128)),
    "swap_pingpong_m128n64": dict(
        swap_ab=True, pingpong=True, mma_tiler_mnk=(128, 64, 128)
    ),
    "swap_m128n8_cga1x2": dict(
        swap_ab=True, mma_tiler_mnk=(128, 8, 128), cluster_shape_mnk=(1, 2, 1)
    ),
    # K=64 tiles (BF16 only): twice the A/B pipeline stages.
    "swap_pingpong_m128n128_k64_tail_split": dict(
        swap_ab=True,
        pingpong=True,
        mma_tiler_mnk=(128, 128, 64),
        cluster_shape_mnk=(1, 2, 1),
        tail_split_pairs=True,
    ),
    "swap_m256n32_k64": dict(swap_ab=True, mma_tiler_mnk=(256, 32, 64)),
    "native_coop_m64n256_k64": dict(swap_ab=False, mma_tiler_mnk=(64, 256, 64)),
    "reuse_dispatch_warps": dict(
        swap_ab=False,
        mma_tiler_mnk=(64, 128, 128),
        token_back_mode="reuse_dispatch_warps",
    ),
    "standalone_warps": dict(
        swap_ab=False,
        mma_tiler_mnk=(64, 128, 128),
        token_back_mode="standalone_warps",
    ),
    "dedup_dispatch": dict(swap_ab=True, dedup_dispatch=True),
    "grouped_token_back": dict(
        swap_ab=True,
        token_back_mode="reuse_dispatch_warps",
        grouped_token_back=True,
    ),
}


@pytest.mark.arch_hopper
@pytest.mark.parametrize("case", sorted(_CASES))
def test_sm90_pull_bf16_layer_matches_oracle(monkeypatch, case):
    _require_sm90_tree()
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    hidden, inter, experts, topk, tokens = 1024, 512, 8, 4, 64
    w13, w2 = _weights(experts, hidden, inter)
    layer = _layer(
        hidden=hidden,
        intermediate=inter,
        num_experts=experts,
        topk=topk,
        max_tokens=tokens,
        w13=w13,
        w2=w2,
        **_CASES[case],
    )
    try:
        _run_rounds(layer, w13, w2, _rounds(tokens, experts, topk, hidden))
    finally:
        layer.destroy()


@pytest.mark.arch_hopper
def test_sm90_pull_bf16_in_kernel_fc2_reduce(monkeypatch):
    """In-kernel FC2 reduce accumulates BF16 terms with bf16x2 atomics: one
    extra BF16 rounding per added term, so the band widens accordingly."""
    import torch

    _require_sm90_tree()
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    hidden, inter, experts, topk, tokens = 1024, 512, 8, 4, 64
    w13, w2 = _weights(experts, hidden, inter)
    layer = _layer(
        hidden=hidden,
        intermediate=inter,
        num_experts=experts,
        topk=topk,
        max_tokens=tokens,
        w13=w13,
        w2=w2,
        swap_ab=False,
        mma_tiler_mnk=(64, 128, 128),
        enable_in_kernel_fc2_reduce=True,
    )

    def check(y, ref, *, label):
        out = y.float()
        assert torch.isfinite(out).all(), label
        rel_l2 = ((out - ref).norm() / ref.norm()).item()
        torch.testing.assert_close(out, ref, atol=4e-2, rtol=4e-2, msg=label)
        assert rel_l2 < 1e-2, (label, rel_l2)

    try:
        _run_rounds(layer, w13, w2, _rounds(tokens, experts, topk, hidden), check=check)
    finally:
        layer.destroy()


@pytest.mark.arch_hopper
@pytest.mark.parametrize("tokens", [8, 128, 512])
def test_sm90_pull_bf16_production_width(monkeypatch, tokens):
    """DeepSeek-V3-like widths (H=7168, I=2048, top-8) on the BF16 heuristic
    rows: the doubled BF16 A/B stage leaves 2-6 SMEM stages here."""
    _require_sm90_tree()
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    hidden, inter, experts, topk = 7168, 2048, 8, 8
    w13, w2 = _weights(experts, hidden, inter)
    layer = _layer(
        hidden=hidden,
        intermediate=inter,
        num_experts=experts,
        topk=topk,
        max_tokens=tokens,
        w13=w13,
        w2=w2,
    )
    try:
        rounds = _rounds(tokens, experts, topk, hidden)
        _run_rounds(layer, w13, w2, [rounds[0], rounds[1]])
    finally:
        layer.destroy()


@pytest.mark.arch_hopper
def test_sm90_pull_bf16_rejects_tile_without_two_smem_stages(monkeypatch):
    """Swap-AB M256xN128 is an FP8 tile; with BF16 operands only one A/B stage
    fits at H=7168, which must fail at compile time with a clear message."""
    _require_sm90_tree()
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    hidden, inter, experts, topk = 7168, 2048, 2, 2
    w13, w2 = _weights(experts, hidden, inter)
    layer = _layer(
        hidden=hidden,
        intermediate=inter,
        num_experts=experts,
        topk=topk,
        max_tokens=64,
        w13=w13,
        w2=w2,
        swap_ab=True,
        mma_tiler_mnk=(256, 128, 128),
    )
    try:
        with pytest.raises(ValueError, match="SMEM stage"):
            layer.warmup()
    finally:
        layer.destroy()


@pytest.mark.arch_hopper
def test_sm90_pull_bf16_preprocess_layout():
    """Kernel legs: interleave-8 gate/up, K-major BF16 views, unit scale slots."""
    import torch

    _require_sm90_tree()
    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.mega.kernel.sm90.bf16_bf16_bf16_pull_cutedsl.weights import (
        preprocess_mega_weights,
        validate_transformed_mega_weights,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_fp8_bf16_pull_cutedsl.weights import (
        _interleave_gate_up_8,
    )
    from flashinfer.moe_ep.core.validation.common import MoEEpConfigError

    from ._mxfp8_reference import mxfp8_quantize_ref

    experts, hidden, inter = 4, 512, 256
    w13, w2 = _weights(experts, hidden, inter)
    for source in (w13, w13.float()):
        l1, l2 = preprocess_mega_weights(
            MoEWeightPack(w13=source, w2=w2),
            intermediate_size=inter,
            hidden_size=hidden,
        )
        assert l1[0].dtype == torch.bfloat16 and l1[0].shape == (
            experts,
            hidden,
            2 * inter,
        )
        assert l1[0].stride(1) == 1 and l2[0].stride(1) == 1
        assert torch.equal(
            l1[0].transpose(1, 2),
            _interleave_gate_up_8(w13, intermediate_size=2 * inter),
        )
        assert torch.equal(l2[0].transpose(1, 2), w2)
        assert all(torch.all(leg[i] == 1) for leg in (l1, l2) for i in (2, 3))
        validate_transformed_mega_weights(
            (l1, l2),
            intermediate_size=inter,
            hidden_size=hidden,
            world_size=1,
            num_experts=experts,
        )

    scaled = (l1[0], l1[1], l1[2], l1[3] * 2)
    with pytest.raises(MoEEpConfigError, match="ones"):
        validate_transformed_mega_weights(
            (scaled, l2),
            intermediate_size=inter,
            hidden_size=hidden,
            world_size=1,
            num_experts=experts,
        )
    row_major = (l1[0].contiguous(), *l1[1:])
    with pytest.raises(MoEEpConfigError, match="K-major"):
        validate_transformed_mega_weights(
            (row_major, l2),
            intermediate_size=inter,
            hidden_size=hidden,
            world_size=1,
            num_experts=experts,
        )

    (q13, s13), (q2, s2) = mxfp8_quantize_ref(w13), mxfp8_quantize_ref(w2)
    with pytest.raises(MoEEpConfigError, match="PrequantizedMoEWeights"):
        preprocess_mega_weights(
            MoEWeightPack(w13=q13, w2=q2, w13_scale=s13, w2_scale=s2),
            intermediate_size=inter,
            hidden_size=hidden,
        )
