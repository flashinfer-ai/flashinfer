"""Single-GPU checks: ``sm90_bf16_nvfp4_bf16_pull_cutedsl`` (W4A16) vs a pure-torch oracle.

Packed NVFP4 weights decoded inside the SM90 pull-style megakernel
(register-sourced BF16 WGMMA, swap-AB), driven through ``MoEEpMegaLayer`` at
EP1 (``MEGA_NO_DIST=1``).  The decode is exact (E2M1 x E4M3 fits BF16), so
the oracle is the BF16 oracle (``_sm90_pull_bf16_reference``) on the dequantized
weights times the per-expert global scales.  Routing rounds (full / masked /
short / hot / empty / full) replay on one layer.

Process isolation as for the other SM90 pull tests: excluded from
run_tests.sh ``unit``, run by ``oracle_sm90``.  Directly::

    MEGA_NO_DIST=1 CUDA_VISIBLE_DEVICES=0 pytest \\
        tests/moe_ep/test_sm90_pull_w4a16_kernel_vs_reference.py -v -m arch_hopper
"""

from __future__ import annotations

import pytest

from ._sm90_pull_bf16_reference import assert_bf16_close, sm90_bf16_moe_reference
from .test_sm90_pull_bf16_kernel_vs_reference import (
    GATE_UP_CLAMP,
    _require_sm90_tree,
    _rounds,
)


def _nvfp4_pack(num_experts, hidden, intermediate, *, seed=11, special_scales=False):
    """Random NVFP4 checkpoint pack + per-expert global scales (O(1) outputs).

    ``special_scales``: also zero ~10% and negate ~25% of the block scales
    (the host canonicalizes both before the kernel sees them).
    """
    import torch

    from flashinfer.moe_ep import MoEWeightPack

    g = torch.Generator(device="cuda").manual_seed(seed)
    legs = []
    for n, k in ((2 * intermediate, hidden), (hidden, intermediate)):
        codes = torch.randint(
            0, 16, (num_experts, n, k), device="cuda", generator=g, dtype=torch.uint8
        )
        packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous()
        scales = (
            (torch.rand(num_experts, n, k // 16, device="cuda", generator=g) + 0.5)
            * (k**-0.5)
            / 3
        ).to(torch.float8_e4m3fn)
        if special_scales:
            pick = torch.rand(scales.shape, device="cuda", generator=g)
            value = scales.float()
            value = torch.where(
                pick < 0.1, 0.0, torch.where(pick < 0.35, -value, value)
            )
            scales = value.to(torch.float8_e4m3fn)
        legs.append((packed, scales))
    alphas = [
        torch.rand(num_experts, device="cuda", generator=g) + 0.5 for _ in range(2)
    ]
    pack = MoEWeightPack(
        w13=legs[0][0], w2=legs[1][0], w13_scale=legs[0][1], w2_scale=legs[1][1]
    )
    return pack, alphas


def _dequant(pack, alphas):
    from flashinfer.moe_ep.backends.mega.kernel.sm90.common.nvfp4 import (
        dequantize_nvfp4,
    )

    return (
        dequantize_nvfp4(pack.w13, pack.w13_scale, alphas[0]),
        dequantize_nvfp4(pack.w2, pack.w2_scale, alphas[1]),
    )


def _layer(*, hidden, intermediate, num_experts, topk, max_tokens, pack, **cfg):
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpMegaLayer,
        Sm90_Bf16_Nvfp4_Bf16_PullCutedsl_MegaMoeConfig,
    )

    return MoEEpMegaLayer(
        bootstrap=BootstrapConfig(world_size=1, rank=0, auto_bootstrap=False),
        fleet_params=FleetParams(
            num_experts=num_experts,
            max_tokens_per_rank=max_tokens,
            token_hidden_size=hidden,
        ),
        weights=pack,
        backend=MegaConfig(
            megakernel=Sm90_Bf16_Nvfp4_Bf16_PullCutedsl_MegaMoeConfig(
                intermediate_size=intermediate,
                top_k=topk,
                gate_up_clamp=GATE_UP_CLAMP,
                **cfg,
            ),
        ),
    )


def _check_rounds(layer, w13, w2, rounds, *, check=assert_bf16_close):
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


_CASES = {
    "heuristic": {},
    "swap_m128n8": dict(mma_tiler_mnk=(128, 8, 128)),
    "swap_m256n16": dict(mma_tiler_mnk=(256, 16, 128)),
    "swap_m256n64": dict(mma_tiler_mnk=(256, 64, 128)),
    "pingpong_m128n128": dict(pingpong=True, mma_tiler_mnk=(128, 128, 128)),
    "cga2x1": dict(mma_tiler_mnk=(256, 32, 128), cluster_shape_mnk=(2, 1, 1)),
    "pingpong_cga1x2_tail_split": dict(
        pingpong=True,
        mma_tiler_mnk=(128, 64, 128),
        cluster_shape_mnk=(1, 2, 1),
        tail_split_pairs=True,
    ),
    "reuse_dispatch_warps": dict(
        mma_tiler_mnk=(256, 32, 128), token_back_mode="reuse_dispatch_warps"
    ),
    "standalone_warps": dict(
        mma_tiler_mnk=(256, 32, 128), token_back_mode="standalone_warps"
    ),
    "dedup_dispatch": dict(mma_tiler_mnk=(256, 32, 128), dedup_dispatch=True),
    "grouped_token_back": dict(
        mma_tiler_mnk=(256, 32, 128),
        token_back_mode="reuse_dispatch_warps",
        grouped_token_back=True,
    ),
}


@pytest.mark.arch_hopper
@pytest.mark.parametrize("case", sorted(_CASES))
def test_sm90_pull_w4a16_layer_matches_oracle(monkeypatch, case):
    _require_sm90_tree()
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    hidden, inter, experts, topk, tokens = 1024, 512, 8, 4, 64
    pack, alphas = _nvfp4_pack(experts, hidden, inter)
    w13, w2 = _dequant(pack, alphas)
    layer = _layer(
        hidden=hidden,
        intermediate=inter,
        num_experts=experts,
        topk=topk,
        max_tokens=tokens,
        pack=pack,
        fc1_alpha=alphas[0],
        fc2_alpha=alphas[1],
        **_CASES[case],
    )
    try:
        _check_rounds(layer, w13, w2, _rounds(tokens, experts, topk, hidden))
    finally:
        layer.destroy()


@pytest.mark.arch_hopper
def test_sm90_pull_w4a16_runtime_alpha_overrides_config(monkeypatch):
    """Runtime MoEEpTensors alphas apply to that forward only."""
    import dataclasses

    _require_sm90_tree()
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    hidden, inter, experts, topk, tokens = 1024, 512, 8, 4, 64
    pack, alphas = _nvfp4_pack(experts, hidden, inter)
    layer = _layer(
        hidden=hidden,
        intermediate=inter,
        num_experts=experts,
        topk=topk,
        max_tokens=tokens,
        pack=pack,
        fc1_alpha=alphas[0],
        fc2_alpha=alphas[1],
    )
    try:
        _, t = _rounds(tokens, experts, topk, hidden)[0]
        runtime = [a.flip(0) * 1.5 for a in alphas]
        w13, w2 = _dequant(pack, runtime)
        _check_rounds(
            layer,
            w13,
            w2,
            [
                (
                    "runtime",
                    dataclasses.replace(t, fc1_alpha=runtime[0], fc2_alpha=runtime[1]),
                )
            ],
        )
        # fc2 only at runtime: fc1 falls back to the config alpha.
        w13_cfg, _ = _dequant(pack, alphas)
        _check_rounds(
            layer,
            w13_cfg,
            w2,
            [("fc2_runtime", dataclasses.replace(t, fc2_alpha=runtime[1]))],
        )
        w13, w2 = _dequant(pack, alphas)
        _check_rounds(layer, w13, w2, [("config_again", t)])
    finally:
        layer.destroy()


@pytest.mark.arch_hopper
def test_sm90_pull_w4a16_in_kernel_fc2_reduce(monkeypatch):
    import torch

    _require_sm90_tree()
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    hidden, inter, experts, topk, tokens = 1024, 512, 8, 4, 64
    pack, alphas = _nvfp4_pack(experts, hidden, inter)
    w13, w2 = _dequant(pack, alphas)
    layer = _layer(
        hidden=hidden,
        intermediate=inter,
        num_experts=experts,
        topk=topk,
        max_tokens=tokens,
        pack=pack,
        fc1_alpha=alphas[0],
        fc2_alpha=alphas[1],
        mma_tiler_mnk=(256, 32, 128),
        enable_in_kernel_fc2_reduce=True,
    )

    def check(y, ref, *, label):
        out = y.float()
        assert torch.isfinite(out).all(), label
        torch.testing.assert_close(out, ref, atol=4e-2, rtol=4e-2, msg=label)
        assert ((out - ref).norm() / ref.norm()).item() < 1e-2, label

    try:
        _check_rounds(
            layer, w13, w2, _rounds(tokens, experts, topk, hidden), check=check
        )
    finally:
        layer.destroy()


@pytest.mark.arch_hopper
@pytest.mark.parametrize("tokens", [8, 128, 512, 2048])
def test_sm90_pull_w4a16_production_width(monkeypatch, tokens):
    """DeepSeek-V3-like widths (H=7168, I=2048, top-8) on the heuristic rows."""
    _require_sm90_tree()
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    hidden, inter, experts, topk = 7168, 2048, 8, 8
    pack, alphas = _nvfp4_pack(experts, hidden, inter)
    w13, w2 = _dequant(pack, alphas)
    layer = _layer(
        hidden=hidden,
        intermediate=inter,
        num_experts=experts,
        topk=topk,
        max_tokens=tokens,
        pack=pack,
        fc1_alpha=alphas[0],
        fc2_alpha=alphas[1],
    )
    try:
        rounds = _rounds(tokens, experts, topk, hidden)
        _check_rounds(layer, w13, w2, [rounds[0], rounds[1]])
    finally:
        layer.destroy()


@pytest.mark.arch_hopper
def test_sm90_pull_w4a16_zero_and_negative_scales(monkeypatch):
    """Zero and negative block scales stay exact through host canonicalization."""
    _require_sm90_tree()
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    hidden, inter, experts, topk, tokens = 1024, 512, 8, 4, 64
    pack, alphas = _nvfp4_pack(experts, hidden, inter, special_scales=True)
    w13, w2 = _dequant(pack, alphas)
    layer = _layer(
        hidden=hidden,
        intermediate=inter,
        num_experts=experts,
        topk=topk,
        max_tokens=tokens,
        pack=pack,
        fc1_alpha=alphas[0],
        fc2_alpha=alphas[1],
        mma_tiler_mnk=(256, 32, 128),
    )
    try:
        _check_rounds(layer, w13, w2, _rounds(tokens, experts, topk, hidden)[:2])
    finally:
        layer.destroy()


@pytest.mark.arch_hopper
def test_sm90_pull_w4a16_quantizes_bf16_checkpoints(monkeypatch):
    """A BF16 pack is quantized at preprocess (global scale folded into alpha)."""
    import torch

    _require_sm90_tree()
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.mega.kernel.sm90.common.nvfp4 import (
        dequantize_nvfp4,
        quantize_nvfp4,
    )

    from .test_sm90_pull_bf16_kernel_vs_reference import _weights

    hidden, inter, experts, topk, tokens = 1024, 512, 8, 4, 64
    w13_bf16, w2_bf16 = _weights(experts, hidden, inter)
    w13 = dequantize_nvfp4(*quantize_nvfp4(w13_bf16))
    w2 = dequantize_nvfp4(*quantize_nvfp4(w2_bf16))
    # NVFP4 of 1/sqrt(K)-scaled randn stays close to the source weights.
    assert ((w13 - w13_bf16.float()).norm() / w13_bf16.float().norm()) < 0.15
    layer = _layer(
        hidden=hidden,
        intermediate=inter,
        num_experts=experts,
        topk=topk,
        max_tokens=tokens,
        pack=MoEWeightPack(w13=w13_bf16, w2=w2_bf16),
        mma_tiler_mnk=(256, 32, 128),
    )
    try:
        _check_rounds(layer, w13, w2, _rounds(tokens, experts, topk, hidden)[:2])
    finally:
        layer.destroy()
    del torch


@pytest.mark.arch_hopper
def test_sm90_pull_w4a16_preprocess_layout():
    """Augmented legs decode back to the interleaved NVFP4 weights; bad packs raise."""
    import torch

    _require_sm90_tree()
    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.mega.kernel.sm90.bf16_nvfp4_bf16_pull_cutedsl.weights import (
        preprocess_mega_weights,
        validate_transformed_mega_weights,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm90.common.nvfp4 import (
        dequantize_nvfp4,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_fp8_bf16_pull_cutedsl.weights import (
        _interleave_gate_up_8,
    )
    from flashinfer.moe_ep.core.validation.common import MoEEpConfigError

    experts, hidden, inter = 4, 256, 128
    pack, alphas = _nvfp4_pack(experts, hidden, inter, special_scales=True)
    l1, l2 = preprocess_mega_weights(
        pack,
        intermediate_size=inter,
        hidden_size=hidden,
        fc1_alpha=alphas[0],
        fc2_alpha=alphas[1],
    )
    validate_transformed_mega_weights(
        (l1, l2),
        intermediate_size=inter,
        hidden_size=hidden,
        world_size=1,
        num_experts=experts,
    )
    assert l1[0].dtype == torch.uint8 and l1[0].shape == (
        experts,
        inter,
        hidden // 128 * 144,
    )
    assert torch.equal(l1[3], alphas[0]) and torch.equal(l2[3], alphas[1])

    def unaugment(aug, rows, k):
        # Pair tile: [row 2p payload | row 2p+1 payload | 2p scales | 2p+1 scales],
        # payload byte (kb, h, lane) stored at lane * 16 + kb * 2 + h, with
        # each 32-bit word's byte b codes (even, odd k) in nibbles b, b + 4.
        tiles = aug.view(experts, rows // 2, k // 128, 144)
        words = tiles[..., :128].reshape(-1, 4)
        nibbles = torch.stack((words & 0xF, words >> 4), dim=-1).reshape(-1, 2, 4)
        payload = nibbles[:, 0] | (nibbles[:, 1] << 4)  # byte b = nibbles b, b + 4
        payload = payload.reshape(experts, rows // 2, k // 128, 2, 4, 8, 2)
        payload = payload.permute(
            0, 1, 3, 2, 5, 6, 4
        )  # (e, pair, row, kt, kb, h, lane)
        packed = payload.reshape(experts, rows, k // 2)
        scales = tiles[..., 128:].reshape(experts, rows // 2, k // 128, 2, 8)
        scales = scales.permute(0, 1, 3, 2, 4).reshape(experts, rows, k // 16)
        return dequantize_nvfp4(packed, scales)

    want13 = _interleave_gate_up_8(
        dequantize_nvfp4(pack.w13, pack.w13_scale), intermediate_size=2 * inter
    )
    assert torch.equal(unaugment(l1[0], 2 * inter, hidden), want13)
    assert torch.equal(
        unaugment(l2[0], hidden, inter), dequantize_nvfp4(pack.w2, pack.w2_scale)
    )

    bad = MoEWeightPack(
        w13=pack.w13[:, :, :-1].contiguous(),
        w2=pack.w2,
        w13_scale=pack.w13_scale,
        w2_scale=pack.w2_scale,
    )
    with pytest.raises(MoEEpConfigError, match="packed w13"):
        preprocess_mega_weights(bad, intermediate_size=inter, hidden_size=hidden)
    with pytest.raises(MoEEpConfigError, match="fc1_alpha"):
        preprocess_mega_weights(
            pack,
            intermediate_size=inter,
            hidden_size=hidden,
            fc1_alpha=alphas[0].double(),
        )
    nan_scale = pack.w2_scale.clone()
    nan_scale.view(torch.uint8)[0, 0, 0] = 0x7F
    with pytest.raises(MoEEpConfigError, match="NaN"):
        preprocess_mega_weights(
            MoEWeightPack(
                w13=pack.w13, w2=pack.w2, w13_scale=pack.w13_scale, w2_scale=nan_scale
            ),
            intermediate_size=inter,
            hidden_size=hidden,
        )
