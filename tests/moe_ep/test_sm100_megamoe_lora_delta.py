"""FC1 LoRA delta (Mn-bias) and post-SwiGLU activation snapshot tests for SM100 MegaMoE."""

from __future__ import annotations

import os

import pytest
import torch

from flashinfer import moe_ep
from flashinfer.moe_ep.backends.mega.kernel.sm100.standard_cutedsl import (
    _validate_gemm1_lora_delta,
)
from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel
from flashinfer.moe_ep.core.validation.common import MoEEpConfigError
from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.sources import (
    Sm100BlockScaledMoeConfig,
    output_error,
)
from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.sources.kernel_helpers import (
    compute_megamoe_reference_block_scaled,
)


def _require_sm100():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("requires SM100/SM103")


def test_lora_config_validation():
    cfg = moe_ep.Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=256,
        top_k=2,
        enable_fc1_lora_delta=True,
    )
    assert cfg.enable_fc1_lora_delta is True

    default_cfg = moe_ep.Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=256, top_k=2
    )
    assert default_cfg.enable_fc1_lora_delta is False

    with pytest.raises(ValueError, match="enable_fc1_lora_delta must be a bool"):
        moe_ep.Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig(
            intermediate_size=256,
            top_k=2,
            enable_fc1_lora_delta=1,  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="kernel_variant='inference'"):
        moe_ep.Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(
            intermediate_size=256,
            top_k=2,
            kernel_variant="genphase",
            enable_fc1_lora_delta=True,
        )

    with pytest.raises(ValueError, match="batch_invariant"):
        moe_ep.Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig(
            intermediate_size=256,
            top_k=2,
            batch_invariant=True,
            enable_fc1_lora_delta=True,
        )

    with pytest.raises(ValueError, match="finalize_mode=NONE"):
        moe_ep.Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig(
            intermediate_size=256,
            top_k=2,
            finalize_mode=moe_ep.FinalizeMode.NONE,
            enable_fc1_lora_delta=True,
        )

    with pytest.raises(ValueError, match="standard provider export"):
        moe_ep.Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
            intermediate_size=256,
            top_k=2,
            kernel_name="sm100_nvfp4_nvfp4_bf16_cutedsl_legacy",
            enable_fc1_lora_delta=True,
        )

    with pytest.raises(ValueError, match="batch_invariant"):
        Sm100BlockScaledMoeConfig(
            num_total_experts=4,
            max_tokens_per_rank=64,
            num_topk=2,
            hidden=256,
            intermediate=256,
            rank=0,
            world_size=1,
            quant_kind="mxfp4_mxfp8",
            batch_invariant=True,
            enable_fc1_lora_delta=True,
        )


def test_lora_workspace_config_propagation(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    cfg = moe_ep.Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=256,
        top_k=2,
        enable_fc1_lora_delta=True,
        knobs={"mma_tiler_mnk": (256, 64, 128), "schedule_mode": "grouped"},
    )
    backend = create_mega_kernel(cfg)
    backend.bind_ep_bootstrap(
        moe_ep.BootstrapConfig(world_size=1, rank=0, auto_bootstrap=False)
    )
    fleet = moe_ep.FleetParams(
        num_experts=4, max_tokens_per_rank=64, token_hidden_size=256
    )
    native = backend._workspace_config(fleet)
    assert native.enable_fc1_lora_delta is True
    assert native.quant_kind == "mxfp4_mxfp8"


def test_lora_delta_input_validation():
    cpu_dev = torch.device("cpu")
    _validate_gemm1_lora_delta(
        None,
        num_tokens=4,
        top_k=2,
        intermediate_size=128,
        enabled=False,
        device=cpu_dev,
    )
    dummy = torch.zeros(4, 2, 256, dtype=torch.bfloat16)
    with pytest.raises(MoEEpConfigError, match="enable_fc1_lora_delta=False"):
        _validate_gemm1_lora_delta(
            dummy,
            num_tokens=4,
            top_k=2,
            intermediate_size=128,
            enabled=False,
            device=cpu_dev,
        )
    with pytest.raises(
        MoEEpConfigError, match="required when enable_fc1_lora_delta=True"
    ):
        _validate_gemm1_lora_delta(
            None,
            num_tokens=4,
            top_k=2,
            intermediate_size=128,
            enabled=True,
            device=cpu_dev,
        )
    with pytest.raises(MoEEpConfigError, match="CUDA tensor"):
        _validate_gemm1_lora_delta(
            dummy,
            num_tokens=4,
            top_k=2,
            intermediate_size=128,
            enabled=True,
            device=cpu_dev,
        )


def test_positional_constructor_compatibility():
    x = torch.zeros(4, 16, dtype=torch.bfloat16)
    topk_ids = torch.zeros(4, 2, dtype=torch.int32)
    topk_weights = torch.zeros(4, 2, dtype=torch.float32)
    recv_count = torch.tensor([4], dtype=torch.int32)
    num_tokens_per_expert = torch.zeros(2, dtype=torch.int32)
    out = torch.zeros(4, 16, dtype=torch.bfloat16)
    # 9th positional argument must remain recv_count, not gemm1_lora_delta.
    bundle = moe_ep.MoEEpTensors(
        x,
        topk_ids,
        topk_weights,
        None,
        None,
        None,
        None,
        None,
        recv_count,
        num_tokens_per_expert,
        out,
    )
    assert bundle.recv_count is recv_count
    assert bundle.num_tokens_per_expert is num_tokens_per_expert
    assert bundle.output is out
    assert bundle.gemm1_lora_delta is None

    # Positional fields on Sm100BlockScaledMoeConfig after enable_fc1_activation_per_token_scale
    # must remain use_custom_finalize / use_persistent_finalize_kernel, not enable_fc1_lora_delta.
    import dataclasses

    pos_names = [
        f.name for f in dataclasses.fields(Sm100BlockScaledMoeConfig) if not f.kw_only
    ]
    assert "enable_fc1_lora_delta" not in pos_names
    idx = pos_names.index("enable_fc1_activation_per_token_scale")
    assert pos_names[idx + 1 : idx + 3] == [
        "use_custom_finalize",
        "use_persistent_finalize_kernel",
    ]


def _make_lora_weights_and_delta(
    x_bf16: torch.Tensor,
    topk_ids: torch.Tensor,
    adapter_ids: torch.Tensor,
    num_experts: int,
    intermediate: int,
    hidden: int,
    num_adapters: int,
    rank: int,
    rng: torch.Generator,
):
    """Build multi-adapter gate/up and down LoRA banks + precomputed delta1 [T, K, 2*H_int]."""
    device = x_bf16.device
    scale = 0.5
    # Gate/up LoRA weights: [num_adapters, E, rank, H] and [num_adapters, E, 2*H_int, rank]
    a1 = (
        torch.randn(
            num_adapters, num_experts, rank, hidden, device=device, generator=rng
        )
        * 0.15
    ).bfloat16()
    b1 = (
        torch.randn(
            num_adapters,
            num_experts,
            2 * intermediate,
            rank,
            device=device,
            generator=rng,
        )
        * 0.15
    ).bfloat16()
    # Down LoRA weights: [num_adapters, E, rank, H_int] and [num_adapters, E, H, rank]
    a2 = (
        torch.randn(
            num_adapters, num_experts, rank, intermediate, device=device, generator=rng
        )
        * 0.15
    ).bfloat16()
    b2 = (
        torch.randn(
            num_adapters, num_experts, hidden, rank, device=device, generator=rng
        )
        * 0.15
    ).bfloat16()

    num_tokens, top_k = topk_ids.shape
    delta1 = torch.zeros(
        num_tokens, top_k, 2 * intermediate, dtype=torch.bfloat16, device=device
    )
    for t in range(num_tokens):
        a_id = int(adapter_ids[t].item())
        if a_id < 0:
            continue
        xt = x_bf16[t].float()
        for k in range(top_k):
            e = int(topk_ids[t, k].item())
            if e < 0 or e >= num_experts:
                continue
            mid = a1[a_id, e].float() @ xt
            delta1[t, k] = (scale * (b1[a_id, e].float() @ mid)).to(torch.bfloat16)
    return delta1, a2, b2, scale


def _apply_down_lora_from_snapshot(
    act: torch.Tensor,
    token_id: torch.Tensor,
    expert_offsets: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    adapter_ids: torch.Tensor,
    a2: torch.Tensor,
    b2: torch.Tensor,
    scale: float,
    num_tokens: int,
    hidden: int,
) -> torch.Tensor:
    """Caller-side down-LoRA loop matching #5468's proposed kernel graph."""
    device = act.device
    delta2 = torch.zeros(num_tokens, hidden, dtype=torch.float32, device=device)
    offsets = expert_offsets.tolist()
    tok_ids = token_id.tolist()
    num_local_experts = len(offsets) - 1
    for e in range(num_local_experts):
        for r in range(offsets[e], offsets[e + 1]):
            t = tok_ids[r]
            if t == -1:
                continue
            a_id = int(adapter_ids[t].item())
            if a_id < 0:
                continue
            slots = (topk_ids[t] == e).nonzero(as_tuple=False)
            assert slots.numel() == 1
            w = float(topk_weights[t, int(slots[0].item())].item())
            mid = a2[a_id, e].float() @ act[r].float()
            delta2[t] += w * scale * (b2[a_id, e].float() @ mid)
    return delta2.to(torch.bfloat16)


def _check_snapshot_and_output(
    layer,
    res,
    delta1: torch.Tensor,
    num_tokens: int,
    *,
    num_experts: int,
    intermediate: int,
    out_tol: float = 0.03,
    act_tol: float = 0.03,
):
    assert isinstance(res, moe_ep.MegaMoeLoraOutput)
    y, act, token_id, expert_offsets, src_rank = res
    ws = layer._ensure_workspace()
    cfg = ws.config
    r_cap = int(ws.kernel.max_tokens)

    assert y.shape == (num_tokens, cfg.hidden)
    assert y.dtype == torch.bfloat16
    assert act.shape == (r_cap, intermediate)
    assert act.dtype == torch.bfloat16
    assert token_id.shape == (r_cap,)
    assert token_id.dtype == torch.int32
    assert src_rank.shape == (r_cap,)
    assert src_rank.dtype == torch.int32
    assert expert_offsets.shape == (num_experts + 1,)
    assert expert_offsets.dtype == torch.int32
    assert torch.isfinite(y).all()

    ref_y, ref_post_act = compute_megamoe_reference_block_scaled(
        ws.x,
        ws.x_sf,
        ws.topk_idx,
        ws.topk_weights,
        layer._transformed[0][0].permute(0, 2, 1),
        layer._transformed[0][1],
        layer._transformed[1][0].permute(0, 2, 1),
        layer._transformed[1][1],
        quant_kind=cfg.quant_kind,
        combine_dtype=cfg.combine_dtype,
        local_expert_offset=0,
        gate_up_clamp=cfg.gate_up_clamp,
        apply_routing_weights_before_fc2=cfg.apply_routing_weights_before_fc2,
        num_tokens=num_tokens,
        weight_scales_are_swizzled=True,
        fc1_alpha=getattr(ws, "fc1_alpha", None),
        fc2_alpha=getattr(ws, "fc2_alpha", None),
        fc1_norm_const=getattr(ws, "fc1_norm_const", None),
        gemm1_lora_delta=delta1,
        return_post_act=True,
    )
    ref_y = ref_y[:num_tokens]
    if num_tokens > 0:
        indices = torch.arange(num_tokens, device="cuda")
        err = output_error(y, indices, ref_y)
        assert err < out_tol, f"output relative L2 error {err:.4f} >= {out_tol}"

    offsets = expert_offsets.tolist()
    assert offsets[0] == 0
    assert all(offsets[i] <= offsets[i + 1] for i in range(num_experts))
    assert offsets[-1] <= r_cap

    # Trailing unused rows must be marked -1.
    if offsets[-1] < r_cap:
        assert (token_id[offsets[-1] :] == -1).all()
        assert (src_rank[offsets[-1] :] == -1).all()

    for e in range(num_experts):
        slab_tokens = token_id[offsets[e] : offsets[e + 1]]
        slab_ranks = src_rank[offsets[e] : offsets[e + 1]]
        slab_act = act[offsets[e] : offsets[e + 1]]
        valid_mask = slab_tokens >= 0
        pad_mask = ~valid_mask
        assert (slab_tokens[pad_mask] == -1).all()
        assert (slab_ranks[pad_mask] == -1).all()
        assert (slab_ranks[valid_mask] == cfg.rank).all()
        valid_tok_list = slab_tokens[valid_mask].tolist()
        ref_toks, ref_acts = ref_post_act[e]
        ref_tok_list = ref_toks.tolist()
        # Same multiset of routed tokens (router sorts within each expert in arbitrary thread order).
        assert sorted(valid_tok_list) == sorted(ref_tok_list)
        if not valid_tok_list:
            continue
        assert torch.isfinite(slab_act[valid_mask]).all()
        ref_by_tok = {tok: ref_acts[i] for i, tok in enumerate(ref_tok_list)}
        actual_valid = slab_act[valid_mask].float()
        expected_valid = torch.stack(
            [ref_by_tok[tok] for tok in valid_tok_list]
        ).float()
        denom = expected_valid.norm().clamp_min(1e-12)
        act_err = float((actual_valid - expected_valid).norm() / denom)
        assert act_err < act_tol, (
            f"expert {e} post-act rel L2 error {act_err:.4f} >= {act_tol}"
        )
    return ref_y, ref_post_act


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("apply_before_fc2", [True, False])
@pytest.mark.parametrize("gate_up_clamp", [None, 3.0])
@pytest.mark.parametrize("mma_n", [64, 128])
def test_mxfp4_mxfp8_lora_delta_and_snapshot(
    apply_before_fc2, gate_up_clamp, mma_n, monkeypatch
):
    _require_sm100()
    monkeypatch.setenv("MEGA_NO_DIST", "1")

    capacity, num_tokens, hidden, intermediate, top_k, num_experts = (
        160,
        145,
        256,
        256,
        2,
        2,
    )
    rng = torch.Generator(device="cuda").manual_seed(5468)
    w13 = (
        torch.randn(num_experts, 2 * intermediate, hidden, device="cuda", generator=rng)
        / hidden**0.5
    ).bfloat16()
    w2 = (
        torch.randn(num_experts, hidden, intermediate, device="cuda", generator=rng)
        / intermediate**0.5
    ).bfloat16()
    weights = moe_ep.MoEWeightPack(w13, w2)
    bootstrap = moe_ep.BootstrapConfig(world_size=1, rank=0)
    fleet = moe_ep.FleetParams(
        num_experts=num_experts,
        max_tokens_per_rank=capacity,
        token_hidden_size=hidden,
    )
    knobs = {"mma_tiler_mnk": (256, mma_n, 128), "schedule_mode": "grouped"}

    base_layer = moe_ep.MoEEpLayer(
        bootstrap,
        fleet,
        weights,
        backend=moe_ep.MegaConfig(
            megakernel=moe_ep.Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig(
                intermediate_size=intermediate,
                top_k=top_k,
                gate_up_clamp=gate_up_clamp,
                apply_routing_weights_before_fc2=apply_before_fc2,
                knobs=knobs,
            )
        ),
    )
    lora_layer = moe_ep.MoEEpLayer(
        bootstrap,
        fleet,
        weights,
        backend=moe_ep.MegaConfig(
            megakernel=moe_ep.Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig(
                intermediate_size=intermediate,
                top_k=top_k,
                gate_up_clamp=gate_up_clamp,
                apply_routing_weights_before_fc2=apply_before_fc2,
                enable_fc1_lora_delta=True,
                knobs=knobs,
            ),
            preprocess_weights=False,
            transformed_weights=base_layer._transformed,
        ),
    )
    try:
        x = torch.randn(num_tokens, hidden, device="cuda", generator=rng).bfloat16()
        # Unique expert per (token, slot)
        base_e = torch.arange(num_tokens, device="cuda", dtype=torch.int32)
        topk_ids = torch.stack(
            [base_e % num_experts, (base_e + 1) % num_experts], dim=-1
        )
        topk_weights = torch.softmax(
            torch.randn(num_tokens, top_k, device="cuda", generator=rng), dim=-1
        ).float()
        adapter_ids = torch.randint(
            -1, 3, (num_tokens,), device="cuda", dtype=torch.int32, generator=rng
        )

        # 1. Non-LoRA layer rejects delta and keeps exact single-tensor return ABI.
        with pytest.raises(MoEEpConfigError, match="enable_fc1_lora_delta=False"):
            base_layer.forward(
                moe_ep.MoEEpTensors(
                    hidden_states=x,
                    topk_ids=topk_ids,
                    topk_weights=topk_weights,
                    gemm1_lora_delta=torch.zeros(
                        num_tokens,
                        top_k,
                        2 * intermediate,
                        dtype=torch.bfloat16,
                        device="cuda",
                    ),
                )
            )
        t_base = moe_ep.MoEEpTensors(
            hidden_states=x, topk_ids=topk_ids, topk_weights=topk_weights
        )
        base_layer.warmup(t_base)
        y_base = base_layer.forward(t_base)
        assert isinstance(y_base, torch.Tensor)

        # 2. Zero delta matches base output and produces accurate post-SwiGLU snapshot.
        lora_layer.warmup()
        zero_delta = torch.zeros(
            num_tokens, top_k, 2 * intermediate, dtype=torch.bfloat16, device="cuda"
        )
        res_zero = lora_layer.forward(
            moe_ep.MoEEpTensors(
                hidden_states=x,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                gemm1_lora_delta=zero_delta,
            )
        )
        _check_snapshot_and_output(
            lora_layer,
            res_zero,
            zero_delta,
            num_tokens,
            num_experts=num_experts,
            intermediate=intermediate,
        )
        torch.testing.assert_close(res_zero.output, y_base, rtol=1e-2, atol=1e-2)

        # 3. Real multi-adapter LoRA delta (gate != up) + caller-side down-LoRA end-to-end check.
        delta1, a2, b2, scale = _make_lora_weights_and_delta(
            x,
            topk_ids,
            adapter_ids,
            num_experts,
            intermediate,
            hidden,
            num_adapters=3,
            rank=16,
            rng=rng,
        )
        assert delta1.abs().max().item() > 0
        res_lora = lora_layer.forward(
            moe_ep.MoEEpTensors(
                hidden_states=x,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                gemm1_lora_delta=delta1,
            )
        )
        # Delta must materially affect the output (fails if silently dropped).
        assert (
            res_lora.output.float() - res_zero.output.float()
        ).abs().max().item() > 0.05
        ref_y, ref_post_act = _check_snapshot_and_output(
            lora_layer,
            res_lora,
            delta1,
            num_tokens,
            num_experts=num_experts,
            intermediate=intermediate,
        )

        # End-to-end down-LoRA from kernel snapshot vs reference post-SwiGLU activations.
        delta2_kernel = _apply_down_lora_from_snapshot(
            res_lora.act,
            res_lora.token_id,
            res_lora.expert_offsets,
            topk_ids,
            topk_weights,
            adapter_ids,
            a2,
            b2,
            scale,
            num_tokens,
            hidden,
        )
        delta2_ref = torch.zeros(num_tokens, hidden, dtype=torch.float32, device="cuda")
        for e in range(num_experts):
            ref_toks, ref_acts = ref_post_act[e]
            for i, t in enumerate(ref_toks.tolist()):
                a_id = int(adapter_ids[t].item())
                if a_id < 0:
                    continue
                slots = (topk_ids[t] == e).nonzero(as_tuple=False)
                w = float(topk_weights[t, int(slots[0].item())].item())
                mid = a2[a_id, e].float() @ ref_acts[i].float()
                delta2_ref[t] += w * scale * (b2[a_id, e].float() @ mid)
        y_total_kernel = (res_lora.output.float() + delta2_kernel.float()).to(
            torch.bfloat16
        )
        y_total_ref = (ref_y.float() + delta2_ref).to(torch.bfloat16)
        total_err = output_error(
            y_total_kernel, torch.arange(num_tokens, device="cuda"), y_total_ref
        )
        assert total_err < 0.03, f"end-to-end LoRA rel L2 error {total_err:.4f} >= 0.03"
    finally:
        lora_layer.destroy()
        base_layer.destroy()


@pytest.mark.arch_blackwell
def test_lora_delta_masked_routes_nan_poison_and_zero_tokens(monkeypatch):
    _require_sm100()
    monkeypatch.setenv("MEGA_NO_DIST", "1")

    capacity, num_tokens, hidden, intermediate, top_k, num_experts = (
        64,
        19,
        256,
        256,
        3,
        4,
    )
    rng = torch.Generator(device="cuda").manual_seed(1337)
    w13 = (
        torch.randn(num_experts, 2 * intermediate, hidden, device="cuda", generator=rng)
        / hidden**0.5
    ).bfloat16()
    w2 = (
        torch.randn(num_experts, hidden, intermediate, device="cuda", generator=rng)
        / intermediate**0.5
    ).bfloat16()
    layer = moe_ep.MoEEpLayer(
        moe_ep.BootstrapConfig(world_size=1, rank=0),
        moe_ep.FleetParams(
            num_experts=num_experts,
            max_tokens_per_rank=capacity,
            token_hidden_size=hidden,
        ),
        moe_ep.MoEWeightPack(w13, w2),
        backend=moe_ep.MegaConfig(
            megakernel=moe_ep.Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig(
                intermediate_size=intermediate,
                top_k=top_k,
                enable_fc1_lora_delta=True,
                knobs={"mma_tiler_mnk": (256, 64, 128), "schedule_mode": "grouped"},
            )
        ),
    )
    try:
        x = torch.randn(num_tokens, hidden, device="cuda", generator=rng).bfloat16()
        topk_ids = torch.full((num_tokens, top_k), -1, dtype=torch.int32, device="cuda")
        # Leave expert 2 completely empty to verify zero-sized expert slabs.
        topk_ids[:, 1] = torch.where(
            torch.arange(num_tokens, device="cuda") % 2 == 0, 0, 1
        ).to(torch.int32)
        topk_ids[:, 2] = 3
        topk_weights = torch.full(
            (num_tokens, top_k), 0.5, dtype=torch.float32, device="cuda"
        )
        topk_weights[:, 0] = float("nan")
        delta1 = (
            torch.randn(
                num_tokens, top_k, 2 * intermediate, device="cuda", generator=rng
            )
            * 0.4
        ).bfloat16()
        # Poison masked slot 0 with NaN: kernel must never read it.
        delta1[:, 0, :] = float("nan")

        t = moe_ep.MoEEpTensors(
            hidden_states=x,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            gemm1_lora_delta=delta1,
        )
        layer.warmup(t)
        res = layer.forward(t)
        _check_snapshot_and_output(
            layer,
            res,
            delta1,
            num_tokens,
            num_experts=num_experts,
            intermediate=intermediate,
        )
        # Empty expert 2 has empty slab.
        assert int(res.expert_offsets[2].item()) == int(res.expert_offsets[3].item())

        # Zero-token invocation: all expert offsets 0, all token_ids -1.
        t_zero = moe_ep.MoEEpTensors(
            hidden_states=torch.empty(0, hidden, dtype=torch.bfloat16, device="cuda"),
            topk_ids=torch.empty(0, top_k, dtype=torch.int32, device="cuda"),
            topk_weights=torch.empty(0, top_k, dtype=torch.float32, device="cuda"),
            gemm1_lora_delta=torch.empty(
                0, top_k, 2 * intermediate, dtype=torch.bfloat16, device="cuda"
            ),
        )
        res_empty = layer.forward(t_zero)
        assert res_empty.output.shape == (0, hidden)
        assert (res_empty.expert_offsets == 0).all()
        assert (res_empty.token_id == -1).all()
        assert (res_empty.src_rank == -1).all()
    finally:
        layer.destroy()


@pytest.mark.arch_blackwell
def test_lora_delta_cuda_graph_replay(monkeypatch):
    _require_sm100()
    monkeypatch.setenv("MEGA_NO_DIST", "1")

    capacity, num_tokens, hidden, intermediate, top_k, num_experts = (
        64,
        24,
        256,
        256,
        2,
        4,
    )
    rng = torch.Generator(device="cuda").manual_seed(2026)
    w13 = (
        torch.randn(num_experts, 2 * intermediate, hidden, device="cuda", generator=rng)
        / hidden**0.5
    ).bfloat16()
    w2 = (
        torch.randn(num_experts, hidden, intermediate, device="cuda", generator=rng)
        / intermediate**0.5
    ).bfloat16()
    layer = moe_ep.MoEEpLayer(
        moe_ep.BootstrapConfig(world_size=1, rank=0),
        moe_ep.FleetParams(
            num_experts=num_experts,
            max_tokens_per_rank=capacity,
            token_hidden_size=hidden,
        ),
        moe_ep.MoEWeightPack(w13, w2),
        backend=moe_ep.MegaConfig(
            megakernel=moe_ep.Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig(
                intermediate_size=intermediate,
                top_k=top_k,
                enable_fc1_lora_delta=True,
                knobs={"mma_tiler_mnk": (256, 64, 128), "schedule_mode": "grouped"},
            )
        ),
    )
    graph = torch.cuda.CUDAGraph()
    try:
        x = torch.randn(num_tokens, hidden, device="cuda", generator=rng).bfloat16()
        base_e = torch.arange(num_tokens, device="cuda", dtype=torch.int32)
        topk_ids = torch.stack(
            [base_e % num_experts, (base_e + 1) % num_experts], dim=-1
        )
        topk_weights = torch.full(
            (num_tokens, top_k), 0.5, dtype=torch.float32, device="cuda"
        )
        delta1 = (
            torch.randn(
                num_tokens, top_k, 2 * intermediate, device="cuda", generator=rng
            )
            * 0.3
        ).bfloat16()
        t = moe_ep.MoEEpTensors(
            hidden_states=x,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            gemm1_lora_delta=delta1,
        )
        layer.warmup(t)
        with torch.cuda.graph(graph):
            captured = layer.forward(t, return_workspace_view=True)
        ptrs = tuple(tensor.data_ptr() for tensor in captured)

        # Replay 1: original routes and delta.
        graph.replay()
        torch.cuda.synchronize()
        _check_snapshot_and_output(
            layer,
            captured,
            delta1,
            num_tokens,
            num_experts=num_experts,
            intermediate=intermediate,
        )

        # Replay 2: change routes so fewer tokens land per expert (old valid rows become pad rows)
        # and change delta1 values.
        topk_ids.fill_(-1)
        topk_ids[:6, 0] = 0
        topk_ids[:6, 1] = 3
        delta1.copy_(
            (torch.randn_like(delta1, dtype=torch.float32) * 0.5).to(torch.bfloat16)
        )
        graph.replay()
        torch.cuda.synchronize()
        assert tuple(tensor.data_ptr() for tensor in captured) == ptrs
        _check_snapshot_and_output(
            layer,
            captured,
            delta1,
            num_tokens,
            num_experts=num_experts,
            intermediate=intermediate,
        )
    finally:
        graph.reset()
        layer.destroy()


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("mode", ["mxfp8", "nvfp4"])
def test_mxfp8_and_nvfp4_lora_delta_parity(mode, monkeypatch):
    _require_sm100()
    monkeypatch.setenv("MEGA_NO_DIST", "1")

    capacity, num_tokens, hidden, intermediate, top_k, num_experts = (
        64,
        20,
        256,
        256,
        2,
        4,
    )
    rng = torch.Generator(device="cuda").manual_seed(99 if mode == "mxfp8" else 101)
    w13 = (
        torch.randn(num_experts, 2 * intermediate, hidden, device="cuda", generator=rng)
        / hidden**0.5
    ).bfloat16()
    w2 = (
        torch.randn(num_experts, hidden, intermediate, device="cuda", generator=rng)
        / intermediate**0.5
    ).bfloat16()
    knobs = {"mma_tiler_mnk": (256, 64, 128), "schedule_mode": "grouped"}
    if mode == "mxfp8":
        megakernel = moe_ep.Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(
            intermediate_size=intermediate,
            top_k=top_k,
            enable_fc1_lora_delta=True,
            knobs=knobs,
        )
    else:
        megakernel = moe_ep.Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
            intermediate_size=intermediate,
            top_k=top_k,
            fc1_alpha=torch.tensor(
                [0.75, 1.25, 0.9, 1.1], device="cuda", dtype=torch.float32
            ),
            fc2_alpha=torch.tensor(
                [1.0, 0.8, 1.2, 0.95], device="cuda", dtype=torch.float32
            ),
            enable_fc1_lora_delta=True,
            knobs=knobs,
        )
    layer = moe_ep.MoEEpLayer(
        moe_ep.BootstrapConfig(world_size=1, rank=0),
        moe_ep.FleetParams(
            num_experts=num_experts,
            max_tokens_per_rank=capacity,
            token_hidden_size=hidden,
        ),
        moe_ep.MoEWeightPack(w13, w2),
        backend=moe_ep.MegaConfig(megakernel=megakernel),
    )
    try:
        x = torch.randn(num_tokens, hidden, device="cuda", generator=rng).bfloat16()
        base_e = torch.arange(num_tokens, device="cuda", dtype=torch.int32)
        topk_ids = torch.stack(
            [base_e % num_experts, (base_e + 1) % num_experts], dim=-1
        )
        topk_weights = torch.full(
            (num_tokens, top_k), 0.5, dtype=torch.float32, device="cuda"
        )
        delta1 = (
            torch.randn(
                num_tokens, top_k, 2 * intermediate, device="cuda", generator=rng
            )
            * 0.4
        ).bfloat16()
        t = moe_ep.MoEEpTensors(
            hidden_states=x,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            gemm1_lora_delta=delta1,
        )
        layer.warmup(t)
        res = layer.forward(t)
        _check_snapshot_and_output(
            layer,
            res,
            delta1,
            num_tokens,
            num_experts=num_experts,
            intermediate=intermediate,
            out_tol=0.06 if mode == "nvfp4" else 0.03,
            act_tol=0.06 if mode == "nvfp4" else 0.03,
        )
    finally:
        layer.destroy()


@pytest.mark.arch_blackwell
def test_nvfp4_lora_delta_with_per_token_scale_and_multi_subtile(monkeypatch):
    """Exercise token_0_32, token_32_64, subtile_idx > 0, tile_n_idx > 0, and per-token scales together."""
    _require_sm100()
    monkeypatch.setenv("MEGA_NO_DIST", "1")

    capacity, num_tokens, hidden, intermediate, top_k, num_experts = (
        160,
        140,
        256,
        256,
        2,
        2,
    )
    rng = torch.Generator(device="cuda").manual_seed(777)
    w13 = (
        torch.randn(num_experts, 2 * intermediate, hidden, device="cuda", generator=rng)
        / hidden**0.5
    ).bfloat16()
    w2 = (
        torch.randn(num_experts, hidden, intermediate, device="cuda", generator=rng)
        / intermediate**0.5
    ).bfloat16()
    layer = moe_ep.MoEEpLayer(
        moe_ep.BootstrapConfig(world_size=1, rank=0),
        moe_ep.FleetParams(
            num_experts=num_experts,
            max_tokens_per_rank=capacity,
            token_hidden_size=hidden,
        ),
        moe_ep.MoEWeightPack(w13, w2),
        backend=moe_ep.MegaConfig(
            megakernel=moe_ep.Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
                intermediate_size=intermediate,
                top_k=top_k,
                fc1_alpha=torch.tensor([0.8, 1.2], device="cuda", dtype=torch.float32),
                fc2_alpha=torch.tensor([1.1, 0.9], device="cuda", dtype=torch.float32),
                enable_fc1_activation_per_token_scale=True,
                enable_fc1_lora_delta=True,
                knobs={"mma_tiler_mnk": (256, 128, 128), "schedule_mode": "grouped"},
            )
        ),
    )
    try:
        x = torch.randn(num_tokens, hidden, device="cuda", generator=rng).bfloat16()
        base_e = torch.arange(num_tokens, device="cuda", dtype=torch.int32)
        topk_ids = torch.stack(
            [base_e % num_experts, (base_e + 1) % num_experts], dim=-1
        )
        topk_weights = torch.full(
            (num_tokens, top_k), 0.5, dtype=torch.float32, device="cuda"
        )
        pt_scale = (0.5 + torch.rand(num_tokens, device="cuda", generator=rng)).float()
        delta1 = (
            torch.randn(
                num_tokens, top_k, 2 * intermediate, device="cuda", generator=rng
            )
            * 0.4
        ).bfloat16()
        t = moe_ep.MoEEpTensors(
            hidden_states=x,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            fc1_activation_per_token_scale=pt_scale,
            gemm1_lora_delta=delta1,
        )
        layer.warmup(t)
        res = layer.forward(t)
        ws = layer._ensure_workspace()
        cfg = ws.config
        ref_y, ref_post_act = compute_megamoe_reference_block_scaled(
            ws.x,
            ws.x_sf,
            ws.topk_idx,
            ws.topk_weights,
            layer._transformed[0][0].permute(0, 2, 1),
            layer._transformed[0][1],
            layer._transformed[1][0].permute(0, 2, 1),
            layer._transformed[1][1],
            quant_kind=cfg.quant_kind,
            combine_dtype=cfg.combine_dtype,
            local_expert_offset=0,
            gate_up_clamp=cfg.gate_up_clamp,
            apply_routing_weights_before_fc2=cfg.apply_routing_weights_before_fc2,
            num_tokens=num_tokens,
            weight_scales_are_swizzled=True,
            fc1_alpha=ws.fc1_alpha,
            fc2_alpha=ws.fc2_alpha,
            fc1_norm_const=ws.fc1_norm_const,
            fc1_activation_per_token_scale=pt_scale,
            gemm1_lora_delta=delta1,
            return_post_act=True,
        )
        err = output_error(
            res.output, torch.arange(num_tokens, device="cuda"), ref_y[:num_tokens]
        )
        assert err < 0.06, f"output rel L2 error {err:.4f} >= 0.06"
        offsets = res.expert_offsets.tolist()
        for e in range(num_experts):
            slab_tokens = res.token_id[offsets[e] : offsets[e + 1]]
            slab_ranks = res.src_rank[offsets[e] : offsets[e + 1]]
            slab_act = res.act[offsets[e] : offsets[e + 1]]
            valid_mask = slab_tokens >= 0
            assert (slab_tokens[~valid_mask] == -1).all()
            assert (slab_ranks[~valid_mask] == -1).all()
            assert (slab_ranks[valid_mask] == 0).all()
            valid_toks = slab_tokens[valid_mask].tolist()
            ref_toks, ref_acts = ref_post_act[e]
            assert sorted(valid_toks) == sorted(ref_toks.tolist())
            ref_by_tok = {tok: ref_acts[i] for i, tok in enumerate(ref_toks.tolist())}
            actual = slab_act[valid_mask].float()
            expected = torch.stack([ref_by_tok[tok] for tok in valid_toks]).float()
            act_err = float(
                (actual - expected).norm() / expected.norm().clamp_min(1e-12)
            )
            assert act_err < 0.06, (
                f"expert {e} post-act rel L2 error {act_err:.4f} >= 0.06"
            )
    finally:
        layer.destroy()


@pytest.mark.arch_blackwell
def test_direct_kernel_lora_operand_validation():
    """Direct kernel boundary rejects undersized, misstrided, and unaligned LoRA operands."""
    _require_sm100()
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_tensor

    from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.sources.standard import (
        Sm100BlockScaledSymmBuffer,
    )

    cfg = Sm100BlockScaledMoeConfig(
        num_total_experts=4,
        max_tokens_per_rank=64,
        num_topk=2,
        hidden=256,
        intermediate=256,
        rank=0,
        world_size=1,
        quant_kind="mxfp4_mxfp8",
        mma_tiler_mnk=(256, 64, 128),
        enable_fc1_lora_delta=True,
    )
    kernel = Sm100BlockScaledSymmBuffer._build_kernel(cfg)
    max_tokens = int(kernel.max_tokens)
    local_e = int(kernel.local_expert_count)

    @cute.jit
    def _check(topk, delta, snap, tok, rank, off):
        kernel._validate_lora_operands(topk, delta, snap, tok, rank, off)

    valid_topk = make_fake_compact_tensor(
        cutlass.Int32, (64, 2), stride_order=(1, 0), assumed_align=16
    )
    valid_delta = make_fake_compact_tensor(
        cutlass.BFloat16, (64, 2, 512), stride_order=(2, 1, 0), assumed_align=16
    )
    valid_snap = make_fake_compact_tensor(
        cutlass.BFloat16, (max_tokens, 256), stride_order=(1, 0), assumed_align=16
    )
    valid_tok = make_fake_compact_tensor(
        cutlass.Int32, (max_tokens,), stride_order=(0,), assumed_align=4
    )
    valid_rank = make_fake_compact_tensor(
        cutlass.Int32, (max_tokens,), stride_order=(0,), assumed_align=4
    )
    valid_off = make_fake_compact_tensor(
        cutlass.Int32, (local_e + 1,), stride_order=(0,), assumed_align=4
    )

    cute.compile(
        _check, valid_topk, valid_delta, valid_snap, valid_tok, valid_rank, valid_off
    )

    # Short expert_offsets (local_e instead of local_e + 1).
    short_off = make_fake_compact_tensor(
        cutlass.Int32, (local_e,), stride_order=(0,), assumed_align=4
    )
    with pytest.raises(ValueError, match="fc1_snapshot_expert_offsets length"):
        cute.compile(
            _check,
            valid_topk,
            valid_delta,
            valid_snap,
            valid_tok,
            valid_rank,
            short_off,
        )

    # Short token_ids / src_ranks / snapshot.
    short_1d = make_fake_compact_tensor(
        cutlass.Int32, (max_tokens - 1,), stride_order=(0,), assumed_align=4
    )
    with pytest.raises(ValueError, match="fc1_snapshot_token_ids length"):
        cute.compile(
            _check, valid_topk, valid_delta, valid_snap, short_1d, valid_rank, valid_off
        )
    with pytest.raises(ValueError, match="fc1_snapshot_src_ranks length"):
        cute.compile(
            _check, valid_topk, valid_delta, valid_snap, valid_tok, short_1d, valid_off
        )
    short_snap = make_fake_compact_tensor(
        cutlass.BFloat16, (max_tokens - 1, 256), stride_order=(1, 0), assumed_align=16
    )
    with pytest.raises(ValueError, match="fc1_post_act_snapshot dim 0"):
        cute.compile(
            _check,
            valid_topk,
            valid_delta,
            short_snap,
            valid_tok,
            valid_rank,
            valid_off,
        )

    # Short delta token rows.
    short_delta = make_fake_compact_tensor(
        cutlass.BFloat16, (32, 2, 512), stride_order=(2, 1, 0), assumed_align=16
    )
    with pytest.raises(ValueError, match="fc1_lora_delta has 32 token rows"):
        cute.compile(
            _check,
            valid_topk,
            short_delta,
            valid_snap,
            valid_tok,
            valid_rank,
            valid_off,
        )

    # Non-unit inner stride on delta and snapshot.
    strided_delta = make_fake_tensor(
        cutlass.BFloat16, (64, 2, 512), stride=(2048, 1024, 2), assumed_align=16
    )
    with pytest.raises(ValueError, match="fc1_lora_delta must have unit inner stride"):
        cute.compile(
            _check,
            valid_topk,
            strided_delta,
            valid_snap,
            valid_tok,
            valid_rank,
            valid_off,
        )
    strided_snap = make_fake_tensor(
        cutlass.BFloat16, (max_tokens, 256), stride=(512, 2), assumed_align=16
    )
    with pytest.raises(
        ValueError, match="fc1_post_act_snapshot must have unit inner stride"
    ):
        cute.compile(
            _check,
            valid_topk,
            valid_delta,
            strided_snap,
            valid_tok,
            valid_rank,
            valid_off,
        )

    # Overlapping snapshot rows (stride[0] < intermediate_downproj).
    overlapping_snap = make_fake_tensor(
        cutlass.BFloat16, (max_tokens, 256), stride=(8, 1), assumed_align=16
    )
    with pytest.raises(ValueError, match="fc1_post_act_snapshot stride\\[0\\]"):
        cute.compile(
            _check,
            valid_topk,
            valid_delta,
            overlapping_snap,
            valid_tok,
            valid_rank,
            valid_off,
        )

    # Dynamic delta outer stride without guaranteed 8-element (16 B) divisibility.
    unproven_dyn_delta = make_fake_tensor(
        cutlass.BFloat16,
        (64, 2, 512),
        stride=(cute.sym_int64(divisibility=1), 512, 1),
        assumed_align=16,
    )
    with pytest.raises(ValueError, match="fc1_lora_delta dynamic stride\\[0\\]"):
        cute.compile(
            _check,
            valid_topk,
            unproven_dyn_delta,
            valid_snap,
            valid_tok,
            valid_rank,
            valid_off,
        )

    # Unaligned delta / snapshot pointers.
    unaligned_delta = make_fake_compact_tensor(
        cutlass.BFloat16, (64, 2, 512), stride_order=(2, 1, 0), assumed_align=2
    )
    with pytest.raises(
        ValueError, match="fc1_lora_delta must be declared 16 B aligned"
    ):
        cute.compile(
            _check,
            valid_topk,
            unaligned_delta,
            valid_snap,
            valid_tok,
            valid_rank,
            valid_off,
        )
    unaligned_snap = make_fake_compact_tensor(
        cutlass.BFloat16, (max_tokens, 256), stride_order=(1, 0), assumed_align=2
    )
    with pytest.raises(
        ValueError, match="fc1_post_act_snapshot must be declared 16 B aligned"
    ):
        cute.compile(
            _check,
            valid_topk,
            valid_delta,
            unaligned_snap,
            valid_tok,
            valid_rank,
            valid_off,
        )


@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    "mma_n,expect_overlap",
    [(128, False), (256, True)],
)
def test_mxfp4_mxfp8_overlapping_accum_modes(mma_n, expect_overlap, monkeypatch):
    """Verify both overlapping_accum=False and overlapping_accum=True with LoRA TMEM transpose."""
    _require_sm100()
    monkeypatch.setenv("MEGA_NO_DIST", "1")

    capacity, num_tokens, hidden, intermediate, top_k, num_experts = (
        320,
        290,
        256,
        256,
        2,
        2,
    )
    rng = torch.Generator(device="cuda").manual_seed(54680 + mma_n)
    w13 = (
        torch.randn(num_experts, 2 * intermediate, hidden, device="cuda", generator=rng)
        / hidden**0.5
    ).bfloat16()
    w2 = (
        torch.randn(num_experts, hidden, intermediate, device="cuda", generator=rng)
        / intermediate**0.5
    ).bfloat16()
    layer = moe_ep.MoEEpLayer(
        moe_ep.BootstrapConfig(world_size=1, rank=0),
        moe_ep.FleetParams(
            num_experts=num_experts,
            max_tokens_per_rank=capacity,
            token_hidden_size=hidden,
        ),
        moe_ep.MoEWeightPack(w13, w2),
        backend=moe_ep.MegaConfig(
            megakernel=moe_ep.Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig(
                intermediate_size=intermediate,
                top_k=top_k,
                enable_fc1_lora_delta=True,
                knobs={"mma_tiler_mnk": (256, mma_n, 128), "schedule_mode": "grouped"},
            )
        ),
    )
    try:
        ws = layer._ensure_workspace()
        assert ws.kernel.epilogue.overlapping_accum is expect_overlap
        x = torch.randn(num_tokens, hidden, device="cuda", generator=rng).bfloat16()
        base_e = torch.arange(num_tokens, device="cuda", dtype=torch.int32)
        topk_ids = torch.stack(
            [base_e % num_experts, (base_e + 1) % num_experts], dim=-1
        )
        topk_weights = torch.softmax(
            torch.randn(num_tokens, top_k, device="cuda", generator=rng), dim=-1
        ).float()
        delta1 = (
            torch.randn(
                num_tokens, top_k, 2 * intermediate, device="cuda", generator=rng
            )
            * 0.4
        ).bfloat16()
        t = moe_ep.MoEEpTensors(
            hidden_states=x,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            gemm1_lora_delta=delta1,
        )
        layer.warmup(t)
        res = layer.forward(t)
        _check_snapshot_and_output(
            layer,
            res,
            delta1,
            num_tokens,
            num_experts=num_experts,
            intermediate=intermediate,
        )
    finally:
        layer.destroy()


@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    "mode,intermediate",
    [("mxfp8", 192), ("nvfp4", 192)],
)
def test_partial_intermediate_tile_lora_delta_and_snapshot(
    mode, intermediate, monkeypatch
):
    """Exercise intermediate-tail guards (intermediate not divisible by cluster M/2=128)."""
    _require_sm100()
    monkeypatch.setenv("MEGA_NO_DIST", "1")

    capacity, num_tokens, hidden, top_k, num_experts = 96, 77, 256, 2, 2
    rng = torch.Generator(device="cuda").manual_seed(9000 + intermediate)
    w13 = (
        torch.randn(num_experts, 2 * intermediate, hidden, device="cuda", generator=rng)
        / hidden**0.5
    ).bfloat16()
    w2 = (
        torch.randn(num_experts, hidden, intermediate, device="cuda", generator=rng)
        / intermediate**0.5
    ).bfloat16()
    knobs = {"mma_tiler_mnk": (256, 128, 128), "schedule_mode": "grouped"}
    if mode == "mxfp8":
        megakernel = moe_ep.Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(
            intermediate_size=intermediate,
            top_k=top_k,
            enable_fc1_lora_delta=True,
            knobs=knobs,
        )
    else:
        megakernel = moe_ep.Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
            intermediate_size=intermediate,
            top_k=top_k,
            fc1_alpha=torch.tensor([0.9, 1.15], device="cuda", dtype=torch.float32),
            fc2_alpha=torch.tensor([1.05, 0.85], device="cuda", dtype=torch.float32),
            enable_fc1_lora_delta=True,
            knobs=knobs,
        )
    layer = moe_ep.MoEEpLayer(
        moe_ep.BootstrapConfig(world_size=1, rank=0),
        moe_ep.FleetParams(
            num_experts=num_experts,
            max_tokens_per_rank=capacity,
            token_hidden_size=hidden,
        ),
        moe_ep.MoEWeightPack(w13, w2),
        backend=moe_ep.MegaConfig(megakernel=megakernel),
    )
    try:
        ws = layer._ensure_workspace()
        assert intermediate % (ws.kernel.epilogue.mma_tiler_mnk[0] // 2) != 0
        x = torch.randn(num_tokens, hidden, device="cuda", generator=rng).bfloat16()
        base_e = torch.arange(num_tokens, device="cuda", dtype=torch.int32)
        topk_ids = torch.stack(
            [base_e % num_experts, (base_e + 1) % num_experts], dim=-1
        )
        topk_weights = torch.full(
            (num_tokens, top_k), 0.5, dtype=torch.float32, device="cuda"
        )
        delta1 = (
            torch.randn(
                num_tokens, top_k, 2 * intermediate, device="cuda", generator=rng
            )
            * 0.4
        ).bfloat16()
        t = moe_ep.MoEEpTensors(
            hidden_states=x,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            gemm1_lora_delta=delta1,
        )
        layer.warmup(t)
        res = layer.forward(t)
        _check_snapshot_and_output(
            layer,
            res,
            delta1,
            num_tokens,
            num_experts=num_experts,
            intermediate=intermediate,
            out_tol=0.06 if mode == "nvfp4" else 0.03,
            act_tol=0.06 if mode == "nvfp4" else 0.03,
        )
    finally:
        layer.destroy()


@pytest.mark.arch_blackwell
@pytest.mark.gpu_2
def test_multirank_ep_lora_delta_and_snapshot_src_rank(monkeypatch):
    """Multi-rank EP test with colliding local token IDs and distinct per-rank LoRA deltas."""
    _require_sm100()
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world < 2:
        pytest.skip("requires multi-GPU launch (WORLD_SIZE >= 2)")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", str(rank))))
    if not dist.is_initialized():
        dist.init_process_group("nccl")

    capacity, num_tokens, hidden, intermediate, top_k, local_experts = (
        64,
        16,
        256,
        256,
        2,
        2,
    )
    num_experts = local_experts * world
    rng = torch.Generator(device="cuda").manual_seed(546800 + rank)
    w13 = (
        torch.randn(
            local_experts, 2 * intermediate, hidden, device="cuda", generator=rng
        )
        / hidden**0.5
    ).bfloat16()
    w2 = (
        torch.randn(local_experts, hidden, intermediate, device="cuda", generator=rng)
        / intermediate**0.5
    ).bfloat16()
    layer = moe_ep.MoEEpLayer(
        moe_ep.BootstrapConfig(world_size=world, rank=rank),
        moe_ep.FleetParams(
            num_experts=num_experts,
            max_tokens_per_rank=capacity,
            token_hidden_size=hidden,
        ),
        moe_ep.MoEWeightPack(w13, w2),
        backend=moe_ep.MegaConfig(
            megakernel=moe_ep.Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig(
                intermediate_size=intermediate,
                top_k=top_k,
                enable_fc1_lora_delta=True,
                knobs={"mma_tiler_mnk": (256, 64, 128), "schedule_mode": "grouped"},
            )
        ),
    )
    try:
        x = torch.randn(num_tokens, hidden, device="cuda", generator=rng).bfloat16()
        # Every rank routes token 0..num_tokens-1 to expert 0 (on rank 0) and a remote expert,
        # creating colliding local token_id values across source ranks on the same local expert.
        topk_ids = torch.zeros((num_tokens, top_k), dtype=torch.int32, device="cuda")
        topk_ids[:, 0] = 0
        topk_ids[:, 1] = 1 + (
            torch.arange(num_tokens, device="cuda", dtype=torch.int32)
            % (num_experts - 1)
        )
        topk_weights = torch.full(
            (num_tokens, top_k), 0.5, dtype=torch.float32, device="cuda"
        )
        # Rank-distinct LoRA deltas so cross-rank peer reads and src_rank tagging are verified.
        delta1 = (
            torch.randn(
                num_tokens, top_k, 2 * intermediate, device="cuda", generator=rng
            )
            * 0.4
            + (0.25 if rank == 0 else -0.25)
        ).bfloat16()
        t = moe_ep.MoEEpTensors(
            hidden_states=x,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            gemm1_lora_delta=delta1,
        )
        layer.warmup(t)
        res = layer.forward(t)
        assert isinstance(res, moe_ep.MegaMoeLoraOutput)
        y, act, token_id, expert_offsets, src_rank = res
        ws = layer._ensure_workspace()

        # Gather staged inputs from all ranks to verify finalized output and local expert
        # post-act snapshots against the exact (src_rank, src_token) pairs routed to this rank.
        # View FP8 / E8M0 tensors as uint8 for NCCL all_gather (float8_e8m0fnu is unsupported by NCCL).
        local_x_u8 = ws.x[:num_tokens].contiguous().view(torch.uint8)
        local_sf_u8 = ws.x_sf[:num_tokens].contiguous().view(torch.uint8)
        all_x_u8 = [torch.empty_like(local_x_u8) for _ in range(world)]
        all_sf_u8 = [torch.empty_like(local_sf_u8) for _ in range(world)]
        all_ids = [torch.empty_like(ws.topk_idx[:num_tokens]) for _ in range(world)]
        all_w = [torch.empty_like(ws.topk_weights[:num_tokens]) for _ in range(world)]
        all_d1 = [torch.empty_like(delta1) for _ in range(world)]
        dist.all_gather(all_x_u8, local_x_u8)
        dist.all_gather(all_sf_u8, local_sf_u8)
        dist.all_gather(all_ids, ws.topk_idx[:num_tokens])
        dist.all_gather(all_w, ws.topk_weights[:num_tokens])
        dist.all_gather(all_d1, delta1)

        cat_x = torch.cat(all_x_u8, dim=0).view(ws.x.dtype)
        cat_sf = torch.cat(all_sf_u8, dim=0).view(ws.x_sf.dtype)
        cat_ids = torch.cat(all_ids, dim=0)
        cat_w = torch.cat(all_w, dim=0)
        cat_d1 = torch.cat(all_d1, dim=0)
        ref_y_partial, ref_post_act = compute_megamoe_reference_block_scaled(
            cat_x,
            cat_sf,
            cat_ids,
            cat_w,
            layer._transformed[0][0].permute(0, 2, 1),
            layer._transformed[0][1],
            layer._transformed[1][0].permute(0, 2, 1),
            layer._transformed[1][1],
            quant_kind=ws.config.quant_kind,
            combine_dtype=ws.config.combine_dtype,
            local_expert_offset=rank * local_experts,
            gate_up_clamp=ws.config.gate_up_clamp,
            apply_routing_weights_before_fc2=ws.config.apply_routing_weights_before_fc2,
            num_tokens=world * num_tokens,
            weight_scales_are_swizzled=True,
            return_fp32=True,
            gemm1_lora_delta=cat_d1,
            return_post_act=True,
        )
        dist.all_reduce(ref_y_partial, op=dist.ReduceOp.SUM)
        ref_y_rank = ref_y_partial[rank * num_tokens : (rank + 1) * num_tokens].to(
            torch.bfloat16
        )
        out_err = output_error(y, torch.arange(num_tokens, device="cuda"), ref_y_rank)
        assert out_err < 0.03, f"rank {rank} output rel L2 error {out_err:.4f} >= 0.03"

        offsets = expert_offsets.tolist()
        r_cap = int(ws.kernel.max_tokens)
        assert offsets[0] == 0
        assert all(offsets[i] <= offsets[i + 1] for i in range(local_experts))
        assert offsets[-1] <= r_cap
        if offsets[-1] < r_cap:
            assert (token_id[offsets[-1] :] == -1).all()
            assert (src_rank[offsets[-1] :] == -1).all()
        for e in range(local_experts):
            slab_tokens = token_id[offsets[e] : offsets[e + 1]]
            slab_ranks = src_rank[offsets[e] : offsets[e + 1]]
            slab_act = act[offsets[e] : offsets[e + 1]]
            valid_mask = slab_tokens >= 0
            assert (slab_tokens[~valid_mask] == -1).all()
            assert (slab_ranks[~valid_mask] == -1).all()
            pairs = list(
                zip(
                    slab_ranks[valid_mask].tolist(),
                    slab_tokens[valid_mask].tolist(),
                    strict=True,
                )
            )
            # Every (src_rank, src_token) pair must be unique even when local token_id collides.
            assert len(pairs) == len(set(pairs))
            ref_toks, ref_acts = ref_post_act[e]
            ref_pairs = [
                (tok // num_tokens, tok % num_tokens) for tok in ref_toks.tolist()
            ]
            assert sorted(pairs) == sorted(ref_pairs)
            if rank == 0 and e == 0:
                # Token 0 from both rank 0 and rank 1 must both appear on expert 0 with distinct src_rank.
                assert (0, 0) in pairs and (1, 0) in pairs
            if pairs:
                ref_by_pair = {p: ref_acts[i] for i, p in enumerate(ref_pairs)}
                actual = slab_act[valid_mask].float()
                expected = torch.stack([ref_by_pair[p] for p in pairs]).float()
                act_err = float(
                    (actual - expected).norm() / expected.norm().clamp_min(1e-12)
                )
                assert act_err < 0.03, (
                    f"rank {rank} expert {e} post-act rel L2 error {act_err:.4f} >= 0.03"
                )
    finally:
        layer.destroy()
