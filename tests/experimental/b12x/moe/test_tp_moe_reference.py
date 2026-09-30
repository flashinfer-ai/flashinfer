#!/usr/bin/env python3
"""Independent Torch reference for the TP MoE path.

This verifies auto-dispatched `b12x_moe_fp4(...)` against a pure PyTorch
reference that models the NVFP4 block-scaled FC1/FC2 math directly, rather
than using another kernel implementation as the correctness oracle.
"""

from __future__ import annotations

from contextlib import contextmanager

import argparse
import pathlib
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from b12x._lib.intrinsics import fp4_quantize_values_torch
from benchmarks.experimental.b12x.benchmark_moe import (
    MODEL_PATH,
    TP_RANK,
    TP_SIZE,
    ModelSpec,
    get_scale_contract_params,
    load_expert_weights,
)
from b12x.moe.fused_moe._impl import (
    b12x_moe_fp4,
    clear_tp_moe_caches,
)


def _unswizzle_block_scale(swizzled_scale: torch.Tensor, rows: int, cols_blocks: int) -> torch.Tensor:
    cols_padded = ((cols_blocks + 3) // 4) * 4
    rows_padded = ((rows + 127) // 128) * 128
    unswizzled = swizzled_scale.view(torch.float8_e4m3fn).reshape(
        rows_padded // 128, cols_padded // 4, 32, 4, 4,
    )
    unswizzled = unswizzled.permute(0, 3, 2, 1, 4).contiguous()
    unswizzled = unswizzled.reshape(rows_padded, cols_padded)
    return unswizzled[:rows, :cols_blocks].to(torch.float32)


def _moe_reference_f32(
    x: torch.Tensor,
    w1_fp4: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w1_alphas: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_alphas: torch.Tensor,
    a1_gscale: torch.Tensor,
    a2_gscale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    E: int,
    K: int,
    I_tp: int,
) -> torch.Tensor:
    """Pure PyTorch f32 MoE reference matching the NVFP4 block-scaled GEMM."""
    del E
    block_size = 16
    fp8_e4m3_max = float(torch.finfo(torch.float8_e4m3fn).max)

    fp4_lut = torch.tensor(
        [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ],
        dtype=torch.float32,
        device=x.device,
    )
    def _dequant_fp4(packed_u8: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
        lo = (packed_u8 & 0x0F).to(torch.int64)
        hi = ((packed_u8 >> 4) & 0x0F).to(torch.int64)
        return torch.stack([fp4_lut[lo], fp4_lut[hi]], dim=-1).reshape(rows, cols)

    def _apply_block_scales(raw: torch.Tensor, sf_f32: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
        n_blocks = cols // block_size
        sf = sf_f32[:rows, :n_blocks]
        return raw * sf.unsqueeze(-1).expand(rows, n_blocks, block_size).reshape(rows, cols)

    def _quantize_vec_to_fp4_dequant(vals_f32: torch.Tensor, global_scale: float) -> torch.Tensor:
        cols = vals_f32.shape[0]
        n_blocks = cols // block_size
        blocked = vals_f32.reshape(n_blocks, block_size)
        block_max = blocked.abs().amax(dim=-1)

        raw_scale = (block_max * global_scale / 6.0).clamp(max=fp8_e4m3_max)
        sf_e4m3 = raw_scale.to(torch.float8_e4m3fn).to(torch.float32)

        sf_times_gs = sf_e4m3.unsqueeze(-1).expand(n_blocks, block_size).reshape(cols) / global_scale
        scaled = vals_f32 / sf_times_gs.clamp(min=1e-30)
        quant = fp4_quantize_values_torch(scaled)
        sf_only = sf_e4m3.unsqueeze(-1).expand(n_blocks, block_size).reshape(cols)
        return quant * sf_only

    device = x.device
    m = x.shape[0]
    top_k = topk_ids.shape[1]
    output = torch.zeros(m, K, dtype=torch.float32, device=device)

    for t in range(m):
        x_f32 = x[t].float()
        for k_idx in range(top_k):
            eid = int(topk_ids[t, k_idx].item())
            router_w = float(topk_weights[t, k_idx].item())
            alpha_fc1 = float(w1_alphas[eid].item())
            alpha_fc2 = float(w2_alphas[eid].item())

            gs_fc1 = float(a1_gscale[eid].item()) if a1_gscale.numel() > 1 else float(a1_gscale.item())
            gs_fc2 = float(a2_gscale[eid].item()) if a2_gscale.numel() > 1 else float(a2_gscale.item())

            x_dequant = _quantize_vec_to_fp4_dequant(x_f32, gs_fc1)

            w13_sf = _unswizzle_block_scale(w1_blockscale[eid], 2 * I_tp, K // block_size)
            w2_sf = _unswizzle_block_scale(w2_blockscale[eid], K, I_tp // block_size)

            up_dequant = _apply_block_scales(
                _dequant_fp4(w1_fp4[eid, :I_tp], I_tp, K), w13_sf[:I_tp], I_tp, K,
            )
            gate_dequant = _apply_block_scales(
                _dequant_fp4(w1_fp4[eid, I_tp:], I_tp, K), w13_sf[I_tp:], I_tp, K,
            )

            gate_out = (gate_dequant @ x_dequant) * alpha_fc1
            up_out = (up_dequant @ x_dequant) * alpha_fc1
            intermediate = torch.sigmoid(gate_out) * gate_out * up_out

            int_dequant = _quantize_vec_to_fp4_dequant(intermediate, gs_fc2)
            down_dequant = _apply_block_scales(
                _dequant_fp4(w2_fp4[eid], K, I_tp), w2_sf, K, I_tp,
            )
            down_out = (down_dequant @ int_dequant) * alpha_fc2
            output[t] += router_w * down_out

    return output


def _moe_reference_nvfp4(
    x: torch.Tensor,
    w1_fp4: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w1_alphas: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_alphas: torch.Tensor,
    a1_gscale: torch.Tensor,
    a2_gscale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    E: int,
    K: int,
    I_tp: int,
) -> torch.Tensor:
    """Reference that mirrors the kernel's BF16 staging more closely."""
    block_size = 16
    fp8_e4m3_max = float(torch.finfo(torch.float8_e4m3fn).max)

    fp4_lut = torch.tensor(
        [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ],
        dtype=torch.float32,
        device=x.device,
    )
    def _dequant_fp4(packed_u8: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
        lo = (packed_u8 & 0x0F).to(torch.int64)
        hi = ((packed_u8 >> 4) & 0x0F).to(torch.int64)
        return torch.stack([fp4_lut[lo], fp4_lut[hi]], dim=-1).reshape(rows, cols)

    def _apply_block_scales(raw: torch.Tensor, sf_f32: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
        n_blocks = cols // block_size
        sf = sf_f32[:rows, :n_blocks]
        return raw * sf.unsqueeze(-1).expand(rows, n_blocks, block_size).reshape(rows, cols)

    def _quantize_vec_to_fp4_dequant(vals_f32: torch.Tensor, global_scale: float) -> torch.Tensor:
        cols = vals_f32.shape[0]
        n_blocks = cols // block_size
        blocked = vals_f32.reshape(n_blocks, block_size)
        block_max = blocked.abs().amax(dim=-1)

        raw_scale = (block_max * global_scale / 6.0).clamp(max=fp8_e4m3_max)
        sf_e4m3 = raw_scale.to(torch.float8_e4m3fn).to(torch.float32)

        sf_times_gs = sf_e4m3.unsqueeze(-1).expand(n_blocks, block_size).reshape(cols) / global_scale
        scaled = vals_f32 / sf_times_gs.clamp(min=1e-30)
        quant = fp4_quantize_values_torch(scaled)
        sf_only = sf_e4m3.unsqueeze(-1).expand(n_blocks, block_size).reshape(cols)
        return quant * sf_only

    m = x.shape[0]
    top_k = topk_ids.shape[1]
    output = torch.zeros(m, K, dtype=torch.bfloat16, device=x.device)

    contribs: list[list[tuple[int, torch.Tensor]]] = [[] for _ in range(E)]

    for t in range(m):
        x_f32 = x[t].float()
        for k_idx in range(top_k):
            eid = int(topk_ids[t, k_idx].item())
            router_w = float(topk_weights[t, k_idx].item())
            alpha_fc1 = float(w1_alphas[eid].item())
            alpha_fc2 = float(w2_alphas[eid].item())
            gs_fc1 = float(a1_gscale[eid].item()) if a1_gscale.numel() > 1 else float(a1_gscale.item())
            gs_fc2 = float(a2_gscale[eid].item()) if a2_gscale.numel() > 1 else float(a2_gscale.item())

            x_dequant = _quantize_vec_to_fp4_dequant(x_f32, gs_fc1)

            w13_sf = _unswizzle_block_scale(w1_blockscale[eid], 2 * I_tp, K // block_size)
            w2_sf = _unswizzle_block_scale(w2_blockscale[eid], K, I_tp // block_size)

            up_dequant = _apply_block_scales(
                _dequant_fp4(w1_fp4[eid, :I_tp], I_tp, K), w13_sf[:I_tp], I_tp, K,
            )
            gate_dequant = _apply_block_scales(
                _dequant_fp4(w1_fp4[eid, I_tp:], I_tp, K), w13_sf[I_tp:], I_tp, K,
            )

            gate_out = (gate_dequant @ x_dequant) * alpha_fc1
            up_out = (up_dequant @ x_dequant) * alpha_fc1
            intermediate = (torch.sigmoid(gate_out) * gate_out * up_out).to(torch.bfloat16).float()

            int_dequant = _quantize_vec_to_fp4_dequant(intermediate, gs_fc2)
            down_dequant = _apply_block_scales(
                _dequant_fp4(w2_fp4[eid], K, I_tp), w2_sf, K, I_tp,
            )
            down_out = ((down_dequant @ int_dequant) * alpha_fc2).to(torch.bfloat16)
            contribs[eid].append((t, (router_w * down_out.float()).to(torch.bfloat16)))

    for eid in range(E):
        for t, contrib in contribs[eid]:
            output[t] = (output[t].float() + contrib.float()).to(torch.bfloat16)

    return output


@contextmanager
def _make_b12x_moe_binding(
    x: torch.Tensor,
    weights,
    scale_params,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
):
    from b12x.testing.reference.helpers import make_tp_moe_fp4_binding, prepare_tp_moe_fp4_experts

    experts = prepare_tp_moe_fp4_experts(
        a=x,
        a1_gscale=scale_params.a1_gscale,
        w1_fp4=weights.w13_weight,
        w1_blockscale=weights.w13_blockscale_swizzled,
        w1_alphas=scale_params.g1_alphas,
        a2_gscale=scale_params.a2_gscale,
        w2_fp4=weights.w2_weight,
        w2_blockscale=weights.w2_blockscale_swizzled,
        w2_alphas=scale_params.g2_alphas,
        source_format=weights.source_format,
        w13_layout=weights.w13_layout,
    )
    with make_tp_moe_fp4_binding(
        a=x,
        experts=experts,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        input_scales_static=True,
    ) as binding:
        yield binding


def _run_impl(
    impl: str,
    x: torch.Tensor,
    weights,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    scale_contract: str,
    *,
    clear_state: bool = True,
) -> torch.Tensor:

    if clear_state:
        clear_tp_moe_caches()
    scale_params = get_scale_contract_params(weights, scale_contract)
    with _make_b12x_moe_binding(
        x,
        weights,
        scale_params,
        topk_weights,
        topk_ids,
    ) as binding:
        out = b12x_moe_fp4(binding=binding)
        torch.cuda.synchronize()
        return out.detach().clone()


def _run_impl_sequence(
    impl: str,
    x: torch.Tensor,
    weights_sequence: list,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    scale_contract: str,
    *,
    clear_state_between_calls: bool,
) -> list[torch.Tensor]:
    if not clear_state_between_calls:
        clear_tp_moe_caches()

    outputs = []
    for weights in weights_sequence:

        scale_params = get_scale_contract_params(weights, scale_contract)
        if clear_state_between_calls:
            clear_tp_moe_caches()
        binding = _make_b12x_moe_binding(
            x,
            weights,
            scale_params,
            topk_weights,
            topk_ids,
        )
        out = b12x_moe_fp4(binding=binding)
        torch.cuda.synchronize()
        outputs.append(out.detach().clone())
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 128, 512, 4096])
    parser.add_argument("--layer-indices", type=int, nargs="+", default=[0])
    parser.add_argument("--sequence-repeats", type=int, default=1)
    parser.add_argument("--clear-state-between-calls", action="store_true")
    parser.add_argument("--impls", nargs="+", choices=["b12x"], default=["b12x"])
    parser.add_argument("--activation-scale", type=float, default=10.0)
    parser.add_argument("--oracle-mode", choices=["f32", "nvfp4"], default="nvfp4")
    parser.add_argument("--scale-contract", choices=["shared", "per-expert"], default="per-expert")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda")
    torch.empty(1, device=device)

    spec = ModelSpec(
        hidden_size=4096,
        intermediate_size=1024,
        num_experts=512,
        top_k=10,
        tp_size=TP_SIZE,
        tp_rank=TP_RANK,
    )

    weights_by_layer = [
        load_expert_weights(MODEL_PATH, spec, layer_idx=layer_idx)
        for layer_idx in args.layer_indices
    ]
    clear_tp_moe_caches()
    torch.cuda.empty_cache()

    print("Independent TP MoE verification")
    print(
        f"Model: Qwen3.5-397B layers={args.layer_indices}  "
        f"TP={spec.tp_size}  K={spec.hidden_size}  I_tp={spec.I_tp}  E={spec.num_experts}"
    )
    print(f"Activation scale: x *= {args.activation_scale}")
    print(f"Oracle mode: {args.oracle_mode}")
    print(f"Scale contract: {args.scale_contract}")
    print(
        f"Layer sequence repeats: {args.sequence_repeats}  "
        f"clear_state_between_calls={args.clear_state_between_calls}"
    )
    print()

    oracle_fn = _moe_reference_nvfp4 if args.oracle_mode == "nvfp4" else _moe_reference_f32

    for batch_size in args.batch_sizes:
        print(f"batch_size={batch_size}")
        impl_stats: dict[str, list[dict[str, float]]] = defaultdict(list)
        ref_norms: list[float] = []
        ref_maxes: list[float] = []

        for trial_idx in range(args.trials):
            seed = args.base_seed + batch_size + trial_idx * 9973
            torch.manual_seed(seed)
            x = torch.randn(batch_size, spec.hidden_size, dtype=torch.bfloat16, device=device)
            x = (x.float() * args.activation_scale).to(torch.bfloat16)
            routing_logits = torch.randn(batch_size, spec.num_experts, device=device, dtype=torch.float32)
            topk_logits, topk_ids = torch.topk(routing_logits, spec.top_k, dim=-1)
            topk_weights = torch.softmax(topk_logits, dim=-1)
            weights_sequence = weights_by_layer * args.sequence_repeats
            refs = []
            for seq_idx, weights in enumerate(weights_sequence):
                scale_params = get_scale_contract_params(weights, args.scale_contract)
                ref = oracle_fn(
                    x,
                    weights.w13_weight,
                    weights.w13_blockscale_swizzled,
                    scale_params.g1_alphas,
                    weights.w2_weight,
                    weights.w2_blockscale_swizzled,
                    scale_params.g2_alphas,
                    scale_params.a1_gscale,
                    scale_params.a2_gscale,
                    topk_ids,
                    topk_weights,
                    spec.num_experts,
                    spec.hidden_size,
                    spec.I_tp,
                )
                refs.append(ref)
                ref_norm = ref.float().norm().item()
                ref_max = ref.float().abs().max().item()
                ref_norms.append(ref_norm)
                ref_maxes.append(ref_max)
                if len(weights_sequence) == 1:
                    if args.trials == 1:
                        print(f"  reference norm={ref_norm:.5f} max={ref_max:.5f}")
                    else:
                        print(
                            f"  trial={trial_idx} seed={seed} "
                            f"ref_norm={ref_norm:.5f} ref_max={ref_max:.5f}"
                        )
                else:
                    print(
                        f"  reference layer={weights.layer_idx} call={seq_idx} "
                        f"norm={ref_norm:.5f} max={ref_max:.5f}"
                    )

            for impl in args.impls:
                outs = _run_impl_sequence(
                    impl,
                    x,
                    weights_sequence,
                    topk_weights,
                    topk_ids,
                    args.scale_contract,
                    clear_state_between_calls=args.clear_state_between_calls,
                )
                for seq_idx, (weights, ref, out) in enumerate(
                    zip(weights_sequence, refs, outs, strict=True)
                ):
                    diff = (out.float() - ref.float()).abs()
                    out_norm = out.float().norm().item()
                    cos = F.cosine_similarity(
                        out.float().reshape(batch_size, -1),
                        ref.float().reshape(batch_size, -1),
                        dim=1,
                    ).mean().item()
                    stat = {
                        "max_abs": diff.max().item(),
                        "rmse": diff.square().mean().sqrt().item(),
                        "mean_abs": diff.mean().item(),
                        "norm": out_norm,
                        "cos": cos,
                    }
                    impl_stats[impl].append(stat)
                    label = (
                        f"{impl:<10} "
                        if len(weights_sequence) == 1
                        else f"{impl:<10} layer={weights.layer_idx} call={seq_idx} "
                    )
                    print(
                        f"  {label}max_abs={stat['max_abs']:.5f} "
                        f"rmse={stat['rmse']:.5f} "
                        f"mean_abs={stat['mean_abs']:.5f} "
                        f"norm={stat['norm']:.5f} "
                        f"cos={stat['cos']:.6f} cos_dist={1.0 - stat['cos']:.6f}",
                    )

        if args.trials > 1:
            print(
                f"  reference summary norm=[{min(ref_norms):.5f}, {max(ref_norms):.5f}] "
                f"max=[{min(ref_maxes):.5f}, {max(ref_maxes):.5f}]"
            )
            for impl in args.impls:
                stats = impl_stats[impl]
                print(
                    f"  {impl:<10} summary "
                    f"max_abs=[{min(s['max_abs'] for s in stats):.5f}, {max(s['max_abs'] for s in stats):.5f}] "
                    f"rmse=[{min(s['rmse'] for s in stats):.5f}, {max(s['rmse'] for s in stats):.5f}] "
                    f"mean_abs=[{min(s['mean_abs'] for s in stats):.5f}, {max(s['mean_abs'] for s in stats):.5f}] "
                    f"cos=[{min(s['cos'] for s in stats):.6f}, {max(s['cos'] for s in stats):.6f}]"
                )
        print()


if __name__ == "__main__":
    main()
