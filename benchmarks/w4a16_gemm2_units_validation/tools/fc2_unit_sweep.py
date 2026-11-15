"""Kernel-level W4A16 GEMM2 (FC2) harness: bit-exact unit-size A/B and CUPTI timing.

Builds real MoE inputs (quantized NVFP4 weights, random top-k routing), runs moe_sort and
FC1 once, then launches FC2 through FlashInfer's ``_run_grouped_gemm`` with the kernel's
``m_tiles_per_unit`` forced per arm. Arm ``auto`` uses the kernel's own compile-time choice;
arm ``1`` reproduces the pre-change one-tile units. Every arm's whole output buffer
(pre-filled with a sentinel) must match arm ``1`` bit for bit.

Run from the FlashInfer checkout root with PYTHONPATH=$PWD.
"""

import argparse
import json
import math
import sys
import time

import torch

from flashinfer.fused_moe.cute_dsl.blackwell import moe_w4a16 as launcher
from flashinfer.fused_moe.cute_dsl.blackwell.moe_w4a16_kernel import (
    Sm100W4A16GroupedGemmKernel as Kernel,
)
from flashinfer.fused_moe.cute_dsl.moe_utils import (
    allocate_moe_sort_buffers,
    get_max_num_permuted_tokens,
    moe_sort,
)
from flashinfer.testing import bench_gpu_time_with_cupti
from flashinfer.tllm_enums import ActivationType
from tests.moe.utils import create_moe_tensors

_auto_select = Kernel._select_m_tiles_per_unit
_FORCED = {"tiles": None}


def _forced_select(self):
    auto = _auto_select(self)
    if _FORCED["tiles"] is None or self.gather_input or self.activation_type is not None:
        return auto
    return _FORCED["tiles"]


Kernel._select_m_tiles_per_unit = _forced_select


def _compiled_units():
    """m_tiles_per_unit of the most recently traced FC2 kernel (for reporting)."""
    return _LAST.get("tiles")


_LAST = {}
_LAST_BY_KEY = {}
_orig_setup = Kernel._setup_attributes


def _recording_setup(self):
    _orig_setup(self)
    if not self.gather_input:
        _LAST["tiles"] = self.m_tiles_per_unit
        _LAST["load_stages"] = self.num_load2trans_stage
        _LAST["trans_stages"] = self.num_trans2mma_stage
        _LAST["acc_cols"] = self.num_acc_tmem_cols


Kernel._setup_attributes = _recording_setup


def build_inputs(args, num_tokens, route_tile, seed):
    tensors = create_moe_tensors(
        num_tokens=num_tokens,
        hidden_size=args.hidden,
        intermediate_size=args.intermediate,
        num_experts=args.experts,
        num_local_experts=args.experts,
        top_k=args.top_k,
        seed=seed,
        swap_ab=False,
    )
    x = tensors["x_bf16"]
    top_k = args.top_k
    sort_buffers = allocate_moe_sort_buffers(
        num_tokens=num_tokens,
        num_experts=args.experts,
        top_k=top_k,
        num_local_experts=args.experts,
        tile_tokens_dim=route_tile,
        device=x.device,
    )
    (
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        expanded_idx_to_permuted_idx,
        permuted_idx_to_expanded_idx,
        _,
        num_non_exiting_tiles,
    ) = moe_sort(
        token_selected_experts=tensors["token_selected_experts"],
        token_final_scales=tensors["token_final_scales"],
        num_experts=args.experts,
        top_k=top_k,
        local_expert_offset=0,
        num_local_experts=args.experts,
        tile_tokens_dim=route_tile,
        enable_pdl=False,
        **sort_buffers,
    )
    slots = get_max_num_permuted_tokens(num_tokens, top_k, args.experts, route_tile)
    intermediate = torch.empty(
        (slots, args.intermediate), dtype=torch.bfloat16, device=x.device
    )
    fc1_tactic = ((128, route_tile, 256), (1, 1), True)
    launcher._run_grouped_gemm(
        weight=tensors["w1_weight"],
        weight_sf=tensors["w1_weight_sf"],
        activations=x,
        tile_idx_to_expert_idx=tile_idx_to_expert_idx,
        tile_idx_to_mn_limit=tile_idx_to_mn_limit,
        num_non_exiting_tiles=num_non_exiting_tiles,
        alpha=tensors["w1_alpha"],
        output=intermediate,
        num_local_experts=args.experts,
        activation_type=ActivationType.Swiglu,
        swiglu_alpha=1.0,
        swiglu_beta=0.0,
        swiglu_limit=float("inf"),
        situ_beta=None,
        situ_linear_beta=None,
        use_fused_finalize=False,
        permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
        token_final_scales=None,
        enable_pdl=False,
        tactic=fc1_tactic,
        swap_ab=False,
        gather_input=True,
        gather_top_k=top_k,
    )
    torch.cuda.synchronize()
    return dict(
        tensors=tensors,
        intermediate=intermediate,
        tile_idx_to_expert_idx=tile_idx_to_expert_idx,
        tile_idx_to_mn_limit=tile_idx_to_mn_limit,
        num_non_exiting_tiles=num_non_exiting_tiles,
        slots=slots,
        num_route_tiles=int(num_non_exiting_tiles[0].item()),
    )


def run_fc2(inp, args, tactic, output, enable_pdl):
    t = inp["tensors"]
    launcher._run_grouped_gemm(
        weight=t["w2_weight"],
        weight_sf=t["w2_weight_sf"],
        activations=inp["intermediate"],
        tile_idx_to_expert_idx=inp["tile_idx_to_expert_idx"],
        tile_idx_to_mn_limit=inp["tile_idx_to_mn_limit"],
        num_non_exiting_tiles=inp["num_non_exiting_tiles"],
        alpha=t["w2_alpha"],
        output=output,
        num_local_experts=args.experts,
        activation_type=None,
        swiglu_alpha=1.0,
        swiglu_beta=0.0,
        swiglu_limit=float("inf"),
        situ_beta=None,
        situ_linear_beta=None,
        use_fused_finalize=False,
        permuted_idx_to_expanded_idx=None,
        token_final_scales=None,
        enable_pdl=enable_pdl,
        tactic=tactic,
        swap_ab=False,
        gather_input=False,
        gather_top_k=1,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hidden", type=int, required=True)
    p.add_argument("--intermediate", type=int, required=True, help="per-rank FC2 K")
    p.add_argument("--experts", type=int, default=256)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--tokens", default="1,2,4,8,16,32,64,128")
    p.add_argument("--route-tiles", default="8,16")
    p.add_argument("--clusters", default="1x1")
    p.add_argument("--arms", default="1,auto", help="comma list: 1, auto, or integers")
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--no-time", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label", default="")
    args = p.parse_args()

    torch.cuda.set_device(0)
    # One compiled-kernel cache per arm: the unit size is not part of the launcher key.
    arm_caches = {arm: {} for arm in args.arms.split(",")}
    for num_tokens in [int(v) for v in args.tokens.split(",")]:
        for route_tile in [int(v) for v in args.route_tiles.split(",")]:
            inp = build_inputs(args, num_tokens, route_tile, args.seed + num_tokens)
            for cluster in args.clusters.split(","):
                cm, cn = (int(v) for v in cluster.split("x"))
                tactic = ((128, route_tile, 256), (cm, cn), True)
                outputs = {}
                for arm in args.arms.split(","):
                    _FORCED["tiles"] = None if arm == "auto" else int(arm)
                    launcher._kernel_cache = arm_caches[arm]
                    _LAST.clear()
                    out = torch.full(
                        (inp["slots"], args.hidden), -7.25, dtype=torch.bfloat16, device="cuda"
                    )
                    rec = dict(
                        label=args.label, hidden=args.hidden, k=args.intermediate,
                        experts=args.experts, top_k=args.top_k, tokens=num_tokens,
                        route_tile=route_tile, cluster=cluster, arm=arm,
                        route_tiles_active=inp["num_route_tiles"],
                    )
                    try:
                        run_fc2(inp, args, tactic, out, enable_pdl=False)
                        torch.cuda.synchronize()
                    except Exception as e:  # infeasible forced unit size, etc.
                        rec["error"] = repr(e)[:300]
                        print(json.dumps(rec), flush=True)
                        continue
                    rec.update({k: v for k, v in _LAST.items()})
                    _LAST_BY_KEY.setdefault((arm, route_tile, cluster), dict(_LAST))
                    rec.update(_LAST_BY_KEY[(arm, route_tile, cluster)])
                    outputs[arm] = out.clone()
                    if "1" in outputs and arm != "1":
                        a = outputs["1"].view(torch.int16)
                        b = out.view(torch.int16)
                        rec["bitwise_vs_1"] = bool(torch.equal(a, b))
                        rec["ndiff_vs_1"] = int((a != b).sum().item())
                    if not args.no_time:
                        times = bench_gpu_time_with_cupti(
                            lambda: run_fc2(inp, args, tactic, out, enable_pdl=False),
                            dry_run_iters=args.warmup,
                            repeat_iters=args.iters,
                            use_cuda_graph=True,
                            cold_l2_cache=True,
                        )
                        times = sorted(float(t) * 1000.0 for t in times)  # ms -> us
                        rec["us_median"] = times[len(times) // 2]
                        rec["us_p10"] = times[len(times) // 10]
                        rec["us_p90"] = times[(9 * len(times)) // 10]
                        rec["n"] = len(times)
                    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
