"""Single-GPU per-tactic latency sweep at Rong Song's exact per-rank MoE shape.

Goal: turn Issue 1 ("autotuner can pick a tactic ~2x slower than default")
from a *mechanism argument* into *direct evidence*, without needing the old
0.6.12 build.

For the exact post-pad shape Rong Song's gpt-oss-120b EP4 decode runs
(``trtllm-gen bmm_MxE4m3_MxE2m1MxE4m3``, hidden=intermediate=3072 after
TRT-LLM pads 2880 up to a 256-multiple, 128 experts / EP4 -> 32 local experts,
top-4, swiglu), this script:

  1. enumerates *every* valid (tile_N, config) tactic the autotuner may pick;
  2. CUDA-graph-times each one, plus the heuristic default (tactic=-1), at the
     real decode M;
  3. reports the latency spread and flags how many tactics are slower than the
     default -- i.e. the "trap" tactics a naive fastest-wins-with-noise tuner
     could freeze into the decode CUDA graph;
  4. applies 改动① 's rule (only switch off the default if a tactic beats it by
     ``FLASHINFER_AUTOTUNE_SWITCH_MARGIN``) and prints what *our* autotuner would
     select -- demonstrating it can never land on a slower-than-default tactic.

Run on one Blackwell (SM100) GPU:

    cd /lustre/fsw/coreai_dlfw_dev/albecheng/flashinfer-autotuner-align
    python tests/moe/sweep_tactic_latency.py
"""

import os
import sys

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from flashinfer import ActivationType, RoutingMethodType
from flashinfer.autotuner import AutoTuner
from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe, WeightLayout  # noqa: F401
from flashinfer.fused_moe.utils import make_balanced_local_topk_ids
from flashinfer.jit.fused_moe import gen_trtllm_gen_fused_moe_sm100_module
from flashinfer.utils import device_support_pdl, get_compute_capability

from tests.moe.test_trtllm_gen_moe_autotune_tactics import (
    _build_fp4_routed_moe_inputs,
    _enumerate_valid_tactics,
    _force_tactic_in_autotuner_cache,
    _last_positive_power_of_2,
    _moe_profile_shapes,
    _TEST_OP_FP4,
)

# --- Rong Song's exact post-pad per-rank shape (override via env) -------------
HIDDEN = int(os.environ.get("HIDDEN", "3072"))
INTERMEDIATE = int(os.environ.get("INTERMEDIATE", "3072"))
NUM_LOCAL_EXPERTS = int(os.environ.get("NUM_LOCAL_EXPERTS", "32"))  # 128 / EP4
TOP_K = int(os.environ.get("TOP_K", "4"))
QUANT_MODE = os.environ.get("QUANT_MODE", "MxFP4xMxFP8")  # -> bmm_MxE4m3_MxE2m1MxE4m3
MS = [int(x) for x in os.environ.get("MS", "80,160,640").split(",")]
TIME_ITERS = int(os.environ.get("TIME_ITERS", "50"))
WARMUP_ITERS = int(os.environ.get("WARMUP_ITERS", "10"))
MARGIN = float(os.environ.get("FLASHINFER_AUTOTUNE_SWITCH_MARGIN", "0.03"))


def _time_call_graph(call, iters, warmup):
    """Steady-state latency under a captured CUDA graph (deployment-faithful)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            call()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            call()
    except Exception as e:  # noqa: BLE001
        print(f"    [graph-capture-failed] {e}", flush=True)
        return float("inf")

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms/iter


def _build_call(inputs, packed_topk, enable_pdl, routing_method_type, tune_max):
    def _call():
        return trtllm_fp4_block_scale_routed_moe(
            topk_ids=packed_topk,
            routing_bias=None,
            hidden_states=inputs["hidden_states"],
            hidden_states_scale=inputs["hidden_states_scale"],
            gemm1_weights=inputs["w13"],
            gemm1_weights_scale=inputs["w13_scale"],
            gemm1_bias=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm2_weights=inputs["w2"],
            gemm2_weights_scale=inputs["w2_scale"],
            gemm2_bias=None,
            output1_scale_scalar=inputs["output1_scale_scalar"],
            output1_scale_gate_scalar=inputs["output1_scale_gate_scalar"],
            output2_scale_scalar=inputs["output2_scale_scalar"],
            num_experts=NUM_LOCAL_EXPERTS,
            top_k=TOP_K,
            n_group=None,
            topk_group=None,
            intermediate_size=INTERMEDIATE,
            local_expert_offset=0,
            local_num_experts=NUM_LOCAL_EXPERTS,
            routed_scaling_factor=None,
            routing_method_type=routing_method_type.value,
            do_finalize=True,
            enable_pdl=enable_pdl,
            activation_type=ActivationType.Swiglu.value,
            tune_max_num_tokens=tune_max,
        )[0]

    return _call


def main():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    if get_compute_capability(device)[0] != 10:
        print("This sweep only runs on SM100/SM103 (Blackwell).")
        return

    enable_pdl = device_support_pdl(device)
    routing_method_type = RoutingMethodType.Renormalize
    moe_op = gen_trtllm_gen_fused_moe_sm100_module().build_and_load()

    print(
        f"\n=== per-tactic latency sweep (Rong Song shape) ===\n"
        f"hidden={HIDDEN}  intermediate={INTERMEDIATE}  "
        f"local_experts={NUM_LOCAL_EXPERTS}  top_k={TOP_K}  quant={QUANT_MODE}\n"
        f"kernel=bmm_MxE4m3_MxE2m1MxE4m3  switch_margin={MARGIN}\n"
        f"decode M={MS}\n",
        flush=True,
    )

    for M in MS:
        torch.manual_seed(42)
        inputs = _build_fp4_routed_moe_inputs(
            num_tokens=M,
            hidden_size=HIDDEN,
            intermediate_size=INTERMEDIATE,
            top_k=TOP_K,
            num_experts=NUM_LOCAL_EXPERTS,
            quant_mode=QUANT_MODE,
            routing_method_type=routing_method_type,
            device=device,
        )
        # Balanced post-all-to-all local routing (matches EP-aware tuning dummy).
        topk_ids = make_balanced_local_topk_ids(
            num_tokens=M,
            top_k=TOP_K,
            num_local_experts=NUM_LOCAL_EXPERTS,
            local_expert_offset=0,
            device=device,
        )
        expert_weights = torch.ones(M, TOP_K, device=device, dtype=torch.bfloat16)
        packed_topk = (topk_ids.to(torch.int32) << 16) | expert_weights.to(
            torch.bfloat16
        ).view(torch.int16)
        inputs["packed_topk"] = packed_topk
        inputs["expert_weights"] = expert_weights

        tune_max = max(_last_positive_power_of_2(M), 16)
        bucket_m = min(_last_positive_power_of_2(M), tune_max)
        profile_shapes = _moe_profile_shapes(inputs, M, bucket_m)
        call = _build_call(inputs, packed_topk, enable_pdl, routing_method_type, tune_max)

        valid_tactics = _enumerate_valid_tactics(
            moe_op,
            QUANT_MODE,
            TOP_K,
            HIDDEN,
            INTERMEDIATE,
            NUM_LOCAL_EXPERTS,
            M,
        )

        # Heuristic default baseline (tactic=-1).
        _force_tactic_in_autotuner_cache(profile_shapes, None, custom_op=_TEST_OP_FP4)
        t_default = _time_call_graph(call, TIME_ITERS, WARMUP_ITERS)

        # Every explicit tactic.
        rows = []  # (tactic, ms)
        for tac in valid_tactics:
            _force_tactic_in_autotuner_cache(
                profile_shapes, list(tac), custom_op=_TEST_OP_FP4
            )
            t = _time_call_graph(call, TIME_ITERS, WARMUP_ITERS)
            rows.append((list(tac), t))

        rows_finite = [(t, ms) for t, ms in rows if ms != float("inf")]
        rows_finite.sort(key=lambda x: x[1])
        if not rows_finite:
            print(f"M={M}: no tactic ran successfully", flush=True)
            continue

        fastest_tac, fastest_ms = rows_finite[0]
        slowest_tac, slowest_ms = rows_finite[-1]
        n_slower_than_default = sum(1 for _, ms in rows_finite if ms > t_default)
        n_2x = sum(1 for _, ms in rows_finite if ms >= 1.8 * fastest_ms)

        # What 改动① would select: the fastest tactic, but only if it beats the
        # default by the margin; otherwise keep the default (tactic=-1).
        if fastest_ms < t_default * (1.0 - MARGIN):
            chosen = f"tactic={fastest_tac} ({fastest_ms*1e3:.1f}us)"
        else:
            chosen = f"DEFAULT (-1) ({t_default*1e3:.1f}us)"

        print(
            f"M={M:>5}  valid_tactics={len(valid_tactics)}\n"
            f"    default(-1) = {t_default*1e3:7.1f}us\n"
            f"    fastest     = {fastest_ms*1e3:7.1f}us  {fastest_tac}\n"
            f"    slowest     = {slowest_ms*1e3:7.1f}us  {slowest_tac}"
            f"   (slowest/fastest={slowest_ms/fastest_ms:.2f}x, "
            f"slowest/default={slowest_ms/t_default:.2f}x)\n"
            f"    #tactics slower than default = {n_slower_than_default}/{len(rows_finite)}"
            f"   |  #tactics >=1.8x fastest = {n_2x}\n"
            f"    -> 改动① selects: {chosen}",
            flush=True,
        )
        # Full sorted distribution for the record.
        dist = "  ".join(f"{ms*1e3:.0f}" for _, ms in rows_finite)
        print(f"    sorted us: [{dist}]", flush=True)

    AutoTuner.get().profiling_cache.clear()
    AutoTuner.get()._file_configs.clear()


if __name__ == "__main__":
    main()
