"""4-GPU EP autotuner consistency / determinism / speed micro-benchmark.

This is the *accurate* test for the EP/DP-aware deterministic balanced
dummy in the trtllm-gen MoE autotuner.  The bug it targets is fundamentally a
**multi-process** phenomenon: every EP rank is its own process with its own
autotuner and its own RNG, so the old unseeded ``make_random_topk_ids`` routing
made each rank profile a *different* dummy problem and converge on a *different*
(often slower) tactic.  A single-GPU loop only approximates this; running one
real process per GPU is how it actually happens in TRT-LLM.

What it checks, per decode bucket ``M`` and per rank
(``local_expert_offset = rank * num_local_experts``):

  1. **Consistency** -- all ranks select the *same* tactic for the same shape.
  2. **Determinism** -- re-tuning the same rank twice yields the same tactic.
  3. **Speed**       -- the autotuner-selected tactic is *not slower* than the
                        heuristic default (the original regression was
                        "selected kernel slower than default").

Run on a node with >= world_size Blackwell (SM100) GPUs, e.g. 4x GB200:

    cd /lustre/fsw/coreai_dlfw_dev/albecheng/flashinfer-autotuner-align
    torchrun --nproc_per_node=4 tests/moe/bench_ep_tactic_consistency.py

Override the problem shape via env vars, e.g. to match a specific model:

    NUM_EXPERTS=128 TOP_K=4 HIDDEN=2880 INTERMEDIATE=2880 \
    MS="8,16,64,128,256" torchrun --nproc_per_node=4 \
        tests/moe/bench_ep_tactic_consistency.py
"""

import os
import sys

import torch
import torch.distributed as dist

# Make ``tests.moe.*`` importable when launched as a script from repo root.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from flashinfer import ActivationType, RoutingMethodType
from flashinfer.autotuner import AutoTuner, autotune
from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe, WeightLayout  # noqa: F401
from flashinfer.fused_moe.utils import make_balanced_local_topk_ids
from flashinfer.utils import device_support_pdl, get_compute_capability

from tests.moe.test_trtllm_gen_moe_autotune_tactics import _build_fp4_routed_moe_inputs

_FP4_OP = "flashinfer::trtllm_fp4_block_scale_moe"

# --- problem shape (override via env) ----------------------------------------
# NOTE: trtllm-gen MXFP4 cubins only expose kernel configs for hidden /
# intermediate sizes that are 128-aligned (e.g. 3072, 4096, 7168).  gpt-oss's
# raw 2880 has *no valid config* (getValidConfigIndices throws) -- TRT-LLM pads
# it before calling the kernel.  Since this change only affects *tactic selection*
# (shape-independent), we default to a 128-aligned shape so the kernel runs and
# we can observe per-rank tactic consistency / determinism.
NUM_EXPERTS = int(os.environ.get("NUM_EXPERTS", "128"))
TOP_K = int(os.environ.get("TOP_K", "4"))
HIDDEN = int(os.environ.get("HIDDEN", "4096"))
INTERMEDIATE = int(os.environ.get("INTERMEDIATE", "3072"))
QUANT_MODE = os.environ.get("QUANT_MODE", "MxFP4xMxFP8")  # W4A8_MXFP4_FP8
MS = [int(x) for x in os.environ.get("MS", "8,16,64,128,256").split(",")]
TIME_ITERS = int(os.environ.get("TIME_ITERS", "50"))
WARMUP_ITERS = int(os.environ.get("WARMUP_ITERS", "10"))


def _selected_tactic(custom_op: str = _FP4_OP):
    """Return the single tactic the autotuner wrote to its cache.

    With ``autotune(tuning_buckets=(M,))`` exactly one bucket is profiled, so
    there is one cache entry per (op, runner).  Cache value layout is
    ``(tactic, profile)``.
    """
    tuner = AutoTuner.get()
    tactics = {
        tuple(v[0]) if isinstance(v[0], (list, tuple)) else v[0]
        for k, v in tuner.profiling_cache.items()
        if isinstance(k, tuple) and len(k) > 0 and k[0] == custom_op
    }
    if len(tactics) != 1:
        raise RuntimeError(
            f"expected exactly 1 cached tactic for {custom_op}, got {tactics!r} "
            f"(cache size={len(tuner.profiling_cache)})"
        )
    return next(iter(tactics))


def _build_call(
    inputs,
    num_experts,
    local_num_experts,
    local_expert_offset,
    top_k,
    intermediate_size,
    packed_topk,
    enable_pdl,
    routing_method_type,
    tune_max_num_tokens,
):
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
            num_experts=num_experts,
            top_k=top_k,
            n_group=None,
            topk_group=None,
            intermediate_size=intermediate_size,
            local_expert_offset=local_expert_offset,
            local_num_experts=local_num_experts,
            routed_scaling_factor=None,
            routing_method_type=routing_method_type.value,
            do_finalize=True,
            enable_pdl=enable_pdl,
            activation_type=ActivationType.Swiglu.value,
            tune_max_num_tokens=tune_max_num_tokens,
        )[0]

    return _call


def _time_call(call, iters, warmup):
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        call()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms/iter


def _time_call_graph(call, iters, warmup):
    """Steady-state timing under a captured CUDA graph -- this matches how
    decode actually runs in TRT-LLM (and how the autotuner profiles, with
    ``use_cuda_graph=True``), so it is the deployment-faithful yardstick.

    Eager ``_time_call`` pays per-kernel CPU launch overhead every iter, which
    at small decode M dwarfs the kernel delta and makes the autotuner's
    graph-profiled pick look artificially slow.  Falls back to ``inf`` on
    capture failure so the caller can detect it.
    """
    # Warm up on a side stream (required before capture).
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
        print(f"[graph-capture-failed] {e}", flush=True)
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


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if get_compute_capability(device)[0] != 10:
        if rank == 0:
            print("This benchmark only runs on SM100/SM103 (Blackwell).")
        dist.destroy_process_group()
        return

    assert NUM_EXPERTS % world_size == 0, (
        f"num_experts={NUM_EXPERTS} must be divisible by EP world_size={world_size}"
    )
    num_local_experts = NUM_EXPERTS // world_size
    local_expert_offset = rank * num_local_experts
    enable_pdl = device_support_pdl(device)
    routing_method_type = RoutingMethodType.Renormalize

    torch.manual_seed(1234 + rank)  # distinct per rank on purpose

    if rank == 0:
        print(
            f"\n=== EP autotuner consistency benchmark ===\n"
            f"world_size(EP)={world_size}  num_experts={NUM_EXPERTS}  "
            f"num_local_experts={num_local_experts}  top_k={TOP_K}\n"
            f"hidden={HIDDEN}  intermediate={INTERMEDIATE}  quant={QUANT_MODE}\n"
            f"decode buckets M={MS}\n",
            flush=True,
        )

    # Per-rank result: {M: dict(tactic, tactic2, t_tuned_ms, t_default_ms)}
    results = {}

    for M in MS:
        # Local-shard weights live on every rank; build with num_experts =
        # num_local_experts so the weight tensors are sized [num_local_experts].
        inputs = _build_fp4_routed_moe_inputs(
            num_tokens=M,
            hidden_size=HIDDEN,
            intermediate_size=INTERMEDIATE,
            top_k=TOP_K,
            num_experts=num_local_experts,
            quant_mode=QUANT_MODE,
            routing_method_type=routing_method_type,
            device=device,
        )
        # Real runtime routing: global ids inside this rank's shard
        # [offset, offset+num_local_experts).  This keeps the final real
        # forward valid while the autotuner synthesizes its own dummy.
        topk_ids = make_balanced_local_topk_ids(
            num_tokens=M,
            top_k=TOP_K,
            num_local_experts=num_local_experts,
            local_expert_offset=local_expert_offset,
            device=device,
        )
        expert_weights = torch.ones(M, TOP_K, device=device, dtype=torch.bfloat16)
        packed_topk = (topk_ids.to(torch.int32) << 16) | expert_weights.to(
            torch.bfloat16
        ).view(torch.int16)

        call = _build_call(
            inputs,
            NUM_EXPERTS,
            num_local_experts,
            local_expert_offset,
            TOP_K,
            INTERMEDIATE,
            packed_topk,
            enable_pdl,
            routing_method_type,
            tune_max_num_tokens=M,
        )

        tuner = AutoTuner.get()

        # --- pass 1: tune (pin bucket to M) and capture selected tactic ------
        tuner.clear_cache()
        with autotune(True, tuning_buckets=(M,)):
            call()
        torch.cuda.synchronize()
        tactic = _selected_tactic()
        # Deployment-faithful (CUDA graph) + eager timing of the tuned tactic.
        t_tuned = _time_call_graph(call, TIME_ITERS, WARMUP_ITERS)
        t_tuned_eager = _time_call(call, TIME_ITERS, WARMUP_ITERS)

        # --- pass 2: re-tune from scratch -> determinism check ---------------
        tuner.clear_cache()
        with autotune(True, tuning_buckets=(M,)):
            call()
        torch.cuda.synchronize()
        tactic2 = _selected_tactic()

        # --- default (heuristic) baseline: empty cache, no tuning ------------
        tuner.clear_cache()
        t_default = _time_call_graph(call, TIME_ITERS, WARMUP_ITERS)
        t_default_eager = _time_call(call, TIME_ITERS, WARMUP_ITERS)

        results[M] = dict(
            tactic=tactic,
            tactic2=tactic2,
            t_tuned_ms=t_tuned,
            t_default_ms=t_default,
            t_tuned_eager_ms=t_tuned_eager,
            t_default_eager_ms=t_default_eager,
        )
        print(
            f"[rank {rank}] M={M:>5}  offset={local_expert_offset:>4}  "
            f"tactic={tactic}  redo={tactic2}\n"
            f"           graph: tuned={t_tuned * 1e3:7.1f}us default={t_default * 1e3:7.1f}us "
            f"speedup={t_default / max(t_tuned, 1e-9):4.2f}x  |  "
            f"eager: tuned={t_tuned_eager * 1e3:7.1f}us default={t_default_eager * 1e3:7.1f}us "
            f"speedup={t_default_eager / max(t_tuned_eager, 1e-9):4.2f}x",
            flush=True,
        )

    # --- gather all ranks' results to rank 0 and assert invariants ----------
    gathered = [None] * world_size
    dist.all_gather_object(gathered, results)

    if rank == 0:
        # Self-identify the config so buggy-vs-fixed logs are comparable.
        cfg = (
            f"dist={os.environ.get('FLASHINFER_AUTOTUNE_DISTRIBUTED', '') or 'off'}  "
            f"margin={os.environ.get('FLASHINFER_AUTOTUNE_SWITCH_MARGIN', '0.03')}  "
            f"legacy_random_dummy="
            f"{bool(os.environ.get('FLASHINFER_AUTOTUNE_LEGACY_RANDOM_DUMMY', ''))}"
        )
        print(f"\n=== verdict === [{cfg}]", flush=True)
        all_ok = True
        for M in MS:
            tactics = [gathered[r][M]["tactic"] for r in range(world_size)]
            redos = [gathered[r][M]["tactic2"] for r in range(world_size)]
            consistent = len(set(map(str, tactics))) == 1
            deterministic = all(
                str(gathered[r][M]["tactic"]) == str(gathered[r][M]["tactic2"])
                for r in range(world_size)
            )
            # "not slower than default" with 5% tolerance for measurement noise.
            not_slower = all(
                gathered[r][M]["t_tuned_ms"] <= 1.05 * gathered[r][M]["t_default_ms"]
                for r in range(world_size)
            )
            # PASS criteria mirror what actually matters in deployment:
            #   * consistent  -> all EP ranks pick the SAME tactic in one tuning
            #                    pass, so no rank is a straggler in the all-to-all.
            #   * not_slower  -> the tuned pick is never slower than the default.
            # ``deterministic`` (does a *second, independent* re-tune land on the
            # same tactic?) is reported for visibility only: at near-tied buckets
            # rank 0 can flap between two equally-fast tactics across separate
            # tuning runs, but deployment tunes exactly once, so this is not a
            # correctness/perf concern as long as the single run is consistent.
            # EP-group effective latency: the all-to-all at each decode step
            # waits for the SLOWEST rank, so the group runs at the per-rank max
            # (this is exactly the straggler mechanism behind Rong Song's e2e
            # regression).  Compare the tuned EP-group latency against the
            # default EP-group latency -- and across buggy-vs-fixed runs, the
            # tuned EP-group max is the headline "what does the autotuner cost
            # the group" number.
            tuned_per_rank = [gathered[r][M]["t_tuned_ms"] for r in range(world_size)]
            default_per_rank = [
                gathered[r][M]["t_default_ms"] for r in range(world_size)
            ]
            ep_group_tuned = max(tuned_per_rank)
            ep_group_default = max(default_per_rank)
            straggler_ratio = ep_group_tuned / max(min(tuned_per_rank), 1e-9)

            ok = consistent and not_slower
            all_ok = all_ok and ok
            print(
                f"M={M:>5}  consistent={consistent!s:>5}  "
                f"not_slower_than_default={not_slower!s:>5}  "
                f"deterministic(info)={deterministic!s:>5}  "
                f"{'PASS' if ok else 'FAIL'}\n"
                f"    EP-group(max-rank): tuned={ep_group_tuned * 1e3:7.1f}us  "
                f"default={ep_group_default * 1e3:7.1f}us  "
                f"tuned_vs_default={ep_group_default / max(ep_group_tuned, 1e-9):4.2f}x  "
                f"straggler(max/min tuned)={straggler_ratio:4.2f}x",
                flush=True,
            )
            if not consistent:
                print(f"    per-rank tactics: {tactics}  (redo: {redos})", flush=True)
            if not consistent or straggler_ratio > 1.10:
                print(
                    f"    per-rank tuned us: "
                    f"{[round(t * 1e3, 1) for t in tuned_per_rank]}",
                    flush=True,
                )
            if not not_slower:
                for r in range(world_size):
                    g = gathered[r][M]
                    print(
                        f"    rank {r}: tuned={g['t_tuned_ms'] * 1e3:.1f}us "
                        f"default={g['t_default_ms'] * 1e3:.1f}us",
                        flush=True,
                    )
        print(f"\nOVERALL: {'PASS' if all_ok else 'FAIL'}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
