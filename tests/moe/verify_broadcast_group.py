"""Correctness checks for the process-group-scoped broadcast.

Three sub-checks, run together under a 4-GPU torchrun:

  A. WORLD (group=None) regression -- distinct per-rank tactics all converge to
     global rank 0 (the default-WORLD path still works after adding the
     process_group arg).
  B. Sub-group scoping -- two independent EP groups [0,1] and [2,3]; each rank
     ends on its *group-local* rank-0's tactic and the groups do NOT bleed into
     each other (a WORLD broadcast would have made all four identical). Uses
     synthetic injected tactics so the scoping is unambiguous.
  C. Real-op API path -- run the actual trtllm_fp4_block_scale_moe under
     ``autotune(distributed_process_group=<EP group>)`` on two 2-rank EP groups
     and confirm the public API path aligns real tactics within each group
     without error. (Skipped off SM100.)

Single-process no-op (most users run 1 process) is checked separately, see the
command in the PR / chat -- it does not require torchrun.

Run on a 4-GPU node:
    cd /lustre/fsw/coreai_dlfw_dev/albecheng/flashinfer-autotuner-align
    FLASHINFER_AUTOTUNE_DISTRIBUTED=broadcast \
    python -m torch.distributed.run --nproc_per_node=4 \
        tests/moe/verify_broadcast_group.py
"""

import os
import sys

import torch
import torch.distributed as dist

# Make ``tests.moe.*`` importable when launched as a script from repo root.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from flashinfer.autotuner import AutoTuner, autotune

_KEY = ("flashinfer::dummy_op", "MoERunner", (1,), ())
_FP4_OP = "flashinfer::trtllm_fp4_block_scale_moe"


def _inject_distinct(rank):
    tuner = AutoTuner.get()
    tuner.profiling_cache.clear()
    tuner.profiling_cache[_KEY] = (0, [rank, rank * 100], None)
    return tuner


def test_world_synthetic(rank, world):
    """A. group=None (WORLD): all ranks converge to global rank 0."""
    tuner = _inject_distinct(rank)
    tuner._maybe_sync_distributed_cache(process_group=None)
    got = tuner.profiling_cache[_KEY][1]
    gathered = [None] * world
    dist.all_gather_object(gathered, (rank, got))
    dist.barrier()
    if rank == 0:
        d = dict(gathered)
        ok = all(d[r] == [0, 0] for r in range(world))
        print(f"[A WORLD ] final={d}  {'PASS' if ok else 'FAIL'}", flush=True)
        return ok
    return None


def test_subgroup_synthetic(rank, world):
    """B. two EP groups [0,1] and [2,3]: scoped, no cross-group bleed."""
    g01 = dist.new_group([0, 1])
    g23 = dist.new_group([2, 3])
    my_group = g01 if rank in (0, 1) else g23
    expected_src = 0 if rank in (0, 1) else 2

    src_api = dist.get_global_rank(my_group, 0)
    assert src_api == expected_src, (
        f"[rank {rank}] get_global_rank(group,0)={src_api} != {expected_src}"
    )

    tuner = _inject_distinct(rank)
    tuner._maybe_sync_distributed_cache(process_group=my_group)
    got = tuner.profiling_cache[_KEY][1]

    gathered = [None] * world
    dist.all_gather_object(gathered, (rank, got))
    dist.barrier()
    if rank == 0:
        d = dict(gathered)
        want = {0: [0, 0], 1: [0, 0], 2: [2, 200], 3: [2, 200]}
        ok = all(d[r] == want[r] for r in range(world))
        print(
            f"[B scope ] final={d}  {'PASS' if ok else 'FAIL'} "
            f"(WORLD broadcast would make all [0,0])",
            flush=True,
        )
        return ok
    return None


def _selected_fp4_tactic(tuner):
    tacs = {
        tuple(v[1]) if isinstance(v[1], (list, tuple)) else v[1]
        for k, v in tuner.profiling_cache.items()
        if isinstance(k, tuple) and len(k) > 0 and k[0] == _FP4_OP
    }
    return next(iter(tacs)) if len(tacs) == 1 else None


def test_subgroup_realop(rank, world, device):
    """C. real MoE op via autotune(distributed_process_group=...) on 2-rank EP groups."""
    from flashinfer import ActivationType, RoutingMethodType
    from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe
    from flashinfer.fused_moe.utils import make_balanced_local_topk_ids
    from flashinfer.utils import device_support_pdl, get_compute_capability

    from tests.moe.test_trtllm_gen_moe_autotune_tactics import (
        _build_fp4_routed_moe_inputs,
    )

    g01 = dist.new_group([0, 1])
    g23 = dist.new_group([2, 3])
    my_group = g01 if rank in (0, 1) else g23

    if get_compute_capability(device)[0] != 10:
        if rank == 0:
            print("[C realop] SKIP (needs SM100)", flush=True)
        return None

    NUM_EXPERTS, EP = 128, 2
    num_local = NUM_EXPERTS // EP
    offset = (rank % 2) * num_local
    HIDDEN = INTER = 3072
    TOPK, M = 4, 128
    enable_pdl = device_support_pdl(device)
    rmt = RoutingMethodType.Renormalize

    torch.manual_seed(1000 + rank)
    inputs = _build_fp4_routed_moe_inputs(
        num_tokens=M,
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        top_k=TOPK,
        num_experts=num_local,
        quant_mode="MxFP4xMxFP8",
        routing_method_type=rmt,
        device=device,
    )
    topk = make_balanced_local_topk_ids(
        num_tokens=M,
        top_k=TOPK,
        num_local_experts=num_local,
        local_expert_offset=offset,
        device=device,
    )
    ew = torch.ones(M, TOPK, device=device, dtype=torch.bfloat16)
    packed = (topk.to(torch.int32) << 16) | ew.to(torch.bfloat16).view(torch.int16)

    def call():
        return trtllm_fp4_block_scale_routed_moe(
            topk_ids=packed,
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
            num_experts=NUM_EXPERTS,
            top_k=TOPK,
            n_group=None,
            topk_group=None,
            intermediate_size=INTER,
            local_expert_offset=offset,
            local_num_experts=num_local,
            routed_scaling_factor=None,
            routing_method_type=rmt.value,
            do_finalize=True,
            enable_pdl=enable_pdl,
            activation_type=ActivationType.Swiglu.value,
            tune_max_num_tokens=M,
        )[0]

    tuner = AutoTuner.get()
    tuner.clear_cache()
    # The public API path under test: pass the EP group to autotune().
    with autotune(True, tuning_buckets=(M,), distributed_process_group=my_group):
        call()
    torch.cuda.synchronize()
    tac = _selected_fp4_tactic(tuner)

    gathered = [None] * world
    dist.all_gather_object(gathered, (rank, str(tac)))
    dist.barrier()
    if rank == 0:
        d = dict(gathered)
        # within-group consistency: pair (0,1) agree and (2,3) agree
        ok = d[0] == d[1] and d[2] == d[3] and tac is not None
        print(
            f"[C realop] per-rank tactic={d}  "
            f"{'PASS' if ok else 'FAIL'} (within-group consistent via API)",
            flush=True,
        )
        return ok
    return None


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', rank))}")
    assert world == 4, f"this check expects 4 ranks, got {world}"

    results = []
    results.append(test_world_synthetic(rank, world))
    results.append(test_subgroup_synthetic(rank, world))
    results.append(test_subgroup_realop(rank, world, device))

    dist.barrier()
    if rank == 0:
        graded = [r for r in results if r is not None]
        print(
            f"\n=== verify_broadcast_group: "
            f"{'ALL PASS' if all(graded) else 'FAIL'} "
            f"({sum(graded)}/{len(graded)} graded checks) ===",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
