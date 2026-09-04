"""Authoritative regression artifact: zero-token idle-batch livelock repro
for in_kernel_fc2_reduce (MXFP8).

SGLang's mechanism for keeping DP ranks in lockstep (prepare_mlp_sync_batch_raw
/ dp_attn.py) is: every idle DP rank still calls forward() every step, but
with a synthetic ZERO-TOKEN batch. The bug: mxfp8_mega_moe()
(kernel_src/cutedsl_megamoe/shim/mxfp8.py) had

    if n == 0 and symm_buffer._frontend.config.in_kernel_fc2_reduce:
        return symm_buffer.output_activation[:0] if y is None else None

-- i.e. a rank with num_tokens=0 locally SKIPPED issuing the actual CUDA
kernel launch entirely for that call. Sm100MegaMoEMxfp8Kernel is a persistent
megakernel whose CTA grid is sized from hardware occupancy, never from
num_tokens, so even a genuinely zero-token launch would still run every CTA
and the warp-specialized dispatch/token-back/tail-cleanup logic that keeps a
rank's cross-rank REDG atomic-add session in lockstep with its EP peers. The
shortcut instead silently skipped that round's participation for the local
rank, desynchronizing session bookkeeping from its peers.

This mirrors SGLang's actual idle-batch mechanism: rank 0 always sends a
real (nonzero) batch; ranks 1-3 independently coin-flip zero-token vs real
every iteration (no cross-rank coordination -- exactly like real DP ranks
with independently-arriving requests), all under in_kernel_fc2_reduce=True.

This is proven to reliably livelock the unpatched kernel as a plain
torchrun-launched script (confirmed multiple times, including directly
against this repo's kernel_src_restructure-branch checkout) and to complete
cleanly after the fix. It could NOT be made to reliably reproduce the hang
when run under `torchrun -m pytest` (see the twin pytest regression tests in
tests/moe_ep/test_moe_ep_{mxfp8,nvfp4}_cutedsl_mega_multirank.py for
details) -- this script remains the authoritative artifact for verifying a
fix or bisecting a regression.

Run (single node, 4 GPUs), from a flashinfer checkout root:
    torchrun --nproc_per_node=4 --standalone tests/repro_ikr_zero_token_idle.py

Env knobs:
    REAL_TOKENS    token count for a "real" batch (default 4)
    NUM_ITERS      iterations (default 60)
    MAX_TOKENS     symmetric buffer size per rank (default 16384)
    IKR            1 (default) or 0
    WATCHDOG_S     seconds of no-progress before dumping stacks (default 20)
"""

from __future__ import annotations

import faulthandler
import os
import random
import sys
import threading
import time

import torch
import torch.distributed as dist


def log(rank: int, msg: str) -> None:
    print(f"[rank {rank} t={time.time():.1f}] {msg}", flush=True)


def start_watchdog(rank: int, progress: dict, watchdog_s: float) -> None:
    def _watch():
        last_seen = -1
        stuck_since = None
        while True:
            time.sleep(2)
            cur = progress["iter"]
            if cur == last_seen:
                if stuck_since is None:
                    stuck_since = time.time()
                elif time.time() - stuck_since > watchdog_s:
                    log(
                        rank,
                        f"WATCHDOG: no progress for {watchdog_s}s, stuck at "
                        f"iter={cur} phase={progress['phase']!r} "
                        f"n_tokens={progress.get('n_tokens')}. Dumping stacks:",
                    )
                    faulthandler.dump_traceback(file=sys.stdout)
                    stuck_since = time.time()
            else:
                stuck_since = None
                last_seen = cur

    t = threading.Thread(target=_watch, daemon=True)
    t.start()


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)

    real_tokens = int(os.environ.get("REAL_TOKENS", "4"))
    num_iters = int(os.environ.get("NUM_ITERS", "60"))
    max_tokens = int(os.environ.get("MAX_TOKENS", "16384"))
    ikr = os.environ.get("IKR", "1") not in ("0", "false", "False")
    watchdog_s = float(os.environ.get("WATCHDOG_S", "20"))

    hidden = 2048
    intermediate = 768
    num_experts = 128
    top_k = 8
    num_local_experts = num_experts // world_size

    log(
        rank,
        f"world_size={world_size} real_tokens={real_tokens} ikr={ikr} num_iters={num_iters}",
    )

    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpMegaLayer,
        MoEEpTensors,
        MoEWeightPack,
        Mxfp8CutedslMegaMoeConfig,
    )

    g = torch.Generator(device="cuda").manual_seed(13 + rank)
    w13 = torch.randn(
        num_local_experts,
        2 * intermediate,
        hidden,
        dtype=torch.bfloat16,
        device="cuda",
        generator=g,
    )
    w2 = torch.randn(
        num_local_experts,
        hidden,
        intermediate,
        dtype=torch.bfloat16,
        device="cuda",
        generator=g,
    )
    megakernel_config = Mxfp8CutedslMegaMoeConfig(
        intermediate_size=intermediate,
        top_k=top_k,
        kind="mxfp8_e4m3",
        in_kernel_fc2_reduce=ikr,
    )

    log(rank, "constructing MoEEpMegaLayer...")
    mega = MoEEpMegaLayer(
        bootstrap=BootstrapConfig(world_size=world_size, rank=rank),
        fleet_params=FleetParams(
            num_experts=num_experts,
            max_tokens_per_rank=max_tokens,
            token_hidden_size=hidden,
        ),
        weights=MoEWeightPack(w13=w13, w2=w2),
        backend=MegaConfig(megakernel=megakernel_config, preprocess_weights=True),
    )
    log(rank, "MoEEpMegaLayer constructed.")
    if dist.is_initialized():
        dist.barrier()

    def make_inputs(n: int, salt: int):
        gi = torch.Generator(device="cuda").manual_seed(9000 + salt * 17 + rank)
        hidden_states = torch.randn(
            n, hidden, dtype=torch.bfloat16, device="cuda", generator=gi
        )
        scores = torch.randn(
            n, num_experts, dtype=torch.float32, device="cuda", generator=gi
        )
        topk_weights, topk_ids = torch.topk(
            scores, top_k, dim=-1, largest=True, sorted=False
        )
        return MoEEpTensors(
            hidden_states=hidden_states,
            topk_ids=topk_ids.to(torch.int64),
            topk_weights=topk_weights.to(torch.float32),
        )

    # Matched-count collective warmup (real tokens on every rank, once).
    mega.forward(make_inputs(real_tokens, salt=1))
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    log(
        rank,
        "starting loop: rank0 always real; ranks>=1 randomly zero-token "
        "or real each iter, independently, no barrier",
    )

    rnd = random.Random(4242 + rank)
    progress = {"iter": 0, "phase": "start", "n_tokens": None}
    start_watchdog(rank, progress, watchdog_s)

    for it in range(num_iters):
        progress["iter"] = it
        if rank == 0:
            n = real_tokens
        else:
            n = 0 if rnd.random() < 0.5 else real_tokens
        progress["n_tokens"] = n
        progress["phase"] = "build_inputs"
        t = make_inputs(n, salt=1000 + it)
        progress["phase"] = "forward"
        y = mega.forward(t)
        progress["phase"] = "sync"
        torch.cuda.synchronize()
        progress["phase"] = "done"
        if it % 10 == 0 or it == num_iters - 1:
            log(
                rank,
                f"iter {it}/{num_iters} ok, n_tokens={n}, y.shape={tuple(y.shape)}",
            )

    log(rank, "ALL ITERATIONS COMPLETE")
    if dist.is_initialized():
        dist.barrier()
    log(rank, "final barrier passed, exiting cleanly")


if __name__ == "__main__":
    main()
