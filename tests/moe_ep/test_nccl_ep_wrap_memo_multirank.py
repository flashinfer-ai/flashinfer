"""The nccl_ep wrap memo must not pin tensors it can never hit on, 4+ GPUs.

Launched via torchrun::

    torchrun --nproc_per_node=4 -m pytest \
        tests/moe_ep/test_nccl_ep_wrap_memo_multirank.py -v -m "nvep and gpu_4"

``NcclEpHandle._wrap`` memoizes ``nccl.ep.Tensor`` descriptors keyed by
``(data_ptr, dtype, shape)``, and the wrapper keeps its torch tensor alive. This
allows descriptor reuse for stable workspace buffers.

It is not fine for combine's ``out``, which eager ``forward()`` allocates fresh
every call. Caching on first sight pinned it, which stopped the allocator
recycling that address, so the next forward got a new one, the memo missed 100%
of the time, retaining one output buffer per forward until cache eviction.
This is bounded retention (roughly 512 MiB at the memo's limits), not an
unbounded leak. Measured before the fix at 128 tokens x 4096 hidden:
50/50 distinct ``out`` pointers, the memo
growing by exactly one entry per forward, and torch's allocated bytes climbing
27 -> 227 MiB across 200 forwards. The knock-on cost was an ``empty_like`` going
from 1.1us to ~103us once its predecessors were pinned, which is what made the
benchmark's amortized eager number exceed its own per-call number.

The shape here is deliberate. 128 tokens x 4096 hidden x 2 B = 1 MiB, under
``_WRAP_MEMO_MAX_BYTES`` (2 MiB), so ``out`` goes through the memo rather than
the large-tensor bypass -- at 512 tokens it would bypass and the retention would not
reproduce at all.
"""

from __future__ import annotations

import os
from datetime import timedelta

import pytest

pytestmark = pytest.mark.usefixtures("require_split_backend")

_PG_TIMEOUT = timedelta(minutes=60)

TOKENS_PER_RANK = 128
NUM_EXPERTS = 8
HIDDEN = 4096  # 128 x 4096 x 2 B = 1 MiB: memoized, not bypassed
TOPK = 4
FORWARDS = 40


def _init_dist():
    import torch
    import torch.distributed as dist

    if not dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            device_id=torch.device(f"cuda:{local_rank}"),
            timeout=_PG_TIMEOUT,
        )
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    return rank, world


@pytest.mark.nvep
@pytest.mark.gpu_4
@pytest.mark.parametrize("algo_name", ["low_latency", "high_throughput"])
def test_eager_forwards_do_not_grow_the_wrap_memo(algo_name):
    """Repeated eager forwards must not retain outputs in the fleet memo.

    Both assertions target the same defect from different sides, because either
    alone is weak: the memo could stay small while another owner retained
    outputs, and memory could remain flat if a different allocation was freed.
    Together they check the identity path's output lifetime. The identity
    kernel deliberately reuses dispatch buffers; transient packed payloads
    and expert outputs from other kernels are outside this test's scope.

    Fails against the pre-fix code, where the memo grew by one entry and torch's
    allocated bytes by one output buffer on every single forward.
    """
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import (
        BootstrapConfig,
        EpAlgorithm,
        FleetParams,
        MoEEpSplitLayer,
        MoEEpTensors,
        dummy_moe_weights,
    )

    rank, world_size = _init_dist()
    assert world_size >= 4, f"needs >=4 ranks, got {world_size}"
    algorithm = {
        "low_latency": EpAlgorithm.LOW_LATENCY,
        "high_throughput": EpAlgorithm.HIGH_THROUGHPUT,
    }[algo_name]

    layer = MoEEpSplitLayer(
        bootstrap=BootstrapConfig(
            world_size=world_size,
            rank=rank,
            stream=torch.cuda.current_stream().cuda_stream,
        ),
        fleet_params=FleetParams(
            num_experts=NUM_EXPERTS,
            max_tokens_per_rank=TOKENS_PER_RANK,
            token_hidden_size=HIDDEN,
            dtype_bytes=2,
            algorithm=algorithm,
        ),
        weights=dummy_moe_weights(
            num_local_experts=NUM_EXPERTS // world_size, hidden=HIDDEN
        ),
        backend="nccl_ep",
    )

    g = torch.Generator(device="cuda").manual_seed(11 + rank)
    t = MoEEpTensors(
        hidden_states=torch.randn(
            TOKENS_PER_RANK, HIDDEN, dtype=torch.bfloat16, device="cuda", generator=g
        ),
        topk_ids=torch.randint(
            0,
            NUM_EXPERTS,
            (TOKENS_PER_RANK, TOPK),
            device="cuda",
            dtype=torch.int64,
            generator=g,
        ),
        topk_weights=torch.softmax(
            torch.randn(TOKENS_PER_RANK, TOPK, device="cuda", generator=g), dim=-1
        ),
    )

    # All collective work first; assertions only after teardown. A bare assert
    # mid-collective strands the other ranks at the next barrier.
    for _ in range(5):  # warmup: let steady-state addresses settle
        layer.forward(t)
    torch.cuda.synchronize()
    dist.barrier()

    fleet = layer._ensure_fleet()
    memo_before = len(fleet._hot_cache)
    mem_before = torch.cuda.memory_allocated()

    for _ in range(FORWARDS):
        layer.forward(t)
    torch.cuda.synchronize()
    dist.barrier()

    memo_after = len(fleet._hot_cache)
    mem_after = torch.cuda.memory_allocated()

    layer.destroy()
    dist.barrier()

    # --- assertions: no collectives beyond this point ---------------------
    memo_growth = memo_after - memo_before
    assert memo_growth <= 2, (
        f"wrap memo grew by {memo_growth} entries over {FORWARDS} eager "
        f"forwards ({memo_before} -> {memo_after}). A per-call tensor is being "
        "cached, which pins it and prevents the allocator reuse the memo "
        "depends on."
    )
    # One output buffer is 1 MiB here; pre-fix this grew by FORWARDS MiB.
    retained_mib = (mem_after - mem_before) / (1 << 20)
    assert retained_mib < 4.0, (
        f"device memory grew {retained_mib:.1f} MiB over {FORWARDS} eager "
        f"forwards (~{retained_mib / FORWARDS:.2f} MiB per forward). Eager "
        "forward allocates one output buffer per call; if it is still "
        "reachable afterwards the memo is pinning it."
    )
