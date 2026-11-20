"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0
"""

"""The owner/helper split of the unbounded direct N32 body.

A call whose tasks leave a spare SM per helper (at most two helpers per task)
and whose sequences reach ``OWNER_HELPER_MIN_TOKENS`` launches the owner/helper
program cooperatively with ``tasks * (1 + helpers)`` CTAs; every other call
keeps the one-CTA program.  The split moves prepare work between CTAs and
leaves every value the recurrence computes unchanged, so out, the final state
pool and the FP32 checkpoint rows are the bits of the one-CTA program, launch
after launch (the program leaves its ring words zero for the next launch).

Where the per-task split has one helper per task and the SMs it leaves idle
give each head at least 13/12 as many helpers (and every sequence has the same
chunk count), the call takes the pooled program instead: the helpers of a head
form one pool of ``pool`` CTAs serving the head's chunks of every sequence in
turn, launched with ``tasks + heads * pool`` CTAs. It moves the same prepare
work between other CTAs, so its results are the same bits as well; the bitwise
checks also force the pool where the route keeps the per-task split (two
helpers per task).
"""

import pytest
import torch

import flashinfer.cake_kda_tf32_runtime as runtime
from flashinfer import KDAPrefillPlanCache, prepare_bf16_kda_prefill

HEAD_DIM = 128
CHECKPOINT_EVERY = 64


@pytest.mark.parametrize(
    ("sm_count", "tasks", "max_seq_len", "expected"),
    [
        (148, 48, 4096, 2),  # 4 x 4096, H12: 144 CTAs
        (148, 64, 4096, 1),  # 4 x 4096, H16: one spare SM per task
        (152, 64, 4096, 1),
        (148, 24, 8192, 2),  # helpers stop at two
        (148, 75, 4096, 0),  # no spare SM per task
        (148, 48, 1023, 0),  # sequences too short
        (148, 192, 1088, 0),
    ],
)
@pytest.mark.parametrize("gpu_arch", ["sm_100a", "sm_103a"])
def test_owner_helper_count(gpu_arch, sm_count, tasks, max_seq_len, expected):
    assert (
        runtime._owner_helper_count(
            gpu_arch=gpu_arch,
            sm_count=sm_count,
            total_tasks=tasks,
            max_seq_len=max_seq_len,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("sm_count", "seqs", "heads", "helpers", "uniform", "expected"),
    [
        (148, 4, 16, 1, True, 5),  # 4 x 4096, H16: 64 + 16 x 5 = 144 CTAs
        (152, 4, 16, 1, True, 5),
        (148, 5, 12, 1, True, 7),  # 60 + 12 x 7 = 144 CTAs
        (148, 71, 1, 1, True, 77),  # 12 x 77 >= 13 x 71: 148 CTAs
        (148, 72, 1, 1, True, 0),  # 76 helpers for 72 sequences: below 13/12
        (148, 4, 12, 2, True, 0),  # 4 x 4096, H12: two helpers per task
        (148, 2, 12, 2, True, 0),  # two helpers per task (a pool of 5 fits)
        (148, 4, 16, 1, False, 0),  # chunk counts differ
        (148, 1, 64, 1, True, 0),  # one sequence: the pool stays at the helper
        (148, 4, 16, 0, True, 0),  # no owner/helper split
    ],
)
def test_owner_helper_pool_count(sm_count, seqs, heads, helpers, uniform, expected):
    pool = runtime._owner_helper_pool_count(
        sm_count=sm_count,
        num_seqs=seqs,
        num_heads=heads,
        helpers=helpers,
        uniform_chunks=uniform,
    )
    assert pool == expected
    if pool:
        tasks = seqs * heads
        grid = tasks + heads * pool
        # Every CTA resident (one per SM) and the program reads the same
        # helpers per task from its grid as the per-task split.
        assert grid <= sm_count and grid // tasks - 1 == helpers
        assert 12 * pool >= 13 * helpers * seqs


def test_owner_helper_count_other_arch():
    assert (
        runtime._owner_helper_count(
            gpu_arch="sm_120a", sm_count=148, total_tasks=48, max_seq_len=4096
        )
        == 0
    )


requires_sm100 = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="KDA export requires SM100a or SM103a",
)


def _inputs(lengths, heads):
    torch.manual_seed(0)
    tokens = sum(lengths)
    shape = (1, tokens, heads, HEAD_DIM)
    q, k, v, g = [
        torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(4)
    ]
    # Strided logit beta, as the serving adapter passes it.
    beta = torch.randn(1, tokens, 40, device="cuda", dtype=torch.bfloat16)[
        :, :, 8 : 8 + heads
    ]
    pool = len(lengths) + 5
    state = (
        torch.randn(pool, heads, HEAD_DIM, HEAD_DIM, device="cuda", dtype=torch.float32)
        * 0.01
    )
    indices = torch.arange(
        pool - 1, pool - len(lengths) - 1, -1, device="cuda", dtype=torch.int32
    )
    offsets, starts = [0], [0]
    for n in lengths:
        offsets.append(offsets[-1] + n)
        starts.append(starts[-1] + (n + CHECKPOINT_EVERY - 1) // CHECKPOINT_EVERY)
    return dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=torch.randn(heads, device="cuda") * 0.2,
        dt_bias=torch.randn(heads, HEAD_DIM, device="cuda") * 0.1,
        state=state,
        indices=indices,
        cu_seqlens=torch.tensor(offsets, device="cuda", dtype=torch.int64),
        starts=torch.tensor(starts, device="cuda", dtype=torch.int64),
        rows=starts[-1],
    )


def _poison_free_cuda_memory(reserve=2 << 30, small_reserve=16 << 20):
    """Fill every free block of the CUDA caching allocator (this device and
    stream, outside graph pools) with 0xFF bytes, NaN in BF16 and FP32, after
    growing the free pools to ``reserve`` / ``small_reserve`` bytes.  A launch
    that skips a producer then reads poison, not what another arm or an
    earlier call left in memory the allocator hands out again (a copy of
    test_cake_kda_prefill_generated.py's)."""
    torch.cuda.synchronize()
    device = torch.cuda.current_device()
    stream = torch.cuda.current_stream().cuda_stream
    sizes, free = [], {"large": 0, "small": 0}
    for segment in torch.cuda.memory_snapshot():
        if (
            segment["device"] != device
            or segment["stream"] != stream
            or tuple(segment.get("segment_pool_id", (0, 0))) != (0, 0)
        ):
            continue
        kind = segment["segment_type"]
        for block in segment["blocks"]:
            if block["state"] != "inactive":
                continue
            size = block["size"]
            free[kind] += size
            # The small pool serves requests up to 1 MiB.
            step = 1 << 20 if kind == "small" else size
            sizes += [min(step, size - offset) for offset in range(0, size, step)]
    sizes.sort(reverse=True)
    grow = max(0, reserve - free["large"])
    sizes += [256 << 20] * -(-grow // (256 << 20))
    sizes += [1 << 20] * -(-max(0, small_reserve - free["small"]) // (1 << 20))
    held = [torch.empty(n, dtype=torch.uint8, device="cuda") for n in sizes]
    for tensor in held:
        tensor.fill_(0xFF)
    torch.cuda.synchronize()


def _run(d, lengths, heads, launches, *, checkpoints=True):
    """Launch one prepared call ``launches`` times from the same initial pool
    (``checkpoints=False``: without checkpoint rows).  Free CUDA memory is
    poisoned before the call is prepared and before every launch, the output
    and checkpoint rows before every launch."""
    state0 = d["state"].clone()
    out = torch.empty_like(d["q"])
    rows = (
        torch.empty(
            (d["rows"], heads, HEAD_DIM, HEAD_DIM), device="cuda", dtype=torch.float32
        )
        if checkpoints
        else None
    )
    pool = d["state"].clone()
    _poison_free_cuda_memory()
    call = prepare_bf16_kda_prefill(
        d["q"],
        d["k"],
        d["v"],
        d["g"],
        d["beta"],
        A_log=d["A_log"],
        dt_bias=d["dt_bias"],
        out=out,
        initial_state=pool,
        final_state=pool,
        lower_bound=None,
        cu_seqlens=d["cu_seqlens"],
        sequence_lengths=tuple(lengths),
        state_indices=d["indices"],
        state_checkpoints=rows,
        checkpoint_cu_starts=d["starts"] if checkpoints else None,
        checkpoint_every_n_tokens=CHECKPOINT_EVERY if checkpoints else 0,
        beta_is_logit=True,
        plan_cache=KDAPrefillPlanCache(4),
    )
    results = []
    for _ in range(launches):
        _poison_free_cuda_memory()
        pool.copy_(state0)
        out.fill_(float("nan"))
        if checkpoints:
            rows.fill_(float("nan"))
        call.launch()
        torch.cuda.synchronize()
        results.append(
            (out.clone(), pool.clone(), rows.clone() if checkpoints else out[:0])
        )
    return str(call.schedule), results


def _bits(tensor):
    return tensor.contiguous().view(torch.uint8)


def _supported_pool_count(*, sm_count, num_seqs, num_heads, helpers, uniform_chunks):
    """The largest pool the program supports, also where the route keeps the split."""
    if helpers < 1 or not uniform_chunks:
        return 0
    total_tasks = num_seqs * num_heads
    pool = min((sm_count - total_tasks) // num_heads, (helpers + 1) * num_seqs - 1)
    return pool if pool > helpers * num_seqs else 0


def _expected_suffix(lengths, heads):
    """The schedule suffix of the owner/helper launch the call takes here."""
    from flashinfer.jit import cake_kda_tf32 as registry

    tasks = len(lengths) * heads
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    helpers = min(runtime.OWNER_HELPER_MAX_HELPERS, sm_count // tasks - 1)
    if helpers < 1:
        pytest.skip(f"{sm_count} SMs leave no helper for {tasks} tasks")
    pool = runtime._owner_helper_pool_count(
        sm_count=sm_count,
        num_seqs=len(lengths),
        num_heads=heads,
        helpers=helpers,
        uniform_chunks=len({(n + 31) // 32 for n in lengths}) == 1,
    )
    if pool and registry._RENDERED_OWNER_HELPER_POOL_ROUTES.get(registry.device_arch()):
        return f"_owner{helpers}helper_pool{pool}"
    return f"_owner{helpers}helper"


@requires_sm100
@pytest.mark.parametrize("heads", [12, 16])
def test_owner_helper_is_bitwise_the_one_cta_program(heads, monkeypatch):
    lengths = (4096,) * 4
    suffix = _expected_suffix(lengths, heads)
    d = _inputs(lengths, heads)
    split_schedule, split = _run(d, lengths, heads, launches=3)
    assert split_schedule.endswith(suffix), (split_schedule, suffix)
    monkeypatch.setattr(runtime, "OWNER_HELPER_MAX_HELPERS", 0)
    one_schedule, one = _run(d, lengths, heads, launches=1)
    assert "owner" not in one_schedule, one_schedule
    assert split_schedule.startswith(one_schedule), (split_schedule, one_schedule)
    for launch in split:
        for got, want in zip(launch, one[0], strict=True):
            assert torch.equal(_bits(got), _bits(want))


@requires_sm100
def test_short_sequences_keep_the_one_cta_program():
    lengths = (512,) * 4
    d = _inputs(lengths, 12)
    schedule, _ = _run(d, lengths, 12, launches=1)
    assert "owner" not in schedule, schedule


@requires_sm100
@pytest.mark.parametrize(
    ("lengths", "heads", "forced"),
    [
        ((4096,) * 4, 16, False),
        ((2048,) * 5, 12, False),
        ((4096, 4090, 4096, 4065), 16, False),  # one chunk count, other lengths
        ((4096, 4096, 4096, 4000), 16, False),  # chunk counts differ: the split
        ((1024,) * 2, 12, True),  # two helpers per task: the pool forced
        ((1024,) * 3, 12, True),
    ],
)
def test_owner_helper_pool_is_bitwise_the_per_task_split(
    lengths, heads, forced, monkeypatch
):
    if forced:
        monkeypatch.setattr(runtime, "_owner_helper_pool_count", _supported_pool_count)
    suffix = _expected_suffix(lengths, heads)
    if forced:
        assert "_pool" in suffix, suffix
    d = _inputs(lengths, heads)
    with monkeypatch.context() as m:
        m.setattr(runtime, "_owner_helper_pool_count", lambda **_: 0)
        split_schedule, split = _run(d, lengths, heads, launches=1)
    # _expected_suffix skipped a part that leaves no helper: every call here
    # takes the per-task split, a call that routes elsewhere fails.
    assert split_schedule.endswith(suffix.split("_pool")[0]), (split_schedule, suffix)
    pool_schedule, pooled = _run(d, lengths, heads, launches=3)
    assert pool_schedule.endswith(suffix), (pool_schedule, suffix)
    assert pool_schedule.startswith(split_schedule), (pool_schedule, split_schedule)
    for launch in pooled:
        for got, want in zip(launch, split[0], strict=True):
            assert torch.equal(_bits(got), _bits(want))


@requires_sm100
def test_owner_helper_pool_without_checkpoint_rows_is_bitwise_the_per_task_split(
    monkeypatch,
):
    """The pooled program on a call without checkpoint rows (the FP32-carrier
    module serves both): the same out and final state pool as the per-task
    split, launch after launch."""
    lengths, heads = (4096,) * 4, 16
    suffix = _expected_suffix(lengths, heads)
    assert "_pool" in suffix, suffix
    d = _inputs(lengths, heads)
    with monkeypatch.context() as m:
        m.setattr(runtime, "_owner_helper_pool_count", lambda **_: 0)
        split_schedule, split = _run(d, lengths, heads, launches=1, checkpoints=False)
    assert "_owner" in split_schedule and "_pool" not in split_schedule, split_schedule
    pool_schedule, pooled = _run(d, lengths, heads, launches=3, checkpoints=False)
    assert pool_schedule.endswith(suffix), (pool_schedule, suffix)
    assert pool_schedule.startswith(split_schedule), (pool_schedule, split_schedule)
    for launch in pooled:
        for got, want in zip(launch, split[0], strict=True):
            assert torch.equal(_bits(got), _bits(want))
