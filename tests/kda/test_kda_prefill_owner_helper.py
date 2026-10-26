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


def _run(d, lengths, heads, launches):
    """Launch one prepared call ``launches`` times from the same initial pool."""
    state0 = d["state"].clone()
    out = torch.empty_like(d["q"])
    rows = torch.empty(
        (d["rows"], heads, HEAD_DIM, HEAD_DIM), device="cuda", dtype=torch.float32
    )
    pool = d["state"].clone()
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
        checkpoint_cu_starts=d["starts"],
        checkpoint_every_n_tokens=CHECKPOINT_EVERY,
        beta_is_logit=True,
        plan_cache=KDAPrefillPlanCache(4),
    )
    results = []
    for _ in range(launches):
        pool.copy_(state0)
        call.launch()
        torch.cuda.synchronize()
        results.append((out.clone(), pool.clone(), rows.clone()))
    return str(call.schedule), results


def _bits(tensor):
    return tensor.contiguous().view(torch.uint8)


@requires_sm100
@pytest.mark.parametrize("heads", [12, 16])
def test_owner_helper_is_bitwise_the_one_cta_program(heads, monkeypatch):
    lengths = (4096,) * 4
    tasks = len(lengths) * heads
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    helpers = min(runtime.OWNER_HELPER_MAX_HELPERS, sm_count // tasks - 1)
    if helpers < 1:
        pytest.skip(f"{sm_count} SMs leave no helper for {tasks} tasks")
    d = _inputs(lengths, heads)
    split_schedule, split = _run(d, lengths, heads, launches=3)
    assert split_schedule.endswith(f"_owner{helpers}helper"), split_schedule
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
