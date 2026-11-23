# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""Non-finite tokens of one packed sequence must not reach another sequence in Cake prefill.

The last chunk of a packed sequence that is not a whole chunk long covers tokens of the
next sequence.  A kernel that loads that chunk whole must clear those rows instead of
multiplying them by a zero beta or a zero key, because ``0 * NaN`` and ``0 * inf`` are
NaN.  Every case runs ``recurrent_kda(backend="cake")`` once with finite inputs and then
with every other sequence poisoned (V alone with NaN or inf, or every input with NaN);
the output rows and final state of the other sequences must keep the bytes of the
finite launch.  The cases reach the direct, persistent, small-BH and BT16 schedules,
and the TF32 BT16 chain of ``prepare_tf32_kda_prefill``.
"""

import importlib

import pytest
import torch

from flashinfer import prepare_tf32_kda_prefill
from flashinfer.jit import cake_kda_tf32
from flashinfer.jit.flash_kda import _GeneratedFlashKDASelectorNotFoundError
from flashinfer.kda import recurrent_kda

kda_prefill_api = importlib.import_module("flashinfer.kda_prefill")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="Cake KDA prefill requires SM100a or SM103a",
)

HEAD_DIM = 128
LOWER_BOUND = -5.0
_PERSISTENT = {
    "persistent_m128",
    "piece_persistent_m128",
    "head_grouped_persistent_m128",
    "lpt_persistent_m128",
    "scalar_chunk_lpt_m128",
}


def _sm_count():
    return torch.cuda.get_device_properties(0).multi_processor_count


def _many(lengths_of):
    # more (sequence, head) tasks than SMs at 64 heads, so a persistent schedule is chosen
    return lambda: lengths_of(2 * _sm_count() // 64 + 2)


# (id, lengths or a callable returning them, heads, state, the schedules the call may take)
CASES = [
    ("direct", (37, 40, 100, 17, 130), 16, "bf16", {"direct_m128"}),
    ("direct_fp32_pool", (37, 40, 100, 17, 130), 16, "fp32_pool", {"direct_m128"}),
    (
        "persistent_uniform",
        _many(lambda n: (513,) * n),
        64,
        "bf16",
        _PERSISTENT,
    ),
    (
        "persistent_mixed",
        _many(lambda n: tuple(700 - 13 * i for i in range(n))),
        64,
        "bf16",
        _PERSISTENT,
    ),
    ("small_bh", (4100, 2049), 4, "bf16", {"small_bh_owner_helper_m128"}),
    (
        "small_bh_fp32_pool",
        (4100, 2049),
        4,
        "fp32_pool",
        {"small_bh_owner_helper_m128"},
    ),
    ("small_bh_no_state", (4100, 2049), 2, "none", {"small_bh_owner_helper_m128"}),
    ("bt16", (4100, 4097, 5000, 4099), 8, "bf16", {"bt16_prepare_chain_m64"}),
    (
        "bt16_fp32_pool",
        (4100, 4097, 5000, 4099),
        8,
        "fp32_pool",
        {"bt16_prepare_chain_m64"},
    ),
    ("bt16_no_state", (4100, 2049), 8, "none", {"bt16_prepare_chain_m64"}),
]
POISONS = {
    "v_nan": (("v",), float("nan")),
    "v_inf": (("v",), float("inf")),
    "all_nan": (("q", "k", "v", "g", "beta"), float("nan")),
}


def _inputs(lengths, heads, state, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    tokens = sum(lengths)
    shape = (1, tokens, heads, HEAD_DIM)

    def randn(*size, scale=1.0, shift=0.0):
        return torch.randn(size, generator=gen, device="cuda") * scale + shift

    d = dict(
        q=randn(*shape).bfloat16(),
        k=randn(*shape).bfloat16(),
        v=randn(*shape).bfloat16(),
        g=randn(*shape).bfloat16(),
        beta=randn(*shape[:3]).bfloat16(),
        A_log=randn(heads, scale=0.2),
        dt_bias=randn(heads * HEAD_DIM, scale=0.1),
    )
    offsets = [0]
    for n in lengths:
        offsets.append(offsets[-1] + n)
    d["offsets"] = offsets
    d["cu_seqlens"] = torch.tensor(offsets, device="cuda", dtype=torch.int64)
    count = len(lengths)
    d["indices"] = None
    if state == "bf16":
        d["state"] = randn(count, heads, HEAD_DIM, HEAD_DIM, scale=0.05).bfloat16()
    elif state == "fp32_pool":
        d["state"] = randn(count + 3, heads, HEAD_DIM, HEAD_DIM, scale=0.05)
        d["indices"] = torch.arange(count + 2, 1, -1, device="cuda", dtype=torch.int32)[
            :count
        ].contiguous()
    else:
        d["state"] = None
    return d


def _launch(d, operands):
    state = None if d["state"] is None else d["state"].clone()
    out, final_state = recurrent_kda(
        q=operands["q"],
        k=operands["k"],
        v=operands["v"],
        g=operands["g"],
        beta=operands["beta"],
        A_log=d["A_log"],
        dt_bias=d["dt_bias"],
        initial_state=state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        lower_bound=LOWER_BOUND,
        cu_seqlens=d["cu_seqlens"],
        ssm_state_indices=d["indices"],
        beta_is_logit=True,
        backend="cake",
    )
    torch.cuda.synchronize()
    if d["indices"] is not None:
        final_state = state[d["indices"].long()]
    return out, final_state


def _sequence(d, launch, seq):
    out, final_state = launch
    start, end = d["offsets"][seq], d["offsets"][seq + 1]
    return {"output": out[0, start:end], "final_state": final_state[seq]}


@pytest.mark.parametrize("poison", list(POISONS))
@pytest.mark.parametrize(("case", "lengths", "heads", "state", "schedules"), CASES)
def test_cake_prefill_keeps_nonfinite_tokens_inside_their_sequence(
    case, lengths, heads, state, schedules, poison
):
    lengths = lengths() if callable(lengths) else lengths
    route = kda_prefill_api.select_bf16_schedule_route(
        compute_capability=torch.cuda.get_device_capability(),
        sm_count=_sm_count(),
        fixed_layout=False,
        sequence_lengths=tuple(lengths),
        num_heads=heads,
        state_dtype_is_fp32=state == "fp32_pool",
    )
    if route not in schedules:
        pytest.skip(f"this device schedules {case} as {route}")
    d = _inputs(lengths, heads, state)
    names, value = POISONS[poison]
    clean = {name: d[name] for name in ("q", "k", "v", "g", "beta")}
    finite = _launch(d, clean)
    count = len(lengths)
    for parity in (0, 1):
        poisoned = {name: tensor.clone() for name, tensor in clean.items()}
        for seq in range(parity, count, 2):
            start, end = d["offsets"][seq], d["offsets"][seq + 1]
            for name in names:
                poisoned[name][0, start:end] = value
        launch = _launch(d, poisoned)
        for seq in range(1 - parity, count, 2):
            expected = _sequence(d, finite, seq)
            for part, actual in _sequence(d, launch, seq).items():
                where = f"{case} {route}: sequence {seq} {part} (poisoned neighbours)"
                assert torch.isfinite(actual.float()).all(), f"{where} is not finite"
                assert torch.equal(
                    actual.view(torch.uint8), expected[part].view(torch.uint8)
                ), f"{where} differs from the finite launch"


@pytest.mark.parametrize("kernels", ["generated", "static"])
def test_bt16_static_and_generated_chains_keep_nonfinite_tokens_inside_their_sequence(
    kernels, monkeypatch
):
    """The compact BF16 BT16 route runs the generated chain, or the hand-written chain when
    the generated selector misses; both must clear the V rows past a sequence end."""
    lengths, heads = (4100, 4097, 5000, 4099), 8
    route = kda_prefill_api.select_bf16_schedule_route(
        compute_capability=torch.cuda.get_device_capability(),
        sm_count=_sm_count(),
        fixed_layout=False,
        sequence_lengths=lengths,
        num_heads=heads,
    )
    if route != "bt16_prepare_chain_m64":
        pytest.skip(f"this device schedules the BT16 case as {route}")
    launches = []
    generated = kda_prefill_api._run_generated_bt16_prepare_chain
    static = kda_prefill_api._run_bt16_prepare_chain

    def run_generated(**kwargs):
        launches.append("generated")
        if kernels == "static":
            raise _GeneratedFlashKDASelectorNotFoundError("hand-written BT16 kernels")
        return generated(**kwargs)

    def run_static(**kwargs):
        launches.append("static")
        return static(**kwargs)

    monkeypatch.setattr(
        kda_prefill_api, "_run_generated_bt16_prepare_chain", run_generated
    )
    monkeypatch.setattr(kda_prefill_api, "_run_bt16_prepare_chain", run_static)
    d = _inputs(lengths, heads, "bf16")
    clean = {name: d[name] for name in ("q", "k", "v", "g", "beta")}
    finite = _launch(d, clean)
    poisoned = {name: tensor.clone() for name, tensor in clean.items()}
    for seq in (1, 3):
        poisoned["v"][0, d["offsets"][seq] : d["offsets"][seq + 1]] = float("nan")
    launch = _launch(d, poisoned)
    expected_launches = (
        ["generated"] if kernels == "generated" else ["generated", "static"]
    )
    assert launches == expected_launches * 2
    for seq in (0, 2):
        expected = _sequence(d, finite, seq)
        for part, actual in _sequence(d, launch, seq).items():
            assert torch.isfinite(actual.float()).all(), (kernels, seq, part)
            assert torch.equal(
                actual.view(torch.uint8), expected[part].view(torch.uint8)
            ), (kernels, seq, part)


# The TF32 BT16 chain modules: one shared kernel body, selected per module by its knobs.
_TF32_BT16_CHAIN_BODY = (
    "de0b648b181aea585317a33cbb59f25b403246d990627eb3e6cdab1a9c874ab0"
)


@pytest.mark.parametrize(
    ("lengths", "heads", "checkpoints"),
    [
        pytest.param((4100, 2049), 8, False, id="long_h8"),
        pytest.param((4100, 2049), 1, False, id="long_h1"),
        pytest.param((300, 517, 1023, 77), 12, False, id="mixed_h12"),
        pytest.param((4100, 2049), 8, True, id="long_h8_ckpt"),
    ],
)
@pytest.mark.parametrize("poison", list(POISONS))
def test_prepared_tf32_bt16_chain_keeps_nonfinite_tokens_inside_their_sequence(
    lengths, heads, checkpoints, poison, monkeypatch
):
    """The TF32 BT16 chain loads a tail chunk's V whole; it must clear the rows past the end."""
    chain_modules = set(cake_kda_tf32._TF32_FOLDS[_TF32_BT16_CHAIN_BODY][1])
    prepared = []
    init = cake_kda_tf32.PreparedKernel.__init__

    def record(self, name, arch):
        prepared.append(name)
        init(self, name, arch)

    monkeypatch.setattr(cake_kda_tf32.PreparedKernel, "__init__", record)
    d = _inputs(lengths, heads, "fp32_pool")
    count = len(lengths)
    out = torch.empty_like(d["q"])
    pool = d["state"]
    kwargs = {}
    if checkpoints:
        starts = [0]
        for n in lengths:
            starts.append(starts[-1] + (n + 63) // 64)
        kwargs = dict(
            state_checkpoints=torch.empty(
                (starts[-1], heads, HEAD_DIM, HEAD_DIM),
                device="cuda",
                dtype=torch.bfloat16,
            ),
            checkpoint_cu_starts=torch.tensor(starts, device="cuda", dtype=torch.int64),
            checkpoint_every_n_tokens=64,
        )
    operands = {name: d[name].clone() for name in ("q", "k", "v", "g", "beta")}
    call = prepare_tf32_kda_prefill(
        *(operands[name] for name in ("q", "k", "v", "g", "beta")),
        A_log=d["A_log"],
        dt_bias=d["dt_bias"].reshape(heads, HEAD_DIM),
        out=out,
        initial_state=pool,
        final_state=pool,
        lower_bound=LOWER_BOUND,
        cu_seqlens=d["cu_seqlens"],
        sequence_lengths=tuple(lengths),
        state_indices=d["indices"],
        **kwargs,
    )
    if not chain_modules & set(prepared):
        pytest.skip(
            f"this device prepares the call without the TF32 BT16 chain: {prepared}"
        )
    initial = pool.clone()

    def launch():
        pool.copy_(initial)
        out.fill_(0)
        call.launch()
        torch.cuda.synchronize()
        rows = {
            "output": out.clone(),
            "final_state": pool[d["indices"].long()].clone(),
        }
        if checkpoints:
            rows["checkpoints"] = kwargs["state_checkpoints"].clone()
        return rows

    finite = launch()
    names, value = POISONS[poison]
    for seq in range(1, count, 2):
        start, end = d["offsets"][seq], d["offsets"][seq + 1]
        for name in names:
            operands[name][0, start:end] = value
    poisoned = launch()
    for seq in range(0, count, 2):
        start, end = d["offsets"][seq], d["offsets"][seq + 1]
        parts = {
            "output": (slice(None), slice(start, end)),
            "final_state": (seq,),
        }
        if checkpoints:
            first, last = (
                int(kwargs["checkpoint_cu_starts"][seq]),
                int(kwargs["checkpoint_cu_starts"][seq + 1]),
            )
            parts["checkpoints"] = (slice(first, last),)
        for part, index in parts.items():
            actual, expected = poisoned[part][index], finite[part][index]
            where = f"TF32 BT16 chain: sequence {seq} {part} (poisoned neighbours)"
            assert torch.isfinite(actual.float()).all(), f"{where} is not finite"
            assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)), (
                f"{where} differs from the finite launch"
            )
