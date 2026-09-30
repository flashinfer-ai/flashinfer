"""Oracle and contract tests for chunked lower-bounded KDA prefill.

The CPU tests pin the reference algebra and the mirror's rounding policy; the
GPU tests (added with the kernels) compare the CuTe DSL op against them.
"""

from __future__ import annotations

import gc
import math

import pytest
import torch

from b12x.preparation import PreparationSession, PreparedCall, require_prepared

from b12x.sequence._shared.kda_math import kda_beta, kda_log_decay, l2_normalize
from b12x.sequence.kda_prefill.reference import (
    MirrorPolicy,
    prefill_kda,
    prefill_kda_chunk_mirror,
    recurrent_kda,
)

HEAD_DIM = 128
CPU = torch.device("cpu")
PURE_FP32 = MirrorPolicy(shadow=False, inv_operand="fp32", u_operand="fp32", operands="fp32")
FLASHKDA_LIKE = MirrorPolicy(state_master="bf16", single_rounding=False, scale_dtype="bf16")


def rmse_ratio(reference: torch.Tensor, actual: torch.Tensor) -> float:
    delta = (reference.float() - actual.float()).flatten()
    base = reference.float().flatten()
    return (delta.square().mean().sqrt() / (base.square().mean().sqrt() + 1e-8)).item()


def assert_kda_close(
    name: str,
    reference: torch.Tensor,
    actual: torch.Tensor,
    *,
    ratio: float,
    peak_ratio: float = 4e-2,
    exact_atol: float = 1e-6,
) -> None:
    assert torch.isfinite(actual.float()).all(), f"{name}: non-finite values"
    delta = (reference.float() - actual.float()).abs()
    if delta.max().item() <= exact_atol:
        return
    observed = rmse_ratio(reference, actual)
    assert observed < ratio, f"{name}: rmse ratio {observed:.3e} >= {ratio}"
    rms = reference.float().square().mean().sqrt().item()
    peak = reference.float().abs().max().item()
    assert delta.max().item() <= peak_ratio * rms + 2**-6 * peak, f"{name}: peak error"


def make_inputs(
    *,
    lengths: list[int],
    heads: int = 2,
    device: torch.device = CPU,
    seed: int = 0,
    gate_profile: str = "random",
    key_profile: str = "random",
    lower_bound: float = -5.0,
    token_capacity: int | None = None,
    state_slots: int | None = None,
    initial: list[int] | None = None,
    final: list[int] | None = None,
    checkpoint: list[tuple[int, int]] | None = None,
    null_state_index: int | None = None,
) -> dict:
    """Build packed inputs; slot assignment defaults to distinct slots."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    tokens = sum(lengths)
    count = len(lengths)
    capacity = tokens if token_capacity is None else token_capacity

    def bf16(*shape, scale=0.25):
        return (torch.randn(*shape, generator=generator) * scale).to(torch.bfloat16)

    q, k, v = bf16(capacity, heads, HEAD_DIM), bf16(capacity, heads, HEAD_DIM), bf16(capacity, heads, HEAD_DIM)
    raw_g = bf16(capacity, heads, HEAD_DIM, scale=1.0)
    raw_beta = bf16(capacity, heads, scale=1.0)
    if gate_profile == "long_memory":
        raw_g[:, :, :32] = -12.0
    elif gate_profile == "saturated":
        raw_g.fill_(12.0)
    elif gate_profile == "zero":
        raw_g.fill_(-12.0)
    if key_profile in ("repeated", "alternating"):
        unit = torch.randn(heads, HEAD_DIM, generator=generator)
        unit = unit / unit.norm(dim=-1, keepdim=True)
        pattern = unit[None].expand(capacity, heads, HEAD_DIM).clone()
        if key_profile == "alternating":
            pattern[1::2] *= -1.0
        k = pattern.to(torch.bfloat16)
        raw_beta.fill_(12.0)
    A_log = torch.randn(heads, generator=generator) * 0.1
    dt_bias = torch.randn(heads, HEAD_DIM, generator=generator) * 0.1
    slots = 3 * count + 2 if state_slots is None else state_slots
    pool = torch.randn(slots, heads, HEAD_DIM, HEAD_DIM, generator=generator) * 0.1
    initial = list(range(count)) if initial is None else initial
    final = list(range(count, 2 * count)) if final is None else final
    checkpoint = [(0, 0)] * count if checkpoint is None else checkpoint
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    to = lambda t: t.to(device)  # noqa: E731
    return {
        "q": to(q), "k": to(k), "v": to(v), "raw_g": to(raw_g), "raw_beta": to(raw_beta),
        "A_log": to(A_log), "dt_bias": to(dt_bias), "pool": to(pool),
        "cu_seqlens": torch.tensor(cu, dtype=torch.int32, device=device),
        "initial": torch.tensor(initial, dtype=torch.int32, device=device),
        "final": torch.tensor(final, dtype=torch.int32, device=device),
        "checkpoint_slots": torch.tensor([c[1] for c in checkpoint], dtype=torch.int32, device=device),
        "checkpoint_offsets": torch.tensor([c[0] for c in checkpoint], dtype=torch.int32, device=device),
        "num_seqs": count, "num_tokens": tokens, "lower_bound": lower_bound,
        "null_state_index": null_state_index,
    }


def run_oracle(inputs: dict, fn=prefill_kda, **extra):
    pool = inputs["pool"].clone()
    output = fn(
        inputs["q"], inputs["k"], inputs["v"], inputs["raw_g"], inputs["raw_beta"],
        inputs["A_log"], inputs["dt_bias"], pool, inputs["cu_seqlens"], inputs["initial"],
        inputs["final"], inputs["checkpoint_slots"], inputs["checkpoint_offsets"],
        inputs["num_seqs"], inputs["num_tokens"], lower_bound=inputs["lower_bound"],
        null_state_index=inputs["null_state_index"], **extra,
    )
    return output, pool


def test_shared_gate_helper_matches_decode_kda_expression() -> None:
    torch.manual_seed(3)
    raw_g = torch.randn(5, 2, HEAD_DIM).to(torch.bfloat16)
    dt_bias = torch.randn(2, HEAD_DIM) * 0.1
    A_log = torch.randn(2) * 0.1
    helper = kda_log_decay(raw_g, dt_bias, A_log, -5.0)
    for token in range(5):
        for head in range(2):
            rate = torch.exp(A_log[head].float())
            expected = -5.0 * torch.sigmoid(rate * (raw_g[token, head].float() + dt_bias[head].float()))
            torch.testing.assert_close(helper[token, head], expected, rtol=0, atol=0)
    beta = torch.randn(5, 2).to(torch.bfloat16)
    torch.testing.assert_close(kda_beta(beta), torch.sigmoid(beta.float()), rtol=0, atol=0)
    x = torch.randn(5, 2, HEAD_DIM).to(torch.bfloat16).float()
    torch.testing.assert_close(
        l2_normalize(x), x * torch.rsqrt(x.square().sum(-1, keepdim=True) + 1e-6), rtol=0, atol=0
    )


def test_recurrent_oracle_matches_decode_kda_token_by_token() -> None:
    from b12x.sequence.gdn_decode.reference import decode_kda

    heads, tokens = 2, 8
    inputs = make_inputs(lengths=[tokens], heads=heads, seed=7, state_slots=tokens + 2)
    q, k, v = inputs["q"], inputs["k"], inputs["v"]
    mixed = torch.cat([q.reshape(tokens, -1), k.reshape(tokens, -1), v.reshape(tokens, -1)], dim=1)
    pool = inputs["pool"].clone()
    state_indices = torch.arange(0, tokens, dtype=torch.int32)[None]
    z = torch.zeros(tokens, heads, HEAD_DIM, dtype=torch.bfloat16)
    norm_weight = torch.ones(HEAD_DIM)
    decode_out = decode_kda(
        mixed, inputs["raw_g"], inputs["raw_beta"], z, inputs["A_log"], inputs["dt_bias"],
        norm_weight, pool, torch.tensor([0, tokens], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32), state_indices, 1, tokens, heads=heads,
        lower_bound=-5.0,
    )
    del decode_out  # the decode epilogue applies a gated norm; compare states.
    out, final, _ = recurrent_kda(
        q, k, v, inputs["raw_g"], inputs["raw_beta"], inputs["A_log"], inputs["dt_bias"],
        lower_bound=-5.0, initial_state=inputs["pool"][0],
    )
    torch.testing.assert_close(final, pool[tokens - 1], rtol=1e-5, atol=1e-6)
    assert torch.isfinite(out.float()).all()


@pytest.mark.parametrize(
    "lengths,checkpoint",
    [([0], None), ([0, 33], None), ([40], [(16, 5)]), ([40], [(0, 5)]), ([40], [(32, 5)])],
)
def test_prefill_oracle_contract_cases(lengths, checkpoint) -> None:
    inputs = make_inputs(lengths=lengths, seed=11, checkpoint=checkpoint, state_slots=8)
    tail = torch.full((3, 2, HEAD_DIM), float("nan"), dtype=torch.bfloat16)
    output = torch.cat([torch.zeros(inputs["num_tokens"], 2, HEAD_DIM, dtype=torch.bfloat16), tail])
    padded = dict(inputs)
    for name in ("q", "k", "v", "raw_g"):
        padded[name] = torch.cat([inputs[name], torch.zeros(3, 2, HEAD_DIM, dtype=torch.bfloat16)])
    padded["raw_beta"] = torch.cat([inputs["raw_beta"], torch.zeros(3, 2, dtype=torch.bfloat16)])
    out, pool = run_oracle(padded, output=output)
    assert torch.isnan(out[inputs["num_tokens"] :].float()).all()
    for request, length in enumerate(lengths):
        initial = int(inputs["initial"][request])
        final = int(inputs["final"][request])
        if length == 0:
            torch.testing.assert_close(pool[final], inputs["pool"][initial], rtol=0, atol=0)
    if checkpoint is not None:
        offset, slot = checkpoint[0]
        start = 0
        _, _, expected = recurrent_kda(
            inputs["q"][start : start + lengths[0]], inputs["k"][start : start + lengths[0]],
            inputs["v"][start : start + lengths[0]], inputs["raw_g"][start : start + lengths[0]],
            inputs["raw_beta"][start : start + lengths[0]], inputs["A_log"], inputs["dt_bias"],
            lower_bound=-5.0, initial_state=inputs["pool"][0], checkpoint_offset=offset,
        )
        if offset > 0:
            torch.testing.assert_close(pool[slot], expected, rtol=1e-6, atol=1e-6)
        else:
            torch.testing.assert_close(pool[slot], inputs["pool"][slot], rtol=0, atol=0)


def test_prefill_oracle_null_initial_never_reads_the_slot() -> None:
    inputs = make_inputs(lengths=[20], seed=5, initial=[0], final=[1], null_state_index=0)
    inputs["pool"][0].fill_(float("nan"))
    out, pool = run_oracle(inputs)
    assert torch.isfinite(out.float()).all()
    zero_start = make_inputs(lengths=[20], seed=5, initial=[2], final=[1])
    zero_start["pool"][2].zero_()
    expected_out, expected_pool = run_oracle(zero_start)
    torch.testing.assert_close(out, expected_out, rtol=0, atol=0)
    torch.testing.assert_close(pool[1], expected_pool[1], rtol=0, atol=0)


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda i: i["cu_seqlens"].__setitem__(2, 30), "must equal num_tokens"),
        (lambda i: i.update(num_tokens=1_000_000), "exceeds capacity"),
        (lambda i: i["checkpoint_offsets"].__setitem__(0, 17), "not a multiple"),
        (lambda i: i["checkpoint_offsets"].__setitem__(0, 64), "exceeds the sequence"),
        (lambda i: i["final"].__setitem__(0, int(i["final"][1])), "duplicate write"),
        (lambda i: i["final"].__setitem__(1, int(i["initial"][0])), "written by another"),
    ],
)
def test_prefill_oracle_rejects_bad_metadata(mutate, match) -> None:
    inputs = make_inputs(lengths=[20, 20], seed=9, checkpoint=[(16, 6), (0, 0)])
    mutate(inputs)
    with pytest.raises((ValueError, IndexError), match=match):
        run_oracle(inputs)


@pytest.mark.parametrize(
    "lengths",
    [[1], [15], [16], [17], [64], [0, 64, 0, 15], [15, 100, 200, 900]],
    ids=lambda v: "-".join(map(str, v)),
)
@pytest.mark.parametrize("lower_bound", [-5.0, -3.0, -0.5])
@pytest.mark.parametrize("gate_profile", ["random", "long_memory", "saturated"])
def test_chunk_mirror_matches_recurrent_oracle(lengths, lower_bound, gate_profile) -> None:
    checkpoint = [(0, 0)] * len(lengths)
    if lengths[-1] >= 32:
        checkpoint[-1] = (32, 3 * len(lengths))
    inputs = make_inputs(
        lengths=lengths, seed=21, lower_bound=lower_bound, gate_profile=gate_profile,
        checkpoint=checkpoint,
    )
    expected_out, expected_pool = run_oracle(inputs)
    default_out, default_pool = run_oracle(inputs, prefill_kda_chunk_mirror)
    pure_out, pure_pool = run_oracle(inputs, prefill_kda_chunk_mirror, policy=PURE_FP32)
    tokens = inputs["num_tokens"]
    writes = [int(s) for s in inputs["final"]] + [slot for offset, slot in checkpoint if offset > 0]
    if tokens:
        assert_kda_close("out", expected_out[:tokens], default_out[:tokens], ratio=1e-2)
        assert_kda_close("out-fp32", expected_out[:tokens], pure_out[:tokens], ratio=2e-4)
    for slot in writes:
        assert_kda_close(f"state[{slot}]", expected_pool[slot], default_pool[slot], ratio=5e-3)
        assert_kda_close(f"state-fp32[{slot}]", expected_pool[slot], pure_pool[slot], ratio=2e-5)
    untouched = [s for s in range(inputs["pool"].shape[0]) if s not in writes]
    torch.testing.assert_close(default_pool[untouched], inputs["pool"][untouched], rtol=0, atol=0)


def test_mirror_policy_study_fp32_master_beats_bf16_state_on_long_memory() -> None:
    inputs = make_inputs(lengths=[16384], heads=1, seed=31, gate_profile="long_memory")
    _, expected_pool = run_oracle(inputs)
    slot = int(inputs["final"][0])
    _, fp32_pool = run_oracle(inputs, prefill_kda_chunk_mirror)
    _, bf16_pool = run_oracle(inputs, prefill_kda_chunk_mirror, policy=FLASHKDA_LIKE)
    err_fp32 = rmse_ratio(expected_pool[slot], fp32_pool[slot])
    err_bf16 = rmse_ratio(expected_pool[slot], bf16_pool[slot])
    assert err_fp32 <= 5e-3, err_fp32
    assert err_bf16 >= 1.5 * err_fp32, (err_fp32, err_bf16)


@pytest.mark.parametrize("key_profile", ["random", "repeated", "alternating"])
@pytest.mark.parametrize("gate_profile", ["random", "zero", "saturated"])
def test_mirror_inverse_growth_bound_on_adversarial_keys(key_profile, gate_profile) -> None:
    worst = 0.0
    for seed in range(8):
        inputs = make_inputs(
            lengths=[64], heads=2, seed=100 + seed, key_profile=key_profile, gate_profile=gate_profile
        )
        _, trace = prefill_kda_chunk_mirror(
            inputs["q"], inputs["k"], inputs["v"], inputs["raw_g"], inputs["raw_beta"],
            inputs["A_log"], inputs["dt_bias"], inputs["pool"].clone(), inputs["cu_seqlens"],
            inputs["initial"], inputs["final"], inputs["checkpoint_slots"],
            inputs["checkpoint_offsets"], inputs["num_seqs"], inputs["num_tokens"], trace=True,
        )
        for tile in trace.k1.values():
            assert torch.isfinite(tile["inv"]).all()
            worst = max(worst, tile["inv"].abs().max().item())
    assert worst <= 8.0, worst


def test_run_rejects_lower_bound_outside_range() -> None:
    inputs = make_inputs(lengths=[16], seed=1)
    for bad in (-5.5, 0.0, math.nan):
        inputs["lower_bound"] = bad
        with pytest.raises(ValueError, match="lower_bound"):
            run_oracle(inputs)
        with pytest.raises(ValueError, match="lower_bound"):
            run_oracle(inputs, prefill_kda_chunk_mirror)


# ---------------------------------------------------------------------------
# GPU: prologue and prepare kernels against the chunk mirror trace.
_PREPARATIONS: list[tuple[object, PreparationSession]] = []


@pytest.fixture(autouse=True)
def _prepared_scopes():
    try:
        yield
    finally:
        gc.collect()
        while _PREPARATIONS:
            result, session = _PREPARATIONS.pop()
            result.close()
            session.close()


def make_binding(
    inputs: dict,
    *,
    max_tokens: int,
    max_seqs: int,
    final_stride: int = 1,
    recurrent_state: torch.Tensor | None = None,
    config=None,
    **caps_extra,
):
    """Prepare a real KDA plan and bind caller-owned serving tensors."""
    from b12x.sequence.kda_prefill import _impl as impl, invocation_from_tensors

    device = inputs["q"].device
    heads = int(inputs["q"].shape[1])
    recurrent_state = inputs["pool"].clone() if recurrent_state is None else recurrent_state
    caps = impl.Caps(
        device=device, max_tokens=max_tokens, max_seqs=max_seqs,
        max_state_slots=int(recurrent_state.shape[0]), heads=heads,
        null_state_index=inputs["null_state_index"], **caps_extra,
    )

    def pad_rows(t: torch.Tensor) -> torch.Tensor:
        out = torch.zeros((max_tokens,) + tuple(t.shape[1:]), dtype=t.dtype, device=device)
        out[: t.shape[0]] = t
        return out

    def pad_seqs(t: torch.Tensor, extra: int = 0) -> torch.Tensor:
        out = torch.zeros(max_seqs + extra, dtype=t.dtype, device=device)
        out[: t.shape[0]] = t
        return out

    final_storage = torch.zeros((max_seqs, final_stride), dtype=inputs["final"].dtype, device=device)
    final_state_indices = final_storage[:, 0]
    final_state_indices[: inputs["final"].shape[0]] = inputs["final"]
    tensors = {
        "q": pad_rows(inputs["q"]), "k": pad_rows(inputs["k"]), "v": pad_rows(inputs["v"]),
        "raw_g": pad_rows(inputs["raw_g"]), "raw_beta": pad_rows(inputs["raw_beta"]),
        "A_log": inputs["A_log"], "dt_bias": inputs["dt_bias"], "recurrent_state": recurrent_state,
        "cu_seqlens": pad_seqs(inputs["cu_seqlens"], extra=1),
        "initial_state_indices": pad_seqs(inputs["initial"]),
        "final_state_indices": final_state_indices,
        "checkpoint_state_indices": pad_seqs(inputs["checkpoint_slots"]),
        "checkpoint_offsets": pad_seqs(inputs["checkpoint_offsets"]),
        "num_seqs": torch.tensor([inputs["num_seqs"]], dtype=torch.int32, device=device),
        "num_tokens": torch.tensor([inputs["num_tokens"]], dtype=torch.int32, device=device),
        "output": torch.zeros(max_tokens, heads, HEAD_DIM, dtype=torch.bfloat16, device=device),
    }
    override = config
    plan = impl.plan(
        caps, invocation=invocation_from_tensors(**tensors), override=override,
    )

    def prepare_call(state):
        (spec,) = state.layout.scratch_specs()
        binding = state.bind(
            scratch=torch.empty(spec.shape, dtype=spec.dtype, device=device), **tensors
        )
        original_state, original_output = binding.recurrent_state.clone(), binding.output.clone()

        def reset():
            binding.recurrent_state.copy_(original_state)
            binding.output.copy_(original_output)

        return PreparedCall(
            run=lambda: state.run(binding, lower_bound=inputs["lower_bound"]),
            output=binding.output, reset=reset, restore=reset,
        )

    request = plan.request(
        name=f"kda-prefill-{id(tensors)}", prepare_call=prepare_call,
    )
    session = PreparationSession(device=device, autotune=False)
    result = session.prepare((request,))
    state = require_prepared(plan, "sequence.kda_prefill")
    (spec,) = state.layout.scratch_specs()
    _PREPARATIONS.append((result, session))
    return impl.bind(
        plan, scratch=torch.empty(spec.shape, dtype=spec.dtype, device=device), **tensors
    ), tensors

def _prepare_tensors(caps, tensors: dict[str, torch.Tensor], *, lower_bound: float):
    """Session-prepare an already-shaped binding without replacing its views."""
    from b12x.sequence.kda_prefill import _impl as impl, invocation_from_tensors

    plan = impl.plan(caps, invocation=invocation_from_tensors(**tensors))

    def prepare_call(state):
        (spec,) = state.layout.scratch_specs()
        binding = state.bind(
            scratch=torch.empty(spec.shape, dtype=spec.dtype, device=caps.device), **tensors
        )
        original_state, original_output = binding.recurrent_state.clone(), binding.output.clone()

        def reset():
            binding.recurrent_state.copy_(original_state)
            binding.output.copy_(original_output)

        return PreparedCall(
            run=lambda: state.run(binding, lower_bound=lower_bound),
            output=binding.output, reset=reset, restore=reset,
        )

    request = plan.request(
        name=f"kda-prefill-{id(tensors)}", prepare_call=prepare_call,
    )
    session = PreparationSession(device=caps.device, autotune=False)
    result = session.prepare((request,))
    state = require_prepared(plan, "sequence.kda_prefill")
    (spec,) = state.layout.scratch_specs()
    _PREPARATIONS.append((result, session))
    return impl.bind(
        plan,
        scratch=torch.empty(spec.shape, dtype=spec.dtype, device=caps.device),
        **tensors,
    )



def _mirror_trace(inputs: dict):
    _, trace = prefill_kda_chunk_mirror(
        inputs["q"], inputs["k"], inputs["v"], inputs["raw_g"], inputs["raw_beta"],
        inputs["A_log"], inputs["dt_bias"], inputs["pool"].clone(), inputs["cu_seqlens"],
        inputs["initial"], inputs["final"], inputs["checkpoint_slots"], inputs["checkpoint_offsets"],
        inputs["num_seqs"], inputs["num_tokens"], lower_bound=inputs["lower_bound"],
        null_state_index=inputs["null_state_index"], trace=True,
    )
    return trace


def test_prologue_initializes_more_windows_than_threads() -> None:
    """Small tuning windows must initialize the entire table, including its tail."""
    from types import SimpleNamespace

    from ..conftest import require_b12x
    from b12x.sequence._shared.delta_prefill._cute_kernels import _compile_prologue

    device = require_b12x()
    tiles, windows, seqs = 515, 515, 4
    caps = SimpleNamespace(max_seqs=seqs, tiles_capacity=tiles, heads=1)
    layout = SimpleNamespace(caps=caps, window_tiles=1, max_windows=windows, workspace_windows=2)

    def integers(size):
        return torch.full((size,), -123456, dtype=torch.int32, device=device)

    binding = SimpleNamespace(
        _state=layout, output=torch.empty(0, device=device), seq_capacity=seqs,
        cu_seqlens=integers(seqs + 1), num_seqs=integers(1),
        band_base=integers(tiles + 2), sorted_seq=integers(seqs),
        rank_of=integers(seqs), pos_seq=integers(tiles), pos_local=integers(tiles),
        window_table=integers(2 * windows), ready_flags=integers(2),
    )
    _, launch = _compile_prologue(binding)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(binding)
    for tokens in (8192, 17, 0):
        binding.cu_seqlens.zero_()
        binding.cu_seqlens[1:].fill_(tokens)
        binding.num_seqs.fill_(1)
        binding.window_table.fill_(-123456)
        graph.replay()
        table = binding.window_table.reshape(windows, 2).cpu()
        count = (tokens + 15) // 16
        for window, (band, rank) in enumerate(table.tolist()):
            assert band == (window if window < count else tiles - 1)
            assert rank == max(0, window - count)


@pytest.mark.parametrize(
    "lengths",
    [[1], [16], [17], [15, 100, 0, 300, 33], [64, 64]],
    ids=lambda v: "-".join(map(str, v)),
)
@pytest.mark.parametrize("lower_bound", [-5.0, -0.5])
def test_prepare_kernel_matches_chunk_mirror(lengths, lower_bound) -> None:
    from ..conftest import require_b12x
    from b12x.sequence._shared.delta_prefill._cute_kernels import run_prepare, run_prologue, workspace_tiles

    device = require_b12x()
    checkpoint = [(0, 0)] * len(lengths)
    if lengths[-1] >= 32:
        checkpoint[-1] = (32, 3 * len(lengths))
    inputs = make_inputs(
        lengths=lengths, heads=2, seed=41, device=device, lower_bound=lower_bound,
        checkpoint=checkpoint, gate_profile="saturated" if lower_bound == -5.0 else "random",
    )
    binding, _ = make_binding(inputs, max_tokens=512, max_seqs=8)
    run_prologue(binding)
    run_prepare(binding, lower_bound=lower_bound, scale=HEAD_DIM**-0.5, eps=1e-6)
    torch.cuda.synchronize(device)
    trace = _mirror_trace(inputs)
    counts = [(length + 15) // 16 for length in lengths]
    order = sorted(range(len(lengths)), key=lambda seq: (-counts[seq], seq))
    rank_of = binding.rank_of[: len(lengths)].tolist()
    assert sorted(rank_of) == list(range(len(lengths)))
    assert [rank_of[seq] for seq in order] == list(range(len(lengths))) or all(
        counts[order[rank]] == counts[binding.sorted_seq[rank].item()] for rank in range(len(lengths))
    ), "ranks must order sequences by descending tile count"
    band_base = binding.band_base.tolist()
    expected_band = [0]
    for local in range(max(counts) + 1):
        expected_band.append(expected_band[-1] + sum(1 for c in counts if c > local))
    assert band_base[: len(expected_band)] == expected_band
    total = sum(counts)
    assert band_base[binding._state.caps.tiles_capacity + 1] == total
    pos_seq = binding.pos_seq.tolist()
    pos_local = binding.pos_local.tolist()
    for seq in range(len(lengths)):
        for local in range(counts[seq]):
            position = band_base[local] + rank_of[seq]
            assert pos_seq[position] == seq and pos_local[position] == local
    assert all(t == -1 for t in pos_seq[total:])
    tiles = workspace_tiles(binding)
    for (seq, local), record in trace.k1.items():
        tile = band_base[local] + rank_of[seq]
        for name, key in (("q_tilde", "q_tilde"), ("k_tilde", "k_tilde"), ("k_r", "k_r")):
            got = tiles[name][tile].float()
            expected = record[key].float()
            assert torch.isfinite(got).all(), name
            # One bf16 ulp of slack covers the kernel's approximate exp2 and rsqrt.
            torch.testing.assert_close(got, expected, rtol=2**-7, atol=1e-6, msg=f"{name} {seq} {local}")
        for name, key, tol in (("inv", "inv_op", 2**-7), ("mqk", "mqk", 2**-9)):
            got = tiles[name][tile].float()
            expected = record[key].float()
            scale = max(1.0, expected.abs().max().item())
            assert torch.isfinite(got).all(), name
            assert (got - expected).abs().max().item() <= tol * scale, (name, seq, local)
        torch.testing.assert_close(tiles["lambda_c"][tile], record["lambda_c"], rtol=1e-4, atol=0)
        torch.testing.assert_close(tiles["beta"][tile], record["beta"], rtol=1e-5, atol=1e-6)
        rows = min(16, lengths[seq] - 16 * local)
        assert torch.count_nonzero(tiles["k_tilde"][tile, :, rows:]) == 0
        assert torch.count_nonzero(tiles["k_r"][tile, :, rows:]) == 0


# ---------------------------------------------------------------------------
# GPU: the complete op against the oracles and the serving contract.
# ---------------------------------------------------------------------------


def _run(binding, inputs: dict, **extra):
    from b12x.sequence.kda_prefill import _impl as impl

    return impl.run(binding, lower_bound=inputs["lower_bound"], **extra)


def _writes(inputs: dict) -> list[int]:
    offsets = inputs["checkpoint_offsets"].tolist()
    slots = inputs["checkpoint_slots"].tolist()
    return [int(s) for s in inputs["final"]] + [s for o, s in zip(offsets, slots, strict=True) if o > 0]


def _assert_op_matches_oracle(binding, tensors, inputs, *, out_ratio=1e-2, state_ratio=5e-3):
    tokens = inputs["num_tokens"]
    expected_out, expected_pool = run_oracle(inputs)
    if tokens:
        assert_kda_close("out", expected_out[:tokens], binding.output[:tokens], ratio=out_ratio)
    for slot in _writes(inputs):
        assert_kda_close(f"state[{slot}]", expected_pool[slot], tensors["recurrent_state"][slot], ratio=state_ratio)
    untouched = [s for s in range(inputs["pool"].shape[0]) if s not in _writes(inputs)]
    torch.testing.assert_close(tensors["recurrent_state"][untouched], inputs["pool"][untouched], rtol=0, atol=0)


@pytest.mark.parametrize("tokens", [1, 15, 16, 17, 64, 1024, 4096])
def test_op_matches_reference_single_sequence(tokens) -> None:
    from ..conftest import require_b12x

    device = require_b12x()
    inputs = make_inputs(lengths=[tokens], heads=2, seed=61, device=device)
    binding, tensors = make_binding(inputs, max_tokens=4096, max_seqs=4)
    _run(binding, inputs)
    torch.cuda.synchronize(device)
    _assert_op_matches_oracle(binding, tensors, inputs)


@pytest.mark.parametrize("heads", [16, 64])
def test_op_serving_head_geometries(heads) -> None:
    from ..conftest import require_b12x

    device = require_b12x()
    inputs = make_inputs(lengths=[256, 500, 244], heads=heads, seed=62, device=device)
    binding, tensors = make_binding(inputs, max_tokens=1024, max_seqs=4)
    _run(binding, inputs)
    torch.cuda.synchronize(device)
    _assert_op_matches_oracle(binding, tensors, inputs)


@pytest.mark.parametrize(
    "lengths",
    [[0, 0], [0, 64, 0, 79], [1, 1, 1, 1, 100], [17] * 8, [1] * 16, [15, 100, 300, 200]],
    ids=lambda v: "-".join(map(str, v)),
)
def test_op_varlen_packed_with_null_and_inplace_slots(lengths) -> None:
    from ..conftest import require_b12x

    device = require_b12x()
    count = len(lengths)
    slots = 3 * count + 2
    # Sequence 0 starts from the null slot, sequence 1 updates in place.
    initial = list(range(count))
    final = list(range(count, 2 * count))
    initial[0] = slots - 1
    final[1] = initial[1]
    inputs = make_inputs(
        lengths=lengths, heads=2, seed=63, device=device, state_slots=slots,
        initial=initial, final=final, null_state_index=slots - 1,
    )
    inputs["pool"][slots - 1].fill_(float("nan"))
    binding, tensors = make_binding(inputs, max_tokens=1024, max_seqs=16)
    _run(binding, inputs)
    torch.cuda.synchronize(device)
    tokens = inputs["num_tokens"]
    expected_out, expected_pool = run_oracle(inputs)
    if tokens:
        assert_kda_close("out", expected_out[:tokens], binding.output[:tokens], ratio=1e-2)
    for slot in set(final):
        assert_kda_close(f"state[{slot}]", expected_pool[slot], tensors["recurrent_state"][slot], ratio=5e-3)


@pytest.mark.parametrize("lower_bound", [-5.0, -3.0, -0.5])
@pytest.mark.parametrize("gate_profile", ["random", "saturated"])
def test_op_lower_bounds_and_saturated_gates(lower_bound, gate_profile) -> None:
    from ..conftest import require_b12x

    device = require_b12x()
    inputs = make_inputs(
        lengths=[64], heads=2, seed=64, device=device, lower_bound=lower_bound, gate_profile=gate_profile
    )
    binding, tensors = make_binding(inputs, max_tokens=64, max_seqs=2)
    _run(binding, inputs)
    torch.cuda.synchronize(device)
    from b12x.sequence._shared.delta_prefill._cute_kernels import workspace_tiles

    live_tiles = inputs["num_tokens"] // 16
    tiles = workspace_tiles(binding)
    assert torch.isfinite(tiles["inv"][:live_tiles].float()).all()
    assert torch.isfinite(tiles["mqk"][:live_tiles].float()).all()
    _assert_op_matches_oracle(binding, tensors, inputs)


def test_op_checkpoints_match_states_after_offset() -> None:
    from ..conftest import require_b12x

    device = require_b12x()
    lengths = [40, 1, 100]
    checkpoint = [(16, 9), (0, 0), (96, 10)]
    inputs = make_inputs(lengths=lengths, heads=2, seed=65, device=device, checkpoint=checkpoint, state_slots=12)
    binding, tensors = make_binding(inputs, max_tokens=256, max_seqs=4, checkpoint_export=True)
    _run(binding, inputs)
    torch.cuda.synchronize(device)
    _assert_op_matches_oracle(binding, tensors, inputs)
    plain = make_inputs(lengths=lengths, heads=2, seed=65, device=device, state_slots=12)
    plain_binding, plain_tensors = make_binding(plain, max_tokens=256, max_seqs=4)
    _run(plain_binding, plain)
    torch.cuda.synchronize(device)
    for slot in plain["final"].tolist():
        torch.testing.assert_close(
            tensors["recurrent_state"][slot], plain_tensors["recurrent_state"][slot], rtol=0, atol=0
        )
    torch.testing.assert_close(binding.output, plain_binding.output, rtol=0, atol=0)


@pytest.mark.parametrize("tokens", [16384, 32768])
def test_op_long_sequence_accumulation_long_memory(tokens) -> None:
    from ..conftest import require_b12x

    device = require_b12x()
    inputs = make_inputs(lengths=[tokens], heads=2, seed=66, device=device, gate_profile="long_memory")
    binding, tensors = make_binding(inputs, max_tokens=tokens, max_seqs=1)
    _run(binding, inputs)
    torch.cuda.synchronize(device)
    _assert_op_matches_oracle(binding, tensors, inputs)


@pytest.mark.parametrize("high_state_slots", [False, True], ids=["low-slots", "high-slots"])
def test_op_near_collinear_long_sequence_remains_finite(high_state_slots) -> None:
    """Reject non-finite output or state from ill-conditioned key blocks.

    Repeating one BF16 key for every token makes each sixteen-token block
    rank one. A saturated update coefficient and slow decay amplify errors
    in the blockwise triangular inverse. The KDA prefill
    contract requires finite BF16 output and FP32 recurrent state for this
    supported input.
    """
    from ..conftest import require_b12x

    device = require_b12x()
    tokens, heads = 16384, 1
    inputs = make_inputs(
        lengths=[tokens], heads=heads, seed=0, device=device, state_slots=2
    )

    torch.manual_seed(0)
    key = torch.randn(1, heads, HEAD_DIM, dtype=torch.bfloat16, device=device)
    qk = key.expand(tokens, heads, HEAD_DIM).contiguous()
    value_block = torch.randn(
        16, heads, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    inputs.update(
        q=qk,
        k=qk,
        v=value_block.repeat(tokens // 16, 1, 1),
        raw_g=torch.full_like(qk, -12.0),
        raw_beta=torch.full(
            (tokens, heads), 8.0, dtype=torch.bfloat16, device=device
        ),
        A_log=torch.zeros(heads, dtype=torch.float32, device=device),
        dt_bias=torch.zeros(heads, HEAD_DIM, dtype=torch.float32, device=device),
        pool=torch.zeros(
            2, heads, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device=device
        ),
    )

    recurrent_state = None
    if high_state_slots:
        slot_stride = heads * HEAD_DIM * HEAD_DIM
        int32_max = torch.iinfo(torch.int32).max
        high_slot = int32_max // slot_stride + 2
        required_bytes = (high_slot + 2) * slot_stride * torch.float32.itemsize
        free_bytes, _ = torch.cuda.mem_get_info(device)
        if free_bytes < required_bytes + 2 * 1024**3:
            pytest.skip(
                "KDA high-slot stability test requires "
                f"{required_bytes + 2 * 1024**3} bytes free, found {free_bytes}"
            )
        recurrent_state = torch.empty(
            high_slot + 2, heads, HEAD_DIM, HEAD_DIM,
            dtype=torch.float32, device=device,
        )
        recurrent_state[high_slot:].zero_()
        inputs["initial"] += high_slot
        inputs["final"] += high_slot
        assert high_slot * slot_stride > int32_max

    binding, tensors = make_binding(
        inputs, max_tokens=tokens, max_seqs=1, recurrent_state=recurrent_state
    )
    final_slot = int(inputs["final"][0])
    binding.output.fill_(float("nan"))
    tensors["recurrent_state"][final_slot].fill_(float("nan"))
    _run(binding, inputs)
    torch.cuda.synchronize(device)

    output_nonfinite = (~torch.isfinite(binding.output[:tokens])).sum().item()
    state_nonfinite = (
        ~torch.isfinite(tensors["recurrent_state"][final_slot])
    ).sum().item()
    assert output_nonfinite == 0, (
        f"output contains {output_nonfinite} non-finite values"
    )
    assert state_nonfinite == 0, (
        f"state contains {state_nonfinite} non-finite values"
    )
    assert torch.count_nonzero(binding.output[:tokens]).item() > 0
    assert torch.count_nonzero(tensors["recurrent_state"][final_slot]).item() > 0


@pytest.mark.parametrize("key_profile", ["repeated", "alternating"])
def test_op_adversarial_keys(key_profile) -> None:
    from ..conftest import require_b12x

    device = require_b12x()
    inputs = make_inputs(lengths=[256], heads=2, seed=67, device=device, key_profile=key_profile)
    binding, tensors = make_binding(inputs, max_tokens=256, max_seqs=1)
    _run(binding, inputs)
    torch.cuda.synchronize(device)
    _assert_op_matches_oracle(binding, tensors, inputs)


@pytest.mark.parametrize("tiles", [1, 2, 3, 5, 8, 64])
def test_op_state_matches_mirror_per_tile_prefix(tiles) -> None:
    from ..conftest import require_b12x

    device = require_b12x()
    inputs = make_inputs(lengths=[1024], heads=2, seed=68, device=device)
    binding, tensors = make_binding(inputs, max_tokens=1024, max_seqs=1)
    tokens = 16 * tiles
    tensors["num_tokens"].fill_(tokens)
    tensors["cu_seqlens"][1] = tokens
    _run(binding, inputs)
    torch.cuda.synchronize(device)
    prefix = dict(inputs)
    prefix["num_tokens"] = tokens
    prefix["cu_seqlens"] = torch.tensor([0, tokens], dtype=torch.int32, device=device)
    trace = _mirror_trace(prefix)
    slot = int(inputs["final"][0])
    assert_kda_close("state", trace.k2[(0, tiles - 1)]["state_after"], tensors["recurrent_state"][slot], ratio=1.5e-3)
    _, mirror_pool = run_oracle(prefix, prefill_kda_chunk_mirror)
    assert_kda_close("state-mirror", mirror_pool[slot], tensors["recurrent_state"][slot], ratio=1.5e-3)


def test_op_cuda_graph_replay_is_allocation_free_with_poison() -> None:
    from ..conftest import require_b12x

    device = require_b12x()
    inputs = make_inputs(lengths=[100, 30], heads=2, seed=69, device=device, checkpoint=[(32, 7), (0, 0)], state_slots=9)
    binding, tensors = make_binding(
        inputs,
        max_tokens=256,
        max_seqs=4,
        final_stride=3,
        checkpoint_export=True,
    )
    assert binding.final_state_indices.stride() == (3,)
    _run(binding, inputs)
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run(binding, inputs)
    tensors["recurrent_state"].copy_(inputs["pool"])
    binding.output.fill_(float("nan"))
    binding.scratch.fill_(0xFF)
    addresses = (tensors["recurrent_state"].data_ptr(), binding.output.data_ptr(), binding.scratch.data_ptr())
    allocated_before = torch.cuda.memory_allocated(device)
    graph.replay()
    torch.cuda.synchronize(device)
    assert torch.cuda.memory_allocated(device) == allocated_before
    assert addresses == (tensors["recurrent_state"].data_ptr(), binding.output.data_ptr(), binding.scratch.data_ptr())
    _assert_op_matches_oracle(binding, tensors, inputs)
    assert torch.isnan(binding.output[inputs["num_tokens"] :].float()).all()
    graph.reset()

def test_op_three_window_ring_reuse_is_capture_safe() -> None:
    """Three populated windows preserve results and fixed storage on replay."""
    from ..conftest import require_b12x
    from b12x.sequence.kda_prefill import KdaPrefillConfig

    device = require_b12x()
    inputs = make_inputs(lengths=[160], heads=2, seed=74, device=device, state_slots=4)
    config = KdaPrefillConfig(v_split=64, k_split=1, stages=3, window_tiles=4)
    binding, tensors = make_binding(inputs, max_tokens=160, max_seqs=1, config=config)

    assert binding._state.window_tiles == 4
    assert binding._state.launched_windows(inputs["num_tokens"], inputs["num_seqs"]) == 3
    scratch_capacity = binding.scratch.numel()
    addresses = (
        tensors["recurrent_state"].data_ptr(),
        binding.output.data_ptr(),
        binding.scratch.data_ptr(),
        binding.ws.data_ptr(),
    )

    _run(binding, inputs)
    torch.cuda.synchronize(device)
    _assert_op_matches_oracle(binding, tensors, inputs)

    tensors["recurrent_state"].copy_(inputs["pool"])
    binding.output.zero_()
    binding.scratch.zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run(binding, inputs)

    tensors["recurrent_state"].copy_(inputs["pool"])
    binding.output.fill_(float("nan"))
    binding.scratch.fill_(0xFF)
    allocated_before = torch.cuda.memory_allocated(device)
    graph.replay()
    torch.cuda.synchronize(device)

    assert torch.cuda.memory_allocated(device) == allocated_before
    assert binding.scratch.numel() == scratch_capacity
    assert addresses == (
        tensors["recurrent_state"].data_ptr(),
        binding.output.data_ptr(),
        binding.scratch.data_ptr(),
        binding.ws.data_ptr(),
    )
    _assert_op_matches_oracle(binding, tensors, inputs)
    graph.reset()


def test_op_cuda_graph_replay_uses_device_metadata() -> None:
    from ..conftest import require_b12x

    device = require_b12x()
    first = make_inputs(lengths=[100, 30], heads=2, seed=70, device=device, state_slots=9)
    binding, tensors = make_binding(first, max_tokens=256, max_seqs=4)
    _run(binding, first)
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run(binding, first)
    second = make_inputs(lengths=[17, 64, 1], heads=2, seed=71, device=device, state_slots=9, initial=[0, 1, 2], final=[3, 4, 5])
    for name in ("q", "k", "v", "raw_g", "raw_beta"):
        tensors[name].zero_()
        tensors[name][: second["num_tokens"]] = second[name]
    tensors["cu_seqlens"].zero_()
    tensors["cu_seqlens"][:4] = second["cu_seqlens"]
    tensors["initial_state_indices"][:3] = second["initial"]
    tensors["final_state_indices"][:3] = second["final"]
    tensors["num_seqs"].fill_(3)
    tensors["num_tokens"].fill_(second["num_tokens"])
    tensors["recurrent_state"].copy_(second["pool"])
    tensors["A_log"].copy_(second["A_log"])
    tensors["dt_bias"].copy_(second["dt_bias"])
    graph.replay()
    torch.cuda.synchronize(device)
    _assert_op_matches_oracle(binding, tensors, second)
    graph.reset()


def test_op_read_only_inputs_are_immutable() -> None:
    from ..conftest import require_b12x

    device = require_b12x()
    inputs = make_inputs(lengths=[50, 60], heads=2, seed=73, device=device)
    binding, tensors = make_binding(inputs, max_tokens=128, max_seqs=4)
    read_only = {
        name: tensors[name].clone()
        for name in (
            "q", "k", "v", "raw_g", "raw_beta", "A_log", "dt_bias", "cu_seqlens",
            "initial_state_indices", "final_state_indices", "checkpoint_state_indices",
            "checkpoint_offsets", "num_seqs", "num_tokens",
        )
    }
    _run(binding, inputs)
    torch.cuda.synchronize(device)
    for name, before in read_only.items():
        torch.testing.assert_close(tensors[name], before, rtol=0, atol=0)


def test_op_accepts_strided_views() -> None:
    from ..conftest import require_b12x
    from b12x.sequence.kda_prefill import _impl as impl

    device = require_b12x()
    inputs = make_inputs(lengths=[40, 24], heads=2, seed=74, device=device)
    heads, tokens = 2, inputs["num_tokens"]
    packed = torch.zeros(tokens, 3 * heads * HEAD_DIM + 64, dtype=torch.bfloat16, device=device)
    packed[:, : heads * HEAD_DIM] = inputs["q"].reshape(tokens, -1)
    packed[:, heads * HEAD_DIM : 2 * heads * HEAD_DIM] = inputs["k"].reshape(tokens, -1)
    packed[:, 2 * heads * HEAD_DIM : 3 * heads * HEAD_DIM] = inputs["v"].reshape(tokens, -1)
    q = packed[:, : heads * HEAD_DIM].view(tokens, heads, HEAD_DIM)
    k = packed[:, heads * HEAD_DIM : 2 * heads * HEAD_DIM].view(tokens, heads, HEAD_DIM)
    v = packed[:, 2 * heads * HEAD_DIM : 3 * heads * HEAD_DIM].view(tokens, heads, HEAD_DIM)
    beta_storage = torch.zeros(tokens, 2 * heads + 3, dtype=torch.bfloat16, device=device)
    beta_storage[:, 1 : 2 * heads + 1 : 2] = inputs["raw_beta"]
    raw_beta = beta_storage[:, 1 : 2 * heads + 1 : 2]
    final_storage = torch.zeros((2, 3), dtype=torch.int32, device=device)
    final_storage[:, 0] = inputs["final"]
    final_state_indices = final_storage[:, 0]
    assert (
        not q.is_contiguous()
        and not raw_beta.is_contiguous()
        and not final_state_indices.is_contiguous()
    )
    caps = impl.Caps(device=device, max_tokens=tokens, max_seqs=2, max_state_slots=8, heads=heads)
    pool = inputs["pool"].clone()
    binding = _prepare_tensors(caps, {
        "q": q, "k": k, "v": v, "raw_g": inputs["raw_g"], "raw_beta": raw_beta,
        "A_log": inputs["A_log"], "dt_bias": inputs["dt_bias"], "recurrent_state": pool,
        "cu_seqlens": inputs["cu_seqlens"], "initial_state_indices": inputs["initial"],
        "final_state_indices": final_state_indices,
        "checkpoint_state_indices": inputs["checkpoint_slots"],
        "checkpoint_offsets": inputs["checkpoint_offsets"],
        "num_seqs": torch.tensor([2], dtype=torch.int32, device=device),
        "num_tokens": torch.tensor([tokens], dtype=torch.int32, device=device),
        "output": torch.zeros(tokens, heads, HEAD_DIM, dtype=torch.bfloat16, device=device),
    }, lower_bound=-5.0)
    impl.run(binding, lower_bound=-5.0)
    torch.cuda.synchronize(device)
    expected_out, expected_pool = run_oracle(inputs)
    assert_kda_close("out", expected_out, binding.output, ratio=1e-2)
    for slot in inputs["final"].tolist():
        assert_kda_close(f"state[{slot}]", expected_pool[slot], pool[slot], ratio=5e-3)


def test_op_capacity_specialization_is_reused_under_frozen_resolution() -> None:
    from ..conftest import require_b12x
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.sequence._shared.delta_prefill import _cute_kernels as kernels

    device = require_b12x()
    small = make_inputs(lengths=[1], heads=2, seed=75, device=device, state_slots=8)
    binding, tensors = make_binding(small, max_tokens=128, max_seqs=4)
    _run(binding, small)
    torch.cuda.synchronize(device)
    launchers = (
        kernels._PROLOGUE_CACHE[kernels._prologue_key(binding)],
        kernels._PREPARE_CACHE[kernels._prepare_key(binding)],
        kernels._RECURRENCE_CACHE[kernels._recurrence_key(binding)],
    )
    with kernel_resolution_guard('kda prefill reuse test'):
        for lengths in ([128], [40, 41, 47]):
            live = make_inputs(lengths=lengths, heads=2, seed=76, device=device, state_slots=8)
            for name in ("q", "k", "v", "raw_g", "raw_beta"):
                tensors[name].zero_()
                tensors[name][: live["num_tokens"]] = live[name]
            tensors["cu_seqlens"].zero_()
            tensors["cu_seqlens"][: len(lengths) + 1] = live["cu_seqlens"]
            tensors["initial_state_indices"][: len(lengths)] = live["initial"]
            tensors["final_state_indices"][: len(lengths)] = live["final"]
            tensors["num_seqs"].fill_(len(lengths))
            tensors["num_tokens"].fill_(live["num_tokens"])
            tensors["recurrent_state"].copy_(live["pool"])
            tensors["A_log"].copy_(live["A_log"])
            tensors["dt_bias"].copy_(live["dt_bias"])
            _run(binding, live)
            torch.cuda.synchronize(device)
            _assert_op_matches_oracle(binding, tensors, live)
    assert launchers == (
        kernels._PROLOGUE_CACHE[kernels._prologue_key(binding)],
        kernels._PREPARE_CACHE[kernels._prepare_key(binding)],
        kernels._RECURRENCE_CACHE[kernels._recurrence_key(binding)],
    )


def test_op_state_slot_offset_past_int32_boundary() -> None:
    from ..conftest import require_b12x
    from b12x.sequence.kda_prefill import _impl as impl

    device = require_b12x()
    heads = 1
    inputs = make_inputs(lengths=[40], heads=heads, seed=77, device=device, state_slots=3, checkpoint=[(16, 2)])
    slot_stride = HEAD_DIM * HEAD_DIM + 2048
    tail_slot = (1 << 31) // slot_stride + 1
    storage = torch.empty(tail_slot * slot_stride + slot_stride, dtype=torch.float32, device=device)
    pool = torch.as_strided(storage, (tail_slot + 1, heads, HEAD_DIM, HEAD_DIM), (slot_stride, HEAD_DIM * HEAD_DIM, HEAD_DIM, 1))
    pool[tail_slot].copy_(inputs["pool"][0])
    pool[tail_slot - 1].copy_(inputs["pool"][1])
    pool[tail_slot - 2].copy_(inputs["pool"][2])
    tokens = inputs["num_tokens"]
    caps = impl.Caps(device=device, max_tokens=tokens, max_seqs=1, max_state_slots=tail_slot + 1, heads=heads, checkpoint_export=True)
    binding = _prepare_tensors(caps, {
        "q": inputs["q"], "k": inputs["k"], "v": inputs["v"], "raw_g": inputs["raw_g"],
        "raw_beta": inputs["raw_beta"], "A_log": inputs["A_log"], "dt_bias": inputs["dt_bias"],
        "recurrent_state": pool, "cu_seqlens": inputs["cu_seqlens"],
        "initial_state_indices": torch.tensor([tail_slot], dtype=torch.int64, device=device),
        "final_state_indices": torch.tensor([tail_slot - 1], dtype=torch.int64, device=device),
        "checkpoint_state_indices": torch.tensor([tail_slot - 2], dtype=torch.int64, device=device),
        "checkpoint_offsets": torch.tensor([16], dtype=torch.int32, device=device),
        "num_seqs": torch.tensor([1], dtype=torch.int32, device=device),
        "num_tokens": torch.tensor([tokens], dtype=torch.int32, device=device),
        "output": torch.zeros(tokens, heads, HEAD_DIM, dtype=torch.bfloat16, device=device),
    }, lower_bound=-5.0)
    impl.run(binding, lower_bound=-5.0)
    torch.cuda.synchronize(device)
    compact = make_inputs(lengths=[40], heads=heads, seed=77, device=device, state_slots=3, checkpoint=[(16, 2)])
    compact_binding, compact_tensors = make_binding(compact, max_tokens=tokens, max_seqs=1, checkpoint_export=True)
    _run(compact_binding, compact)
    torch.cuda.synchronize(device)
    torch.testing.assert_close(binding.output, compact_binding.output, rtol=0, atol=0)
    torch.testing.assert_close(pool[tail_slot - 1], compact_tensors["recurrent_state"][1], rtol=0, atol=0)
    torch.testing.assert_close(pool[tail_slot - 2], compact_tensors["recurrent_state"][2], rtol=0, atol=0)
    del storage, pool
    torch.cuda.empty_cache()


def test_op_zero_tokens_copies_states_only() -> None:
    from ..conftest import require_b12x

    device = require_b12x()
    inputs = make_inputs(lengths=[0, 0], heads=2, seed=78, device=device)
    binding, tensors = make_binding(inputs, max_tokens=32, max_seqs=2)
    binding.output.fill_(float("nan"))
    _run(binding, inputs)
    torch.cuda.synchronize(device)
    assert torch.isnan(binding.output.float()).all()
    for request in range(2):
        torch.testing.assert_close(
            tensors["recurrent_state"][int(inputs["final"][request])],
            inputs["pool"][int(inputs["initial"][request])], rtol=0, atol=0,
        )
