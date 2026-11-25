# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""``chunk_gated_delta_product(..., backend="cake")`` on SM90: state pools with write slots and BF16 / FP16 state.

A BF16 / FP16 state, or ``output_state_indices``, runs the V6 pool programs with the write slot table (an FP32, a
BF16 and an FP16 pool, each with its warm-up split twin): the split twin when the V6 split plan pays, else the program
over the ``N x H`` tiles.  The cases mirror ``test_cake_gdp_prefill_state.py`` (SM100 / SM103):

* An FP32 pool call (permuted read and write slots, a padded slot stride, a read slot of -1 or past the pool, a
  write slot of -1 or past the pool, disjoint or in-place pools, int32 or int64 slots) equals bit for bit the dense
  call on the gathered rows; an int64 slot outside the int32 range is a slot outside the pool.
* A BF16 / FP16 call is the twin of the FP32 call on the exactly widened state: the output equal bit for bit, the
  stored state the twin's rounded to nearest even or, with ``rand_seed``, the Philox model of
  ``test_cake_gdp_prefill_state.py`` (the software rounding of SM90 is the model of the hardware ``cvt.rs``), with no
  mismatch, on the read slot's counter or, without an initial state, the write slot's; also past 2^32 elements.
* A dense BF16 / FP16 call runs on identity slots and a call without an initial state on read slots -1; both are
  graph-safe; ``g=None`` / ``beta=None`` are filled with ones (within FlashInfer's tolerance of the FP32 unit-gate /
  unit-beta programs, not bit for bit: another program).
* A seed with a dense FP16 state rounds it stochastically on its identity slots.
* Mixed dtypes, a bad pool layout, a bad seed and a seed with a per-sequence state that would need a copy are
  refused.
"""

from __future__ import annotations

import pytest
import torch

import flashinfer
import flashinfer.cake_gdp_backend as backend
from flashinfer.cake_gdp_backend import cake_jit as loader
from flashinfer.jit.env import has_flashinfer_jit_cache
from flashinfer.utils import get_compute_capability
from tests.gdn.test_helpers.cudnn_linear_attention import rel_err
from tests.gdp.test_cake_gdp_prefill_state import (
    DTYPES,
    HEAD,
    K,
    N_HH,
    SEED,
    STRADDLE,
    TOLERANCE,
    V,
    _call,
    _capture,
    _equal_bits,
    _gathered,
    _inputs,
    _mismatches,
    _model,
    _pool,
    _slots,
    _valid,
)

SUPPORTED_CAPABILITIES = ((9, 0),)
# The state dtypes whose SM90 programs round stochastically with ``rand_seed``.
SR_DTYPES = ("bf16", "f16")


def _device() -> torch.device:
    return torch.device("cuda:0")


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and get_compute_capability(_device()) in SUPPORTED_CAPABILITIES
        and backend.is_cake_gdp_prefill_supported(_device())
    )


pytestmark = pytest.mark.skipif(
    not _supported(),
    reason="the Cake GDP SM90 pool programs need an SM90 device with exported routes",
)


def _arch() -> str:
    return loader.arch_for_capability(get_compute_capability(_device()))


def _pool_programs() -> dict[tuple[str, bool], str]:
    """(state dtype, split) -> the pool program with the write slot table on this device's route."""
    arch = _arch()
    return {
        (
            str(loader.MODULES[name]["build"].get("state_dtype", "f32")),
            bool(loader.MODULES[name]["build"].get(loader.PREFILL_SPLIT_FLAG)),
        ): name
        for name in loader.ROUTES[arch][loader.PREFILL_ROUTE]
        if loader.MODULES[name]["select_rule"].get("use_output_state_indices")
    }


@pytest.fixture(autouse=not has_flashinfer_jit_cache(), scope="module")
def warmup_jit():
    if _supported():
        arch = _arch()
        prefixes = tuple(
            f"{loader.library_name(name, arch)}_"
            for name in loader.ROUTES[arch][loader.PREFILL_ROUTE]
        )
        specs = [
            spec
            for spec in loader.specs_for_arch(arch)
            if any(spec.name.startswith(p) for p in prefixes)
        ]
        flashinfer.jit.build_jit_specs(specs, verbose=False)
    yield


@pytest.fixture
def launched(monkeypatch):
    """The program names of the launches the test makes."""
    names: list[str] = []
    original = loader.run_module

    def recording(name, values, **kwargs):
        names.append(name)
        return original(name, values, **kwargs)

    monkeypatch.setattr(loader, "run_module", recording)
    return names


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------


def _split(lengths, H) -> bool:
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    return loader.prefill_v6_sm90_plan(
        cu,
        num_heads=H,
        num_householder=N_HH,
        sm_count=loader.sm_count(_device().index or 0),
        total_tokens=cu[-1],
    ).split


# (id, H, lengths): a batch the V6 program runs over its N x H tiles and one whose V6 split plan pays (an empty
# sequence beside the balanced one).
BATCHES = {
    "serial": (8, [1, 65, 193, 300]),
    "split": (24, [1, 0, 8192, 65]),
}
CASES = ("disjoint", "inplace", "int64")


def _problem(batch, case, dtype, *, seed=0, init=True, pad=12):
    """The call of a slot case: ``(inputs, kwargs, reads, writes)``.  ``disjoint``: separate pools, explicit write
    slots; ``inplace``: one pool, ``output_state_indices`` the read slots; ``int64``: ``disjoint`` with int64
    tables.  Every case passes ``output_state_indices`` (so an FP32 case runs the pool program too)."""
    H, lengths = BATCHES[batch]
    N = len(lengths)
    rows = N + 3
    inputs = _inputs(lengths, H, seed=seed)
    reads, writes = _slots(N, rows, seed)
    index = torch.int64 if case == "int64" else torch.int32
    initial = _pool(rows, H, DTYPES[dtype], pad=pad, seed=seed + 1)
    if case == "inplace":
        writes = list(reads)
        output = initial
    else:
        output = _pool(rows, H, DTYPES[dtype], pad=pad, seed=seed + 2)
    kwargs = {
        "initial_state": initial if init else None,
        "output_state": output,
        "output_final_state": True,
        "state_indices": torch.tensor(reads, dtype=index, device=_device()),
        "output_state_indices": torch.tensor(writes, dtype=index, device=_device()),
    }
    return inputs, kwargs, reads, writes


def _twin(inputs, kwargs):
    """The FP32 pool call on the exactly widened 16-bit pools (the same slots, the same launch plan)."""
    initial, output = kwargs["initial_state"], kwargs["output_state"]
    same = initial is not None and output.data_ptr() == initial.data_ptr()
    initial32 = None if initial is None else initial.float()
    output32 = initial32 if same else output.float()
    return _call(
        inputs,
        **{
            **kwargs,
            "initial_state": initial32,
            "output_state": output32,
            "rand_seed": None,
        },
    )


# ---------------------------------------------------------------------------
# the route
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["f32", "bf16", "f16"])
@pytest.mark.parametrize("batch", sorted(BATCHES))
def test_every_pool_program_is_launched(launched, dtype, batch):
    """Each call is one launch of the pool program of its state dtype: the split twin where the V6 split plan pays."""
    H, lengths = BATCHES[batch]
    split = _split(lengths, H)
    assert split == (batch == "split")
    inputs, kwargs, _, _ = _problem(batch, "disjoint", dtype, seed=1)
    _call(inputs, **kwargs)
    assert launched == [_pool_programs()[(dtype, split)]]
    # a dense 16-bit call and a pool call without an initial state run the same program
    if dtype != "f32":
        N = len(lengths)
        launched.clear()
        _call(
            inputs,
            initial_state=_pool(N, H, DTYPES[dtype], seed=2),
            output_state=torch.empty(N, H, V, K, dtype=DTYPES[dtype], device=_device()),
            output_final_state=True,
        )
        assert launched == [_pool_programs()[(dtype, split)]]
    launched.clear()
    _call(inputs, **{**kwargs, "initial_state": None})
    assert launched == [_pool_programs()[(dtype, split)]]


def test_an_f32_call_without_write_slots_keeps_its_programs(launched):
    """An FP32 pool without ``output_state_indices`` (contiguous: those programs take no padded slot stride) and a
    dense FP32 call run the FP32 programs of the route."""
    inputs, kwargs, _, _ = _problem("serial", "disjoint", "f32", seed=3, pad=0)
    pools = set(_pool_programs().values())
    _call(inputs, **{**kwargs, "output_state_indices": None})
    _call(inputs, initial_state=_gathered(kwargs["initial_state"], [0, 1, 2, 3], 8))
    assert len(launched) == 2 and not set(launched) & pools


# ---------------------------------------------------------------------------
# FP32 pools against the dense call
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("batch", sorted(BATCHES))
def test_f32_pool_is_the_dense_call_on_the_gathered_rows(batch, case):
    H, _ = BATCHES[batch]
    inputs, kwargs, reads, writes = _problem(batch, case, "f32", seed=3)
    initial, output = kwargs["initial_state"], kwargs["output_state"]
    dense_initial = _gathered(initial, reads, H)
    before = output.clone()
    ref_out, ref_state = _call(
        inputs, initial_state=dense_initial, output_final_state=True
    )
    out, state = _call(inputs, **kwargs)
    assert state is output
    assert _equal_bits(out, ref_out)
    rows = int(output.shape[0])
    written = set()
    for seq, w in enumerate(writes):
        if _valid(w, rows):
            assert _equal_bits(output[w], ref_state[seq]), (seq, w)
            written.add(w)
    for slot in range(rows):
        if slot not in written:
            assert _equal_bits(output[slot], before[slot]), slot


@pytest.mark.parametrize("batch", sorted(BATCHES))
def test_a_write_only_pool_starts_from_zero(batch):
    """Without an initial state (read slots -1) every sequence starts from zero and stores at its write slot."""
    inputs, kwargs, _, writes = _problem(batch, "disjoint", "f32", seed=4)
    ref_out, ref_state = _call(inputs, output_final_state=True)
    out, _ = _call(inputs, **{**kwargs, "initial_state": None})
    assert _equal_bits(out, ref_out)
    rows = int(kwargs["output_state"].shape[0])
    for seq, w in enumerate(writes):
        if _valid(w, rows):
            assert _equal_bits(kwargs["output_state"][w], ref_state[seq])


@pytest.mark.parametrize("dtype", ["f32", "bf16"])
def test_a_pool_without_a_final_state_stores_nothing(dtype):
    inputs, kwargs, _, _ = _problem("serial", "disjoint", dtype, seed=5)
    before = kwargs["output_state"].clone()
    ref = _call(inputs, **{**kwargs, "output_final_state": True})[0]
    kwargs["output_state"].copy_(before)
    out = _call(inputs, **{**kwargs, "output_final_state": False})
    assert torch.is_tensor(out) and _equal_bits(out, ref)
    assert _equal_bits(kwargs["output_state"], before)


@pytest.mark.parametrize("dtype", ["f32", "bf16"])
def test_int64_slots_outside_the_int32_range_are_outside_the_pool(dtype):
    """int64 slots 2^32, 2^32 + 3 and 1 - 2^32 would wrap onto rows 0, 3 and 1 in an int32 cast: they read a zero
    state and store nothing, bit for bit as -1 does."""
    inputs, kwargs, reads, writes = _problem("serial", "int64", dtype, seed=31)
    big = 1 << 32
    reads = [reads[0], big, 1 - big, reads[3]]
    writes = [big + 3, writes[1], big, writes[3]]
    output = kwargs["output_state"]
    before = output.clone()

    def run(r, w, index):
        out, state = _call(
            inputs,
            **{
                **kwargs,
                "state_indices": torch.tensor(r, dtype=index, device=_device()),
                "output_state_indices": torch.tensor(w, dtype=index, device=_device()),
            },
        )
        assert state is output
        return out, output.clone()

    got_out, got_state = run(reads, writes, torch.int64)
    output.copy_(before)
    fold = [
        [x if -(1 << 31) <= x < (1 << 31) else -1 for x in t] for t in (reads, writes)
    ]
    ref_out, ref_state = run(*fold, torch.int32)
    assert _equal_bits(got_out, ref_out)
    assert _equal_bits(got_state, ref_state)
    for slot in (0, 3):
        if slot not in fold[1]:
            assert _equal_bits(got_state[slot], before[slot]), slot


# ---------------------------------------------------------------------------
# BF16 / FP16: the twin, round to nearest even
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("batch", sorted(BATCHES))
@pytest.mark.parametrize("dtype", ["bf16", "f16"])
def test_16bit_store_is_the_rounded_twin(dtype, batch, case):
    inputs, kwargs, _, writes = _problem(batch, case, dtype, seed=6)
    output = kwargs["output_state"]
    before = output.clone()
    twin_out, twin_state = _twin(inputs, kwargs)
    out, state = _call(inputs, **kwargs)
    assert state is output and state.dtype == DTYPES[dtype]
    assert _equal_bits(out, twin_out)
    rows = int(output.shape[0])
    written = set()
    for w in writes:
        if _valid(w, rows):
            assert _equal_bits(output[w], twin_state[w].to(DTYPES[dtype])), w
            written.add(w)
    for slot in range(rows):
        if slot not in written:
            assert _equal_bits(output[slot], before[slot]), slot


@pytest.mark.parametrize("batch", sorted(BATCHES))
@pytest.mark.parametrize("dtype", ["bf16", "f16"])
def test_a_dense_16bit_call_is_the_rounded_dense_call(dtype, batch):
    """A dense BF16 / FP16 call (per-sequence tensors, the pool program on identity slots) stores the dense FP32
    call's final state rounded to nearest even; without an initial state it starts from zero."""
    H, lengths = BATCHES[batch]
    N = len(lengths)
    inputs = _inputs(lengths, H, seed=7)
    initial = _pool(N, H, DTYPES[dtype], seed=8)
    output = torch.empty(N, H, V, K, dtype=DTYPES[dtype], device=_device())
    ref_out, ref_state = _call(
        inputs, initial_state=initial.float(), output_final_state=True
    )
    out, state = _call(
        inputs, initial_state=initial, output_state=output, output_final_state=True
    )
    assert state is output
    assert _equal_bits(out, ref_out)
    assert _equal_bits(state, ref_state.to(DTYPES[dtype]))
    ref_out, ref_state = _call(inputs, output_final_state=True)
    out, state = _call(inputs, output_state=output, output_final_state=True)
    assert _equal_bits(out, ref_out)
    assert _equal_bits(state, ref_state.to(DTYPES[dtype]))


def test_bf16_pool_stores_what_the_gather_cast_scatter_pattern_stores():
    """The BF16 pool call (in place, no write slots) against the dense FP32 call on the gathered, widened state, then
    ``.to(bfloat16)`` and a scatter."""
    H, lengths = 24, [1, 1, 8192, 65]
    assert _split(lengths, H)
    inputs = _inputs(lengths, H, seed=9)
    pool = _pool(16, H, torch.bfloat16, seed=10)
    slots = torch.tensor([11, 3, 7, 0], dtype=torch.int32, device=_device())
    expected = pool.clone()
    ref_out, ref_state = _call(
        inputs, initial_state=pool[slots.long()].float(), output_final_state=True
    )
    expected[slots.long()] = ref_state.to(torch.bfloat16)
    out, _ = _call(
        inputs,
        initial_state=pool,
        output_state=pool,
        output_final_state=True,
        state_indices=slots,
    )
    assert _equal_bits(out, ref_out)
    assert _equal_bits(pool, expected)


@pytest.mark.parametrize("dtype", ["bf16", "f16"])
def test_unit_gates_and_betas_are_filled_with_ones(launched, dtype):
    """``g=None`` / ``beta=None`` with a 16-bit state run the pool program on unit tensors: bit for bit the call with
    explicit ones.  That is the math of the FP32 unit-gate / unit-beta programs but not their program (the pool
    program runs its gate / beta path on the ones), so the bits may differ from theirs: from a zero state (those
    programs read no initial state) the output and every stored state agree with the dense FP32 unit-gate /
    unit-beta call within FlashInfer's tolerance."""
    inputs, kwargs, _, writes = _problem("serial", "disjoint", dtype, seed=11)
    output = kwargs["output_state"]
    rows = int(output.shape[0])
    before = output.clone()
    pools = set(_pool_programs().values())
    for name, flag in (("g", "unit_gates"), ("beta", "unit_beta")):
        output.copy_(before)
        ref_out, _ = _call({**inputs, name: torch.ones_like(inputs[name])}, **kwargs)
        ref_state = output.clone()
        output.copy_(before)
        out, _ = _call({**inputs, name: None}, **kwargs)
        assert _equal_bits(out, ref_out) and _equal_bits(output, ref_state), name
        # against the FP32 unit program, from a zero state (read slots -1)
        output.copy_(before)
        launched.clear()
        out, _ = _call({**inputs, name: None}, **{**kwargs, "initial_state": None})
        assert len(launched) == 1 and launched[0] in pools, launched
        launched.clear()
        unit_out, unit_state = _call({**inputs, name: None}, output_final_state=True)
        assert len(launched) == 1 and launched[0] not in pools, launched
        assert loader.MODULES[launched[0]]["select_rule"][flag], launched
        assert rel_err(out, unit_out) < TOLERANCE, name
        for seq, w in enumerate(writes):
            if _valid(w, rows):
                assert rel_err(output[w], unit_state[seq]) < TOLERANCE, (name, seq)


# ---------------------------------------------------------------------------
# stochastic rounding: the Philox model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rounds", [5, 10, 16])
@pytest.mark.parametrize("batch", sorted(BATCHES))
@pytest.mark.parametrize("dtype", SR_DTYPES)
def test_stochastic_store_matches_the_philox_model(dtype, batch, rounds):
    """Zero mismatches against the model on the twin's FP32 state: the read slot's counter where the call reads a
    valid read slot, the write slot's where the read slot is -1 or past the pool."""
    inputs, kwargs, reads, writes = _problem(batch, "disjoint", dtype, seed=14)
    seed = torch.tensor([SEED], dtype=torch.int64, device=_device())
    _, twin_state = _twin(inputs, kwargs)
    _, state = _call(inputs, **kwargs, rand_seed=seed, philox_rounds=rounds)
    assert (
        _mismatches(state, twin_state, kwargs, reads, writes, SEED, rounds, dtype) == 0
    )
    assert _mismatches(
        state, twin_state, kwargs, reads, writes, SEED, rounds + 1, dtype
    )
    assert _mismatches(
        state, twin_state, kwargs, reads, writes, SEED ^ 1, rounds, dtype
    )


@pytest.mark.parametrize("batch", sorted(BATCHES))
@pytest.mark.parametrize("dtype", SR_DTYPES)
def test_stochastic_store_without_an_initial_state_counts_from_the_write_slot(
    dtype, batch
):
    inputs, kwargs, reads, writes = _problem(batch, "disjoint", dtype, seed=15)
    kwargs["initial_state"] = None
    seed = torch.tensor([SEED], dtype=torch.int64, device=_device())
    _, twin_state = _twin(inputs, kwargs)
    _, state = _call(inputs, **kwargs, rand_seed=seed)
    assert _mismatches(state, twin_state, kwargs, reads, writes, SEED, 10, dtype) == 0


@pytest.mark.parametrize("dtype", SR_DTYPES)
def test_rounds_and_seeds_store_other_bits_and_repeat_exactly(dtype):
    inputs, kwargs, _, _ = _problem("serial", "disjoint", dtype, seed=16)
    seed = torch.tensor([SEED], dtype=torch.int64, device=_device())
    stored = {}
    for rounds in (5, 10, 16):
        _call(inputs, **kwargs, rand_seed=seed, philox_rounds=rounds)
        stored[rounds] = kwargs["output_state"].clone()
    assert not _equal_bits(stored[5], stored[10])
    assert not _equal_bits(stored[10], stored[16])
    for _ in range(2):
        _call(inputs, **kwargs, rand_seed=seed, philox_rounds=10)
        assert _equal_bits(kwargs["output_state"], stored[10])
    # philox_rounds without a seed is not read: round to nearest even
    _call(inputs, **kwargs, philox_rounds=3)
    rne = kwargs["output_state"].clone()
    _call(inputs, **kwargs)
    assert _equal_bits(kwargs["output_state"], rne)
    assert not _equal_bits(rne, stored[10])


@pytest.mark.parametrize("init", [True, False], ids=["read-slot", "write-slot"])
@pytest.mark.parametrize("dtype", SR_DTYPES)
def test_the_counter_past_2_32_elements(dtype, init):
    H, lengths = 1, [1, 700]
    inputs = _inputs(lengths, H, seed=17)
    flat = torch.zeros(STRADDLE + HEAD, dtype=DTYPES[dtype], device=_device())
    pool = flat.as_strided((2, H, V, K), (STRADDLE, HEAD, K, 1))
    g = torch.Generator(device=_device()).manual_seed(18)
    for slot in (0, 1):
        pool[slot].copy_((0.1 * torch.randn(H, V, K, generator=g, device=_device())))
    reads, writes = [0, 1], [0, 1]
    kwargs = {
        "initial_state": pool if init else None,
        "output_state": pool,
        "output_final_state": True,
        "state_indices": torch.tensor(reads, dtype=torch.int32, device=_device()),
    }
    dense = _gathered(pool, reads, H) if init else None
    _, ref_state = _call(inputs, initial_state=dense, output_final_state=True)
    twin_state = torch.empty(2, H, V, K, device=_device())
    twin_state[:] = ref_state
    seed = torch.tensor([SEED], dtype=torch.int64, device=_device())
    _call(inputs, **kwargs, rand_seed=seed)
    assert _mismatches(pool, twin_state, kwargs, reads, writes, SEED, 10, dtype) == 0
    want = _model(twin_state[1], STRADDLE, SEED, 10, dtype, low_word_only=True)
    have = pool[1].contiguous().view(torch.int16).cpu().numpy().view("uint16")
    assert (have != want).any()
    del flat, pool


# ---------------------------------------------------------------------------
# empty sequences
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["f32", "bf16", "f16"])
def test_a_batch_of_empty_sequences_copies_the_valid_slots(dtype):
    """Every sequence empty: the initial row at the read slot (zero for -1 or past the pool) is stored as it is at
    the write slot; -1 or past the pool stores nothing."""
    H, N, rows = 4, 4, 7
    inputs = _inputs([0] * N, H, seed=19)
    initial = _pool(rows, H, DTYPES[dtype], pad=4, seed=20)
    output = _pool(rows, H, DTYPES[dtype], pad=4, seed=21)
    reads, writes = [2, -1, rows, 5], [6, 0, 3, -1]
    before = output.clone()
    out, state = _call(
        inputs,
        initial_state=initial,
        output_state=output,
        output_final_state=True,
        state_indices=torch.tensor(reads, dtype=torch.int32, device=_device()),
        output_state_indices=torch.tensor(writes, dtype=torch.int32, device=_device()),
    )
    assert out.shape[0] == 0 and state is output
    assert _equal_bits(output[6], initial[2])
    assert torch.count_nonzero(output[0].float()) == 0
    assert torch.count_nonzero(output[3].float()) == 0
    for slot in (1, 2, 4, 5):
        assert _equal_bits(output[slot], before[slot])


@pytest.mark.parametrize("final", [False, True], ids=["no-final", "final"])
@pytest.mark.parametrize("initial", [False, True], ids=["no-initial", "initial"])
@pytest.mark.parametrize("dtype", ["bf16", "f16"])
def test_no_sequence_with_a_dense_state_launches_nothing(
    launched, dtype, initial, final
):
    if not initial and not final:
        pytest.skip("no state tensor")
    H, T = 2, 5
    device, state_dtype = _device(), DTYPES[dtype]

    def empty():
        return torch.empty(0, H, V, K, dtype=state_dtype, device=device)

    out_state = empty() if final else None
    result = _call(
        {
            "q": torch.zeros(T, H, K, dtype=torch.bfloat16, device=device),
            "k": torch.zeros(T * N_HH, H, K, dtype=torch.bfloat16, device=device),
            "v": torch.zeros(T * N_HH, H, V, dtype=torch.bfloat16, device=device),
            "num_householder": N_HH,
            "cu_seqlens": torch.zeros(1, dtype=torch.int32, device=device),
        },
        initial_state=empty() if initial else None,
        output_state=out_state,
        output_final_state=final,
    )
    out = result[0] if final else result
    assert tuple(out.shape) == (T, H, V) and not launched
    if final:
        assert result[1] is out_state


# ---------------------------------------------------------------------------
# CUDA graphs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch", sorted(BATCHES))
@pytest.mark.parametrize("dtype", SR_DTYPES)
def test_a_graph_reads_the_seed_and_the_slots_at_replay(dtype, batch):
    inputs, kwargs, reads, _ = _problem(batch, "disjoint", dtype, seed=22)
    seed = torch.tensor([SEED], dtype=torch.int64, device=_device())
    out = torch.empty(
        int(inputs["q"].shape[0]),
        BATCHES[batch][0],
        V,
        dtype=torch.bfloat16,
        device=_device(),
    )
    initial0 = kwargs["initial_state"].clone()
    graph = _capture(lambda: _call(inputs, **kwargs, output=out, rand_seed=seed))
    seed.fill_(SEED ^ 0x5555)
    new_reads = [r if r != reads[0] else -1 for r in reads[::-1]]
    kwargs["state_indices"].copy_(torch.tensor(new_reads, device=_device()))
    kwargs["initial_state"].copy_(initial0)
    graph.replay()
    torch.cuda.synchronize()
    replayed_out, replayed_state = out.clone(), kwargs["output_state"].clone()
    kwargs["initial_state"].copy_(initial0)
    eager_out, eager_state = _call(inputs, **kwargs, rand_seed=seed)
    assert _equal_bits(replayed_out, eager_out)
    assert _equal_bits(replayed_state, eager_state)


@pytest.mark.parametrize(
    "tables,initial",
    [("_IDENTITY_SLOTS", True), ("_NO_SLOTS", False)],
    ids=["identity", "no-read"],
)
def test_slot_tables_made_in_a_capture_are_the_graphs_own(tables, initial):
    """A dense BF16 call captured with no cached slot table (the identity slots; without an initial state also the
    -1 read slots) records its own table in the graph; an eager call afterwards makes a complete one; both store the
    same."""
    H, lengths = 8, [1, 65, 193, 300]
    N = len(lengths)
    inputs = _inputs(lengths, H, seed=23)
    initial_state = _pool(N, H, torch.bfloat16, seed=24) if initial else None
    output = torch.empty(N, H, V, K, dtype=torch.bfloat16, device=_device())
    out = torch.empty(sum(lengths), H, V, dtype=torch.bfloat16, device=_device())

    def call():
        return _call(
            inputs,
            initial_state=initial_state,
            output_state=output,
            output_final_state=True,
            output=out,
        )

    stream = torch.cuda.Stream(device=_device())
    stream.wait_stream(torch.cuda.current_stream(_device()))
    with torch.cuda.stream(stream):
        call()
    stream.synchronize()
    expected_out, expected_state = out.clone(), output.clone()
    cache = getattr(backend, tables)
    cache.clear()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        call()
    assert not cache
    out.fill_(float("nan"))
    output.fill_(float("nan"))
    call()  # eager, before any replay
    assert _equal_bits(out, expected_out) and _equal_bits(output, expected_state)
    out.fill_(float("nan"))
    output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert _equal_bits(out, expected_out) and _equal_bits(output, expected_state)


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------


def _small():
    inputs = _inputs([64, 3], 4, seed=25)
    pool = _pool(4, 4, torch.bfloat16, seed=26)
    slots = torch.tensor([1, 2], dtype=torch.int32, device=_device())
    return inputs, pool, slots


def test_a_seed_with_a_dense_float16_state_rounds_stochastically(launched):
    """A seed with a dense FP16 state (per-sequence tensors read in place on identity slots) runs the FP16 pool
    program and stores the Philox model of the dense FP32 call's final state, on each sequence's row of
    ``initial_state`` (zero mismatches); the output is the FP32 call's bit for bit."""
    H, lengths = 4, [64, 3]
    inputs = _inputs(lengths, H, seed=25)
    initial = _pool(len(lengths), H, torch.float16, seed=26)
    output = torch.empty_like(initial)
    seed = torch.tensor([SEED], dtype=torch.int64, device=_device())
    ref_out, ref_state = _call(
        inputs, initial_state=initial.float(), output_final_state=True
    )
    launched.clear()
    out, state = _call(
        inputs,
        initial_state=initial,
        output_state=output,
        output_final_state=True,
        rand_seed=seed,
    )
    assert state is output
    assert launched == [_pool_programs()[("f16", False)]]
    assert _equal_bits(out, ref_out)
    rows = list(range(len(lengths)))
    kwargs = {"initial_state": initial, "output_state": output}
    assert _mismatches(output, ref_state, kwargs, rows, rows, SEED, 10, "f16") == 0
    assert not _equal_bits(output, ref_state.half())


def test_mixed_dtypes_are_refused():
    inputs, pool, slots = _small()
    with pytest.raises(ValueError, match="one dtype"):
        _call(
            inputs,
            initial_state=pool,
            output_state=pool.float(),
            output_final_state=True,
            state_indices=slots,
        )
    with pytest.raises(ValueError, match="one dtype"):
        _call(
            inputs,
            initial_state=pool[:2].half(),
            output_state=torch.empty(
                2, 4, V, K, dtype=torch.bfloat16, device=_device()
            ),
            output_final_state=True,
        )
    with pytest.raises(ValueError, match="float16 output_state"):
        _call(inputs, initial_state=pool[:2].half(), output_final_state=True)


def test_the_pool_layout_is_checked():
    inputs, pool, slots = _small()
    padded_head = torch.empty(4, 4, V + 1, K, dtype=torch.bfloat16, device=_device())[
        :, :, :V
    ]
    with pytest.raises(ValueError, match="dense"):
        _call(
            inputs,
            initial_state=padded_head,
            output_state=padded_head,
            output_final_state=True,
            state_indices=slots,
        )
    odd = torch.empty(4 * (4 * HEAD + 2), dtype=torch.bfloat16, device=_device())
    odd_pool = odd.as_strided((4, 4, V, K), (4 * HEAD + 2, HEAD, K, 1))
    with pytest.raises(ValueError, match="multiple of 4"):
        _call(
            inputs,
            initial_state=odd_pool,
            output_state=odd_pool,
            output_final_state=True,
            state_indices=slots,
        )
    with pytest.raises(ValueError, match="output_state pool"):
        _call(inputs, initial_state=pool, output_final_state=True, state_indices=slots)
    with pytest.raises(ValueError, match="needs state_indices"):
        _call(
            inputs,
            initial_state=pool[:2],
            output_state=pool[:2],
            output_final_state=True,
            output_state_indices=slots,
        )


def test_a_seed_never_copies_a_per_sequence_state():
    """A per-sequence BF16 ``initial_state`` the pool program cannot read in place (a slot stride of ``H x 8192 + 2``
    elements, not a multiple of 4) is copied once without ``rand_seed`` and runs as its contiguous copy; with a seed
    it is refused, since the stochastic rounding counts from the base of the caller's tensor and a copy would change
    the random bits."""
    inputs, _, _ = _small()
    initial = _pool(2, 4, torch.bfloat16, pad=2, seed=27)
    assert initial.stride(0) == 4 * HEAD + 2
    seed = torch.tensor([1], dtype=torch.int64, device=_device())
    output = torch.empty(2, 4, V, K, dtype=torch.bfloat16, device=_device())
    with pytest.raises(ValueError, match="never copied"):
        _call(
            inputs,
            initial_state=initial,
            output_state=output,
            output_final_state=True,
            rand_seed=seed,
        )
    out, state = _call(
        inputs, initial_state=initial, output_state=output, output_final_state=True
    )
    ref_out, ref_state = _call(
        inputs,
        initial_state=initial.contiguous(),
        output_state=torch.empty_like(output),
        output_final_state=True,
    )
    assert state is output
    assert _equal_bits(out, ref_out) and _equal_bits(state, ref_state)


def test_the_seed_contract():
    inputs, pool, slots = _small()
    call = dict(
        initial_state=pool,
        output_state=pool,
        output_final_state=True,
        state_indices=slots,
    )
    seed = torch.tensor([1], dtype=torch.int64, device=_device())
    for rounds in (0, 4, 17):
        with pytest.raises(ValueError, match="philox_rounds"):
            _call(inputs, **call, rand_seed=seed, philox_rounds=rounds)
    with pytest.raises(TypeError, match="philox_rounds"):
        _call(inputs, **call, rand_seed=seed, philox_rounds=True)
    with pytest.raises(ValueError, match="int64"):
        _call(inputs, **call, rand_seed=seed.int())
    with pytest.raises(ValueError, match="one-element"):
        _call(inputs, **call, rand_seed=seed.repeat(2))
    with pytest.raises(ValueError, match="device"):
        _call(inputs, **call, rand_seed=seed.cpu())
    with pytest.raises(TypeError, match="rand_seed"):
        _call(inputs, **call, rand_seed=7)
    with pytest.raises(ValueError, match="float32 state"):
        _call(
            inputs,
            **{**call, "initial_state": pool.float(), "output_state": pool.float()},
            rand_seed=seed,
        )
    with pytest.raises(ValueError, match="float32 state"):
        _call(
            inputs,
            initial_state=pool[:2].float(),
            output_final_state=True,
            rand_seed=seed,
        )
