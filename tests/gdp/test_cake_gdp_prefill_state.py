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

"""``chunk_gated_delta_product(..., backend="cake")`` on SM100 / SM103: state pools and BF16 / FP16 state.

* An FP32 pool call (permuted read and write slots, a padded slot stride, a read slot of -1 or past the pool, a
  write slot of -1 or past the pool, disjoint or in-place pools, int32 or int64 slots) equals bit for bit the dense
  call on the gathered rows: the output, the stored rows at the valid write slots, and every other pool row untouched.
  An int64 slot outside the int32 range is a slot outside the pool (as -1), not the row its int32 cast wraps to.
* A BF16 / FP16 call is the twin of the FP32 call on the exactly widened state: the output is equal bit for bit and
  the stored state is the twin's final state rounded to nearest even (no seed) or, with ``rand_seed``, the rounding of
  an independent model of the stochastic stores (Philox-4x32 under the seed, 16 / 13 random bits per element added
  below the stored precision and truncated, as PTX ``cvt.rs``), with no mismatch, at 5, 10 and 16 rounds, on the
  read slot's counter or, when no initial state is read there, the write slot's; also past 2^32 elements.
* The BF16 pool call stores what the vLLM pattern stores (gather, ``.float()``, the dense FP32 call, ``.to(bf16)``,
  scatter), and agrees with cuDNN's BF16 pool within FlashInfer's tolerance.
* Graph capture: the seed and the slot tables are read on the device at each replay; a dense 16-bit call's
  identity slots made inside a capture are that graph's alone, and the cached table a graph captured stays valid
  after a larger batch grows the cache.
* A batch of no sequence with a dense state of any dtype launches nothing and returns the state tensor.
* No batch-independence case: a BF16 / FP16 split call rounds a state that depends on the launch plan, so a 16-bit
  call is held to its FP32 twin on the same batch, and the batch-independence tests of ``test_cake_gdp_prefill.py``
  run FP32 calls only.
* The arguments the call refuses.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import flashinfer
import flashinfer.cake_gdp_backend as backend
from flashinfer.cake_gdp_backend import cake_jit as loader
from flashinfer.gdp_prefill import chunk_gated_delta_product
from flashinfer.jit.env import has_flashinfer_jit_cache
from flashinfer.utils import get_compute_capability
from tests.gdn.test_helpers.cudnn_linear_attention import CUDNN_AVAILABLE, rel_err

K = 128
V = 64
HEAD = K * V
N_HH = 3
TOLERANCE = 1e-2
SUPPORTED_CAPABILITIES = ((10, 0), (10, 3))
DTYPES = {"f32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16}
MASK32 = 0xFFFF_FFFF
PHILOX_M0, PHILOX_M1 = 0xD2511F53, 0xCD9E8D57
PHILOX_W0, PHILOX_W1 = 0x9E3779B9, 0xBB67AE85


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
    reason="the Cake GDP Blackwell prefill needs an SM100 / SM103 device with exported routes",
)
needs_cudnn = pytest.mark.skipif(
    not CUDNN_AVAILABLE, reason="the cudnn python frontend is not installed"
)


@pytest.fixture(autouse=not has_flashinfer_jit_cache(), scope="module")
def warmup_jit():
    if _supported():
        arch = loader.arch_for_capability(get_compute_capability(_device()))
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


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------


def _arch() -> str:
    return loader.arch_for_capability(get_compute_capability(_device()))


def _plan(lengths, H):
    return loader.prefill_v6_plan(
        [0, *itertools.accumulate(lengths)],
        num_heads=H,
        num_householder=N_HH,
        sm_count=loader.sm_count(_device().index or 0),
        arch=_arch(),
    )


def _inputs(lengths, H, *, seed=0):
    """The model inputs of a batch (L2-normalized q / k, BF16 v, FP32 gates in (0, 1), FP32 betas)."""
    g = torch.Generator(device=_device()).manual_seed(seed)
    device = _device()
    T = sum(lengths)

    def unit(rows):
        x = torch.randn(rows, H, K, generator=g, device=device)
        return F.normalize(x, dim=-1).to(torch.bfloat16)

    cu = [0, *itertools.accumulate(lengths)]
    return {
        "q": unit(T),
        "k": unit(T * N_HH),
        "v": torch.randn(T * N_HH, H, V, generator=g, device=device).to(torch.bfloat16),
        "g": torch.exp(
            -F.softplus(0.5 * torch.randn(T, H, generator=g, device=device) - 2.0)
        ),
        "beta": torch.rand(T * N_HH, H, generator=g, device=device),
        "num_householder": N_HH,
        "cu_seqlens": torch.tensor(cu, dtype=torch.int32, device=device),
        "cu_seqlens_cpu": torch.tensor(cu, dtype=torch.int32),
    }


def _call(inputs, **kwargs):
    return chunk_gated_delta_product(**{**inputs, **kwargs}, backend="cake")


def _pool(rows, H, dtype, *, pad=0, seed=0, fill=True):
    """A ``[rows, H, 64, 128]`` state pool of ``dtype`` whose slot stride is padded by ``pad`` elements (a multiple of
    four), holding 0.1 x normal values representable in ``dtype``."""
    stride = H * HEAD + pad
    flat = torch.empty(rows * stride, dtype=dtype, device=_device())
    pool = flat.as_strided((rows, H, V, K), (stride, HEAD, K, 1))
    if fill:
        g = torch.Generator(device=_device()).manual_seed(seed)
        flat.copy_(
            (0.1 * torch.randn(rows * stride, generator=g, device=_device())).to(dtype)
        )
    return pool


# Slot layouts per batch of N sequences over pools of N + 3 rows: the read slots a permutation with one -1 and one
# past the pool, the write slots another permutation with one -1 and one past the pool (``None``: the read slots).
def _slots(N, rows, seed):
    g = torch.Generator().manual_seed(seed)
    reads = torch.randperm(rows, generator=g)[:N].tolist()
    writes = torch.randperm(rows, generator=g)[:N].tolist()
    if N >= 3:
        reads[1], reads[2] = -1, rows
        writes[0], writes[2] = -1, rows
    return reads, writes


# (id, H, lengths): one batch per build of the launch plan (an empty sequence on the split build: its state copy).
BATCHES = {
    "block": (8, [1, 65, 193, 300]),
    "split": (20, [0, 1, 0, 9000]),
}
CASES = ("disjoint", "inplace", "int64")


def _problem(batch, case, dtype, *, seed=0, init=True, pad=12):
    """The call of a slot case: ``(inputs, kwargs, reads, writes)`` with the pools and slot tables in ``kwargs``.
    ``disjoint``: separate initial / output pools, explicit write slots; ``inplace``: one pool, the read slots
    written; ``int64``: ``disjoint`` with int64 slot tables."""
    H, lengths = BATCHES[batch]
    N = len(lengths)
    rows = N + 3
    inputs = _inputs(lengths, H, seed=seed)
    reads, writes = _slots(N, rows, seed)
    index = torch.int64 if case == "int64" else torch.int32
    initial = _pool(rows, H, DTYPES[dtype], pad=pad, seed=seed + 1)
    if case == "inplace":
        writes = list(reads)
        output, out_slots = initial, None
    else:
        output = _pool(rows, H, DTYPES[dtype], pad=pad, seed=seed + 2)
        out_slots = torch.tensor(writes, dtype=index, device=_device())
    kwargs = {
        "initial_state": initial if init else None,
        "output_state": output,
        "output_final_state": True,
        "state_indices": torch.tensor(reads, dtype=index, device=_device()),
        "output_state_indices": out_slots,
    }
    return inputs, kwargs, reads, writes


def _gathered(pool, reads, H):
    """The dense FP32 initial state of the sequences: each valid read slot's rows widened, zeros otherwise."""
    rows = int(pool.shape[0])
    dense = torch.zeros(len(reads), H, V, K, device=_device())
    for seq, slot in enumerate(reads):
        if 0 <= slot < rows:
            dense[seq] = pool[slot].float()
    return dense


def _valid(slot, rows):
    return 0 <= slot < rows


def _bits(t):
    return t.view(torch.int16) if t.element_size() == 2 else t.view(torch.int32)


def _equal_bits(a, b):
    return torch.equal(_bits(a.contiguous()), _bits(b.contiguous()))


# ---------------------------------------------------------------------------
# the stochastic rounding model (Philox-4x32, cvt.rs)
# ---------------------------------------------------------------------------


def _philox4x32(seed, counters, rounds):
    bits = int(seed) & 0xFFFF_FFFF_FFFF_FFFF
    m32 = np.uint64(MASK32)
    k0, k1 = np.uint64(bits & MASK32), np.uint64(bits >> 32)
    c = np.asarray(counters, dtype=np.uint64).reshape(-1)
    c0, c1 = c & m32, c >> np.uint64(32)
    c2, c3 = np.zeros_like(c0), np.zeros_like(c0)
    for _ in range(rounds):
        p0, p1 = c0 * np.uint64(PHILOX_M0), c2 * np.uint64(PHILOX_M1)
        c0, c1, c2, c3 = (
            ((p1 >> np.uint64(32)) ^ c1 ^ k0) & m32,
            p1 & m32,
            ((p0 >> np.uint64(32)) ^ c3 ^ k1) & m32,
            p0 & m32,
        )
        k0 = (k0 + np.uint64(PHILOX_W0)) & m32
        k1 = (k1 + np.uint64(PHILOX_W1)) & m32
    return np.stack([c0, c1, c2, c3], axis=-1).astype(np.uint32)


def _row_bits(seed, rounds, row_offsets):
    """``[rows, K]`` 16-bit random halves of the stores of whole state rows at the element offsets ``row_offsets``:
    column ``k`` takes word ``(k % 4) // 2 + 2 (k // 64)`` of the draw at ``offset + (k % 64 & ~3)``, its low half
    for an even ``k`` and its high half for an odd one."""
    rows = np.asarray(row_offsets, dtype=np.uint64).reshape(-1)
    columns = np.arange(0, K // 2, 4, dtype=np.uint64)
    words = _philox4x32(seed, rows[:, None] + columns[None, :], rounds).reshape(
        len(rows), len(columns), 4
    )
    k = np.arange(K)
    picked = words[:, (k % 64) // 4, (k % 4) // 2 + 2 * (k // 64)]
    odd = (k % 2 == 1)[None, :]
    return np.where(odd, picked >> np.uint32(16), picked) & np.uint32(0xFFFF)


def _cvt_rs_f16(x, r13):
    a = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    sign, a = a & np.uint64(0x8000_0000), a & np.uint64(0x7FFF_FFFF)
    e = a >> np.uint64(23)
    m = (a & np.uint64(0x7F_FFFF)) | np.uint64(0x80_0000)
    shift = np.clip(np.int64(113) - e.astype(np.int64), 0, 63).astype(np.uint64)
    scaled = np.where(
        e >= np.uint64(113),
        a - np.uint64(112 << 23),
        np.where(e >= np.uint64(90), m >> shift, np.uint64(0)),
    )
    h = np.minimum(
        (scaled + np.asarray(r13, dtype=np.uint64)) >> np.uint64(13), np.uint64(0x7C00)
    )
    return ((sign >> np.uint64(16)) | h).astype(np.uint16)


def _cvt_rs_bf16(x, r16):
    b = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    sign, a = b & np.uint64(0x8000_0000), b & np.uint64(0x7FFF_FFFF)
    h = (a + np.asarray(r16, dtype=np.uint64)) >> np.uint64(16)
    return ((sign >> np.uint64(16)) | h).astype(np.uint16)


def _model(state32, base, seed, rounds, dtype, *, low_word_only=False):
    """The 16-bit bits ``[H, 64, 128]`` the stochastic store of the FP32 rows ``state32`` ``[H, 64, 128]`` takes on
    the counter base ``base`` (finite values); ``low_word_only``: the counters with their high word dropped."""
    H = int(state32.shape[0])
    offsets = (
        base + np.arange(H * V, dtype=np.int64) * K
    )  # head h, row v: base + h 8192 + v 128
    if low_word_only:
        offsets = offsets & MASK32
    r = _row_bits(seed, rounds, offsets)
    x = state32.reshape(H * V, K).cpu().numpy()
    if dtype == "bf16":
        out = _cvt_rs_bf16(x, r)
    else:
        out = _cvt_rs_f16(x, r & np.uint32(0x1FFF))
    return out.reshape(H, V, K)


def _mismatches(got16, twin32, kwargs, reads, writes, seed, rounds, dtype):
    """Elements of the stored 16-bit rows that differ from the model on the twin's FP32 rows."""
    initial, output = kwargs["initial_state"], kwargs["output_state"]
    rows_out = int(output.shape[0])
    bad = 0
    for r, w in zip(reads, writes, strict=True):
        if not _valid(w, rows_out):
            continue
        if initial is not None and _valid(r, int(initial.shape[0])):
            base = r * initial.stride(0)
        else:
            base = w * output.stride(0)
        want = _model(twin32[w], base, seed, rounds, dtype)
        have = got16[w].contiguous().view(torch.int16).cpu().numpy().view(np.uint16)
        bad += int((have != want).sum())
    return bad


def _twin(inputs, kwargs):
    """The FP32 call on the exactly widened 16-bit pools (the same slots; contiguous FP32 pools)."""
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
# FP32 pools against the dense call
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("batch", sorted(BATCHES))
def test_f32_pool_is_the_dense_call_on_the_gathered_rows(batch, case):
    H, lengths = BATCHES[batch]
    assert _plan(lengths, H).split == (batch == "split")
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


def test_a_write_only_pool_starts_from_zero():
    """Without an initial state every sequence starts from zero and stores at its slot."""
    inputs, kwargs, reads, writes = _problem("split", "disjoint", "f32", seed=4)
    ref_out, ref_state = _call(inputs, output_final_state=True)
    out, _ = _call(inputs, **{**kwargs, "initial_state": None})
    assert _equal_bits(out, ref_out)
    rows = int(kwargs["output_state"].shape[0])
    for seq, w in enumerate(writes):
        if _valid(w, rows):
            assert _equal_bits(kwargs["output_state"][w], ref_state[seq])


def test_a_pool_without_a_final_state_stores_nothing():
    inputs, kwargs, reads, _ = _problem("block", "disjoint", "bf16", seed=5)
    before = kwargs["output_state"].clone()
    out = _call(inputs, **{**kwargs, "output_final_state": False})
    assert torch.is_tensor(out)
    assert _equal_bits(kwargs["output_state"], before)


@pytest.mark.parametrize("dtype", ["f32", "bf16"])
def test_int64_slots_outside_the_int32_range_are_outside_the_pool(dtype):
    """int64 slots 2^32, 2^32 + 3 and 1 - 2^32 would wrap onto rows 0, 3 and 1 in an int32 cast: they read a zero
    state and store nothing, bit for bit as -1 does."""
    inputs, kwargs, reads, writes = _problem("block", "int64", dtype, seed=31)
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
    inputs, kwargs, reads, writes = _problem(batch, case, dtype, seed=6)
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
    call's final state rounded to nearest even."""
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


def test_bf16_pool_stores_what_the_gather_cast_scatter_pattern_stores():
    """The BF16 pool call against the dense FP32 call of the previous release on the gathered, widened state, then
    ``.to(bfloat16)`` and a scatter (the pattern a caller with a BF16 pool runs without pools)."""
    H, lengths = 10, [1, 1, 1, 4000]
    N = len(lengths)
    inputs = _inputs(lengths, H, seed=9)
    pool = _pool(16, H, torch.bfloat16, seed=10)
    slots = torch.tensor([11, 3, 7, 0], dtype=torch.int32, device=_device())
    expected = pool.clone()
    ref_out, ref_state = _call(
        inputs,
        initial_state=pool[slots.long()].float(),
        output_final_state=True,
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
    assert slots.numel() == N


@needs_cudnn
@pytest.mark.parametrize("inplace", [True, False], ids=["inplace", "disjoint"])
def test_bf16_pool_matches_cudnn(inplace):
    """cuDNN's BF16 pool (int32 unique slots in range) on the same inputs: FlashInfer's tolerance on the output and
    on the stored rows."""
    H, lengths = 10, [1, 1, 3000, 500]
    inputs = _inputs(lengths, H, seed=11)
    slots = torch.tensor([5, 0, 3, 6], dtype=torch.int32, device=_device())
    results = {}
    for name in ("cake", "cudnn"):
        pool = _pool(8, H, torch.bfloat16, seed=12)
        output = pool if inplace else _pool(8, H, torch.bfloat16, seed=13)
        out, state = chunk_gated_delta_product(
            **inputs,
            initial_state=pool,
            output_state=output,
            output_final_state=True,
            state_indices=slots,
            backend=name,
        )
        results[name] = (out, state[slots.long()].float())
    (out, state), (ref_out, ref_state) = results["cake"], results["cudnn"]
    assert rel_err(out, ref_out) < TOLERANCE
    assert rel_err(state, ref_state) < TOLERANCE


# ---------------------------------------------------------------------------
# stochastic rounding: the Philox model
# ---------------------------------------------------------------------------


SEED = 0x0123_4567_89AB_CDEF


@pytest.mark.parametrize("rounds", [5, 10, 16])
@pytest.mark.parametrize("batch", sorted(BATCHES))
@pytest.mark.parametrize("dtype", ["bf16", "f16"])
def test_stochastic_store_matches_the_philox_model(dtype, batch, rounds):
    """Zero mismatches against the model on the twin's FP32 state: the read slot's counter where the call reads a
    valid read slot, the write slot's where the read slot is -1 or past the pool."""
    inputs, kwargs, reads, writes = _problem(batch, "disjoint", dtype, seed=14)
    seed = torch.tensor([SEED], dtype=torch.int64, device=_device())
    _, twin_state = _twin(inputs, kwargs)
    out, state = _call(inputs, **kwargs, rand_seed=seed, philox_rounds=rounds)
    assert (
        _mismatches(state, twin_state, kwargs, reads, writes, SEED, rounds, dtype) == 0
    )
    # the controls: one round more and another seed disagree with the kernel
    assert _mismatches(
        state, twin_state, kwargs, reads, writes, SEED, rounds + 1, dtype
    )
    assert _mismatches(
        state, twin_state, kwargs, reads, writes, SEED ^ 1, rounds, dtype
    )


@pytest.mark.parametrize("dtype", ["bf16", "f16"])
def test_stochastic_store_without_an_initial_state_counts_from_the_write_slot(dtype):
    inputs, kwargs, reads, writes = _problem("split", "disjoint", dtype, seed=15)
    kwargs["initial_state"] = None
    seed = torch.tensor([SEED], dtype=torch.int64, device=_device())
    _, twin_state = _twin(inputs, kwargs)
    _, state = _call(inputs, **kwargs, rand_seed=seed)
    assert _mismatches(state, twin_state, kwargs, reads, writes, SEED, 10, dtype) == 0


@pytest.mark.parametrize("dtype", ["bf16", "f16"])
def test_rounds_and_seeds_store_other_bits_and_repeat_exactly(dtype):
    inputs, kwargs, reads, writes = _problem("block", "disjoint", dtype, seed=16)
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


# A pool past 2^32 elements: slot 1 starts 4096 elements below 2^32, so the draws of its rows straddle the offset 2^32
# (the counter's high word); the model with the high word dropped disagrees.
STRADDLE = (1 << 32) - 4096


@pytest.mark.parametrize("init", [True, False], ids=["read-slot", "write-slot"])
@pytest.mark.parametrize("dtype", ["bf16", "f16"])
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
    have = pool[1].contiguous().view(torch.int16).cpu().numpy().view(np.uint16)
    assert (have != want).any()
    del flat, pool


# ---------------------------------------------------------------------------
# empty sequences
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["f32", "bf16"])
def test_a_batch_of_empty_sequences_copies_the_valid_slots(dtype):
    """Every sequence empty (no launch): the initial row at the read slot (zero for -1 or past the pool) is stored
    as it is at the write slot; -1 or past the pool stores nothing."""
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
@pytest.mark.parametrize("dtype", ["f32", "bf16", "f16"])
def test_no_sequence_with_a_dense_state_launches_nothing(dtype, initial, final):
    """``cu_seqlens = [0]`` with a dense ``[0, H, 64, 128]`` initial and / or output state of any dtype returns the
    output (and the output state tensor) without a launch."""
    if not initial and not final:
        pytest.skip("no state tensor")
    H, T = 2, 5
    device, state_dtype = _device(), DTYPES[dtype]

    def empty():
        return torch.empty(0, H, V, K, dtype=state_dtype, device=device)

    out_state = empty() if final else None
    result = chunk_gated_delta_product(
        torch.zeros(T, H, K, dtype=torch.bfloat16, device=device),
        torch.zeros(T * N_HH, H, K, dtype=torch.bfloat16, device=device),
        torch.zeros(T * N_HH, H, V, dtype=torch.bfloat16, device=device),
        num_householder=N_HH,
        initial_state=empty() if initial else None,
        output_state=out_state,
        output_final_state=final,
        cu_seqlens=torch.zeros(1, dtype=torch.int32, device=device),
        backend="cake",
    )
    out = result[0] if final else result
    assert tuple(out.shape) == (T, H, V)
    if final:
        assert result[1] is out_state


# ---------------------------------------------------------------------------
# CUDA graphs
# ---------------------------------------------------------------------------


def _capture(fn):
    stream = torch.cuda.Stream(device=_device())
    stream.wait_stream(torch.cuda.current_stream(_device()))
    with torch.cuda.stream(stream):
        fn()  # warm-up, eager
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        fn()
    return graph


@pytest.mark.parametrize("dtype", ["bf16", "f16"])
def test_a_graph_reads_the_seed_and_the_slots_at_replay(dtype):
    inputs, kwargs, reads, writes = _problem("split", "disjoint", dtype, seed=22)
    seed = torch.tensor([SEED], dtype=torch.int64, device=_device())
    out = torch.empty(
        int(inputs["q"].shape[0]),
        BATCHES["split"][0],
        V,
        dtype=torch.bfloat16,
        device=_device(),
    )
    initial0 = kwargs["initial_state"].clone()
    graph = _capture(lambda: _call(inputs, **kwargs, output=out, rand_seed=seed))
    # new seed, new slots, the initial pool restored: the replay stores what an eager call on those values stores
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


def test_identity_slots_made_in_a_capture_are_the_graphs_own():
    """A dense BF16 call captured with no identity slot table cached (the item table cached by an eager call of the
    batch) records its own table in the graph; an eager call afterwards makes a complete one; both store the same."""
    H, lengths = 8, [1, 65, 193, 300]
    N = len(lengths)
    inputs = _inputs(lengths, H, seed=23)
    initial = _pool(N, H, torch.bfloat16, seed=24)
    output = torch.empty(N, H, V, K, dtype=torch.bfloat16, device=_device())
    out = torch.empty(sum(lengths), H, V, dtype=torch.bfloat16, device=_device())

    def call():
        return _call(
            inputs,
            initial_state=initial,
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
    backend._IDENTITY_SLOTS.clear()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        call()
    assert not backend._IDENTITY_SLOTS
    out.fill_(float("nan"))
    output.fill_(float("nan"))
    call()  # eager, before any replay
    assert _equal_bits(out, expected_out) and _equal_bits(output, expected_state)
    out.fill_(float("nan"))
    output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert _equal_bits(out, expected_out) and _equal_bits(output, expected_state)


def test_a_graph_keeps_the_identity_slots_a_larger_batch_replaces():
    """A dense BF16 call captured after an eager call slices the cached identity table, and the graph reads that
    memory at every replay.  An eager call over more sequences than the table holds grows the cache; the replaced
    table stays alive, so after new allocations the replay still stores what the eager call stores."""
    H, lengths = 8, [1, 65, 193, 300]
    N = len(lengths)
    inputs = _inputs(lengths, H, seed=27)
    initial = _pool(N, H, torch.bfloat16, seed=28)
    output = torch.empty(N, H, V, K, dtype=torch.bfloat16, device=_device())
    out = torch.empty(sum(lengths), H, V, dtype=torch.bfloat16, device=_device())

    def call():
        return _call(
            inputs,
            initial_state=initial,
            output_state=output,
            output_final_state=True,
            output=out,
        )

    backend._IDENTITY_SLOTS.clear()
    call()  # on the current stream: the cached table is in this stream's allocator pool, as the filler below
    torch.cuda.synchronize()
    graph = _capture(call)  # slices the cached table
    expected_out, expected_state = out.clone(), output.clone()
    captured = backend._IDENTITY_SLOTS[_device().index or 0]
    size, address = captured.numel(), captured.data_ptr()
    del captured
    # an eager dense BF16 call over size + 1 one-token sequences grows the table
    big = size + 1
    big_inputs = _inputs([1] * big, 1, seed=29)
    big_initial = _pool(big, 1, torch.bfloat16, seed=30)
    big_output = torch.empty(big, 1, V, K, dtype=torch.bfloat16, device=_device())
    ref_out, ref_state = _call(
        big_inputs, initial_state=big_initial.float(), output_final_state=True
    )
    big_out, big_state = _call(
        big_inputs,
        initial_state=big_initial,
        output_state=big_output,
        output_final_state=True,
    )
    assert _equal_bits(big_out, ref_out)
    assert _equal_bits(big_state, ref_state.to(torch.bfloat16))
    grown = backend._IDENTITY_SLOTS[_device().index or 0]
    assert grown.numel() >= big and grown.data_ptr() != address
    # allocations of the old table's size would take its memory had it been freed: their 2s would send every
    # sequence of the replay to row 2
    filler = [
        torch.full((size,), 2, dtype=torch.int32, device=_device()) for _ in range(256)
    ]
    assert all(t.data_ptr() != address for t in filler)
    out.fill_(float("nan"))
    output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert _equal_bits(out, expected_out) and _equal_bits(output, expected_state)
    del filler


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------


def _small():
    inputs = _inputs([64, 3], 4, seed=25)
    pool = _pool(4, 4, torch.bfloat16, seed=26)
    slots = torch.tensor([1, 2], dtype=torch.int32, device=_device())
    return inputs, pool, slots


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


@pytest.mark.parametrize("name", ["auto", "cudnn"])
def test_other_backends_refuse_the_cake_arguments(name):
    inputs, pool, slots = _small()
    seed = torch.tensor([1], dtype=torch.int64, device=_device())
    inputs.pop("cu_seqlens_cpu")
    for kwargs in ({"rand_seed": seed}, {"output_state_indices": slots}):
        with pytest.raises(ValueError, match='backend="cake"'):
            chunk_gated_delta_product(
                **inputs,
                initial_state=pool,
                output_state=pool,
                output_final_state=True,
                state_indices=slots,
                backend=name,
                **kwargs,
            )
