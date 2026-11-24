"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Tests for the Cake Gated DeltaProduct (GDP) chunked varlen prefill kernel
(``K = 128``, ``V = 64``, one to four Householder updates per token,
FP32 state; SM90).

Every output row a sequence covers and every state row the call may touch is
compared with an FP64 token-serial reference of the same math written in this
file (inputs at their BF16 values, the state rounded to FP32 after each real
token, the decay before the token's updates and the readout after the last
one) at ``atol = rtol = 1e-2`` for the output and ``atol = 5e-3``,
``rtol = 1e-3`` for the final state, with no mismatch allowed; every pool row
the call must not touch is checked bitwise.  The matrix covers
``num_householder`` 1 to 4, the head counts 1 to 80, the sequence lengths 0 to
32768 (the chunk-boundary lengths 63 / 64 / 65, 127 / 128 / 129, 191 / 192 /
193 among them) in single and mixed batches, int32 / int64 ``cu_seqlens``, an
initial state none / given / pool-indexed (the output pool the input view or
disjoint memory, invalid slots), the final state kept or dropped, the fused q /
k L2 normalization, ``g = None`` / ``beta = None`` from a zero state (and the
refusal of those calls with an initial state or a pool, which no exported SM90
program serves), FP16 I/O, caller-provided output and state buffers, the scale
values, the 18 prefill performance rows, empty batches, sequence independence,
bitwise reproducibility, the batch body route (one body and one launch per
call: the V6 body's warm-up split programs for gated calls with a long
sequence that fills a slot per SM, the V6 fork-order body for short sequences
and for calls without a state, the merged body's unit and gate-saturation
split programs for unit calls and for batches whose long sequences make the
split the faster whole-batch kernel, the short and empty sequences in the same
launch; checked against the sequences alone), the launch count, CUDA Graph
capture and replay of a mixed batch (and the explicit failure of a V6 split
graph replayed on other lengths), and one launch of every exported prefill
program.  ``test_cake_gdp_prefill_route.py`` covers the route arithmetic and
the registry without a GPU.

This suite exercises the backend package alone: it names the package once,
through ``tests.cake.test_helpers.cake_package``, and imports nothing else from
flashinfer, so a build that vendors the package under another import path runs
it unchanged after pointing that table entry at its package.  The
``flashinfer.gdp_prefill.chunk_gated_delta_product(backend="cake")`` route is
covered by ``test_cake_gdp_prefill_api.py``.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pytest
import torch

from tests.cake.test_helpers.cake_package import cake_package

pkg = cake_package("cake_gdp")
cake_build = cake_package("cake_gdp", "cake_build")
loader = cake_package("cake_gdp", "cake_jit")
cake_chunk_gated_delta_product = pkg.cake_chunk_gated_delta_product
is_cake_gdp_prefill_supported = pkg.is_cake_gdp_prefill_supported

SUPPORTED_CAPABILITIES = ((9, 0),)
K = 128
V = 64
O_ATOL = 1e-2
O_RTOL = 1e-2
S_ATOL = 5e-3
S_RTOL = 1e-3
SEED = 760
NUM_HOUSEHOLDER = (1, 2, 3, 4)
HEADS = (1, 2, 3, 6, 10, 12, 20, 24, 40, 80)
LENGTHS = (0, 1, 2, 63, 64, 65, 127, 128, 129, 191, 192, 193, 1000, 4097, 32768)
# The prefill performance shapes (num_householder = 3): every performance row is a correctness case.
PERF_HEADS = (80, 10, 24)
PERF_NL = ((1, 1024), (1, 8192), (1, 32768), (4, 4096), (16, 1024), (64, 256))
UNUSED_SLOTS = 2  # pool rows no sequence names, which must come back bitwise
SENTINEL = 7.5
L2NORM_EPS = 1e-6
# The flag sets the SM90 prefill route carries (use_initial_state, use_state_indices, store_final_state,
# unit_gates, unit_beta), per kernel body.  The V6 fork-order body (``prefill_sm90_v6``): the three gate-tensor
# programs, every one storing a final state (a call keeping none gets a discard record); the body route of record
# sends it every sequence under ``ROUTE_MIN_CHUNKS_P3`` chunks and every call without an initial and a final state.
V6_FLAGS = {
    (False, False, True, False, False),
    (True, False, True, False, False),
    (True, True, True, False, False),
}
# The merged body (``prefill_sm90``): the unit programs from a zero state (g=None: the unit-gate program, which
# stores the final state; beta=None: the unit-beta program) ...
UNIT_FLAGS = {
    (False, False, True, True, False),
    (False, False, True, False, True),
}
# ... and the gate-saturation split programs (``split_items``) of the three gated flag sets, which run the
# merged-class sequences (at least ``ROUTE_MIN_CHUNKS_P3`` chunks and the split items cheaper than the V6 steps;
# the split rule of record over the host copy of ``cu_seqlens`` -- ``cu_seqlens_cpu``, else ``cu_seqlens.cpu()`` --
# always cuts such a sequence).  The merged body routes no exact gate-tensor program any more.
SPLIT_FLAGS = {
    (False, False, True, False, False),
    (True, False, True, False, False),
    (True, True, True, False, False),
}
# ... and the warm-up split programs of the V6 body (``split_items`` twins of its three programs), which run a gated
# call whose ``prefill_v6_sm90_plan`` pays (a long sequence whose head-major chunks fill a slot per SM -- on 132 SMs
# an 8192-token sequence at n = 3 from H = 8 on -- and the plan lighter than the V6 program's tiles), every sequence
# of the batch in the one launch.
V6_SPLIT_FLAGS = V6_FLAGS
# The exact programs of the route: every routed program but the split ones.
ROUTED_FLAGS = V6_FLAGS | UNIT_FLAGS
# A batch of two 384-chunk sequences at n = 3, merged-class at the head counts the tests pass it (3 and 10; every
# test asserts the class through the route), and the mixed-class batch of record: at H = 10 its long sequences are
# merged-class and the short and the empty one V6-class (``[P3, V6, V6, P3]``).  At H = 10 (and not at H = 3) a
# 384-chunk sequence is balanced on 132 SMs, so these batches run the V6 split program there; at H = 3 the merged
# body's split program.
MERGED_LENS = (8192, 8192)
MIXED_LENS = (8192, 1000, 0, 8192)
MIXED_H = 10
MERGED_H = 3


def _device() -> torch.device:
    return torch.device("cuda:0")


def _capability() -> tuple[int, int]:
    return torch.cuda.get_device_capability(_device())


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and _capability() in SUPPORTED_CAPABILITIES
        and is_cake_gdp_prefill_supported(_device())
    )


pytestmark = pytest.mark.skipif(
    not _supported(),
    reason="cake_gdp prefill needs an SM90 device with exported routes",
)


def _arch() -> str:
    return loader.arch_for_capability(_capability())


def _prefill_libraries(arch: str) -> set[str]:
    return {
        loader.library_name(name, arch)
        for name, record in loader.MODULES.items()
        if record["role"] == "prefill" and arch in record["arches"]
    }


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    # Build the local architecture's prefill libraries that have no prebuilt file
    # before the first test, through the spec objects the package's loader itself
    # builds on first use; a prebuilt or already cached library loads without
    # compiling.
    if _supported():
        arch = _arch()
        prefixes = tuple(f"{library}_" for library in _prefill_libraries(arch))
        for spec in loader.specs_for_arch(arch):
            if any(spec.name[: len(prefix)] == prefix for prefix in prefixes):
                spec.build_and_load()
    yield


def _generator(*keys: int) -> torch.Generator:
    seed = SEED
    for key in keys:
        seed = (seed * 1_000_003 + int(key)) % (2**31 - 1)
    return torch.Generator(device=_device()).manual_seed(seed)


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Case:
    lens: tuple[int, ...]
    H: int
    n: int = 3
    initial: str = "none"  # none | given | pool_same | pool_disjoint
    final: bool = True
    index_dtype: torch.dtype = torch.int32
    l2norm: bool = False
    unit_g: bool = False
    unit_beta: bool = False
    dtype: torch.dtype = torch.bfloat16
    scale: Optional[float] = None
    invalid: tuple[
        int, ...
    ] = ()  # sequence positions whose pool slot is invalid (-1 / pool size, alternating)
    provided_output: bool = False
    provided_state: bool = False  # a caller buffer for the per-sequence final state
    gate_width: float = 0.5  # alpha drawn in (1 - gate_width, 1]
    cu_cpu: bool = (
        False  # pass the host copy of cu_seqlens (else the backend synchronizes it)
    )
    seed: int = 0
    # filled by _make
    q: torch.Tensor = None
    k: torch.Tensor = None
    v: torch.Tensor = None
    g: Optional[torch.Tensor] = None
    beta: Optional[torch.Tensor] = None
    cu: torch.Tensor = None
    cu_host: torch.Tensor = None  # the CPU copy of ``cu``
    offsets: tuple[int, ...] = ()
    initial_state: Optional[torch.Tensor] = None
    output_state: Optional[torch.Tensor] = None
    state_indices: Optional[torch.Tensor] = None
    slots: list = dataclasses.field(default_factory=list)
    valid: list = dataclasses.field(default_factory=list)
    pool_rows: int = 0
    output: Optional[torch.Tensor] = None
    initial_before: Optional[torch.Tensor] = None
    output_state_before: Optional[torch.Tensor] = None
    output_before: Optional[torch.Tensor] = None

    @property
    def T(self) -> int:
        return sum(self.lens)

    @property
    def N(self) -> int:
        return len(self.lens)

    @property
    def pool(self) -> bool:
        return self.initial in ("pool_same", "pool_disjoint")


def _l2norm_fp64(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float64)
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + L2NORM_EPS)


def _make(c: Case) -> Case:
    device = _device()
    generator = _generator(
        c.seed, c.H, c.n, len(c.lens), sum(c.lens), int(c.final), int(c.pool)
    )
    T, N, H, n = c.T, c.N, c.H, c.n

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(
            *shape, device=device, generator=generator, dtype=torch.float32
        )

    def rand(*shape: int) -> torch.Tensor:
        return torch.rand(
            *shape, device=device, generator=generator, dtype=torch.float32
        )

    q, k = randn(T, H, K), randn(T * n, H, K)
    if not c.l2norm:  # pre-normalized inputs, as the kernel reads them
        q = torch.nn.functional.normalize(q, dim=-1, eps=0.0)
        k = torch.nn.functional.normalize(k, dim=-1, eps=0.0)
    c.q, c.k, c.v = q.to(c.dtype), k.to(c.dtype), randn(T * n, H, V).to(c.dtype)
    c.g = None if c.unit_g else 1.0 - c.gate_width * rand(T, H)
    c.beta = None if c.unit_beta else rand(T * n, H)
    c.offsets = (0, *itertools.accumulate(c.lens))
    c.cu = torch.tensor(c.offsets, device=device, dtype=c.index_dtype)
    c.cu_host = torch.tensor(c.offsets, dtype=c.index_dtype)
    c.pool_rows = N + UNUSED_SLOTS if c.pool else N
    state_shape = (c.pool_rows, H, V, K)
    if c.pool:
        slots = torch.randperm(c.pool_rows, device=device, generator=generator)[
            :N
        ].tolist()
        for position, index in enumerate(c.invalid):
            slots[index] = -1 if position % 2 == 0 else c.pool_rows
        c.slots = slots
        c.valid = [0 <= slot < c.pool_rows for slot in slots]
        c.state_indices = torch.tensor(slots, device=device, dtype=c.index_dtype)
        c.initial_state = randn(*state_shape)
        if c.initial == "pool_same":
            c.output_state = c.initial_state
        else:
            c.output_state = randn(*state_shape)
    else:
        c.slots, c.valid = list(range(N)), [True] * N
        c.initial_state = randn(*state_shape) if c.initial == "given" else None
        c.output_state = randn(*state_shape) if c.provided_state else None
    c.initial_before = None if c.initial_state is None else c.initial_state.clone()
    c.output_state_before = None if c.output_state is None else c.output_state.clone()
    if c.provided_output:
        c.output = torch.full((T, H, V), SENTINEL, device=device, dtype=c.dtype)
        c.output_before = c.output.clone()
    return c


def _run(c: Case):
    result = cake_chunk_gated_delta_product(
        c.q,
        c.k,
        c.v,
        g=c.g,
        beta=c.beta,
        num_householder=c.n,
        scale=c.scale,
        initial_state=c.initial_state,
        output_final_state=c.final,
        cu_seqlens=c.cu,
        use_qk_l2norm_in_kernel=c.l2norm,
        output=c.output,
        output_state=c.output_state,
        state_indices=c.state_indices,
        cu_seqlens_cpu=c.cu_host if c.cu_cpu else None,
    )
    if c.final:
        out, state = result
    else:
        out, state = result, None
    return out, state


# ---------------------------------------------------------------------------
# The FP64 token-serial reference (decay, n Householder updates, readout; FP32 state per token)
# ---------------------------------------------------------------------------


def _recurrence(q, k, v, g, beta, row0, length, state, scale, output):
    """Advance ``row0.numel()`` sequences of ``length`` tokens together from ``state`` ``[Ns, H, K, V]`` FP64,
    writing their output rows into ``output`` and returning the final states.  ``k`` / ``v`` / ``beta`` are the
    expanded timeline viewed ``[T, n, H, ...]``."""

    n = k.shape[1]
    for t in range(length):
        rows = row0 + t
        if g is not None:
            state = state * g[rows][:, :, None, None]
        for j in range(n):
            kq = k[rows, j]
            u = torch.einsum("shk,shkv->shv", kq, state)
            vn = v[rows, j] - u
            if beta is not None:
                vn = vn * beta[rows, j][:, :, None]
            state = state + torch.einsum("shk,shv->shkv", kq, vn)
        state = state.to(torch.float32).to(torch.float64)
        output[rows] = scale * torch.einsum("shk,shkv->shv", q[rows], state)
    return state


def _reference(c: Case) -> tuple[torch.Tensor, torch.Tensor]:
    """``(O [T, H, V] FP64 with NaN where no sequence writes, final [N, H, V, K] FP64)``: the final state of a
    sequence is its initial state (zeros without one or on an invalid slot) advanced over its tokens."""

    device = _device()
    T, N, H, n = c.T, c.N, c.H, c.n
    if c.l2norm:
        q = _l2norm_fp64(c.q).to(c.dtype).to(torch.float64)
        k = _l2norm_fp64(c.k).to(c.dtype).to(torch.float64)
    else:
        q, k = c.q.to(torch.float64), c.k.to(torch.float64)
    k = k.contiguous().view(T, n, H, K)
    v = c.v.to(torch.float64).view(T, n, H, V)
    g = None if c.g is None else c.g.to(torch.float64)
    beta = None if c.beta is None else c.beta.to(torch.float64).view(T, n, H)
    scale = float(c.scale) if c.scale not in (None, 0.0) else K**-0.5
    output = torch.full((T, H, V), float("nan"), device=device, dtype=torch.float64)
    final = torch.zeros((N, H, V, K), device=device, dtype=torch.float64)
    initial = None if c.initial_state is None else c.initial_before.to(torch.float64)

    def initial_of(seq: int) -> torch.Tensor:
        if initial is None or not c.valid[seq]:
            return torch.zeros((H, K, V), device=device, dtype=torch.float64)
        return initial[c.slots[seq]].transpose(-1, -2)

    groups: dict[int, list[int]] = {}
    for seq in range(N):
        lo, hi = c.offsets[seq], c.offsets[seq + 1]
        if lo == hi:
            final[seq] = initial_of(seq).transpose(-1, -2)
            continue
        groups.setdefault(hi - lo, []).append(seq)
    for length in sorted(groups):
        members = groups[length]
        row0 = torch.tensor(
            [c.offsets[seq] for seq in members], device=device, dtype=torch.int64
        )
        state = torch.stack([initial_of(seq) for seq in members])
        state = _recurrence(q, k, v, g, beta, row0, length, state, scale, output)
        for position, seq in enumerate(members):
            final[seq] = state[position].transpose(-1, -2)
    return output, final


def _assert_close(
    actual: torch.Tensor, expected: torch.Tensor, what: str, *, atol: float, rtol: float
) -> None:
    actual = actual.to(torch.float64)
    expected = expected.to(torch.float64)
    assert torch.isfinite(actual).all(), f"{what}: non-finite values"
    error = (actual - expected).abs()
    bound = atol + rtol * expected.abs()
    mismatches = int((error > bound).sum())
    assert mismatches == 0, (
        f"{what}: {mismatches} / {actual.numel()} elements beyond atol={atol}, rtol={rtol}; max abs {error.max().item():.3e}"
    )


def _check(c: Case, out: torch.Tensor, state: Optional[torch.Tensor]) -> None:
    ref_out, ref_final = _reference(c)
    assert out.dtype == c.dtype and tuple(out.shape) == (c.T, c.H, V)
    if c.provided_output:
        assert out is c.output
    for seq in range(c.N):
        lo, hi = c.offsets[seq], c.offsets[seq + 1]
        if lo < hi:
            _assert_close(
                out[lo:hi],
                ref_out[lo:hi],
                f"output rows of sequence {seq}",
                atol=O_ATOL,
                rtol=O_RTOL,
            )
    if (
        c.provided_output and c.offsets[-1] < c.T
    ):  # rows past every sequence keep the caller's values
        assert torch.equal(out[c.offsets[-1] :], c.output_before[c.offsets[-1] :])
    if not c.final:
        assert state is None
        if c.initial_state is not None:
            assert torch.equal(c.initial_state, c.initial_before), (
                "initial state modified without output_final_state"
            )
        if c.output_state is not None and c.output_state is not c.initial_state:
            assert torch.equal(c.output_state, c.output_state_before), (
                "output pool modified without output_final_state"
            )
        return
    assert state is not None and state.dtype == torch.float32
    if c.pool:
        assert state is c.output_state
        written = set()
        for seq in range(c.N):
            if c.valid[seq]:
                written.add(c.slots[seq])
                _assert_close(
                    state[c.slots[seq]],
                    ref_final[seq],
                    f"final state of sequence {seq}",
                    atol=S_ATOL,
                    rtol=S_RTOL,
                )
        for row in range(c.pool_rows):
            if row not in written:
                assert torch.equal(state[row], c.output_state_before[row]), (
                    f"pool row {row} touched"
                )
        if (
            c.initial == "pool_disjoint" and c.initial_state is not None
        ):  # None: the write-only pool
            assert torch.equal(c.initial_state, c.initial_before), (
                "initial pool modified"
            )
    else:
        assert tuple(state.shape) == (c.N, c.H, V, K)
        if c.provided_state:
            assert state is c.output_state
        if c.initial_state is not None:
            assert torch.equal(c.initial_state, c.initial_before), (
                "initial state modified"
            )
        for seq in range(c.N):
            _assert_close(
                state[seq],
                ref_final[seq],
                f"final state of sequence {seq}",
                atol=S_ATOL,
                rtol=S_RTOL,
            )


def _run_and_check(c: Case) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    _make(c)
    out, state = _run(c)
    _check(c, out, state)
    return out, state


def _classes(c: Case) -> list[str]:
    """The body of every sequence of ``c`` under the route of record on this device, with the flags the backend
    derives from the call: a program stores a final state when the call keeps one, reads an initial state (a pool
    case carries its initial pool) or runs a unit program."""
    stores = c.final or c.initial != "none" or c.unit_g or c.unit_beta
    return loader.prefill_body_classes(
        (0, *itertools.accumulate(c.lens)),
        c.n,
        c.H,
        loader.sm_count(0),
        store_final_state=stores,
        unit_gates=c.unit_g,
        unit_beta=c.unit_beta,
        total_tokens=c.T,
    )


def _body(c: Case) -> str:
    """The one body of the call ``c`` under the batch route on this device (``prefill_batch_body`` with the flags
    the backend derives from the call, as ``_classes``)."""
    stores = c.final or c.initial != "none" or c.unit_g or c.unit_beta
    return loader.prefill_batch_body(
        (0, *itertools.accumulate(c.lens)),
        c.n,
        c.H,
        loader.sm_count(0),
        store_final_state=stores,
        unit_gates=c.unit_g,
        unit_beta=c.unit_beta,
        total_tokens=c.T,
    )


def _v6_split_of(lens: tuple[int, ...], n: int, H: int) -> bool:
    """Whether a gated call of ``lens`` runs a V6 split program on this device (its plan pays)."""
    return loader.prefill_v6_sm90_plan(
        (0, *itertools.accumulate(lens)),
        num_heads=H,
        num_householder=n,
        sm_count=loader.sm_count(0),
    ).split


def _v6_split(c: Case) -> bool:
    """Whether the call ``c`` runs a V6 split program (a gated call whose plan pays)."""
    return not (c.unit_g or c.unit_beta) and _v6_split_of(c.lens, c.n, c.H)


def _launch(c: Case) -> tuple[str, bool]:
    """The body of the one launch of ``c`` and whether its program is a split program."""
    if _v6_split(c):
        return loader.ROUTE_BODY_V6, True
    body = _body(c)
    return body, body == loader.ROUTE_BODY_P3 and not (c.unit_g or c.unit_beta)


# A mixed bench row: four one-token sequences then one 8192-token sequence at H = 10: one V6 split launch (the long
# sequence balanced on a slot per SM, the one-token rows on the least-loaded CTAs).
BENCH_MIXED_LENS = (1, 1, 1, 1, 8192)


# ---------------------------------------------------------------------------
# Correctness matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", NUM_HOUSEHOLDER)
@pytest.mark.parametrize(
    "lens,H",
    (((129, 0, 1, 1000), 6), ((64, 65, 63), 10), ((4097,), 3), ((192, 191, 193, 2), 2)),
)
def test_prefill_matrix(n: int, lens: tuple[int, ...], H: int) -> None:
    _run_and_check(Case(lens=lens, H=H, n=n, initial="given", seed=1))


@pytest.mark.parametrize("length", LENGTHS)
def test_sequence_lengths(length: int) -> None:
    _run_and_check(Case(lens=(length,), H=2, n=3, initial="given", seed=2))


def test_mixed_batch_of_every_length() -> None:
    _run_and_check(Case(lens=LENGTHS[:-2], H=4, n=2, initial="given", seed=3))


@pytest.mark.parametrize("H", HEADS)
def test_head_counts(H: int) -> None:
    _run_and_check(Case(lens=(300, 129, 7), H=H, n=3, initial="given", seed=4))


@pytest.mark.parametrize("H", PERF_HEADS)
@pytest.mark.parametrize("N,L", PERF_NL)
def test_prefill_performance_rows(H: int, N: int, L: int) -> None:
    _run_and_check(
        Case(lens=(L,) * N, H=H, n=3, initial="given", gate_width=1.0, seed=5)
    )


@pytest.mark.parametrize("initial", ("none", "given", "pool_same", "pool_disjoint"))
@pytest.mark.parametrize("final", (True, False))
def test_initial_and_final_state_modes(initial: str, final: bool) -> None:
    _run_and_check(
        Case(lens=(200, 0, 65, 1), H=6, n=3, initial=initial, final=final, seed=6)
    )


def test_write_only_pool_begins_at_zeros() -> None:
    """``state_indices`` with an output pool and no ``initial_state``: every sequence begins at zeros and the
    final states land in the named slots."""
    c = _make(Case(lens=(130, 7, 64), H=3, n=2, initial="pool_disjoint", seed=7))
    c.initial_state, c.initial_before = None, None
    out, state = _run(c)
    _check(c, out, state)


@pytest.mark.parametrize("H", (MERGED_H, MIXED_H))
def test_write_only_pool_begins_at_zeros_on_the_split_route(H: int) -> None:
    """A single L = 8192, n = 3 sequence (384 chunks) runs a split program on 132 SMs: at H = 3 the merged body's
    (a merged-class row, cut into p > 1 items), at H = 10 the V6 body's (a balanced row).  With ``state_indices``,
    an output pool and no ``initial_state`` the sequence begins at zeros, its final state lands in the named slot
    and the other pool rows keep their pre-launch values."""
    c = _make(Case(lens=(8192,), H=H, n=3, initial="pool_disjoint", seed=11))
    c.initial_state, c.initial_before = None, None
    c.output_state.fill_(SENTINEL)
    c.output_state_before = c.output_state.clone()
    sm_count = loader.sm_count(0)
    if H == MIXED_H:
        assert _v6_split(c)
    else:
        assert not _v6_split(c)
        classes = loader.prefill_body_classes(c.cu_host, c.n, c.H, sm_count)
        assert classes == [loader.ROUTE_BODY_P3], classes
        (chunks,) = loader.route_chunk_steps(c.cu_host, c.n)
        p, _span = loader.split_count(chunks, c.H, c.n, sm_count)
        assert chunks == 384 and p > 1, (chunks, p, sm_count)
    out, state = _run(c)
    _check(c, out, state)


@pytest.mark.parametrize("final", (True, False))
def test_invalid_pool_slots(final: bool) -> None:
    """A negative or out-of-range slot: the sequence begins at zeros, its output rows are written, nothing is
    stored for it."""
    c = Case(
        lens=(100, 65, 0, 129, 3),
        H=4,
        n=3,
        initial="pool_disjoint",
        final=final,
        invalid=(0, 3, 2),
        seed=8,
    )
    _run_and_check(c)


def test_int64_metadata() -> None:
    _run_and_check(
        Case(
            lens=(129, 1000, 0, 65),
            H=3,
            n=3,
            initial="pool_disjoint",
            index_dtype=torch.int64,
            seed=9,
        )
    )
    _run_and_check(
        Case(
            lens=(129, 64), H=2, n=1, initial="given", index_dtype=torch.int64, seed=10
        )
    )
    # The plain fork-order program reads int32 metadata only: the backend casts the int64 cu_seqlens for it.
    _run_and_check(
        Case(lens=(129, 1000, 0, 65), H=3, n=3, index_dtype=torch.int64, seed=12)
    )


def test_qk_l2norm_in_kernel() -> None:
    _run_and_check(
        Case(lens=(257, 64, 1), H=6, n=3, initial="given", l2norm=True, seed=11)
    )


@pytest.mark.parametrize("unit_g,unit_beta", ((True, False), (False, True)))
@pytest.mark.parametrize(
    "lens,H,n",
    (((193, 2, 129), 3, 2), ((65, 0, 193, 1000), 3, 3), ((1000,), 24, 2)),
)
def test_unit_gates(
    unit_g: bool, unit_beta: bool, lens: tuple[int, ...], H: int, n: int
) -> None:
    """``g=None`` runs the unit-gate program and ``beta=None`` the unit-beta program, from zeros (the kernel's own
    unit-gates / unit-beta cases among the shapes); the undecayed state is held to the same final-state
    tolerance."""
    _run_and_check(
        Case(
            lens=lens,
            H=H,
            n=n,
            initial="none",
            unit_g=unit_g,
            unit_beta=unit_beta,
            seed=12,
        )
    )


@pytest.mark.parametrize("unit_g,unit_beta", ((True, False), (False, True)))
@pytest.mark.parametrize("initial", ("given", "pool_disjoint", "pool_same"))
def test_unit_gates_with_a_state_are_refused(
    unit_g: bool, unit_beta: bool, initial: str
) -> None:
    """No exported SM90 program reads an initial state or a slot pool with unit gates or unit betas: the backend
    refuses the call with ``ValueError`` naming the flags instead of approximating it."""
    c = _make(
        Case(
            lens=(193, 2, 129),
            H=3,
            n=2,
            initial=initial,
            unit_g=unit_g,
            unit_beta=unit_beta,
            seed=12,
        )
    )
    with pytest.raises(
        ValueError, match="no exported Cake GDP prefill program on sm_90a for this call"
    ):
        _run(c)
    assert torch.equal(
        c.initial_state, c.initial_before
    )  # nothing launched, nothing touched
    if c.output_state is not None and c.output_state is not c.initial_state:
        assert torch.equal(c.output_state, c.output_state_before)


@pytest.mark.parametrize(
    "lens,H,n,initial",
    (
        ((193, 2, 129), 3, 2, "given"),
        ((193, 2, 129), 3, 2, "none"),
        ((65, 0, 193, 1000), 3, 3, "none"),
        ((1000,), 24, 2, "none"),
    ),
)
def test_both_unit_gates_are_refused(
    lens: tuple[int, ...], H: int, n: int, initial: str
) -> None:
    """``g=None`` together with ``beta=None`` is no exported program: the backend refuses the call with
    ``ValueError`` instead of approximating it."""
    with pytest.raises(
        ValueError,
        match="no exported Cake GDP prefill program for g=None with beta=None",
    ):
        _run_and_check(
            Case(
                lens=lens,
                H=H,
                n=n,
                initial=initial,
                unit_g=True,
                unit_beta=True,
                seed=12,
            )
        )


@pytest.mark.parametrize("provided_output", (False, True))
def test_fp16_io(provided_output: bool) -> None:
    _run_and_check(
        Case(
            lens=(129, 65),
            H=3,
            n=3,
            initial="given",
            dtype=torch.float16,
            provided_output=provided_output,
            seed=13,
        )
    )


def test_provided_output_and_state_buffers() -> None:
    c = Case(
        lens=(129, 0, 65),
        H=3,
        n=3,
        initial="given",
        provided_output=True,
        provided_state=True,
        seed=14,
    )
    _run_and_check(c)


def test_non_contiguous_output_is_copied_back() -> None:
    c = _make(Case(lens=(129, 65), H=4, n=3, initial="given", seed=15))
    wide = torch.full(
        (c.T, c.H, 2 * V), SENTINEL, device=_device(), dtype=torch.bfloat16
    )
    c.output = wide[:, :, :V]
    c.output_before = c.output.clone()
    c.provided_output = True
    out, state = _run(c)
    assert out is c.output
    _check(c, out, state)
    assert torch.equal(wide[:, :, V:], torch.full_like(wide[:, :, V:], SENTINEL))


@pytest.mark.parametrize("scale", (None, 0.0, 1.0, 0.3))
def test_scale(scale: Optional[float]) -> None:
    _run_and_check(Case(lens=(129, 5), H=2, n=3, initial="given", scale=scale, seed=16))


def test_scale_zero_is_the_default() -> None:
    a = _make(Case(lens=(129, 5), H=2, n=3, initial="given", scale=None, seed=16))
    b = _make(Case(lens=(129, 5), H=2, n=3, initial="given", scale=0.0, seed=16))
    assert torch.equal(_run(a)[0], _run(b)[0])


def test_strided_inputs() -> None:
    """Views whose innermost dimension is dense (a head slice of a wider tensor) are read in place."""
    c = _make(Case(lens=(129, 64), H=4, n=3, initial="given", seed=17))
    device = _device()
    generator = _generator(99)
    wide_q = torch.randn(c.T, c.H + 2, K, device=device, generator=generator).to(
        torch.bfloat16
    )
    wide_k = torch.randn(c.T * c.n, c.H + 2, K, device=device, generator=generator).to(
        torch.bfloat16
    )
    wide_v = torch.randn(c.T * c.n, c.H + 2, V, device=device, generator=generator).to(
        torch.bfloat16
    )
    wide_q[:, 1 : 1 + c.H] = c.q
    wide_k[:, 1 : 1 + c.H] = c.k
    wide_v[:, 1 : 1 + c.H] = c.v
    c.q, c.k, c.v = (
        wide_q[:, 1 : 1 + c.H],
        wide_k[:, 1 : 1 + c.H],
        wide_v[:, 1 : 1 + c.H],
    )
    assert not c.q.is_contiguous()
    out, state = _run(c)
    _check(c, out, state)


def test_bf16_initial_state_is_converted() -> None:
    c = _make(Case(lens=(129, 64), H=3, n=3, initial="given", seed=18))
    low = c.initial_state.to(torch.bfloat16)
    c.initial_state = low
    c.initial_before = low.to(torch.float32)
    out, state = _run(c)
    c.initial_state = c.initial_before  # the reference reads the BF16 values
    _check(c, out, state)


# ---------------------------------------------------------------------------
# Empty batches
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("initial", ("none", "given", "pool_disjoint"))
def test_batch_of_empty_sequences(initial: str) -> None:
    """No token: every sequence is an identity state update (its initial state, zeros without one)."""
    _run_and_check(Case(lens=(0, 0, 0), H=3, n=3, initial=initial, seed=19))


def test_no_sequence() -> None:
    device = _device()
    q = torch.zeros((5, 2, K), device=device, dtype=torch.bfloat16)
    k = torch.zeros((15, 2, K), device=device, dtype=torch.bfloat16)
    v = torch.zeros((15, 2, V), device=device, dtype=torch.bfloat16)
    cu = torch.zeros((1,), device=device, dtype=torch.int32)
    out = cake_chunk_gated_delta_product(q, k, v, num_householder=3, cu_seqlens=cu)
    assert tuple(out.shape) == (5, 2, V) and out.dtype == torch.bfloat16
    out, state = cake_chunk_gated_delta_product(
        q, k, v, num_householder=3, cu_seqlens=cu, output_final_state=True
    )
    assert tuple(state.shape) == (0, 2, V, K)


# ---------------------------------------------------------------------------
# Determinism, independence, launch count
# ---------------------------------------------------------------------------


def test_launches_are_bitwise_reproducible() -> None:
    c = _make(Case(lens=(8192, 300, 0, 65), H=10, n=3, initial="given", seed=24))
    results = []
    for _ in range(3):
        out, state = _run(c)
        results.append((out.clone(), state.clone()))
    for out, state in results[1:]:
        assert torch.equal(out, results[0][0]) and torch.equal(state, results[0][1])


def test_sequences_are_independent() -> None:
    """A sequence alone gives the same bits as in its batch (output rows and final state)."""
    c = _make(Case(lens=(300, 0, 129, 1000, 1), H=4, n=3, initial="given", seed=25))
    out, state = _run(c)
    for seq, length in enumerate(c.lens):
        lo, hi = c.offsets[seq], c.offsets[seq + 1]
        alone = cake_chunk_gated_delta_product(
            c.q[lo:hi].contiguous(),
            c.k[lo * c.n : hi * c.n].contiguous(),
            c.v[lo * c.n : hi * c.n].contiguous(),
            c.g[lo:hi].contiguous(),
            c.beta[lo * c.n : hi * c.n].contiguous(),
            c.n,
            initial_state=c.initial_state[seq : seq + 1].contiguous(),
            output_final_state=True,
            cu_seqlens=torch.tensor([0, length], device=_device(), dtype=torch.int32),
        )
        assert torch.equal(alone[0], out[lo:hi]), seq
        assert torch.equal(alone[1][0], state[seq]), seq


@pytest.mark.parametrize("initial", ("none", "given", "pool_disjoint", "pool_same"))
@pytest.mark.parametrize("final", (True, False))
def test_mixed_batch_matches_the_sequences_alone(initial: str, final: bool) -> None:
    """The mixed batch of record (two long sequences, balanced at H = 10 on 132 SMs, around a short and an empty
    one) is one launch of the V6 split program of its flags for the whole batch and satisfies the FP64 reference.
    Every sequence, and the long pair, run alone gives the same bits as in the batch (its output rows, its final
    state slot) when it runs the same program alone: a long sequence's rows depend on its own length only.  A
    sequence that runs another program alone (the short one) is held to the reference only.  An empty sequence
    copies its initial state under every program, so it is always compared bitwise.  The pool rows of the other
    sequences stay untouched."""
    c = Case(lens=MIXED_LENS, H=MIXED_H, n=3, initial=initial, final=final, seed=30)
    V6, P3 = loader.ROUTE_BODY_V6, loader.ROUTE_BODY_P3
    if final or initial != "none":
        assert _classes(c) == [P3, V6, V6, P3]
    else:
        assert _classes(c) == [V6] * c.N
    assert _v6_split(c)
    launch = _launch(c)
    out, state = _run_and_check(c)
    device = _device()
    sm_count = loader.sm_count(0)
    stores = c.final or c.initial != "none"
    groups = [[seq] for seq in range(c.N)] + [
        [seq for seq, cls in enumerate(_classes(c)) if cls == P3]
    ]
    for group in groups:
        if not group:
            continue
        lens = [c.offsets[seq + 1] - c.offsets[seq] for seq in group]
        if _v6_split_of(tuple(lens), c.n, c.H):
            alone_launch = (V6, True)
        else:
            alone_body = loader.prefill_batch_body(
                (0, *itertools.accumulate(lens)),
                c.n,
                c.H,
                sm_count,
                store_final_state=stores,
            )
            alone_launch = (alone_body, alone_body == P3)
        if alone_launch != launch and sum(lens):
            continue  # another program alone: the FP64 reference above holds it
        rows = torch.cat(
            [torch.arange(c.offsets[seq], c.offsets[seq + 1]) for seq in group]
        ).to(device)
        rows_n = (rows[:, None] * c.n + torch.arange(c.n, device=device)).reshape(-1)
        group_t = torch.tensor(group, device=device)
        kwargs = dict(
            num_householder=c.n,
            output_final_state=c.final,
            cu_seqlens=torch.tensor([0, *lens])
            .cumsum(0)
            .to(device=device, dtype=torch.int32),
        )
        pool = None
        if c.pool:
            # The alone call reads the pool values the batch read and writes a fresh copy of the output pool.
            before = (
                c.initial_before if c.initial == "pool_same" else c.output_state_before
            )
            pool = before.clone()
            kwargs.update(
                initial_state=pool if c.initial == "pool_same" else c.initial_before,
                state_indices=c.state_indices.index_select(0, group_t),
            )
            if c.final:
                kwargs["output_state"] = pool
        elif c.initial == "given":
            kwargs["initial_state"] = c.initial_before.index_select(
                0, group_t
            ).contiguous()
        alone = cake_chunk_gated_delta_product(
            c.q.index_select(0, rows),
            c.k.index_select(0, rows_n),
            c.v.index_select(0, rows_n),
            c.g.index_select(0, rows),
            c.beta.index_select(0, rows_n),
            **kwargs,
        )
        out_alone = alone[0] if c.final else alone
        assert torch.equal(out_alone, out.index_select(0, rows)), group
        if not c.final:
            continue
        if c.pool:
            assert alone[1] is pool
            written = {c.slots[seq] for seq in group}
            for row in range(c.pool_rows):
                expected = state[row] if row in written else before[row]
                assert torch.equal(pool[row], expected), (group, row)
        else:
            for i, seq in enumerate(group):
                assert torch.equal(alone[1][i], state[seq]), (group, seq)


@pytest.mark.parametrize("H", (MERGED_H, MIXED_H))
@pytest.mark.parametrize("initial", ["given", "pool_disjoint"])
def test_merged_batch_with_empty_sequences(initial: str, H: int) -> None:
    """Two merged-class long sequences around two empty ones: one split launch over the four sequences -- the
    merged body's at H = 3 (an empty sequence is one item per head with no chunk), the V6 body's at H = 10 (the
    long sequences balanced, an empty sequence one row per head with no chunk); the empty sequences' final states
    are the bitwise copies of their initial states (zeros from a zero record), the long sequences match the FP64
    reference and their output rows are the whole output."""
    c = Case(lens=(8192, 0, 0, 8192), H=H, n=3, initial=initial, final=True, seed=31)
    V6, P3 = loader.ROUTE_BODY_V6, loader.ROUTE_BODY_P3
    assert _classes(c) == [P3, V6, V6, P3]
    assert _launch(c) == ((V6, True) if H == MIXED_H else (P3, True))
    out, state = _run_and_check(c)
    assert out.shape[0] == c.offsets[-1]
    for seq in (1, 2):
        if c.pool:
            row = c.slots[seq]
            before = c.initial_before if c.initial == "pool_same" else c.initial_before
            assert torch.equal(state[row], before[row]), (seq, row)
        else:
            assert torch.equal(state[seq], c.initial_before[seq]), seq


def _count_device_events(call) -> list[str]:
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize(_device())
    try:
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            call()
            torch.cuda.synchronize(_device())
    except Exception as ex:  # noqa: BLE001 - profiler unavailable on this host
        pytest.skip(f"torch.profiler unavailable: {ex}")
    events = [
        e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
    ]
    if not events:
        pytest.skip("torch.profiler captured no device events")
    return [e.name for e in events]


def test_single_launch() -> None:
    """BF16 normalized inputs, FP32 gates, an initial state and a kept final state, the host copy of ``cu_seqlens``
    given and a uniform V6 batch: the call launches the kernel alone (no fill, cast, copy or plan kernel).  Without
    the host copy the call adds the one device-to-host copy of ``cu_seqlens``.  A uniform merged batch adds the
    workspace zero before its one kernel (its item table is cached on the device after the first call), and so does
    a batch with a balanced sequence on the V6 split program.  A mixed batch -- one-token sequences beside a long
    one, and the mixed batch of record -- is likewise one Cake kernel: no second launch, no gather or scatter of its
    tokens, no copy."""
    c = _make(Case(lens=(400,) * 8, H=24, n=3, initial="given", cu_cpu=True, seed=26))
    assert _body(c) == loader.ROUTE_BODY_V6
    _check(c, *_run(c))  # warm-up
    names = _count_device_events(lambda: _run(c))
    assert len(names) == 1 and "cake_gdp" in names[0], names
    c.cu_cpu = False
    names_sync = _count_device_events(lambda: _run(c))
    assert [name for name in names_sync if "cake_gdp" in name] == names, names_sync
    assert sum("Memcpy" in name for name in names_sync) == 1, names_sync
    for lens, H in (
        (MERGED_LENS, MERGED_H),
        (MERGED_LENS, MIXED_H),
        (BENCH_MIXED_LENS, MIXED_H),
        (MIXED_LENS, MIXED_H),
    ):
        m = _make(Case(lens=lens, H=H, n=3, initial="given", cu_cpu=True, seed=26))
        expected = (
            loader.ROUTE_BODY_V6 if H == MIXED_H else loader.ROUTE_BODY_P3,
            True,
        )
        assert _launch(m) == expected, (lens, H)
        _check(m, *_run(m))  # warm-up
        names_m = _count_device_events(lambda m=m: _run(m))
        kernels = [name for name in names_m if "cake_gdp" in name]
        assert len(kernels) == 1 and names_m[-1] == kernels[0], (lens, names_m)
        assert len(names_m) <= 2 and not any("Memcpy" in name for name in names_m), (
            lens,
            names_m,
        )


def _capture(call, *, warm_up: bool):
    """``call`` captured into a CUDA graph on a side stream (after two eager warm-up calls there, the usual
    recipe, when ``warm_up``); the graph and the tensors the captured call returned (graph-pool memory, rewritten
    by every replay)."""
    stream = torch.cuda.Stream(device=_device())
    stream.wait_stream(torch.cuda.current_stream(_device()))
    with torch.cuda.stream(stream):
        for _ in range(2 if warm_up else 0):
            call()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result = call()
    torch.cuda.synchronize(_device())
    return graph, result


@pytest.mark.parametrize("warm", (True, False))
def test_mixed_batch_captures_into_a_cuda_graph(warm: bool) -> None:
    """A mixed bench batch at H = 10 (one-token sequences beside a balanced long one; one V6 split launch) with
    the host copy of ``cu_seqlens`` captures into a CUDA graph -- after eager calls of the batch (its item table
    cached on the device: the graph holds no copy) and as the first call of its batch (the program loaded by an
    eager call of another batch; the table copied in the graph from pinned memory) -- and every replay rewrites the
    eager call's bits (output rows and final states); the eager result satisfies the FP64 reference."""
    other = _make(
        Case(
            lens=BENCH_MIXED_LENS, H=MIXED_H, n=3, initial="given", cu_cpu=True, seed=33
        )
    )
    # loads the program; the batch below is new to the table cache when not warm
    _run(other)
    lens = BENCH_MIXED_LENS if warm else (1, 1, 1, 8191)
    c = _make(Case(lens=lens, H=MIXED_H, n=3, initial="given", cu_cpu=True, seed=34))
    assert _launch(c) == (loader.ROUTE_BODY_V6, True)
    graph, (g_out, g_state) = _capture(lambda: _run(c), warm_up=warm)
    out, state = _run(c)
    _check(c, out, state)
    for _ in range(3):
        g_out.zero_()
        g_state.zero_()
        graph.replay()
        torch.cuda.synchronize(_device())
        assert torch.equal(g_out, out) and torch.equal(g_state, state)


# Batches with a balanced sequence on 132 SMs (a slot per SM of at least 24 chunks, shorter than a head) beside
# one-token, short and empty sequences, n = 1..4: each runs the V6 split program of its flags.
V6_SPLIT_ROWS = (
    (3, 24, (8192,)),
    (3, 24, (1, 1, 8192, 65)),
    (1, 40, (1, 6000, 0, 2)),
    (2, 20, (9000,)),
    (4, 12, (4097, 1)),
    (3, 80, (1, 1, 1, 4096)),
)


@pytest.mark.parametrize("initial", ("none", "given", "pool_same", "pool_disjoint"))
@pytest.mark.parametrize("n,H,lens", V6_SPLIT_ROWS)
def test_v6_split_rows(n: int, H: int, lens: tuple[int, ...], initial: str) -> None:
    """A batch with a balanced sequence runs the V6 split program of its flags and satisfies the FP64 reference
    (output rows, final states, untouched pool rows), with an invalid slot in the disjoint pool."""
    invalid = (0,) if initial == "pool_disjoint" else ()
    c = Case(lens=lens, H=H, n=n, initial=initial, invalid=invalid, seed=40)
    assert _launch(c) == (loader.ROUTE_BODY_V6, True), (n, H, lens)
    _run_and_check(c)


@pytest.mark.parametrize("initial", ("none", "given"))
def test_v6_split_long_sequence_bits_do_not_depend_on_short_rows(initial: str) -> None:
    """The rows of a balanced sequence depend on its own length only, so its output rows and final state are the
    same bits with any number of one-token sequences before it (the one-token rows run on the least-loaded CTAs),
    and the one-token sequences satisfy the FP64 reference."""
    long = _make(Case(lens=(8192,), H=MIXED_H, n=3, initial=initial, seed=41))
    out, state = _run(long)
    for d in (1, 4, 15):
        c = _make(
            Case(lens=(1,) * d + (8192,), H=MIXED_H, n=3, initial=initial, seed=42)
        )
        assert _v6_split(c)
        rows = slice(d, d + 8192)
        c.q[rows], c.k[d * 3 : (d + 8192) * 3] = long.q, long.k
        c.v[d * 3 : (d + 8192) * 3], c.g[rows] = long.v, long.g
        c.beta[d * 3 : (d + 8192) * 3] = long.beta
        if initial == "given":
            c.initial_state[d] = long.initial_state[0]
            c.initial_before = c.initial_state.clone()
        out_d, state_d = _run(c)
        _check(c, out_d, state_d)
        assert torch.equal(out_d[rows], out), d
        assert torch.equal(state_d[d], state[0]), d


# A V6 split graph replayed after the caller rewrote cu_seqlens in place with other lengths (same N and T): the
# kernel compares the planned offsets with the device ones and runs no row; it writes NaN to o[0, 0, :2] and the
# final state [0, 0, 0, 0] and leaves every other value as it was; replayed again on the restored batch, the graph
# gives the eager call's bits.  Each case runs in its own process with CUDA_LAUNCH_BLOCKING=1.
_STALE_CHILD = """
import itertools, json, sys, torch
from tests.gdp.test_cake_gdp_prefill import MIXED_H, Case, _capture, _make, _run, _v6_split
x, y = json.loads(sys.argv[1])
c = _make(Case(lens=tuple(x), H=MIXED_H, n=3, initial="given", cu_cpu=True, seed=43))
assert _v6_split(c)
graph, (g_out, g_state) = _capture(lambda: _run(c), warm_up=True)
eager_out, eager_state = _run(c)
kept = c.cu.clone()
result = {"error": None}
try:
    c.cu.copy_(torch.tensor((0, *itertools.accumulate(y)), dtype=c.cu.dtype))
    g_out.fill_(0.5)
    g_state.fill_(0.5)
    graph.replay()
    torch.cuda.synchronize()
    result["marker"] = bool(g_out[0, 0, :2].isnan().all() and g_state[0, 0, 0, 0].isnan())
    g_out[0, 0, :2] = 0.5
    g_state[0, 0, 0, 0] = 0.5
    result["rest_kept"] = bool((g_out == 0.5).all() and (g_state == 0.5).all())
    c.cu.copy_(kept)
    graph.replay()
    torch.cuda.synchronize()
    result["restored"] = bool(torch.equal(g_out, eager_out) and torch.equal(g_state, eager_state))
except Exception as exc:
    result["error"] = str(exc).splitlines()[0]
try:
    result["context_alive"] = float(torch.ones(4, device="cuda").sum()) == 4.0
except Exception:
    result["context_alive"] = False
print("STALE " + json.dumps(result))
"""


@pytest.mark.parametrize(
    "captured,replayed",
    [
        ([1, 1, 1, 8189], [8189, 1, 1, 1]),
        ([8192, 0], [0, 8192]),
        ([8192, 8192], [8000, 8384]),
    ],
    ids=["tiny-moves", "to-empty", "rebalance"],
)
def test_a_v6_split_graph_replayed_on_other_lengths_fails_explicitly(
    captured, replayed
) -> None:
    env = {**os.environ, "CUDA_LAUNCH_BLOCKING": "1"}
    root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [sys.executable, "-c", _STALE_CHILD, json.dumps([captured, replayed])],
        capture_output=True,
        text=True,
        timeout=900,
        env=env,
        cwd=root,
    )
    lines = [line for line in proc.stdout.splitlines() if line.startswith("STALE ")]
    assert lines, (
        f"the child process died (exit {proc.returncode}): {proc.stderr[-2000:]}"
    )
    result = json.loads(lines[-1][len("STALE ") :])
    assert result == {
        "error": None,
        "marker": True,
        "rest_kept": True,
        "restored": True,
        "context_alive": True,
    }, result


# ---------------------------------------------------------------------------
# Routes, programs, package contract
# ---------------------------------------------------------------------------


def test_every_prefill_program_is_launched(monkeypatch) -> None:
    """One call reaches every program of the ``prefill`` route: the three V6 programs through short batches with
    no / a given / a pooled initial state (a short batch keeping no final state runs the V6 plain program again,
    its final state discarded), the two unit programs, the three merged split programs through merged-class long
    batches at H = 3 and the three V6 split programs through a balanced sequence at H = 10, both with the host
    copy of ``cu_seqlens``; every uniform batch is one launch whose record names the body and the split flag of
    its launch."""
    arch = _arch()
    launched: list[str] = []
    original = loader.run_module

    def recording(name, values, **kwargs):
        launched.append(name)
        return original(name, values, **kwargs)

    monkeypatch.setattr(loader, "run_module", recording)
    V6, P3 = loader.ROUTE_BODY_V6, loader.ROUTE_BODY_P3
    for c, body in (
        (Case(lens=(129, 65), H=3, n=3, initial="none", final=True), V6),
        (Case(lens=(129, 65), H=3, n=3, initial="given", final=True), V6),
        (Case(lens=(129, 65), H=3, n=3, initial="pool_disjoint", final=True), V6),
        (Case(lens=(129, 65), H=3, n=3, initial="none", final=False), V6),
        (
            Case(lens=(129, 65), H=3, n=3, initial="none", final=True, unit_g=True),
            P3,
        ),
        (
            Case(lens=(129, 65), H=3, n=3, initial="none", final=True, unit_beta=True),
            P3,
        ),
        # the split programs: the host copy of cu_seqlens and a merged-class batch
        (
            Case(lens=MERGED_LENS, H=3, n=3, initial="none", final=True, cu_cpu=True),
            P3,
        ),
        (
            Case(lens=MERGED_LENS, H=3, n=3, initial="given", final=True, cu_cpu=True),
            P3,
        ),
        (
            Case(
                lens=MERGED_LENS,
                H=3,
                n=3,
                initial="pool_disjoint",
                final=True,
                cu_cpu=True,
            ),
            P3,
        ),
        # the V6 split programs: the host copy of cu_seqlens and a balanced sequence
        *(
            (
                Case(lens=(8192,), H=MIXED_H, n=3, initial=initial, cu_cpu=True),
                V6,
            )
            for initial in ("none", "given", "pool_disjoint")
        ),
    ):
        c.seed = 28
        body_launched, split = _launch(c)
        assert body_launched == body, (c.lens, c.initial, c.final)
        if not (body == V6 and split):
            assert _classes(c) == [body] * c.N, (c.lens, c.initial, c.final)
            assert _body(c) == body, (c.lens, c.initial, c.final)
        before = len(launched)
        _run_and_check(c)
        assert len(launched) == before + 1, launched  # a uniform batch is one launch
        record = loader.MODULES[launched[-1]]
        assert record["body"] == body, (c.lens, launched[-1])
        assert bool(record["build"].get(loader.PREFILL_SPLIT_FLAG)) == split
    prefill = {
        name
        for name, record in loader.MODULES.items()
        if record["role"] == "prefill" and arch in record["arches"]
    }
    assert set(launched) == prefill, (sorted(set(launched)), sorted(prefill))
    assert len(prefill) == len(V6_FLAGS) + len(UNIT_FLAGS) + len(SPLIT_FLAGS) + len(
        V6_SPLIT_FLAGS
    )
    split = [
        name
        for name in launched
        if loader.MODULES[name]["build"].get(loader.PREFILL_SPLIT_FLAG)
    ]
    count = len(SPLIT_FLAGS) + len(V6_SPLIT_FLAGS)
    assert len(split) == count and launched[-count:] == split, launched


def test_split_programs_follow_the_host_split_rule(monkeypatch) -> None:
    """The per-class launch law over the host copy of ``cu_seqlens``: a merged-class long batch runs the split
    program of its flags with the rule's item table as ``output_slot_indices`` and the items as the tiles; a
    V6-class short batch runs the V6 program of its flags over ``N x H`` tiles without a table; a call without
    ``cu_seqlens_cpu`` synchronizes the copy and selects the same program with the same bits; an int64 host copy
    is accepted; a unit long batch runs its exact unit program and a long batch keeping no state the V6 plain
    program, both over ``N x H`` tiles; the mixed batch of record (balanced at H = 10) is one recorded launch of
    the V6 split program over the rows of ``prefill_v6_sm90_plan`` of the whole batch (every sequence, the short
    and the empty one included) on the plan's grid, its item words ``v6_sm90_item_words``, satisfying the FP64
    reference; ``cu_seqlens_cpu`` is checked for device, rank, length and dtype."""
    seen: list[tuple[str, dict]] = []
    # total_tiles and the item words of each launch
    items: list[tuple[int, tuple[int, ...]]] = []
    original = loader.run_module

    def recording(name, values, **kwargs):
        items.append(
            (values["total_tiles"], tuple(values["output_slot_indices"].tolist()))
        )
        seen.append(
            (
                name,
                {
                    "tiles": kwargs.get("tiles"),
                    "slots": values["output_slot_indices"].numel(),
                    "ws": values["ainv_workspace"].numel(),
                    "num_seqs": values["num_seqs"],
                },
            )
        )
        return original(name, values, **kwargs)

    monkeypatch.setattr(loader, "run_module", recording)
    arch = _arch()
    sm_count = loader.sm_count(0)
    V6, P3 = loader.ROUTE_BODY_V6, loader.ROUTE_BODY_P3
    for initial in ("none", "given"):
        long = Case(
            lens=MERGED_LENS,
            H=3,
            n=3,
            initial=initial,
            final=True,
            cu_cpu=True,
            seed=29,
        )
        assert _classes(long) == [P3] * long.N
        out, state = _run_and_check(long)
        table = loader.prefill_split_table(
            long.offsets,
            num_heads=3,
            num_householder=3,
            sm_count=sm_count,
            total_tokens=long.T,
        )
        assert table is not None and len(table) > long.N * 3
        name, launch = seen[-1]
        assert name == loader.selected_prefill_module(
            arch,
            body=P3,
            use_initial_state=initial == "given",
            use_state_indices=False,
            store_final_state=True,
            split_items=True,
        )
        assert launch == {
            "tiles": len(table),
            "slots": 4 * len(table),
            "ws": 4 * (long.N * 3 + 2 * len(table)),
            "num_seqs": long.N,
        }
        long.cu_cpu = (
            False  # the backend reads cu_seqlens.cpu(): the same program, the same bits
        )
        out_sync, state_sync = _run(long)
        assert (
            seen[-1] == (name, launch)
            and torch.equal(out_sync, out)
            and torch.equal(state_sync, state)
        )
        long.cu_cpu, long.cu_host = (
            True,
            long.cu_host.to(torch.int64),
        )  # an int64 host copy is accepted
        out_i64, state_i64 = _run(long)
        assert (
            seen[-1] == (name, launch)
            and torch.equal(out_i64, out)
            and torch.equal(state_i64, state)
        )
        short = Case(
            lens=(129, 65), H=3, n=3, initial=initial, final=True, cu_cpu=True, seed=29
        )
        assert _classes(short) == [V6] * short.N
        _run_and_check(short)
        name, launch = seen[-1]
        assert name == loader.selected_prefill_module(
            arch,
            body=V6,
            use_initial_state=initial == "given",
            use_state_indices=False,
            store_final_state=True,
        )
        # no item table: formal 9 takes the one-element slot placeholder, the workspace its 16-byte placeholder
        assert launch == {
            "tiles": short.N * 3,
            "slots": 1,
            "ws": 16,
            "num_seqs": short.N,
        }
    # a unit long batch is merged-class at every length: its exact unit program over N x H tiles
    unit = Case(
        lens=MERGED_LENS,
        H=3,
        n=3,
        initial="none",
        final=True,
        unit_g=True,
        cu_cpu=True,
        seed=29,
    )
    assert _classes(unit) == [P3] * unit.N
    _run_and_check(unit)
    assert seen[-1][0] == loader.selected_prefill_module(
        arch,
        body=P3,
        use_initial_state=False,
        use_state_indices=False,
        store_final_state=True,
        unit_gates=True,
    )
    assert seen[-1][1]["tiles"] == unit.N * 3 and seen[-1][1]["slots"] == 1
    # a long batch keeping no state is V6-class at every length: the V6 plain program over N x H tiles
    plain = Case(lens=MERGED_LENS, H=3, n=3, initial="none", final=False, seed=29)
    assert _classes(plain) == [V6] * plain.N
    _run_and_check(plain)
    assert seen[-1][0] == loader.selected_prefill_module(
        arch,
        body=V6,
        use_initial_state=False,
        use_state_indices=False,
        store_final_state=True,
    )
    assert seen[-1][1]["tiles"] == plain.N * 3 and seen[-1][1]["slots"] == 1
    # the mixed batch of record: one launch of the V6 split program over the rows of the whole batch
    mixed = Case(
        lens=MIXED_LENS,
        H=MIXED_H,
        n=3,
        initial="given",
        final=True,
        cu_cpu=True,
        seed=29,
    )
    assert _classes(mixed) == [P3, V6, V6, P3]
    assert _launch(mixed) == (V6, True)
    before = len(seen)
    _run_and_check(mixed)
    launches = seen[before:]
    plan = loader.prefill_v6_sm90_plan(
        mixed.offsets,
        num_heads=MIXED_H,
        num_householder=3,
        sm_count=sm_count,
        total_tokens=mixed.T,
    )
    words = loader.v6_sm90_item_words(plan, mixed.offsets)
    assert plan.split and plan.grid == sm_count
    assert {seq for seq, head, cut, span in plan.rows} == set(range(mixed.N))
    assert len(plan.offsets) == plan.grid + 1 and plan.offsets[-1] == len(plan.rows)
    assert words[-(mixed.N + 2) :] == (mixed.N, *mixed.offsets)
    assert launches == [
        (
            loader.selected_prefill_module(
                arch,
                body=V6,
                use_initial_state=True,
                use_state_indices=False,
                store_final_state=True,
                split_items=True,
            ),
            {
                "tiles": plan.grid,
                "slots": len(words),
                "ws": max(16, 4 * mixed.N * MIXED_H),
                "num_seqs": mixed.N,
            },
        )
    ]
    assert items[-1] == (len(plan.rows), words)
    c = _make(Case(lens=(129, 65), H=3, n=3, seed=29))

    def call(cu_seqlens_cpu):
        return cake_chunk_gated_delta_product(
            c.q,
            c.k,
            c.v,
            g=c.g,
            beta=c.beta,
            num_householder=3,
            cu_seqlens=c.cu,
            cu_seqlens_cpu=cu_seqlens_cpu,
        )

    for bad in (
        c.offsets,
        c.cu,
        c.cu_host[None],
        c.cu_host.to(torch.float32),
        c.cu_host[:1],
    ):
        with pytest.raises(
            ValueError, match="cu_seqlens_cpu must be a CPU int32/int64 1-D tensor"
        ):
            call(bad)
    with pytest.raises(ValueError, match="cu_seqlens_cpu must hold the 3 entries"):
        call(c.cu_host[:2])


def test_route_tables_are_consistent() -> None:
    """The ``prefill`` route of the device: the three V6 records (every one storing a final state, no unit
    program, no segment pass), the two unit records and the three split records of the merged body, then the
    three V6 split records, in that order; ``selected_prefill_module`` resolves every flag set of each body and
    refuses the removed merged exact programs, a V6 unit program, a V6 program without a final state, an unknown
    further flag and both unit flags; the grid rule, the route predicate and the logical names."""
    arch = _arch()
    routes = loader.ROUTES[arch]
    assert [key for key in routes if key.startswith("prefill")] == [
        loader.PREFILL_ROUTE
    ]  # no segmented path on SM90
    single = routes[loader.PREFILL_ROUTE]
    V6, P3 = loader.ROUTE_BODY_V6, loader.ROUTE_BODY_P3
    assert loader.ROUTE_BODIES == (P3, V6)
    assert len(single) == len(V6_FLAGS) + len(UNIT_FLAGS) + len(SPLIT_FLAGS) + len(
        V6_SPLIT_FLAGS
    )
    v6_flags = set()
    unit_flags = set()
    split_flags = set()
    v6_split_flags = set()
    for name in single:
        record = loader.MODULES[name]
        assert (
            record["role"] == "prefill"
            and record["body"] in loader.ROUTE_BODIES
            and record["arches"] == [arch]
        )
        assert record["grid_rule"] == {"kind": "tiles"}
        rule = record["select_rule"]
        assert (
            rule["kind"] == "prefill_flags"
            and not rule["segment_pass"]
            and not (rule["unit_gates"] and rule["unit_beta"])
        )
        assert {flag: record["build"][flag] for flag in loader.PREFILL_FLAGS} == {
            flag: rule[flag] for flag in loader.PREFILL_FLAGS
        }
        five = (
            rule["use_initial_state"],
            rule["use_state_indices"],
            rule["store_final_state"],
            rule["unit_gates"],
            rule["unit_beta"],
        )
        if record["body"] == V6:
            # a V6 record stores the final state and runs no unit program; a V6 split record spells the split flag
            # in its rule too, the others exactly the six flags
            assert (
                rule["store_final_state"]
                and not rule["unit_gates"]
                and not rule["unit_beta"]
            )
            assert record["launch"]["block"] == (384, 1, 1)
            if record["build"].get(loader.PREFILL_SPLIT_FLAG):
                assert (
                    set(rule)
                    == {"kind", loader.PREFILL_SPLIT_FLAG, *loader.PREFILL_FLAGS}
                    and rule[loader.PREFILL_SPLIT_FLAG] is True
                )
                v6_split_flags.add(five)
            else:
                assert set(rule) == {"kind", *loader.PREFILL_FLAGS}
                v6_flags.add(five)
        elif record["build"].get(loader.PREFILL_SPLIT_FLAG):
            # a split program spells the flag in its rule
            assert (
                set(rule) == {"kind", loader.PREFILL_SPLIT_FLAG, *loader.PREFILL_FLAGS}
                and rule[loader.PREFILL_SPLIT_FLAG] is True
            )
            split_flags.add(five)
        else:
            # the exact programs of the merged body are the unit programs alone
            assert set(rule) == {"kind", *loader.PREFILL_FLAGS}
            assert rule["unit_gates"] or rule["unit_beta"]
            unit_flags.add(five)
    assert v6_flags == V6_FLAGS and unit_flags == UNIT_FLAGS
    assert split_flags == SPLIT_FLAGS and v6_split_flags == V6_SPLIT_FLAGS
    assert [
        (
            loader.MODULES[name]["body"],
            loader.MODULES[name]["build"].get(loader.PREFILL_SPLIT_FLAG, False),
        )
        for name in single
    ] == [(V6, False)] * len(V6_FLAGS) + [(P3, False)] * len(UNIT_FLAGS) + [
        (P3, True)
    ] * len(SPLIT_FLAGS) + [(V6, True)] * len(V6_SPLIT_FLAGS)
    for body, table, extra in (
        (V6, V6_FLAGS, {}),
        (P3, UNIT_FLAGS, {}),
        (P3, SPLIT_FLAGS, {loader.PREFILL_SPLIT_FLAG: True}),
        (V6, V6_SPLIT_FLAGS, {loader.PREFILL_SPLIT_FLAG: True}),
    ):
        for use_initial, use_indices, store, unit_g, unit_beta in table:
            name = loader.selected_prefill_module(
                arch,
                body=body,
                use_initial_state=use_initial,
                use_state_indices=use_indices,
                store_final_state=store,
                unit_gates=unit_g,
                unit_beta=unit_beta,
                **extra,
            )
            record = loader.MODULES[name]
            assert name in single and record["body"] == body
            assert bool(record["build"].get(loader.PREFILL_SPLIT_FLAG)) == bool(extra)

    def refused(**kwargs) -> None:
        with pytest.raises(ValueError, match="0 programs of the 'prefill' route"):
            loader.selected_prefill_module(arch, **kwargs)

    # the merged body routes no exact gate-tensor program any more (plain, h0, pool, no-final); without a body the
    # search spans every body: the stored-final sets resolve to the V6 program of the flags alone, no-final refuses
    for use_initial, use_indices, store in (
        (False, False, True),
        (True, False, True),
        (True, True, True),
        (False, False, False),
    ):
        flags = dict(
            use_initial_state=use_initial,
            use_state_indices=use_indices,
            store_final_state=store,
        )
        if store:
            name = loader.selected_prefill_module(arch, **flags)
            assert loader.MODULES[name]["body"] == V6
        else:
            refused(**flags)
        refused(body=P3, **flags)
    # no V6 unit program; no V6 program without a final state, split or not
    refused(
        body=V6,
        use_initial_state=False,
        use_state_indices=False,
        store_final_state=True,
        unit_gates=True,
    )
    refused(
        body=V6,
        use_initial_state=False,
        use_state_indices=False,
        store_final_state=True,
        unit_beta=True,
    )
    refused(
        body=V6,
        use_initial_state=False,
        use_state_indices=False,
        store_final_state=False,
    )
    refused(
        body=V6,
        use_initial_state=False,
        use_state_indices=False,
        store_final_state=False,
        split_items=True,
    )
    # slot indices without an initial state, in either body
    for body in loader.ROUTE_BODIES:
        refused(
            body=body,
            use_initial_state=False,
            use_state_indices=True,
            store_final_state=True,
        )
    # g=None with beta=None is no shipped program
    refused(
        use_initial_state=False,
        use_state_indices=False,
        store_final_state=True,
        unit_gates=True,
        unit_beta=True,
    )
    # a unit program with an initial state
    refused(
        use_initial_state=True,
        use_state_indices=False,
        store_final_state=True,
        unit_gates=True,
    )
    # no split program of the unit-gate flags
    refused(
        use_initial_state=False,
        use_state_indices=False,
        store_final_state=True,
        unit_gates=True,
        split_items=True,
    )
    # a further flag no record spells on, in either body
    for body in loader.ROUTE_BODIES:
        refused(
            body=body,
            use_initial_state=False,
            use_state_indices=False,
            store_final_state=True,
            split_rows=True,
        )
    # an unknown body
    refused(
        body="prefill_sm100",
        use_initial_state=False,
        use_state_indices=False,
        store_final_state=True,
    )
    base = loader.selected_prefill_module(
        arch,
        body=V6,
        use_initial_state=False,
        use_state_indices=False,
        store_final_state=True,
    )
    merged = loader.selected_prefill_module(
        arch,
        body=P3,
        use_initial_state=False,
        use_state_indices=False,
        store_final_state=True,
        unit_gates=True,
    )
    for name in (base, merged):
        record = loader.MODULES[name]
        assert loader.launch_grid(record, 1, 1, tiles=1000, sm_count=132) == (
            132,
            1,
            1,
        )
        assert loader.launch_grid(record, 1, 1, tiles=7, sm_count=132) == (7, 1, 1)
        with pytest.raises(ValueError, match="tiles and sm_count"):
            loader.launch_grid(record, 1, 1)
    for n in NUM_HOUSEHOLDER:
        assert loader.prefill_route_applies(
            device_capability=_capability(), num_householder=n
        )
    assert not loader.prefill_route_applies(
        device_capability=_capability(), num_householder=5
    )
    assert loader.prefill_route_applies(device_capability=(9, 0), num_householder=1)
    assert not loader.prefill_route_applies(
        device_capability=(12, 0), num_householder=1
    )
    assert loader.PREFILL_LOGICAL_TENSORS[:4] == ("q", "k", "v", "o")
    # the logical names of the shared prefill vocabulary but the Blackwell body's own (its item table, workspace,
    # flags word and counts, and the pool programs' write slots, rounding seed and Philox rounds)
    blackwell_only = {
        "items",
        "split_workspace",
        "state_stride_slot",
        "num_items",
        "flags",
        "output_state_indices",
        "rand_seed",
        "philox_rounds",
    }
    for name in (base, merged):
        assert (
            set(loader.MODULES[name]["logical"].values())
            == (
                set(loader.PREFILL_LOGICAL_TENSORS)
                | set(loader.PREFILL_LOGICAL_SCALARS)
            )
            - blackwell_only
        )
    # both bodies take the same 23 formals in the same order: one ``values`` dict serves every launch
    assert loader.MODULES[base]["logical"] == loader.MODULES[merged]["logical"]
    assert loader.MODULES[base]["arg_plan"] == loader.MODULES[merged]["arg_plan"]


@pytest.mark.parametrize("H", (MERGED_H, MIXED_H))
def test_no_final_state_runs_the_v6_body(monkeypatch, H: int) -> None:
    """A long batch reading no initial state and keeping no final state is V6-class at every length (no merged
    program stores none, and the V6 step is the cheaper): at H = 3 the V6 plain program runs it over ``N x H``
    tiles, at H = 10 (its sequences balanced) the V6 split program of the plain flags on one CTA per SM; the final
    state goes to the discard record and the output satisfies the FP64 reference.  The same lengths with a final
    state are merged-class."""
    seen: list[tuple[str, Optional[int]]] = []
    original = loader.run_module

    def recording(name, values, **kwargs):
        seen.append((name, kwargs.get("tiles")))
        return original(name, values, **kwargs)

    monkeypatch.setattr(loader, "run_module", recording)
    V6, P3 = loader.ROUTE_BODY_V6, loader.ROUTE_BODY_P3
    c = Case(
        lens=MERGED_LENS,
        H=H,
        n=3,
        initial="none",
        final=False,
        cu_cpu=True,
        seed=31,
    )
    assert _classes(c) == [V6] * c.N
    kept = Case(lens=MERGED_LENS, H=H, n=3, initial="none", final=True)
    assert _classes(kept) == [P3] * kept.N
    split = H == MIXED_H
    assert _launch(c) == (V6, split)
    _run_and_check(c)
    assert seen == [
        (
            loader.selected_prefill_module(
                _arch(),
                body=V6,
                use_initial_state=False,
                use_state_indices=False,
                store_final_state=True,
                **({loader.PREFILL_SPLIT_FLAG: True} if split else {}),
            ),
            loader.sm_count(0) if split else c.N * c.H,
        )
    ]


def test_prefill_records_reference_the_build_manifest() -> None:
    """Every prefill record of this device's architecture names its ``build.json`` library; the entries compile
    with fast math (their validated setting).  (The Blackwell body's records are covered by
    ``test_cake_gdp_prefill_bw.py``.)"""
    libraries = cake_build.libraries_by_name()
    package = Path(cake_build.__file__).parent
    seen = 0
    v6_seen = 0
    for name, record in loader.MODULES.items():
        if record["role"] != "prefill" or _arch() not in record["arches"]:
            continue
        seen += 1
        assert record["library"] == record["kernel"] == name
        # two prefill bodies: the merged body and the V6 fork-order body (three flag sets and their split twins)
        assert record["body"] in (loader.ROUTE_BODY_P3, loader.ROUTE_BODY_V6), (
            name,
            record["body"],
        )
        v6_seen += record["body"] == loader.ROUTE_BODY_V6
        for arch in record["arches"]:
            entry = libraries[loader.library_name(name, arch)]
            assert entry["arch"] == arch and entry["entry"] == "run"
            assert entry["sources"] == [
                f"csrc/{name}_kernel.cu",
                f"csrc/{name}_binding.cu",
            ]
            for source in entry["sources"]:
                assert (package / source).is_file(), source
            assert re.fullmatch(r"[0-9a-f]{64}", entry["closure_sha256"])
            assert "--use_fast_math" in entry["nvcc_flags"], (name, entry["nvcc_flags"])
    assert seen == len(ROUTED_FLAGS) + len(SPLIT_FLAGS) + len(V6_SPLIT_FLAGS)
    assert v6_seen == len(V6_FLAGS) + len(V6_SPLIT_FLAGS), v6_seen


def test_is_supported_reports_the_routes() -> None:
    assert is_cake_gdp_prefill_supported(_device())
    for n in NUM_HOUSEHOLDER:
        assert is_cake_gdp_prefill_supported(_device(), n)
    assert not is_cake_gdp_prefill_supported(_device(), 5)
    assert not is_cake_gdp_prefill_supported(torch.device("cpu"))


def test_rejects_invalid_arguments() -> None:
    c = _make(Case(lens=(129, 65), H=3, n=3, initial="given", seed=29))
    device = _device()

    def call(q=None, k=None, v=None, **overrides):
        kwargs = dict(
            g=c.g,
            beta=c.beta,
            num_householder=c.n,
            initial_state=c.initial_state,
            output_final_state=True,
            cu_seqlens=c.cu,
        )
        kwargs.update(overrides)
        return cake_chunk_gated_delta_product(
            c.q if q is None else q,
            c.k if k is None else k,
            c.v if v is None else v,
            **kwargs,
        )

    with pytest.raises(ValueError, match=r"\[total_seq_len, H, 128\]"):
        cake_chunk_gated_delta_product(
            c.q.unsqueeze(0), c.k, c.v, cu_seqlens=c.cu, num_householder=c.n
        )
    with pytest.raises(ValueError, match="head size 128"):
        cake_chunk_gated_delta_product(
            c.q[:, :, :64].contiguous(), c.k, c.v, cu_seqlens=c.cu, num_householder=c.n
        )
    with pytest.raises(ValueError, match="bfloat16 or float16"):
        call(q=c.q.float(), k=c.k, v=c.v)
    with pytest.raises(ValueError, match="num_householder must be one of"):
        call(num_householder=5)
    with pytest.raises(ValueError, match="one head count"):
        call(k=c.k[:, :2].contiguous())
    with pytest.raises(ValueError, match="cu_seqlens is required"):
        call(cu_seqlens=None)
    with pytest.raises(ValueError, match="int32 or int64"):
        call(cu_seqlens=c.cu.float())
    with pytest.raises(ValueError, match="k must have shape"):
        call(k=c.k[: c.T].contiguous())
    with pytest.raises(ValueError, match="g must have shape"):
        call(g=c.g[:, :2].contiguous())
    with pytest.raises(ValueError, match="beta must be"):
        call(beta=c.beta.to(torch.float64))
    with pytest.raises(ValueError, match="initial_state must have shape"):
        call(initial_state=c.initial_state[:1])
    with pytest.raises(ValueError, match="initial_state must be float32"):
        call(initial_state=c.initial_state.to(torch.float16))
    with pytest.raises(ValueError, match="output_state must be float32"):
        call(output_state=c.initial_state.to(torch.bfloat16))
    with pytest.raises(ValueError, match="output_state must be contiguous"):
        call(output_state=torch.zeros((c.N, c.H, V, 2 * K), device=device)[:, :, :, :K])
    with pytest.raises(ValueError, match="explicit output_state pool"):
        call(state_indices=torch.zeros((c.N,), device=device, dtype=torch.int32))
    with pytest.raises(ValueError, match="initial_state must be float32"):
        call(
            initial_state=c.initial_state.to(torch.bfloat16),
            state_indices=torch.zeros((c.N,), device=device, dtype=torch.int32),
            output_state=c.initial_state,
        )
    with pytest.raises(ValueError, match="state_indices must have shape"):
        call(
            state_indices=torch.zeros((c.N + 1,), device=device, dtype=torch.int32),
            output_state=c.initial_state,
        )
    with pytest.raises(ValueError, match="output must have shape"):
        call(output=torch.zeros((c.T, c.H, 2 * V), device=device, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="requires CUDA tensors"):
        cake_chunk_gated_delta_product(
            c.q.cpu(), c.k.cpu(), c.v.cpu(), cu_seqlens=c.cu.cpu(), num_householder=c.n
        )
