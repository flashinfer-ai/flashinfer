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
bitwise reproducibility, the launch count and one launch of every exported
prefill program.

This suite exercises the backend package alone: it names the package once,
through ``tests.test_helpers.cake_package``, and imports nothing else from
flashinfer, so a build that vendors the package under another import path runs
it unchanged after pointing that table entry at its package.  The
``flashinfer.gdp_prefill.chunk_gated_delta_product(backend="cake")`` route is
covered by ``test_cake_gdp_prefill_api.py``.
"""

from __future__ import annotations

import dataclasses
import itertools
import re
from pathlib import Path
from typing import Optional

import pytest
import torch

from tests.test_helpers.cake_package import cake_package

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
# unit_gates, unit_beta): the gate-tensor programs and the unit programs from a zero state.
ROUTED_FLAGS = {
    (False, False, True, False, False),
    (True, False, True, False, False),
    (True, True, True, False, False),
    (False, False, False, False, False),
    (
        False,
        False,
        True,
        True,
        False,
    ),  # g=None: the unit-gate program starts from zeros and stores the final state
    (False, False, True, False, True),  # beta=None: the unit-beta program
}
# The gate-saturation split programs (``split_items``): the three gated flag sets that store a final state, reached
# when the split rule of record over the host copy of ``cu_seqlens`` (``cu_seqlens_cpu``, else ``cu_seqlens.cpu()``)
# cuts a long sequence into items.
SPLIT_FLAGS = {
    (False, False, True, False, False),
    (True, False, True, False, False),
    (True, True, True, False, False),
}
SPLIT_LENS = (
    4097,
    1000,
)  # a batch the rule cuts on every exported device (items > N x H)


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
    given and a batch the split rule leaves whole: the call launches the kernel alone (no fill, cast, copy or plan
    kernel).  Without the host copy the call adds the one device-to-host copy of ``cu_seqlens``.  A batch the rule
    cuts adds the item table upload and the workspace zero before its one kernel."""
    sm_count = loader.sm_count(0)
    c = _make(Case(lens=(400,) * 8, H=24, n=3, initial="given", cu_cpu=True, seed=26))
    assert (
        loader.prefill_split_table(
            c.offsets,
            num_heads=24,
            num_householder=3,
            sm_count=sm_count,
            total_tokens=c.T,
        )
        is None
    )
    _check(c, *_run(c))  # warm-up
    names = _count_device_events(lambda: _run(c))
    assert len(names) == 1 and "cake_gdp" in names[0], names
    c.cu_cpu = False
    names_sync = _count_device_events(lambda: _run(c))
    assert [name for name in names_sync if "cake_gdp" in name] == names, names_sync
    assert sum("Memcpy" in name for name in names_sync) == 1, names_sync
    s = _make(Case(lens=(1024,) * 4, H=24, n=3, initial="given", cu_cpu=True, seed=26))
    assert (
        loader.prefill_split_table(
            s.offsets,
            num_heads=24,
            num_householder=3,
            sm_count=sm_count,
            total_tokens=s.T,
        )
        is not None
    )
    _check(s, *_run(s))  # warm-up
    names_split = _count_device_events(lambda: _run(s))
    kernels = [name for name in names_split if "cake_gdp" in name]
    assert (
        len(kernels) == 1 and names_split[-1] == kernels[0] and len(names_split) <= 3
    ), names_split


# ---------------------------------------------------------------------------
# Routes, programs, package contract
# ---------------------------------------------------------------------------


def test_every_prefill_program_is_launched(monkeypatch) -> None:
    arch = _arch()
    launched: list[str] = []
    original = loader.run_module

    def recording(name, values, **kwargs):
        launched.append(name)
        return original(name, values, **kwargs)

    monkeypatch.setattr(loader, "run_module", recording)
    for c in (
        Case(lens=(129, 65), H=3, n=3, initial="none", final=True),
        Case(lens=(129, 65), H=3, n=3, initial="given", final=True),
        Case(lens=(129, 65), H=3, n=3, initial="pool_disjoint", final=True),
        Case(lens=(129, 65), H=3, n=3, initial="none", final=False),
        Case(lens=(129, 65), H=3, n=3, initial="none", final=True, unit_g=True),
        Case(lens=(129, 65), H=3, n=3, initial="none", final=True, unit_beta=True),
        # the split programs: the host copy of cu_seqlens and a batch the split rule cuts
        Case(lens=SPLIT_LENS, H=3, n=3, initial="none", final=True, cu_cpu=True),
        Case(lens=SPLIT_LENS, H=3, n=3, initial="given", final=True, cu_cpu=True),
        Case(
            lens=SPLIT_LENS, H=3, n=3, initial="pool_disjoint", final=True, cu_cpu=True
        ),
    ):
        c.seed = 28
        _run_and_check(c)
    prefill = {
        name
        for name, record in loader.MODULES.items()
        if record["role"] == "prefill" and arch in record["arches"]
    }
    assert set(launched) == prefill, (sorted(set(launched)), sorted(prefill))
    split = [
        name
        for name in launched
        if loader.MODULES[name]["build"].get(loader.PREFILL_SPLIT_FLAG)
    ]
    assert len(split) == len(SPLIT_FLAGS) and launched[-len(SPLIT_FLAGS) :] == split, (
        launched
    )


def test_split_programs_follow_the_host_split_rule(monkeypatch) -> None:
    """The split rule of record over the host copy of ``cu_seqlens`` selects a split program exactly when
    ``prefill_split_table`` cuts (a long batch), with the rule's item table as ``output_slot_indices`` and the items
    as the tiles; a short batch runs the exact program with the same inputs; a call without ``cu_seqlens_cpu``
    synchronizes the copy and selects the same program with the same bits; a long batch whose flags have no split
    program (a unit program, the plain call without a final state) runs its exact program; the results agree with
    the FP64 reference either way; ``cu_seqlens_cpu`` is checked for device, rank, length and dtype."""
    seen: list[tuple[str, dict]] = []
    original = loader.run_module

    def recording(name, values, **kwargs):
        seen.append(
            (
                name,
                {
                    "tiles": kwargs.get("tiles"),
                    "slots": values["output_slot_indices"].numel(),
                    "ws": values["ainv_workspace"].numel(),
                },
            )
        )
        return original(name, values, **kwargs)

    monkeypatch.setattr(loader, "run_module", recording)
    sm_count = loader.sm_count(0)
    for initial in ("none", "given"):
        long = Case(
            lens=SPLIT_LENS, H=3, n=3, initial=initial, final=True, cu_cpu=True, seed=29
        )
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
        assert loader.MODULES[name]["build"][loader.PREFILL_SPLIT_FLAG] is True
        assert launch == {
            "tiles": len(table),
            "slots": 4 * len(table),
            "ws": 4 * (long.N * 3 + 2 * len(table)),
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
        _run_and_check(short)
        assert (
            loader.prefill_split_table(
                short.offsets,
                num_heads=3,
                num_householder=3,
                sm_count=sm_count,
                total_tokens=short.T,
            )
            is None
        )
        name, launch = seen[-1]
        assert (
            not loader.MODULES[name]["build"].get(loader.PREFILL_SPLIT_FLAG)
            and launch["tiles"] == short.N * 3
        )
    # unit programs and the plain no-final program have no split: the exact program runs the long batch
    _run_and_check(
        Case(
            lens=SPLIT_LENS,
            H=3,
            n=3,
            initial="none",
            final=True,
            unit_g=True,
            cu_cpu=True,
            seed=29,
        )
    )
    assert (
        not loader.MODULES[seen[-1][0]]["build"].get(loader.PREFILL_SPLIT_FLAG)
        and seen[-1][1]["tiles"] == 2 * 3
    )
    _run_and_check(
        Case(lens=SPLIT_LENS, H=3, n=3, initial="none", final=False, seed=29)
    )
    assert (
        not loader.MODULES[seen[-1][0]]["build"].get(loader.PREFILL_SPLIT_FLAG)
        and seen[-1][1]["tiles"] == 2 * 3
    )
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
    arch = _arch()
    routes = loader.ROUTES[arch]
    assert [key for key in routes if key.startswith("prefill")] == [
        loader.PREFILL_ROUTE
    ]  # no segmented path on SM90
    single = routes[loader.PREFILL_ROUTE]
    assert len(single) == len(ROUTED_FLAGS) + len(SPLIT_FLAGS)
    flags = set()
    split_flags = set()
    for name in single:
        record = loader.MODULES[name]
        assert (
            record["role"] == "prefill"
            and record["body"] == "prefill_sm90"
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
        if record["build"].get(loader.PREFILL_SPLIT_FLAG):
            # a split program spells the flag in its rule; the exact programs spell no split_items at all
            assert (
                set(rule) == {"kind", loader.PREFILL_SPLIT_FLAG, *loader.PREFILL_FLAGS}
                and rule[loader.PREFILL_SPLIT_FLAG] is True
            )
            split_flags.add(five)
        else:
            assert set(rule) == {"kind", *loader.PREFILL_FLAGS}
            flags.add(five)
    assert flags == ROUTED_FLAGS and split_flags == SPLIT_FLAGS
    assert [
        loader.MODULES[name]["build"].get(loader.PREFILL_SPLIT_FLAG, False)
        for name in single
    ] == [False] * len(ROUTED_FLAGS) + [True] * len(SPLIT_FLAGS)
    for use_initial, use_indices, store, unit_g, unit_beta in split_flags:
        name = loader.selected_prefill_module(
            arch,
            use_initial_state=use_initial,
            use_state_indices=use_indices,
            store_final_state=store,
            unit_gates=unit_g,
            unit_beta=unit_beta,
            split_items=True,
        )
        assert loader.MODULES[name]["build"][loader.PREFILL_SPLIT_FLAG] is True
    for use_initial, use_indices, store, unit_g, unit_beta in flags:
        name = loader.selected_prefill_module(
            arch,
            use_initial_state=use_initial,
            use_state_indices=use_indices,
            store_final_state=store,
            unit_gates=unit_g,
            unit_beta=unit_beta,
        )
        assert name in single
    with pytest.raises(ValueError, match="programs of the 'prefill' route"):
        loader.selected_prefill_module(
            arch,
            use_initial_state=False,
            use_state_indices=True,
            store_final_state=True,
        )
    with pytest.raises(
        ValueError, match="programs of the 'prefill' route"
    ):  # g=None with beta=None is no shipped program
        loader.selected_prefill_module(
            arch,
            use_initial_state=False,
            use_state_indices=False,
            store_final_state=True,
            unit_gates=True,
            unit_beta=True,
        )
    with pytest.raises(
        ValueError, match="programs of the 'prefill' route"
    ):  # a unit program with an initial state
        loader.selected_prefill_module(
            arch,
            use_initial_state=True,
            use_state_indices=False,
            store_final_state=True,
            unit_gates=True,
        )
    with pytest.raises(
        ValueError, match="programs of the 'prefill' route"
    ):  # no split program of the unit-gate flags
        loader.selected_prefill_module(
            arch,
            use_initial_state=False,
            use_state_indices=False,
            store_final_state=True,
            unit_gates=True,
            split_items=True,
        )
    with pytest.raises(
        ValueError, match="programs of the 'prefill' route"
    ):  # a further flag no record spells on
        loader.selected_prefill_module(
            arch,
            use_initial_state=False,
            use_state_indices=False,
            store_final_state=True,
            split_rows=True,
        )
    base = loader.selected_prefill_module(
        arch, use_initial_state=False, use_state_indices=False, store_final_state=True
    )
    assert loader.launch_grid(loader.MODULES[base], 1, 1, tiles=1000, sm_count=132) == (
        132,
        1,
        1,
    )
    assert loader.launch_grid(loader.MODULES[base], 1, 1, tiles=7, sm_count=132) == (
        7,
        1,
        1,
    )
    with pytest.raises(ValueError, match="tiles and sm_count"):
        loader.launch_grid(loader.MODULES[base], 1, 1)
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
    assert set(loader.MODULES[base]["logical"].values()) == set(
        loader.PREFILL_LOGICAL_TENSORS
    ) | set(loader.PREFILL_LOGICAL_SCALARS)


def test_prefill_records_reference_the_build_manifest() -> None:
    """Every prefill record names its ``build.json`` library; the entries compile with fast math (their validated
    setting)."""
    libraries = cake_build.libraries_by_name()
    package = Path(cake_build.__file__).parent
    seen = 0
    for name, record in loader.MODULES.items():
        if record["role"] != "prefill":
            continue
        seen += 1
        assert record["library"] == record["kernel"] == name
        assert record["body"] == "prefill_sm90"
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
    assert seen == len(ROUTED_FLAGS) + len(SPLIT_FLAGS)


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
