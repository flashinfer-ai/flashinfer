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

Tests of the experimental fused MLA value projection + sigmoid output gate
(``flashinfer.cake_mla_value_gate.prepare_mla_value_gate``, Cake backend on
exact SM100a / SM103a).

The reference is a float64 model with the kernel's rounding boundaries: the
projection summed exactly and rounded to BF16, the exact sigmoid, the exact
product.  Every valid element must lie within a bound built from the FP32
accumulation error, the FP32 sigmoid error and the final BF16 rounding, and at
most ``MAX_MISMATCH_FRACTION`` of the valid elements may differ from the
model's BF16-rounded value; the second check catches a wrong rounding mode or
a BF16-rounded sigmoid that the per-element bound alone would admit.  Invalid
rows (request length <= 0) must be +0.0 bit for bit, rows past ``B`` must stay
untouched.

Each architecture has a 16-row and an 8-row program (token rows per CTA).  They
must give identical bits; prepare decides once which one serves each ``B``
(the 8-row program with programmatic dependent launch or when its grid fits one
wave of the device).  The layout, padding, stream, graph, weight-binding and
PDL tests run on each program, forced through the backend's ``tile_rows``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import tvm_ffi

from flashinfer.cake_mla_value_gate import prepare_mla_value_gate
from flashinfer.experimental.cake_mla_value_gate import cake_backend
from flashinfer.experimental.cake_mla_value_gate.cake_jit import (
    ARCH_BY_CAPABILITY,
    ARCHES,
    MODULES,
    records_for,
    tile_rows,
)

LATENT = 512
VALUE = 128
CAPACITY = 64
# The owner's decode widths (timed) and the padding boundaries (correctness only).
DECODE_ROWS = (1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 48, 64)
BOUNDARY_ROWS = (9, 17, 33)
# Kimi-K3 TP8: 12 heads per rank; the gate is a column slice of the fused [B, 3648] projection output.
PROD_HEADS = 12
PROD_PITCH = 3648
PROD_GATE_OFFSET = 2112

# Accuracy contract (tolerance form).  The FP32 sigmoid's relative error against the exact sigmoid: a fixed
# part plus a part that grows with |g| (the exponent argument -g * log2(e) is rounded to FP32 before exp2).
SIGMOID_REL_ERROR = 2.0**-20
SIGMOID_REL_ERROR_PER_UNIT = 2.0**-22
# The largest fraction of valid elements whose value may differ from the model's BF16-rounded value.
MAX_MISMATCH_FRACTION = 0.01
# The out sentinel: rows past B must keep it.
SENTINEL = -777.0
# Lengths of padded requests: a request is padded iff its length is < 1.
PADDED_LENGTHS = (0, -1, -(2**31), -4096)
# The programs of each architecture (token rows per CTA) and the dynamic shared memory each one
# launches with (rows x 544 x 2 bytes).
TILES = (16, 8)
TILE_SMEM_BYTES = {16: 17408, 8: 8704}
# The B on both sides of the one-wave threshold at H = 12 on B200 / B300 (148 SMs) and GB300 (152).
THRESHOLD_ROWS = (24, 25)
# The package directory (the generated bindings live in csrc/).
PACKAGE_DIR = Path(cake_backend.__file__).parent

tiles = pytest.mark.parametrize("tile", TILES, ids=[f"rows{t}" for t in TILES])


def expected_tile(heads: int, rows: int, enable_pdl: bool, sm_count: int) -> int:
    """The selection rule: the 8-row program with PDL or when its ``4 * H * ceil(B / 8)`` CTAs fit
    one wave of the device, else the 16-row program."""
    return 8 if enable_pdl or 4 * heads * -(-rows // 8) <= sm_count else 16


def _sm_count() -> int:
    return torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count


def _arch() -> str | None:
    if not torch.cuda.is_available():
        return None
    return ARCH_BY_CAPABILITY.get(tuple(torch.cuda.get_device_capability()))


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)
requires_kernel = pytest.mark.skipif(
    _arch() is None,
    reason="requires an exact SM100a / SM103a device (compute capability 10.0 / 10.3)",
)


# ----------------------------------------------------------------------------------------------
# Inputs


def _generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def make_w_uv(heads: int, seed: int, scale: float = 0.05) -> torch.Tensor:
    """Logical ``[H, 512, 128]`` view (strides ``(65536, 1, 512)``) of a contiguous ``[H, 128, 512]`` weight."""
    physical = torch.randn((heads, VALUE, LATENT), generator=_generator(seed)) * scale
    return physical.to(torch.bfloat16).cuda().transpose(1, 2)


@dataclass
class Case:
    """One launch's tensors on persistent buffers of ``CAPACITY`` rows: rows past ``rows`` hold NaN
    (inputs) or ``SENTINEL`` (output), the gate's pitch padding holds NaN."""

    rows: int
    heads: int
    q: int
    latent_buf: torch.Tensor
    gate_buf: torch.Tensor
    pitch: int
    offset: int
    lengths: torch.Tensor
    out_buf: torch.Tensor

    @property
    def latent(self) -> torch.Tensor:
        return self.latent_buf[: self.rows]

    @property
    def gate(self) -> torch.Tensor:
        return self.gate_view(self.rows)

    def gate_view(self, rows: int) -> torch.Tensor:
        return self.gate_buf.as_strided(
            (rows, self.heads * VALUE), (self.pitch, 1), self.offset
        )

    @property
    def out(self) -> torch.Tensor:
        return self.out_buf[: self.rows]

    def args(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.latent, self.gate, self.lengths, self.out

    def load(self, other: Case) -> None:
        """Copy ``other``'s input contents into this case's buffers (same geometry)."""
        self.latent_buf.copy_(other.latent_buf)
        self.gate_buf.copy_(other.gate_buf)
        self.lengths.copy_(other.lengths)


def make_case(
    rows: int,
    heads: int,
    *,
    q: int = 1,
    pitch: int | None = None,
    offset: int = 0,
    seed: int = 0,
    invalid_fraction: float = 0.25,
    gate_scale: float = 4.0,
) -> Case:
    width = heads * VALUE
    pitch = width if pitch is None else pitch
    g = _generator(1000 + seed)
    nan = float("nan")
    latent_buf = torch.full((CAPACITY, heads, LATENT), nan, dtype=torch.bfloat16)
    latent_buf[:rows] = torch.randn((rows, heads, LATENT), generator=g).to(
        torch.bfloat16
    )
    gate_buf = torch.full((offset + CAPACITY * pitch,), nan, dtype=torch.bfloat16)
    gate_buf.as_strided((rows, width), (pitch, 1), offset).copy_(
        (torch.randn((rows, width), generator=g) * gate_scale).to(torch.bfloat16)
    )
    requests = rows // q
    lengths = torch.randint(1, 4096, (requests,), generator=g, dtype=torch.int32)
    invalid = torch.rand((requests,), generator=g) < invalid_fraction
    if requests > 1 and invalid_fraction > 0:
        invalid[torch.randint(0, requests, (1,), generator=g)] = True
    if invalid.all():
        invalid[0] = False
    # A request is padded iff its length is < 1: mix zero and negative lengths.
    padded = torch.nonzero(invalid).flatten()
    cycle = torch.tensor(PADDED_LENGTHS, dtype=torch.int32)
    lengths[padded] = cycle[torch.arange(padded.numel()) % cycle.numel()]
    # Padded requests carry garbage: NaN latent and NaN gates must still give +0.0.
    for r in torch.nonzero(invalid).flatten().tolist():
        latent_buf[r * q : (r + 1) * q] = nan
        gate_buf.as_strided((rows, width), (pitch, 1), offset)[r * q : (r + 1) * q] = (
            nan
        )
    out_buf = torch.full((CAPACITY, width), SENTINEL, dtype=torch.bfloat16)
    return Case(
        rows,
        heads,
        q,
        latent_buf.cuda(),
        gate_buf.cuda(),
        pitch,
        offset,
        lengths.cuda(),
        out_buf.cuda(),
    )


# ----------------------------------------------------------------------------------------------
# Reference model and checks


@dataclass
class Model:
    valid: torch.Tensor  # [B] bool
    s: torch.Tensor  # [B, H*128] exact projection (float64)
    s_abs: torch.Tensor  # [B, H*128] sum of |products|
    p: torch.Tensor  # [B, H*128] projection rounded to FP32, then BF16 (float64)
    g: torch.Tensor  # [B, H*128] gate (float64)
    sig: torch.Tensor  # [B, H*128] exact sigmoid
    exact: torch.Tensor  # [B, H*128] p * sig


def model(
    latent: torch.Tensor,
    w_uv: torch.Tensor,
    gate: torch.Tensor,
    lengths: torch.Tensor,
    q: int,
) -> Model:
    rows, heads, _ = latent.shape
    z, w = latent.double(), w_uv.double()
    s = torch.einsum("bhk,hkv->bhv", z, w).reshape(rows, heads * VALUE)
    s_abs = torch.einsum("bhk,hkv->bhv", z.abs(), w.abs()).reshape(rows, heads * VALUE)
    p = s.float().to(torch.bfloat16).double()
    g = gate.double()
    sig = torch.sigmoid(g)
    valid = lengths.long().repeat_interleave(q) > 0
    return Model(valid, s, s_abs, p, g, sig, p * sig)


def _rounded(x: torch.Tensor) -> torch.Tensor:
    return x.float().to(torch.bfloat16).double()


def check_against_model(out: torch.Tensor, m: Model, label: str = "") -> None:
    """Invalid rows +0.0 bit for bit; valid elements within the error bound and at most
    ``MAX_MISMATCH_FRACTION`` of the normal-range ones off the model's BF16-rounded value; NaN exactly
    where the model has NaN."""
    bits = out.view(torch.int16)
    invalid = ~m.valid
    if invalid.any():
        assert torch.all(bits[invalid] == 0), f"{label}: an invalid row is not +0.0"
    if not m.valid.any():
        return
    got = out[m.valid].double()
    exact, p, s, s_abs, g, sig = (
        x[m.valid] for x in (m.exact, m.p, m.s, m.s_abs, m.g, m.sig)
    )
    nan = torch.isnan(exact)
    assert torch.equal(torch.isnan(got), nan), f"{label}: NaN pattern differs"
    accumulation = LATENT * 2.0**-23 * s_abs
    dp = accumulation + 2.0**-7 * (s.abs() + accumulation)
    sig_rel = SIGMOID_REL_ERROR + SIGMOID_REL_ERROR_PER_UNIT * g.abs().clamp(max=128.0)
    bound = sig * dp + (p.abs() + dp) * sig * sig_rel
    tol = bound + 2.0**-8 * (exact.abs() + bound) + (p.abs() + 1.0) * 2.0**-126
    err = (got - exact).abs()
    bad = ~nan & ~(err <= tol)
    if bad.any():
        idx = torch.nonzero(bad)[0].tolist()
        raise AssertionError(
            f"{label}: {int(bad.sum())} elements outside the bound; first at {idx}: "
            f"got {got[tuple(idx)].item()!r}, model {exact[tuple(idx)].item()!r}, "
            f"tol {tol[tuple(idx)].item()!r}, p {p[tuple(idx)].item()!r}, g {g[tuple(idx)].item()!r}"
        )
    # Subnormal sigmoids and products depend on the kernel's flush-to-zero choices; the absolute
    # floor of the bound covers them, the rounding statistics count the normal range only.
    counted = ~nan & (sig >= 2.0**-126) & (exact.abs() >= 2.0**-126)
    mismatch = (got != _rounded(exact)) & counted
    fraction = float(mismatch.sum()) / max(1, int(counted.sum()))
    assert fraction <= MAX_MISMATCH_FRACTION, (
        f"{label}: {fraction:.4%} of the valid elements differ from the model's BF16 value"
    )


def check_case(
    case: Case, w_uv: torch.Tensor, label: str, inputs: Case | None = None
) -> None:
    """``case.out`` against the model of ``inputs`` (default: ``case``'s own inputs); rows past ``B`` of
    ``case.out_buf`` must keep the sentinel."""
    src = case if inputs is None else inputs
    check_against_model(
        case.out, model(src.latent, w_uv, src.gate, src.lengths, src.q), label
    )
    tail = case.out_buf[case.rows :]
    assert torch.all(tail == SENTINEL), f"{label}: rows past B were written"


@pytest.fixture(scope="module")
def prod_w_uv() -> torch.Tensor:
    return make_w_uv(PROD_HEADS, seed=7)


def _prepare(w_uv: torch.Tensor, *, tile: int | None = None, **kwargs):
    """The public API (the selection rule), or the backend forced to the ``tile``-row program."""
    kwargs.setdefault("max_rows", CAPACITY)
    kwargs.setdefault("sigmoid_precision", "fp32")
    if tile is None:
        return prepare_mla_value_gate(w_uv, **kwargs)
    runner = cake_backend.prepare_mla_value_gate(w_uv, tile_rows=tile, **kwargs)
    rows = range(1, runner.max_rows + 1)
    assert {runner.tile_rows_for(b) for b in rows} == {tile}
    return runner


def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.view(torch.int16)


# ----------------------------------------------------------------------------------------------
# Registry and selection rule (no GPU needed)


def _records_by_tile(arch: str) -> dict[int, dict]:
    return {
        tile_rows(MODULES[name]): MODULES[name] for name in records_for(arch, "fp32")
    }


@pytest.mark.parametrize("arch", ARCHES)
def test_registry_has_a_16_row_and_an_8_row_program(arch):
    names = records_for(arch, "fp32")
    by_tile = _records_by_tile(arch)
    assert len(names) == len(TILES) and sorted(by_tile) == sorted(TILES), names
    for key in ("arg_plan", "logical", "ffi_entry", "sigmoid"):
        assert by_tile[8][key] == by_tile[16][key], key
    for record in by_tile.values():
        assert record["grid_rule"]["kind"] == "value_tiles"
        assert record["grid_rule"]["cols"] == 32
        assert record["launch"]["block"] == (128, 1, 1)
        assert record["launch"]["use_pdl"] == "runtime"


@pytest.mark.parametrize("name", sorted(MODULES))
def test_each_program_launches_with_its_own_shared_memory(name):
    """The generated binding of every program launches with the dynamic shared memory its record
    declares, and that is its row tile's size (16 rows: 17408 B, 8 rows: 8704 B)."""
    record = MODULES[name]
    smem = record["launch"]["dynamic_smem_bytes"]
    assert smem == TILE_SMEM_BYTES[tile_rows(record)], (name, smem)
    binding = PACKAGE_DIR / "csrc" / f"{record['library']}_binding.cu"
    launched = re.findall(r"dynamicSmemBytes\s*=\s*(\d+)\s*;", binding.read_text())
    assert launched == [str(smem)], (name, launched)


def _rule_records() -> dict[int, dict]:
    return {
        tile: {
            "library": f"rows{tile}",
            "grid_rule": {"kind": "value_tiles", "cols": 32, "rows": tile},
        }
        for tile in TILES
    }


@pytest.mark.parametrize("sm_count", [148, 152])
def test_selection_rule(sm_count):
    records = _rule_records()

    def pick(heads, rows, enable_pdl=False, recs=records):
        return cake_backend.select_tile_rows(
            recs, rows=rows, heads=heads, enable_pdl=enable_pdl, sm_count=sm_count
        )

    for heads in (1, 6, 12, 16):
        for rows in range(1, CAPACITY + 1):
            for enable_pdl in (True, False):
                assert pick(heads, rows, enable_pdl) == expected_tile(
                    heads, rows, enable_pdl, sm_count
                ), (heads, rows, enable_pdl)
    # The thresholds on B200 / B300 / GB300: H = 12 -> B <= 24, H = 6 -> B <= 48, H = 16 -> B <= 16,
    # H = 1 -> every B; with PDL the 8-row program serves every B.
    assert [pick(12, rows) for rows in THRESHOLD_ROWS] == [8, 16]
    assert [pick(6, rows) for rows in (48, 49)] == [8, 16]
    assert [pick(16, rows) for rows in (16, 17)] == [8, 16]
    assert {pick(1, rows) for rows in range(1, CAPACITY + 1)} == {8}
    assert {pick(16, rows, True) for rows in range(1, CAPACITY + 1)} == {8}
    # One program serves every B; three are a registry error.
    single = {16: records[16]}
    assert {
        pick(12, rows, pdl, single) for rows in (1, 64) for pdl in (True, False)
    } == {16}
    three = {
        **records,
        4: {**records[8], "grid_rule": {**records[8]["grid_rule"], "rows": 4}},
    }
    with pytest.raises(RuntimeError):
        pick(12, 8, False, three)


# ----------------------------------------------------------------------------------------------
# Accuracy over the shape list


@requires_kernel
@pytest.mark.parametrize("rows", DECODE_ROWS + BOUNDARY_ROWS + THRESHOLD_ROWS)
@pytest.mark.parametrize(
    "pitch, offset",
    [(PROD_PITCH, PROD_GATE_OFFSET), (PROD_HEADS * VALUE, 0)],
    ids=["pitch3648_off2112", "pitch1536"],
)
@pytest.mark.parametrize("enable_pdl", [True, False])
def test_accuracy_production_heads(prod_w_uv, rows, pitch, offset, enable_pdl):
    runner = _prepare(prod_w_uv, enable_pdl=enable_pdl)
    case = make_case(rows, PROD_HEADS, pitch=pitch, offset=offset, seed=rows)
    args = case.args()
    assert runner.launch(*args) is args[3]
    torch.cuda.synchronize()
    check_case(case, prod_w_uv, f"B={rows} pitch={pitch}")


@requires_kernel
@pytest.mark.parametrize("heads", [1, 6, 16])
@pytest.mark.parametrize("rows", [1, 5, 9, 16, 33, 64])
@pytest.mark.parametrize("padded", [False, True], ids=["dense", "padded"])
def test_accuracy_other_heads(heads, rows, padded):
    w_uv = make_w_uv(heads, seed=heads)
    width = heads * VALUE
    pitch, offset = (width + 96, 2) if padded else (width, 0)
    case = make_case(rows, heads, pitch=pitch, offset=offset, seed=heads * 100 + rows)
    _prepare(w_uv).launch(*case.args())
    torch.cuda.synchronize()
    check_case(case, w_uv, f"H={heads} B={rows} pitch={pitch}")


@requires_kernel
@tiles
@pytest.mark.parametrize("q, rows", [(2, 2), (2, 8), (2, 64), (4, 4), (4, 16), (4, 48)])
def test_rows_per_request(prod_w_uv, tile, q, rows):
    case = make_case(rows, PROD_HEADS, q=q, seed=31 * q + rows, invalid_fraction=0.4)
    _prepare(prod_w_uv, tile=tile, rows_per_request=q).launch(*case.args())
    torch.cuda.synchronize()
    check_case(case, prod_w_uv, f"q={q} B={rows}")


@requires_kernel
@tiles
def test_sigmoid_is_fp32_not_bf16(prod_w_uv, tile):
    """The product uses the FP32 sigmoid: a BF16-rounded sigmoid gives other bits on a large fraction."""
    case = make_case(64, PROD_HEADS, seed=99, invalid_fraction=0.0, gate_scale=2.0)
    _prepare(prod_w_uv, tile=tile).launch(*case.args())
    torch.cuda.synchronize()
    m = model(case.latent, prod_w_uv, case.gate, case.lengths, 1)
    check_against_model(case.out, m, "fp32 sigmoid")
    bf16_sigmoid = _rounded(m.p * _rounded(m.sig))
    differs = (case.out.double() != bf16_sigmoid).float().mean().item()
    assert differs > 5 * MAX_MISMATCH_FRACTION, (
        f"only {differs:.2%} of the elements differ from the BF16-sigmoid path"
    )


@requires_kernel
@tiles
def test_gate_extremes(prod_w_uv, tile):
    """Saturating, underflowing, infinite and NaN gates on valid rows."""
    special = [
        0.0, 1.0, -1.0, 8.0, -8.0, 16.0, -16.0, 40.0, -40.0, 87.0, -87.0, 88.0, -88.0,
        89.0, -89.0, 100.0, -100.0, 1.0e4, -1.0e4, 3.0e38, -3.0e38,
        float("inf"), float("-inf"),
    ]  # fmt: skip
    rows, width = 8, PROD_HEADS * VALUE
    case = make_case(rows, PROD_HEADS, seed=5, invalid_fraction=0.0)
    values = torch.tensor(special, dtype=torch.float32)
    pattern = values[torch.arange(width) % len(special)].to(torch.bfloat16)
    case.gate.copy_(pattern.cuda().expand(rows, width))
    case.gate[rows - 1, 3] = float("nan")  # a NaN gate on a valid row gives NaN
    _prepare(prod_w_uv, tile=tile).launch(*case.args())
    torch.cuda.synchronize()
    check_case(case, prod_w_uv, "gate extremes")
    gate = case.gate.float()
    out = case.out.float()
    p = model(case.latent, prod_w_uv, case.gate, case.lengths, 1).p.float()
    assert torch.all(out[gate == float("-inf")] == 0.0)
    plus_inf = gate == float("inf")
    # sigmoid(+inf) == 1: the output is the BF16 projection (within its accumulation flip).
    assert torch.allclose(out[plus_inf], p[plus_inf], rtol=2.0**-7, atol=0.0)
    assert torch.isnan(out[rows - 1, 3])
    assert torch.isfinite(out[rows - 1, 4])


@requires_kernel
@tiles
def test_nan_latent_on_a_valid_row_gives_nan(prod_w_uv, tile):
    case = make_case(4, PROD_HEADS, seed=6, invalid_fraction=0.0)
    case.latent[2, 5, 100] = float("nan")
    _prepare(prod_w_uv, tile=tile).launch(*case.args())
    torch.cuda.synchronize()
    out = case.out.view(4, PROD_HEADS, VALUE)
    assert torch.isnan(out[2, 5]).all()
    assert torch.isfinite(out[2, :5]).all() and torch.isfinite(out[2, 6:]).all()
    assert torch.isfinite(out[[0, 1, 3]]).all()


@requires_kernel
@tiles
def test_invalid_rows_are_positive_zero_and_tail_untouched(prod_w_uv, tile):
    case = make_case(16, PROD_HEADS, seed=8, invalid_fraction=0.5)
    case.lengths[0] = 0
    case.latent_buf[0] = float("nan")
    case.gate[0] = float("inf")
    _prepare(prod_w_uv, tile=tile).launch(*case.args())
    torch.cuda.synchronize()
    check_case(case, prod_w_uv, "invalid rows")
    assert torch.all(case.out.view(torch.int16)[0] == 0)


@requires_kernel
@tiles
def test_negative_lengths_are_padded(prod_w_uv, tile):
    """A request is padded iff its length is < 1: negative lengths give +0.0 rows."""
    case = make_case(8, PROD_HEADS, seed=12, invalid_fraction=0.0)
    lengths = [-1, -(2**31), 7, -4096, 1, 0, -2, 2**31 - 1]
    case.lengths.copy_(torch.tensor(lengths, dtype=torch.int32))
    padded = [t for t, n in enumerate(lengths) if n < 1]
    case.latent_buf[padded] = float("nan")
    case.gate[padded] = float("nan")
    _prepare(prod_w_uv, tile=tile).launch(*case.args())
    torch.cuda.synchronize()
    check_case(case, prod_w_uv, "negative lengths")
    bits = case.out.view(torch.int16)
    for t in padded:
        assert torch.all(bits[t] == 0), f"row {t} (length {lengths[t]}) is not +0.0"
    for t in (2, 4, 7):
        assert torch.isfinite(case.out[t]).all(), f"row {t} (length {lengths[t]})"


# ----------------------------------------------------------------------------------------------
# The two programs: the decision of every B, identical bits, the threshold


def _names_by_tile(arch: str) -> dict[int, str]:
    return {tile_rows(MODULES[name]): name for name in records_for(arch, "fp32")}


@requires_kernel
@pytest.mark.parametrize("heads", [1, 6, 12, 16])
@pytest.mark.parametrize("enable_pdl", [True, False])
def test_prepare_decides_the_program_of_every_rows(heads, enable_pdl):
    runner = _prepare(make_w_uv(heads, seed=heads), enable_pdl=enable_pdl)
    sm_count = _sm_count()
    assert runner.sm_count == sm_count
    names = _names_by_tile(runner.arch)
    rows = range(1, CAPACITY + 1)
    decided = [runner.tile_rows_for(b) for b in rows]
    assert decided == [expected_tile(heads, b, enable_pdl, sm_count) for b in rows]
    assert [runner.program_for(b) for b in rows] == [names[t] for t in decided]
    # Only the programs some B uses are loaded; the PDL runner uses the 8-row program alone.
    assert set(runner.programs) == {names[t] for t in decided}
    if enable_pdl:
        assert runner.programs == (names[8],)
    assert runner.program == runner.program_for(CAPACITY)
    for bad in (0, CAPACITY + 1):
        with pytest.raises(ValueError):
            runner.program_for(bad)


@requires_kernel
@pytest.mark.parametrize("heads", [1, 6, 12, 16])
@pytest.mark.parametrize("enable_pdl", [True, False])
def test_both_programs_give_identical_bits(heads, enable_pdl):
    """The 16-row program, the 8-row program and the rule's choice give the same bits for every B,
    and each is within the model's bound (strided, offset gate; padded and NaN-filled requests)."""
    w_uv = make_w_uv(heads, seed=400 + heads)
    width = heads * VALUE
    if heads == PROD_HEADS:
        pitch, offset = PROD_PITCH, PROD_GATE_OFFSET
    else:
        pitch, offset = width + 96, 2
    runners = {tile: _prepare(w_uv, tile=tile, enable_pdl=enable_pdl) for tile in TILES}
    runners["rule"] = _prepare(w_uv, enable_pdl=enable_pdl)
    for rows in sorted(set(DECODE_ROWS + BOUNDARY_ROWS + THRESHOLD_ROWS)):
        case = make_case(rows, heads, pitch=pitch, offset=offset, seed=500 + rows)
        outs = {}
        for key, runner in runners.items():
            case.out_buf.fill_(SENTINEL)
            runner.launch(*case.args())
            torch.cuda.synchronize()
            check_case(case, w_uv, f"H={heads} B={rows} program={key}")
            outs[key] = case.out.clone()
        for key in (8, "rule"):
            assert torch.equal(_bits(outs[key]), _bits(outs[16])), (
                f"H={heads} B={rows}: program {key} differs from the 16-row program"
            )


@requires_kernel
@pytest.mark.parametrize("heads", [6, 12, 16])
def test_threshold_rows_switch_programs(heads):
    """Without PDL the 8-row program serves the largest B whose 8-row grid fits one wave and the
    16-row program the next B; both give the 16-row program's bits there.  With PDL the 8-row
    program serves both."""
    sm_count = _sm_count()
    last = max(b for b in range(1, CAPACITY + 1) if 4 * heads * -(-b // 8) <= sm_count)
    if heads == PROD_HEADS and 144 <= sm_count < 192:  # B200, B300, GB300
        assert (last, last + 1) == THRESHOLD_ROWS
    assert last < CAPACITY
    w_uv = make_w_uv(heads, seed=600 + heads)
    rule = _prepare(w_uv, enable_pdl=False)
    with_pdl = _prepare(w_uv, enable_pdl=True)
    forced = {tile: _prepare(w_uv, tile=tile, enable_pdl=False) for tile in TILES}
    for rows, tile in ((last, 8), (last + 1, 16)):
        assert rule.tile_rows_for(rows) == tile
        assert with_pdl.tile_rows_for(rows) == 8
        case = make_case(rows, heads, seed=700 + rows)
        outs = []
        for runner in (rule, with_pdl, forced[16], forced[8]):
            case.out_buf.fill_(SENTINEL)
            runner.launch(*case.args())
            torch.cuda.synchronize()
            check_case(case, w_uv, f"H={heads} B={rows}")
            outs.append(case.out.clone())
        for out in outs[1:]:
            assert torch.equal(_bits(out), _bits(outs[0])), f"H={heads} B={rows}"


@requires_kernel
def test_forced_program_rejections(prod_w_uv):
    with pytest.raises(NotImplementedError, match="4-row"):
        _prepare(prod_w_uv, tile=4)
    for bad in (True, "8", 8.0):
        with pytest.raises(TypeError):
            cake_backend.prepare_mla_value_gate(
                prod_w_uv, max_rows=8, sigmoid_precision="fp32", tile_rows=bad
            )


# ----------------------------------------------------------------------------------------------
# Layouts accepted and rejected


@requires_kernel
@tiles
@pytest.mark.parametrize(
    "pitch, offset",
    [(1540, 2), (PROD_PITCH + 2, 6), (PROD_PITCH, 10)],
    ids=["even_offset_4B", "pitch3650_off6", "pitch3648_off10"],
)
def test_gate_even_offsets_not_16_byte_aligned(prod_w_uv, tile, pitch, offset):
    case = make_case(33, PROD_HEADS, pitch=pitch, offset=offset, seed=pitch + offset)
    assert case.gate.data_ptr() % 16 != 0
    _prepare(prod_w_uv, tile=tile).launch(*case.args())
    torch.cuda.synchronize()
    check_case(case, prod_w_uv, f"pitch={pitch} offset={offset}")


@requires_kernel
@tiles
def test_single_row_gate_pitch_is_immaterial(prod_w_uv, tile):
    case = make_case(1, PROD_HEADS, pitch=PROD_PITCH, offset=PROD_GATE_OFFSET, seed=3)
    gate = case.gate_buf.as_strided(
        (1, PROD_HEADS * VALUE), (7, 1), PROD_GATE_OFFSET
    )  # a size-1 row dimension may carry any stride
    _prepare(prod_w_uv, tile=tile).launch(case.latent, gate, case.lengths, case.out)
    torch.cuda.synchronize()
    check_case(case, prod_w_uv, "B=1")


def _expect_rejection(
    label: str, error: type[Exception], call, *args, **kwargs
) -> None:
    try:
        call(*args, **kwargs)
    except error:
        return
    except Exception as exc:  # the wrong refusal
        raise AssertionError(f"{label}: raised {type(exc).__name__}: {exc}") from exc
    raise AssertionError(f"{label}: accepted")


def _offset_view(numel: int, offset: int, shape, dtype=torch.bfloat16) -> torch.Tensor:
    flat = torch.zeros(numel + offset + 64, dtype=dtype, device="cuda")
    return flat[offset : offset + numel].view(shape)


@requires_kernel
def test_launch_rejections(prod_w_uv):
    runner = _prepare(prod_w_uv, max_rows=16, rows_per_request=2)
    case = make_case(8, PROD_HEADS, q=2, seed=11)
    latent, gate, lengths, out = case.args()
    width = PROD_HEADS * VALUE
    bad = {
        "latent dtype": (latent.half(), gate, lengths, out, TypeError),
        "gate dtype": (latent, gate.float(), lengths, out, TypeError),
        "lengths dtype": (latent, gate, lengths.long(), out, TypeError),
        "out dtype": (latent, gate, lengths, out.float(), TypeError),
        "cpu latent": (latent.cpu(), gate, lengths, out, ValueError),
        "heads": (latent[:, :6], gate, lengths, out, ValueError),
        "latent width": (latent[..., :256], gate, lengths, out, ValueError),
        "latent not contiguous": (
            latent.transpose(0, 1).contiguous().transpose(0, 1), gate, lengths, out, ValueError,
        ),
        "latent misaligned": (
            _offset_view(8 * width * 4, 4, (8, PROD_HEADS, LATENT)), gate, lengths, out, ValueError,
        ),
        "out misaligned": (latent, gate, lengths, _offset_view(8 * width, 4, (8, width)), ValueError),
        "out not contiguous": (
            latent, gate, lengths, torch.empty(width, 8, dtype=torch.bfloat16, device="cuda").t(), ValueError,
        ),
        "out shape": (latent, gate, lengths, out[:, :-2], ValueError),
        "gate shape": (latent, gate[:, :-2], lengths, out, ValueError),
        "gate odd offset": (latent, case.gate_buf.as_strided((8, width), (width, 1), 1), lengths, out, ValueError),
        "gate odd pitch": (latent, case.gate_buf.as_strided((8, width), (width + 1, 1), 0), lengths, out, ValueError),
        "gate overlapping rows": (latent, case.gate_buf.as_strided((8, width), (width - 2, 1), 0), lengths, out, ValueError),
        "gate column stride": (latent, case.gate_buf.as_strided((8, width), (2 * width, 2), 0), lengths, out, ValueError),
        "lengths shape": (latent, gate, lengths[:3], out, ValueError),
        "rows not a multiple of q": (latent[:7], gate[:7], lengths, out[:7], ValueError),
        "rows above max_rows": (
            torch.zeros(18, PROD_HEADS, LATENT, dtype=torch.bfloat16, device="cuda"),
            torch.zeros(18, width, dtype=torch.bfloat16, device="cuda"),
            torch.ones(9, dtype=torch.int32, device="cuda"),
            torch.zeros(18, width, dtype=torch.bfloat16, device="cuda"),
            ValueError,
        ),
        "out overlaps latent": (latent, gate, lengths, latent.view(-1)[: 8 * width].view(8, width), ValueError),
        "out overlaps gate": (latent, gate, lengths, case.gate_buf[: 8 * width].view(8, width), ValueError),
        "out overlaps w_uv": (
            latent, gate, lengths, prod_w_uv.transpose(1, 2).reshape(-1)[: 8 * width].view(8, width), ValueError,
        ),
    }  # fmt: skip
    bad["no rows"] = (latent[:0], gate[:0], lengths[:0], out[:0], ValueError)
    for label, (lat, gat, lens, dst, error) in bad.items():
        _expect_rejection(label, error, runner.launch, lat, gat, lens, dst)


@requires_cuda
def test_prepare_rejections():
    heads = PROD_HEADS
    w_uv = make_w_uv(heads, seed=1)
    with pytest.raises(ValueError, match="sigmoid_precision"):
        prepare_mla_value_gate(w_uv, max_rows=8, sigmoid_precision="bf16")
    with pytest.raises(TypeError):
        prepare_mla_value_gate(w_uv, max_rows=8)  # sigmoid_precision is required
    for max_rows in (0, 65):
        with pytest.raises(ValueError, match="max_rows"):
            prepare_mla_value_gate(w_uv, max_rows=max_rows, sigmoid_precision="fp32")
    for q in (0, 9):
        with pytest.raises(ValueError, match="rows_per_request"):
            prepare_mla_value_gate(
                w_uv, max_rows=8, sigmoid_precision="fp32", rows_per_request=q
            )
    with pytest.raises(TypeError):
        prepare_mla_value_gate(w_uv, max_rows=8, sigmoid_precision="fp32", enable_pdl=1)
    with pytest.raises(ValueError, match="backend"):
        prepare_mla_value_gate(
            w_uv, max_rows=8, sigmoid_precision="fp32", backend="cudnn"
        )
    packed = w_uv.contiguous()  # [H, 512, 128] row-major: the value axis contiguous
    misaligned = _offset_view(
        heads * VALUE * LATENT, 4, (heads, VALUE, LATENT)
    ).transpose(1, 2)
    wrong = {
        "packed layout": (packed, ValueError),
        "dtype": (w_uv.half(), TypeError),
        "cpu": (w_uv.cpu(), ValueError),
        "latent width": (w_uv[:, :256], ValueError),
        "value width": (w_uv[:, :, :64], ValueError),
        "misaligned": (misaligned, ValueError),
    }
    for label, (weight, error) in wrong.items():
        _expect_rejection(
            label,
            error,
            prepare_mla_value_gate,
            weight,
            max_rows=8,
            sigmoid_precision="fp32",
        )


@requires_cuda
@pytest.mark.skipif(
    _arch() is not None, reason="checks the refusal on other architectures"
)
def test_prepare_refuses_other_architectures():
    with pytest.raises(NotImplementedError, match="SM100a or SM103a"):
        prepare_mla_value_gate(
            make_w_uv(PROD_HEADS, seed=1), max_rows=8, sigmoid_precision="fp32"
        )


# ----------------------------------------------------------------------------------------------
# Determinism and batch invariance


@requires_kernel
@tiles
def test_deterministic_and_batch_invariant(prod_w_uv, tile):
    runner = _prepare(prod_w_uv, tile=tile)
    full = make_case(CAPACITY, PROD_HEADS, seed=21)
    runner.launch(*full.args())
    reference = full.out.clone()
    for _ in range(3):
        full.out.fill_(SENTINEL)
        runner.launch(*full.args())
        assert torch.equal(full.out.view(torch.int16), reference.view(torch.int16))
    for rows in DECODE_ROWS + BOUNDARY_ROWS + THRESHOLD_ROWS:
        out = torch.full_like(full.out_buf, SENTINEL)
        runner.launch(
            full.latent_buf[:rows],
            full.gate_view(rows),
            full.lengths[:rows],
            out[:rows],
        )
        torch.cuda.synchronize()
        assert torch.equal(
            out[:rows].view(torch.int16), reference[:rows].view(torch.int16)
        ), f"B={rows} differs from the first B rows of B={CAPACITY}"
        assert torch.all(out[rows:] == SENTINEL)


# ----------------------------------------------------------------------------------------------
# Streams, CUDA graphs and the weight binding


@requires_kernel
@tiles
def test_launches_on_the_current_stream(prod_w_uv, tile):
    runner = _prepare(prod_w_uv, tile=tile)
    case = make_case(16, PROD_HEADS, seed=31)
    staged = make_case(16, PROD_HEADS, seed=32)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        # The inputs are written on the side stream after a long delay: a launch on any other
        # stream would read the old contents.
        torch.cuda._sleep(100_000_000)
        case.load(staged)
        runner.launch(*case.args())
    side.synchronize()
    check_case(case, prod_w_uv, "side stream", inputs=staged)


@requires_kernel
def test_prepare_inside_capture_raises(prod_w_uv):
    graph = torch.cuda.CUDAGraph()
    scratch = torch.zeros(1, device="cuda")
    with torch.cuda.graph(graph):
        scratch.add_(1)
        with pytest.raises(RuntimeError, match="outside CUDA graph capture"):
            prepare_mla_value_gate(prod_w_uv, max_rows=8, sigmoid_precision="fp32")


@requires_kernel
@pytest.mark.parametrize("tile", [None, *TILES], ids=["rule", "rows16", "rows8"])
@pytest.mark.parametrize("rows", [16, 25])
@pytest.mark.parametrize("enable_pdl", [True, False])
def test_graph_capture_replays_new_contents(prod_w_uv, tile, rows, enable_pdl):
    runner = _prepare(prod_w_uv, tile=tile, enable_pdl=enable_pdl)
    case = make_case(
        rows, PROD_HEADS, pitch=PROD_PITCH, offset=PROD_GATE_OFFSET, seed=40
    )
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        runner.launch(*case.args())  # warm-up outside capture
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        runner.launch(*case.args())
    for seed in (41, 42, 43):
        staged = make_case(
            rows, PROD_HEADS, pitch=PROD_PITCH, offset=PROD_GATE_OFFSET, seed=seed,
            invalid_fraction=0.1 * (seed - 40),
        )  # fmt: skip
        case.load(staged)
        case.out_buf.fill_(SENTINEL)
        graph.replay()
        torch.cuda.synchronize()
        check_case(case, prod_w_uv, f"replay {seed}", inputs=staged)
        eager = torch.full_like(staged.out_buf, SENTINEL)
        runner.launch(staged.latent, staged.gate, staged.lengths, eager[:rows])
        torch.cuda.synchronize()
        assert torch.equal(_bits(case.out), _bits(eager[:rows]))


@requires_kernel
@tiles
def test_weight_binding_and_re_prepare(tile):
    old = make_w_uv(PROD_HEADS, seed=50)
    new = make_w_uv(PROD_HEADS, seed=51)
    case = make_case(8, PROD_HEADS, seed=52)
    old_runner = _prepare(old, tile=tile)
    old_runner.launch(*case.args())
    torch.cuda.synchronize()
    check_case(case, old, "old weight")
    with_old = case.out.clone()
    # The model replaces its weight storage: the old runner still reads the storage it bound
    # (and keeps it alive), so it must be prepared again for the new weight.
    new_runner = _prepare(new, tile=tile)
    assert old_runner.w_uv is old and old_runner.w_uv.data_ptr() != new.data_ptr()
    assert new_runner.w_uv is new
    old_runner.launch(*case.args())
    torch.cuda.synchronize()
    assert torch.equal(case.out.view(torch.int16), with_old.view(torch.int16))
    new_runner.launch(*case.args())
    torch.cuda.synchronize()
    check_case(case, new, "new weight")
    with_new = case.out.clone()
    # An in-place update of the bound storage is seen by the next launch.
    old.copy_(new)
    old_runner.launch(*case.args())
    torch.cuda.synchronize()
    assert torch.equal(case.out.view(torch.int16), with_new.view(torch.int16))


# ----------------------------------------------------------------------------------------------
# Programmatic dependent launch: a predecessor that lets its dependents start before it writes,
# and a successor that starts early and waits before it reads.

_SPIN_CYCLES = 1_000_000  # about half a millisecond

_PROBE_SOURCE = r"""
#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include <cstdint>

using tvm::ffi::TensorView;

struct Copies {
  uint8_t* dst[4];
  const uint8_t* src[4];
  int64_t bytes[4];
};

__global__ void TriggerSpinCopyKernel(Copies copies, int64_t spin_cycles) {
  // Let the dependent launch start now, before anything is written.
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  const int64_t start = clock64();
  while (clock64() - start < spin_cycles) {
  }
  const int64_t stride = int64_t(gridDim.x) * blockDim.x;
  for (int c = 0; c < 4; ++c) {
    for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < copies.bytes[c]; i += stride) {
      copies.dst[c][i] = copies.src[c][i];
    }
  }
}

__global__ void WaitCopyKernel(uint8_t* dst, const uint8_t* src, int64_t bytes) {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const int64_t stride = int64_t(gridDim.x) * blockDim.x;
  for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < bytes; i += stride) {
    dst[i] = src[i];
  }
}

static int64_t Bytes(const TensorView& t) { return t.numel() * ((t.dtype().bits + 7) / 8); }

static cudaStream_t StreamOf(const TensorView& t) {
  return static_cast<cudaStream_t>(TVMFFIEnvGetStream(t.device().device_type, t.device().device_id));
}

// Copies src_i over dst_i (whole contiguous tensors) after triggering its dependents and spinning.
void DelayedCopies(TensorView d0, TensorView s0, TensorView d1, TensorView s1, TensorView d2, TensorView s2,
                   TensorView d3, TensorView s3, int64_t spin_cycles) {
  Copies copies{};
  const TensorView* pairs[4][2] = {{&d0, &s0}, {&d1, &s1}, {&d2, &s2}, {&d3, &s3}};
  for (int c = 0; c < 4; ++c) {
    const TensorView& d = *pairs[c][0];
    const TensorView& s = *pairs[c][1];
    TVM_FFI_ICHECK(d.IsContiguous() && s.IsContiguous() && Bytes(d) == Bytes(s)) << "pair " << c;
    copies.dst[c] = static_cast<uint8_t*>(d.data_ptr()) + d.byte_offset();
    copies.src[c] = static_cast<const uint8_t*>(s.data_ptr()) + s.byte_offset();
    copies.bytes[c] = Bytes(d);
  }
  int sms = 0;
  TVM_FFI_ICHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, d0.device().device_id) ==
                 cudaSuccess);
  // One resident CTA per SM: every CTA triggers before a dependent can take a slot.
  TriggerSpinCopyKernel<<<sms, 256, 0, StreamOf(d0)>>>(copies, spin_cycles);
  const cudaError_t status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
}

// Copies src over dst, launched with programmatic dependent launch; waits on its predecessor first.
void WaitThenCopy(TensorView dst, TensorView src) {
  TVM_FFI_ICHECK(dst.IsContiguous() && src.IsContiguous() && Bytes(dst) == Bytes(src));
  uint8_t* d = static_cast<uint8_t*>(dst.data_ptr()) + dst.byte_offset();
  const uint8_t* s = static_cast<const uint8_t*>(src.data_ptr()) + src.byte_offset();
  int64_t bytes = Bytes(dst);
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(32, 1, 1);
  config.blockDim = dim3(256, 1, 1);
  config.dynamicSmemBytes = 0;
  config.stream = StreamOf(dst);
  cudaLaunchAttribute attrs[1] = {};
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attrs;
  config.numAttrs = 1;
  const cudaError_t status = cudaLaunchKernelEx(&config, WaitCopyKernel, d, s, bytes);
  TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(delayed_copies, DelayedCopies);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(wait_then_copy, WaitThenCopy);
"""


@pytest.fixture(scope="module")
def probe():
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.core import gen_jit_spec, sm100a_nvcc_flags, sm103a_nvcc_flags
    from flashinfer.jit.utils import write_if_different

    arch = _arch()
    if arch is None:
        pytest.skip("requires an exact SM100a / SM103a device")
    flags = {"sm_100a": sm100a_nvcc_flags, "sm_103a": sm103a_nvcc_flags}[arch]
    name = f"test_cake_mla_value_gate_pdl_probe_{arch}"
    source = jit_env.FLASHINFER_GEN_SRC_DIR / name / "probe.cu"
    write_if_different(source, _PROBE_SOURCE)
    return gen_jit_spec(name, [source], extra_cuda_cflags=flags).build_and_load()


class _PdlChain:
    """predecessor (triggers, spins, then writes the inputs and snapshots ``out``) -> the PDL runner ->
    successor (PDL launch, waits, then copies ``out``), on the current stream."""

    def __init__(self, probe, runner, case: Case, tile: int) -> None:
        self.probe, self.runner, self.case, self.tile = probe, runner, case, tile
        self.staged = make_case(
            case.rows, case.heads, pitch=case.pitch, offset=case.offset, seed=0
        )
        self.previous_out = torch.empty_like(case.out_buf)
        self.snapshot = torch.empty_like(case.out_buf)
        self.copied = torch.empty_like(case.out_buf)

    def stage(self, seed: int) -> Case:
        staged = make_case(
            self.case.rows,
            self.case.heads,
            pitch=self.case.pitch,
            offset=self.case.offset,
            seed=seed,
        )
        self.staged.load(staged)
        # The buffers the chain overwrites hold stale contents: the previous inputs and a known old output.
        self.previous_out.copy_(
            torch.randn(self.previous_out.shape, generator=_generator(seed)).to(
                torch.bfloat16
            )
        )
        self.case.out_buf.copy_(self.previous_out)
        return staged

    def run(self) -> None:
        c, s = self.case, self.staged
        with tvm_ffi.use_torch_stream():
            self.probe.delayed_copies(
                c.latent_buf, s.latent_buf, c.gate_buf, s.gate_buf, c.lengths, s.lengths,
                self.snapshot, c.out_buf, _SPIN_CYCLES,
            )  # fmt: skip
        self.runner.launch(*c.args())
        with tvm_ffi.use_torch_stream():
            self.probe.wait_then_copy(self.copied, c.out_buf)

    def check(self, staged: Case, w_uv: torch.Tensor, label: str) -> None:
        c = self.case
        # The runner waited for the predecessor's writes before reading its inputs ...
        check_against_model(
            c.out, model(staged.latent, w_uv, staged.gate, staged.lengths, c.q), label
        )
        plain = torch.full_like(c.out_buf, SENTINEL)
        _prepare(w_uv, tile=self.tile, enable_pdl=False).launch(
            staged.latent, staged.gate, staged.lengths, plain[: c.rows]
        )
        torch.cuda.synchronize()
        assert torch.equal(
            c.out.view(torch.int16), plain[: c.rows].view(torch.int16)
        ), label
        # ... did not write out before the predecessor read it ...
        assert torch.equal(
            self.snapshot.view(torch.int16), self.previous_out.view(torch.int16)
        ), label
        # ... and the successor saw the finished output.
        assert torch.equal(
            self.copied.view(torch.int16), c.out_buf.view(torch.int16)
        ), label


@requires_kernel
@tiles
def test_pdl_predecessor_and_successor_eager(probe, prod_w_uv, tile):
    case = make_case(64, PROD_HEADS, pitch=PROD_PITCH, offset=PROD_GATE_OFFSET, seed=60)
    runner = _prepare(prod_w_uv, tile=tile, enable_pdl=True)
    chain = _PdlChain(probe, runner, case, tile)
    for seed in (61, 62, 63):
        staged = chain.stage(seed)
        torch.cuda.synchronize()
        chain.run()
        torch.cuda.synchronize()
        chain.check(staged, prod_w_uv, f"eager {seed}")


@requires_kernel
@tiles
def test_pdl_predecessor_and_successor_graph(probe, prod_w_uv, tile):
    case = make_case(32, PROD_HEADS, pitch=PROD_PITCH, offset=PROD_GATE_OFFSET, seed=70)
    runner = _prepare(prod_w_uv, tile=tile, enable_pdl=True)
    chain = _PdlChain(probe, runner, case, tile)
    chain.stage(71)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        chain.run()  # warm-up outside capture
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        chain.run()
    for seed in (72, 73, 74):
        staged = chain.stage(seed)
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        chain.check(staged, prod_w_uv, f"graph {seed}")
