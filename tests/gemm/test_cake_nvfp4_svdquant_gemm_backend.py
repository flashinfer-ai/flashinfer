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

Tests of the Cake NVFP4 SVDQuant GEMM backend package (SM100 / SM103), the
programs behind ``mm_nvfp4_svdquant(..., backend="cake")``.

Every program computes ``out = bf16(alpha * (deq(A) @ deq(B)^T + D @ L1^T) +
bias)`` with FP32 accumulation; the caller has divided ``L1`` by ``alpha``
beforehand.  The suite compares each kernel output with a plain-torch FP32
reference of that formula on the kernel-test operands (random E2M1 codes,
UE4M3 block scales between 2^-3 and 2^1 in the padded 128 x 4 swizzled
layout, ``2 randn`` low-rank factors, ``4 randn`` bias): every element within
``atol 1 + rtol 0.1`` and a signal-to-quantization-noise ratio above 40 dB.
The matrix covers the family's 28 performance rows (every shipped program),
the off-table rows ``M in {128, 129, 135, 191, 200, 333, 2048, 4097}`` x
``rank in {32, 64, 96, 128}`` x bias at ``N = K = 3072``, the per-K routing of
the M 129 rows and the output-store form of the 2-CTA routes, the named
``ValueError`` of each shape rule, the host plan (route table, grid rules,
raster groups, scale-factor views), CUDA Graph capture and replay (bitwise
equal to eager, two layers interleaved in one graph and in alternating
graphs), pooled scale-factor buffers, run-to-run determinism, the
``enable_pdl`` contract and the ``alpha`` first-element rule.

This suite exercises the backend package alone: it names the package once,
through ``tests.test_helpers.cake_package``, and imports nothing else from
flashinfer, so a build that vendors the package under another import path
runs it unchanged after pointing that table entry at its package.  The
``mm_nvfp4_svdquant`` public entry (backend selection, the CUDA 13.0 gate,
``enable_pdl`` plumbing) is covered by ``test_nvfp4_svdquant_gemm.py``.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest
import torch

from tests.test_helpers.cake_package import cake_package

pkg = cake_package("cake_nvfp4_svdquant_gemm")
cake_build = cake_package("cake_nvfp4_svdquant_gemm", "cake_build")
loader = cake_package("cake_nvfp4_svdquant_gemm", "cake_jit")

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3))
# The public ``backend="cake"`` requires CUDA 13.0; the suite follows that gate.
MIN_CUDA = (13, 0)
# The contract's tolerances against the FP32 reference.
ATOL, RTOL = 1.0, 0.1
MIN_SQNR_DB = 40.0
SF_BLOCK = 16
E2M1 = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)
SEED = 1351

# The family's performance rows ``(M, N, K, rank, bias)``: the SVDQuant linear
# shapes of the Cake program (decode M = 129, prefill M up to 16384, the
# 3072 / 12288 N and K pairs, every rank).  Together they route to every
# shipped program.
PERF_ROWS: tuple[tuple[int, int, int, int, bool], ...] = (
    (129, 3072, 3072, 32, True),
    (129, 3072, 3072, 32, False),
    (129, 3072, 3072, 64, True),
    (129, 3072, 3072, 96, True),
    (129, 3072, 3072, 128, True),
    (129, 3072, 3072, 128, False),
    (129, 3072, 12288, 32, False),
    (129, 3072, 12288, 32, True),
    (129, 3072, 12288, 128, False),
    (129, 3072, 12288, 128, True),
    (6912, 3072, 3072, 32, True),
    (6912, 3072, 3072, 64, True),
    (6912, 3072, 3072, 96, True),
    (6912, 3072, 3072, 128, True),
    (6912, 3072, 12288, 32, True),
    (6912, 3072, 12288, 128, True),
    *(
        (m, n, k, 32, True)
        for n, k in ((3072, 3072), (3072, 12288), (12288, 3072))
        for m in (4096, 6889, 9216, 16384)
    ),
)
# Off-table correctness rows: ragged and tiny M at one (N, K) inside every
# program's K tile, every rank, with and without bias.
CONTRACT_M = (128, 129, 135, 191, 200, 333, 2048, 4097)
CONTRACT_N = 3072
CONTRACT_K = 3072
RANKS = (32, 64, 96, 128)
# Coverage rows: the programs the two lists above do not reach under the
# current route table (tactic 25: K 12288 at rank 32 between M 256 and 4095;
# the rank 64 / 96 tail-split builds; the hinted-store persistent builds of
# ranks 64 / 96 / 128 at a deep K).  Update them when the route table changes;
# ``test_matrix_reaches_every_program`` says when a program is missed.
COVERAGE_ROWS: tuple[tuple[int, int, int, int, bool], ...] = (
    (333, 3072, 12288, 32, True),
    (6912, 3072, 12288, 64, True),
    (6912, 3072, 12288, 96, True),
    (4096, 3072, 12288, 64, True),
    (4096, 3072, 12288, 96, True),
    (4096, 3072, 12288, 128, True),
)
CONTRACT_ROWS: tuple[tuple[int, int, int, int, bool], ...] = tuple(
    (m, CONTRACT_N, CONTRACT_K, rank, bias)
    for m in CONTRACT_M
    for rank in RANKS
    for bias in (False, True)
)
# (M, N, K, rank, bias) -> the substring of the ValueError the shape rule raises.
REJECTED_SHAPES = (
    ((65, 3072, 3072, 32, True), "M >= 128"),
    ((127, 3072, 3072, 32, False), "M >= 128"),
    ((129, 3104, 3072, 32, True), "multiple of 128"),
    ((129, 64, 3072, 32, True), "multiple of 128"),
    ((333, 3072, 3200, 96, True), "multiple of 256 for rank=96 with bias"),
    ((129, 3072, 3136, 32, True), "multiple of 128 for rank=32 with bias"),
    ((333, 3072, 3200, 96, False), "multiple of 256 for rank=96, got K=3200"),
    ((129, 3072, 3072, 160, True), "between 32 and 128"),
    ((129, 3072, 3072, 48, False), "between 32 and 128"),
    ((129, 3072, 3072, 0, True), "between 32 and 128"),
)
GRAPH_ROWS = (
    (129, 3072, 3072, 32, True),
    (129, 3072, 3072, 32, False),
    (129, 3072, 3072, 96, True),
    (129, 3072, 12288, 32, False),
    (333, 3072, 3072, 64, False),
    (4096, 3072, 3072, 64, True),
    (4097, 3072, 12288, 128, True),
    (6912, 3072, 12288, 32, True),
)
GRAPH_REPLAYS = 3
# The build keywords of the 64-row tactic-14 program per K class of the M 129
# rows: the K12288 class (K >= 12288, K % 256 == 0) and every other K.
M129_FORM_K3072 = {"prologue": "overlap", "epi": "fused64"}
M129_FORM_K12288 = {
    "prologue": "overlap",
    "epi": "fused64",
    "k_box": 256,
    "stages": 11,
    "epi_alias": True,
}
# The 2-CTA programs' output-store forms by launch: the kernel hints its stores
# when ``K_tiles >= 24`` or ``w_tiles >= 24`` on 256-wide tiles.
OUT_HINT_MIN_TILES = 24


def _hinted_launch(n: int, k: int) -> bool:
    return k // 256 >= OUT_HINT_MIN_TILES or -(-n // 256) >= OUT_HINT_MIN_TILES


def _device() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device())


def _capability() -> tuple[int, int]:
    return tuple(torch.cuda.get_device_capability(_device()))


def _toolkit() -> tuple[int, int] | None:
    try:
        return cake_build.record_cuda_version(loader.arch_for_capability(_capability()))
    except Exception:
        return None


def _supported() -> bool:
    if not torch.cuda.is_available() or _capability() not in SUPPORTED_CAPABILITIES:
        return False
    toolkit = _toolkit()
    return toolkit is not None and toolkit >= MIN_CUDA


pytestmark = pytest.mark.skipif(
    not _supported(),
    reason="cake_nvfp4_svdquant_gemm needs an SM100 / SM103 device and a CUDA 13.0+ toolkit",
)


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    # Build the local architecture's libraries that have no prebuilt file before
    # the first test, through the spec objects the package's loader itself builds
    # on first use (``cake_build.load_library``); a prebuilt or already cached
    # library loads without compiling.
    if _supported():
        arch = loader.arch_for_capability(_capability())
        for spec in loader.specs_for_arch(arch):
            spec.build_and_load()
    yield


def _row_id(row) -> str:
    m, n, k, rank, bias = row
    return f"M{m}-N{n}-K{k}-r{rank}-{'bias' if bias else 'nobias'}"


def _generator(*keys: int) -> torch.Generator:
    seed = SEED
    for key in keys:
        seed = (seed * 1_000_003 + int(key)) % (2**31 - 1)
    return torch.Generator(device=_device()).manual_seed(seed)


def swizzle_128x4(scale: torch.Tensor) -> torch.Tensor:
    """``[rows, K/16]`` scale bytes in FlashInfer's padded 128 x 4 swizzled layout, flat.

    Scale ``(r, c)`` lands at ``(r // 128) 128 K/16 + (c // 4) 512 + (r % 32) 16
    + (r % 128 // 32) 4 + c % 4``; the padding rows are zero.
    """
    rows, cols = scale.shape
    padded = -(-rows // 128) * 128
    r = torch.arange(rows, device=scale.device)[:, None]
    c = torch.arange(cols, device=scale.device)[None, :]
    index = (
        (r // 128) * 128 * cols
        + (c // 4) * 512
        + (r % 32) * 16
        + (r % 128 // 32) * 4
        + c % 4
    )
    out = torch.zeros(padded * cols, device=scale.device, dtype=torch.uint8)
    out[index.flatten()] = scale.flatten()
    return out


def dequantize(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """FP32 values of ``[rows, K/2]`` packed E2M1 pairs (even element in the low
    nibble) under ``[rows, K/16]`` UE4M3 block scales."""
    lut = torch.tensor(E2M1, device=packed.device)
    values = torch.stack(
        [lut[(packed & 0xF).long()], lut[(packed >> 4).long()]], dim=-1
    ).flatten(-2)
    scales = scale.view(torch.float8_e4m3fn).float().repeat_interleave(SF_BLOCK, dim=-1)
    return values * scales


class Problem:
    """The operands of one ``(M, N, K, rank)`` problem and its FP32 reference."""

    def __init__(self, m: int, n: int, k: int, rank: int, *, key: int = 0):
        device = _device()
        generator = _generator(m, n, k, rank, key)

        def randint(low: int, high: int, *shape: int) -> torch.Tensor:
            return torch.randint(
                low, high, shape, device=device, generator=generator, dtype=torch.uint8
            )

        def randn(*shape: int, scale: float) -> torch.Tensor:
            return (
                torch.randn(*shape, device=device, generator=generator)
                .mul_(scale)
                .to(torch.bfloat16)
            )

        self.m, self.n, self.k, self.rank = m, n, k, rank
        self.a = randint(0, 256, m, k // 2)
        self.b = randint(0, 256, n, k // 2)
        # UE4M3 bytes 32 .. 63: exponent field 4 .. 7, scales 2^-3 .. 2^1 (x 1.875).
        self.a_scale = randint(32, 64, m, k // SF_BLOCK)
        self.b_scale = randint(32, 64, n, k // SF_BLOCK)
        self.a_sf = swizzle_128x4(self.a_scale)
        self.b_sf = swizzle_128x4(self.b_scale)
        self.d = randn(m, rank, scale=2.0)
        self.l1 = randn(n, rank, scale=2.0)
        self.alpha = torch.tensor([0.5], device=device, dtype=torch.float32)
        self.bias = randn(n, scale=4.0)

    def reference(self, has_bias: bool) -> torch.Tensor:
        acc = dequantize(self.a, self.a_scale) @ dequantize(self.b, self.b_scale).T
        acc = acc + self.d.float() @ self.l1.float().T
        out = acc * self.alpha
        if has_bias:
            out = out + self.bias.float()
        return out

    def out(self) -> torch.Tensor:
        return torch.empty(self.m, self.n, dtype=torch.bfloat16, device=_device())

    def run(
        self,
        has_bias: bool,
        *,
        out: torch.Tensor | None = None,
        enable_pdl: bool | None = None,
        alpha: torch.Tensor | None = None,
        a_sf: torch.Tensor | None = None,
        b_sf: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return pkg.run(
            a=self.a,
            b=self.b,
            a_sf=self.a_sf if a_sf is None else a_sf,
            b_sf=self.b_sf if b_sf is None else b_sf,
            alpha=self.alpha if alpha is None else alpha,
            d=self.d,
            l1=self.l1,
            bias=self.bias if has_bias else None,
            out=self.out() if out is None else out,
            enable_pdl=enable_pdl,
        )


def _sqnr_db(reference: torch.Tensor, actual: torch.Tensor) -> float:
    noise = (actual - reference).square().sum()
    if noise == 0:
        return math.inf
    return float(10 * torch.log10(reference.square().sum() / noise))


def _check(reference: torch.Tensor, out: torch.Tensor, what: str) -> None:
    assert out.dtype == torch.bfloat16 and out.shape == reference.shape, what
    got = out.float()
    assert torch.isfinite(got).all(), f"{what}: non-finite output"
    error = (got - reference).abs()
    tolerance = ATOL + RTOL * reference.abs()
    outside = error > tolerance
    assert not outside.any(), (
        f"{what}: {int(outside.sum())} of {outside.numel()} elements outside "
        f"atol {ATOL} + rtol {RTOL}; max |error| {float(error.max()):.4g}"
    )
    sqnr = _sqnr_db(reference, got)
    assert sqnr > MIN_SQNR_DB, f"{what}: SQNR {sqnr:.2f} dB <= {MIN_SQNR_DB} dB"


def _arch() -> str:
    return loader.device_arch(_device().index)


def _route(row) -> str:
    m, n, k, rank, bias = row
    return loader.select_record(m=m, n=n, k=k, rank=rank, has_bias=bias, arch=_arch())


# --------------------------------------------------------------------------- host plan


def test_registry_records_reference_the_build_manifest() -> None:
    """Every ``MODULES`` record names its ``build.json`` library and nothing to compile.

    The manifest (schema ``cake-build/1``) carries exactly one entry per (library,
    architecture) the registry references, each with the kernel and binding
    sources under ``csrc/`` of this package, their digests, a closure digest and
    the family's ``--use_fast_math``; the registry itself records the host plan
    only, in the vocabularies it publishes.
    """
    manifest = cake_build.manifest()
    assert manifest["schema"] == "cake-build/1"
    assert manifest["package"] == pkg.__name__
    assert set(manifest["arch_flags"]) == set(loader.ARCHES) == {"sm_100a", "sm_103a"}
    assert manifest["common"]["requires"]["cuda_toolkit"]["min"]
    libraries = cake_build.libraries_by_name()
    package = Path(cake_build.__file__).parent
    referenced = set()
    for name, record in loader.MODULES.items():
        assert record["library"] == record["kernel"] == name
        assert re.fullmatch(r"cake_nvfp4_svdquant_gemm_[0-9a-f]{20}", name)
        assert record["kernel_symbol"] == f"kernel_{name}"
        assert record["ffi_entry"] == "run"
        assert not {"sources", "source_sha256", "compile_flags", "cache_names"} & set(
            record
        ), name
        assert record["arches"] and set(record["arches"]) <= set(loader.ARCHES), name
        launch = record["launch"]
        assert launch["cooperative"] is False
        assert len(launch["block"]) == 3 and launch["block"][0] >= 128
        assert len(launch["cluster"]) == 3 and all(c >= 1 for c in launch["cluster"])
        assert launch["dynamic_smem_bytes"] > 0
        plan = record["plan"]
        assert plan["pdl"] is launch["use_pdl"], name
        assert plan["variant"] in loader.VARIANTS
        assert plan["grid_rule"] in loader.GRID_RULES
        assert plan["scale_elems"] in loader.SCALE_ELEMS
        # M tiles are a multiple of the 128-row tensor-map box (the pair programs) or an even divisor of it (the
        # 64-row programs).
        block_m = plan["block_m"]
        assert block_m % loader.BLOCK_M == 0 or loader.BLOCK_M % block_m == 0, name
        assert plan["block_n"] > 0
        assert plan["block_k"] % 64 == 0
        storage = plan["rank_storage"]
        assert storage is None or storage in RANKS, name
        assert record["build"].get("rank", storage) == storage, name
        parameters = {key for kind, key in record["arg_plan"] if kind != "grid"}
        assert parameters == set(record["logical"]), name
        assert [key for kind, key in record["arg_plan"] if kind == "grid"] == [
            "grid_x",
            "grid_y",
            "grid_z",
        ]
        assert set(record["logical"].values()) <= set(loader.LOGICAL_TENSORS) | set(
            loader.LOGICAL_SCALARS
        )
        for arch in record["arches"]:
            library = loader.library_name(name, arch)
            referenced.add(library)
            entry = libraries[library]
            assert entry["arch"] == arch
            assert entry["entry"] == "run"
            assert entry["sources"] == [
                f"csrc/{record['library']}_kernel.cu",
                f"csrc/{record['library']}_binding.cu",
            ]
            assert set(entry["sha256"]) == set(entry["sources"])
            for source in entry["sources"]:
                assert (package / source).is_file(), source
            assert re.fullmatch(r"[0-9a-f]{64}", entry["closure_sha256"])
            assert entry["nvcc_flags"] == ["--use_fast_math"], library
            assert re.fullmatch(r"\d+\.\d+", entry["min_cuda"]), library
    assert referenced == set(libraries)
    assert (package / "csrc" / "cake_ffi_checks.h").is_file()


def test_route_table_is_total_and_reaches_every_program() -> None:
    """The last route row admits everything, every row names shipped records
    (and, when it names architectures, some but not all of the exported ones),
    and every shipped record is the target of some row."""
    table = loader.ROUTE_TABLE
    assert set(table[-1]) == {"record"}
    targets = set()
    for row in table:
        arches = row.get("arches")
        assert arches is None or (arches and set(arches) < set(loader.ARCHES))
        if "records_by_rank" in row:
            assert "record" not in row
            for rank, name in row["records_by_rank"].items():
                assert rank in RANKS
                assert loader.MODULES[name]["plan"]["rank_storage"] in (None, rank)
                targets.add(name)
        else:
            targets.add(row["record"])
    assert targets == set(loader.MODULES)
    assert set(loader.RASTER_GROUP_TABLE[-1]) == {"group"}


@pytest.mark.parametrize("row", PERF_ROWS, ids=_row_id)
def test_perf_rows_route_to_an_admitting_program(row) -> None:
    m, n, k, rank, bias = row
    name = _route(row)
    plan = loader.MODULES[name]["plan"]
    assert k % plan["block_k"] == 0
    assert not bias or plan["variant"] == "bias"
    assert plan["rank_storage"] in (None, rank)
    assert pkg.is_supported(m=m, n=n, k=k, rank=rank, has_bias=bias, device=_device())


def test_matrix_reaches_every_program() -> None:
    """The perf, off-table and coverage rows together exercise every shipped program."""
    reached = {_route(row) for row in (*PERF_ROWS, *CONTRACT_ROWS, *COVERAGE_ROWS)}
    assert reached == set(loader.MODULES), sorted(set(loader.MODULES) - reached)


def test_route_rows_may_name_their_architectures(monkeypatch) -> None:
    """A row with ``arches`` applies on those architectures only: one problem
    dispatches per architecture, an architecture-blind call that meets such a
    row is refused, and an unexported architecture is refused."""
    m, n, k, rank, bias = 4096, 3072, 3072, 32, True
    before = {
        arch: loader.select_record(m=m, n=n, k=k, rank=rank, has_bias=bias, arch=arch)
        for arch in loader.ARCHES
    }
    beside = {
        arch: loader.select_record(
            m=m + 1, n=n, k=k, rank=rank, has_bias=bias, arch=arch
        )
        for arch in loader.ARCHES
    }
    other = next(
        name
        for name, record in loader.MODULES.items()
        if name not in before.values()
        and record["plan"]["variant"] == "bias"
        and record["plan"]["rank_storage"] in (None, rank)
        and k % record["plan"]["block_k"] == 0
    )
    scoped = {
        "m_min": m,
        "m_max": m,
        "n_min": n,
        "n_max": n,
        "k_min": k,
        "k_max": k,
        "ranks": (rank,),
        "bias": True,
        "arches": ("sm_103a",),
        "record": other,
    }
    monkeypatch.setattr(loader, "ROUTE_TABLE", (scoped, *loader.ROUTE_TABLE))
    routed = {
        arch: loader.select_record(m=m, n=n, k=k, rank=rank, has_bias=bias, arch=arch)
        for arch in loader.ARCHES
    }
    assert routed == {"sm_100a": before["sm_100a"], "sm_103a": other}
    # A neighbouring problem the scoped row does not admit routes as before.
    for arch in loader.ARCHES:
        assert (
            loader.select_record(m=m + 1, n=n, k=k, rank=rank, has_bias=bias, arch=arch)
            == beside[arch]
        )
    with pytest.raises(TypeError, match="per architecture"):
        loader.select_record(m=m, n=n, k=k, rank=rank, has_bias=bias)
    with pytest.raises(RuntimeError, match="not an exported"):
        loader.select_record(m=m, n=n, k=k, rank=rank, has_bias=bias, arch="sm_120a")
    assert pkg.is_supported(m=m, n=n, k=k, rank=rank, has_bias=bias, device=_device())
    assert _route((m, n, k, rank, bias)) == routed[_arch()]


@pytest.mark.parametrize("case", REJECTED_SHAPES, ids=lambda case: _row_id(case[0]))
def test_shape_rule_names_the_violated_constraint(case) -> None:
    (m, n, k, rank, bias), message = case
    with pytest.raises(ValueError, match=re.escape(message)):
        loader.select_record(m=m, n=n, k=k, rank=rank, has_bias=bias, arch=_arch())
    assert not pkg.is_supported(
        m=m, n=n, k=k, rank=rank, has_bias=bias, device=_device()
    )


def test_run_rejects_a_shape_before_launching() -> None:
    device = _device()
    m, n, k, rank = 64, 3072, 3072, 32
    a = torch.zeros(m, k // 2, dtype=torch.uint8, device=device)
    b = torch.zeros(n, k // 2, dtype=torch.uint8, device=device)
    a_sf = torch.zeros(128 * k // SF_BLOCK, dtype=torch.uint8, device=device)
    b_sf = torch.zeros(n * k // SF_BLOCK, dtype=torch.uint8, device=device)
    with pytest.raises(ValueError, match="M >= 128"):
        pkg.run(
            a=a,
            b=b,
            a_sf=a_sf,
            b_sf=b_sf,
            alpha=torch.ones(1, device=device),
            d=torch.zeros(m, rank, dtype=torch.bfloat16, device=device),
            l1=torch.zeros(n, rank, dtype=torch.bfloat16, device=device),
            bias=None,
            out=torch.empty(m, n, dtype=torch.bfloat16, device=device),
        )


def test_is_supported_needs_a_cuda_device() -> None:
    assert not pkg.is_supported(
        m=129, n=3072, k=3072, rank=32, has_bias=True, device=torch.device("cpu")
    )
    assert pkg.is_supported(
        m=129, n=3072, k=3072, rank=32, has_bias=True, device=_device()
    )


@pytest.mark.parametrize(
    "m,n", [(129, 3072), (333, 12288), (4097, 3072), (16384, 3072)]
)
def test_launch_grid_follows_the_record_rule(m, n) -> None:
    grid_m = -(-m // loader.BLOCK_M)
    sms = 148
    seen = set()
    for name, record in loader.MODULES.items():
        plan = record["plan"]
        grid_n = n // plan["block_n"]
        rule = plan["grid_rule"]
        seen.add(rule)
        grid = loader.launch_grid(record, m=m, n=n, k=3072, sm_count=sms)
        if rule == "linear":
            assert grid == (grid_m * grid_n, 1, 1), name
        elif rule == "cluster_n2":
            tiles_m = -(-m // plan["block_m"])
            assert grid == (tiles_m, grid_n, 1) and grid_n % 2 == 0, name
        elif rule == "pair_persistent":
            tok = -(-m // plan["block_m"])
            w = -(-n // plan["block_n"])
            assert grid == (2 * tok * w, 1, 1), name
        elif rule == "pair_tailsplit":
            tiles, units, full, split = loader.tailsplit_units(m=m, n=n, sm_count=sms)
            assert grid == (2 * units, 1, 1), name
            assert units == full + (tiles - full) * 2 and split in (1, 2), name
            with pytest.raises(ValueError, match="sm_count"):
                loader.launch_grid(record, m=m, n=n, k=3072)
        else:
            pytest.fail(f"{name}: unknown grid rule {rule!r}")
        assert all(axis >= 1 for axis in grid), name
    assert seen == set(loader.GRID_RULES)


def test_tailsplit_units_schedule() -> None:
    """The tail-split schedule: the tail tiles of the last partial wave become
    half units when they fit in half the CTA pairs, whole tiles otherwise."""
    units = loader.tailsplit_units
    # 324 pair tiles: 76 pairs -> 4 waves + 20 tail tiles -> 40 half units;
    # 74 pairs -> 4 waves + 28 -> 56 half units.
    assert units(m=6912, n=3072, sm_count=152) == (324, 344, 304, 2)
    assert units(m=6889, n=3072, sm_count=148) == (324, 352, 296, 2)
    # 768 pair tiles: 10 waves + 8 (76 pairs) / 10 waves + 28 (74 pairs).
    assert units(m=16384, n=3072, sm_count=152) == (768, 776, 760, 2)
    assert units(m=4096, n=12288, sm_count=148) == (768, 796, 740, 2)
    # 192 pair tiles on 76 pairs: 2 waves + 40 tail tiles, 80 > 76 -> whole tiles.
    assert units(m=4096, n=3072, sm_count=152) == (192, 192, 192, 1)
    # A whole number of waves: nothing to split.
    assert units(m=4096, n=3072, sm_count=128) == (192, 192, 192, 1)
    # Forced schedules and the admitted values.
    assert units(m=6912, n=3072, sm_count=152, tail_split=1) == (324, 324, 324, 1)
    assert units(m=4096, n=3072, sm_count=152, tail_split=2) == (192, 232, 152, 2)
    with pytest.raises(ValueError, match="tail_split"):
        units(m=6912, n=3072, sm_count=152, tail_split=4)


def test_tailsplit_rows_split_on_this_device() -> None:
    """Every row routed to the tail-split program splits its tail on the
    device the suite runs on (148 or 152 SMs)."""
    sms = loader.sm_count(_device().index or 0)
    for row in (*PERF_ROWS, *COVERAGE_ROWS):
        m, n, k, rank, bias = row
        if loader.MODULES[_route(row)]["plan"]["grid_rule"] != "pair_tailsplit":
            continue
        tiles, units, full, split = loader.tailsplit_units(m=m, n=n, sm_count=sms)
        assert split == 2 and full < units, (_row_id(row), sms)


def test_raster_group_follows_its_table() -> None:
    assert loader.raster_group(m=4096, n=3072, k=12288, rank=32) == 2
    assert loader.raster_group(m=16384, n=12288, k=3072, rank=32) == 2
    assert loader.raster_group(m=16384, n=12288, k=12288, rank=32) == 2
    assert loader.raster_group(m=4096, n=3072, k=3072, rank=32) == 8
    assert loader.raster_group(m=6912, n=3072, k=3072, rank=128) == 8


def test_two_cta_routes_pick_the_store_form_of_the_launch() -> None:
    """Every 2-CTA route (whole-tile and tail-split) lands on the build whose
    ``out_store`` matches the kernel's hint rule for that launch, and both forms
    are reached by the performance rows."""
    forms = set()
    for row in (*PERF_ROWS, *CONTRACT_ROWS, *COVERAGE_ROWS):
        m, n, k, rank, bias = row
        record = loader.MODULES[_route(row)]
        if record["plan"]["grid_rule"] not in ("pair_persistent", "pair_tailsplit"):
            continue
        build = record["build"]
        assert build["rank"] == rank, _row_id(row)
        expected = "v8hint" if _hinted_launch(n, k) else "v8"
        assert build["out_store"] == expected, (_row_id(row), build)
        if row in PERF_ROWS:
            forms.add(build["out_store"])
    assert forms == {"v8", "v8hint"}
    # The boundary of the hint rule on the whole-tile program: 23 vs 24 tiles.
    for (n, k), form in (
        ((3072, 5888), "v8"),
        ((3072, 6144), "v8hint"),
        ((5888, 3072), "v8"),
        ((6016, 3072), "v8hint"),
    ):
        name = loader.select_record(
            m=4096, n=n, k=k, rank=32, has_bias=True, arch=_arch()
        )
        record = loader.MODULES[name]
        assert record["plan"]["grid_rule"] == "pair_persistent", (n, k)
        assert record["build"]["out_store"] == form, (n, k, record["build"])
    # Every tail-split route is a hinted launch, so the program ships hinted only.
    for record in loader.MODULES.values():
        if record["plan"]["grid_rule"] == "pair_tailsplit":
            assert record["build"]["out_store"] == "v8hint"


@pytest.mark.parametrize("bias", [False, True], ids=["nobias", "bias"])
def test_m129_rows_route_by_k_class(bias) -> None:
    """The M 129 / N 3072 rows take the 64-row tactic-14 build of their K class:
    the K12288 form for K >= 12288 with K % 256 == 0, the K3072 form otherwise
    (including K % 256 != 0 and the K between the two classes); the bias form
    follows the problem."""
    for k, form in (
        (3072, M129_FORM_K3072),
        (3200, M129_FORM_K3072),
        (6144, M129_FORM_K3072),
        (12288, M129_FORM_K12288),
        (12416, M129_FORM_K3072),
        (24576, M129_FORM_K12288),
    ):
        name = loader.select_record(
            m=129, n=3072, k=k, rank=32, has_bias=bias, arch=_arch()
        )
        record = loader.MODULES[name]
        assert record["build"] == {"bias": bias, **form}, (k, bias, record["build"])
        assert record["plan"]["variant"] == ("bias" if bias else "nobias")
        assert record["plan"]["block_m"] == 64 and record["plan"]["block_k"] == 128


def test_out_ptr_binds_the_out_tensor() -> None:
    """The register-direct store builds take the raw ``out`` pointer as the
    trailing ``out_ptr`` parameter after ``L1_tail``; its logical name is ``out``,
    so the host passes the ``out`` tensor twice (as the tensor map and the
    pointer).  No other record has the parameter."""
    assert loader.PARAMETER_LOGICAL["out_ptr"] == "out"
    with_ptr = {
        name
        for name, record in loader.MODULES.items()
        if "out_ptr" in record["logical"]
    }
    assert with_ptr
    for name, record in loader.MODULES.items():
        direct = record["build"].get("out_store") in ("v8", "v8hint")
        assert (name in with_ptr) == direct, name
        if not direct:
            continue
        assert record["logical"]["out_ptr"] == "out"
        plan = [tuple(pair) for pair in record["arg_plan"]]
        assert (
            plan.index(("buffer", "out_ptr")) == plan.index(("buffer", "L1_tail")) + 1
        )
        assert plan[-3:] == [("grid", "grid_x"), ("grid", "grid_y"), ("grid", "grid_z")]
        logical = [
            logical
            for logical, _axis in loader._argument_plan(name)
            if logical is not None
        ]
        assert logical.count("out") == 2, name


@pytest.mark.parametrize("has_bias", [False, True])
def test_plan_values_view_the_scale_factors_in_place(has_bias) -> None:
    """Each record views the caller's scale bytes per its ``scale_elems`` (no
    copy), carries ``out`` as the bias when there is none and derives the
    kernel scalars from the plan."""
    m, n, k, rank = 129, 3072, 3072, 32
    p = Problem(m, n, k, rank)
    out = p.out()
    padded_m = 256
    quads = k // SF_BLOCK // 4
    shapes = {
        "uint8": (padded_m // 128, quads, 4, 8, 16),
        "uint16": (padded_m // 128, quads, 4, 64),
        "uint32": (padded_m // 128, quads, 128),
    }
    seen = set()
    for name, record in loader.MODULES.items():
        plan = record["plan"]
        if plan["rank_storage"] not in (None, rank):
            continue
        values = loader.plan_values(
            record,
            a=p.a,
            b=p.b,
            a_sf=p.a_sf,
            b_sf=p.b_sf,
            alpha=p.alpha,
            d=p.d,
            l1=p.l1,
            bias=p.bias if has_bias else None,
            out=out,
        )
        kind = plan["scale_elems"]
        seen.add(kind)
        assert tuple(values["a_sf"].shape) == shapes[kind], name
        assert values["a_sf"].data_ptr() == p.a_sf.data_ptr(), name
        assert values["b_sf"].data_ptr() == p.b_sf.data_ptr(), name
        assert values["b_sf"].shape[0] == n // 128, name
        assert values["alpha"].numel() == 1
        assert (values["bias"] is p.bias) if has_bias else (values["bias"] is out)
        assert values["has_bias"] == int(has_bias)
        assert values["M"] == m and values["N"] == n
        assert values["K_tiles"] == k // plan["block_k"]
        assert values["rank_tiles"] == rank // loader.RANK_GRANULARITY
        assert values["rank"] == rank and values["rank_chunks"] == 1
        assert values["grid_n"] == n // plan["block_n"]
        sms = loader.sm_count(p.a.device.index)
        assert values["grid"] == loader.launch_grid(record, m=m, n=n, k=k, sm_count=sms)
        if plan["grid_rule"] in ("pair_persistent", "pair_tailsplit"):
            assert values["raster_group"] == loader.raster_group(
                m=m, n=n, k=k, rank=rank
            )
            assert values["num_tiles"] == values["tok_tiles"] * values["w_tiles"]
        else:
            assert "raster_group" not in values
        if plan["grid_rule"] == "pair_tailsplit":
            tiles, units, full, split = loader.tailsplit_units(m=m, n=n, sm_count=sms)
            assert (
                values["num_units"],
                values["full_units"],
                values["tail_split"],
            ) == (
                units,
                full,
                split,
            )
            assert values["grid"] == (2 * units, 1, 1)
        else:
            assert "num_units" not in values
    assert seen == set(loader.SCALE_ELEMS)


# --------------------------------------------------------------------------- correctness


@pytest.mark.parametrize("row", PERF_ROWS, ids=_row_id)
def test_perf_rows_match_the_reference(row) -> None:
    m, n, k, rank, bias = row
    p = Problem(m, n, k, rank)
    out = p.run(bias)
    torch.cuda.synchronize()
    _check(p.reference(bias), out, f"{_row_id(row)} via {_route(row)}")


@pytest.mark.parametrize("bias", [False, True], ids=["nobias", "bias"])
@pytest.mark.parametrize("rank", RANKS)
@pytest.mark.parametrize("m", CONTRACT_M)
def test_off_table_shapes_match_the_reference(m, rank, bias) -> None:
    row = (m, CONTRACT_N, CONTRACT_K, rank, bias)
    p = Problem(*row[:4])
    out = p.run(bias)
    torch.cuda.synchronize()
    _check(p.reference(bias), out, f"{_row_id(row)} via {_route(row)}")


@pytest.mark.parametrize("row", COVERAGE_ROWS, ids=_row_id)
def test_coverage_rows_match_the_reference(row) -> None:
    m, n, k, rank, bias = row
    p = Problem(m, n, k, rank)
    out = p.run(bias)
    torch.cuda.synchronize()
    _check(p.reference(bias), out, f"{_row_id(row)} via {_route(row)}")


def test_run_returns_out_and_writes_it_in_place() -> None:
    p = Problem(129, 3072, 3072, 64)
    out = p.out()
    assert p.run(True, out=out) is out
    torch.cuda.synchronize()
    _check(p.reference(True), out, "M129-r64-bias")


@pytest.mark.parametrize("row", GRAPH_ROWS, ids=_row_id)
def test_runs_are_deterministic(row) -> None:
    m, n, k, rank, bias = row
    p = Problem(m, n, k, rank)
    first = p.run(bias)
    second = p.run(bias)
    torch.cuda.synchronize()
    assert torch.equal(first, second)


def test_alpha_uses_its_first_element() -> None:
    p = Problem(129, 3072, 3072, 32)
    expected = p.run(True)
    pooled = torch.tensor(
        [0.5, 7.0, -1.0, float("nan")], device=_device(), dtype=torch.float32
    )
    actual = p.run(True, alpha=pooled)
    torch.cuda.synchronize()
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("bias", [False, True], ids=["nobias", "bias"])
def test_pooled_scale_buffers_read_only_the_leading_bytes(bias) -> None:
    """A larger contiguous scale-factor buffer (a pooled allocation) gives the
    same bytes as the exact one: only ``padded_rows * K / 16`` bytes are viewed."""
    p = Problem(129, 3072, 3072, 32)
    expected = p.run(bias)
    padding = torch.full((4096,), 255, dtype=torch.uint8, device=_device())
    actual = p.run(
        bias, a_sf=torch.cat((p.a_sf, padding)), b_sf=torch.cat((p.b_sf, padding))
    )
    torch.cuda.synchronize()
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("row", GRAPH_ROWS, ids=_row_id)
def test_cuda_graph_replay_matches_eager_bitwise(row) -> None:
    m, n, k, rank, bias = row
    p = Problem(m, n, k, rank)
    out = p.out()
    p.run(bias, out=out)
    torch.cuda.synchronize()
    eager = out.clone()
    _check(p.reference(bias), eager, _row_id(row))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        p.run(bias, out=out)
    for _ in range(GRAPH_REPLAYS):
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, eager)


def test_layers_interleave_in_one_graph_and_in_alternating_graphs() -> None:
    """Three layers with their own shapes, weights, rank, bias and program (the
    64-row cluster program, the PDL tail-split pair program and the linear
    fallback program) replay correctly from one graph and from alternating
    graphs: every launch carries its own tensor maps by value, so no layer
    observes another's descriptors."""
    rows = (
        (129, 3072, 3072, 32, True),
        (6912, 3072, 12288, 32, True),
        (333, 3072, 3072, 32, True),
    )
    routes = [_route(row) for row in rows]
    assert len(set(routes)) == 3, routes
    layers = [Problem(*row[:4], key=index + 1) for index, row in enumerate(rows)]
    outs = [layer.out() for layer in layers]
    for layer, row, out in zip(layers, rows, outs, strict=True):
        layer.run(row[4], out=out)
    torch.cuda.synchronize()
    eager = [out.clone() for out in outs]
    for layer, row, got in zip(layers, rows, eager, strict=True):
        _check(layer.reference(row[4]), got, _row_id(row))

    together = torch.cuda.CUDAGraph()
    with torch.cuda.graph(together):
        for layer, row, out in zip(layers, rows, outs, strict=True):
            layer.run(row[4], out=out)
    for _ in range(GRAPH_REPLAYS):
        for out in outs:
            out.fill_(float("nan"))
        together.replay()
        torch.cuda.synchronize()
        for out, expected in zip(outs, eager, strict=True):
            assert torch.equal(out, expected)

    graphs = []
    for layer, row, out in zip(layers, rows, outs, strict=True):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            layer.run(row[4], out=out)
        graphs.append(graph)
    for _ in range(GRAPH_REPLAYS):
        for out in outs:
            out.fill_(float("nan"))
        for graph in (*graphs, graphs[1], graphs[0]):
            graph.replay()
        torch.cuda.synchronize()
        for out, expected in zip(outs, eager, strict=True):
            assert torch.equal(out, expected)


def test_enable_pdl_must_match_the_selected_program() -> None:
    """``enable_pdl`` is ``None`` or the selected program's launch attribute; the
    opposite value is a ``ValueError`` naming the required one.  Both a PDL and a
    non-PDL program are exercised when the route table reaches both."""
    by_pdl: dict[bool, tuple[int, int, int, int, bool]] = {}
    for row in PERF_ROWS:
        pdl = bool(loader.MODULES[_route(row)]["launch"]["use_pdl"])
        by_pdl.setdefault(pdl, row)
    assert by_pdl, "no perf row routes"
    for program_pdl, row in sorted(by_pdl.items()):
        m, n, k, rank, bias = row
        p = Problem(m, n, k, rank)
        expected = p.run(bias)
        matching = p.run(bias, enable_pdl=program_pdl)
        torch.cuda.synchronize()
        assert torch.equal(matching, expected), _row_id(row)
        with pytest.raises(ValueError, match=f"enable_pdl={program_pdl}"):
            p.run(bias, enable_pdl=not program_pdl)
