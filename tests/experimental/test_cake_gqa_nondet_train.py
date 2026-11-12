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

Backend suite of the Cake non-deterministic (``nondet``) packed-varlen GQA training attention package (head
dimension 256, ``flashinfer.experimental.cake.cake_gqa_nondet_train``, the backend of
``flashinfer.cake_gqa_nondet_train``), whose every output nonetheless repeats bit for bit run to run.

Without a device: the registry (one record per stage and architecture, the argument plans the host passes), the
build manifest (every record's library, the recorded source digests), the standalone loader and its pinned bytes,
no environment read, the workspace chunk plan and the launch grids against the exact dS^T tile counts of generated
layouts (every KV head and key block in exactly one chunk, every chunk's tiles within its stride, every Q-tile pair
in the dQ grid), the device tile numbering against the same counts, and the scratch layout.

On SM100 / SM103 / SM107: the output, the LSE and every gradient against an FP64 reference of the same BF16 inputs on
packed layouts (unequal lengths, tails that are no multiple of a tile, keys fewer than queries, empty segments) in
the three dK / dV forms, under the accuracy budget declared below; rows that see no key exactly (``out = 0``,
``lse = -inf``, zero gradients); every output repeating bit for bit run to run (repeated row-map destination rows
into non-zero destinations included), beside resident kernels on other streams and through any workspace; small
workspace budgets (KV-head and key-block chunks) under the same accuracy budget, bitwise reproducible, with the
dK / dV bits of the unchunked call; the row-map adds equal, bit for bit, to the ordered FP32 fold of the FP32 window
gradients; document isolation (bitwise); dynamic shapes through one reused workspace; launches on the operands'
device and stream while another device is current; the dS^T chunk and the window gradients allocated per call
(nothing but the scratch kept between calls); non-reentrant checkpointing (nothing retained, gradients bitwise equal
to the saved-set run); no device synchronization in steady state, with one chunk and with the minimum chunks; the
speed of the forward and the backward beside a kernel resident on another stream; under
``torch.use_deterministic_algorithms(True)``, strict and with ``warn_only=True``, no alert and no warning, and the
bits of a run outside that mode.  Also: a caller's ``dq`` off a 32-byte boundary (the bits of an allocated one); the
workspace scratch and the caller's ``dq`` filled with 0x00 and then with 0xFF bytes before the backward (the same
bits: no unwritten read).

The suite names its package once, through the shared alias helper, and imports nothing else from flashinfer.
"""

from __future__ import annotations

import ast
import bisect
import hashlib
import itertools
import json
import math
import random
import re
import statistics
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from tests.cake.test_helpers.cake_package import cake_package

pkg = cake_package("cake.cake_gqa_nondet_train")
cake_backend = cake_package("cake.cake_gqa_nondet_train", "cake_backend")
cake_build = cake_package("cake.cake_gqa_nondet_train", "cake_build")
cake_jit = cake_package("cake.cake_gqa_nondet_train", "cake_jit")

PACKAGE_DIR = Path(cake_jit.__file__).resolve().parent
PACKAGE_IMPORT = cake_jit.PACKAGE
ARCHES = tuple(cake_jit.ARCHES)
D = 256
SCALE = 0.0625
MIB = 2**20

# Accuracy budget (declared before the first run; unchanged from the package's first release): per layout and tensor,
# the relative L2 error against the FP64 reference of the same BF16 inputs is at most 1.10 times FlashAttention CuTe's
# (98c1cee0, its dedicated head-dim-256 backward) on B300 against the same reference: out 2.3e-3, lse 1.5e-7,
# dq 2.6e-3, dk 2.6e-3, dv 2.4e-3 (the largest of its calibration layouts).  FP32 window gradients and destinations are
# compared in FP32.
FA4_REL_L2 = {"out": 2.3e-3, "lse": 1.5e-7, "dq": 2.6e-3, "dk": 2.6e-3, "dv": 2.4e-3}
BUDGET = 1.10
# Run to run every output is compared bitwise.  Two workspace budgets split the backward into different key-block
# chunks, so the FP32 dQ accumulator adds the chunks' partials in another association before its one BF16 rounding:
# dQ of two budgets is compared as two dQ of one input were compared when they were reduced with atomics, one BF16
# rounding step (DQ_RTOL) plus an absolute floor for the FP32 reassociation error, DQ_ATOL_FRACTION of the tensor's
# largest magnitude (2^9 times below one BF16 step there).  O, LSE, dK and dV of two budgets are compared bitwise.
DQ_RTOL = 2**-7
DQ_ATOL_FRACTION = 2**-16
# Checkpointing: memory kept after a checkpointed forward, less the output storage.
CKPT_RETAINED_LIMIT_MIB = 2.0
# Beside a resident side-stream kernel: busy over alone within MAX_RATIO or at most MAX_EXTRA_MS above it.
SIDE_SLEEP_CYCLES = 10_000_000
PRE_SLEEP_CYCLES = 1_000_000
WARMUP = 3
RUNS = 7
MAX_RATIO = 1.5
MAX_EXTRA_MS = 0.020
# Run-to-run repetitions of the bitwise reproducibility test.
REPEATS = 5


# ----------------------------------------------------------------------------------------------------------------
# Layouts: (name, q_lens, k_lens, H_q, H_kv)

LAYOUTS = (
    ("tails", (77, 300, 1, 129), (77, 700, 1000, 129), 24, 4),
    ("keys_fewer_than_queries", (200, 150, 64), (77, 0, 63), 24, 4),
    ("empty_segments", (0, 100, 0, 37, 0), (0, 300, 50, 37, 0), 24, 4),
    ("zigzag_cp2", (250, 250, 129, 130), (250, 1000, 129, 517), 24, 4),
    ("long_keys", (256, 255), (4099, 2049), 24, 4),
    ("group_2", (300, 77), (513, 77), 8, 4),
    # 133 key blocks: the dQ GEMM flushes its accumulator every 64 key blocks.
    ("over_64_blocks", (200, 64), (9000, 17000), 24, 4),
)
LAYOUT_IDS = [layout[0] for layout in LAYOUTS]
LAYOUTS_BY_NAME = {layout[0]: layout for layout in LAYOUTS}
DKV_FORMS = ("bf16", "fp32", "rowmap")
OUTPUTS = ("out", "lse", "dq", "dk", "dv")
# The stages whose CTAs run as pairs (``cta_group::2``), and the first kernel parameter of each stage's plan.
PAIR_STAGES = ("fwd", "bwd_main_fp32", "bwd_main_bf16", "bwd_dq")
FIRST_PARAMETER = {
    "fwd": "Q",
    "bwd_pre": "O",
    "bwd_plan": "cu_seqlens_q",
    "bwd_main_fp32": "Q",
    "bwd_main_bf16": "Q",
    "bwd_dq": "dsw",
    "bwd_fold": "dk_win",
}


def _cu(lens) -> list[int]:
    cu = [0]
    for n in lens:
        cu.append(cu[-1] + int(n))
    return cu


def _device_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    arch = cake_backend.arch_for_capability(torch.cuda.get_device_capability(0))
    if arch is None:
        pytest.skip(
            f"the package ships {ARCHES}; this device is {torch.cuda.get_device_capability(0)}"
        )
    if not (PACKAGE_DIR / "build.json").is_file():
        pytest.skip("the package has no build.json yet (not exported)")
    return torch.device("cuda:0")


@pytest.fixture(scope="module")
def device() -> torch.device:
    dev = _device_or_skip()
    # Build the local architecture's libraries before the first timed or sync-checked call.
    for spec in cake_backend.specs_for_arch(cake_backend.device_arch(dev.index)):
        spec.build_and_load()
    return dev


def _manifest_or_skip() -> dict:
    path = PACKAGE_DIR / "build.json"
    if not path.is_file():
        pytest.skip("the package has no build.json yet (not exported)")
    return json.loads(path.read_text())


_BITS = {torch.bfloat16: torch.int16, torch.float32: torch.int32}


def _assert_same_bits(got: torch.Tensor, want: torch.Tensor, label: str) -> None:
    """``got`` and ``want`` hold the same bits (dtype, shape and every element's bit pattern)."""
    assert got.dtype == want.dtype and got.shape == want.shape, (
        label,
        got.dtype,
        want.dtype,
        tuple(got.shape),
        tuple(want.shape),
    )
    bits = _BITS[got.dtype]
    if torch.equal(got.contiguous().view(bits), want.contiguous().view(bits)):
        return
    diff = (got.double() - want.double()).abs()
    diff = diff[torch.isfinite(diff)]
    raise AssertionError(
        f"{label}: the bits differ ({int((got != want).sum())} elements by value, max |diff| "
        f"{float(diff.max()) if diff.numel() else float('nan'):.3e})"
    )


def _assert_outputs_same_bits(got: dict, want: dict, label: str, names=OUTPUTS) -> None:
    for t in names:
        _assert_same_bits(got[t], want[t], f"{label}: {t}")


# ----------------------------------------------------------------------------------------------------------------
# Without a device


def test_registry_records_are_well_formed() -> None:
    assert tuple(cake_jit.STAGES) == (
        "fwd",
        "bwd_pre",
        "bwd_plan",
        "bwd_main_fp32",
        "bwd_main_bf16",
        "bwd_dq",
        "bwd_fold",
    )
    for arch in ARCHES:
        for stage in cake_jit.STAGES:
            serving = [
                n
                for n, rec in cake_jit.MODULES.items()
                if rec["stage"] == stage and arch in rec["arches"]
            ]
            assert len(serving) == 1, (stage, arch, serving)
    prefix = cake_jit.LIBRARY_PREFIX + "_"
    for name, rec in cake_jit.MODULES.items():
        assert set(rec["arches"]) <= set(ARCHES), name
        assert rec["ffi_entry"] == "run", name
        assert re.fullmatch(re.escape(prefix) + "[0-9a-f]{20}", rec["library"]), (
            name,
            rec["library"],
        )
        assert rec["kernel_symbol"] == "kernel_" + rec["library"], name
        launch = rec["launch"]
        assert launch["cooperative"] is False and launch["use_pdl"] is False, name
        pair = rec["stage"] in PAIR_STAGES
        assert tuple(launch["cluster"]) == ((2, 1, 1) if pair else (1, 1, 1)), name
        assert launch["block"][0] % 32 == 0, name
        # Every parameter the host passes is in the plan, in the order the kernel declares.
        assert cake_backend.parameters(name)[0] == FIRST_PARAMETER[rec["stage"]], name
    # The two precisions of the main kernel take one argument plan (one launch of the host).
    for arch in ARCHES:
        assert cake_backend.parameters(
            cake_jit.stage_record("bwd_main_fp32", arch)
        ) == cake_backend.parameters(cake_jit.stage_record("bwd_main_bf16", arch)), arch


def test_manifest_covers_every_record_and_its_sources() -> None:
    manifest = _manifest_or_skip()
    assert manifest["schema"] == "cake-build/1"
    assert manifest["package"] == PACKAGE_IMPORT
    entries = {entry["name"]: entry for entry in manifest["libraries"]}
    wanted = {
        f"{rec['library']}_{arch}"
        for rec in cake_jit.MODULES.values()
        for arch in rec["arches"]
    }
    assert set(entries) == wanted
    for name, entry in entries.items():
        assert entry["arch"] in ARCHES, name
        assert entry["path"] == f"aot/{name}.so", name
        assert entry["entry"] == "run", name
        for rel, digest in entry["sha256"].items():
            assert (
                hashlib.sha256((PACKAGE_DIR / rel).read_bytes()).hexdigest() == digest
            ), (name, rel)
    assert (PACKAGE_DIR / "csrc" / "cake_ffi_checks.h").is_file()


def test_the_loader_imports_standalone_without_flashinfer() -> None:
    """The vendorable form: ``cake_build.py`` loaded by path in a process where ``flashinfer`` cannot be imported
    reads the manifest (the prebuilt route imports nothing from FlashInfer)."""
    _manifest_or_skip()
    code = (
        "import importlib.util, sys\n"
        "sys.modules['flashinfer'] = None\n"
        f"spec = importlib.util.spec_from_file_location('vendored_cake_build', {str(PACKAGE_DIR / 'cake_build.py')!r})\n"
        "b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)\n"
        "m = b.manifest(); assert m['schema'] == 'cake-build/1'\n"
        "print('ok', m['package'], len(m['libraries']))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr
    records = sum(len(rec["arches"]) for rec in cake_jit.MODULES.values())
    assert result.stdout.strip() == f"ok {PACKAGE_IMPORT} {records}"


def test_the_loader_and_header_are_the_pinned_bytes_of_every_cake_package() -> None:
    _manifest_or_skip()
    # Another package's copy, read from the repository (not imported: the vendored gate blocks flashinfer): the
    # KDA det backend's, wherever the KDA rename (#8928) has put its directory.
    experimental = Path(__file__).resolve().parents[2] / "flashinfer/experimental"
    other = next(
        path
        for path in (
            experimental / "cake" / "cake_kda_det_train",
            experimental / "cake_kda_det_train",
        )
        if path.is_dir()
    )
    for relative in ("cake_build.py", "csrc/cake_ffi_checks.h"):
        assert (PACKAGE_DIR / relative).read_bytes() == (
            other / relative
        ).read_bytes(), relative


def test_the_prebuilt_library_is_preferred() -> None:
    """A builder's ``aot/<name>.so`` is served before the JIT path; without one the JIT spec names the entry."""
    _manifest_or_skip()
    for name, rec in cake_jit.MODULES.items():
        for arch in rec["arches"]:
            library = cake_backend.library_name(name, arch)
            assert (
                cake_build.prebuilt_path(library)
                == PACKAGE_DIR / "aot" / f"{library}.so"
            )
            assert (
                cake_backend.is_prebuilt(name, arch)
                == cake_build.prebuilt_path(library).is_file()
            )


def test_the_package_reads_no_environment_variable() -> None:
    offenders = []
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Attribute) and node.attr in ("environ", "getenv"):
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, offenders


class _ExactTiles:
    """The exact dS^T tiles of one head pair: ``count(n, b)`` 64-row Q tiles of segment ``b`` have a row that sees a
    key of the 128-key block ``n`` (none when the segment has no such block), counted from the rows' last visible key
    (bottom-right causal: row ``i`` sees the keys ``j <= i + k_len - q_len``)."""

    def __init__(self, q_lens, k_lens) -> None:
        self.k_lens = list(k_lens)
        rows, keys = cake_jit.BWD_TILE_ROWS, cake_jit.BWD_TILE_KEYS
        self.keys = keys
        # Per segment, ascending: the last key each Q tile's last row sees.
        self.reach = [
            [min(rows * m + rows - 1, q - 1) + k - q for m in range(-(-q // rows))]
            for q, k in zip(q_lens, k_lens, strict=True)
        ]
        self._rows: dict[int, list[int]] = {}

    def row(self, n: int) -> list[int]:
        """``[count(n, b) for b]``."""
        found = self._rows.get(n)
        if found is None:
            found = [
                0
                if self.keys * n >= k
                else len(reach) - bisect.bisect_left(reach, self.keys * n)
                for k, reach in zip(self.k_lens, self.reach, strict=True)
            ]
            self._rows[n] = found
        return found


def _ranges(stop: int, step: int) -> list[tuple[int, int]]:
    """The backend's chunk ranges of one axis: ``[start, min(start + step, stop))`` for ``start`` in
    ``range(0, stop, step)``."""
    return [(start, min(start + step, stop)) for start in range(0, stop, step)]


def test_chunk_plan_and_grids_cover_every_tile() -> None:
    """The workspace chunks of random layouts and budgets: the KV-head ranges and the key-block ranges each tile
    their axis exactly once (so every (KV head, key block) is in exactly one chunk), each chunk's exact dS^T tiles
    of one head pair fit its ``tile_stride``, the dS^T workspace fits the budget unless at the minimum chunk (one KV
    head, two key blocks), and the grids of every chunk's launches; the forward, preprocess and fold grids against
    the exact counts."""
    rng = random.Random(7073)
    sm_count = 148
    for trial in range(200):
        n = rng.randint(1, 24)
        q_lens = [
            rng.choice((0, 1, rng.randint(1, 600), rng.randint(1, 5000)))
            for _ in range(n)
        ]
        k_lens = [rng.choice((0, q, q + rng.randint(0, 12000))) for q in q_lens]
        h_q, h_kv = rng.choice(((24, 4), (8, 4), (16, 2), (32, 8)))
        pairs = h_q // h_kv // 2
        t_q, t_k = sum(q_lens), sum(k_lens)
        dims = dict(
            total_q=t_q,
            total_k=t_k,
            num_heads_q=h_q,
            num_heads_kv=h_kv,
            num_segments=n,
            sm_count=sm_count,
        )
        exact = _ExactTiles(q_lens, k_lens)
        longest = max(-(-k // cake_jit.BWD_TILE_KEYS) for k in k_lens)
        per_block = cake_jit.tiles_per_key_block(t_q, n)
        # One head pair's exact tiles per key block (none past the longest segment) and their prefix sums.
        block_tiles = [sum(exact.row(m)) for m in range(longest)]
        assert all(tiles <= per_block for tiles in block_tiles), trial
        q_tile_pairs = sum(-(-q // (2 * cake_jit.BWD_TILE_ROWS)) for q in q_lens)
        for max_k in (None, max(k_lens)):
            budgets = (
                cake_jit.DEFAULT_WORKSPACE_BUDGET,
                rng.randint(1, 64) * cake_jit.DS_TILE_BYTES * pairs * per_block,
                rng.randint(1, 1 << 30),
                1,
            )
            for budget in budgets:
                plan = cake_jit.chunk_plan(
                    total_q=t_q,
                    total_k=t_k,
                    num_heads_q=h_q,
                    num_heads_kv=h_kv,
                    num_segments=n,
                    max_seqlen_k=max_k,
                    budget_bytes=budget,
                )
                label = (trial, max_k, budget, plan)
                blocks = plan["key_blocks"]
                h_chunk, n_chunk = plan["h_chunk"], plan["n_chunk"]
                assert blocks % 2 == 0 and blocks >= max(1, longest), label
                assert 1 <= h_chunk <= h_kv and 2 <= n_chunk <= blocks, label
                assert n_chunk % 2 == 0 and (h_chunk == 1 or n_chunk == blocks), label
                assert plan["tile_stride"] == n_chunk * per_block, label
                assert plan["zero_tile"] == h_chunk * pairs * plan["tile_stride"], label
                assert plan["ws_tiles"] == plan["zero_tile"] + 1, label
                assert plan["ws_tiles"] * cake_jit.DS_TILE_BYTES <= budget or (
                    h_chunk,
                    n_chunk,
                ) == (1, 2), label
                head_ranges = _ranges(h_kv, h_chunk)
                block_ranges = _ranges(blocks, n_chunk)
                # Each axis tiled exactly once, in order.
                for ranges, stop in ((head_ranges, h_kv), (block_ranges, blocks)):
                    assert ranges[0][0] == 0 and ranges[-1][1] == stop, label
                    assert all(
                        a[1] == b[0] for a, b in zip(ranges, ranges[1:], strict=False)
                    ), label
                assert all((n1 - n0) % 2 == 0 for n0, n1 in block_ranges), label
                # Every chunk's tiles of one head pair within its stride.
                widest = max(sum(block_tiles[n0:n1]) for n0, n1 in block_ranges)
                assert widest <= plan["tile_stride"], (label, widest)
                for hc, chunk_blocks in {
                    (h1 - h0, n1 - n0)
                    for h0, h1 in head_ranges
                    for n0, n1 in block_ranges
                }:
                    chunk = dict(chunk_blocks=chunk_blocks, chunk_heads=hc)
                    for stage in ("bwd_main_fp32", "bwd_main_bf16"):
                        # A CTA pair per (key block of the chunk, KV head of the chunk, segment).
                        assert cake_jit.grid(stage, **dims, **chunk) == (
                            2 * chunk_blocks,
                            hc,
                            n,
                        ), (label, stage)
                    # The persistent dQ GEMM: its work items bound every Q-tile pair of the chunk's head pairs.
                    gx, gy, gz = cake_jit.grid("bwd_dq", **dims, **chunk)
                    items = (-(-t_q // 128) + n) * hc * pairs
                    assert q_tile_pairs * hc * pairs <= items, label
                    assert (gy, gz) == (1, 1), label
                    assert gx == 2 * max(1, min(sm_count // 2, items)), label
        # The forward is persistent: any pair count >= 1 covers every tile; the bound caps it at one pair per two
        # SMs and at the tile count bound.
        fwd_tiles = sum(-(-q // cake_jit.FWD_TILE_ROWS) for q in q_lens)
        bound = t_q // cake_jit.FWD_TILE_ROWS + n
        assert bound >= fwd_tiles
        assert cake_jit.grid("fwd", **dims) == (
            2 * max(1, min(sm_count // 2, h_q * bound)),
            1,
            1,
        )
        rows = cake_jit.rows_pad(t_q, n)
        assert rows % 128 == 0 and rows >= sum(-(-2 * q // 128) * 128 for q in q_lens)
        assert cake_jit.grid("bwd_pre", **dims) == (rows // 32, h_q // 2, 1)
        assert cake_jit.grid("bwd_plan", **dims) == (1, 1, 1)
        # The fold: one warp per (window row, key head), every one in the grid, no CTA without one.
        gx, gy, gz = cake_jit.grid("bwd_fold", **dims)
        items = t_k * h_kv
        assert (gy, gz) == (1, 1)
        assert (
            gx * cake_jit.FOLD_ITEMS_PER_CTA
            >= items
            > (gx - 1) * cake_jit.FOLD_ITEMS_PER_CTA
        )


def test_the_device_tile_numbering_counts_the_exact_tiles() -> None:
    """The backend's tile numbering (``tile_base``, ``pair_base``; run here on host tensors) is the exclusive prefix
    sum of the exact tile counts in (key block, segment) order and of the segments' Q-tile pairs; ``pair_order`` lists
    the pairs by visible key blocks, most first."""
    rng = random.Random(7074)
    for trial in range(100):
        n = rng.randint(1, 16)
        q_lens = [
            rng.choice((0, 1, rng.randint(1, 600), rng.randint(1, 3000)))
            for _ in range(n)
        ]
        k_lens = [rng.choice((0, q, q + rng.randint(0, 6000))) for q in q_lens]
        blocks = cake_jit.key_blocks(sum(k_lens), max(k_lens))
        exact = _ExactTiles(q_lens, k_lens)
        counts = [c for m in range(blocks + 1) for c in exact.row(m)]
        want_tile = [0, *itertools.accumulate(counts)][:-1]
        pairs = [-(-q // (2 * cake_jit.BWD_TILE_ROWS)) for q in q_lens]
        want_pair = [0, *itertools.accumulate(pairs)]
        cu_q = torch.tensor(_cu(q_lens), dtype=torch.int32)
        cu_k = torch.tensor(_cu(k_lens), dtype=torch.int32)
        tile_base, pair_base, pair_order = cake_backend._tile_plan(
            cu_q, cu_k, blocks, sum(q_lens)
        )
        assert tile_base.dtype == torch.int32 and pair_base.dtype == torch.int32
        assert tile_base.tolist() == want_tile, trial
        assert pair_base.tolist() == want_pair, trial
        # The dQ work order: every pair once, first, by the key blocks its later Q tile sees, most first, ties in pair
        # order; the bound's spare positions after them.
        seen = [
            min(
                max(-(-(64 * min(2 * i + 2, -(-q // 64)) + k - q) // 128), 0),
                -(-k // 128),
            )
            for q, k in zip(q_lens, k_lens, strict=True)
            for i in range(-(-q // 128))
        ]
        assert pair_order.dtype == torch.int32
        assert pair_order.numel() == -(-sum(q_lens) // 128) + len(q_lens), trial
        assert sorted(pair_order.tolist()) == list(range(pair_order.numel())), trial
        want_order = sorted(range(len(seen)), key=lambda i: -seen[i])
        assert pair_order.tolist()[: len(seen)] == want_order, trial


def test_the_host_boundary_plan_counts_the_exact_tiles() -> None:
    """With host boundaries the plan counts each key block's exact tiles (the counts the device numbering holds) and
    cuts the key blocks into the widest even ranges within the budget: the ranges tile the key blocks once, each
    within the stride (the most tiles of one range), no range but the last could take two more key blocks, the
    workspace fits the budget unless a range is the two-block minimum, and the plan launches no more chunks than the
    bound's plan.  Malformed host boundaries are refused."""
    rng = random.Random(7075)
    for trial in range(200):
        n = rng.randint(1, 24)
        q_lens = [
            rng.choice((0, 1, rng.randint(1, 600), rng.randint(1, 5000)))
            for _ in range(n)
        ]
        k_lens = [rng.choice((0, q, q + rng.randint(0, 12000))) for q in q_lens]
        h_q, h_kv = rng.choice(((24, 4), (8, 4), (16, 2), (32, 8)))
        pairs = h_q // h_kv // 2
        t_q, t_k = sum(q_lens), sum(k_lens)
        exact = _ExactTiles(q_lens, k_lens)
        longest = max(k_lens)
        for max_k in (None, longest, longest // 2):
            blocks = cake_jit.key_blocks(
                t_k, longest if max_k is None else min(max_k, longest)
            )
            want = [sum(exact.row(m)) for m in range(blocks)]
            assert cake_jit.exact_tiles(q_lens, k_lens, blocks) == want, trial
            prefix = [0, *itertools.accumulate(want)]
            budgets = (
                cake_jit.DEFAULT_WORKSPACE_BUDGET,
                rng.randint(1, 64) * cake_jit.DS_TILE_BYTES * pairs * max(1, *want),
                rng.randint(1, 1 << 30),
                1,
            )
            for budget in budgets:
                sizes = dict(
                    total_q=t_q,
                    total_k=t_k,
                    num_heads_q=h_q,
                    num_heads_kv=h_kv,
                    num_segments=n,
                    max_seqlen_k=max_k,
                    budget_bytes=budget,
                )
                plan = cake_jit.chunk_plan(**sizes, seqlens=(q_lens, k_lens))
                bound = cake_jit.chunk_plan(**sizes)
                label = (trial, max_k, budget, plan)
                ranges, h_chunk = plan["ranges"], plan["h_chunk"]
                assert plan["key_blocks"] == blocks, label
                assert ranges[0][0] == 0 and ranges[-1][1] == blocks, label
                assert all(
                    a[1] == b[0] for a, b in zip(ranges, ranges[1:], strict=False)
                ), label
                assert all(n0 % 2 == 0 and n1 - n0 >= 2 for n0, n1 in ranges), label
                assert all(n1 % 2 == 0 for n0, n1 in ranges), label
                tiles = [prefix[n1] - prefix[n0] for n0, n1 in ranges]
                assert plan["tile_stride"] == max(1, *tiles), label
                assert plan["n_chunk"] == max(n1 - n0 for n0, n1 in ranges), label
                assert plan["zero_tile"] == h_chunk * pairs * plan["tile_stride"], label
                assert plan["ws_tiles"] == plan["zero_tile"] + 1, label
                assert 1 <= h_chunk <= h_kv, label
                assert h_chunk == 1 or ranges == [(0, blocks)], label
                limit = max(0, budget // cake_jit.DS_TILE_BYTES - 1) // pairs
                for n0, n1 in ranges[:-1]:
                    # Greedy: the next two key blocks would pass the limit.
                    assert prefix[n1 + 2] - prefix[n0] > limit, label
                if plan["ws_tiles"] * cake_jit.DS_TILE_BYTES > budget:
                    assert h_chunk == 1, label
                    assert plan["tile_stride"] == 1 or any(
                        n1 - n0 == 2 and t == plan["tile_stride"]
                        for (n0, n1), t in zip(ranges, tiles, strict=True)
                    ), label
                launches = -(-h_kv // h_chunk) * len(ranges)
                assert launches <= -(-h_kv // bound["h_chunk"]) * len(
                    bound["ranges"]
                ), label
                assert bound["ranges"] == _ranges(
                    bound["key_blocks"], bound["n_chunk"]
                ), label
    for cu_q, cu_k in (
        ([0, 5], [0, 7]),  # the last boundary is not T_q
        ([1, 6], [0, 8]),  # the first boundary is not 0
        ([0, 6, 6], [0, 8, 8]),  # one boundary too many
    ):
        with pytest.raises(ValueError):
            cake_jit.host_seqlens(cu_q, cu_k, 6, 8, 1)
    with pytest.raises(ValueError, match="non-decreasing"):
        cake_jit.host_seqlens([0, 3, 6], [0, 9, 8], 6, 8, 2)
    with pytest.raises(ValueError):
        cake_backend._host_lists([0, 6], None)
    with pytest.raises(ValueError):
        cake_backend._host_lists(torch.tensor([0, 6], device="meta"), [0, 8])
    assert cake_backend._host_lists(torch.tensor([0, 6]), (0, 8)) == ([0, 6], [0, 8])


def test_scratch_layout_is_aligned_and_sized() -> None:
    for total_q, heads, segs in ((1, 24, 1), (30193, 24, 12), (77, 8, 3)):
        rows = cake_jit.rows_pad(total_q, segs)
        layout = cake_jit.scratch_layout(total_q, heads, segs)
        assert layout["dq_accum"][1] == (heads // 2, rows, D)
        assert layout["lse_log2"][1] == layout["dpsum"][1] == (heads // 2, rows)
        for offset, _shape in layout.values():
            assert offset % cake_jit.SCRATCH_ALIGN == 0
        floats = (heads // 2) * rows * (D + 2)
        scratch = cake_backend.cake_gqa_nondet_train_workspace_bytes(
            total_q, heads, segs
        )
        assert 4 * floats <= scratch <= 4 * floats + 3 * 256
        # With the key sizes: plus the dS^T chunk of the budget; with a row map: plus the two FP32 windows.
        for total_k, max_k, budget in (
            (4 * total_q, None, cake_jit.DEFAULT_WORKSPACE_BUDGET),
            (4 * total_q, total_q, 1 << 26),
        ):
            plan = cake_jit.chunk_plan(
                total_q=total_q,
                total_k=total_k,
                num_heads_q=heads,
                num_heads_kv=4,
                num_segments=segs,
                max_seqlen_k=max_k,
                budget_bytes=budget,
            )
            kwargs = dict(
                total_k=total_k, num_heads_kv=4, max_seqlen_k=max_k, budget_bytes=budget
            )
            ds = plan["ws_tiles"] * cake_jit.DS_TILE_BYTES
            assert (
                cake_backend.cake_gqa_nondet_train_workspace_bytes(
                    total_q, heads, segs, **kwargs
                )
                == scratch + ds
            )
            assert (
                cake_backend.cake_gqa_nondet_train_workspace_bytes(
                    total_q, heads, segs, row_map=True, **kwargs
                )
                == scratch + ds + 2 * 4 * total_k * 4 * D
            )


# ----------------------------------------------------------------------------------------------------------------
# The FP64 reference


def _inputs(q_lens, k_lens, h_q, h_kv, device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    t_q, t_k = sum(q_lens), sum(k_lens)

    def randn(*shape):
        return torch.randn(*shape, generator=g, device=device, dtype=torch.float32).to(
            torch.bfloat16
        )

    return {
        "q": randn(t_q, h_q, D),
        "k": randn(t_k, h_kv, D),
        "v": randn(t_k, h_kv, D),
        "dout": randn(t_q, h_q, D),
        "cu_q": torch.tensor(_cu(q_lens), dtype=torch.int32, device=device),
        "cu_k": torch.tensor(_cu(k_lens), dtype=torch.int32, device=device),
        "cq": _cu(q_lens),
        "ck": _cu(k_lens),
    }


def _reference(x, scale=SCALE):
    """FP64 out, lse and gradients of the same BF16 inputs (bottom-right causal per segment), computed on the CPU and
    returned on the inputs' device: on a GPU, PyTorch may run some FP64 ops of this graph (the outer products of
    one-row segments) as Triton kernels, which need a ptxas that knows the GPU."""
    device = x["q"].device
    q = x["q"].cpu().double().requires_grad_()
    k = x["k"].cpu().double().requires_grad_()
    v = x["v"].cpu().double().requires_grad_()
    group = q.shape[1] // k.shape[1]
    outs, lses = [], []
    for b in range(len(x["cq"]) - 1):
        q0, q1, k0, k1 = x["cq"][b], x["cq"][b + 1], x["ck"][b], x["ck"][b + 1]
        ql, kl = q1 - q0, k1 - k0
        qs = q[q0:q1]
        ks = k[k0:k1].repeat_interleave(group, dim=1)
        vs = v[k0:k1].repeat_interleave(group, dim=1)
        s = torch.einsum("qhd,khd->hqk", qs, ks) * scale
        seen = torch.arange(kl, device=q.device)[None, :] <= torch.arange(
            ql, device=q.device
        )[:, None] + (kl - ql)
        has_key = seen.any(dim=-1)
        # Rows without a key take finite dummy scores (no NaN reaches the gradients) and a zero P.
        s = s.masked_fill(~seen, -math.inf).masked_fill(~has_key[None, :, None], 0.0)
        lse = torch.logsumexp(s, dim=-1)
        p = torch.exp(s - lse[..., None]) * seen
        outs.append(torch.einsum("hqk,khd->qhd", p, vs))
        lses.append(lse.detach().masked_fill(~has_key[None, :], -math.inf))
    out = torch.cat(outs)
    lse = torch.cat(lses, dim=1)
    out.backward(x["dout"].cpu().double())
    ref = {
        "out": out.detach(),
        "lse": lse.detach(),
        "dq": q.grad,
        "dk": k.grad,
        "dv": v.grad,
    }
    return {name: t.to(device) for name, t in ref.items()}


def _assert_dq_close(got: torch.Tensor, ref: torch.Tensor) -> None:
    """Two dQ of one input under two workspace budgets (see DQ_RTOL / DQ_ATOL_FRACTION)."""
    floor = DQ_ATOL_FRACTION * float(ref.float().abs().max()) if ref.numel() else 0.0
    torch.testing.assert_close(got, ref, rtol=DQ_RTOL, atol=floor)


def _rel_l2(got: torch.Tensor, ref: torch.Tensor) -> float:
    got, ref = got.double(), ref.double()
    finite = torch.isfinite(ref)
    num = torch.linalg.vector_norm((got - ref)[finite])
    den = torch.linalg.vector_norm(ref[finite])
    return float(num / den) if den > 0 else float(num)


def _no_key_rows(x):
    """Mask ``[T_q]`` of the query rows that see no key (``i + k_len - q_len < 0``)."""
    rows = []
    for b in range(len(x["cq"]) - 1):
        ql, kl = x["cq"][b + 1] - x["cq"][b], x["ck"][b + 1] - x["ck"][b]
        rows += [i + kl - ql < 0 for i in range(ql)]
    return torch.tensor(rows, dtype=torch.bool, device=x["q"].device)


def _longest_k(x) -> int:
    ck = x["ck"]
    return max(ck[i + 1] - ck[i] for i in range(len(ck) - 1))


# ``_run``'s default ``max_seqlen_k``: none for the BF16 form (the T_k bound), the exact longest key segment else.
_BY_FORM = object()


def _run(
    x,
    form,
    *,
    workspace=None,
    dst_init=None,
    row_map=None,
    max_seqlen_k=_BY_FORM,
    host=False,
):
    """Forward and backward of the package: ``{"out", "lse", "dq", "dk", "dv"}`` (dk / dv as the form returns them;
    the row-map form adds into copies of ``dst_init``; ``host``: the backward gets the host boundaries too)."""
    out, lse = cake_backend.cake_gqa_nondet_train_forward(
        x["q"],
        x["k"],
        x["v"],
        x["cu_q"],
        x["cu_k"],
        softmax_scale=SCALE,
        workspace=workspace,
    )
    if max_seqlen_k is _BY_FORM:
        max_seqlen_k = None if form == "bf16" else _longest_k(x)
    kwargs = {
        "softmax_scale": SCALE,
        "workspace": workspace,
        "max_seqlen_k": max_seqlen_k,
    }
    if host:
        kwargs.update(cu_seqlens_q_host=x["cq"], cu_seqlens_k_host=x["ck"])
    if form == "rowmap":
        kwargs.update(
            dk_acc=dst_init[0].clone(), dv_acc=dst_init[1].clone(), row_map=row_map
        )
    elif form == "fp32":
        kwargs["dkv_dtype"] = torch.float32
    dq, dk, dv = cake_backend.cake_gqa_nondet_train_backward(
        x["q"], x["k"], x["v"], out, lse, x["dout"], x["cu_q"], x["cu_k"], **kwargs
    )
    return {"out": out, "lse": lse, "dq": dq, "dk": dk, "dv": dv}


def _window_row_map(x):
    """A row map sending every window row to a destination row, the windows of a segment pair sharing rows
    (repeated destination rows across windows), plus untouched destination rows."""
    ck = x["ck"]
    rows = []
    for b in range(len(ck) - 1):
        start = (
            0 if b % 2 else 3
        )  # odd windows restart at row 0: they overlap the even windows' rows
        rows += [start + j for j in range(ck[b + 1] - ck[b])]
    n_dst = (max(rows) + 1 if rows else 0) + 5
    return torch.tensor(rows, dtype=torch.int32, device=x["q"].device), n_dst


def _scattered_row_map(x, seed=0):
    """A row map sending the window rows to destination rows in no particular order, most destination rows from
    several window rows, plus untouched destination rows."""
    t_k = x["ck"][-1]
    n_dst = t_k // 3 + 5
    rng = random.Random(seed)
    rows = [rng.randrange(n_dst - 5) for _ in range(t_k)]
    return torch.tensor(rows, dtype=torch.int32, device=x["q"].device), n_dst


def _row_maps(x):
    return {"windows": _window_row_map(x), "scattered": _scattered_row_map(x)}


def _destinations(n_dst, h_kv, device, seed=1):
    """Two non-zero FP32 destinations ``[n_dst, H_kv, 256]``."""
    g = torch.Generator(device=device).manual_seed(seed)
    return tuple(
        torch.randn(n_dst, h_kv, D, generator=g, device=device) for _ in range(2)
    )


def _ordered_fold(dst: torch.Tensor, window: torch.Tensor, row_map: torch.Tensor):
    """FP32 adds of the window rows into a copy of ``dst`` in ascending window order, on the device: round ``r``
    adds every destination row's ``r``-th window row (distinct destination rows per round), so a destination row
    becomes ``((dst + w_j1) + w_j2) + ...`` over its window rows ``j1 < j2 < ...``."""
    out = dst.clone()
    count: dict[int, int] = {}
    rounds: list[list[int]] = []
    for j, row in enumerate(row_map.tolist()):
        r = count.get(row, 0)
        count[row] = r + 1
        if r == len(rounds):
            rounds.append([])
        rounds[r].append(j)
    rows = row_map.long()
    for js in rounds:
        index = torch.tensor(js, dtype=torch.long, device=dst.device)
        target = rows[index]
        out[target] = out[target] + window[index]
    return out


def _assert_matches_reference(label, x, form, got, ref, dst_init=None, row_map=None):
    """``got`` within the accuracy budget of the FP64 reference ``ref``; rows that see no key and keys no query
    sees exactly; untouched row-map destination rows unchanged."""
    got, ref = dict(got), dict(ref)
    device = x["q"].device
    if form == "rowmap":
        rm = row_map.long()
        for t, init in (("dk", dst_init[0]), ("dv", dst_init[1])):
            expected = init.double().index_add(0, rm, ref[t])
            touched = torch.zeros(init.shape[0], dtype=torch.bool, device=device)
            touched[rm] = True
            _assert_same_bits(
                got[t][~touched], init[~touched], f"{label}: untouched {t} rows"
            )
            ref[t] = expected - init.double()
            got[t] = got[t].double() - init.double()
    no_key = _no_key_rows(x)
    errors = {}
    for t in OUTPUTS:
        assert torch.isfinite(got[t][torch.isfinite(ref[t])]).all(), (
            f"{label}: non-finite {t}"
        )
        errors[t] = _rel_l2(got[t], ref[t])
    print(f"REL_L2 {label} " + " ".join(f"{t}={e:.3e}" for t, e in errors.items()))
    for t, e in errors.items():
        assert e <= BUDGET * FA4_REL_L2[t], (
            f"{label}: {t} rel-L2 {e:.3e} > {BUDGET} x {FA4_REL_L2[t]:.1e}"
        )
    # Rows that see no key: out 0, lse -inf, dq 0, exactly.
    assert torch.equal(got["out"][no_key], torch.zeros_like(got["out"][no_key]))
    assert bool((got["lse"][:, no_key] == -math.inf).all())
    assert bool((got["lse"][:, ~no_key] > -math.inf).all())
    assert torch.equal(got["dq"][no_key], torch.zeros_like(got["dq"][no_key]))
    if form != "rowmap":
        # Keys no query sees (a segment without queries) get exactly zero gradients.
        unseen = ref["dk"].abs().amax(dim=(1, 2)) == 0
        unseen &= ref["dv"].abs().amax(dim=(1, 2)) == 0
        assert torch.equal(got["dk"][unseen], torch.zeros_like(got["dk"][unseen]))
        assert torch.equal(got["dv"][unseen], torch.zeros_like(got["dv"][unseen]))


# ----------------------------------------------------------------------------------------------------------------
# On SM100 / SM103 / SM107


def test_the_plan_kernel_writes_the_pytorch_tables(device) -> None:
    """``bwd_plan`` (one CTA) writes the integers of the PyTorch plan: random layouts with empty and one-row segments,
    more segments than the CTA has threads, the pair bound at the kernel's limit, and the PyTorch fallback past it."""
    rng = random.Random(7076)
    layouts = []
    for _ in range(40):
        n = rng.randint(1, 24)
        q_lens = [
            rng.choice((0, 1, rng.randint(1, 600), rng.randint(1, 5000)))
            for _ in range(n)
        ]
        k_lens = [rng.choice((0, q, q + rng.randint(0, 9000))) for q in q_lens]
        layouts.append((q_lens, k_lens))
    many = [rng.randint(0, 300) for _ in range(1100)]
    layouts.append((many, [q + rng.randint(0, 200) for q in many]))
    for extra in (0, 1):
        q = 2 * cake_jit.BWD_TILE_ROWS * (cake_jit.PLAN_MAX_PAIRS - 1 + extra)
        assert cake_jit.pair_bound(q, 1) == cake_jit.PLAN_MAX_PAIRS + extra
        layouts.append(([q], [q]))
    for trial, (q_lens, k_lens) in enumerate(layouts):
        total_q, total_k = sum(q_lens), sum(k_lens)
        blocks = cake_jit.key_blocks(total_k, max(k_lens))
        dims = {
            "total_q": total_q,
            "total_k": total_k,
            "num_heads_q": 8,
            "num_heads_kv": 2,
            "num_segments": len(q_lens),
        }
        cu_q = torch.tensor(_cu(q_lens), dtype=torch.int32)
        cu_k = torch.tensor(_cu(k_lens), dtype=torch.int32)
        got = cake_backend._plan_tables(
            cu_q.to(device), cu_k.to(device), blocks, total_q, dims, device.index
        )
        want = cake_backend._tile_plan(cu_q, cu_k, blocks, total_q)
        for name, g, w in zip(
            ("tile_base", "pair_base", "pair_order"), got, want, strict=True
        ):
            assert g.dtype == torch.int32 and g.device == device, (trial, name)
            assert torch.equal(g.cpu(), w), (trial, name)


@pytest.mark.parametrize("form", DKV_FORMS)
@pytest.mark.parametrize("layout", LAYOUTS, ids=LAYOUT_IDS)
def test_matches_the_fp64_reference(layout, form, device) -> None:
    name, q_lens, k_lens, h_q, h_kv = layout
    x = _inputs(q_lens, k_lens, h_q, h_kv, device)
    ref = _reference(x)
    dst_init = row_map = None
    if form == "rowmap":
        row_map, n_dst = _window_row_map(x)
        dst_init = _destinations(n_dst, h_kv, device)
    got = _run(x, form, dst_init=dst_init, row_map=row_map)
    torch.cuda.synchronize()
    _assert_matches_reference(f"{name} {form}", x, form, got, ref, dst_init, row_map)


@pytest.mark.parametrize(
    "h_q, h_kv", ((24, 4), (4, 2), (8, 1)), ids=("group_6", "group_2", "group_8")
)
def test_the_forward_of_every_group_matches_the_reference(h_q, h_kv, device) -> None:
    """Forward tiles of two query heads of one key head match the FP64 reference for several groups and repeat bit
    for bit, also into a caller's ``out`` 16 bytes past a 32-byte boundary."""
    q_lens, k_lens = (77, 300, 1, 129, 0, 40), (77, 700, 1000, 129, 50, 0)
    x = _inputs(q_lens, k_lens, h_q, h_kv, device)
    ref = _reference(x)
    args = (x["q"], x["k"], x["v"], x["cu_q"], x["cu_k"])
    out, lse = cake_backend.cake_gqa_nondet_train_forward(*args)
    again, lse_again = cake_backend.cake_gqa_nondet_train_forward(*args)
    buf = torch.empty(out.numel() + 8, dtype=torch.bfloat16, device=device)
    off = buf[8:].view(out.shape)
    assert off.data_ptr() % 32 == 16
    off_lse = torch.empty_like(lse)
    cake_backend.cake_gqa_nondet_train_forward(*args, out=off, lse=off_lse)
    torch.cuda.synchronize()
    for t, got in (("out", out), ("lse", lse)):
        e = _rel_l2(got, ref[t])
        assert e <= BUDGET * FA4_REL_L2[t], f"{t} rel-L2 {e:.3e}"
    no_key = _no_key_rows(x)
    assert torch.equal(out[no_key], torch.zeros_like(out[no_key]))
    assert bool((lse[:, no_key] == -math.inf).all())
    _assert_same_bits(again, out, "out")
    _assert_same_bits(lse_again, lse, "lse")
    _assert_same_bits(off, out, "out off a 32-byte boundary")
    _assert_same_bits(off_lse, lse, "lse beside it")


@pytest.mark.parametrize("layout", LAYOUTS, ids=LAYOUT_IDS)
def test_a_dq_off_a_32_byte_boundary_gets_the_bits_of_an_allocated_one(
    layout, device
) -> None:
    """The dQ GEMM stores dQ in 32-byte pieces: a caller's ``dq`` on a 32-byte boundary, 16 bytes past one and 2
    bytes past one, each full of NaN before the call, gets the dQ, dK and dV bits of an allocated ``dq``."""
    name, q_lens, k_lens, h_q, h_kv = layout
    x = _inputs(q_lens, k_lens, h_q, h_kv, device)
    ws = cake_backend.CakeGQANondetWorkspace(device)
    out, lse = cake_backend.cake_gqa_nondet_train_forward(
        x["q"], x["k"], x["v"], x["cu_q"], x["cu_k"], softmax_scale=SCALE, workspace=ws
    )
    args = (x["q"], x["k"], x["v"], out, lse, x["dout"], x["cu_q"], x["cu_k"])
    names = ("dq", "dk", "dv")

    def backward(dq=None):
        grads = cake_backend.cake_gqa_nondet_train_backward(
            *args, softmax_scale=SCALE, workspace=ws, dq=dq
        )
        return dict(zip(names, grads, strict=True))

    want = backward()
    for skip, offset in ((0, 0), (8, 16), (1, 2)):
        # Caching-allocator blocks are 512-byte aligned, so ``buf[skip:]`` is ``offset`` bytes past a 32-byte boundary.
        buf = torch.empty(x["q"].numel() + skip, dtype=torch.bfloat16, device=device)
        buf.view(torch.int16).fill_(-1)  # every bit set: NaN
        dq = buf[skip:].view(x["q"].shape)
        assert dq.data_ptr() % 32 == offset
        got = backward(dq)
        torch.cuda.synchronize()
        assert got["dq"] is dq
        label = f"{name}: dq {offset} bytes past a 32-byte boundary"
        _assert_outputs_same_bits(got, want, label, names)


@pytest.mark.parametrize("layout", LAYOUTS, ids=LAYOUT_IDS)
def test_every_output_repeats_bit_for_bit(layout, device) -> None:
    """``REPEATS`` runs of every form, through the per-stream default workspace, a reused workspace and fresh ones,
    give the same bits for out, lse, dq and dK / dV (BF16, FP32 window, row-map adds into non-zero destinations
    with repeated destination rows, in window order and scattered)."""
    name, q_lens, k_lens, h_q, h_kv = layout
    x = _inputs(q_lens, k_lens, h_q, h_kv, device)
    shared = cake_backend.CakeGQANondetWorkspace(device)
    cases = [(form, None, None) for form in ("bf16", "fp32")]
    for map_name, (row_map, n_dst) in _row_maps(x).items():
        assert len(set(row_map.tolist())) < row_map.numel(), (name, map_name)
        cases.append(
            (f"rowmap/{map_name}", row_map, _destinations(n_dst, h_kv, device))
        )
    for label, row_map, dst_init in cases:
        form = label.split("/")[0]
        runs = []
        for i in range(REPEATS):
            workspace = (None, shared, cake_backend.CakeGQANondetWorkspace(device))[
                i % 3
            ]
            runs.append(
                _run(x, form, workspace=workspace, dst_init=dst_init, row_map=row_map)
            )
        torch.cuda.synchronize()
        for i, got in enumerate(runs[1:], 1):
            _assert_outputs_same_bits(got, runs[0], f"{name} {label} run {i}")


def _timed(call, *, sides=()):
    """Milliseconds and result of ``call`` on the current stream, launched behind a short sleep kernel while a long
    sleep kernel is resident on each stream of ``sides``."""
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    torch.cuda.synchronize()
    for side in sides:
        with torch.cuda.stream(side):
            torch.cuda._sleep(SIDE_SLEEP_CYCLES)
    torch.cuda._sleep(PRE_SLEEP_CYCLES)
    start.record()
    result = call()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end), result


@pytest.mark.parametrize("form", DKV_FORMS)
def test_the_bits_do_not_depend_on_resident_kernels(form, device) -> None:
    """Kernels resident on other streams change which SMs the launches get, not the bits."""
    x = _inputs((3000, 1100, 77), (3000, 17000, 300), 24, 4, device, seed=5)
    row_map, n_dst = _scattered_row_map(x)
    dst_init = _destinations(n_dst, 4, device)
    ws = cake_backend.CakeGQANondetWorkspace(device)

    def call():
        return _run(x, form, workspace=ws, dst_init=dst_init, row_map=row_map)

    reference = _timed(call)[1]
    sides = [torch.cuda.Stream(device) for _ in range(8)]
    for count in (1, 8, 1, 8):
        _ms, got = _timed(call, sides=sides[:count])
        _assert_outputs_same_bits(
            got, reference, f"{form} beside {count} resident kernel(s)"
        )


BUDGET_KINDS = ("kv_heads", "key_blocks", "minimum")


def _budget_plan(kind, x, h_q, h_kv):
    """``(budget_bytes, expected (h_chunk, n_chunk), default plan)`` of a budget kind for the exact ``max_seqlen_k``:
    half the KV heads per chunk, about a third of the key blocks per chunk (one KV head), or the minimum chunk."""
    t_q, t_k, segments = x["cq"][-1], x["ck"][-1], len(x["cq"]) - 1
    sizes = dict(
        total_q=t_q,
        total_k=t_k,
        num_heads_q=h_q,
        num_heads_kv=h_kv,
        num_segments=segments,
        max_seqlen_k=_longest_k(x),
    )
    default = cake_jit.chunk_plan(
        **sizes, budget_bytes=cake_jit.DEFAULT_WORKSPACE_BUDGET
    )
    blocks = default["key_blocks"]
    per_pair_block = (h_q // h_kv // 2) * cake_jit.tiles_per_key_block(t_q, segments)
    if kind == "minimum":
        return 1, (1, 2), default
    if kind == "kv_heads":
        tiles, want = (h_kv // 2) * blocks * per_pair_block, (h_kv // 2, blocks)
    else:
        n = max(2, (blocks // 3) & ~1)
        tiles, want = n * per_pair_block, (1, min(n, blocks))
    return (tiles + 1) * cake_jit.DS_TILE_BYTES, want, default


@pytest.mark.parametrize("kind", BUDGET_KINDS)
@pytest.mark.parametrize("layout", LAYOUTS, ids=LAYOUT_IDS)
def test_a_small_budget_matches_the_reference_and_repeats(layout, kind, device) -> None:
    """A budget that splits the backward into KV-head chunks, key-block chunks or the minimum chunks: every form
    within the accuracy budget of the FP64 reference, bitwise reproducible, and out, lse, dK and dV with the bits
    of the unchunked call (only the dQ rounding may move)."""
    name, q_lens, k_lens, h_q, h_kv = layout
    x = _inputs(q_lens, k_lens, h_q, h_kv, device)
    budget, (h_chunk, n_chunk), default = _budget_plan(kind, x, h_q, h_kv)
    assert (default["h_chunk"], default["n_chunk"]) == (h_kv, default["key_blocks"])
    plan = cake_jit.chunk_plan(
        total_q=x["cq"][-1],
        total_k=x["ck"][-1],
        num_heads_q=h_q,
        num_heads_kv=h_kv,
        num_segments=len(q_lens),
        max_seqlen_k=_longest_k(x),
        budget_bytes=budget,
    )
    assert (plan["h_chunk"], min(plan["n_chunk"], plan["key_blocks"])) == (
        h_chunk,
        min(n_chunk, plan["key_blocks"]),
    ), (kind, plan)
    ref = _reference(x)
    row_map, n_dst = _window_row_map(x)
    dst_init = _destinations(n_dst, h_kv, device)
    whole = cake_backend.CakeGQANondetWorkspace(device)
    small = cake_backend.CakeGQANondetWorkspace(device, budget_bytes=budget)
    assert small.budget_bytes == budget
    max_k = _longest_k(x)
    for form in DKV_FORMS:
        label = f"{name} {form} budget={kind}"
        runs = [
            _run(
                x,
                form,
                workspace=ws,
                dst_init=dst_init,
                row_map=row_map,
                max_seqlen_k=max_k,
            )
            for ws in (whole, small, small)
        ]
        torch.cuda.synchronize()
        base, got, again = runs
        _assert_outputs_same_bits(again, got, f"{label}: repeat")
        _assert_outputs_same_bits(
            got, base, f"{label}: vs one chunk", ("out", "lse", "dk", "dv")
        )
        _assert_dq_close(got["dq"], base["dq"])
        _assert_matches_reference(label, x, form, got, ref, dst_init, row_map)


HOST_BUDGETS = ("default", "key_blocks", "minimum")


def _host_budget(kind, x, h_q, h_kv) -> int | None:
    """``None`` (the default budget), a budget the exact plan meets with about three key-block ranges, or 1 (the
    minimum)."""
    if kind == "default":
        return None
    if kind == "minimum":
        return 1
    q_lens = [b - a for a, b in zip(x["cq"], x["cq"][1:], strict=False)]
    k_lens = [b - a for a, b in zip(x["ck"], x["ck"][1:], strict=False)]
    blocks = cake_jit.key_blocks(x["ck"][-1], max(k_lens))
    total = sum(cake_jit.exact_tiles(q_lens, k_lens, blocks))
    return ((h_q // h_kv // 2) * max(1, total // 3) + 1) * cake_jit.DS_TILE_BYTES


@pytest.mark.parametrize("kind", HOST_BUDGETS)
@pytest.mark.parametrize("layout", LAYOUTS, ids=LAYOUT_IDS)
def test_host_boundaries_match_the_reference_and_repeat(layout, kind, device) -> None:
    """Host boundaries (the exact plan) at the default budget, at a budget of about three key-block ranges and at
    the minimum: every form within the accuracy budget of the FP64 reference, bitwise reproducible, out, lse, dK and
    dV with the bits of the call without them (only the dQ rounding may move), and the workspace bytes of the exact
    plan."""
    _check_host_boundaries(layout, kind, device)


@pytest.mark.parametrize("kind", ("default", "minimum"))
@pytest.mark.parametrize("layout", LAYOUTS, ids=LAYOUT_IDS)
def test_host_boundaries_without_the_plan_kernel_match_the_reference_and_repeat(
    layout, kind, device, monkeypatch
) -> None:
    """The same past ``PLAN_MAX_PAIRS`` pair positions (0 here, so every call is past it): the PyTorch ops of
    ``_tile_plan`` build the tile tables, both host-boundary checks run on the current stream and the row-map fold
    order on the first side stream."""
    monkeypatch.setattr(cake_jit, "PLAN_MAX_PAIRS", 0)
    _check_host_boundaries(layout, kind, device)


def _check_host_boundaries(layout, kind, device) -> None:
    """The checks of :func:`test_host_boundaries_match_the_reference_and_repeat` on ``layout`` at budget ``kind``."""
    name, q_lens, k_lens, h_q, h_kv = layout
    x = _inputs(q_lens, k_lens, h_q, h_kv, device)
    budget = _host_budget(kind, x, h_q, h_kv)
    plan = cake_jit.chunk_plan(
        total_q=x["cq"][-1],
        total_k=x["ck"][-1],
        num_heads_q=h_q,
        num_heads_kv=h_kv,
        num_segments=len(q_lens),
        max_seqlen_k=None,
        budget_bytes=budget or cake_jit.DEFAULT_WORKSPACE_BUDGET,
        seqlens=(q_lens, k_lens),
    )
    if kind == "key_blocks" and plan["key_blocks"] >= 6:
        assert len(plan["ranges"]) >= 2 and plan["h_chunk"] == 1, plan
    if x["cq"][-1] and x["ck"][-1]:
        want = cake_jit.scratch_bytes(x["cq"][-1], h_q, len(q_lens)) + (
            plan["ws_tiles"] * cake_jit.DS_TILE_BYTES
        )
        assert (
            pkg.cake_gqa_nondet_train_workspace_bytes(
                x["cq"][-1],
                h_q,
                len(q_lens),
                total_k=x["ck"][-1],
                num_heads_kv=h_kv,
                budget_bytes=budget,
                cu_seqlens_q_host=x["cq"],
                cu_seqlens_k_host=torch.tensor(x["ck"]),
            )
            == want
        )
    ref = _reference(x)
    row_map, n_dst = _window_row_map(x)
    dst_init = _destinations(n_dst, h_kv, device)
    ws = cake_backend.CakeGQANondetWorkspace(device, budget_bytes=budget)
    for form in DKV_FORMS:
        label = f"{name} {form} host budget={kind}"
        base, got, again = (
            _run(x, form, workspace=ws, dst_init=dst_init, row_map=row_map, host=host)
            for host in (False, True, True)
        )
        torch.cuda.synchronize()
        _assert_outputs_same_bits(again, got, f"{label}: repeat")
        _assert_outputs_same_bits(
            got, base, f"{label}: vs no host boundaries", ("out", "lse", "dk", "dv")
        )
        _assert_dq_close(got["dq"], base["dq"])
        _assert_matches_reference(label, x, form, got, ref, dst_init, row_map)


# Layouts of the unwritten-read test: rows that see no key and a segment without keys, empty and short segments, and
# a key range past 64 key blocks.
UNWRITTEN_READ_LAYOUTS = ("keys_fewer_than_queries", "empty_segments", "over_64_blocks")


@pytest.mark.parametrize("host", (False, True), ids=("device_bounds", "host_bounds"))
@pytest.mark.parametrize("kind", HOST_BUDGETS)
@pytest.mark.parametrize("name", UNWRITTEN_READ_LAYOUTS)
def test_no_kernel_reads_scratch_or_dq_bytes_it_did_not_write(
    name, kind, host, device
) -> None:
    """The workspace scratch (the FP32 dQ accumulator, ``LSE * log2(e)`` and ``dPsum``) and the caller's ``dq``
    filled with 0x00 bytes and then with 0xFF bytes (NaN) before the backward give the same dQ, dK and dV bits, at
    the default budget, at about three key-block ranges and at the minimum chunks, with and without host boundaries:
    no kernel reads a byte of them that the call did not write."""
    _name, q_lens, k_lens, h_q, h_kv = LAYOUTS_BY_NAME[name]
    x = _inputs(q_lens, k_lens, h_q, h_kv, device)
    budget = _host_budget(kind, x, h_q, h_kv)
    ws = cake_backend.CakeGQANondetWorkspace(device, budget_bytes=budget)
    out, lse = cake_backend.cake_gqa_nondet_train_forward(
        x["q"], x["k"], x["v"], x["cu_q"], x["cu_k"], softmax_scale=SCALE, workspace=ws
    )
    args = (x["q"], x["k"], x["v"], out, lse, x["dout"], x["cu_q"], x["cu_k"])
    kwargs = {"softmax_scale": SCALE, "workspace": ws}
    if host:
        kwargs.update(cu_seqlens_q_host=x["cq"], cu_seqlens_k_host=x["ck"])
    key = (x["cq"][-1], h_q, len(q_lens))
    names = ("dq", "dk", "dv")
    runs = {}
    for byte in (0x00, 0xFF):
        # The views the backward gets (one shape, one key): the scratch grows here when it must, not in the call.
        scratch = ws.scratch(*key)
        scratch[0].untyped_storage().fill_(byte)
        dq = torch.empty_like(x["q"])
        dq.untyped_storage().fill_(byte)
        grads = cake_backend.cake_gqa_nondet_train_backward(*args, dq=dq, **kwargs)
        runs[byte] = dict(zip(names, grads, strict=True))
        assert ws.scratch(*key)[0].data_ptr() == scratch[0].data_ptr()
    torch.cuda.synchronize()
    label = f"{name} budget={kind} host={host}: 0xFF vs 0x00"
    _assert_outputs_same_bits(runs[0xFF], runs[0x00], label, names)


def _child_backward(
    cu_q, cu_k, host_q, host_k, budget, plan_kernel=True
) -> subprocess.CompletedProcess:
    """A child process (a device-side assert ends its CUDA context) running the forward, then the backward with the
    host boundaries ``host_q`` / ``host_k`` on device boundaries ``cu_q`` / ``cu_k``; it prints ``forward`` and
    ``backward`` after each synchronizes.  ``plan_kernel=False`` sets ``PLAN_MAX_PAIRS`` to 0 in the child, so the
    PyTorch ops build the tile tables and the checks run on the current stream."""
    t_q, t_k = cu_q[-1], cu_k[-1]
    code = (
        "import importlib, sys\n"
        f"sys.path[:0] = {sys.path!r}\n"
        "import torch\n"
        f"b = importlib.import_module({cake_backend.__name__!r})\n"
        + ("" if plan_kernel else "b.cake_jit.PLAN_MAX_PAIRS = 0\n")
        + "d = torch.device('cuda:0')\n"
        "g = torch.Generator(device=d).manual_seed(0)\n"
        "r = lambda *s: torch.randn(*s, generator=g, device=d).to(torch.bfloat16)\n"
        f"q, k, v, do = r({t_q}, 24, 256), r({t_k}, 4, 256), r({t_k}, 4, 256), r({t_q}, 24, 256)\n"
        f"cq = torch.tensor({list(cu_q)!r}, dtype=torch.int32, device=d)\n"
        f"ck = torch.tensor({list(cu_k)!r}, dtype=torch.int32, device=d)\n"
        f"ws = b.CakeGQANondetWorkspace(d, budget_bytes={budget!r})\n"
        "out, lse = b.cake_gqa_nondet_train_forward(q, k, v, cq, ck, workspace=ws)\n"
        "torch.cuda.synchronize()\n"
        "print('forward', flush=True)\n"
        "b.cake_gqa_nondet_train_backward(q, k, v, out, lse, do, cq, ck, workspace=ws,\n"
        f"    cu_seqlens_q_host={list(host_q)!r}, cu_seqlens_k_host={list(host_k)!r})\n"
        "torch.cuda.synchronize()\n"
        "print('backward', flush=True)\n"
    )
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
    )


def test_host_boundaries_that_disagree_past_the_plan_stop_the_call(device) -> None:
    """Host boundaries that disagree with the device ones: within the plan (the same tiles per key block) the call
    runs and gives out, lse, dK and dV of the device boundaries; a device key segment longer than the plan's key
    blocks, or device tiles that overfill a range, fire the device-side assert before the main kernel writes (each in
    a child process, with the tile tables of the plan kernel and of the PyTorch ops)."""
    h_q, h_kv = 24, 4
    device_lens = ((2990, 10), (2990, 10))
    swapped = ((10, 2990), (10, 2990))  # other boundaries, the same tiles per key block
    # 12 key blocks; the device's 2990-key segment needs 24
    short_keys = ((1500, 1500), (1500, 1500))
    # 24 key blocks, fewer tiles per range than the device's
    fewer_tiles = ((1500, 1500), (2990, 10))
    plan = cake_jit.chunk_plan(
        total_q=3000,
        total_k=3000,
        num_heads_q=h_q,
        num_heads_kv=h_kv,
        num_segments=2,
        max_seqlen_k=None,
        budget_bytes=1,
        seqlens=fewer_tiles,
    )
    assert max(device_lens[1]) <= cake_jit.BWD_TILE_KEYS * plan["key_blocks"]
    counts = cake_jit.exact_tiles(*device_lens, plan["key_blocks"])
    over = [sum(counts[n0:n1]) - plan["tile_stride"] for n0, n1 in plan["ranges"]]
    assert max(over) > 0, over  # the device numbering overfills a range of that plan
    one = cake_jit.chunk_plan(
        total_q=3000,
        total_k=3000,
        num_heads_q=h_q,
        num_heads_kv=h_kv,
        num_segments=2,
        max_seqlen_k=None,
        budget_bytes=cake_jit.DEFAULT_WORKSPACE_BUDGET,
        seqlens=fewer_tiles,
    )
    assert one["ranges"] == [(0, one["key_blocks"])], one["ranges"]
    # The device numbering overfills that one range too.
    assert sum(counts[: one["key_blocks"]]) > one["tile_stride"], one
    x = _inputs(*device_lens, h_q, h_kv, device)
    ws = cake_backend.CakeGQANondetWorkspace(device, budget_bytes=1)
    base = _run(x, "bf16", workspace=ws)
    within = dict(x, cq=_cu(swapped[0]), ck=_cu(swapped[1]))
    got = _run(within, "bf16", workspace=ws, host=True)
    torch.cuda.synchronize()
    _assert_outputs_same_bits(got, base, "within the plan", ("out", "lse", "dk", "dv"))
    _assert_dq_close(got["dq"], base["dq"])
    dev = (_cu(device_lens[0]), _cu(device_lens[1]))
    cases = {
        "keys past the key blocks": (
            *dev,
            _cu(short_keys[0]),
            _cu(short_keys[1]),
            None,
        ),
        # A 1024-key device segment against 544-key host segments (6 key blocks), at the default budget.
        "keys past the key blocks, one chunk": (
            [0, 64, 128],
            [0, 64, 1088],
            [0, 64, 128],
            [0, 544, 1088],
            None,
        ),
        "tiles past a range": (*dev, _cu(fewer_tiles[0]), _cu(fewer_tiles[1]), 1),
        # The same host boundaries at the default budget: one range (the int32 one-subtraction check).
        "tiles past the one range": (
            *dev,
            _cu(fewer_tiles[0]),
            _cu(fewer_tiles[1]),
            None,
        ),
    }
    for (label, args), plan_kernel in itertools.product(cases.items(), (True, False)):
        result = _child_backward(*args, plan_kernel=plan_kernel)
        where = (label, "plan kernel" if plan_kernel else "PyTorch tables")
        assert result.stdout.split() == ["forward"], (
            where,
            result.stdout,
            result.stderr[-2000:],
        )
        assert result.returncode != 0, (where, result.stderr[-2000:])
        assert "device-side assert" in result.stderr or "disagree" in result.stderr, (
            where,
            result.stderr[-2000:],
        )


def _child_check_ranges(tile_base, plan, total_q) -> subprocess.CompletedProcess:
    """A child process (a device-side assert ends its CUDA context) queuing :func:`cake_backend._check_ranges` of one
    segment on the int32 tile table ``tile_base`` (wrapped as int32 stores it); it prints ``checked`` after the
    synchronization."""
    code = (
        "import importlib, sys\n"
        f"sys.path[:0] = {sys.path!r}\n"
        "import torch\n"
        f"b = importlib.import_module({cake_backend.__name__!r})\n"
        "d = torch.device('cuda:0')\n"
        f"tb = torch.tensor({list(tile_base)!r}, dtype=torch.int32, device=d)\n"
        f"b._check_ranges(tb, {dict(plan)!r}, 1, {total_q!r})\n"
        "torch.cuda.synchronize()\n"
        "print('checked', flush=True)\n"
    )
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
    )


def test_a_range_past_two_to_the_31_tiles_fails_the_range_check(device) -> None:
    """The device tile table is an int32 prefix sum, which wraps past 2^31 tiles: a range of 2^31 + 64 tiles reads
    as a negative int32 difference, below any ``tile_stride``.  With ``T_q`` large enough for the wrap, the check
    rebuilds the counts in int64: that range fires the assert, and a wrapped table whose ranges fit passes (each in
    a child process)."""
    total_q = 2**31 - 1
    per_block = cake_jit.tiles_per_key_block(total_q, 1)  # 2^25
    marks = [n * per_block for n in range(65)]  # 64 key blocks, the last mark 2^31
    table = [(m + 2**31) % 2**32 - 2**31 for m in marks]
    assert marks[-1] >= 2**31 and table[-1] < 0
    fit = {
        "ranges": [(0, 32), (32, 64)],
        "tile_stride": 32 * per_block,
        "key_blocks": 64,
    }
    over = {"ranges": [(0, 64)], "tile_stride": 10, "key_blocks": 64}
    assert table[-1] - table[0] <= over["tile_stride"]  # an int32 difference would pass
    result = _child_check_ranges(table, fit, total_q)
    assert result.returncode == 0 and result.stdout.split() == ["checked"], (
        result.stderr[-2000:]
    )
    result = _child_check_ranges(table, over, total_q)
    assert result.returncode != 0 and "checked" not in result.stdout, result.stdout
    assert "device-side assert" in result.stderr or "disagree" in result.stderr, (
        result.stderr[-2000:]
    )


def test_the_tile_tables_wrap_and_the_range_marks_rebuild_them(device) -> None:
    """Past 2^31 tiles of one head pair, both tile-table builders (the PyTorch plan and the ``bwd_plan`` kernel) wrap
    modulo 2^32 (no saturation), and :func:`cake_backend._range_marks` gives the exact counts of
    :func:`.cake_jit.exact_tiles` before every key block."""
    for q_lens, k_lens, kernel in (
        ((6_000_000, 0), (6_000_000, 0), False),  # past PLAN_MAX_PAIRS: the PyTorch ops
        ((2_000_000,), (10_000_000,), True),  # one bwd_plan launch
    ):
        segments, total_q, total_k = len(q_lens), sum(q_lens), sum(k_lens)
        key_blocks = cake_jit.key_blocks(total_k, max(k_lens))
        counts = cake_jit.exact_tiles(q_lens, k_lens, key_blocks)
        want = [0, *itertools.accumulate(counts)]
        assert want[-1] >= 2**31, want[-1]
        cq = torch.tensor(_cu(q_lens), dtype=torch.int32, device=device)
        ck = torch.tensor(_cu(k_lens), dtype=torch.int32, device=device)
        builders = {"torch": cake_backend._tile_plan(cq, ck, key_blocks, total_q)}
        if kernel:
            assert cake_jit.pair_bound(total_q, segments) <= cake_jit.PLAN_MAX_PAIRS
            dims = {
                "total_q": total_q,
                "total_k": total_k,
                "num_heads_q": 24,
                "num_heads_kv": 4,
                "num_segments": segments,
            }
            builders["bwd_plan"] = cake_backend._plan_tables(
                cq, ck, key_blocks, total_q, dims, device.index
            )
        for name, tables in builders.items():
            marks = cake_backend._range_marks(tables[0], segments, total_q, key_blocks)
            assert marks.dtype == torch.int64, name
            assert marks.tolist() == want, (q_lens, k_lens, name)


def test_host_boundaries_come_together_at_every_size(device) -> None:
    """One host boundary vector without the other is refused by the workspace estimator at every size and by the
    backward also when the call has no query or no key (it returns early there); the empty backward still checks the
    host boundaries against the operands."""
    for args in ((0, 24, 1), (377, 24, 2)):
        with pytest.raises(ValueError, match="together"):
            cake_backend.cake_gqa_nondet_train_workspace_bytes(
                *args, cu_seqlens_q_host=[0, 0]
            )
    for q_lens, k_lens in (((0,), (8,)), ((8,), (0,))):
        x = _inputs(q_lens, k_lens, 24, 4, device)
        out, lse = cake_backend.cake_gqa_nondet_train_forward(
            x["q"], x["k"], x["v"], x["cu_q"], x["cu_k"], softmax_scale=SCALE
        )
        args = (x["q"], x["k"], x["v"], out, lse, x["dout"], x["cu_q"], x["cu_k"])
        with pytest.raises(ValueError, match="together"):
            cake_backend.cake_gqa_nondet_train_backward(
                *args, cu_seqlens_q_host=x["cq"]
            )
        with pytest.raises(ValueError, match="boundaries from 0"):
            cake_backend.cake_gqa_nondet_train_backward(
                *args, cu_seqlens_q_host=[0, 1], cu_seqlens_k_host=x["ck"]
            )
        dq, dk, dv = cake_backend.cake_gqa_nondet_train_backward(
            *args, cu_seqlens_q_host=x["cq"], cu_seqlens_k_host=x["ck"]
        )
        torch.cuda.synchronize()
        assert not dq.any() and not dk.any() and not dv.any()


def test_nine_argument_apply_calls_keep_working(device) -> None:
    """``CakeGQANondetFunction.apply`` without the host boundaries (its nine arguments before they were added) gives
    the bits of the call with ``None`` for them."""
    x = _inputs((257, 129), (513, 129), 24, 4, device)
    grads = []
    for extra in ((), (None,)):
        leaves = [x[n].clone().requires_grad_() for n in ("q", "k", "v")]
        out, _ = cake_backend.CakeGQANondetFunction.apply(
            *leaves, x["cu_q"], x["cu_k"], SCALE, None, None, None, *extra
        )
        out.backward(x["dout"])
        grads.append([out.detach()] + [t.grad for t in leaves])
    torch.cuda.synchronize()
    for got, want in zip(grads[0], grads[1], strict=True):
        assert torch.equal(got, want)


@pytest.mark.parametrize("host", (False, True), ids=("device_bounds", "host_bounds"))
@pytest.mark.parametrize("form", DKV_FORMS)
def test_a_cuda_graph_replays_the_eager_bits(form, host, device) -> None:
    """The forward and the backward captured in one CUDA graph with the minimum chunks (the side-stream fork and
    join of the tile tables and the fold order inside the capture, and with host boundaries the device-side range
    check): every replay gives the eager bits."""
    x = _inputs((700, 77, 300), (3000, 77, 300), 24, 4, device)
    row_map, n_dst = _window_row_map(x)
    dst_init = _destinations(n_dst, 4, device)
    ws = cake_backend.CakeGQANondetWorkspace(device, budget_bytes=1)

    def run():
        return _run(
            x, form, workspace=ws, dst_init=dst_init, row_map=row_map, host=host
        )

    eager = run()
    side = torch.cuda.Stream(device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        run()
    torch.cuda.current_stream(device).wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    for replay in range(3):
        graph.replay()
        torch.cuda.synchronize()
        _assert_outputs_same_bits(
            captured, eager, f"{form} host={host} replay {replay}"
        )


@pytest.mark.parametrize("map_name", ("windows", "scattered"))
@pytest.mark.parametrize("layout", LAYOUTS, ids=LAYOUT_IDS)
def test_row_map_adds_are_the_ordered_fold_of_the_window(
    layout, map_name, device
) -> None:
    """The row-map destinations are, bit for bit, the non-zero destinations with the FP32 window gradients of the
    same call added row by row in ascending window order; out, lse and dQ are those of the FP32 window form."""
    name, q_lens, k_lens, h_q, h_kv = layout
    x = _inputs(q_lens, k_lens, h_q, h_kv, device)
    row_map, n_dst = _row_maps(x)[map_name]
    dst_init = _destinations(n_dst, h_kv, device)
    window = _run(x, "fp32")
    got = _run(x, "rowmap", dst_init=dst_init, row_map=row_map)
    torch.cuda.synchronize()
    for t, init in (("dk", dst_init[0]), ("dv", dst_init[1])):
        expected = _ordered_fold(init, window[t], row_map)
        _assert_same_bits(got[t], expected, f"{name} {map_name}: {t}")
    _assert_outputs_same_bits(got, window, f"{name} {map_name}", ("out", "lse", "dq"))


def test_documents_are_isolated(device) -> None:
    q_lens, k_lens = (130, 77, 300, 41), (500, 77, 301, 1000)
    x = _inputs(q_lens, k_lens, 24, 4, device)
    base = _run(x, "fp32")
    y = dict(x)
    b = 2
    q0, q1, k0, k1 = x["cq"][b], x["cq"][b + 1], x["ck"][b], x["ck"][b + 1]
    for key, lo, hi in (("q", q0, q1), ("dout", q0, q1), ("k", k0, k1), ("v", k0, k1)):
        y[key] = x[key].clone()
        y[key][lo:hi] = -y[key][lo:hi] * 1.5
    pert = _run(y, "fp32")
    torch.cuda.synchronize()
    q_other = torch.ones(sum(q_lens), dtype=torch.bool, device=device)
    q_other[q0:q1] = False
    k_other = torch.ones(sum(k_lens), dtype=torch.bool, device=device)
    k_other[k0:k1] = False
    assert not torch.equal(pert["out"][~q_other], base["out"][~q_other])
    _assert_same_bits(pert["out"][q_other], base["out"][q_other], "out")
    _assert_same_bits(pert["lse"][:, q_other], base["lse"][:, q_other], "lse")
    _assert_same_bits(pert["dq"][q_other], base["dq"][q_other], "dq")
    _assert_same_bits(pert["dk"][k_other], base["dk"][k_other], "dk")
    _assert_same_bits(pert["dv"][k_other], base["dv"][k_other], "dv")


def test_dynamic_shapes_reuse_one_workspace(device) -> None:
    ws = cake_backend.CakeGQANondetWorkspace(device)
    shapes = (
        ((3000, 77), (9000, 77)),
        ((1,), (1,)),
        ((500, 0, 260), (500, 40, 4000)),
        ((2900, 300), (3000, 8000)),
    )
    sizes = []
    for i, (q_lens, k_lens) in enumerate(shapes):
        x = _inputs(q_lens, k_lens, 24, 4, device, seed=i)
        shared = _run(x, "fp32", workspace=ws)
        fresh = _run(x, "fp32", workspace=cake_backend.CakeGQANondetWorkspace(device))
        torch.cuda.synchronize()
        _assert_outputs_same_bits(shared, fresh, str(q_lens))
        sizes.append(ws.nbytes)
    assert sizes == sorted(sizes), "the workspace shrank"
    # The workspace keeps the largest scratch seen and nothing else (the dS^T chunk is allocated per call).
    need = max(
        cake_backend.cake_gqa_nondet_train_workspace_bytes(sum(q), 24, len(q))
        for q, _k in shapes
    )
    assert ws.nbytes - ws.tile_counter().untyped_storage().nbytes() == need


def test_launches_on_the_operands_device_and_stream(device) -> None:
    """Operands on ``device`` with another device current: every launch runs on ``device``'s current stream."""
    if torch.cuda.device_count() < 2:
        pytest.skip("needs a second CUDA device to make current")
    other = 1 if device.index == 0 else 0
    x = _inputs((700, 300, 41), (2000, 300, 900), 24, 4, device, seed=3)
    with torch.cuda.device(device):
        ref = _run(x, "fp32")
        torch.cuda.synchronize(device)
    prev_device = torch.cuda.current_device()
    prev_stream = torch.cuda.current_stream(device)
    side = torch.cuda.Stream(device)
    try:
        torch.cuda.set_stream(side)  # also makes ``device`` current
        torch.cuda.set_device(other)  # ``device`` keeps ``side`` as its current stream
        assert torch.cuda.current_device() == other
        assert torch.cuda.current_stream(device) == side
        side.wait_stream(prev_stream)
        got = _run(x, "fp32")
        side.synchronize()
    finally:
        torch.cuda.set_stream(prev_stream)
        torch.cuda.set_device(prev_device)
    for t in OUTPUTS:
        assert got[t].device == device, t
    _assert_outputs_same_bits(got, ref, "beside another current device")


@pytest.mark.parametrize("budget", (None, 1), ids=("default", "minimum"))
@pytest.mark.parametrize("form", DKV_FORMS)
def test_the_chunk_and_the_windows_live_for_one_call(form, budget, device) -> None:
    """The dS^T chunk (``ws_tiles`` 32 KiB tiles of the call's plan) and, in the row-map form, the two FP32 window
    gradients are allocated per call and freed when it returns: a backward into preallocated outputs through a
    workspace already sized for the shape leaves the allocated memory unchanged, and its peak is the per-call bytes
    ``cake_gqa_nondet_train_workspace_bytes`` adds to the scratch (plus small device plan tables)."""
    q_lens, k_lens = (1000, 77, 2600), (3000, 77, 2600)
    x = _inputs(q_lens, k_lens, 24, 4, device)
    ws = cake_backend.CakeGQANondetWorkspace(device, budget_bytes=budget)
    row_map, n_dst = _window_row_map(x)
    dst_init = _destinations(n_dst, 4, device)
    _run(x, form, workspace=ws, dst_init=dst_init, row_map=row_map)
    out, lse = cake_backend.cake_gqa_nondet_train_forward(
        x["q"], x["k"], x["v"], x["cu_q"], x["cu_k"], softmax_scale=SCALE, workspace=ws
    )
    max_k = None if form == "bf16" else _longest_k(x)
    kwargs = {
        "softmax_scale": SCALE,
        "workspace": ws,
        "max_seqlen_k": max_k,
        "dq": torch.empty_like(x["q"]),
    }
    if form == "rowmap":
        kwargs.update(
            dk_acc=dst_init[0].clone(), dv_acc=dst_init[1].clone(), row_map=row_map
        )
    else:
        dtype = torch.bfloat16 if form == "bf16" else torch.float32
        kwargs.update(
            dkv_dtype=dtype,
            dk=torch.empty(x["k"].shape, dtype=dtype, device=device),
            dv=torch.empty(x["v"].shape, dtype=dtype, device=device),
        )
    sizes = dict(
        total_k=sum(k_lens), num_heads_kv=4, max_seqlen_k=max_k, budget_bytes=budget
    )
    scratch = cake_backend.cake_gqa_nondet_train_workspace_bytes(
        sum(q_lens), 24, len(q_lens)
    )
    per_call = (
        cake_backend.cake_gqa_nondet_train_workspace_bytes(
            sum(q_lens), 24, len(q_lens), row_map=form == "rowmap", **sizes
        )
        - scratch
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    before = torch.cuda.memory_allocated(device)
    held = ws.nbytes
    cake_backend.cake_gqa_nondet_train_backward(
        x["q"], x["k"], x["v"], out, lse, x["dout"], x["cu_q"], x["cu_k"], **kwargs
    )
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated(device) - before
    assert torch.cuda.memory_allocated(device) == before, form
    assert ws.nbytes == held
    assert per_call <= peak <= per_call + 4 * MIB, (form, budget, per_call, peak)


def _ckpt_retained_mib(fn, *inputs) -> float:
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    out = checkpoint(fn, *inputs, use_reentrant=False)
    torch.cuda.synchronize()
    kept = torch.cuda.memory_allocated() - before - out.untyped_storage().nbytes()
    del out
    return kept / MIB


def _autograd(x, form, *, workspace, row_map=None, dst_init=None, use_checkpoint=False):
    """``cake_gqa_nondet`` forward and backward through autograd: ``{"out", "dq", "dk", "dv"}`` (dk / dv: the leaves'
    BF16 gradients, the FP32 window or the row-map destinations, by form), with ``max_seqlen_k`` as ``_run`` passes
    it; under non-reentrant checkpointing when asked, then also ``"retained_mib"``."""
    holder: dict = {}

    def dkv_acc():
        if form == "fp32":
            holder["dk"], holder["dv"] = (
                torch.empty(x[t].shape, dtype=torch.float32, device=x[t].device)
                for t in ("k", "v")
            )
            return holder["dk"], holder["dv"]
        holder["dk"], holder["dv"] = dst_init[0].clone(), dst_init[1].clone()
        return holder["dk"], holder["dv"], row_map

    def fn(q, k, v):
        return pkg.cake_gqa_nondet(
            q,
            k,
            v,
            x["cu_q"],
            x["cu_k"],
            softmax_scale=SCALE,
            workspace=workspace,
            dkv_acc=None if form == "bf16" else dkv_acc,
            max_seqlen_k=None if form == "bf16" else _longest_k(x),
        )

    q, k, v = (x[t].clone().requires_grad_() for t in ("q", "k", "v"))
    result: dict = {}
    if use_checkpoint:
        result["retained_mib"] = _ckpt_retained_mib(fn, q, k, v)
        out = checkpoint(fn, q, k, v, use_reentrant=False)
    else:
        out = fn(q, k, v)
    out.backward(x["dout"])
    if form == "bf16":
        dk, dv = k.grad, v.grad
    else:
        assert k.grad is None and v.grad is None
        dk, dv = holder["dk"], holder["dv"]
    result.update(out=out.detach(), dq=q.grad, dk=dk, dv=dv)
    return result


@pytest.mark.parametrize("form", DKV_FORMS)
def test_checkpoint_keeps_nothing_alive_and_matches_the_saved_set(form, device) -> None:
    """A checkpointed forward retains nothing; its replay gives the bits of the saved-set (non-checkpointed) run,
    which are those of the two low-level halves."""
    q_lens, k_lens = (1000, 3000), (1000, 24000)
    x = _inputs(q_lens, k_lens, 24, 4, device)
    ws = cake_backend.CakeGQANondetWorkspace(device)
    row_map, n_dst = _window_row_map(x)
    dst_init = _destinations(n_dst, 4, device)
    kwargs = dict(workspace=ws, row_map=row_map, dst_init=dst_init)
    eager = _autograd(x, form, **kwargs)
    ckpt = _autograd(x, form, use_checkpoint=True, **kwargs)
    eager2 = _autograd(x, form, **kwargs)
    low = _run(x, form, workspace=ws, dst_init=dst_init, row_map=row_map)
    torch.cuda.synchronize()
    assert ckpt["retained_mib"] <= CKPT_RETAINED_LIMIT_MIB, ckpt["retained_mib"]
    names = ("out", "dq", "dk", "dv")
    _assert_outputs_same_bits(ckpt, eager, f"{form} checkpointed", names)
    _assert_outputs_same_bits(eager2, eager, f"{form} eager again", names)
    _assert_outputs_same_bits(eager, low, f"{form} autograd vs halves", names)


@pytest.mark.parametrize("host", (False, True), ids=("device_bounds", "host_bounds"))
@pytest.mark.parametrize("budget", (None, 1), ids=("default", "minimum"))
@pytest.mark.parametrize("form", DKV_FORMS)
def test_steady_state_does_not_synchronize(form, budget, host, device) -> None:
    """Steady-state autograd steps of two shapes, with one chunk per call (the default budget) and with the
    minimum chunks (one KV head, two key blocks), with and without host boundaries, never synchronize the
    device."""
    xs = [
        _inputs(q, k, 24, 4, device, seed=i)
        for i, (q, k) in enumerate((((700, 77), (3000, 77)), ((300,), (300,))))
    ]
    ws = cake_backend.CakeGQANondetWorkspace(device, budget_bytes=budget)
    steps = []
    for x in xs:
        row_map, n_dst = _window_row_map(x)
        dst = tuple(torch.zeros(n_dst, 4, D, device=device) for _ in range(2))
        win = tuple(
            torch.empty(x[t].shape, dtype=torch.float32, device=device)
            for t in ("k", "v")
        )
        dkv_acc = {
            "bf16": None,
            "fp32": lambda win=win: win,
            "rowmap": lambda dst=dst, row_map=row_map: (*dst, row_map),
        }[form]
        steps.append((x, dkv_acc, None if form == "bf16" else _longest_k(x)))

    def step(x, dkv_acc, max_k):
        q, k, v = (x[t].clone().requires_grad_() for t in ("q", "k", "v"))
        out = pkg.cake_gqa_nondet(
            q,
            k,
            v,
            x["cu_q"],
            x["cu_k"],
            softmax_scale=SCALE,
            workspace=ws,
            dkv_acc=dkv_acc,
            max_seqlen_k=max_k,
            cu_seqlens_q_host=x["cq"] if host else None,
            cu_seqlens_k_host=x["ck"] if host else None,
        )
        out.backward(x["dout"])

    for args in steps:
        step(*args)
    torch.cuda.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for args in steps + steps:
            step(*args)
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    torch.cuda.synchronize()


@pytest.mark.parametrize("stage", ("fwd", "bwd"))
def test_keeps_its_speed_beside_a_resident_kernel(stage, device) -> None:
    q_lens, k_lens = (3000, 1100), (3000, 17000)
    x = _inputs(q_lens, k_lens, 24, 4, device)
    ws = cake_backend.CakeGQANondetWorkspace(device)
    out, lse = cake_backend.cake_gqa_nondet_train_forward(
        x["q"], x["k"], x["v"], x["cu_q"], x["cu_k"], softmax_scale=SCALE
    )

    def call():
        if stage == "fwd":
            return cake_backend.cake_gqa_nondet_train_forward(
                x["q"],
                x["k"],
                x["v"],
                x["cu_q"],
                x["cu_k"],
                softmax_scale=SCALE,
                workspace=ws,
            )
        return cake_backend.cake_gqa_nondet_train_backward(
            x["q"],
            x["k"],
            x["v"],
            out,
            lse,
            x["dout"],
            x["cu_q"],
            x["cu_k"],
            softmax_scale=SCALE,
            workspace=ws,
        )

    side = torch.cuda.Stream()
    for _ in range(WARMUP):
        _timed(call)
    alone, busy = [], []
    for _ in range(RUNS):
        alone.append(_timed(call))
        busy.append(_timed(call, sides=(side,)))
    alone_ms = statistics.median(t for t, _ in alone)
    busy_ms = statistics.median(t for t, _ in busy)
    print(
        f"STALL {stage} alone_ms={alone_ms:.4f} busy_ms={busy_ms:.4f} ratio={busy_ms / alone_ms:.3f}"
    )
    reference = alone[0][1]
    for _t, result in alone[1:] + busy:
        for i, (got, want) in enumerate(zip(result, reference, strict=True)):
            _assert_same_bits(
                got, want, f"{stage} output {i}: the result depends on the scheduling"
            )
    assert busy_ms / alone_ms <= MAX_RATIO or busy_ms - alone_ms <= MAX_EXTRA_MS, (
        f"{stage}: {busy_ms:.3f} ms beside a resident kernel vs {alone_ms:.3f} ms alone"
    )


def test_rejects_unsupported_inputs(device) -> None:
    x = _inputs((10,), (10,), 24, 4, device)
    q, k, v, cq, ck, dout = x["q"], x["k"], x["v"], x["cu_q"], x["cu_k"], x["dout"]
    with pytest.raises(ValueError, match="even multiple"):
        cake_backend.cake_gqa_nondet_train_forward(
            q[:, :12], k[:, :4], v[:, :4], cq, ck
        )  # group 3
    with pytest.raises(ValueError, match="bfloat16"):
        cake_backend.cake_gqa_nondet_train_forward(q.half(), k, v, cq, ck)
    with pytest.raises(ValueError, match="int32"):
        cake_backend.cake_gqa_nondet_train_forward(q, k, v, cq.long(), ck)
    with pytest.raises(ValueError, match="num_segments"):
        cake_backend.cake_gqa_nondet_train_forward(q, k, v, cq, ck[:1])
    with pytest.raises(NotImplementedError):
        pkg.cake_gqa_nondet(q, k, v, cq, ck, causal=False)
    with pytest.raises(TypeError, match="dkv_acc"):
        pkg.cake_gqa_nondet(q, k, v, cq, ck, dkv_acc=object())
    for budget in (0, -1):
        with pytest.raises(ValueError, match="budget_bytes"):
            cake_backend.CakeGQANondetWorkspace(device, budget_bytes=budget)
    out, lse = cake_backend.cake_gqa_nondet_train_forward(q, k, v, cq, ck)
    args = (q, k, v, out, lse, dout, cq, ck)
    with pytest.raises(ValueError, match="max_seqlen_k"):
        cake_backend.cake_gqa_nondet_train_backward(*args, max_seqlen_k=-1)
    with pytest.raises(ValueError, match="dkv_dtype"):
        cake_backend.cake_gqa_nondet_train_backward(*args, dkv_dtype=torch.float16)
    row_map = torch.zeros(10, dtype=torch.int32, device=device)
    dst = torch.zeros(1, 4, D, device=device)
    with pytest.raises(ValueError, match="row_map needs"):
        cake_backend.cake_gqa_nondet_train_backward(*args, row_map=row_map, dk_acc=dst)
    with pytest.raises(ValueError, match="need a row_map"):
        cake_backend.cake_gqa_nondet_train_backward(*args, dk_acc=dst, dv_acc=dst)
    with pytest.raises(ValueError, match="row_map must be int32"):
        cake_backend.cake_gqa_nondet_train_backward(
            *args, row_map=row_map.long(), dk_acc=dst, dv_acc=dst
        )


# ----------------------------------------------------------------------------------------------------------------
# torch.use_deterministic_algorithms (#8928 rule 8)


@pytest.fixture
def deterministic_algorithms():
    """Restores torch's deterministic-algorithms mode after a test that sets it."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    yield
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


@pytest.mark.parametrize("warn_only", (False, True), ids=("strict", "warn_only"))
@pytest.mark.parametrize("form", DKV_FORMS)
def test_deterministic_mode_raises_no_alert_and_keeps_the_bits(
    form, warn_only, device, deterministic_algorithms
) -> None:
    """Under ``torch.use_deterministic_algorithms(True)``, strict or ``warn_only=True``: no alert, no warning, and
    the two halves and autograd give the bits of a run outside that mode.  The workspaces of the run in the mode are
    allocated in it, so PyTorch fills their uninitialized memory (and the outputs') with NaN: the bits also show
    that no kernel reads memory it did not write."""
    x = _inputs((130, 77, 300), (200, 77, 301), 24, 4, device)
    row_map, n_dst = _scattered_row_map(x)
    dst_init = _destinations(n_dst, 4, device)

    def both():
        low = _run(
            x,
            form,
            workspace=cake_backend.CakeGQANondetWorkspace(device),
            dst_init=dst_init,
            row_map=row_map,
        )
        auto = _autograd(
            x,
            form,
            workspace=cake_backend.CakeGQANondetWorkspace(device),
            row_map=row_map,
            dst_init=dst_init,
        )
        return low, auto

    torch.use_deterministic_algorithms(False)
    base_low, base_auto = both()
    torch.cuda.synchronize()
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got_low, got_auto = both()
        torch.cuda.synchronize()
    assert not caught, [f"{w.category.__name__}: {w.message}" for w in caught]
    _assert_outputs_same_bits(got_low, base_low, f"{form} halves")
    _assert_outputs_same_bits(
        got_auto, base_auto, f"{form} autograd", ("out", "dq", "dk", "dv")
    )


def test_deterministic_mode_runs_an_empty_backward(
    device, deterministic_algorithms
) -> None:
    """No (query, key) pair launches nothing: zero gradients, no alert."""
    x = _inputs((0, 0), (5, 9), 24, 4, device)
    q, k, v, cq, ck = x["q"], x["k"], x["v"], x["cu_q"], x["cu_k"]
    out, lse = cake_backend.cake_gqa_nondet_train_forward(q, k, v, cq, ck)
    torch.use_deterministic_algorithms(True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dq, dk, dv = cake_backend.cake_gqa_nondet_train_backward(
            q, k, v, out, lse, x["dout"], cq, ck
        )
    assert dq.numel() == 0
    assert not dk.any() and not dv.any()
