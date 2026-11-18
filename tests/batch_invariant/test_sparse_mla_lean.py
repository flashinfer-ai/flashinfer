# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The Lean coupling of the sparse MLA classes ``mla.sparse.<bf16|e4m3>.d512[r64]`` (``flashinfer/batch_invariant/lean/``:
``SparseMLA.lean`` and the generated ``SparseMLACases.lean``).

Device-free (every CI job): the committed ``SparseMLACases.lean`` is byte for byte what the Python spec and config emit,
so a change to the dataflow, the configuration space or the dispatcher that was not propagated to the Lean cases fails
here without a toolchain; the Python-side facts of the space and of the engine plans are checked as well. With ``lake``
installed: ``lake build`` elaborates the class, its theorems and the generated cases (kernel ``decide`` throughout: the
cases add nothing to the axiom policy), ``lake exe sparsemlaconformance`` prints the Lean model's terms, as the class
term and through the fixup kernel's model, compared here with the Python S-expressions of every case (the full-tile
ones, megabytes each, included), and ``lake exe axioms`` enforces the axiom policy of the whole library.
"""

import os
import shutil
import subprocess

import pytest

from flashinfer.batch_invariant.fp_interface.term import TermAlgebra, sexpr
from flashinfer.mla.batch_invariant import config, lean_cases, spec

LAKE = shutil.which("lake") or os.path.expanduser("~/.elan/bin/lake")
needs_lake = pytest.mark.skipif(
    not os.path.exists(LAKE), reason="Lean toolchain (elan / lake) not installed"
)


def test_generated_cases_are_current():
    with open(lean_cases.CASES_PATH, encoding="utf-8") as f:
        committed = f.read()
    assert committed == lean_cases.render(), (
        "regenerate with: python -m flashinfer.mla.batch_invariant.lean_cases"
    )
    cases = lean_cases.mla_cases()
    names = [c[0] for c in cases]
    assert len(names) == len(set(names)) >= 10
    for name, _, sx, _, as_thm, _ in cases:
        assert as_thm == (len(sx) <= lean_cases.THEOREM_MAX_BYTES), name
        assert (f"theorem smla_{name}_spec :" in committed) == as_thm, name
        assert f"theorem smla_{name}_kernel :" in committed, name
    by_name = {c[0]: c for c in lean_cases.CASES}
    # what the cases cover: n = 0, one position, both layouts' K steps, a full tile with a partial one, the e4m3 member
    assert any(c[3] == 0 for c in lean_cases.CASES)
    assert {c[2] for c in lean_cases.CASES if c[3] == 1 and c[1] == "bf16"} == {
        spec.d_qk(r) for r in spec.ROPE_DIMS
    }
    assert any(spec.TILE_KEYS < c[3] < 2 * spec.TILE_KEYS for c in lean_cases.CASES)
    assert {c[1] for c in lean_cases.CASES} == set(spec.CACHE_FMTS)
    assert by_name["empty"][3] == 0
    theorems = {c[0] for c in cases if c[4]}
    assert {"empty", "one_d512", "one_d512r64", "part3", "e4m3_part2"} <= theorems
    assert "sorry" not in committed and "native_decide" not in committed


def test_cases_are_the_spec_terms():
    # the generator's reduced-K path and the class's own symbolic view build one term at the class's K
    for fmt in spec.CACHE_FMTS:
        for rope in spec.ROPE_DIMS:
            s = spec.SparseMlaSpec(rope, fmt)
            for n, d in ((0, 0), (1, 0), (5, 3)):
                assert sexpr(
                    s.output(spec.SparseRow(n, 2), d)
                ) == lean_cases.case_sexpr(fmt, s.D, n, 2, d)
    # the e4m3 member wraps the query slice, the cache slice and the V slice
    sx = lean_cases.case_sexpr("e4m3", 16, 1, 0, 0)
    for leaf in ("(a 0 0)", "(b 0 0)", "(op:v_slice 0 0)"):
        assert f"(op:{spec.CVT_OP} {leaf})" in sx, leaf
    assert spec.CVT_OP not in lean_cases.case_sexpr("bf16", 16, 1, 0, 0)


def test_configuration_space_and_dispatcher():
    space = config.all_valid_configs()
    assert len(space) == len(set(space)) > 0
    assert all(c.well_formed() for c in space)
    members = set(space)
    picks = lean_cases.dispatch_cases()
    assert len(picks) == (
        len(lean_cases.DISPATCH_ROWS)
        * len(lean_cases.DISPATCH_HEADS)
        * len(lean_cases.DISPATCH_WIDTHS)
        * len(lean_cases.DISPATCH_SMS)
    )
    for r, h, w, s, cfg in picks:
        assert cfg in members, (r, h, w, s, cfg)
    # the dispatcher's three outcomes are all on the grid
    assert {(cfg.fold, cfg.tiles_per_unit) for *_, cfg in picks} == {
        ("fixup", 1),
        ("fixup", 4),
        ("inkernel", 0),
    }
    assert {cfg.mma_split for *_, cfg in picks} == set(config.MMA_SPLITS)
    with open(lean_cases.CASES_PATH, encoding="utf-8") as f:
        committed = f.read()
    assert f"smlaSpace.length = {len(space)} " in committed
    for c in space:
        assert c.lean() in committed


def test_key_plans():
    spec.verify_key_plans(TermAlgebra())
    keys = lean_cases.key_cases()
    assert {k[1] for k in keys} == set(spec.KEY_SPLITS) == set(config.MMA_SPLITS)
    assert {k[2] for k in keys} == {"max", "sum"}
    assert len(keys) == 2 * len(spec.KEY_SPLITS) * len(lean_cases.KEY_NS)
    with open(lean_cases.CASES_PATH, encoding="utf-8") as f:
        committed = f.read()
    for split in spec.KEY_SPLITS:
        assert f"theorem smla_key_placement_w{split}_ok :" in committed
        assert (
            f"smlaKeyPlacementW{split} = SparseMLA.keyPlacementOf {split} " in committed
        )
    assert spec.KEY_DIGITS == (("e", 2), ("b", 16), ("c", 4))
    assert spec.TILE_KEYS == 128 == 2 * 16 * 4
    assert sorted(
        spec.key_of(b, c, e) for b in range(16) for c in range(4) for e in range(2)
    ) == list(range(spec.TILE_KEYS))


def _lake(*args, timeout):
    env = dict(
        os.environ,
        ELAN_NO_OVERRIDE_NOTICE="1",
        PATH=os.path.dirname(LAKE) + os.pathsep + os.environ["PATH"],
    )
    return subprocess.run(
        [LAKE, *args],
        cwd=lean_cases.LEAN_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@needs_lake
def test_lean_model_matches_python_spec():
    r = _lake("build", timeout=3600)
    assert r.returncode == 0, "Lean proofs failed:\n" + (r.stdout + r.stderr)[-4000:]
    r = _lake("build", "sparsemlaconformance", timeout=1800)
    assert r.returncode == 0, (r.stdout + r.stderr)[-4000:]
    r = _lake("exe", "sparsemlaconformance", timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    lean = {"smla": {}, "smlafixup": {}, "smlakeys": {}, "smlakeysspec": {}}
    for line in r.stdout.splitlines():
        tag, name, sx = line.split(" ", 2)
        assert tag in lean, f"unexpected conformance line: {line[:80]}"
        assert name not in lean[tag], f"duplicate conformance case {tag} {name}"
        lean[tag][name] = sx
    cases = lean_cases.mla_cases()
    for name, _, sx, _, _, _ in cases:
        assert lean["smla"][name] == sx, f"model differs from the Python spec: {name}"
        assert lean["smlafixup"][name] == sx, f"fixup model differs: {name}"
    assert len(lean["smla"]) == len(lean["smlafixup"]) == len(cases)
    keys = lean_cases.key_cases()
    for name, _, _, _, sx in keys:
        assert lean["smlakeys"][name] == sx, f"executed key plan differs: {name}"
        assert lean["smlakeysspec"][name] == sx, f"key reduction differs: {name}"
    assert len(lean["smlakeys"]) == len(lean["smlakeysspec"]) == len(keys)


@needs_lake
def test_axioms_policy():
    r = _lake("build", "axioms", timeout=1800)
    assert r.returncode == 0, (r.stdout + r.stderr)[-4000:]
    r = _lake("exe", "axioms", timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    assert "0 with sorry, 0 violations" in r.stdout, r.stdout
