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

flashinfer public-API tests of the Cake QSA NVFP4 sparse paged attention
(``flashinfer.experimental.cake_qsa_nvfp4``).

The backend suite ``test_cake_qsa_nvfp4.py`` exercises the package.  This file covers
what needs flashinfer itself: the ``backend`` keyword of
``flashinfer.sparse.qsa_sparse_paged_attention`` -- ``backend="cake"`` reaches the
package with every operand, the default ``"auto"`` is the package on every compute
capability above 9.0 it exports a library for (10.0, 10.3, 10.7, 12.0: capability-exact) and the
existing implementation elsewhere, ``"default"`` keeps the existing
implementation, an unknown value is refused -- device runs of the route and of the auto
default against the package, bit for bit, and
the NVFP4 paged writer against the torch port the backend suite fills its caches with.
It also holds the host's program rule on the CPU: which records a call launches for its
form, architecture and grid classes, when a call is wide (on sm_80 above one resident
wave with at least 16 tiles per CTA; on sm_120a at least 11 tiles per CTA whatever the
grid) and when the sm_103a / sm_107a persistent walk serves it (a multi-wave
split-form grid within the architecture's wave bound whose longest walked CTA is
shorter than the plain program's).
"""

import contextlib
import glob
import inspect
import re
import warnings
from pathlib import Path

import pytest
import torch

import flashinfer
import flashinfer.experimental.qsa_nvfp4.backend as existing_backend
import flashinfer.sparse as public_api
from flashinfer.api_logging import ExperimentalWarning
from flashinfer.experimental.cake_qsa_nvfp4 import cake_backend, cake_jit
from tests.test_helpers.cake_qsa_nvfp4_reference import write_slots

PACKAGE_DIR = Path(cake_jit.__file__).resolve().parent
HEAD_DIM = 256


@contextlib.contextmanager
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        yield


def _device_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    index = torch.cuda.current_device()
    capability = tuple(torch.cuda.get_device_capability(index))
    if cake_jit.arch_for_capability(capability) is None:
        pytest.skip(
            f"compute capability {capability} is not an exported target {cake_jit.ARCHES}"
        )
    if not (PACKAGE_DIR / "build.json").exists():
        pytest.skip("the package has no build.json yet (not exported)")
    return torch.device("cuda", index)


def _recorder(calls, tag):
    def run(*args, **kwargs):
        calls.append((tag, args, kwargs))
        return tag

    return run


def _patch_both_backends(monkeypatch):
    """Replace both implementations' entries by recorders; returns the call log."""
    calls = []
    monkeypatch.setattr(
        cake_backend, "qsa_sparse_paged_attention", _recorder(calls, "cake")
    )
    monkeypatch.setattr(
        existing_backend, "qsa_sparse_paged_attention", _recorder(calls, "existing")
    )
    return calls


def test_entry_point_is_marked_experimental():
    assert public_api.qsa_sparse_paged_attention.is_experimental


def test_backend_is_a_keyword_with_a_default():
    parameter = inspect.signature(public_api.qsa_sparse_paged_attention).parameters[
        "backend"
    ]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default == "auto"


def test_unknown_backend_is_rejected(monkeypatch):
    calls = _patch_both_backends(monkeypatch)
    t = torch.empty(0)
    with pytest.raises(ValueError, match="backend"), _quiet():
        public_api.qsa_sparse_paged_attention(
            t, t, t, t, t, t, False, output_gate=t, k_sf=t, v_sf=t, backend="other"
        )
    assert calls == []


def test_cake_backend_reaches_the_package_with_every_operand(monkeypatch):
    signature = inspect.signature(cake_backend.qsa_sparse_paged_attention)
    calls = _patch_both_backends(monkeypatch)
    q, k_cache, v_cache, indices, table, token_to_req, out, gate, k_sf, v_sf = (
        torch.empty(i) for i in range(10)
    )
    with _quiet():
        result = public_api.qsa_sparse_paged_attention(
            q,
            k_cache,
            v_cache,
            indices,
            table,
            token_to_req,
            True,
            out,
            0.5,
            0.25,
            output_gate=gate,
            k_sf=k_sf,
            v_sf=v_sf,
            backend="cake",
        )
    assert result == "cake"
    assert [tag for tag, _, _ in calls] == ["cake"]
    _, args, kwargs = calls[0]
    received = signature.bind(*args, **kwargs).arguments
    tensors = dict(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        logical_indices=indices,
        block_table=table,
        token_to_req=token_to_req,
        out=out,
        output_gate=gate,
        k_sf=k_sf,
        v_sf=v_sf,
    )
    for name, value in tensors.items():
        assert received[name] is value, name
    assert received["use_prefill_config"] is True
    assert received["k_scale"] == 0.5 and received["v_scale"] == 0.25


def test_explicit_default_backend_keeps_the_existing_implementation(monkeypatch):
    calls = _patch_both_backends(monkeypatch)
    t = torch.empty(0)
    with _quiet():
        result = public_api.qsa_sparse_paged_attention(
            t, t, t, t, t, t, False, output_gate=t, k_sf=t, v_sf=t, backend="default"
        )
    assert result == "existing"
    assert [tag for tag, _, _ in calls] == ["existing"]


def test_auto_backend_resolves_by_capability(monkeypatch):
    """``auto`` is the Cake route for every capability above 9.0 the package exports (10.0,
    10.3, 10.7, 12.0) and the reference implementation for 8.x, 9.0, any other capability or a
    non-CUDA device: the rule is capability-exact, so 12.1 (no delivered library) stays on
    the reference until the package ships for it; the table below needs no GPU, the capability
    lookup is replaced."""
    assert public_api._qsa_auto_backend(torch.device("cpu")) == "default"
    device = torch.device("cuda", 0)
    table = {}
    monkeypatch.setattr(public_api, "get_compute_capability", lambda d: table[d])
    assert sorted(cake_jit.ARCH_BY_CAPABILITY) == [
        (8, 0),
        (8, 6),
        (8, 7),
        (8, 9),
        (9, 0),
        (10, 0),
        (10, 3),
        (10, 7),
        (12, 0),
    ]
    for capability in sorted(cake_jit.ARCH_BY_CAPABILITY):
        table[device] = capability
        want = "default" if capability[0] in (8, 9) else "cake"
        assert public_api._qsa_auto_backend(device) == want, capability
    for capability in ((7, 5), (12, 1)):
        table[device] = capability
        assert public_api._qsa_auto_backend(device) == "default", capability


def test_auto_rule_covers_every_exported_capability(monkeypatch):
    """The entry's rule is the package's capability table above 9.0 and nothing else: over
    every capability of majors 7 through 12, exactly the exported ones above 9.0 resolve to the
    Cake route, so ``auto`` never selects a route that refuses the device and never leaves an
    exported device above 9.0 on the reference."""
    device = torch.device("cuda", 0)
    table = {}
    monkeypatch.setattr(public_api, "get_compute_capability", lambda d: table[d])
    routed = set()
    for major in range(7, 13):
        for minor in range(10):
            table[device] = (major, minor)
            if public_api._qsa_auto_backend(device) == "cake":
                routed.add((major, minor))
    assert routed == {c for c in cake_jit.ARCH_BY_CAPABILITY if c[0] > 9}


def test_omitted_backend_takes_the_auto_route(monkeypatch):
    """On the device at hand an omitted ``backend`` reaches the package when the device's
    capability is an exported one above 9.0 and the existing implementation otherwise
    (recorders stand in for both, so this needs a CUDA device but no exported target)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    calls = _patch_both_backends(monkeypatch)
    device = torch.device("cuda", torch.cuda.current_device())
    capability = tuple(torch.cuda.get_device_capability(device))
    t = torch.empty(0, device=device)
    with _quiet():
        result = public_api.qsa_sparse_paged_attention(
            t, t, t, t, t, t, False, output_gate=t, k_sf=t, v_sf=t
        )
    exported = cake_jit.arch_for_capability(capability) is not None
    want = "cake" if exported and capability[0] > 9 else "existing"
    assert result == want
    assert [tag for tag, _, _ in calls] == [want]


@pytest.mark.parametrize("kv_dtype", [torch.float8_e4m3fn, torch.bfloat16])
def test_omitted_backend_sends_fp8_and_bf16_to_the_existing_implementation(
    monkeypatch, kv_dtype
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    calls = _patch_both_backends(monkeypatch)
    t = torch.empty(0, device="cuda")
    kv = t.to(kv_dtype)
    with _quiet():
        result = public_api.qsa_sparse_paged_attention(
            t, kv, kv, t, t, t, False, output_gate=t
        )
    assert result == "existing"
    assert [tag for tag, _, _ in calls] == ["existing"]


def test_omitted_backend_on_a_cpu_tensor_takes_the_existing_implementation(monkeypatch):
    calls = _patch_both_backends(monkeypatch)
    t = torch.empty(0)
    with _quiet():
        result = public_api.qsa_sparse_paged_attention(
            t, t, t, t, t, t, False, output_gate=t, k_sf=t, v_sf=t
        )
    assert result == "existing"
    assert [tag for tag, _, _ in calls] == ["existing"]


def test_generated_sources_are_declared_as_package_data():
    """A non-editable install carries what the loader resolves next to cake_jit.py: every
    source under csrc/ (the generated kernels, bindings and headers; not the formatter's
    dotfile) and build.json.  pyproject.toml sets include-package-data = false, so only
    this package's package-data globs put them into a wheel; the auto default makes that
    a requirement of every install on an exported capability."""
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if not pyproject.is_file():
        pytest.skip("pyproject.toml is only available in source-tree test runs")
    key = '"flashinfer.experimental.cake_qsa_nvfp4" = ['
    text = pyproject.read_text(encoding="utf-8")
    assert key in text, "pyproject.toml has no package-data entry for the package"
    globs = re.findall(r'"([^"]*)"', text.split(key, 1)[1].split("]", 1)[0])
    shipped = {
        path
        for pattern in globs
        for path in glob.glob(pattern, root_dir=PACKAGE_DIR, recursive=True)
        if (PACKAGE_DIR / path).is_file()
    }
    needed = {
        str(path.relative_to(PACKAGE_DIR))
        for path in (PACKAGE_DIR / "csrc").rglob("*")
        if path.is_file() and not path.name.startswith(".")
    }
    needed.add("build.json")
    assert needed, "the package ships no generated sources"
    assert needed <= shipped, sorted(needed - shipped)


def test_program_rule_selects_the_records_of_a_call():
    """The records are filtered by the device's architecture first: the ``mma.sync`` decode's
    on sm_80 / sm_90a / sm_120a, the tcgen05 decode's on sm_100a / sm_103a / sm_107a.  On sm_80
    and sm_120a a wide split-form call launches the whole-tile decode (sm_80: a multi-wave,
    long-stream grid; sm_120a: a long per-CTA stream), every other call the K-first decode
    (sm_90a never selects the whole-tile record); the tcgen05 architectures launch the
    tcgen05 decode whatever the grid, except that on sm_103a / sm_107a a walked split-form
    call (``walk`` = the persistent walk's CTA count) launches the persistent walk;
    the merge follows above one split on every architecture; the fused form is the
    architecture's cooperative decode alone; an unknown form, architecture or split count is
    refused, as is a walk on an architecture whose decode cannot launch it (the plan)."""
    select = cake_jit.select_records
    assert select(form="split", splits=4, arch="sm_80", wide=True) == (
        "decode_whole_tile",
        "merge",
    )
    assert select(form="split", splits=4, arch="sm_80", wide=False) == (
        "decode",
        "merge",
    )
    assert select(form="split", splits=1, arch="sm_80", wide=True) == (
        "decode_whole_tile",
    )
    assert select(form="split", splits=1, arch="sm_80", wide=False) == ("decode",)
    assert select(form="split", splits=3, arch="sm_120a", wide=True) == (
        "decode_whole_tile",
        "merge",
    )
    assert select(form="split", splits=1, arch="sm_120a", wide=True) == (
        "decode_whole_tile",
    )
    assert select(form="split", splits=5, arch="sm_120a", wide=False) == (
        "decode",
        "merge",
    )
    assert select(form="split", splits=1, arch="sm_120a", wide=False) == ("decode",)
    for wide in (False, True):
        assert select(form="split", splits=4, arch="sm_90a", wide=wide) == (
            "decode",
            "merge",
        )
        assert select(form="split", splits=1, arch="sm_90a", wide=wide) == ("decode",)
        for arch in ("sm_80", "sm_90a", "sm_120a"):
            assert select(form="fused", splits=4, arch=arch, wide=wide) == (
                "decode_coop",
            )
        for arch in ("sm_100a", "sm_103a", "sm_107a"):
            assert select(form="split", splits=4, arch=arch, wide=wide) == (
                "decode_tcgen05",
                "merge",
            )
            assert select(form="split", splits=1, arch=arch, wide=wide) == (
                "decode_tcgen05",
            )
            assert select(form="fused", splits=4, arch=arch, wide=wide) == (
                "decode_tcgen05_coop",
            )
    # The persistent walk: on sm_103a / sm_107a a walked split-form call launches the walk,
    # then the merge above one partial slot; the fused form ignores the walk; sm_100a has
    # no record reading the walked grid class, so selection falls through to the plain
    # tcgen05 decode and the plan refuses the walk there (its split count would be the
    # walk's slots, not the plain cut).
    for arch, ctas in (("sm_103a", 152), ("sm_107a", 212)):
        assert select(form="split", splits=2, arch=arch, wide=True, walk=ctas) == (
            "decode_tcgen05_persistent",
            "merge",
        )
        assert select(form="split", splits=1, arch=arch, wide=False, walk=1) == (
            "decode_tcgen05_persistent",
        )
        assert select(form="fused", splits=2, arch=arch, wide=False, walk=ctas) == (
            "decode_tcgen05_coop",
        )
    for arch in ("sm_100a",):
        assert select(form="split", splits=2, arch=arch, wide=False, walk=148) == (
            "decode_tcgen05",
            "merge",
        )
        with pytest.raises(ValueError, match="persistent walk"):
            cake_jit.resolved_launches(
                form="split",
                splits=2,
                arch=arch,
                wide=False,
                rows=128,
                kv_heads=2,
                qo_heads=24,
                walk=148,
            )
    with pytest.raises(ValueError, match="walk"):
        select(form="split", splits=1, arch="sm_103a", wide=False, walk=-1)
    assert cake_jit.ARCHES == (
        "sm_80",
        "sm_90a",
        "sm_100a",
        "sm_103a",
        "sm_107a",
        "sm_120a",
    )
    for name in cake_jit.MODULES:
        assert cake_jit.MODULES[name]["stage"] in cake_jit.STAGES
        assert set(cake_jit.MODULES[name]["arches"]) <= set(cake_jit.ARCHES)
    # Every (form, architecture, grid classes) selects exactly one decode record; the merge is one record on every
    # architecture.
    for arch in cake_jit.ARCHES:
        for form in cake_jit.FORMS:
            for wide in (False, True):
                for walk in (0, 1):
                    cake_jit.select_record(
                        "decode", form=form, arch=arch, wide=wide, walk=walk
                    )
        assert (
            cake_jit.select_record("merge", form="split", arch=arch, wide=False)
            == "merge"
        )
    with pytest.raises(ValueError, match="form"):
        select(form="other", splits=1, arch="sm_80", wide=False)
    with pytest.raises(ValueError, match="arch"):
        select(form="split", splits=1, arch="sm_121a", wide=False)
    with pytest.raises(ValueError, match="splits"):
        select(form="split", splits=0, arch="sm_80", wide=False)
    with pytest.raises(ValueError, match="merge"):
        cake_jit.select_record("merge", form="fused", arch="sm_90a", wide=False)


def test_two_wave_walk_rule_on_sm_103a():
    """``select_walk`` / ``num_splits`` at the owner's shape (2 KV heads, 33 tiles) on a
    GB300's 152 SMs and on 148: rows 128 (256 plain CTAs in two waves) take the persistent
    walk on one CTA per SM with 2 partial slots per (row, KV head); rows 1-64 (one wave) and
    rows 256 / 512 (4 / 7 waves, past ``WALK_MAX_WAVES["sm_103a"]``) keep the plain cut, as
    does every architecture outside ``WALK_ARCHES``.  On an R200's 212 SMs (sm_107a, bound
    three waves) rows 128 and 256 are walked, rows 1-64 and 512 keep the plain cut.  The walk
    arithmetic: ``walk_ctas`` never exceeds the slots, ``walk_splits`` is the most segments
    any pair is cut into, ``walk_applies`` needs the architecture's waves at most and a
    shorter longest CTA; the walk's launch grid is ``(walk, 1, 1)`` and the merge's ignores
    the walk."""
    tiles = cake_backend.num_tiles(2051)
    assert tiles == 33
    assert cake_backend.WALK_ARCHES == ("sm_103a", "sm_107a")
    assert (
        cake_backend.WALK_MAX_WAVES
        == cake_jit.WALK_MAX_WAVES
        == {"sm_103a": 2, "sm_107a": 3}
    )
    assert tuple(cake_jit.WALK_MAX_WAVES) == cake_backend.WALK_ARCHES
    assert (
        tuple(cake_jit.MODULES["decode_tcgen05_persistent"]["arches"])
        == cake_backend.WALK_ARCHES
    )
    for rows in (128, 256):
        assert cake_backend.select_walk(rows, 2, tiles, 212, "sm_107a") == 212, rows
        assert cake_backend.plain_splits(rows, 2, 2051, 212, "sm_107a") == 1, rows
        assert cake_backend.num_splits(rows, 2, 2051, 212, "sm_107a") == 2, rows
    for rows in (1, 2, 4, 8, 16, 32, 64, 512):
        assert cake_backend.select_walk(rows, 2, tiles, 212, "sm_107a") == 0, rows
        assert cake_backend.num_splits(
            rows, 2, 2051, 212, "sm_107a"
        ) == cake_backend.plain_splits(rows, 2, 2051, 212, "sm_107a"), rows
    for sm_count in (148, 152):
        assert cake_backend.select_walk(128, 2, tiles, sm_count, "sm_103a") == sm_count
        assert cake_backend.plain_splits(128, 2, 2051, sm_count, "sm_103a") == 1
        assert cake_backend.num_splits(128, 2, 2051, sm_count, "sm_103a") == 2
        for rows in (1, 2, 4, 8, 16, 32, 64, 256, 512):
            assert cake_backend.select_walk(rows, 2, tiles, sm_count, "sm_103a") == 0, (
                rows
            )
            assert cake_backend.num_splits(
                rows, 2, 2051, sm_count, "sm_103a"
            ) == cake_backend.plain_splits(rows, 2, 2051, sm_count, "sm_103a"), rows
        for arch in ("sm_80", "sm_90a", "sm_100a", "sm_120a"):
            assert cake_backend.select_walk(128, 2, tiles, sm_count, arch) == 0, arch
            assert cake_backend.num_splits(
                128, 2, 2051, sm_count, arch
            ) == cake_backend.plain_splits(128, 2, 2051, sm_count, arch), arch
    # The wave window: 256 pairs at one split on 152 SMs are two waves (the walk applies),
    # rows 256 four (past sm_103a's bound), rows 64 one (nothing to balance); on 212 SMs rows
    # 256 are three waves (within sm_107a's bound), rows 512 five (past it).
    assert cake_jit.walk_applies(128, 2, tiles, 1, 152, "sm_103a")
    assert not cake_jit.walk_applies(256, 2, tiles, 1, 152, "sm_103a")
    assert not cake_jit.walk_applies(64, 2, tiles, 1, 152, "sm_103a")
    assert cake_jit.walk_applies(256, 2, tiles, 1, 212, "sm_107a")
    assert not cake_jit.walk_applies(512, 2, tiles, 1, 212, "sm_107a")
    assert not cake_jit.walk_applies(64, 2, tiles, 1, 212, "sm_107a")
    with pytest.raises(ValueError, match="wave bound"):
        cake_jit.walk_applies(128, 2, tiles, 1, 148, "sm_100a")
    # The shortening condition: 3 pairs of 1 tile on 2 SMs (two waves) cannot be shortened
    # (the longest CTA walks 2 tiles either way); 3 pairs of 4 tiles can (6 < 8).
    assert not cake_jit.walk_applies(3, 1, 1, 1, 2, "sm_103a")
    assert cake_jit.walk_applies(3, 1, 4, 1, 2, "sm_103a")
    assert (
        cake_jit.walk_ctas(152, 256, 33) == 152 and cake_jit.walk_ctas(152, 2, 4) == 8
    )
    assert cake_jit.walk_splits(152, 256, 33) == 2
    assert cake_jit.walk_splits(148, 256, 33) == 2
    # 5 CTAs over 2 x 33 slots: 13-14 slots each, both pairs meet three CTAs; one CTA holds
    # everything in one slot; 66 CTAs over 66 slots hold one tile each: still one segment
    # per pair?  No -- each pair's 33 tiles sit in 33 CTAs.
    assert cake_jit.walk_splits(5, 2, 33) == 3
    assert cake_jit.walk_splits(1, 4, 33) == 1
    assert cake_jit.walk_splits(66, 2, 33) == 33
    for bad in ((0, 2, 33), (67, 2, 33)):
        with pytest.raises(ValueError):
            cake_jit.walk_splits(*bad)
    for bad in (
        (0, 2, tiles, 152, "sm_103a"),
        (1, 0, tiles, 152, "sm_103a"),
        (1, 2, 0, 152, "sm_103a"),
        (1, 2, tiles, 0, "sm_103a"),
        (1, 2, tiles, 152, "sm_999a"),
    ):
        with pytest.raises(ValueError):
            cake_backend.select_walk(*bad)
    grid = cake_jit.launch_grid
    assert grid(
        "decode_tcgen05_persistent",
        rows=128,
        kv_heads=2,
        qo_heads=24,
        splits=2,
        walk=152,
    ) == (152, 1, 1)
    with pytest.raises(ValueError, match="CTA count"):
        grid("decode_tcgen05_persistent", rows=128, kv_heads=2, qo_heads=24, splits=2)
    assert grid("merge", rows=128, kv_heads=2, qo_heads=24, splits=2, walk=152) == (
        128,
        6,
        1,
    )
    assert grid("decode_tcgen05", rows=128, kv_heads=2, qo_heads=24, splits=1) == (
        128,
        2,
        1,
    )


def test_wide_grid_is_multi_wave_with_long_streams():
    """``select_wide`` on sm_80 at 108 SMs (216 resident CTAs) over the owner's rows at two
    KV heads with the host's split counts and the 33 tiles of a 2051-token selection: rows
    128 (2 splits, 512 CTAs, 17 tiles per CTA) are wide; rows 64 (4 splits, 512 CTAs, 9
    tiles per CTA) and rows 1-32 (at most one wave) are not; the 2048-row single split (33
    tiles per CTA) is; the fused form never is.  The grid bound is strict and the stream
    bound inclusive: 16 tiles per CTA (1024 selected tokens) above one wave is wide, 15 is
    not.  The wave is the architecture's: on sm_100a at 148 SMs (one CTA per SM) the rows
    128 single-split grid of 256 CTAs is wide too, which selects nothing there (the tcgen05
    decode has one split-form record).  On sm_120a the bound is the stream alone
    (``long_stream``): at 188 SMs (376 resident CTAs) rows 32 (3 splits, 11 tiles per CTA),
    64 and 128 (1 split, 33 tiles) are wide within one wave, rows 1-16 (at most 7
    tiles per CTA) are not; the stream bound is inclusive at 11 tiles."""
    sm_count, arch = 108, "sm_80"
    tiles = cake_backend.num_tiles(2051)
    assert tiles == 33
    assert cake_jit.DECODE_LONG_STREAM_TILES == 16
    assert cake_backend.LONG_STREAM_TILES == cake_jit.DECODE_LONG_STREAM_TILES
    assert cake_jit.DECODE_STREAM_TILES == 11
    assert cake_backend.STREAM_TILES == cake_jit.DECODE_STREAM_TILES
    assert cake_backend.WIDE_BOUNDS == {
        "sm_80": "multi_wave_long",
        "sm_120a": "long_stream",
    }
    assert cake_backend.CTAS_PER_SM == {
        "sm_80": 2,
        "sm_90a": 2,
        "sm_100a": 1,
        "sm_103a": 1,
        "sm_107a": 1,
        "sm_120a": 2,
    }
    rows_to_splits = {
        rows: cake_backend.num_splits(rows, 2, 2051, sm_count, arch)
        for rows in (1, 2, 4, 8, 16, 32, 64, 128)
    }
    assert rows_to_splits[64] == 4 and rows_to_splits[128] == 2
    wide = cake_backend.select_wide
    assert wide(128, 2, rows_to_splits[128], tiles, "split", sm_count, arch)
    assert not wide(128, 2, rows_to_splits[128], tiles, "fused", sm_count, arch)
    # Rows 64: 512 CTAs exceed one wave, but ceil(33 / 4) = 9 tiles per CTA is a short stream.
    assert not wide(64, 2, rows_to_splits[64], tiles, "split", sm_count, arch)
    for rows in (1, 2, 4, 8, 16, 32):
        splits = rows_to_splits[rows]
        assert rows * 2 * splits <= cake_backend.CTAS_PER_SM[arch] * sm_count, rows
        assert not wide(rows, 2, splits, tiles, "split", sm_count, arch), rows
    assert wide(2048, 2, 1, tiles, "split", sm_count, arch)
    assert not wide(2048, 2, 1, tiles, "fused", sm_count, arch)
    # The grid bound is strict: one full wave of long streams is not wide.
    capacity = cake_backend.CTAS_PER_SM[arch] * sm_count
    assert not wide(capacity, 1, 1, 16, "split", sm_count, arch)
    assert wide(capacity + 1, 1, 1, 16, "split", sm_count, arch)
    # The stream bound is inclusive and counts ceil(tiles / splits).
    assert wide(512, 1, 1, 16, "split", sm_count, arch)
    assert not wide(512, 1, 1, 15, "split", sm_count, arch)
    assert wide(128, 2, 2, 31, "split", sm_count, arch)  # ceil(31 / 2) = 16
    assert not wide(128, 2, 2, 30, "split", sm_count, arch)  # 15
    # On 132 SMs (264 CTAs) the owner's rows 1-128 never exceed one wave in the split form.
    for rows in (1, 2, 4, 8, 16, 32, 64, 128):
        splits = cake_backend.num_splits(rows, 2, 2051, 132, "sm_90a")
        assert not wide(rows, 2, splits, tiles, "split", 132, "sm_90a"), rows
    # sm_100a at 148 SMs: the wave is 148 CTAs; rows 128 at one split (256 CTAs, 33 tiles) is wide, rows 1-64 are not.
    for rows in (1, 2, 4, 8, 16, 32, 64):
        splits = cake_backend.num_splits(rows, 2, 2051, 148, "sm_100a")
        assert not wide(rows, 2, splits, tiles, "split", 148, "sm_100a"), rows
    assert cake_backend.num_splits(128, 2, 2051, 148, "sm_100a") == 1
    assert wide(128, 2, 1, tiles, "split", 148, "sm_100a")
    assert cake_jit.select_records(
        form="split", splits=1, arch="sm_100a", wide=True
    ) == ("decode_tcgen05",)
    # sm_120a at 188 SMs: the stream alone decides; every owner grid is within one wave of 376.
    splits_188 = {
        rows: cake_backend.num_splits(rows, 2, 2051, 188, "sm_120a")
        for rows in (1, 2, 4, 8, 16, 32, 64, 128)
    }
    assert [splits_188[rows] for rows in (16, 32, 64, 128)] == [5, 3, 1, 1]
    for rows in (1, 2, 4, 8, 16):
        splits = splits_188[rows]
        assert -(-tiles // splits) < 11, rows
        assert not wide(rows, 2, splits, tiles, "split", 188, "sm_120a"), rows
    for rows in (32, 64, 128):
        splits = splits_188[rows]
        assert rows * 2 * splits <= 376, rows
        assert wide(rows, 2, splits, tiles, "split", 188, "sm_120a"), rows
        assert not wide(rows, 2, splits, tiles, "fused", 188, "sm_120a"), rows
    assert wide(2048, 2, 1, tiles, "split", 188, "sm_120a")
    # The stream bound is inclusive at 11 tiles per CTA and reads ceil(tiles / splits).
    assert wide(1, 1, 1, 11, "split", 188, "sm_120a")
    assert not wide(1, 1, 1, 10, "split", 188, "sm_120a")
    assert wide(64, 2, 2, 21, "split", 188, "sm_120a")  # ceil(21 / 2) = 11
    assert not wide(64, 2, 2, 20, "split", 188, "sm_120a")  # 10
    assert cake_jit.select_records(
        form="split", splits=3, arch="sm_120a", wide=True
    ) == ("decode_whole_tile", "merge")
    for bad in (
        (0, 2, 1, tiles, "split", sm_count, arch),
        (1, 0, 1, tiles, "split", sm_count, arch),
        (1, 2, 0, tiles, "split", sm_count, arch),
        (1, 2, 1, 0, "split", sm_count, arch),
        (1, 2, 1, tiles, "split", 0, arch),
        (1, 2, 1, tiles, "other", sm_count, arch),
        (1, 2, 1, tiles, "split", sm_count, "sm_121a"),
    ):
        with pytest.raises(ValueError):
            wide(*bad)


def _operands(device):
    """A small random fixture: 4 rows of 24 x 2 heads over 40 blocks of 16 tokens, two
    requests of 16 pages, 200 distinct selections per row (4 tiles); random E2M1 codes
    and finite E4M3 block scales from {0.5, 0.75, 1, 1.5, 2}."""
    rows, num_qo_heads, num_kv_heads = 4, 24, 2
    blocks, page, table_width, width = 40, 16, 16, 200
    gen = torch.Generator(device=device).manual_seed(3)
    q, gate = (
        torch.randn(
            rows, num_qo_heads, HEAD_DIM, device=device, generator=gen
        ).bfloat16()
        for _ in range(2)
    )
    data = tuple(
        torch.randint(
            0,
            256,
            (blocks, page, num_kv_heads, HEAD_DIM // 2),
            dtype=torch.int32,
            device=device,
            generator=gen,
        ).to(torch.uint8)
        for _ in range(2)
    )
    sf_bytes = torch.tensor(
        [0x30, 0x34, 0x38, 0x3C, 0x40], dtype=torch.uint8, device=device
    )
    sf = tuple(
        sf_bytes[
            torch.randint(
                0,
                5,
                (blocks, page, num_kv_heads, HEAD_DIM // 16),
                device=device,
                generator=gen,
            )
        ].view(torch.float8_e4m3fn)
        for _ in range(2)
    )
    table = torch.randperm(blocks, device=device, generator=gen)[: 2 * table_width]
    table = table.view(2, table_width).int()
    token_to_req = torch.tensor([0, 1, 1, 0], dtype=torch.int32, device=device)
    indices = torch.full((rows, width + 1), -1, dtype=torch.int32, device=device)
    for row in range(rows):
        tokens = torch.randperm(table_width * page, device=device, generator=gen)
        indices[row, :width] = tokens[:width].int()
    indices[:, width] = width
    return dict(
        q=q,
        k_cache=data[0],
        v_cache=data[1],
        logical_indices=indices,
        block_table=table,
        token_to_req=token_to_req,
        use_prefill_config=False,
        k_scale=0.05,
        v_scale=0.05,
        output_gate=gate,
        k_sf=sf[0],
        v_sf=sf[1],
    )


def test_cake_route_runs_the_package():
    device = _device_or_skip()
    operands = _operands(device)
    want = cake_backend.qsa_sparse_paged_attention(**operands)
    with _quiet():
        got = public_api.qsa_sparse_paged_attention(**operands, backend="cake")
    torch.cuda.synchronize()
    assert torch.isfinite(got.float()).all() and got.any()
    assert torch.equal(got, want)
    out = torch.empty_like(want)
    with _quiet():
        result = public_api.qsa_sparse_paged_attention(
            **operands, out=out, backend="cake"
        )
    assert result is out
    assert torch.equal(out, want)


def test_auto_route_is_the_cake_route():
    """Without ``backend`` the public entry runs the package on an exported capability above
    9.0: the result is bitwise the explicit ``backend="cake"`` result."""
    device = _device_or_skip()
    if torch.cuda.get_device_capability(device)[0] <= 9:
        pytest.skip("auto runs the existing implementation on 8.x and 9.0")
    operands = _operands(device)
    with _quiet():
        want = public_api.qsa_sparse_paged_attention(**operands, backend="cake")
        got = public_api.qsa_sparse_paged_attention(**operands)
    torch.cuda.synchronize()
    assert torch.isfinite(got.float()).all() and got.any()
    assert torch.equal(got, want)


def test_writer_format_is_the_port_the_backend_suite_uses():
    """The production NVFP4 paged writer and the backend suite's torch port of it write the
    same format: equal block-scale bytes, equal signs, and codes that agree or differ by
    one ladder step.  The writer's reciprocal of the scale is nvcc's fast-math one (a unit
    low at times), so a product exactly at a ladder midpoint rounds down in the writer
    where the port rounds to even; a layout, nibble-order or scale mismatch would move
    most codes by more than one step.  The writer's cache then runs through the Cake
    route."""
    device = _device_or_skip()
    blocks, page, num_kv_heads = 40, 16, 2
    slots = blocks * page
    gen = torch.Generator(device=device).manual_seed(7)
    k, v = (
        torch.randn(slots, num_kv_heads, HEAD_DIM, device=device, generator=gen)
        .mul(scale)
        .bfloat16()
        for scale in (1.0, 3.0)
    )
    scales = tuple(x.abs().max().item() / (6 * 448) for x in (k, v))
    caches = []
    for layout in ("NHD", "HND"):
        dims = (num_kv_heads, page) if layout == "HND" else (page, num_kv_heads)
        for writer in ("flashinfer", "port"):
            data = tuple(
                torch.zeros(
                    blocks, *dims, HEAD_DIM // 2, dtype=torch.uint8, device=device
                )
                for _ in range(2)
            )
            sf = tuple(
                torch.zeros(
                    blocks,
                    *dims,
                    HEAD_DIM // 16,
                    dtype=torch.float8_e4m3fn,
                    device=device,
                )
                for _ in range(2)
            )
            if writer == "flashinfer":
                flashinfer.nvfp4_quantize_append_paged_kv_cache_with_slot_mapping(
                    k,
                    v,
                    torch.arange(slots, device=device),
                    data,
                    sf,
                    *scales,
                    kv_layout=layout,
                )
            else:
                write_slots(k, v, data, sf, *scales, kv_layout=layout)
            caches.append((layout, writer, data, sf))
        (_, _, data_fi, sf_fi), (_, _, data_port, sf_port) = caches[-2:]
        torch.cuda.synchronize()
        for side in range(2):
            sf_bytes = tuple(x[side].view(torch.uint8) for x in (sf_fi, sf_port))
            assert torch.equal(*sf_bytes), (layout, side, "block scales")
            for shift in (0, 4):
                code_fi = (data_fi[side] >> shift) & 15
                code_port = (data_port[side] >> shift) & 15
                assert torch.equal(code_fi >> 3, code_port >> 3), (layout, side, "sign")
                step = ((code_fi & 7).int() - (code_port & 7).int()).abs()
                assert step.max().item() <= 1, (layout, side, shift, step.max().item())
                differing = (step != 0).sum().item()
                assert differing <= step.numel() * 1e-2, (layout, side, differing)
    # The writer's bytes through the Cake route, in the NHD layout.
    _, _, data, sf = caches[0]
    rows, num_qo_heads, width, table_width = 4, 24, 200, 16
    q, gate = (
        torch.randn(
            rows, num_qo_heads, HEAD_DIM, device=device, generator=gen
        ).bfloat16()
        for _ in range(2)
    )
    table = torch.randperm(blocks, device=device, generator=gen)[: 2 * table_width]
    table = table.view(2, table_width).int()
    indices = torch.full((rows, width + 1), -1, dtype=torch.int32, device=device)
    for row in range(rows):
        tokens = torch.randperm(table_width * page, device=device, generator=gen)
        indices[row, :width] = tokens[:width].int()
    indices[:, width] = width
    with _quiet():
        out = public_api.qsa_sparse_paged_attention(
            q,
            data[0],
            data[1],
            indices,
            table,
            torch.tensor([0, 1, 1, 0], dtype=torch.int32, device=device),
            False,
            None,
            *scales,
            output_gate=gate,
            k_sf=sf[0],
            v_sf=sf[1],
            backend="cake",
        )
    torch.cuda.synchronize()
    assert torch.isfinite(out.float()).all() and out.any()
