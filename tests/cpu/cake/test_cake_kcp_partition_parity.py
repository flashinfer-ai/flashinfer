# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team
"""The four KCP packages build the same partition and the same base context.

The deterministic and the non-deterministic KDA and GDN KCP packages each keep
their own copy of the partition and of the context builder (#8928): there is no
shared module, because each package must vendor on its own.  This test catches
drift between the four copies.  For seeded inputs of every layout it builds the
partition and the context of every rank in each package, and compares:

- the partition: ``to_dict()``, the fragments, each rank's local
  ``cu_seqlens``, the crossing counts, the document chains and the longest
  chain (the layouts, cut points, ``zigzag_per_document`` and explicit fragment
  tables);
- the base context fields of :data:`CONTEXT_FIELDS` (crossing slots, chains,
  chunk geometry and tables, local ``cu_seqlens``, the halo plan and tables, the
  fragment table, ``conv_kernel_size``, the ``sm_count`` default), tensors by
  dtype, shape and values, and ``seg_chunk_start`` over the context's chunk
  table;
- the error type and message (each package's own name masked) of invalid
  partition and context inputs, and the process-group check
  (``torch.distributed`` stubbed).

Not compared: the segment plan, the ``P`` rule, the wave rule, the head plan,
the byte queries and the arena layouts.  These differ between the packages by
design.

The packages and their entries carry their final #8928 names: the package of
``(op, mode)`` is the one row of PACKAGES whose import path ends in
``cake_<op>_<mode>_train_kcp``, and its entries are
``cake_<op>_<mode>_train_kcp_partition`` and
``cake_<op>_<mode>_train_kcp_build_context`` (:func:`_kcp_package` and
:func:`_entry`).  A missing package or entry, a KDA or GDN KCP row or entry
under another name, or a package that does not import fails the test; nothing
is skipped.

CI runs this file in the unit-test run over ``tests/``.  CPU only: no GPU, no
process group and no kernel.
"""

from __future__ import annotations

import collections.abc
import dataclasses
import functools
import inspect
import itertools
import random
import re
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest
import torch

from tests.cake.test_helpers.cake_package import PACKAGES, cake_package

pytestmark = pytest.mark.cpu_only

FAMILIES = (("kda", "det"), ("kda", "nondet"), ("gdn", "det"), ("gdn", "nondet"))

# A row of PACKAGES is the KCP package of (op, mode) when the last part of its
# import path is ``cake_<op>_<mode>_train_kcp`` (#8928 names, for example
# ``flashinfer.experimental.cake.cake_gdn_nondet_train_kcp``).  The key of the
# row does not matter.  A KDA or GDN KCP row under any other name (the
# pre-#8928 ``cake_<op>_train_kcp``, for example) fails the test.
_KCP_PACKAGE = re.compile(r"cake_(kda|gdn)_(det|nondet)_train_kcp")
_KCP_PACKAGE_LIKE = re.compile(r"cake_(?:kda|gdn)\w*_kcp\w*")

# The base context: what a context derives from its partition, the same in the
# four packages.  The p_rule, the head plans and each package's upload state
# are not compared.
CONTEXT_FIELDS = (
    "rank",
    "world_size",
    "device",
    "conv_kernel_size",
    "sm_count",
    "T_r",
    "n_frag",
    "cu_seqlens_local_cpu",
    "cu_seqlens_local",
    "frag_table",
    "in_slot",
    "out_slot",
    "n_in",
    "n_out",
    "N_in",
    "N_out",
    "n_in_by_rank",
    "n_out_by_rank",
    "in_slot_all",
    "out_slot_all",
    "chains",
    "n_d_max",
    "exchange_scheme",
    "NC",
    "NC_max",
    "NC_pad",
    "R",
    "chunk_bos",
    "chunk_len",
    "seq_chunk_start",
    "seq_order",
    "tables",
    "halo",
    "halo_tables",
)
SEGMENTS_PER_SEQUENCE = (1, 2, 3, 8)  # the P of the seg_chunk_start checks

LAYOUT_CASES = (
    "contiguous",
    "zigzag",
    "zigzag_per_document",
    "cut_points",  # zigzag_per_document with random cut points
    "explicit",  # an explicit fragment table, one list per rank
    "explicit_flat",  # an explicit fragment table, (doc, off, len, rank) entries
)
CASES_PER_LAYOUT = 16


def _package_name(op: str, mode: str) -> re.Pattern[str]:
    """The package's own name in an error message, and nothing else: its snake-case
    prefix ``cake_<op>_<mode>_train_kcp`` (``cake_gdn_nondet_train_kcp_build_context``)
    and its class prefix ``Cake<OP><Mode>Kcp`` (``CakeKDADetKcpPartition``)."""
    snake = f"cake_{op}_{mode}_train_kcp"
    camel = f"Cake{op.upper()}{mode.capitalize()}Kcp"
    return re.compile(
        rf"(?<![A-Za-z0-9_])(?:{snake}(?![A-Za-z0-9])|{camel}(?![a-z0-9]))"
    )


@dataclasses.dataclass(frozen=True)
class Kcp:
    """The plan module of one KCP package and its partition and build-context entries."""

    path: str
    plan: ModuleType
    partition: Callable[..., Any]
    build_context: Callable[..., Any]
    name: re.Pattern[str]  # the package's own name in an error message


def _kcp_rows() -> tuple[dict[tuple[str, str], list[str]], list[str]]:
    """The keys of the KCP rows of PACKAGES by ``(op, mode)``, and the import paths
    of the KDA and GDN KCP rows under any other name."""
    found: dict[tuple[str, str], list[str]] = {}
    other = []
    for key, path in PACKAGES.items():
        last = path.rsplit(".", 1)[-1]
        match = _KCP_PACKAGE.fullmatch(last)
        if match:
            found.setdefault((match[1], match[2]), []).append(key)
        elif _KCP_PACKAGE_LIKE.fullmatch(last):
            other.append(path)
    return found, other


def _kcp_package(op: str, mode: str) -> str:
    """The key of the one row of PACKAGES that holds the KCP package of ``(op, mode)``."""
    keys = _kcp_rows()[0].get((op, mode), [])
    if len(keys) != 1:
        pytest.fail(
            f"PACKAGES must hold one row for the {op} {mode} KCP package (import "
            f"path ending in cake_{op}_{mode}_train_kcp); found {keys}",
            pytrace=False,
        )
    return keys[0]


def _entry(plan: ModuleType, op: str, mode: str, what: str) -> Callable[..., Any]:
    """``cake_<op>_<mode>_train_kcp_<what>`` of ``plan``: a public function of the
    module itself, and the module's only public function whose name is ``what``
    plus package-name words only (no pre-#8928 name such as ``partition_det_kcp``,
    no alias).  A helper that names ``what`` alongside other words, such as
    ``kcp_partition_digest``, is a different function and is allowed."""
    name = f"cake_{op}_{mode}_train_kcp_{what}"
    words = tuple(what.split("_"))
    package_words = {"cake", "kda", "gdn", "det", "nondet", "train", "kcp"}
    value = vars(plan).get(name)
    if not inspect.isfunction(value) or value.__module__ != plan.__name__:
        pytest.fail(
            f"{plan.__name__}: expected the public function {name}, found {value!r}",
            pytrace=False,
        )
    others = []
    for other, obj in vars(plan).items():
        parts = other.split("_")
        if (
            other != name
            and not other.startswith("_")
            and inspect.isfunction(obj)
            and "kcp" in parts
            and any(
                tuple(parts[i : i + len(words)]) == words
                and set(parts[:i] + parts[i + len(words) :]) <= package_words
                for i in range(len(parts))
            )
        ):
            others.append(other)
    if others:
        pytest.fail(
            f"{plan.__name__}: {name} must be the only public KCP {what} function; "
            f"also found {sorted(others)}",
            pytrace=False,
        )
    return value


@functools.cache
def _kcp(op: str, mode: str) -> Kcp:
    key = _kcp_package(op, mode)
    plan = cake_package(key, "cake_kcp_plan")
    return Kcp(
        path=PACKAGES[key],
        plan=plan,
        partition=_entry(plan, op, mode, "partition"),
        build_context=_entry(plan, op, mode, "build_context"),
        name=_package_name(op, mode),
    )


@pytest.fixture(scope="module")
def kcps() -> dict[str, Kcp]:
    return {f"{op} {mode}": _kcp(op, mode) for op, mode in FAMILIES}


def _plain(x: Any) -> Any:
    """``x`` as comparable plain values: tensors by dtype, shape and values, dataclasses by field."""
    if isinstance(x, torch.Tensor):
        return {"dtype": str(x.dtype), "shape": list(x.shape), "values": x.tolist()}
    if isinstance(x, torch.device):
        return str(x)
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {f.name: _plain(getattr(x, f.name)) for f in dataclasses.fields(x)}
    if isinstance(x, collections.abc.Mapping):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x


def _partition_record(part: Any) -> Any:
    ranks = range(part.world_size)
    n_in, n_out = part.crossing_counts()
    return _plain(
        {
            "to_dict": part.to_dict(),
            "fragments": part.fragments,
            "local_cu_seqlens": [part.local_cu_seqlens(r) for r in ranks],
            "crossing_in": n_in,
            "crossing_out": n_out,
            "chains": part.chains(),
            "max_chain_fragments": part.max_chain_fragments(),
        }
    )


def _context_record(kcp: Kcp, ctx: Any) -> dict[str, Any]:
    record = {name: _plain(getattr(ctx, name)) for name in CONTEXT_FIELDS}
    for P in SEGMENTS_PER_SEQUENCE:
        record[f"seg_chunk_start(P={P})"] = kcp.plan.seg_chunk_start(
            list(ctx.seq_chunk_start), P
        )
    return record


def _outcome(kcp: Kcp, call: Callable[[], Any]) -> dict[str, Any]:
    """The result of ``call``, or its error type and message with the package's own name masked."""
    try:
        return {"result": call()}
    except Exception as e:
        return {
            "error": type(e).__name__,
            "message": kcp.name.sub("<package>", str(e)),
        }


def _assert_same(what: str, records: dict[str, Any]) -> None:
    """Every package's record equals the first package's."""
    (first, expected), *others = records.items()
    differ = []
    for name, record in others:
        if record != expected:
            keys = (
                sorted(
                    k
                    for k in expected.keys() | record.keys()
                    if expected.get(k) != record.get(k)
                )
                if isinstance(record, dict) and isinstance(expected, dict)
                else None
            )
            differ.append(f"{name} (keys {keys})")
    if differ:
        pytest.fail(f"{what}: {', '.join(differ)} differ from {first}", pytrace=False)


@dataclasses.dataclass(frozen=True)
class Case:
    cu_seqlens: Any
    world_size: int
    layout: str
    fragments: list | None
    cut_points: list | None
    conv_kernel_size: int
    sm_count: int | None


def _length(rng: random.Random) -> int:
    u = rng.random()
    if u < 0.1:
        return 0
    if u < 0.4:
        return rng.randint(1, 300)
    if u < 0.85:
        return rng.randint(300, 9000)
    return rng.randint(9000, 40000)


def _case(kind: str, seed: int) -> Case:
    """A valid partition input of layout case ``kind``, fixed by ``seed``."""
    rng = random.Random(f"{kind}/{seed}")
    W = rng.choice((1, 2, 3, 4, 4, 8, 8, 16))
    lengths = [_length(rng) for _ in range(rng.randint(1, 8))]
    cu = [0, *itertools.accumulate(lengths)]
    layout, fragments, cut_points = kind, None, None
    if kind == "cut_points":
        layout = "zigzag_per_document"
        cut_points = [
            [0, *sorted(rng.randint(0, n) for _ in range(2 * W - 1)), n]
            for n in lengths
        ]
    elif kind.startswith("explicit"):
        layout = "contiguous"  # not read when fragments are given
        entries = []
        for doc, n in enumerate(lengths):
            k = rng.randint(1, W + 2)
            cuts = [0, *sorted(rng.randint(0, n) for _ in range(k - 1)), n]
            entries += [
                (doc, cuts[i], cuts[i + 1] - cuts[i], rng.randrange(W))
                for i in range(k)
            ]
        entries.sort(key=lambda e: (e[0], e[1], e[3]))  # each rank's local order
        if kind == "explicit":
            fragments = [[e[:3] for e in entries if e[3] == r] for r in range(W)]
        else:
            fragments = [list(e) for e in entries]
    return Case(
        cu_seqlens=torch.tensor(cu, dtype=torch.int64) if seed % 2 else cu,
        world_size=W,
        layout=layout,
        fragments=fragments,
        cut_points=cut_points,
        conv_kernel_size=rng.choice((1, 2, 3, 4, 4)),
        sm_count=rng.choice((None, 132, 148, 152, 160)),
    )


def _partition(kcp: Kcp, cu, W, layout="contiguous", fragments=None, cut_points=None):
    return kcp.partition(cu, W, layout, fragments, cut_points=cut_points)


def test_the_four_kcp_packages_and_no_other_are_found() -> None:
    """PACKAGES holds exactly one row for each of the four KCP packages, under its
    #8928 name, and no KDA or GDN KCP row under another name."""
    found, other = _kcp_rows()
    assert not other, (
        f"KDA / GDN KCP rows not named cake_<op>_<mode>_train_kcp: {other}"
    )
    assert {k: len(v) for k, v in found.items()} == dict.fromkeys(FAMILIES, 1), found


@pytest.mark.parametrize("op, mode", FAMILIES, ids=[f"{o}-{m}" for o, m in FAMILIES])
def test_kcp_package_and_entries_are_found(op: str, mode: str) -> None:
    kcp = _kcp(op, mode)
    part = _partition(kcp, [0, 100, 300], 2)
    ctx = kcp.build_context(part, None, 1, "cpu")
    assert ctx.partition is part and (ctx.rank, ctx.world_size) == (1, 2), kcp.path


@pytest.mark.parametrize("seed", range(CASES_PER_LAYOUT))
@pytest.mark.parametrize("kind", LAYOUT_CASES)
def test_partition_and_base_context_match(
    kcps: dict[str, Kcp], kind: str, seed: int
) -> None:
    case = _case(kind, seed)
    parts = {
        name: _partition(
            kcp,
            case.cu_seqlens,
            case.world_size,
            case.layout,
            case.fragments,
            case.cut_points,
        )
        for name, kcp in kcps.items()
    }
    _assert_same("partition", {name: _partition_record(p) for name, p in parts.items()})
    for rank in range(case.world_size):
        records = {}
        for name, kcp in kcps.items():
            ctx = kcp.build_context(
                parts[name],
                None,
                rank,
                "cpu",
                case.conv_kernel_size,
                sm_count=case.sm_count,
            )
            records[name] = _context_record(kcp, ctx)
        _assert_same(f"rank {rank} context", records)


INVALID_PARTITIONS = {
    "cu_seqlens_not_from_0": ([1, 50, 90], 2, {}),
    "cu_seqlens_decreasing": ([0, 50, 40], 2, {}),
    "cu_seqlens_2d": (torch.zeros(2, 3, dtype=torch.int64), 2, {}),
    "cu_seqlens_float": (torch.tensor([0.0, 50.0]), 2, {}),
    "world_size_0": ([0, 50], 0, {}),
    "unknown_layout": ([0, 50], 2, {"layout": "striped"}),
    "cut_points_contiguous": ([0, 50], 2, {"cut_points": [[0, 10, 20, 30, 50]]}),
    "cut_points_zigzag": (
        [0, 50],
        2,
        {"layout": "zigzag", "cut_points": [[0, 10, 20, 30, 50]]},
    ),
    "cut_points_per_document": (
        [0, 50, 90],
        2,
        {"layout": "zigzag_per_document", "cut_points": [[0, 10, 20, 30, 50]]},
    ),
    "cut_points_count": (
        [0, 50],
        2,
        {"layout": "zigzag_per_document", "cut_points": [[0, 25, 50]]},
    ),
    "cut_points_decreasing": (
        [0, 50],
        2,
        {"layout": "zigzag_per_document", "cut_points": [[0, 30, 20, 40, 50]]},
    ),
    "fragment_document": ([0, 50], 2, {"fragments": [[(1, 0, 50)], []]}),
    "fragment_rank": ([0, 50], 2, {"fragments": [(0, 0, 50, 2)]}),
    "fragment_past_document_end": ([0, 50], 2, {"fragments": [[(0, 0, 60)], []]}),
    "fragment_gap": ([0, 50], 2, {"fragments": [[(0, 1, 49)], []]}),
    "fragment_overlap": ([0, 50], 2, {"fragments": [[(0, 0, 30)], [(0, 20, 30)]]}),
    "document_not_covered": ([0, 50], 2, {"fragments": [[(0, 0, 30)], []]}),
    "fragments_out_of_order": (
        [0, 50, 90],
        2,
        {"fragments": [[(1, 0, 40), (0, 0, 50)], []]},
    ),
    "nested_entry_arity": ([0, 50], 2, {"fragments": [[(0, 0)], [(0, 0, 50)]]}),
    "flat_entry_arity": ([0, 50], 2, {"fragments": [(0, 0, 50)]}),
}


@pytest.mark.parametrize("case", INVALID_PARTITIONS.keys())
def test_invalid_partition_inputs_fail_alike(kcps: dict[str, Kcp], case: str) -> None:
    cu, W, kwargs = INVALID_PARTITIONS[case]
    outcomes = {
        name: _outcome(
            kcp, lambda kcp=kcp: _partition_record(_partition(kcp, cu, W, **kwargs))
        )
        for name, kcp in kcps.items()
    }
    assert all("error" in o for o in outcomes.values()), outcomes
    _assert_same("error", outcomes)


INVALID_CONTEXTS = {
    "rank_is_world_size": {"rank": 4},
    "rank_negative": {"rank": -1},
    "conv_kernel_size_0": {"conv_kernel_size": 0},
    "sm_count_0": {"sm_count": 0},
    "not_a_partition": {"part": None},
}


@pytest.mark.parametrize("case", INVALID_CONTEXTS.keys())
def test_invalid_context_inputs_fail_alike(kcps: dict[str, Kcp], case: str) -> None:
    def build(kcp: Kcp) -> dict[str, Any]:
        args = {"rank": 1, "conv_kernel_size": 4, "sm_count": None}
        args.update(INVALID_CONTEXTS[case])
        part = args.pop("part", _partition(kcp, [0, 300, 700], 4, "zigzag"))
        ctx = kcp.build_context(
            part,
            None,
            args["rank"],
            "cpu",
            args["conv_kernel_size"],
            sm_count=args["sm_count"],
        )
        return _context_record(kcp, ctx)

    outcomes = {
        name: _outcome(kcp, lambda kcp=kcp: build(kcp)) for name, kcp in kcps.items()
    }
    assert all("error" in o for o in outcomes.values()), outcomes
    _assert_same("error", outcomes)


@pytest.mark.parametrize(
    "group_size, group_rank, ok",
    [(4, 1, True), (2, 1, False), (4, 3, False)],
    ids=["matches", "world_size_differs", "rank_differs"],
)
def test_process_group_check_matches(
    kcps: dict[str, Kcp], monkeypatch, group_size: int, group_rank: int, ok: bool
) -> None:
    """With a group and an initialized default group, every package checks the
    group's size and rank against the partition and the call (rank 1 of 4)."""
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.distributed, "get_world_size", lambda group=None: group_size
    )
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: group_rank)
    group = object()

    def build(kcp: Kcp) -> dict[str, Any]:
        part = _partition(kcp, [0, 300, 700], 4, "zigzag")
        return _context_record(kcp, kcp.build_context(part, group, 1, "cpu"))

    outcomes = {
        name: _outcome(kcp, lambda kcp=kcp: build(kcp)) for name, kcp in kcps.items()
    }
    assert all(("error" not in o) == ok for o in outcomes.values()), outcomes
    _assert_same("process-group check", outcomes)
