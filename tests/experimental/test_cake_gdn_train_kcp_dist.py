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

Multi-GPU rows of the context-parallel (KCP) packed GDN training route (``cake_gdn_train.cake_kcp``): ``W`` ranks over
NCCL, each holding the fragments of one partition of a packed batch, compared with the exact FP64 per-document
recurrence of the same operands and with one another.  Per row and rank the test checks

* vs FP64 (rank 0, the outputs reassembled to the global order): ``out, dq, dk, dv`` as stored BF16, ``dg, dbeta`` FP32;
  the correctness rows (regime ``normal``) by the elementwise clause ``|cand - ref| <= 1e-2 + 1e-2 |ref|`` with zero
  mismatches on every tensor; the accuracy rows (regimes A1-A4) by the per-(row, CP, tensor) budgets of
  ``cake_gdn_train_kcp_g1_budgets.json`` (rel-RMS <= budget and max-abs <= budget + one storage ulp of |ref| at the
  worst element, all six tensors; their elementwise counts are recorded as information); rel-RMS / max-abs recorded
  for every tensor;
* bitwise (G3): repeat; saved == replay (``boundary_mode``); ``save_mode`` light == full; autograd == explicit;
  ``torch.utils.checkpoint(use_reentrant=False)`` == direct; ``save_transfer`` True == False; ring == a2a where the
  partition allows the ring; strided packed-QKV views == contiguous operands (``layout`` tma and copy);
  ``.backward()`` twice accumulates exactly twice the gradient;
* semantics (G4): zero-gradient leak (``dout`` zero outside one document -> exactly zero gradients on the others) and
  independence (the other documents' operands replaced -> one document's outputs bitwise unchanged);
* contracts (G10): #7039 retained bytes after a checkpointed forward (``graph="off"``), the exchange byte counts against
  ``gdn_kcp_comm_bytes``, CUDA-graph capture of forward + backward (no steady-state D2H / sync), the #7038 side-stream
  ratio of the backward (asserted for the kernel backend; reported for the torch emulation).

Two launch forms: ``torch.multiprocessing.spawn`` (``pytest`` on a node with >= ``W`` GPUs) and one process per Slurm
task (``SLURM_PROCID`` / ``SLURM_NTASKS``: the WORLD group is initialised once per process, every row runs on a
sub-group of its ``W`` lowest ranks; the other ranks skip the row).  ``KCP_TEST_BACKEND`` selects ``kernels`` |
``fake`` (default: ``kernels`` when the package's ``build.json`` lists the programs, else ``fake``);
``KCP_TEST_ROWS`` selects rows by name (comma list; ``all``) or by stage (``int1`` CP <= 4, ``int3`` CP8, ``int4`` CP16);
``KCP_TEST_RESULT_DIR`` receives one JSON per row.
Operands come from the seeded CPU generator (bitwise identical on every rank and host).
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import gc
import importlib
import json
import math
import os
import socket
import statistics
import tempfile
import time
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

PACKAGE = "flashinfer.experimental.cake_gdn_train"


def _mod(sub: str | None = None):
    try:
        from tests.test_helpers.cake_package import cake_package

        return cake_package("cake_gdn_train", sub)
    except (ImportError, KeyError):
        return importlib.import_module(PACKAGE if sub is None else f"{PACKAGE}.{sub}")


kcp = _mod("cake_kcp")
plan = _mod("cake_kcp_plan")
PACKAGE_DIR = Path(kcp.__file__).resolve().parent
TENSORS = ("out", "dq", "dk", "dv", "dg", "dbeta")
GRADS = (
    "dq",
    "dk",
    "dv",
    "dg",
    "dbeta",
)  # autograd order of the leaves (q, k, v, g, beta)
ATOL = RTOL = 1e-2
STATE_REL_RMS_MAX = 1e-2  # forward-only mode: S_in (FP32, owner-merged) vs the FP64 cut state, per crossing-in fragment
WORLD_TIMEOUT_S = 300  # NCCL / rendezvous timeout of the WORLD group (printed at init)
SCALE = kcp.SCALE
C_CONV = 10240
ALLOC_GRANULE = 512  # the caching allocator rounds every block to 512 B
SIDE_STREAM_RATIO_MAX = 1.10  # #7038; the per-row ratio is recorded, never a gate (the module-timing protocol decides)
SIDE_STREAM_ITERS = 7


def _reference():
    """The FP64 reference: the FlashInfer test helper once it ships, else the session harness on PYTHONPATH."""
    try:
        from tests.test_helpers import cake_gdn_train_reference as ref  # type: ignore[attr-defined]

        return ref
    except ImportError:
        return importlib.import_module("gdn_reference")


def _inputs():
    """The seeded operand generator (``make_inputs(cu, H_qk, HV, *, regime, seed, token_range)``)."""
    try:
        from tests.test_helpers import cake_gdn_train_inputs as gen  # type: ignore[attr-defined]

        return gen
    except ImportError:
        return importlib.import_module("gdn_inputs")


BUDGETS_FILE = Path(__file__).resolve().parent / "cake_gdn_train_kcp_g1_budgets.json"
STORAGE = {
    "out": "bf16",
    "dq": "bf16",
    "dk": "bf16",
    "dv": "bf16",
    "dg": "fp32",
    "dbeta": "fp32",
}


def _budgets() -> dict:
    """The accuracy budgets of the multi-rank rows (the file beside this test): per (row, CP, tensor) the rel-RMS and
    max-abs budget = 1.10 x max(single-rank local-path error of this package, FLA tf32x3 error at the FLA layout of
    record), declared before the rows ran and vendored unchanged."""
    with open(BUDGETS_FILE) as fh:
        return json.load(fh)


def _budget_cell(row: Row) -> tuple[dict, int] | None:
    """The budget cell of a row: the row's base name (``A1-cp16`` -> ``A1``) at CP = ``row.W``.  When the table has no
    cell at that CP the row is judged at its nearest CP of record (closest log2 distance, ties -> the lower CP; the CP
    used is recorded in the report): ``A1-cp16`` -> the A1 CP8 cell.  ``None`` when the table has no such row."""
    cells = _budgets()["rows"].get(row.name.split("-")[0])
    if not cells:
        return None
    cps = sorted(int(c) for c in cells)
    if row.W in cps:
        return cells[str(row.W)], row.W
    cp = min(cps, key=lambda c: (abs(math.log2(c / row.W)), c))
    return cells[str(cp)], cp


def _judge(
    report: dict,
    row: Row,
    name: str,
    cand: torch.Tensor,
    ref: torch.Tensor,
    ref_mod,
    budget: tuple[dict, int] | None,
) -> None:
    """Rank-0 verdict of one reassembled tensor against FP64, written to ``report["metrics"][name]`` and folded into
    ``report["pass"]``.  Correctness rows (regime ``normal``): the elementwise clause, zero elements beyond
    ``ATOL + RTOL |ref|``.  Accuracy rows (regimes A1-A4): the row's budget cell (``_budget_cell``): rel-RMS within the
    budget and max-abs within the budget plus one storage ulp of |ref| at the worst element (the reference module's
    ``judge_tensor``), on all six tensors; the elementwise count is recorded as information.  Finiteness gates every
    row; an accuracy row without a budget cell cannot pass."""
    err = cand - ref
    excess = err.abs() - (ATOL + RTOL * ref.abs())
    mism = int((excess > 0).sum())
    elementwise = row.regime == "normal"
    m = {
        "mismatch": mism,
        "rel_rms": float(
            err.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt().clamp_min(1e-300)
        ),
        "max_abs": float(err.abs().max()) if err.numel() else 0.0,
        "ref_rms": float(ref.pow(2).mean().sqrt()) if ref.numel() else 0.0,
        "finite": bool(torch.isfinite(cand).all()),
        "gated": elementwise,  # the elementwise count is the verdict of the correctness rows only
        "verdict": "elementwise" if elementwise else "budget",
    }
    if elementwise:
        ok = m["finite"] and mism == 0
    elif budget is None:
        m["budget"] = None
        ok = False
    else:
        cell, cp = budget
        b = cell["tensors"][name]
        m["budget"] = ref_mod.judge_tensor(
            ref_mod.error_metrics(cand, ref),
            b["budget_rel_rms"],
            b["budget_max_abs"],
            b["storage"],
        )
        m["budget"]["cp_of_cell"] = cp
        ok = m["finite"] and bool(m["budget"]["pass_u1"])
    report["metrics"][name] = m
    report["pass"] &= ok


# ----------------------------------------------------------------------------------------------------------------
# rows (C1-C9 correctness rows, A1-A4 accuracy rows, the no-leak rows; the int4 stage = the CP16 rows of G4: at least
# one correctness row at W = 16, here the C6 zigzag row and the A1 contiguous row at the (2, 6) head pair)


@dataclasses.dataclass(frozen=True)
class Row:
    name: str
    lengths: tuple[int, ...]
    heads: tuple[int, int]  # (H_qk, HV)
    W: int
    layout: str  # contiguous | zigzag | zigzag_per_document | ragged:<T_r list> | explicit:<name>
    regime: str = "normal"
    leak: bool = False  # run the zero-gradient leak / independence checks
    stage: str = "int1"  # int1 (CP <= 4, small), int3 (CP8 / large), int4 (CP16)


def _c5_fragments() -> list[list[tuple[int, int, int]]]:
    """C5: fragments of exactly 63 / 64 / 65 tokens at cut points with offsets ``1`` and ``63 (mod 64)``: document 0 =
    [63, 64, 65, 66] (offsets 0, 63, 127, 192), document 1 = [1, 64, 63, 65] (offsets 0, 1, 65, 128); rank ``r`` holds
    piece ``r`` of both documents."""
    pieces = [(63, 64, 65, 66), (1, 64, 63, 65)]
    frags: list[list[tuple[int, int, int]]] = [[] for _ in range(4)]
    for doc, lens in enumerate(pieces):
        off = 0
        for r, n in enumerate(lens):
            frags[r].append((doc, off, n))
            off += n
    return frags


EXPLICIT = {"c5": (_c5_fragments, (63 + 64 + 65 + 66, 1 + 64 + 63 + 65))}

ROWS = [
    Row("C1-cp2", (100, 640, 77, 1000, 129), (2, 6), 2, "contiguous", leak=True),
    Row("C1-cp4", (100, 640, 77, 1000, 129), (2, 6), 4, "contiguous"),
    Row("C2-C", (3000,), (4, 12), 4, "contiguous"),
    Row("C2-Z", (3000,), (4, 12), 4, "zigzag"),
    Row("C2-D", (3000,), (4, 12), 4, "zigzag_per_document"),
    Row("C3", (2048, 5, 3000, 1), (1, 3), 4, "zigzag_per_document", leak=True),
    Row("C4a", (1000, 2000), (8, 24), 4, "ragged:700,1,1300,999"),
    Row("C4b", (1000, 2000), (8, 24), 4, "ragged:1200,0,900,900"),
    Row("C5", EXPLICIT["c5"][1], (4, 4), 4, "explicit:c5"),
    Row("C9a", (1,), (2, 6), 2, "contiguous"),
    Row("C9b", (0,), (2, 6), 2, "contiguous"),
    Row("leak-C", (300, 500, 200), (2, 6), 2, "contiguous", leak=True),
    Row("leak-D", (300, 500, 200), (2, 6), 2, "zigzag_per_document", leak=True),
    Row(
        "A3",
        (32801, 32769),
        (2, 6),
        4,
        "zigzag_per_document",
        regime="A3",
        stage="int3",
    ),
    Row("C6", (65553,), (16, 48), 8, "zigzag", stage="int3"),
    Row("C8", (10240, 8192, 2085), (16, 48), 8, "contiguous", stage="int3"),
    Row("A1", (32801,), (2, 6), 8, "contiguous", regime="A1", stage="int3"),
    Row("A2", (40000,), (2, 6), 8, "contiguous", regime="A2", stage="int3"),
    Row("A4", (131071,), (2, 6), 8, "contiguous", regime="A4", stage="int3"),
    Row("C6-cp16", (65553,), (16, 48), 16, "zigzag", stage="int4"),
    Row("A1-cp16", (32801,), (2, 6), 16, "contiguous", regime="A1", stage="int4"),
]


def _selected_rows() -> list[Row]:
    sel = os.environ.get("KCP_TEST_ROWS", "int1")
    if sel == "all":
        return list(ROWS)
    if sel in ("int1", "int3", "int4"):
        return [r for r in ROWS if r.stage == sel]
    names = {s.strip() for s in sel.split(",") if s.strip()}
    rows = [r for r in ROWS if r.name in names]
    missing = names - {r.name for r in rows}
    if missing:
        raise ValueError(f"KCP_TEST_ROWS names unknown rows {sorted(missing)}")
    return rows


def _row_id(row: Row) -> str:
    return (
        f"{row.name}-w{row.W}-h{row.heads[0]}x{row.heads[1]}-{row.layout.split(':')[0]}"
    )


# ----------------------------------------------------------------------------------------------------------------
# partition / slicing helpers


def _cu(lengths) -> list[int]:
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + n)
    return cu


def _ragged(cu: list[int], t_r: list[int]) -> list[list[tuple[int, int, int]]]:
    """Contiguous ragged split of the global buffer into per-rank token counts, cut at document boundaries."""
    nested: list[list[tuple[int, int, int]]] = [[] for _ in t_r]
    pos = 0
    for r, n in enumerate(t_r):
        end = pos + n
        d = 0
        while d < len(cu) - 1 and cu[d + 1] <= pos:
            d += 1
        p = pos
        while p < end and d < len(cu) - 1:
            e = min(end, cu[d + 1])
            if e > p:
                nested[r].append((d, p - cu[d], e - p))
            p = e
            d += 1
        pos = end
    return nested


def _partition(row: Row):
    cu = _cu(row.lengths)
    if row.layout.startswith("ragged:"):
        t_r = [int(x) for x in row.layout.split(":")[1].split(",")]
        assert len(t_r) == row.W and sum(t_r) == cu[-1]
        return plan.kcp_partition(cu, row.W, fragments=_ragged(cu, t_r)), cu
    if row.layout.startswith("explicit:"):
        make, lengths = EXPLICIT[row.layout.split(":")[1]]
        assert tuple(lengths) == tuple(row.lengths)
        return plan.kcp_partition(cu, row.W, fragments=make()), cu
    return plan.kcp_partition(cu, row.W, layout=row.layout), cu


def _programs_exported() -> bool:
    """Every shipped record of the registry has a library in the package's ``build.json`` -- else the kernel path
    cannot run."""
    manifest = PACKAGE_DIR / "build.json"
    if not manifest.exists():
        return False
    try:
        with open(manifest) as fh:
            libraries = {lib["name"] for lib in json.load(fh)["libraries"]}
        jit = _mod("cake_jit")
    except (OSError, ValueError, KeyError, TypeError, ImportError):
        return False
    if not libraries or not hasattr(jit, "MODULES"):
        return False
    try:
        return all(
            any(jit.library_name(name, arch) in libraries for arch in record["arches"])
            for name, record in jit.MODULES.items()
        )
    except Exception:  # noqa: BLE001 - a registry without library_name: the manifest's names decide
        return bool(libraries)


def _backend_name() -> str:
    name = os.environ.get("KCP_TEST_BACKEND")
    if name:
        return name
    return "kernels" if _programs_exported() else "fake"


def _local_inputs(
    row: Row, part, cu: list[int], rank: int, device, *, seed=None, doc_override=None
):
    """This rank's operands (fragments in slot order) generated per fragment range (bitwise the global slice);
    ``doc_override = (doc, other_seed)`` takes every document except ``doc`` from ``other_seed`` (independence)."""
    gen = _inputs()
    H_qk, HV = row.heads
    parts: dict[str, list[torch.Tensor]] = {
        n: [] for n in ("q", "k", "v", "g", "beta", "dout")
    }
    for f in part.fragments[rank]:
        if f.length == 0:
            continue
        a = cu[f.doc] + f.off
        kw = {"regime": row.regime, "token_range": (a, a + f.length)}
        if seed is not None:
            kw["seed"] = seed
        if doc_override is not None and f.doc != doc_override[0]:
            kw["seed"] = doc_override[1]
        got = gen.make_inputs(cu, H_qk, HV, **kw)
        for n in parts:
            parts[n].append(got[n].reshape((-1,) + tuple(got[n].shape[2:])))
    out = {}
    for n, lst in parts.items():
        if lst:
            out[n] = torch.cat(lst, 0).to(device)
        else:
            shape = {
                "q": (0, H_qk, 128),
                "k": (0, H_qk, 128),
                "v": (0, HV, 128),
                "g": (0, HV),
                "beta": (0, HV),
                "dout": (0, HV, 128),
            }[n]
            dtype = torch.float32 if n in ("g", "beta") else torch.bfloat16
            out[n] = torch.empty(shape, dtype=dtype, device=device)
    return out


def _global_inputs(row: Row, cu: list[int], device, *, seed=None):
    gen = _inputs()
    H_qk, HV = row.heads
    kw = {"regime": row.regime}
    if seed is not None:
        kw["seed"] = seed
    got = gen.make_inputs(cu, H_qk, HV, **kw)
    return {
        n: got[n].reshape((-1,) + tuple(got[n].shape[2:])).to(device)
        for n in ("q", "k", "v", "g", "beta", "dout")
    }


def _to_global(
    part, rank_tensors: list[torch.Tensor], cu: list[int], total: int
) -> torch.Tensor:
    ref = rank_tensors[0]
    g = torch.zeros((total,) + tuple(ref.shape[1:]), dtype=ref.dtype, device=ref.device)
    for r, t in enumerate(rank_tensors):
        pos = 0
        for f in part.fragments[r]:
            if f.length:
                g[cu[f.doc] + f.off : cu[f.doc] + f.off + f.length] = t[
                    pos : pos + f.length
                ]
            pos += f.length
    return g


def _doc_mask_local(part, cu: list[int], rank: int, doc: int, device) -> torch.Tensor:
    """Boolean ``[T_r]``: the local tokens of document ``doc``."""
    pieces = []
    for f in part.fragments[rank]:
        pieces.append(torch.full((f.length,), f.doc == doc, dtype=torch.bool))
    m = torch.cat(pieces) if pieces else torch.zeros(0, dtype=torch.bool)
    return m.to(device)


def _packed_views(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """Strided packed-QKV views ``[T, ., 128]`` of one ``[T, (2 H_qk + HV) * 128]`` buffer holding ``q | k | v``."""
    T, H_qk = q.shape[0], q.shape[1]
    HV = v.shape[1]
    buf = torch.empty((T, (2 * H_qk + HV) * 128), dtype=q.dtype, device=q.device)
    qv = buf[:, : H_qk * 128].view(T, H_qk, 128)
    kv = buf[:, H_qk * 128 : 2 * H_qk * 128].view(T, H_qk, 128)
    vv = buf[:, 2 * H_qk * 128 :].view(T, HV, 128)
    qv.copy_(q)
    kv.copy_(k)
    vv.copy_(v)
    return buf, qv, kv, vv


def _eq(a: dict, b: dict, names=TENSORS) -> bool:
    return all(torch.equal(a[n], b[n]) for n in names)


def _rounded(nbytes: int) -> int:
    return -(-nbytes // ALLOC_GRANULE) * ALLOC_GRANULE if nbytes else 0


# ----------------------------------------------------------------------------------------------------------------
# one rank


def _live_cuda_tensors(device) -> dict[int, tuple]:
    """``data_ptr -> (bytes, shape, dtype)`` of every live CUDA tensor on ``device`` (diagnostic only)."""
    out: dict[int, tuple] = {}
    for obj in gc.get_objects():
        try:
            if (
                torch.is_tensor(obj)
                and obj.is_cuda
                and obj.device == device
                and obj.numel()
            ):
                out[obj.data_ptr()] = (
                    obj.numel() * obj.element_size(),
                    tuple(obj.shape),
                    str(obj.dtype),
                )
        except Exception:  # noqa: BLE001 - foreign objects may refuse attribute access
            continue
    return out


def _new_cuda_tensors(before: dict[int, tuple], device, *, skip=()) -> list[dict]:
    """The live CUDA tensors that are new since ``before`` with the type names of their referrers (bounded)."""
    found: list[dict] = []
    for obj in gc.get_objects():
        try:
            if not (
                torch.is_tensor(obj)
                and obj.is_cuda
                and obj.device == device
                and obj.numel()
            ):
                continue
            ptr = obj.data_ptr()
            if ptr in before or ptr in skip:
                continue
            refs = [type(r).__name__ for r in gc.get_referrers(obj)[:6]]
            found.append(
                {
                    "bytes": obj.numel() * obj.element_size(),
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                    "referrers": refs,
                }
            )
        except Exception:  # noqa: BLE001
            continue
        if len(found) >= 40:
            break
    return found


def _alloc_stats(device) -> dict[str, int]:
    """``requested`` (bytes the callers asked for) and ``allocated`` (block bytes) of the caching allocator."""
    stats = torch.cuda.memory_stats(device)
    return {
        "requested": int(stats.get("requested_bytes.all.current", 0)),
        "allocated": int(stats.get("allocated_bytes.all.current", 0)),
    }


def _block_of(t: torch.Tensor) -> dict | None:
    """Size / requested size of the allocator block that holds ``t`` (from the snapshot's segments; diagnostic)."""
    ptr = t.data_ptr()
    for seg in torch.cuda.memory._snapshot().get("segments", []):
        addr = int(seg.get("address", 0))
        for blk in seg.get("blocks", []):
            size = int(blk.get("size", 0))
            if addr <= ptr < addr + size:
                return {
                    "size": size,
                    "requested_size": int(blk.get("requested_size", 0)),
                    "state": str(blk.get("state", "")),
                    "segment_size": int(seg.get("total_size", 0)),
                }
            addr += size
    return None


def _trace_mark(device) -> int:
    """Start the allocator's history (diagnostic only) and return the current trace length of ``device``."""
    torch.cuda.memory._record_memory_history(max_entries=200000)
    snap = torch.cuda.memory._snapshot()
    traces = snap.get("device_traces") or []
    return len(traces[device.index]) if device.index < len(traces) else 0


def _live_blocks_since(mark: int, device, *, skip=()) -> list[dict]:
    """Allocator blocks allocated after ``mark`` and not freed, with sizes and their non-torch Python frames."""
    snap = torch.cuda.memory._snapshot()
    torch.cuda.memory._record_memory_history(enabled=None)
    traces = snap.get("device_traces") or []
    events = traces[device.index][mark:] if device.index < len(traces) else []
    live: dict[int, dict] = {}
    for ev in events:
        action = ev.get("action")
        if action == "alloc":
            live[ev["addr"]] = ev
        elif action in ("free_requested", "free_completed"):
            live.pop(ev["addr"], None)
    rows = []
    for ev in live.values():
        if ev["addr"] in skip:
            continue
        frames = []
        for f in ev.get("frames", []):
            fn = str(f.get("filename", ""))
            if (
                "/torch/" in fn
                and "checkpoint" not in fn
                and "cake_gdn_train" not in fn
            ):
                continue
            frames.append(f"{fn.rsplit('/', 1)[-1]}:{f.get('line')} {f.get('name')}")
            if len(frames) >= 10:
                break
        rows.append({"size": int(ev["size"]), "frames": frames})
        if len(rows) >= 40:
            break
    return rows


def _test_mode() -> str:
    """``KCP_TEST_MODE``: ``full`` (forward + backward rows, default) or ``forward`` (the INT1 forward-only rows:
    #1-#5, ``out`` and the carried states against FP64, no backward program)."""
    mode = os.environ.get("KCP_TEST_MODE", "full")
    if mode not in ("full", "forward"):
        raise ValueError(f"KCP_TEST_MODE must be 'full' or 'forward', got {mode!r}")
    return mode


def _run_rank_fwd(
    rank: int,
    W: int,
    row: Row,
    group,
    device: torch.device,
    result_dir: str | None,
    backend: str,
) -> dict:
    """Forward-only body of one rank (INT1): the forward's bitwise forms, ``out`` against the FP64 forward and the
    crossing-in fragments' ``S_in`` against the FP64 cut states; nothing of the backward runs."""
    H_qk, HV = row.heads
    part, cu = _partition(row)
    ctx = plan.build_kcp_context(part, group, rank, device)
    total = cu[-1]
    ops = _local_inputs(row, part, cu, rank, device)
    q, k, v, g, beta = (ops[n] for n in ("q", "k", "v", "g", "beta"))
    ws = kcp.KcpWorkspace()
    base = {"backend": backend}
    t_start = time.time()

    def fwd(qq=q, kk=k, vv=v, **opts):
        return kcp.chunk_gdn_kcp_fwd(
            qq, kk, vv, g, beta, ctx, workspace=ws, **{**base, **opts}
        )

    flags: dict[str, bool] = {}
    info: dict[str, object] = {}
    out, saved = fwd()
    info["fwd_record"] = {
        k_: v_
        for k_, v_ in kcp.LAST_CALL["fwd"].items()
        if k_ not in ("sequence", "pool_bytes", "fresh_bytes", "merges")
    }
    info["fwd_sequence"] = list(kcp.LAST_CALL["fwd"]["sequence"])
    flags["repeat"] = torch.equal(out, fwd()[0])
    flags["save_transfer_eq"] = torch.equal(out, fwd(save_transfer=False)[0])
    flags["light_eq_full"] = torch.equal(out, fwd(save_mode="full")[0])
    flags["saved_eq_replay"] = torch.equal(out, fwd(boundary_mode="replay")[0])
    ring_eligible = W > 1 and int(ctx.n_d_max) - 1 <= kcp.RING_MAX_HOPS
    info["ring_eligible"] = ring_eligible
    flags["ring_eq_a2a"] = (not ring_eligible) or torch.equal(
        out, fwd(exchange="ring", save_transfer=False)[0]
    )
    _buf, qs, ks, vs = _packed_views(q, k, v)
    flags["strided_eq_contiguous"] = torch.equal(out, fwd(qs, ks, vs)[0])
    flags["copy_layout_eq_contiguous"] = torch.equal(
        out, fwd(qs, ks, vs, layout="copy")[0]
    )
    # the carried states: S_in per crossing-in fragment (in_slot order) with its global cut
    s_in = (
        kcp.saved_states(saved)
        if saved is not None
        else out.new_zeros((0, HV, 128, 128), dtype=torch.float32)
    )
    cuts = [None] * int(ctx.n_in)
    for f in (
        ctx.fragments[rank] if isinstance(ctx.fragments[0], tuple) else ctx.fragments
    ):
        if f.length > 0 and f.off > 0:
            cuts[int(ctx.in_slot[f.slot])] = int(cu[f.doc] + f.off)
    info["s_in_cuts"] = cuts
    # gather
    objs_out: list = [None] * W
    objs_s: list = [None] * W
    objs_cuts: list = [None] * W
    flag_list: list = [None] * W
    info_list: list = [None] * W
    if W > 1:
        dist.all_gather_object(objs_out, out.cpu(), group=group)
        dist.all_gather_object(objs_s, s_in.cpu(), group=group)
        dist.all_gather_object(objs_cuts, cuts, group=group)
        dist.all_gather_object(flag_list, flags, group=group)
        dist.all_gather_object(info_list, info, group=group)
    else:
        objs_out, objs_s, objs_cuts, flag_list, info_list = (
            [out.cpu()],
            [s_in.cpu()],
            [cuts],
            [flags],
            [info],
        )
    report = {
        "row": dataclasses.asdict(row),
        "mode": "forward",
        "rank": rank,
        "world": W,
        "backend": backend,
        "host": socket.gethostname(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "cpu",
        "nccl": ".".join(str(x) for x in torch.cuda.nccl.version())
        if torch.cuda.is_available()
        else None,
        "torch": torch.__version__,
        "utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pass": True,
        "flags": {},
        "gated": [],
        "reported": [],
        "metrics": {},
        "states": [],
        "info_by_rank": info_list,
    }
    for key in flag_list[0]:
        ok = all(bool(fl[key]) for fl in flag_list)
        report["flags"][key] = ok
        report["gated"].append(key)
        report["pass"] &= ok
    if rank == 0 and total > 0:
        ref_mod = _reference()
        glob = _global_inputs(row, cu, device)
        cu_t = torch.tensor(cu, dtype=torch.int64)
        ref_out = ref_mod.gdn_exact_forward(
            glob["q"], glob["k"], glob["v"], glob["g"], glob["beta"], cu_t, scale=SCALE
        )
        ref_out = ref_out[0] if isinstance(ref_out, tuple) else ref_out
        cand = _to_global(part, [t.to(device) for t in objs_out], cu, total).to(
            torch.float64
        )
        ref = ref_out.to(device=device, dtype=torch.float64).reshape(cand.shape)
        _judge(report, row, "out", cand, ref, ref_mod, _budget_cell(row))
        all_cuts = sorted({c for cs in objs_cuts for c in cs if c is not None})
        if all_cuts:
            states = ref_mod.cut_states(
                glob["q"],
                glob["k"],
                glob["v"],
                glob["g"],
                glob["beta"],
                cu_t,
                all_cuts,
                scale=SCALE,
            )
            for r_, (cs, st) in enumerate(zip(objs_cuts, objs_s, strict=True)):
                for slot, c in enumerate(cs):
                    if c is None:
                        continue
                    cand_s = st[slot].to(device=device, dtype=torch.float64)
                    ref_s = (
                        states[c]
                        .to(device=device, dtype=torch.float64)
                        .reshape(cand_s.shape)
                    )
                    e = cand_s - ref_s
                    rel = float(
                        e.pow(2).mean().sqrt()
                        / ref_s.pow(2).mean().sqrt().clamp_min(1e-300)
                    )
                    doc = max(i for i in range(len(cu) - 1) if cu[i] <= c)
                    entry = {
                        "rank": r_,
                        "slot": slot,
                        "cut": c,
                        "doc": doc,
                        "off": c - cu[doc],
                        "rel_rms": rel,
                        "max_abs": float(e.abs().max()),
                        "ref_rms": float(ref_s.pow(2).mean().sqrt()),
                        "finite": bool(torch.isfinite(cand_s).all()),
                    }
                    report["states"].append(entry)
                    report["pass"] &= entry["finite"] and rel <= STATE_REL_RMS_MAX
    report["seconds"] = round(time.time() - t_start, 2)
    if W > 1:
        verdict: list = [None] * W
        dist.all_gather_object(verdict, report["pass"], group=group)
        report["pass"] = all(verdict)
        dist.barrier(group=group)
    if rank == 0 and result_dir:
        os.makedirs(result_dir, exist_ok=True)
        with open(os.path.join(result_dir, _row_id(row) + "-fwd.json"), "w") as fh:
            json.dump(report, fh, indent=1, default=str)
    return report


def _kept_record(T_r: int) -> dict:
    """The last backward's launch inventory under ``reuse_recompute_tapes``: where its forward tapes came from and
    whether anything ran before #6.  With tokens the kept path must have been taken (no silent fallback to the
    replay); a rank without tokens has no tapes and runs the replay form as before."""
    bwd = kcp.LAST_CALL["bwd"]
    seq = list(bwd["sequence"])
    forward_programs = {"fwd_prep", "kcp_fwdh_offset", "kcp_fwdh_tape", "fwd_o"}
    first_local = next(
        (s_ for s_ in seq if s_ in kcp.LOCAL_PROGRAMS or s_ in forward_programs), None
    )
    source = bwd.get("tapes_source")
    if T_r > 0:
        path_ok = (
            source == "kept"
            and first_local == "bwd_prep"
            and not (forward_programs & set(seq))
            and bool(kcp.LAST_CALL["fwd"].get("hand_off"))
        )
    else:
        path_ok = source == "replay" and not bool(kcp.LAST_CALL["fwd"].get("hand_off"))
    return {
        "tapes_source": source,
        "first_local_program": first_local,
        "forward_programs_in_bwd": sorted(forward_programs & set(seq)),
        "fwd_hand_off": bool(kcp.LAST_CALL["fwd"].get("hand_off")),
        "P": bwd.get("P"),
        "path_ok": bool(path_ok),
        "bwd_sequence": seq,
    }


def _run_rank(
    rank: int,
    W: int,
    row: Row,
    group,
    device: torch.device,
    result_dir: str | None,
    backend: str,
) -> dict:
    """The body of one rank: every mode of the KCP forward / backward, gathered to rank 0 and compared there."""
    H_qk, HV = row.heads
    part, cu = _partition(row)
    ctx = plan.build_kcp_context(part, group, rank, device)
    total = cu[-1]
    ops = _local_inputs(row, part, cu, rank, device)
    q, k, v, g, beta, dout = (ops[n] for n in ("q", "k", "v", "g", "beta", "dout"))
    ws = kcp.KcpWorkspace()
    base = {"backend": backend}
    t_start = time.time()

    def saved_pass(qq=q, kk=k, vv=v, gg=g, bb=beta, dd=dout, **opts) -> dict:
        o = {**base, **opts}
        fwd_opts = dict(o)
        out, saved = kcp.chunk_gdn_kcp_fwd(
            qq, kk, vv, gg, bb, ctx, workspace=ws, **fwd_opts
        )
        dq, dk, dv, dbeta, dg = kcp.chunk_gdn_kcp_bwd(
            qq, kk, vv, gg, bb, ctx, dd, saved=saved, workspace=ws, **o
        )
        return {
            "out": out,
            "dq": dq,
            "dk": dk,
            "dv": dv,
            "dg": dg,
            "dbeta": dbeta,
            "saved": saved,
        }

    def autograd_pass(ckpt: bool, *, twice: bool = False, workspace=ws, **opts) -> dict:
        leaves = [x.detach().clone().requires_grad_(True) for x in (q, k, v, g, beta)]
        o = {**base, **opts}

        def fn(*a):
            return kcp.chunk_gdn_kcp(*a, ctx, workspace=workspace, **o)

        if ckpt:
            from torch.utils.checkpoint import checkpoint

            out = checkpoint(fn, *leaves, use_reentrant=False)
        else:
            out = fn(*leaves)
        out.backward(dout, retain_graph=twice)
        if twice:
            out.backward(dout)
        return {
            "out": out.detach(),
            **{n: lf.grad for n, lf in zip(GRADS, leaves, strict=True)},
        }

    flags: dict[str, bool] = {}
    info: dict[str, object] = {}
    res = saved_pass()
    info["fwd_record"] = {
        k_: v_
        for k_, v_ in kcp.LAST_CALL["fwd"].items()
        if k_ not in ("sequence", "pool_bytes", "fresh_bytes", "merges")
    }
    info["bwd_record"] = {
        k_: v_
        for k_, v_ in kcp.LAST_CALL["bwd"].items()
        if k_ not in ("sequence", "pool_bytes", "fresh_bytes", "merges")
    }
    info["fwd_sequence"] = list(kcp.LAST_CALL["fwd"]["sequence"])
    info["bwd_sequence"] = list(kcp.LAST_CALL["bwd"]["sequence"])
    flags["repeat"] = _eq(res, saved_pass())
    flags["saved_eq_replay"] = _eq(res, saved_pass(boundary_mode="replay"))
    flags["light_eq_full"] = _eq(res, saved_pass(save_mode="full"))
    res_auto = autograd_pass(False)
    flags["autograd_eq_saved"] = _eq(res, res_auto)
    res_ckpt = autograd_pass(True)
    flags["ckpt_eq_direct"] = _eq(res_auto, res_ckpt)
    flags["ckpt_full_eq_direct"] = _eq(res_auto, autograd_pass(True, save_mode="full"))
    flags["ckpt_replay_eq_direct"] = _eq(
        res_auto, autograd_pass(True, boundary_mode="replay")
    )
    # reuse_recompute_tapes: the checkpoint backward binds the forward tapes torch's recompute just wrote instead
    # of replaying #1 and the tape walk; bitwise the direct and the switch-off checkpoint gradients; the kept path
    # is proven from the backward's launch inventory (a rank without tokens has no tapes and runs the replay form);
    # the registry is empty afterwards; the per-rank peak of the checkpointed call with the switch is not above the
    # switch-off peak (G8 evidence; the row of record is the memory trace)
    KEPT = {"reuse_recompute_tapes": True}
    # both states are named (res_ckpt above is the default's)
    OFF = {"reuse_recompute_tapes": False}
    T_r_here = int(q.shape[0])
    res_ckpt_off = autograd_pass(True, **OFF)
    kept_stats0 = dict(kcp.KEPT_STATS)
    res_kept = autograd_pass(True, **KEPT)
    info["kept"] = _kept_record(T_r_here)
    flags["kept_eq_direct"] = _eq(res_auto, res_kept)
    flags["kept_eq_ckpt_light"] = _eq(res_ckpt_off, res_kept) and _eq(
        res_ckpt, res_kept
    )
    flags["kept_path_taken"] = bool(info["kept"]["path_ok"])
    res_kept_nt = autograd_pass(True, save_transfer=False, **KEPT)
    info["kept_no_transfer"] = _kept_record(T_r_here)
    # the direct reference inside the stats window names the replay form: under the shipped default a direct
    # backward looks the registry up and counts a "miss" (by design), which would break the miss = 0 expectation
    flags["kept_no_transfer_eq_direct"] = _eq(
        autograd_pass(False, save_transfer=False, **OFF), res_kept_nt
    ) and bool(info["kept_no_transfer"]["path_ok"])
    res_kept_twice = autograd_pass(True, twice=True, **KEPT)
    flags["kept_twice_accumulates"] = all(
        torch.equal(res_kept_twice[n], res_auto[n] + res_auto[n]) for n in GRADS
    )
    kept_delta = {k: kcp.KEPT_STATS[k] - kept_stats0[k] for k in kept_stats0}
    info["kept_stats_delta"] = kept_delta
    n_kept_bwd = 4  # three kept passes, one of them with two backwards
    expect = (
        {"put": n_kept_bwd, "take": n_kept_bwd, "miss": 0}
        if T_r_here > 0
        else {"put": 0, "take": 0, "miss": n_kept_bwd}
    )
    flags["kept_registry_empty"] = kcp.kept_registry_size() == 0 and all(
        kept_delta[k] == v_ for k, v_ in expect.items()
    )

    # issue #7074 (2) + G10: the "auto" form (the shipped default) in a region the GDN node closes takes the kept
    # path on a rank with tokens (the frame proof "dead"; a rank without tokens replays as before), is bitwise the
    # direct result, and its checkpointed forward + backward -- the recompute forward, the frame proof over the
    # autograd graph (structure only, never tensor contents), the kept-tape take -- make no synchronizing CUDA call
    # (outside the kept-stats window above: the "auto" pass has its own accounting below)
    AUTO = {"reuse_recompute_tapes": "auto"}
    auto_stats0 = dict(kcp.KEPT_STATS)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        res_auto_form = autograd_pass(True, **AUTO)
        flags["auto_no_sync"] = True
    except RuntimeError as exc:  # a synchronizing call in the "auto" pass
        info["auto_no_sync_error"] = str(exc)[:200]
        flags["auto_no_sync"] = False
        res_auto_form = None
    finally:
        torch.cuda.set_sync_debug_mode("default")
    if res_auto_form is None:
        res_auto_form = autograd_pass(True, **AUTO)
    info["auto_form"] = _kept_record(T_r_here)
    info["auto_form"]["hand_off_reason"] = kcp.LAST_CALL["fwd"].get("hand_off_reason")
    flags["auto_eq_direct"] = _eq(res_auto, res_auto_form)
    # a rank with tokens hands off on the frame proof ("dead"); a rank without tokens has no tapes and switches the
    # hand-off off before the "auto" decision, as every form does there (recorded "off", the replay form)
    flags["auto_path_taken"] = bool(info["auto_form"]["path_ok"]) and (
        info["auto_form"]["hand_off_reason"] == ("dead" if T_r_here > 0 else "off")
        or not kcp.torch_release_verified()
    )
    auto_delta = {k: kcp.KEPT_STATS[k] - auto_stats0[k] for k in auto_stats0}
    info["auto_stats_delta"] = auto_delta
    # one hand-off on a rank with tokens (the proof "dead"); none without tokens or on an unverified release -- and no
    # miss is counted for an "auto" forward that did not hand off
    n_auto = 1 if (T_r_here > 0 and kcp.torch_release_verified()) else 0
    flags["auto_registry_empty"] = kcp.kept_registry_size() == 0 and all(
        auto_delta[k] == v_
        for k, v_ in {"put": n_auto, "take": n_auto, "miss": 0}.items()
    )

    def _ckpt_peak(**o) -> int:
        torch.cuda.synchronize(device)
        gc.collect()
        torch.cuda.reset_peak_memory_stats(device)
        live = torch.cuda.memory_allocated(device)
        autograd_pass(True, **o)
        torch.cuda.synchronize(device)
        return int(torch.cuda.max_memory_allocated(device) - live)

    peaks = {"off": _ckpt_peak(**OFF), "on": _ckpt_peak(**KEPT)}
    info["ckpt_peak_bytes"] = peaks
    flags["kept_peak_le_off"] = peaks["on"] <= peaks["off"]
    res_recompute = saved_pass(save_transfer=False)
    flags["save_transfer_eq"] = _eq(res, res_recompute)
    ring_eligible = W > 1 and int(ctx.n_d_max) - 1 <= kcp.RING_MAX_HOPS
    info["ring_eligible"] = ring_eligible
    flags["ring_eq_a2a"] = (not ring_eligible) or _eq(
        res_recompute, saved_pass(exchange="ring", save_transfer=False)
    )
    res_host = saved_pass(group_reduce="host")
    info["host_reduce_eq_fused"] = _eq(
        res, res_host
    )  # reported: both forms are fixed-order FP32 + one rounding
    # strided packed-QKV views: zero-copy (layout="tma") and the copy adapter, both bitwise the contiguous call
    _buf, qs, ks, vs = _packed_views(q, k, v)
    flags["strided_eq_contiguous"] = _eq(res, saved_pass(qs, ks, vs))
    flags["copy_layout_eq_contiguous"] = _eq(res, saved_pass(qs, ks, vs, layout="copy"))
    # gradient accumulation: .backward() twice == exactly twice the gradient (x + x is exact in every float format)
    res_twice = autograd_pass(False, twice=True)
    flags["backward_twice_accumulates"] = all(
        torch.equal(res_twice[n], res_auto[n] + res_auto[n]) for n in GRADS
    )
    # G10 (issue #7074): preparing a fresh context of this rank's layout -- the host plan, its table uploads and the
    # per-key device tables -- makes no synchronizing CUDA call (pinned asynchronous copies), so a pipeline can prepare
    # the next batch's metadata behind queued device work
    torch.cuda.synchronize(device)
    torch.cuda.set_sync_debug_mode("error")
    try:
        fresh_ctx = plan.build_kcp_context(part, group, rank, device)
        fresh_ctx.prepare(H_qk, HV)
        flags["prepare_no_sync"] = True
    except RuntimeError as exc:  # "called a synchronizing CUDA operation"
        info["prepare_no_sync_error"] = str(exc)[:200]
        flags["prepare_no_sync"] = False
    finally:
        torch.cuda.set_sync_debug_mode("default")
    del fresh_ctx
    torch.cuda.synchronize(device)
    # #7039: retained bytes after a checkpointed forward (graph off), per-call and caller-held workspace forms.
    # the plan's per-key device tables are uploaded first (ctx.prepare + one warm-up call of the
    # same form) so they are not counted; graph="replay" keeps a static pool and is reported, not asserted.
    prepare = getattr(ctx, "prepare", None)
    if callable(prepare):
        prepare(H_qk, HV)
    retained: dict[str, int] = {}
    retained_block: dict[str, int] = {}
    for form, wsp, form_opts in (
        ("per_call", None, {**base, **OFF}),
        ("caller_ws", ws, {**base, **OFF}),
        ("per_call_kept", None, {**base, **KEPT}),
        ("caller_ws_kept", ws, {**base, **KEPT}),
    ):
        leaves = [x.detach().clone().requires_grad_(True) for x in (q, k, v, g, beta)]

        def fn(*a, _wsp=wsp, _o=form_opts):
            return kcp.chunk_gdn_kcp(*a, ctx, workspace=_wsp, **_o)

        from torch.utils.checkpoint import checkpoint

        warm = checkpoint(
            fn, *leaves, use_reentrant=False
        )  # warm-up of the same form (tables, allocator)
        warm.backward(dout)
        del warm
        for leaf in leaves:
            leaf.grad = None
        torch.cuda.synchronize(device)
        gc.collect()
        if wsp is not None:
            # the warm-up backward's replay re-bound the arena tapes an eager forward drops before its output
            # (FWD_DEAD_BEFORE_O); start from the state a forward leaves so the delta is the output alone
            for name in kcp.FWD_DEAD_BEFORE_O + kcp.FWD_DEAD_AFTER_PREP:
                wsp.release_tape(name)
            for name in kcp.FWD_DEAD_POOLS:
                wsp.release_pool(name)
            torch.cuda.synchronize(device)
            gc.collect()
        debug = bool(os.environ.get("KCP_TEST_DEBUG_RETAINED"))
        live_before = _live_cuda_tensors(device) if debug else None
        trace_from = _trace_mark(device) if debug else None
        before = _alloc_stats(device)
        out = checkpoint(fn, *leaves, use_reentrant=False)
        torch.cuda.synchronize(device)
        after = _alloc_stats(device)
        # The gate counts REQUESTED bytes: ``memory_allocated`` counts block sizes, and the caching allocator hands
        # out an unsplit cached block when its remainder is below the split threshold (1 MiB for the large pool),
        # so the block holding ``out`` may be larger than ``out`` without any tensor being retained.
        out_bytes = out.numel() * out.element_size()
        retained[form] = int(after["requested"] - before["requested"] - out_bytes)
        # the block delta minus out's rounded block stays under the allocator's 1 MiB split threshold
        retained_block[form] = int(
            after["allocated"] - before["allocated"] - _rounded(out_bytes)
        )
        dbg = info.setdefault("retained_debug", {}).setdefault(form, {})
        dbg["requested_delta"] = int(after["requested"] - before["requested"])
        dbg["allocated_delta"] = int(after["allocated"] - before["allocated"])
        dbg["out_bytes"] = int(out_bytes)
        dbg["out_block"] = _block_of(out)
        if wsp is not None:
            # an eager forward leaves no dead forward tape in the caller's arena (its output took their place)
            dbg["arena_tapes_after_forward"] = sorted(wsp._tapes)
            dead_tapes = set(kcp.FWD_DEAD_BEFORE_O + kcp.FWD_DEAD_AFTER_PREP)
            no_dead_tape = not (set(wsp._tapes) & dead_tapes)
            flags["fwd_drops_dead_tapes"] = (
                flags.get("fwd_drops_dead_tapes", True) and no_dead_tape
            )
            dbg["arena_pools_after_forward"] = sorted(wsp._pools)
            no_dead_pool = not (set(wsp._pools) & set(kcp.FWD_DEAD_POOLS))
            flags["fwd_drops_dead_pools"] = (
                flags.get("fwd_drops_dead_pools", True) and no_dead_pool
            )
        if debug:
            dbg["tensors"] = _new_cuda_tensors(
                live_before, device, skip={out.data_ptr()}
            )
            dbg["blocks"] = _live_blocks_since(
                trace_from, device, skip={out.data_ptr()}
            )
        out.backward(dout)
        if wsp is not None and form == "caller_ws_kept":
            # a kept backward leaves the caller's arena as a forward leaves it: of the forward tapes only the four
            # #5 reads are bound (no dead tape, no kept tape beyond them), no entry survives the backward
            dbg["arena_tapes_after_backward"] = sorted(wsp._tapes)
            flags["kept_ws_as_forward_leaves"] = (
                set(wsp._tapes) & set(kcp.FWD_TAPES)
            ) <= set(kcp.FWD_TAPES_ARENA_AFTER_FWD) and kcp.kept_registry_size() == 0
        del out, leaves
    info["retained_bytes"] = retained
    info["retained_block_bytes"] = retained_block
    # INT3 save_mode decision inputs (G8, M1): this rank's saved-set bytes per save_mode -- the public byte query and
    # the measured bytes of the saved objects (recorded, not gated: the default stays "light")
    try:
        sb: dict[str, int] = {"T_r": int(q.shape[0])}
        for mode in ("light", "full"):
            sb[f"{mode}_query"] = int(
                kcp.gdn_kcp_saved_bytes(ctx, H_qk=H_qk, HV=HV, save_mode=mode)
            )
            _, s_obj = kcp.chunk_gdn_kcp_fwd(
                q, k, v, g, beta, ctx, workspace=ws, save_mode=mode, **base
            )
            sb[f"{mode}_actual"] = int(s_obj.bytes())
            del s_obj
        info["saved_bytes"] = sb
    except Exception as exc:  # noqa: BLE001 - recorded for the decision table
        info["saved_bytes"] = f"unavailable: {type(exc).__name__}: {str(exc)[:200]}"
    flags["retained_zero"] = all(v_ == 0 for v_ in retained.values())
    flags["retained_block_lt_1mib"] = all(v_ < 2**20 for v_ in retained_block.values())
    # exchange bytes: the tables' split sums (what the collectives move) against the public byte query
    tb = ws.tables(ctx, HV)
    moved = {}
    for coll in ("c1", "c2", "b2"):
        s, r = tb.split_elems(coll)
        moved[coll] = {"send": 4 * sum(s), "recv": 4 * sum(r)}
    s, r = tb.split_elems("b1_d")
    moved["b1"] = {"send": 4 * sum(s), "recv": 4 * sum(r)}
    s, r = tb.split_elems("b1_dm")
    moved["b1_dm"] = {"send": 4 * sum(s), "recv": 4 * sum(r)}
    info["moved_bytes"] = moved
    try:
        query = kcp.gdn_kcp_comm_bytes(ctx, H_qk=H_qk, HV=HV, C=C_CONV)
        info["query_bytes"] = {
            c: query[c] for c in ("c1", "c2", "b1", "b2", "b1_design") if c in query
        }
        # save_transfer=False: "b1" = the plan's split tables (maps paired with the crossing-IN slots; what the
        # collective moves) must equal the b1_dm split sums; "b1_design" is the per-rank lower bound
        query_dm = kcp.gdn_kcp_comm_bytes(
            ctx, H_qk=H_qk, HV=HV, C=C_CONV, save_transfer=False
        )
        info["query_bytes_dm"] = {
            c: query_dm[c] for c in ("b1", "b1_design") if c in query_dm
        }
        flags["bytes_match_query"] = (
            all(
                int(query[c]["send"]) == moved[c]["send"]
                and int(query[c]["recv"]) == moved[c]["recv"]
                for c in ("c1", "c2", "b2")
            )
            and int(query["b1"]["send"]) == moved["b1"]["send"]
            and int(query["b1"]["recv"]) == moved["b1"]["recv"]
            and int(query_dm["b1"]["send"]) == moved["b1_dm"]["send"]
            and int(query_dm["b1"]["recv"]) == moved["b1_dm"]["recv"]
        )
    except (
        ImportError,
        AttributeError,
        KeyError,
        TypeError,
    ) as exc:  # the byte module is another lane's deliverable
        info["query_bytes"] = f"unavailable: {type(exc).__name__}: {exc}"
        flags["bytes_match_query"] = False
    # G10: CUDA-graph capture of forward + backward (pointer-stable operands, caller-held workspace): the third call
    # of a key replays the captured graph and must be bitwise the eager result.  The backward's key includes the
    # saved tensors, which change once (the eager forward's fresh tensors -> the captured forward's static tensors),
    # so the backward reaches its replay one call after the forward: four forward + backward calls are made and the
    # per-call events are recorded (forward: eager, capture, replay, replay; backward: eager, eager, capture, replay).
    graph_info: dict[str, object] = {}
    try:
        kcp.graph_cache_clear()
        g_res = None
        events: list[list[str]] = []
        for _ in range(4):
            out, saved = kcp.chunk_gdn_kcp_fwd(
                q, k, v, g, beta, ctx, workspace=ws, graph="replay", **base
            )
            grads = kcp.chunk_gdn_kcp_bwd(
                q,
                k,
                v,
                g,
                beta,
                ctx,
                dout,
                saved=saved,
                workspace=ws,
                graph="replay",
                **base,
            )
            g_res = {
                "out": out.clone(),
                **{
                    n: t.clone()
                    for n, t in zip(
                        ("dq", "dk", "dv", "dbeta", "dg"), grads, strict=True
                    )
                },
            }
            events.append(
                [
                    str(kcp.LAST_CALL["fwd"]["graph_event"]),
                    str(kcp.LAST_CALL["bwd"]["graph_event"]),
                ]
            )
        graph_info["events"] = events
        graph_info["fwd_event"] = kcp.LAST_CALL["fwd"]["graph_event"]
        graph_info["bwd_event"] = kcp.LAST_CALL["bwd"]["graph_event"]
        graph_info["fwd_fallback"] = kcp.LAST_CALL["fwd"].get("graph_fallback")
        graph_info["bwd_fallback"] = kcp.LAST_CALL["bwd"].get("graph_fallback")
        graph_info["stats"] = dict(kcp.GRAPH_STATS)
        graph_ok = (
            graph_info["fwd_event"] == "replay"
            and graph_info["bwd_event"] == "replay"
            and _eq(res, g_res)
        )
        torch.cuda.synchronize(device)
        kcp.graph_cache_clear()
    except Exception as exc:  # noqa: BLE001 - recorded; the gate decides below
        graph_info["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        graph_ok = False
        kcp.graph_cache_clear()
    info["graph"] = graph_info
    flags["graph_capture"] = bool(graph_ok)
    # #7038: the backward's time with a resident one-thread kernel on a side stream, over its time alone
    ratio = _side_stream_ratio(
        lambda: kcp.chunk_gdn_kcp_bwd(
            q, k, v, g, beta, ctx, dout, saved=res["saved"], workspace=ws, **base
        ),
        device,
    )
    info["side_stream"] = ratio
    flags["side_stream_ratio"] = ratio["ratio"] <= SIDE_STREAM_RATIO_MAX
    # no-leak rows: zero gradients outside the selected document and independence of its outputs
    if row.leak and total > 0:
        n_docs = len(cu) - 1
        doc = 1 if n_docs > 1 else 0
        mask = _doc_mask_local(part, cu, rank, doc, device)
        dout_sel = dout * mask.view(-1, 1, 1).to(dout.dtype)
        res_sel = saved_pass(dd=dout_sel)
        others = ~mask
        flags["zero_grad_leak"] = all(
            bool(torch.equal(res_sel[n][others], torch.zeros_like(res_sel[n][others])))
            for n in GRADS
        )
        gen = _inputs()
        alt = _local_inputs(
            row,
            part,
            cu,
            rank,
            device,
            doc_override=(doc, int(getattr(gen, "SEED", 20261023)) + 1),
        )
        res_alt = saved_pass(
            alt["q"], alt["k"], alt["v"], alt["g"], alt["beta"], alt["dout"]
        )
        flags["independence"] = all(
            torch.equal(res_alt[n][mask], res[n][mask]) for n in TENSORS
        )
        flags["independence_changed_others"] = (
            bool(others.any())
            and any(
                not torch.equal(res_alt[n][others], res[n][others]) for n in ("out",)
            )
            or not bool(others.any())
        )
    # gather
    gathered = {}
    for n in TENSORS:
        objs: list = [None] * W
        if W > 1:
            dist.all_gather_object(objs, res[n].cpu(), group=group)
        else:
            objs = [res[n].cpu()]
        gathered[n] = objs
    flag_list: list = [None] * W
    info_list: list = [None] * W
    if W > 1:
        dist.all_gather_object(flag_list, flags, group=group)
        dist.all_gather_object(info_list, info, group=group)
    else:
        flag_list, info_list = [flags], [info]
    report: dict = {
        "row": dataclasses.asdict(row),
        "rank": rank,
        "world": W,
        "backend": backend,
        "host": socket.gethostname(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "nccl": ".".join(map(str, torch.cuda.nccl.version())) if W > 1 else None,
        "torch": torch.__version__,
        "utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pass": True,
        "flags": {},
        "gated": [],
        "reported": [],
        "metrics": {},
        "info_by_rank": info_list,
        "seconds": None,
    }
    # gates: bitwise / semantics / retained / bytes on every backend; graph capture is a gate of the kernel backend
    # (the torch emulation reports it); the per-row side-stream ratio is recorded on every backend -- the #7038 gate
    # of record is the module-timing protocol (slowest-rank median over >= 3 groups on an exclusive node)
    kernel_only = {"graph_capture"}
    info_only = {"side_stream_ratio"}
    for key in flags:
        ok = all(f[key] for f in flag_list)
        report["flags"][key] = ok
        if key in info_only or (key in kernel_only and backend != "kernels"):
            report["reported"].append(key)
        else:
            report["gated"].append(key)
            report["pass"] &= ok
    if rank == 0 and total > 0:
        ref_mod = _reference()
        glob = _global_inputs(row, cu, device)
        cu_t = torch.tensor(cu, dtype=torch.int64)
        refs = ref_mod.gdn_exact_with_grads(
            glob["q"],
            glob["k"],
            glob["v"],
            glob["g"],
            glob["beta"],
            glob["dout"],
            cu_t,
            scale=SCALE,
        )
        budget = _budget_cell(row)
        for n in TENSORS:
            cand = _to_global(part, [t.to(device) for t in gathered[n]], cu, total).to(
                torch.float64
            )
            ref = refs[n].to(device=device, dtype=torch.float64).reshape(cand.shape)
            _judge(report, row, n, cand, ref, ref_mod, budget)
    report["seconds"] = round(time.time() - t_start, 2)
    if W > 1:
        verdict: list = [None] * W
        dist.all_gather_object(verdict, report["pass"], group=group)
        report["pass"] = all(verdict)
        dist.barrier(group=group)
    if rank == 0 and result_dir:
        os.makedirs(result_dir, exist_ok=True)
        with open(os.path.join(result_dir, _row_id(row) + ".json"), "w") as fh:
            json.dump(report, fh, indent=1, default=str)
    return report


def _side_stream_ratio(bwd, device: torch.device) -> dict:
    """#7038 repro form: the backward's wall time (CUDA events, median of ``SIDE_STREAM_ITERS``) alone and with a
    one-thread ``torch.cuda._sleep`` kernel resident on a side stream for the whole call."""

    def timed() -> float:
        torch.cuda.synchronize(device)
        e0, e1 = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        e0.record()
        bwd()
        e1.record()
        torch.cuda.synchronize(device)
        return float(e0.elapsed_time(e1))

    bwd()  # warm
    alone = [timed() for _ in range(SIDE_STREAM_ITERS)]
    med_alone = statistics.median(alone)
    clock_khz = (
        torch.cuda.get_device_properties(device).clock_rate
        if hasattr(torch.cuda.get_device_properties(device), "clock_rate")
        else 1_900_000
    )
    cycles = int(
        max(med_alone, 1.0) * 3.0 * clock_khz
    )  # ~3x the backward's duration in SM cycles
    side = torch.cuda.Stream(device=device)
    with_side = []
    for _ in range(SIDE_STREAM_ITERS):
        torch.cuda.synchronize(device)
        with torch.cuda.stream(side):
            torch.cuda._sleep(cycles)
        with_side.append(timed())
        side.synchronize()
    med_side = statistics.median(with_side)
    return {
        "alone_ms": alone,
        "with_side_ms": with_side,
        "median_alone_ms": med_alone,
        "median_with_side_ms": med_side,
        "ratio": med_side / max(med_alone, 1e-9),
        "sleep_cycles": cycles,
    }


# ----------------------------------------------------------------------------------------------------------------
# launch forms


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _spawn_entry(
    rank: int, W: int, row: Row, port: int, result_dir: str, backend: str
) -> None:
    """One spawned rank (the verdict travels through rank 0's report file)."""
    device = torch.device("cuda", rank % max(torch.cuda.device_count(), 1))
    torch.cuda.set_device(device)
    try:
        if W > 1:
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", str(port))
            dist.init_process_group("nccl", rank=rank, world_size=W, device_id=device)
        group = dist.group.WORLD if W > 1 else None
        report = _run_rank(rank, W, row, group, device, result_dir, backend)
        if not report["pass"]:
            raise AssertionError(
                f"rank {rank}: {json.dumps(report, default=str)[:4000]}"
            )
    finally:
        kcp.graph_cache_clear()
        if dist.is_initialized():
            dist.destroy_process_group()


def _expand_hostlist(s: str) -> list[str]:
    import re

    out: list[str] = []
    for item in re.findall(r"[^,\[\]]+(?:\[[^\]]*\])?", s):
        m = re.match(r"^([^\[]*)\[([^\]]*)\](.*)$", item)
        if not m:
            if item:
                out.append(item)
            continue
        pre, body, post = m.groups()
        for part in body.split(","):
            if "-" in part:
                a, b = part.split("-")
                for i in range(int(a), int(b) + 1):
                    out.append(f"{pre}{str(i).zfill(len(a))}{post}")
            elif part:
                out.append(f"{pre}{part}{post}")
    return out


def _slurm_rendezvous(rank: int, world: int) -> tuple[str, int]:
    """MASTER_ADDR from the STEP's nodelist; the port published per Slurm step by rank 0 in ``KCP_TEST_PORT_DIR``
    (shared filesystem) unless ``MASTER_PORT`` is set."""
    addr = os.environ.get("MASTER_ADDR")
    if not addr:
        hl = (
            os.environ.get("SLURM_STEP_NODELIST")
            or os.environ.get("SLURM_NODELIST")
            or socket.gethostname()
        )
        addr = _expand_hostlist(hl)[0]
    port_env = os.environ.get("MASTER_PORT")
    if port_env:
        return addr, int(port_env)
    job, step = (
        os.environ.get("SLURM_JOB_ID", "0"),
        os.environ.get("SLURM_STEP_ID", "0"),
    )
    d = os.environ.get("KCP_TEST_PORT_DIR") or os.path.join(
        os.environ.get("W", tempfile.gettempdir()), "tmp"
    )
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"kcp_test_port.{job}.{step}")
    if rank == 0:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("", 0))
            port = s.getsockname()[1]
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "w") as fh:
            fh.write(f"{port}\n")
        os.replace(tmp, path)
        return addr, port
    t0 = time.time()
    while time.time() - t0 < 600:
        try:
            with open(path) as fh:
                txt = fh.read().strip()
            if txt:
                return addr, int(txt)
        except FileNotFoundError:
            pass
        time.sleep(0.2)
    raise RuntimeError(f"rank {rank}: no rendezvous port file {path} within 600 s")


class _Slurm:
    """The one-process-per-task form: WORLD initialised once; sub-groups of the ``W`` lowest ranks per row."""

    def __init__(self) -> None:
        self.rank = int(os.environ["SLURM_PROCID"])
        self.world = int(os.environ.get("SLURM_NTASKS", "1"))
        local = int(os.environ.get("SLURM_LOCALID", self.rank))
        n = torch.cuda.device_count()
        self.device = torch.device("cuda", 0 if n == 1 else local % max(n, 1))
        torch.cuda.set_device(self.device)
        self.groups: dict[int, object] = {}
        if self.world > 1 and not dist.is_initialized():
            addr, port = _slurm_rendezvous(self.rank, self.world)
            os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = addr, str(port)
            # bounded: a rendezvous that never completes (a port collision with a live socket) or a failed row
            # must fail the step instead of holding its block; the longest legitimate collective wait (rank 0's
            # FP64 oracle of the largest row) stays well below it
            dist.init_process_group(
                "nccl",
                rank=self.rank,
                world_size=self.world,
                device_id=self.device,
                timeout=_dt.timedelta(seconds=WORLD_TIMEOUT_S),
            )
        print(
            f"[kcp_dist] init timeout {WORLD_TIMEOUT_S} s; "
            f"rank {self.rank}/{self.world} local {local} host {socket.gethostname()} device {self.device} "
            f"{torch.cuda.get_device_name(self.device)} flashinfer={_mod().__file__} MASTER={os.environ.get('MASTER_ADDR')}:{os.environ.get('MASTER_PORT')}",
            flush=True,
        )

    def group(self, W: int):
        """The sub-group of ranks ``0..W-1`` (every rank of WORLD must call this in the same order)."""
        if self.world == W:
            return dist.group.WORLD if self.world > 1 else None
        g = self.groups.get(W)
        if g is None:
            g = self.groups[W] = dist.new_group(ranks=list(range(W)))
        return g

    def close(self) -> None:
        kcp.graph_cache_clear()
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


_SLURM: _Slurm | None = None


@pytest.fixture(scope="session")
def slurm():
    global _SLURM
    if os.environ.get("SLURM_PROCID") is None:
        yield None
        return
    if _SLURM is None:
        _SLURM = _Slurm()
    yield _SLURM
    _SLURM.close()


def _skip_unless_runnable(W: int, backend: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    if torch.cuda.device_count() < W:
        pytest.skip(f"{W} CUDA devices required, {torch.cuda.device_count()} visible")
    if backend == "kernels" and not _programs_exported():
        pytest.skip("the KCP programs are not exported in the package's build.json yet")


def _slurm_row(row: Row, slurm, backend: str, body) -> None:
    """One row under Slurm: sub-group of the row's ranks, WORLD-wide verdict (see :func:`test_multi_gpu_rows`)."""
    W = row.W
    if slurm.world < W:
        pytest.skip(f"row needs {W} ranks, the step has {slurm.world}")
    group = slurm.group(
        W
    )  # collective over WORLD: every rank calls it, members and non-members alike
    if slurm.world > 1:
        dist.barrier()
    if backend == "kernels" and not _programs_exported():
        pytest.skip("the KCP programs are not exported in the package's build.json yet")
    report = None
    if slurm.rank < W:
        report = body(
            slurm.rank,
            W,
            row,
            group,
            slurm.device,
            os.environ.get("KCP_TEST_RESULT_DIR"),
            backend,
        )
    ok = report is None or bool(report["pass"])
    if slurm.world > 1:
        verdicts: list = [None] * slurm.world
        dist.all_gather_object(verdicts, ok)
        ok = all(verdicts)
    if slurm.rank >= W:
        if not ok:
            pytest.fail(f"rank {slurm.rank}: row failed on its {W} member ranks")
        pytest.skip(f"rank {slurm.rank} is not part of the row's {W}-rank group")
    assert ok, json.dumps(
        {k: v for k, v in report.items() if k != "info_by_rank"}, default=str
    )[:6000]


@pytest.mark.parametrize("row", _selected_rows(), ids=_row_id)
def test_forward_rows(row: Row, slurm) -> None:
    """INT1 forward-only rows (``KCP_TEST_MODE=forward``): #1-#5, ``out`` vs FP64, ``S_in`` vs the FP64 cut states."""
    if _test_mode() != "forward":
        pytest.skip("KCP_TEST_MODE=forward selects the forward-only rows")
    backend = _backend_name()
    if slurm is not None:
        _slurm_row(row, slurm, backend, _run_rank_fwd)
        return
    _skip_unless_runnable(row.W, backend)
    if row.W != 1:
        pytest.skip("forward-only rows outside Slurm run at W = 1 only")
    report = _run_rank_fwd(
        0,
        1,
        row,
        None,
        torch.device("cuda", 0),
        os.environ.get("KCP_TEST_RESULT_DIR"),
        backend,
    )
    assert report["pass"], report


@pytest.mark.parametrize("row", _selected_rows(), ids=_row_id)
def test_multi_gpu_rows(row: Row, slurm) -> None:
    if _test_mode() != "full":
        pytest.skip("KCP_TEST_MODE=forward: the forward-only rows run instead")
    backend = _backend_name()
    if slurm is not None:
        W = row.W
        if slurm.world < W:
            pytest.skip(f"row needs {W} ranks, the step has {slurm.world}")
        group = slurm.group(
            W
        )  # collective over WORLD: every rank calls it, members and non-members alike
        if slurm.world > 1:
            dist.barrier()
        if backend == "kernels" and not _programs_exported():
            pytest.skip(
                "the KCP programs are not exported in the package's build.json yet"
            )
        report = None
        if slurm.rank < W:
            report = _run_rank(
                slurm.rank,
                W,
                row,
                group,
                slurm.device,
                os.environ.get("KCP_TEST_RESULT_DIR"),
                backend,
            )
        # the verdict is WORLD-wide: every rank (member or not) passes or fails the same row, so ``-x`` and the
        # session teardown keep the collectives aligned across ranks
        ok = report is None or bool(report["pass"])
        if slurm.world > 1:
            verdicts: list = [None] * slurm.world
            dist.all_gather_object(verdicts, ok)
            ok = all(verdicts)
        if slurm.rank >= W:
            if not ok:
                pytest.fail(f"rank {slurm.rank}: row failed on its {W} member ranks")
            pytest.skip(f"rank {slurm.rank} is not part of the row's {W}-rank group")
        assert ok, json.dumps(
            {k: v for k, v in report.items() if k != "info_by_rank"}, default=str
        )[:6000]
        return
    _skip_unless_runnable(row.W, backend)
    with tempfile.TemporaryDirectory(prefix="cake_gdn_kcp_dist_") as tmp:
        result_dir = os.environ.get("KCP_TEST_RESULT_DIR") or tmp
        mp.start_processes(
            _spawn_entry,
            args=(row.W, row, _free_port(), result_dir, backend),
            nprocs=row.W,
            join=True,
            start_method="spawn",
        )
        with open(os.path.join(result_dir, _row_id(row) + ".json")) as fh:
            report = json.load(fh)
    assert report["pass"], report


def test_accuracy_budget_cells_cover_the_accuracy_rows() -> None:
    """Every accuracy row (regime != ``normal``) maps to a budget cell with all six tensors (finite, positive budgets,
    the declared storage per tensor); the rows at a CP of record use their own CP, the CP16 row ``A1-cp16`` (no CP16
    cell) the A1 CP8 cell.  CPU-only."""
    table = _budgets()
    assert table["status"] == "FINAL" and table["cells"] == 216
    rows = [r for r in ROWS if r.regime != "normal"]
    assert {r.name for r in rows} == {"A3", "A1", "A2", "A4", "A1-cp16"}
    for row in rows:
        got = _budget_cell(row)
        assert got is not None, row.name
        cell, cp = got
        assert cp == (8 if row.name == "A1-cp16" else row.W), (row.name, cp)
        assert set(cell["tensors"]) == set(TENSORS), row.name
        for n, b in cell["tensors"].items():
            assert b["storage"] == STORAGE[n], (row.name, n)
            assert 0.0 < b["budget_rel_rms"] < 1.0, (row.name, n)
            assert 0.0 < b["budget_max_abs"] < 1.0, (row.name, n)
    assert _budget_cell(Row("Zz", (1,), (2, 6), 2, "contiguous")) is None


def test_public_api_cannot_select_test_programs() -> None:
    """The precision policy: the shipped record is ``precision_mode=0`` / ``transfer_walk="tf32_hilo"``;
    the test-only records (1 ``_three_mma``, 2 ``_plain_tf32``, ``"bf16x3"``) are refused unless ``test_knobs`` names
    the same value; nothing in the environment changes that (there is no environment hook)."""
    from flashinfer.experimental.cake_gdn_train import cake_options

    for mode in (1, 2):
        with pytest.raises(ValueError):
            cake_options.make_options(precision_mode=mode)
    with pytest.raises(ValueError):
        cake_options.make_options(transfer_walk="bf16x3")
    with pytest.raises(TypeError):
        cake_options.make_options(_kcp_test_knobs={"precision_mode": 1})
    o = cake_options.make_options()
    assert o.precision_mode == 0 and o.transfer_walk == "tf32_hilo"
    o = cake_options.make_options(precision_mode=2, test_knobs={"precision_mode": 2})
    assert o.precision_mode == 2


# ----------------------------------------------------------------------------------------------------------------
# C7: the convolution halo row.  The user calls FlashInfer's ``cake_causal_conv1d_kcp(x_local, weight, ctx)`` with
# THIS package's context: the conv package reads the halo plan (``ctx.halo``, ``ctx.halo_tables``), the local
# ``cu_seqlens`` and the process group by duck typing.  The row's partition: document 0 = [1, 2, 300] on ranks 0,
# 1 and 3 (the three history rows of rank 3's first outputs come from TWO predecessor ranks), rank 2 holds no token
# (an empty predecessor), document 1 = [200] on rank 3 (a document start inside the continuing rank); C = 10,240.

CONV_C = 10240
CONV_WIDTH = 4
CONV_Y_ATOL, CONV_Y_RTOL = (
    1e-3,
    2.0**-7,
)  # one BF16 ulp, no element outside (the conv package's own gate)
CONV_DW_ATOL, CONV_DW_RTOL = 5e-2, 1e-4  # FP32 dweight against the FP64 oracle


def _c7_fragments() -> list[list[tuple[int, int, int]]]:
    return [[(0, 0, 1)], [(0, 1, 2)], [], [(0, 3, 300), (1, 0, 200)]]


EXPLICIT["c7"] = (_c7_fragments, (303, 200))
C7_ROW = Row("C7", (303, 200), (2, 6), 4, "explicit:c7", stage="conv")


def _conv_reference(x: torch.Tensor, w: torch.Tensor, cu: list[int], dy: torch.Tensor):
    """FP64 oracle of the unpartitioned width-4 depthwise causal convolution + SiLU with the conv package's KCP
    semantics (``weight[c, 3]`` multiplies the current token; zero history at a document start; the SiLU
    derivative on the unrounded preactivation): BF16 ``y`` and ``dx`` (one rounding each), FP64 ``dpre`` for the
    per-rank ``dweight`` partials."""
    S, _ = x.shape
    xf, wd = x.double(), w.double()
    doc = torch.zeros(S, dtype=torch.int64, device=x.device)
    for i in range(len(cu) - 1):
        doc[cu[i] : cu[i + 1]] = i

    def shifted(t, k):  # t[i - k] of the same document, else 0
        if k == 0:
            return t.clone()
        out = torch.zeros_like(t)
        if k < S:
            same = doc[k:] == doc[:-k]
            out[k:] = torch.where(
                same[:, None], t[: S - k], torch.zeros_like(t[: S - k])
            )
        return out

    def shifted_up(t, k):  # t[i + k] of the same document, else 0
        if k == 0:
            return t.clone()
        out = torch.zeros_like(t)
        if k < S:
            same = doc[k:] == doc[:-k]
            out[: S - k] = torch.where(same[:, None], t[k:], torch.zeros_like(t[k:]))
        return out

    halo = CONV_WIDTH - 1
    pre = sum(wd[:, k][None, :] * shifted(xf, halo - k) for k in range(CONV_WIDTH))
    s = torch.sigmoid(pre)
    y = (pre * s).to(torch.bfloat16)
    dpre = dy.double() * s * (1 + pre * (1 - s))
    dx = sum(
        wd[:, k][None, :] * shifted_up(dpre, halo - k) for k in range(CONV_WIDTH)
    ).to(torch.bfloat16)
    taps = [shifted(xf, halo - k) for k in range(CONV_WIDTH)]
    return y, dx, dpre, taps


def _outside(
    actual: torch.Tensor, expected: torch.Tensor, *, atol: float, rtol: float
) -> dict:
    """Elements of ``actual`` beyond ``atol + rtol |expected|`` and the comparison's statistics."""
    a = actual.detach().reshape(-1).double()
    b = expected.detach().reshape(-1).double()
    if a.numel() == 0:
        return {"count_outside": 0, "max_abs_diff": 0.0, "finite": True, "n": 0}
    diff = (a - b).abs()
    return {
        "count_outside": int((diff > atol + rtol * b.abs()).sum().item()),
        "max_abs_diff": float(diff.max().item()),
        "finite": bool(torch.isfinite(a).all().item()),
        "n": int(a.numel()),
    }


def _run_rank_conv(
    rank: int,
    W: int,
    row: Row,
    group,
    device: torch.device,
    result_dir: str | None,
    backend: str,
) -> dict:
    """C7 on one rank: ``y`` / ``dx`` / ``dweight`` of the rank's tokens against the FP64 oracle of the whole pack,
    ``y`` bitwise equal to the single-rank conv of the whole pack on the rank's tokens, repeat bitwise, and the
    halo-row gradients delivered to the predecessor ranks (rank 0 / 1: 1 / 2 tokens of document 0)."""
    from flashinfer import cake_causal_conv1d_train as conv

    part, cu = _partition(row)
    ctx = plan.build_kcp_context(part, group, rank, device)
    total = cu[-1]
    gen = torch.Generator(device="cpu").manual_seed(43567)
    x_all = (torch.randn(total, CONV_C, generator=gen) * 0.5).to(torch.bfloat16)
    w = (torch.randn(CONV_C, CONV_WIDTH, generator=gen) * 0.5).float()
    dy_all = (torch.randn(total, CONV_C, generator=gen) * 0.5).to(torch.bfloat16)
    pieces = [
        torch.arange(cu[f.doc] + f.off, cu[f.doc] + f.off + f.length)
        for f in part.fragments[rank]
        if f.length
    ]
    l2g = torch.cat(pieces) if pieces else torch.zeros(0, dtype=torch.int64)
    x_local = x_all[l2g].to(device).requires_grad_(True)
    w_dev = w.to(device).requires_grad_(True)
    dy_local = dy_all[l2g].to(device)
    t0 = time.time()
    y = conv.cake_causal_conv1d_kcp(x_local, w_dev, ctx)
    y.backward(dy_local)
    dx, dw = x_local.grad, w_dev.grad
    y2 = conv.cake_causal_conv1d_kcp(x_local.detach(), w_dev.detach(), ctx)
    torch.cuda.synchronize(device)
    # the oracle (FP64 on this rank's device, the whole pack) and the single-rank conv of the whole pack
    y_ref, dx_ref, dpre, taps = _conv_reference(
        x_all.to(device), w.to(device), cu, dy_all.to(device)
    )
    sel = l2g.to(device)
    dw_ref = torch.stack(
        [(dpre[sel] * tap[sel]).sum(0) for tap in taps], dim=1
    )  # this rank's partial of dweight (the framework reduces the ranks)
    cu_dev = torch.tensor(cu, dtype=torch.int32, device=device)
    y_single = conv.cake_causal_conv1d(x_all.to(device), w.to(device), cu_dev)
    flags: dict[str, bool] = {}
    metrics: dict[str, dict] = {}
    metrics["y"] = _outside(y, y_ref[sel], atol=CONV_Y_ATOL, rtol=CONV_Y_RTOL)
    metrics["dx"] = _outside(dx, dx_ref[sel], atol=CONV_Y_ATOL, rtol=CONV_Y_RTOL)
    metrics["dw"] = _outside(dw, dw_ref, atol=CONV_DW_ATOL, rtol=CONV_DW_RTOL)
    flags["y_within"] = metrics["y"]["count_outside"] == 0 and metrics["y"]["finite"]
    flags["dx_within"] = metrics["dx"]["count_outside"] == 0 and metrics["dx"]["finite"]
    flags["dw_within"] = metrics["dw"]["count_outside"] == 0 and metrics["dw"]["finite"]
    flags["y_eq_single_rank"] = torch.equal(y, y_single[sel])
    flags["repeat"] = torch.equal(y, y2)
    flags["shapes"] = tuple(y.shape) == tuple(x_local.shape) and tuple(
        dx.shape
    ) == tuple(x_local.shape)
    # the halo rows: the predecessor ranks' tokens feed rank 3's first outputs, so their dx must carry that term
    # (non-zero), and a rank without tokens has empty y / dx
    if rank in (0, 1):
        flags["halo_grad_nonzero"] = bool((dx.float().abs() > 0).any().item())
    if rank == 2:
        flags["empty_rank"] = y.numel() == 0 and dx.numel() == 0
    report = {
        "row": dataclasses.asdict(row),
        "rank": rank,
        "world": W,
        "backend": "conv",
        "mode": "conv",
        "host": socket.gethostname(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pass": all(flags.values()),
        "flags": flags,
        "gated": sorted(flags),
        "reported": [],
        "metrics": metrics,
        "info_by_rank": [
            {
                "rank": rank,
                "T_r": int(x_local.shape[0]),
                "l2g_first": int(l2g[0]) if len(l2g) else None,
            }
        ],
        "seconds": round(time.time() - t0, 2),
        "C": CONV_C,
    }
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)
        with open(
            os.path.join(result_dir, f"{row.name}-conv-rank{rank}.json"), "w"
        ) as fh:
            json.dump(report, fh, indent=1, default=str)
    return report


def test_conv_halo_row(slurm) -> None:
    """C7: FlashInfer's ``cake_causal_conv1d_kcp`` through this package's context (duck typing): ``y`` / ``dx`` /
    ``dweight`` vs FP64 on every rank, ``y`` bitwise the single-rank conv, history rows from two predecessor ranks,
    an empty predecessor and a document start.  Runs under Slurm with >= 4 ranks."""
    if slurm is None:
        pytest.skip("the conv halo row runs under Slurm (4 ranks)")
    conv = pytest.importorskip("flashinfer.cake_causal_conv1d_train")
    if not conv.is_cake_causal_conv1d_train_supported(slurm.device):
        pytest.skip("the conv package does not support this device")
    _slurm_row(C7_ROW, slurm, "conv", _run_rank_conv)
