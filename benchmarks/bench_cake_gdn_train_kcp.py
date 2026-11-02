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

Benchmark of the context-parallel (KCP) Cake packed GDN training step on both numerical routes: one process per
rank (``torchrun --nproc_per_node N``), every rank timing the module boundary of its fragments -- the KCP causal
convolution (``flashinfer.cake_causal_conv1d_train.cake_causal_conv1d_kcp``, 10,240 channels; ``--no-conv`` leaves
it out) followed by the KCP GDN step -- with CUDA events, the slowest rank's time counting for the iteration.

Arms (``--arms``; every one on the same packed batch, the same partition and the same input storage):

* ``det``: the deterministic route, the objects of ``flashinfer.cake_gdn_det_train_kcp`` (what
  ``deterministic=True`` dispatches to), with its own partition and context;
* ``nondet``: ``flashinfer.cake_gdn_train_kcp`` with ``deterministic=False`` (the package
  ``flashinfer.experimental.cake_gdn_train_kcp``), default options;
* ``det_graph`` / ``nondet_graph``: ``det`` / ``nondet`` with a caller-owned workspace (``WorkspaceDetKcp`` /
  ``WorkspaceKcp``, one per arm and configuration) and ``graph="replay"``: the GDN call's steady state replays a captured
  CUDA graph (the package captures only with a caller-owned workspace); the convolution stays eager; no ``B3h`` form
  (a recompute handle needs ``graph="off"``); the graphs are released after each configuration;
* ``nondet_graph`` runs on a context built with ``graph_replay=True`` (the package's plan for graph replay);
* ``nondet_graph_p_<rule>``: ``nondet_graph`` on a non-deterministic context built with the plan rule ``<rule>``
  (``--graph-p-rules``; e.g. ``row_cost_lpt``, the deterministic twin's plan);
* ``sota_fla_kcp``: flash-linear-attention's own context parallelism (``fla.ops.cp.context.build_cp_context`` with
  ``use_tf32x3_affine_chain=True``, layouts ``C`` and ``Z`` only) around its Triton kernels -- the convolution
  ``fla.modules.causal_conv1d.causal_conv1d(activation="silu")`` and ``chunk_gated_delta_rule``.  FLA splits the
  pack itself, evenly, after padding it with one zero document to a multiple of ``2 x world`` tokens; the padding is
  computed and timed;
* ``sota_head_cp_fla``: head parallelism: the same even token split (padded to a multiple of ``world``), an
  all-to-all to whole sequences of ``H_qk / W`` query/key heads, ``HV / W`` value heads and ``C / W`` channels (the
  world size must divide each), FLA's single-rank varlen kernels, and the all-to-all back (backward: the reverse
  exchanges); every layout runs the same program;
* ``sota_head_cp_frost`` / ``sota_head_cp_frost_ck64``: the same head parallelism around cuDNN frontend's GDN
  (``cudnn.linear_attention.ops.gated_delta_net``, plan ``gdn_frost``, ``checkpoint_every_n_tokens`` 0: the
  backward recomputes the states, or 64: stored states; FLA's convolution).

The non-Cake arms run FLA's Triton kernels (the benchmark sets ``FLA_FLASH_QLA=0`` before importing FLA); a
requested arm that cannot be imported stops the run, one whose calls did not run fails its probe, and one that does
not apply to a row (layout, head count) is reported unavailable with the reason.  They need ``nondet`` in the arm
list: once per (row, layout) each non-Cake arm's outputs and input gradients (``y, out, dq, dk, dv, dg, dbeta, dx``
per token, ``dweight`` summed over the ranks) are compared with ``nondet``'s on the unpadded pack, and the arm is
``qualified`` when every tensor is finite with rel-RMS <= 1e-2 and max-abs <= 2e-2 x max |ref|.  They have no
recompute-handle form (``B3h``).

Boundaries (``--boundaries``): ``B1`` the module forward with autograd recording; ``B2`` forward + backward;
``B3`` ``torch.utils.checkpoint(use_reentrant=False)`` forward + recompute + backward (``2 x forward + backward``)
without a recompute handle (the deterministic benchmark's B3); ``B3h`` the same with a recompute handle
(``RecomputeHandleDetKcp`` on ``det``, ``RecomputeHandleKcp`` on the non-deterministic arms): one handle per step,
created outside the checkpoint and passed as an argument of the checkpointed function, shared by the convolution
(key ``"conv"``) and the GDN call (key ``"gdn"``).

Rows (``--rows``): rows of the shape module ``gdn_shapes`` on ``PYTHONPATH`` (the packs ``P1`` .. ``P5``: document
lengths, ``(H_qk, HV)``, CP sizes, layouts and, for layout ``D``, ``cut_points_D``) and ``DET``, the deterministic
benchmark's default pack (``--seqlen`` tokens in ``--docs`` documents, ``--qk-heads`` / ``--v-heads``, the
package's default cut points) and the small-pair rows ``S26`` / ``S13`` (``(H_qk, HV)`` = (2, 6) / (1, 3), three
documents of 8,192 tokens: about 4k tokens per rank at CP2).  Layouts (``--layouts``): ``C`` contiguous, ``Z`` zigzag, ``D``
zigzag_per_document.  The CP size is the launch's world size; a row whose CP set or layout set excludes it is
skipped.

Protocol: per (row, layout, boundary), one untimed call per arm (JIT builds, first exchanges) and one probe call per
arm (peak memory per phase, the routed package's ``LAST_CALL`` records), then ``--groups`` groups (at least 3); in
every group the arms run in the palindromic pattern ``[a1 .. an, an .. a1] x k`` (ABBA for two arms); every call
takes ``--samples`` timed iterations (at least 8) after ``--warmup`` untimed ones, each after an L2 flush (unless
``--no-l2-flush``), a device synchronize and a barrier; ``--cool-s`` idles every rank that many seconds before each
call (the cool-down of a sustained, power-capped protocol: long ``--samples`` blocks).  Every rank times every
iteration with CUDA events; the slowest rank's time is the iteration's.  Each call records its wall-clock window
(``t_unix``, rank 0) for joining with GPU power / clock telemetry, and each arm its per-call (block) medians.
Steady samples as in the deterministic benchmark (every sample when
``--warmup`` > 0).  Per arm: the median, p10 and p90 of the slowest rank's steady samples, the group medians and
their drift, the per-rank medians, tokens per second; on every rank the peak memory of the probe call
(``torch.cuda.max_memory_allocated`` minus the allocation before the call: forward phase, backward phase, bytes
retained after the forward) and the ``LAST_CALL`` records of its forward, checkpoint recompute and backward
(``merges``, launched programs, ``backend_path``).  Ratios: every non-deterministic arm over ``det``, every
non-Cake arm over ``nondet`` (medians of the slowest rank).  Rank 0 prints one line per arm and the ratios
and writes one JSON per (row, layout, CP) under ``--out-dir`` with every per-iteration time (slowest rank and every
rank), the files every arm ran from, the host map and the exact argv.

Inputs are drawn on the device from one seeded generator (``--seed``): every rank draws the same whole-pack tensors
and keeps its fragments.  No environment variable selects anything; the FlashInfer tree importable first
(``PYTHONPATH``) is the one measured (this checkout only when none is importable).

Examples::

    # CP8 on one 8-GPU node
    torchrun --nproc_per_node 8 benchmarks/bench_cake_gdn_train_kcp.py --rows P1,P2,P3,P4,P5 --out-dir kcp_cp8

    # CP8 on two 4-GPU nodes (one torchrun per node; HEAD = the first node of the allocation)
    srun -N 2 --ntasks-per-node 1 bash -c 'torchrun --nnodes 2 --nproc_per_node 4 --rdzv_backend c10d \\
        --rdzv_endpoint "$HEAD:29500" --rdzv_id "$SLURM_JOB_ID" benchmarks/bench_cake_gdn_train_kcp.py \\
        --rows P2,P4 --layouts C,Z,D --out-dir kcp_cp8'
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import importlib
import importlib.util
import json
import math
import os
import re
import socket
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

if (
    importlib.util.find_spec("flashinfer") is None
):  # no FlashInfer tree on PYTHONPATH: this checkout's
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import flashinfer  # noqa: E402
from flashinfer import cake_causal_conv1d_train as conv_api  # noqa: E402

BASE_ARMS = ("det", "nondet")
# CUDA-graph replay arms: the route each one runs
GRAPH_ARMS = {"det_graph": "det", "nondet_graph": "nondet"}
PRULE_ARM = re.compile(r"^nondet_p_(\w+)$")
# graph replay on a context built with a plan rule: nondet_graph_p_<rule>
GRAPH_PRULE_ARM = re.compile(r"^nondet_graph_p_(\w+)$")


def is_graph(arm: str) -> bool:
    """Whether ``arm`` replays a captured graph (caller-owned workspace, ``graph="replay"``)."""
    return arm in GRAPH_ARMS or GRAPH_PRULE_ARM.match(arm) is not None


def arm_p_rule(arm: str) -> Optional[str]:
    """The plan rule of a ``nondet_p_<rule>`` / ``nondet_graph_p_<rule>`` arm (``None``: the route's context)."""
    m = PRULE_ARM.match(arm) or GRAPH_PRULE_ARM.match(arm)
    return m.group(1) if m else None


BOUNDARIES = ("B1", "B2", "B3", "B3h")
BOUNDARY_LABELS = {
    "B1": "forward with autograd recording",
    "B2": "forward + backward",
    "B3": "checkpoint(use_reentrant=False): forward + recompute + backward, no recompute handle",
    "B3h": "checkpoint(use_reentrant=False): forward + recompute + backward, one recompute handle per step "
    "(created outside the checkpoint, passed as its argument, shared by the convolution and the GDN call)",
}
LAYOUTS = {"C": "contiguous", "Z": "zigzag", "D": "zigzag_per_document"}
DET_ROW = "DET"
DEFAULT_ROWS = ("P1", "P2", "P3", "P4", "P5", DET_ROW)
# Small head pairs on short shards (about 4k tokens per rank at CP2): the occupancy-limited regime of the dqkg
# kernel (``NC x H_qk`` well below the SM count), where a value-head split of dqkg could pay off.
SMALL_ROWS = {
    "S26": ((2049, 3001, 3142), (2, 6)),
    "S13": ((2049, 3001, 3142), (1, 3)),
}
L2_FLUSH_BYTES = 256 << 20
WIDTH = 4
HEAD_DIM = 128
MIB = float(1 << 20)
SCALE = HEAD_DIM**-0.5
# Non-Cake arms (the second baseline of a table: the minimum over the qualified ones)
SOTA_ARMS = (
    "sota_fla_kcp",
    "sota_head_cp_fla",
    "sota_head_cp_frost",
    "sota_head_cp_frost_ck64",
)
# The GDN call of each non-Cake arm (a library call the probe must see) and the cuDNN arms' state checkpoint stride
SOTA_GDN_CALL = {
    "sota_fla_kcp": "chunk",
    "sota_head_cp_fla": "chunk",
    "sota_head_cp_frost": "frost",
    "sota_head_cp_frost_ck64": "frost",
}
FROST_CHECKPOINT = {"sota_head_cp_frost": 0, "sota_head_cp_frost_ck64": 64}
FLA_KCP_LAYOUTS = {"C": "contiguous", "Z": "zigzag"}
# Agreement of a non-Cake arm with ``nondet`` on every output and input gradient (the generic budgets of the Cake GDN
# training tests): rel-RMS and max-abs over max |ref|
SOTA_REL_RMS = 1e-2
SOTA_MAX_ABS = 2e-2
AGREEMENT_BLOCK_ELEMS = 1 << 28
# LAST_CALL fields kept per record (plus every ``n_*`` counter and the launched programs).
CALL_KEYS = (
    "merges",
    "backend",
    "backend_path",
    "exchange",
    "P",
    "NSEG",
    "T_r",
    "n_in",
    "n_out",
    "save_mode",
    "save_transfer",
    "schedule",
    "fused_walks",
    "cols_rule",
    "precision_mode",
    "transfer_walk",
    "workspace_bytes",
    "graph_event",
    "tapes_source",
    "hand_off",
    "hand_off_reason",
    "fwd_o",
    "fwd_o_reason",
    "recompute_handle",
)


# ----------------------------------------------------------------------------------------------------------------
# Routes and arms


def _jsonable(v: Any) -> Any:
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_jsonable(x) for x in v]
    if isinstance(v, torch.Tensor):
        return f"tensor{list(v.shape)}:{v.dtype}"
    return repr(v)


def _query(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """A byte query's value, or the error it raised (the deterministic tree on PYTHONPATH may predate a keyword)."""
    try:
        return _jsonable(fn(*args, **kwargs))
    except Exception as exc:  # noqa: BLE001 -- recorded, not fatal: the queries are evidence, not the measurement
        return {"error": f"{type(exc).__name__}: {exc}"}


class Route:
    """The entry points of one numerical route.  ``det``: the objects of ``flashinfer.cake_gdn_det_train_kcp`` (what
    ``deterministic=True`` dispatches to); ``nondet``: ``flashinfer.cake_gdn_train_kcp`` with
    ``deterministic=False`` (the package ``flashinfer.experimental.cake_gdn_train_kcp``)."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        if kind == "det":
            self.api = importlib.import_module("flashinfer.cake_gdn_det_train_kcp")
            self.pkg = importlib.import_module(
                "flashinfer.experimental.cake_gdn_det_train_kcp"
            )
            self.mode: dict[str, Any] = {}
            self._partition = self.api.partition_det_kcp
            self._context = self.api.build_context_det_kcp
            self._chunk = self.api.chunk_gdn_det_kcp
            self._workspace = self.api.gdn_det_workspace_bytes_kcp
            self._saved = self.api.gdn_det_saved_bytes_kcp
            self._comm = self.api.gdn_det_comm_bytes_kcp
            self.handle_name = "RecomputeHandleDetKcp"
            self.workspace_name = "WorkspaceDetKcp"
        else:
            self.api = importlib.import_module("flashinfer.cake_gdn_train_kcp")
            self.pkg = importlib.import_module(
                "flashinfer.experimental.cake_gdn_train_kcp"
            )
            self.mode = {"deterministic": False}
            self._partition = self.api.partition_kcp
            self._context = self.api.build_context_kcp
            self._chunk = self.api.chunk_gdn_kcp
            self._workspace = self.api.gdn_workspace_bytes_kcp
            self._saved = self.api.gdn_saved_bytes_kcp
            self._comm = self.api.gdn_comm_bytes_kcp
            self.handle_name = "RecomputeHandleKcp"
            self.workspace_name = "WorkspaceKcp"

    def partition(
        self, cu: torch.Tensor, world: int, layout: str, cut_points: Any
    ) -> Any:
        kw = dict(self.mode)
        if cut_points is not None:
            kw["cut_points"] = cut_points
        return self._partition(cu, world, layout, **kw)

    def context(
        self,
        part: Any,
        rank: int,
        device: torch.device,
        p_rule: Optional[str] = None,
        graph_replay: bool = False,
    ) -> Any:
        if (
            p_rule is not None
        ):  # the package's own context builder: the facade selects no plan rule
            return self.pkg.cake_kcp_plan.build_context_kcp(
                part, dist.group.WORLD, rank, device, WIDTH, p_rule=p_rule
            )
        kw = dict(self.mode)
        if graph_replay:  # the facade's hint for contexts used with graph="replay"
            kw["graph_replay"] = True
        return self._context(
            part, dist.group.WORLD, rank, device, conv_kernel_size=WIDTH, **kw
        )

    def chunk(
        self,
        q,
        k,
        v,
        g,
        beta,
        ctx,
        save_mode: str,
        options: Any,
        handle: Any,
        workspace: Any = None,
    ) -> torch.Tensor:
        kw = dict(self.mode)
        if options is not None:
            kw["options"] = options
        if handle is not None:
            kw["recompute_handle"] = handle
        if workspace is not None:
            kw["workspace"] = workspace
        return self._chunk(q, k, v, g, beta, ctx, save_mode=save_mode, **kw)

    def workspace(self, ctx: Any, heads: tuple[int, int], device: torch.device) -> Any:
        """A caller-owned workspace of this route for ``ctx`` (the graph arms)."""
        cls = getattr(self.api, self.workspace_name, None) or getattr(
            self.pkg, self.workspace_name
        )
        return cls(ctx, heads[0], heads[1], device)

    def graph_cache_clear(self) -> None:
        """Release this route's captured graphs (before the next configuration and the process group's end)."""
        fn = getattr(self.pkg, "graph_cache_clear", None) or getattr(
            getattr(self.pkg, "cake_kcp", None), "graph_cache_clear", None
        )
        if fn is not None:
            fn()

    def handle_class(self) -> tuple[Optional[Callable[[], Any]], Optional[str]]:
        try:
            return getattr(self.api, self.handle_name), None
        except AttributeError as exc:
            return None, f"{self.api.__name__} has no {self.handle_name} ({exc})"

    def bytes(
        self, ctx: Any, heads: dict[str, int], channels: int, save_mode: str
    ) -> dict[str, Any]:
        return {
            "workspace": _query(self._workspace, ctx, **heads, **self.mode),
            "saved": _query(
                self._saved, ctx, **heads, save_mode=save_mode, **self.mode
            ),
            "saved_handle": _query(
                self._saved,
                ctx,
                **heads,
                save_mode="light",
                save_transfer=False,
                **self.mode,
            ),
            "comm": _query(self._comm, ctx, **heads, C=channels, **self.mode),
        }

    def modules(self) -> dict[str, Any]:
        kcp = getattr(self.pkg, "cake_kcp", None)
        return {
            "facade": self.api.__name__,
            "facade_file": self.api.__file__,
            "package": self.pkg.__name__,
            "package_file": self.pkg.__file__,
            "cake_kcp_file": getattr(kcp, "__file__", None),
        }

    def clear_last_call(self) -> None:
        self.pkg.LAST_CALL.clear()

    def last_call(self, kind: str) -> Optional[dict[str, Any]]:
        rec = self.pkg.LAST_CALL.get(kind)
        if not rec:
            return None
        out = {
            k: _jsonable(v)
            for k, v in rec.items()
            if k in CALL_KEYS or k.startswith("n_")
        }
        records = rec.get("records") or []
        out["n_launches"] = len(records)
        out["programs"] = sorted(
            {str(r[0]) if isinstance(r, (list, tuple)) else str(r) for r in records}
        )
        return out


def arm_route(arm: str) -> str:
    if arm in SOTA_ARMS:
        return "sota"
    if arm in GRAPH_ARMS:
        return GRAPH_ARMS[arm]
    # nondet_p_<rule> / nondet_graph_p_<rule> run the non-deterministic route too
    return "det" if arm == "det" else "nondet"


def arm_ctx(arm: str) -> str:
    """The context an arm runs on: its route's, or (``nondet_p_<rule>``) a non-deterministic context built with the
    plan rule ``<rule>``."""
    rule = arm_p_rule(arm)
    if rule:
        return f"nondet:p_rule={rule}"
    if arm == "nondet_graph":  # the context the package documents for graph replay
        return "nondet:graph_replay"
    return arm_route(arm)


def arm_options(arm: str) -> Optional[dict[str, Any]]:
    """The ``options`` of an arm's GDN calls (``None``: the route's defaults)."""
    if is_graph(arm):
        return {"graph": "replay"}
    if arm in ("det", "nondet") or arm in SOTA_ARMS or PRULE_ARM.match(arm):
        return None
    raise ValueError(f"unknown arm {arm!r}")


# ----------------------------------------------------------------------------------------------------------------
# Rows and inputs


def _shapes() -> Any:
    try:
        return importlib.import_module("gdn_shapes")
    except ImportError as exc:
        raise SystemExit(
            f"the rows need the shape module gdn_shapes on PYTHONPATH: {exc}"
        ) from exc


def split_docs(seqlen: int, docs: int) -> tuple[int, ...]:
    """The deterministic benchmark's pack: ``docs`` unequal document lengths summing to ``seqlen``; no length is a
    multiple of 64 and at least one has ``length mod 64 < 32``."""
    if docs < 1 or seqlen < docs:
        raise ValueError(f"cannot split {seqlen} tokens into {docs} documents")
    weights = [1.0 + 0.37 * i for i in range(docs)]
    total = sum(weights)
    lengths = [max(1, int(seqlen * w / total)) for w in weights]
    lengths[-1] += seqlen - sum(lengths)
    for i in range(docs - 1):
        if lengths[i] % 64 == 0 and lengths[i + 1] > 1:
            lengths[i] += 1
            lengths[i + 1] -= 1
    return tuple(lengths)


@dataclasses.dataclass(frozen=True)
class RowSpec:
    name: str
    lengths: tuple[int, ...]
    heads: tuple[int, int]
    cp: Optional[tuple[int, ...]]  # the CP sizes the row runs at (None: any)
    layouts: tuple[str, ...]
    with_conv: bool
    source: str
    shapes_row: Any = None  # the gdn_shapes.Row (cut points of layout D)

    @property
    def cu(self) -> list[int]:
        cu = [0]
        for n in self.lengths:
            cu.append(cu[-1] + n)
        return cu

    def skip_reason(self, layout: str, world: int) -> Optional[str]:
        if self.cp is not None and world not in self.cp:
            return f"CP {world} is not in the row's CP set {list(self.cp)}"
        if layout not in self.layouts:
            return f"layout {layout} is not in the row's layouts {list(self.layouts)}"
        return None

    def cut_points(self, layout: str, world: int) -> Optional[list[list[int]]]:
        if layout != "D" or self.shapes_row is None:
            return None
        return [list(c) for c in _shapes().cut_points_D(self.shapes_row, world)]


def make_rows(names: list[str], args) -> list[RowSpec]:
    rows: list[RowSpec] = []
    for name in names:
        if name in SMALL_ROWS:
            lengths, heads = SMALL_ROWS[name]
            rows.append(
                RowSpec(
                    name,
                    lengths,
                    heads,
                    None,
                    tuple(LAYOUTS),
                    True,
                    f"small head pair {heads} on short shards: lengths {list(lengths)}",
                )
            )
            continue
        if name == DET_ROW:
            rows.append(
                RowSpec(
                    DET_ROW,
                    split_docs(args.seqlen, args.docs),
                    (args.qk_heads, args.v_heads),
                    None,
                    tuple(LAYOUTS),
                    True,
                    f"deterministic benchmark default pack: split_docs({args.seqlen}, {args.docs})",
                )
            )
            continue
        gs = _shapes()
        row = gs.get_row(name)
        if getattr(row, "regime", "normal") != "normal":
            raise SystemExit(
                f"row {name}: input regime {row.regime!r} is not implemented here (normal only)"
            )
        heads = (int(row.heads[0]), int(row.heads[1]))
        label = row.name
        if args.heads is not None and args.heads != heads:
            heads = args.heads  # the pack's documents with another head pair
            label = f"{row.name}h{heads[0]}x{heads[1]}"
        rows.append(
            RowSpec(
                label,
                tuple(int(n) for n in row.lengths),
                heads,
                tuple(int(w) for w in row.cp),
                tuple(row.layouts),
                bool(row.with_conv),
                f"gdn_shapes {gs.__file__} (seed {gs.SEED})",
                row,
            )
        )
    return rows


def _shard(x: torch.Tensor, cu: list[int], fragments) -> torch.Tensor:
    """The rows of ``x`` of this rank's fragments, in the partition's slot order (a fresh tensor)."""
    pieces = [
        x[cu[f.doc] + f.off : cu[f.doc] + f.off + f.length]
        for f in fragments
        if f.length > 0
    ]
    return torch.cat(pieces, 0) if pieces else x[:0].clone()


def make_local_inputs(
    cu: list[int],
    heads: tuple[int, int],
    channels: int,
    seed: int,
    fragments,
    device: torch.device,
    rows: Optional[torch.Tensor] = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """This rank's operands, cut from whole-pack tensors drawn on the device by one seeded generator (every rank draws
    the same pack; each whole-pack tensor is dropped right after the cut): the rows of ``fragments`` or, given
    ``rows`` (pack row indices on the device), those rows, zero past the pack's end.  ``q, k`` BF16 ``[T_r, H_qk, 128]`` (rows
    L2-normalised in FP32), ``v, dout`` BF16 ``[T_r, HV, 128]``, ``g`` FP32 ``[T_r, HV]`` (natural-log decay,
    ``<= 0``), ``beta`` FP32 ``[T_r, HV]`` in (0, 1), the convolution input ``x`` and its upstream gradient ``dy``
    BF16 ``[T_r, C]``, and the FP32 weight ``[C, 4]`` (whole)."""
    qk_heads, v_heads = heads
    total = cu[-1]
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen, device=device, dtype=torch.float32)

    def shard(x: torch.Tensor) -> torch.Tensor:
        if rows is None:
            return _shard(x, cu, fragments)
        out = x.new_zeros((rows.numel(), *x.shape[1:]))
        keep = rows < total
        out[keep] = x[rows[keep]]
        return out

    def l2(x: torch.Tensor) -> torch.Tensor:
        return (x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)).to(torch.bfloat16)

    def scaled_bf16(rows: int, cols: int, s: float) -> torch.Tensor:
        t = randn(rows, cols)
        t.mul_(s)
        return t.to(torch.bfloat16)

    a_log = torch.rand(
        v_heads, generator=gen, device=device, dtype=torch.float32
    ) * math.log(16.0)
    local = {
        "g": shard(
            -torch.exp(a_log) * torch.nn.functional.softplus(randn(total, v_heads))
        )
    }
    local["q"] = shard(l2(randn(total, qk_heads, HEAD_DIM)))
    local["k"] = shard(l2(randn(total, qk_heads, HEAD_DIM)))
    local["v"] = shard(randn(total, v_heads, HEAD_DIM).to(torch.bfloat16))
    local["beta"] = shard(torch.sigmoid(randn(total, v_heads)))
    local["dout"] = shard(randn(total, v_heads, HEAD_DIM).to(torch.bfloat16))
    local["x"] = shard(scaled_bf16(total, channels, 0.5))
    local["dy"] = shard(scaled_bf16(total, channels, 0.1))
    weight = randn(channels, WIDTH) * 0.3
    return local, weight


def make_problem(
    row: RowSpec,
    layout: str,
    args,
    routes: dict[str, Route],
    rank: int,
    world: int,
    device: torch.device,
) -> dict[str, Any]:
    """Partition and context of every route (one plan: the fragment tables must agree), the contexts of the
    ``nondet_p_<rule>`` arms (the non-deterministic partition, plan rule ``<rule>``) and this rank's operands."""
    cu = row.cu
    cu_cpu = torch.tensor(cu, dtype=torch.int64)
    cut = row.cut_points(layout, world)
    parts = {
        kind: route.partition(cu_cpu, world, LAYOUTS[layout], cut)
        for kind, route in routes.items()
    }
    ctxs = {
        kind: route.context(parts[kind], rank, device) for kind, route in routes.items()
    }
    for arm in args.arm_list:
        rule = arm_p_rule(arm)
        if rule and arm_ctx(arm) not in ctxs:
            ctxs[arm_ctx(arm)] = routes["nondet"].context(
                parts["nondet"], rank, device, p_rule=rule
            )
        if arm == "nondet_graph" and arm_ctx(arm) not in ctxs:
            ctxs[arm_ctx(arm)] = routes["nondet"].context(
                parts["nondet"], rank, device, graph_replay=True
            )
    tables = {
        kind: [
            [(int(f.doc), int(f.off), int(f.length)) for f in frs]
            for frs in part.fragments
        ]
        for kind, part in parts.items()
    }
    first = next(iter(tables))
    for kind, table in tables.items():
        if table != tables[first]:
            raise RuntimeError(
                f"{row.name}/{layout}: the {kind} partition differs from the {first} one; the arms would not run the "
                "same tokens"
            )
    local, weight = make_local_inputs(
        cu, row.heads, args.channels, args.seed, parts[first].fragments[rank], device
    )
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "cu": cu,
        "cut_points": cut,
        "ctx": ctxs,
        "fragments": tables[first],
        "local": local,
        "weight": weight,
        "tokens_total": cu[-1],
    }


# ----------------------------------------------------------------------------------------------------------------
# Steps: one callable per (arm, boundary)


def _leaves(local: dict[str, torch.Tensor], weight: torch.Tensor) -> list[torch.Tensor]:
    return [
        local[n].detach().requires_grad_(True)
        for n in ("q", "k", "v", "g", "beta", "x")
    ] + [weight.detach().requires_grad_(True)]


def _forward_fn(
    route: Route,
    ctx: Any,
    with_conv: bool,
    save_mode: str,
    options: Any,
    workspace: Any = None,
) -> Callable[..., tuple]:
    conv = conv_api.cake_causal_conv1d_kcp

    def forward(q, k, v, g, beta, x, w, handle=None):
        hk = {} if handle is None else {"recompute_handle": handle}
        if with_conv:
            y = conv(x, w, ctx, **hk)
            return y, route.chunk(
                q, k, v, g, beta, ctx, save_mode, options, handle, workspace
            )
        return (
            route.chunk(q, k, v, g, beta, ctx, save_mode, options, handle, workspace),
        )

    return forward


def _run_forward(
    boundary: str,
    forward: Callable[..., tuple],
    leaves: list[torch.Tensor],
    handle_cls: Optional[Callable[[], Any]],
) -> tuple:
    if boundary in ("B1", "B2"):
        return forward(*leaves)
    if boundary == "B3":
        return checkpoint(forward, *leaves, use_reentrant=False)
    return checkpoint(forward, *leaves, handle_cls(), use_reentrant=False)


def _step(forward, leaves_of, grads_out, boundary, handle_cls) -> Callable[[], None]:
    def step() -> None:
        with torch.enable_grad():
            outs = _run_forward(boundary, forward, leaves_of(), handle_cls)
            if boundary != "B1":
                torch.autograd.backward(list(outs), grads_out)

    return step


def _probe(
    arm: str,
    forward,
    leaves_of,
    grads_out,
    boundary: str,
    handle_cls,
    routes: dict[str, Route],
    device: torch.device,
    lib: Optional[_Fla] = None,
) -> dict[str, Any]:
    """One call of ``(arm, boundary)`` with its phases split: peak memory over the allocation before the call
    (forward, backward), the bytes retained after the forward (outputs and what the forward keeps), and the
    ``LAST_CALL`` records of the forward, the checkpoint recompute and the backward.  Every route's ``LAST_CALL`` is
    cleared first: the arm must have written its own route's records and no other.  A non-Cake arm (``lib``: its
    library) must have made its library calls -- the GDN call, and the convolution with two upstream gradients --
    produced every input gradient after a backward, and written no route's records."""
    for route in routes.values():
        route.clear_last_call()
    sota = arm in SOTA_ARMS
    if sota and lib is None:
        raise ValueError(f"arm {arm}: the probe of a non-Cake arm needs its library")
    own = None if sota else routes[arm_route(arm)]
    before = dict(lib.calls) if sota else {}
    missing: list[int] = []
    torch.cuda.synchronize(device)
    base = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    mem: dict[str, int] = {}
    calls: dict[str, Any] = {}
    with torch.enable_grad():
        leaves = leaves_of()
        outs = _run_forward(boundary, forward, leaves, handle_cls)
        torch.cuda.synchronize(device)
        mem["fwd_peak_bytes"] = torch.cuda.max_memory_allocated(device) - base
        mem["fwd_retained_bytes"] = torch.cuda.memory_allocated(device) - base
        calls["fwd"] = None if own is None else own.last_call("fwd")
        if boundary != "B1":
            torch.cuda.reset_peak_memory_stats(device)
            torch.autograd.backward(list(outs), grads_out)
            torch.cuda.synchronize(device)
            mem["bwd_peak_bytes"] = torch.cuda.max_memory_allocated(device) - base
            if own is not None:
                if boundary in ("B3", "B3h"):
                    calls["fwd_recompute"] = own.last_call("fwd")
                calls["bwd"] = own.last_call("bwd")
            n_in = len(leaves) if len(grads_out) == 2 else 5
            missing = [i for i in range(n_in) if leaves[i].grad is None]
        del outs, leaves
    torch.cuda.synchronize(device)
    mem["after_bytes"] = torch.cuda.memory_allocated(device) - base
    mem["step_peak_bytes"] = max(mem["fwd_peak_bytes"], mem.get("bwd_peak_bytes", 0))
    ran = sorted(kind for kind, route in routes.items() if route.pkg.LAST_CALL)
    if sota:
        made = {key: lib.calls[key] - before[key] for key in lib.calls}
        need = (SOTA_GDN_CALL[arm],) + (("conv",) if len(grads_out) == 2 else ())
        if ran or missing or any(made[key] < 1 for key in need):
            raise RuntimeError(
                f"arm {arm}: the probe call made the library calls {made} (needs {list(need)}), left the input "
                f"gradients {missing} unset and wrote LAST_CALL records of the routes {ran} (expected none)"
            )
        return {
            "memory": mem,
            "last_call": {},
            "routes_recorded": ran,
            "sota_calls": made,
        }
    if ran != [own.kind] or calls["fwd"] is None:
        raise RuntimeError(
            f"arm {arm}: the probe call wrote LAST_CALL records of the routes {ran}, expected exactly {[own.kind]}"
        )
    return {"memory": mem, "last_call": calls, "routes_recorded": ran}


# ----------------------------------------------------------------------------------------------------------------
# Non-Cake arms: flash-linear-attention's context parallelism, and head parallelism around its single-rank kernels


class _Fla:
    """flash-linear-attention's ops (``fla-core``) on their Triton kernels: ``FLA_FLASH_QLA=0`` is set before the
    import, so no call dispatches to FlashQLA; with ``frost``, cuDNN frontend's GDN too.  Every call is counted
    (:func:`_probe` checks them)."""

    def __init__(self, frost: bool = False) -> None:
        os.environ["FLA_FLASH_QLA"] = "0"
        try:
            import fla  # type: ignore
            from fla.modules.causal_conv1d import causal_conv1d  # type: ignore
            from fla.ops.cp.context import build_cp_context  # type: ignore
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # type: ignore
        except Exception as exc:  # noqa: BLE001 -- a requested arm must run: any import failure stops the run
            raise SystemExit(
                f"the non-Cake arms need flash-linear-attention (fla-core) importable: {exc!r}"
            ) from exc
        self._conv = causal_conv1d
        self._chunk = chunk_gated_delta_rule
        self.build_cp_context = build_cp_context
        self.calls = {"chunk": 0, "conv": 0, "frost": 0}
        self.files = {
            "fla": fla.__file__,
            "conv": sys.modules[causal_conv1d.__module__].__file__,
            "chunk": sys.modules[chunk_gated_delta_rule.__module__].__file__,
            "cp_context": sys.modules[build_cp_context.__module__].__file__,
        }
        self._frost = None
        if frost:
            try:
                from cudnn.linear_attention.ops import gated_delta_net  # type: ignore
            except Exception as exc:  # noqa: BLE001 -- a requested arm must run
                raise SystemExit(
                    f"the cuDNN arms need nvidia-cudnn-frontend importable: {exc!r}"
                ) from exc
            self._frost = gated_delta_net
            self.files["frost"] = sys.modules[gated_delta_net.__module__].__file__

    def versions(self) -> dict[str, Any]:
        from importlib import metadata

        def version(name: str) -> Optional[str]:
            try:
                return metadata.version(name)
            except metadata.PackageNotFoundError:
                return None

        return {
            "fla-core": version("fla-core"),
            "flash-linear-attention": version("flash-linear-attention"),
            "triton": version("triton"),
            "nvidia-cudnn-frontend": version("nvidia-cudnn-frontend"),
            "nvidia-cutlass-dsl": version("nvidia-cutlass-dsl"),
            "FLA_FLASH_QLA": os.environ.get("FLA_FLASH_QLA"),
            "files": self.files,
        }

    def conv(self, x: torch.Tensor, w: torch.Tensor, **kw: Any) -> torch.Tensor:
        self.calls["conv"] += 1
        y, _ = self._conv(x.unsqueeze(0), w, activation="silu", **kw)
        return y[0]

    def chunk(self, q, k, v, g, beta, **kw: Any) -> torch.Tensor:
        self.calls["chunk"] += 1
        out, _ = self._chunk(
            *(t.unsqueeze(0) for t in (q, k, v, g, beta)), scale=SCALE, **kw
        )
        return out[0]

    def frost(
        self, q, k, v, g, beta, cu_seqlens: torch.Tensor, ck: int
    ) -> torch.Tensor:
        self.calls["frost"] += 1
        out = self._frost(
            q,
            k,
            v,
            g,
            beta,
            cu_seqlens,
            scale=SCALE,
            use_qk_l2norm_in_kernel=False,
            batch_invariant=False,
            checkpoint_every_n_tokens=ck,
            plan_name="gdn_frost",
        )
        return out[0] if isinstance(out, (tuple, list)) else out


def _all_to_all(x: torch.Tensor, world: int, to_heads: bool) -> torch.Tensor:
    """Sequence <-> head redistribution (rank ``r``: tokens ``[r t, (r + 1) t)`` of the padded pack, heads
    ``[r h, (r + 1) h)``): ``[t, W h, ...]`` -> ``[W t, h, ...]`` (``to_heads``) or back."""
    if world == 1:
        return x
    rest = tuple(x.shape[2:])
    if to_heads:
        t, h = x.shape[0], x.shape[1] // world
        send = x.reshape(t, world, h, *rest).transpose(0, 1).contiguous()
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send)
        return recv.reshape(world * t, h, *rest)
    t, h = x.shape[0] // world, x.shape[1]
    send = x.contiguous()
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send)
    return recv.reshape(world, t, h, *rest).transpose(0, 1).reshape(t, world * h, *rest)


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, world: int, to_heads: bool) -> torch.Tensor:  # type: ignore[override]
        ctx.world, ctx.to_heads = world, to_heads
        return _all_to_all(x, world, to_heads)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):  # type: ignore[override]
        return _all_to_all(grad.contiguous(), ctx.world, not ctx.to_heads), None, None


def sota_unavailable(
    arm: str,
    layout: str,
    heads: tuple[int, int],
    channels: int,
    with_conv: bool,
    world: int,
) -> Optional[str]:
    """Why a non-Cake arm has no form for (layout, heads, channels, world), or ``None``."""
    if arm == "sota_fla_kcp":
        if layout not in FLA_KCP_LAYOUTS:
            return f"layout {layout} ({LAYOUTS[layout]}) has no FLA context-parallel form (FLA: contiguous, zigzag)"
        return None
    sizes = [("H_qk", heads[0]), ("HV", heads[1])] + (
        [("C", channels)] if with_conv else []
    )
    bad = [f"{name}={n}" for name, n in sizes if n % world]
    if bad:
        return (
            f"head parallelism needs the world size {world} to divide {', '.join(bad)}"
        )
    return None


def make_sota_problem(
    arm: str,
    row: RowSpec,
    layout: str,
    args,
    lib: _Fla,
    rank: int,
    world: int,
    device: torch.device,
) -> dict[str, Any]:
    """A non-Cake arm's tokens: the pack padded with one zero document (to a multiple of ``2 x world`` tokens for
    FLA's context parallelism, of ``world`` for head parallelism), this rank's rows of it (FLA's even split:
    contiguous, or the zigzag pieces ``rank`` and ``2 W - 1 - rank``; checked against FLA's local ``cu_seqlens``),
    its operands (the same seeded pack as the Cake arms) and, for ``sota_fla_kcp``, FLA's context."""
    cu = list(row.cu)
    total = cu[-1]
    pad = (-total) % (2 * world if arm == "sota_fla_kcp" else world)
    cu_pad = cu + ([total + pad] if pad else [])
    padded = total + pad
    cu_cpu = torch.tensor(cu_pad, dtype=torch.long)
    cu_dev = cu_cpu.to(device)
    fla_ctx = None
    if arm == "sota_fla_kcp" and layout == "Z":
        piece = padded // (2 * world)
        rows = torch.cat(
            [
                torch.arange(rank * piece, (rank + 1) * piece),
                torch.arange(
                    (2 * world - 1 - rank) * piece, (2 * world - rank) * piece
                ),
            ]
        )
    else:
        share = padded // world
        rows = torch.arange(rank * share, (rank + 1) * share)
    if arm == "sota_fla_kcp":
        fla_ctx = lib.build_cp_context(
            cu_dev,
            dist.group.WORLD,
            conv1d_kernel_size=WIDTH,
            cu_seqlens_cpu=cu_cpu,
            use_tf32x3_affine_chain=True,
            layout=FLA_KCP_LAYOUTS[layout],
        )
        t_r = int(fla_ctx.cu_seqlens_cpu[-1])
        if t_r != rows.numel():
            raise RuntimeError(
                f"{arm} {row.name}/{layout}: FLA gives rank {rank} {t_r} tokens, the even split {rows.numel()}"
            )
    rows = rows.to(device)
    local, weight = make_local_inputs(
        cu, row.heads, args.channels, args.seed, None, device, rows=rows
    )
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "rows": rows,
        "pad": pad,
        "tokens_padded": padded,
        "cu_cpu": cu_cpu,
        "cu_dev": cu_dev,
        "cu_dev32": cu_cpu.to(device=device, dtype=torch.int32),
        "fla_ctx": fla_ctx,
        "local": local,
        "weight": weight,
    }


def sota_forward(
    arm: str, lib: _Fla, sp: dict[str, Any], with_conv: bool, rank: int, world: int
) -> Callable[..., tuple]:
    """``forward(q, k, v, g, beta, x, w) -> (y, out)`` (``(out,)`` without the convolution) of a non-Cake arm on its
    rank's rows."""
    if arm == "sota_fla_kcp":
        fla_ctx = sp["fla_ctx"]

        def forward_cp(q, k, v, g, beta, x, w):
            out = lib.chunk(q, k, v, g, beta, cp_context=fla_ctx)
            if with_conv:
                return lib.conv(x, w, cp_context=fla_ctx), out
            return (out,)

        return forward_cp
    varlen = {"cu_seqlens": sp["cu_dev"], "cu_seqlens_cpu": sp["cu_cpu"]}

    def heads_of(t: torch.Tensor) -> torch.Tensor:
        return _AllToAll.apply(t, world, True)

    def tokens_of(t: torch.Tensor) -> torch.Tensor:
        return _AllToAll.apply(t, world, False)

    ck = FROST_CHECKPOINT.get(arm)
    cu32 = sp["cu_dev32"]

    def gdn(q, k, v, g, beta):
        if ck is None:
            return lib.chunk(q, k, v, g, beta, **varlen)
        return lib.frost(q, k, v, g, beta, cu32, ck)

    def forward_heads(q, k, v, g, beta, x, w):
        out = tokens_of(gdn(*(heads_of(t) for t in (q, k, v, g, beta))))
        if with_conv:
            c = w.shape[0] // world
            y = lib.conv(heads_of(x), w[rank * c : (rank + 1) * c], **varlen)
            return tokens_of(y), out
        return (out,)

    return forward_heads


GRAD_NAMES = ("dq", "dk", "dv", "dg", "dbeta", "dx", "dw")


def _outputs_and_grads(
    forward: Callable[..., tuple],
    leaves: list[torch.Tensor],
    grads_out,
    with_conv: bool,
) -> dict[str, torch.Tensor]:
    """One forward + backward: the outputs and every input gradient."""
    with torch.enable_grad():
        outs = forward(*leaves)
        torch.autograd.backward(list(outs), grads_out)
    res = {
        name: o.detach()
        for name, o in zip(("y", "out") if with_conv else ("out",), outs, strict=True)
    }
    for name, leaf in zip(GRAD_NAMES, leaves, strict=True):
        if leaf.grad is not None:
            res[name] = leaf.grad
    return res


def _cake_rows(
    fragments: list[tuple[int, int, int]], cu: list[int], device
) -> torch.Tensor:
    """The pack rows of a rank's fragments, in slot order."""
    pieces = [torch.arange(cu[d] + o, cu[d] + o + n) for d, o, n in fragments if n > 0]
    return (torch.cat(pieces) if pieces else torch.zeros(0, dtype=torch.long)).to(
        device
    )


def sota_agreement(
    ref: dict[str, torch.Tensor],
    ref_rows: torch.Tensor,
    cand: dict[str, torch.Tensor],
    cand_rows: torch.Tensor,
    total: int,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """Per tensor of ``ref`` (``nondet``), the non-Cake arm's error on the unpadded pack: the per-token tensors
    gathered by pack row (every row is owned by one rank on each side; a block of columns at a time, at most
    :data:`AGREEMENT_BLOCK_ELEMS` FP32 elements of the whole pack), ``dw`` summed over the ranks on both sides."""
    out: dict[str, dict[str, float]] = {}
    keep = cand_rows < total
    kept_rows = cand_rows[keep]
    for name, r_local in ref.items():
        if name not in cand:
            raise RuntimeError(f"the non-Cake arm produced no {name}")
        sums = torch.zeros(3, dtype=torch.float64, device=device)
        maxes = torch.zeros(2, dtype=torch.float32, device=device)

        def add(c: torch.Tensor, r: torch.Tensor) -> None:
            err = c - r
            sums[0] += err.double().square().sum()
            sums[1] += r.double().square().sum()
            sums[2] += (~torch.isfinite(c)).sum()
            if err.numel():
                maxes[0] = torch.maximum(maxes[0], err.abs().max())
                maxes[1] = torch.maximum(maxes[1], r.abs().max())

        if name == "dw":
            r = r_local.float().clone()
            c = cand[name].float().clone()
            dist.all_reduce(r)
            dist.all_reduce(c)
            add(c, r)
        else:
            r_flat = r_local.reshape(r_local.shape[0], -1)
            c_flat = cand[name][keep].reshape(int(kept_rows.numel()), -1)
            cols = r_flat.shape[1]
            step = max(1, AGREEMENT_BLOCK_ELEMS // max(total, 1))
            for c0 in range(0, cols, step):
                c1 = min(cols, c0 + step)
                full = torch.zeros((total, c1 - c0), dtype=torch.float32, device=device)
                full[ref_rows] = r_flat[:, c0:c1].float()
                dist.all_reduce(full)
                add(c_flat[:, c0:c1].float(), full[kept_rows])
                del full
            dist.all_reduce(sums)
            dist.all_reduce(maxes, op=dist.ReduceOp.MAX)
        e2, r2, nonfinite = sums.tolist()
        max_err, max_ref = maxes.tolist()
        if nonfinite:
            max_err = math.inf
        out[name] = {
            "rel_rms": math.sqrt(e2 / r2) if r2 > 0 else (0.0 if e2 == 0 else math.inf),
            "max_abs": max_err,
            "max_ref": max_ref,
            "max_abs_over_max_ref": max_err / max_ref
            if max_ref > 0
            else (0.0 if max_err == 0 else math.inf),
            "nonfinite": int(nonfinite),
        }
    return out


def sota_qualified(acc: dict[str, dict[str, float]]) -> tuple[bool, list[str]]:
    """Whether every tensor agrees within :data:`SOTA_REL_RMS` / :data:`SOTA_MAX_ABS` (and the failing ones)."""
    bad = [
        name
        for name, a in acc.items()
        if a["nonfinite"]
        or not a["rel_rms"] <= SOTA_REL_RMS
        or not a["max_abs_over_max_ref"] <= SOTA_MAX_ABS
    ]
    return not bad, bad


# ----------------------------------------------------------------------------------------------------------------
# Timing


def _time_block(
    step: Callable[[], None],
    samples: int,
    warmup: int,
    flush: Optional[torch.Tensor],
    device: torch.device,
    world: int,
) -> list[list[float]]:
    """Every rank's milliseconds of ``samples`` timed iterations (after ``warmup`` untimed ones), ``[world][samples]``;
    every iteration after an optional L2 flush, a device synchronize and a barrier."""
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(samples)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(samples)]
    for i in range(warmup + samples):
        if flush is not None:
            flush.fill_(i & 0xFF)
        torch.cuda.synchronize(device)
        dist.barrier()
        j = i - warmup
        if j >= 0:
            starts[j].record()
        step()
        if j >= 0:
            ends[j].record()
        torch.cuda.synchronize(device)
    mine = torch.tensor(
        [s.elapsed_time(e) for s, e in zip(starts, ends, strict=True)],
        dtype=torch.float64,
        device=device,
    )
    every = torch.empty(world * samples, dtype=torch.float64, device=device)
    dist.all_gather_into_tensor(every, mine)
    return every.view(world, samples).tolist()


def _measure(
    steps: dict[str, Callable[[], None]], arms: list[str], args, flush, device, world
) -> dict[str, list]:
    """The palindromic pattern ``[a1 .. an, an .. a1] x repeat`` in each of ``groups`` groups; per call the
    per-rank times, the slowest rank's time per iteration and the first steady sample (the deterministic benchmark's
    rule: every sample is steady after warmup iterations or after a call of the same arm)."""
    pattern = (arms + arms[::-1]) * args.repeat
    calls: dict[str, list] = {arm: [] for arm in arms}
    for group in range(args.groups):
        previous: Optional[str] = None
        for index, arm in enumerate(pattern):
            if args.cool_s > 0:
                torch.cuda.synchronize(device)
                time.sleep(args.cool_s)
                dist.barrier()
            t0 = time.time()
            per_rank = _time_block(
                steps[arm], args.samples, args.warmup, flush, device, world
            )
            t1 = time.time()
            slowest = [max(col) for col in zip(*per_rank, strict=True)]
            steady_all = previous == arm or (
                previous is None and group == 0 and index == 0
            )
            calls[arm].append(
                {
                    "group": group,
                    "index": index,
                    "previous": previous,
                    "steady_from": 0 if steady_all or args.warmup > 0 else 1,
                    "slowest_ms": slowest,
                    "per_rank_ms": per_rank,
                    "t_unix": [t0, t1],
                }
            )
            previous = arm
    return calls


def _quantile(xs: list[float], q: float) -> float:
    s = sorted(xs)
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def _summarise(
    calls: list[dict[str, Any]], groups: int, world: int, tokens: int
) -> dict[str, Any]:
    steady: list[float] = []
    per_group: list[list[float]] = [[] for _ in range(groups)]
    per_rank: list[list[float]] = [[] for _ in range(world)]
    for c in calls:
        s = c["steady_from"]
        steady.extend(c["slowest_ms"][s:])
        per_group[c["group"]].extend(c["slowest_ms"][s:])
        for r in range(world):
            per_rank[r].extend(c["per_rank_ms"][r][s:])
    group_medians = [statistics.median(x) for x in per_group]
    median = statistics.median(steady)
    median_of_groups = statistics.median(group_medians)
    rank_medians = [statistics.median(x) for x in per_rank]
    return {
        "median_ms": median,
        "p10_ms": _quantile(steady, 0.1),
        "p90_ms": _quantile(steady, 0.9),
        "min_ms": min(steady),
        "max_ms": max(steady),
        "n_steady": len(steady),
        "group_medians_ms": group_medians,
        "block_medians_ms": [
            statistics.median(c["slowest_ms"][c["steady_from"] :]) for c in calls
        ],
        "median_of_group_medians_ms": median_of_groups,
        "drift": abs(group_medians[-1] - group_medians[0]) / median_of_groups,
        "per_rank_median_ms": rank_medians,
        "slowest_rank_by_median": max(range(world), key=lambda r: rank_medians[r]),
        "tokens_per_s": tokens / (median * 1e-3),
        "slowest_ms_steady": steady,
        "calls": calls,
    }


def _ratios(stats: dict[str, dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for arm, st in stats.items():
        if arm == "det":
            continue
        if "det" in stats:
            out[f"{arm}/det"] = st["median_ms"] / stats["det"]["median_ms"]
        if arm in SOTA_ARMS:
            if "nondet" in stats:
                out[f"{arm}/nondet"] = st["median_ms"] / stats["nondet"]["median_ms"]
    return out


def _plan_summary(probe_ranks: list[dict[str, Any]]) -> str:
    """The launch recorder's view of the arm's plan: the forward's ``P`` per rank and the largest launch count of the
    forward and the backward over the ranks (``LAST_CALL``)."""
    Ps, launches = [], {}
    for pr in probe_ranks:
        rec = pr["last_call"].get("fwd") or {}
        Ps.append(str(rec.get("P", "?")))
        for phase in ("fwd", "bwd"):
            r = pr["last_call"].get(phase) or {}
            if "n_launches" in r:
                launches[phase] = max(launches.get(phase, 0), int(r["n_launches"]))
    n = " ".join(f"{k}={v}" for k, v in sorted(launches.items()))
    return f"P[{','.join(Ps)}] launches[{n}]"


# ----------------------------------------------------------------------------------------------------------------
# One (row, layout, CP) configuration


def run_config(
    row: RowSpec,
    layout: str,
    args,
    arms: list[str],
    routes: dict[str, Route],
    env: dict[str, Any],
    flush: Optional[torch.Tensor],
    rank: int,
    world: int,
    device: torch.device,
    lib: Optional[_Fla] = None,
) -> dict[str, Any]:
    t0 = time.time()
    with_conv = row.with_conv and not args.no_conv
    h_qk, hv = row.heads
    head = (
        f"KCP x{world} {row.name}/{layout}: S={row.cu[-1]} docs={len(row.lengths)} H_qk={h_qk} HV={hv} "
        f"C={args.channels} conv={'on' if with_conv else 'off'} save_mode={args.save_mode} groups={args.groups} "
        f"repeat={args.repeat} samples={args.samples} warmup={args.warmup} l2_flush={'on' if flush is not None else 'off'}"
    )
    if rank == 0:
        print(f"{head} -- building the problem", flush=True)
    problem = make_problem(row, layout, args, routes, rank, world, device)
    local, weight = problem["local"], problem["weight"]
    grads_out = [local["dy"], local["dout"]] if with_conv else [local["dout"]]

    def leaves_of() -> list[torch.Tensor]:
        return _leaves(local, weight)

    forwards = {
        arm: _forward_fn(
            routes[arm_route(arm)],
            problem["ctx"][arm_ctx(arm)],
            with_conv,
            args.save_mode,
            arm_options(arm),
            routes[arm_route(arm)].workspace(
                problem["ctx"][arm_ctx(arm)], row.heads, device
            )
            if is_graph(arm)
            else None,
        )
        for arm in arms
        if arm not in SOTA_ARMS
    }
    leaves_by_arm = {arm: leaves_of for arm in forwards}
    grads_by_arm = {arm: grads_out for arm in forwards}
    sota: dict[str, dict[str, Any]] = {}
    sota_arms = [arm for arm in arms if arm in SOTA_ARMS]
    if sota_arms:
        if lib is None or "nondet" not in forwards:
            raise ValueError("the non-Cake arms need their library and the nondet arm")
        ref = _outputs_and_grads(forwards["nondet"], leaves_of(), grads_out, with_conv)
        ref_rows = _cake_rows(problem["fragments"][rank], problem["cu"], device)
        for arm in sota_arms:
            why = sota_unavailable(
                arm, layout, row.heads, args.channels, with_conv, world
            )
            if why is not None:
                sota[arm] = {"available": False, "reason": why}
                continue
            sp = make_sota_problem(arm, row, layout, args, lib, rank, world, device)
            forwards[arm] = sota_forward(arm, lib, sp, with_conv, rank, world)
            sl, sw = sp["local"], sp["weight"]
            leaves_by_arm[arm] = lambda sl=sl, sw=sw: _leaves(sl, sw)
            grads_by_arm[arm] = [sl["dy"], sl["dout"]] if with_conv else [sl["dout"]]
            cand = _outputs_and_grads(
                forwards[arm], leaves_by_arm[arm](), grads_by_arm[arm], with_conv
            )
            acc = sota_agreement(
                ref, ref_rows, cand, sp["rows"], problem["tokens_total"], device
            )
            ok, failing = sota_qualified(acc)
            del cand
            sota[arm] = {
                "available": True,
                "qualified": ok,
                "failing": failing,
                "accuracy_vs_nondet": acc,
                "budget": {
                    "rel_rms": SOTA_REL_RMS,
                    "max_abs_over_max_ref": SOTA_MAX_ABS,
                },
                "pad_tokens": sp["pad"],
                "tokens_padded": sp["tokens_padded"],
                "tokens_per_rank": int(sp["rows"].numel()),
            }
            if rank == 0:
                worst = max(acc, key=lambda n: acc[n]["rel_rms"])
                print(
                    f"  {arm} accuracy vs nondet: {'qualified' if ok else 'NOT qualified ' + ','.join(failing)}; "
                    f"worst rel-RMS {acc[worst]['rel_rms']:.3e} ({worst}), worst max-abs/max|ref| "
                    f"{max(a['max_abs_over_max_ref'] for a in acc.values()):.3e}; pad {sp['pad']} tokens",
                    flush=True,
                )
        del ref
        gc.collect()
        torch.cuda.empty_cache()
    handles = {kind: route.handle_class() for kind, route in routes.items()}
    heads = {"H_qk": h_qk, "HV": hv}
    my_bytes = {
        kind: route.bytes(problem["ctx"][kind], heads, args.channels, args.save_mode)
        for kind, route in routes.items()
    }
    bytes_ranks: list[Any] = [None] * world
    dist.all_gather_object(bytes_ranks, my_bytes)
    report: dict[str, Any] = {
        "bench": Path(__file__).name,
        "config": {
            "row": row.name,
            "row_source": row.source,
            "lengths": list(row.lengths),
            "cu_seqlens": problem["cu"],
            "tokens_total": problem["tokens_total"],
            **heads,
            "channels": args.channels,
            "conv": with_conv,
            "layout": layout,
            "layout_name": LAYOUTS[layout],
            "cut_points": problem["cut_points"],
            "world_size": world,
            "arms": arms,
            "arm_route": {arm: arm_route(arm) for arm in arms},
            "arm_options": {arm: arm_options(arm) for arm in arms},
            "boundaries": args.boundary_list,
            "pattern": (arms + arms[::-1]) * args.repeat,
            "groups": args.groups,
            "repeat": args.repeat,
            "samples": args.samples,
            "warmup": args.warmup,
            "cool_s": args.cool_s,
            "l2_flush": flush is not None,
            "l2_flush_bytes": L2_FLUSH_BYTES if flush is not None else 0,
            "seed": args.seed,
            "save_mode": args.save_mode,
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "env": env,
        "modules": {arm: env["ranks"][0]["files"][arm_route(arm)] for arm in arms},
        "partition": {
            "fragments": problem["fragments"],
            "tokens_per_rank": [sum(f[2] for f in frs) for frs in problem["fragments"]],
        },
        "bytes": bytes_ranks,
        "sota": sota,
        "boundaries": {},
    }
    drift_ok = True
    for boundary in args.boundary_list:
        available: list[str] = []
        unavailable: dict[str, str] = {}
        for arm in arms:
            if arm in SOTA_ARMS:
                if not sota[arm]["available"]:
                    unavailable[arm] = sota[arm]["reason"]
                elif boundary == "B3h":
                    unavailable[arm] = (
                        "no recompute-handle form (its checkpointed step is B3)"
                    )
                else:
                    available.append(arm)
                continue
            cls, why = handles[arm_route(arm)]
            if boundary == "B3h" and is_graph(arm):
                unavailable[arm] = "a recompute handle needs graph='off'"
                continue
            if boundary == "B3h" and cls is None:
                unavailable[arm] = why or "no recompute handle"
            else:
                available.append(arm)
        if not available:
            report["boundaries"][boundary] = {
                "label": BOUNDARY_LABELS[boundary],
                "arms": {},
                "unavailable": unavailable,
            }
            continue
        steps = {
            arm: _step(
                forwards[arm],
                leaves_by_arm[arm],
                grads_by_arm[arm],
                boundary,
                None if arm in SOTA_ARMS else handles[arm_route(arm)][0],
            )
            for arm in available
        }
        # one untimed call per arm: JIT builds and the first exchanges
        for arm in available:
            steps[arm]()
        torch.cuda.synchronize(device)
        dist.barrier()
        probes: dict[str, list[Any]] = {}
        for arm in available:
            mine = _probe(
                arm,
                forwards[arm],
                leaves_by_arm[arm],
                grads_by_arm[arm],
                boundary,
                None if arm in SOTA_ARMS else handles[arm_route(arm)][0],
                routes,
                device,
                lib,
            )
            gathered: list[Any] = [None] * world
            dist.all_gather_object(gathered, mine)
            probes[arm] = gathered
        torch.cuda.synchronize(device)
        dist.barrier()
        calls = _measure(steps, available, args, flush, device, world)
        stats = {
            arm: _summarise(calls[arm], args.groups, world, problem["tokens_total"])
            for arm in available
        }
        ratios = _ratios(stats)
        for arm, st in stats.items():
            st["drift_ok"] = st["drift"] <= args.max_drift
            drift_ok = drift_ok and st["drift_ok"]
            st["memory"] = [p["memory"] for p in probes[arm]]
            st["memory_max_over_ranks"] = {
                key: max(p["memory"].get(key, 0) for p in probes[arm])
                for key in probes[arm][0]["memory"]
            }
            st["last_call"] = [p["last_call"] for p in probes[arm]]
            st["routes_recorded"] = probes[arm][0]["routes_recorded"]
            if arm in SOTA_ARMS:
                st["sota_calls"] = [p["sota_calls"] for p in probes[arm]]
                st["qualified"] = sota[arm]["qualified"]
        report["boundaries"][boundary] = {
            "label": BOUNDARY_LABELS[boundary],
            "arms": stats,
            "ratios": ratios,
            "unavailable": unavailable,
        }
        if rank == 0:
            print(f"{head} {boundary} ({BOUNDARY_LABELS[boundary]})", flush=True)
            for arm, st in stats.items():
                m = st["memory_max_over_ranks"]
                tail = (
                    f"library calls {probes[arm][0]['sota_calls']}  "
                    f"{'qualified' if st['qualified'] else 'NOT QUALIFIED'}"
                    if arm in SOTA_ARMS
                    else _plan_summary(probes[arm])
                )
                print(
                    f"  {arm:16s} {st['median_ms']:9.3f} ms  p10 {st['p10_ms']:9.3f}  p90 {st['p90_ms']:9.3f}  "
                    f"{st['tokens_per_s'] / 1e6:8.3f} Mtok/s  peak {m['step_peak_bytes'] / MIB:9.1f} MiB  "
                    f"kept {m['fwd_retained_bytes'] / MIB:9.1f} MiB  drift {st['drift'] * 100:5.2f}%"
                    f"{'' if st['drift_ok'] else ' (ABOVE LIMIT)'}  {tail}",
                    flush=True,
                )
            for arm, why in unavailable.items():
                print(f"  {arm:16s} unavailable: {why}", flush=True)
            if ratios:
                print(
                    "  " + "  ".join(f"{k} = {v:.4f}" for k, v in ratios.items()),
                    flush=True,
                )
        del steps, probes, calls
    if any(is_graph(arm) for arm in arms):
        del forwards
        for kind in {arm_route(arm) for arm in arms if is_graph(arm)}:
            routes[kind].graph_cache_clear()
    report["drift_ok"] = drift_ok
    report["max_drift"] = args.max_drift
    report["wall_s"] = time.time() - t0
    return report


# ----------------------------------------------------------------------------------------------------------------
# Process, environment, CLI


def _init() -> tuple[int, int, torch.device]:
    """The process group of ``torchrun``: rank, world size and this rank's device (``LOCAL_RANK``)."""
    local = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local % torch.cuda.device_count())
    torch.cuda.set_device(device)
    if not dist.is_initialized():
        dist.init_process_group("nccl", device_id=device)
    return dist.get_rank(), dist.get_world_size(), device


def _env(
    routes: dict[str, Route],
    device: torch.device,
    world: int,
    lib: Optional[_Fla] = None,
) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(device)
    me = {
        "rank": dist.get_rank(),
        "host": socket.gethostname(),
        "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
        "device_index": device.index,
        "device": torch.cuda.get_device_name(device),
        "uuid": str(getattr(props, "uuid", "")),
        "pid": os.getpid(),
        "files": {
            "flashinfer": flashinfer.__file__,
            "conv": conv_api.__file__,
            **{kind: route.modules() for kind, route in routes.items()},
            **({"sota": lib.files} if lib is not None else {}),
        },
    }
    ranks: list[Any] = [None] * world
    dist.all_gather_object(ranks, me)
    host_map: dict[str, list[int]] = {}
    for r in ranks:
        host_map.setdefault(r["host"], []).append(r["rank"])
    nccl = torch.cuda.nccl.version()
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nccl": ".".join(str(x) for x in nccl)
        if isinstance(nccl, tuple)
        else str(nccl),
        "python": sys.version.split()[0],
        "device": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "sm_count": props.multi_processor_count,
        "total_memory_bytes": props.total_memory,
        "world_size": world,
        "host_map": host_map,
        "ranks": ranks,
        "argv": [sys.executable, *sys.argv],
        "cwd": os.getcwd(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "sota": lib.versions() if lib is not None else None,
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _csv(text: str) -> list[str]:
    out: list[str] = []
    for item in text.split(","):
        item = item.strip()
        if item and item not in out:
            out.append(item)
    return out


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KCP packed GDN training benchmark: deterministic and non-deterministic routes"
    )
    parser.add_argument(
        "--rows",
        default=",".join(DEFAULT_ROWS),
        help=f"comma-separated gdn_shapes rows, {DET_ROW} and/or the small-pair rows "
        f"{', '.join(SMALL_ROWS)} (default %(default)s)",
    )
    parser.add_argument(
        "--layouts", default="C,Z,D", help="comma-separated layouts among C, Z, D"
    )
    parser.add_argument(
        "--arms",
        default=",".join(BASE_ARMS),
        help="comma-separated arms among det, nondet, nondet_p_<plan rule>, det_graph, "
        "nondet_graph, nondet_graph_p_<plan rule> and the "
        "non-Cake arms sota_fla_kcp, sota_head_cp_fla, sota_head_cp_frost, sota_head_cp_frost_ck64 (these need nondet)",
    )
    parser.add_argument(
        "--p-rules",
        default="",
        help="comma-separated plan rules of the non-deterministic package; adds the arms nondet_p_<rule> (contexts "
        "built with that rule)",
    )
    parser.add_argument(
        "--graph-p-rules",
        default="",
        help="comma-separated plan rules of the non-deterministic package; adds the arms nondet_graph_p_<rule> (graph "
        "replay on contexts built with that rule)",
    )
    parser.add_argument(
        "--boundaries",
        default=",".join(BOUNDARIES),
        help="comma-separated among B1, B2, B3, B3h",
    )
    parser.add_argument(
        "--seqlen", type=int, default=65536, help=f"tokens of the {DET_ROW} pack"
    )
    parser.add_argument(
        "--docs", type=int, default=3, help=f"documents of the {DET_ROW} pack"
    )
    parser.add_argument(
        "--qk-heads",
        type=int,
        default=16,
        help=f"Q/K heads (H_qk) of the {DET_ROW} pack",
    )
    parser.add_argument(
        "--v-heads",
        type=int,
        default=48,
        help=f"value heads (HV, a multiple of H_qk) of the {DET_ROW} pack",
    )
    parser.add_argument(
        "--heads",
        default="",
        help="H_QK,HV: run the gdn_shapes rows' documents with this head pair instead of the row's own (the row is "
        "labelled <row>h<H_QK>x<HV>)",
    )
    parser.add_argument(
        "--channels", type=int, default=10240, help="channels of the causal convolution"
    )
    parser.add_argument(
        "--no-conv",
        action="store_true",
        help="time the GDN step without the convolution",
    )
    parser.add_argument("--save-mode", default="light", choices=("light", "full"))
    parser.add_argument(
        "--groups", type=int, default=3, help="measurement groups (at least 3)"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="repetitions of the palindromic arm pattern per group",
    )
    parser.add_argument(
        "--samples", type=int, default=8, help="timed iterations per call (at least 8)"
    )
    parser.add_argument(
        "--warmup", type=int, default=2, help="untimed iterations per call"
    )
    parser.add_argument(
        "--no-l2-flush", action="store_true", help="no L2 flush before an iteration"
    )
    parser.add_argument(
        "--cool-s",
        type=float,
        default=0.0,
        help="idle seconds before every call (sustained protocol cool-down; 0 = none)",
    )
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument(
        "--max-drift",
        type=float,
        default=0.02,
        help="largest accepted first-to-last group drift of an arm (exit status 1 above it)",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="rank 0 writes one JSON per (row, layout, CP) here",
    )
    args = parser.parse_args(argv)
    if args.groups < 3 or args.samples < 8:
        parser.error("the protocol takes at least 3 groups and 8 samples per call")
    if args.repeat < 1 or args.warmup < 0 or args.cool_s < 0:
        parser.error("--repeat must be >= 1, --warmup >= 0 and --cool-s >= 0")
    if args.v_heads % args.qk_heads:
        parser.error("--v-heads must be a multiple of --qk-heads")
    args.heads = None if not args.heads else tuple(int(x) for x in _csv(args.heads))
    if args.heads is not None and (
        len(args.heads) != 2 or args.heads[1] % args.heads[0]
    ):
        parser.error("--heads takes H_QK,HV with HV a multiple of H_QK")
    args.row_list = _csv(args.rows)
    args.layout_list = _csv(args.layouts)
    args.boundary_list = _csv(args.boundaries)
    arms = _csv(args.arms)
    arms += [f"nondet_p_{r}" for r in _csv(args.p_rules) if f"nondet_p_{r}" not in arms]
    arms += [
        f"nondet_graph_p_{r}"
        for r in _csv(args.graph_p_rules)
        if f"nondet_graph_p_{r}" not in arms
    ]
    args.arm_list = arms
    if not args.row_list or not args.layout_list or not args.boundary_list or not arms:
        parser.error(
            "--rows, --layouts, --boundaries and --arms must name at least one item each"
        )
    bad = [x for x in args.layout_list if x not in LAYOUTS]
    bad += [x for x in args.boundary_list if x not in BOUNDARIES]
    bad += [
        x
        for x in arms
        if x not in BASE_ARMS
        and x not in SOTA_ARMS
        and x not in GRAPH_ARMS
        and not PRULE_ARM.match(x)
        and not GRAPH_PRULE_ARM.match(x)
    ]
    if bad:
        parser.error(f"unknown layout / boundary / arm names: {bad}")
    if any(x in SOTA_ARMS for x in arms) and "nondet" not in arms:
        parser.error("the non-Cake arms are checked against nondet: add it to --arms")
    return args


def main() -> int:
    args = parse_args()
    if torch.are_deterministic_algorithms_enabled():
        raise SystemExit(
            "torch deterministic algorithms are enabled: deterministic=False would run the deterministic route"
        )
    rank, world, device = _init()
    arms = args.arm_list
    routes = {
        kind: Route(kind)
        for kind in ("det", "nondet")
        if any(arm_route(a) == kind for a in arms)
    }
    rows = make_rows(args.row_list, args)
    lib = (
        _Fla(frost=any(a in FROST_CHECKPOINT for a in arms))
        if any(a in SOTA_ARMS for a in arms)
        else None
    )
    env = _env(routes, device, world, lib)
    out_dir = Path(args.out_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"bench_cake_gdn_train_kcp: {env['device']} x{world} (sm_count {env['sm_count']}, torch {env['torch']}, "
            f"CUDA {env['cuda']}) hosts {env['host_map']}",
            flush=True,
        )
        for kind in routes:
            print(f"  route {kind}: {env['ranks'][0]['files'][kind]}", flush=True)
        print(f"  conv: {env['ranks'][0]['files']['conv']}  arms: {arms}", flush=True)
        if lib is not None:
            print(f"  non-Cake arms: {env['sota']}", flush=True)
    flush = (
        None
        if args.no_l2_flush
        else torch.empty(L2_FLUSH_BYTES, dtype=torch.uint8, device=device)
    )
    ok = True
    for row in rows:
        for layout in args.layout_list:
            why = row.skip_reason(layout, world)
            if why is not None:
                if rank == 0:
                    print(
                        f"KCP x{world} {row.name}/{layout}: skipped ({why})", flush=True
                    )
                continue
            report = run_config(
                row, layout, args, arms, routes, env, flush, rank, world, device, lib
            )
            ok = ok and report["drift_ok"]
            if rank == 0:
                path = out_dir / f"kcp_train_{row.name}_{layout}_cp{world}.json"
                path.write_text(json.dumps(report, indent=1) + "\n")
                print(f"  wrote {path} ({report['wall_s']:.1f} s)", flush=True)
            del report
            gc.collect()
            torch.cuda.empty_cache()
            dist.barrier()
    dist.barrier()
    dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
