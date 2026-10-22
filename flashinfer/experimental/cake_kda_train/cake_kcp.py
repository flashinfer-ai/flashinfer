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

Host backend of the context-parallel (KCP) route of the Cake packed KDA
chunked training kernels (SM100 / SM103 / SM107): the transient
:class:`KcpWorkspace`, the forward and backward launch sequences with their
collectives, the saved set :class:`KcpSaved`, the autograd Function and the
entry points ``chunk_kda_kcp`` / ``chunk_kda_kcp_fwd`` / ``chunk_kda_kcp_bwd``.

Every rank holds its own token fragments (``[T_r, H, 128]`` operands, local
``cu_seqlens`` with one sequence per fragment) and only recurrent-state
summaries cross the ranks.  Forward: the crossing-out fragments' affine
summaries ``(B_f, M_f^T)`` (the walk from zero and the FP32-class transfer
walk) go to the merge owner of each head slice (``c1``, one
``all_to_all_single``), the owner composes every document chain in document
order with the exact FP32 affine merge and returns each continuing fragment's
initial state ``S_in`` (``c2``); the local recurrence then runs from ``S_in``.
Backward: the adjoint offsets ``d_f`` (the reverse walk from zero) and, unless
the forward's transfer was kept, ``M_f^T`` again go to the owners (``b1``), the
owner walks each chain backwards (``X <- M_f^T X + d_f``) and returns every
crossing-out fragment's ``dS_out`` (``b2``); the local adjoint recurrence runs
from it.  One compute stream; the collectives are asynchronous on the caller's
process group and waited at their first consumer.  Shipped schedule (round-1
adoption of levers L-A + L-B, rulings P2-R1 / P2-R4): ``wait_first`` -- no merge
or tape walk is issued while a collective is in flight, and the tape walks of a
phase are one fused launch from the carry plane; the round-0 ``overlap`` form
(the document-start tape walk under ``c1``, the document-last adjoint tape walk
under ``b2``: the two R7 overlaps) and the separate walks stay selectable through
the test knobs ``schedule`` / ``fused_walks``.  Nothing launched between an issue
and its wait reads that collective's buffers, so the bits are independent of the
schedule and a checkpoint replay re-runs the same program order on every rank.

The metadata (partition, context, head ownership, chain tables, split sizes,
segment plan, compact-buffer addressing, byte queries) is
:mod:`.cake_kcp_plan`; this module reads tensor contents nowhere, dispatches
from metadata only, allocates the outputs and the saved set fresh per call and
keeps only transient buffers in the workspace.  The kernels are launched
through a dispatch table (:class:`Backend`): the shipped backend runs the
exported programs of :mod:`.cake_kcp_jit` and the local chunk stages through
:mod:`.cake_backend`; a test may install another backend (an FP64 emulation of
the same ABI) for the orchestration tests.

Semantics and contracts follow the KCP design note (its sections 3.5, 3.6, 6.4,
6.5, 7.2-7.5, 9, 10 and 16).


P > 1 staging (ruling P1-R11, 2026-10-21): the K1 ``_offset`` / K2 ``_offset`` / K3 passes of the segmented rank write two
dense planes ``stage_b`` / ``stage_m`` ([NSEG, H, 128, 128] FP32 each, ``tile_stride`` 1); K4 reads them element-wise through
``elem_m`` / ``elem_c`` (map / carry / transposed-map / adjoint-carry families); the pair convention of P1-R1 / P1-R3 stays
for K4's states (c1 / b1 pairs, ``tile_stride`` 2) and the carry plane; under ``save_transfer=True`` at ``P > 1`` the
``stage_m`` plane itself is the saved ``M_seg`` (no copy; ``kcp_saved_bytes`` keeps the ``NSEG * s`` term).
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Sequence
from typing import Any

import torch
import torch.distributed as dist

from . import cake_backend, cake_kcp_plan

D = cake_backend.D  # head dimension (128)
BT = cake_backend.BT  # tokens per chunk (64)
SCALE = cake_backend.SCALE
SAVE_MODES = cake_backend.SAVE_MODES
TILE = D * D  # elements of one FP32 state tile [128, 128]
SUMMARY_TILE = 2 * TILE  # one c1 summary tile of one head: [2][128][128] = (B_f, M_f^T)
BOUNDARY_MODES = ("saved", "replay")
EXCHANGES = ("auto", "a2a", "ring")
OFFSET_CLASSES = ("fla", "fp32")
RING_MAX_HOPS = 2
TRANSFER_WALKS = (
    "tf32_hilo",
    "bf16x3",
    "bf16x3_hw64",
)  # K3 program family (P1-R15; non-default only via _kcp_test_knobs)  # exchange="auto" picks the ring when every chain has at most this many hops (SPEC-2)
KCP_SAVED_VERSION = 1  # saved set {Aqk, Akk, tables, S_in [, c1 receive tensor, M_seg]}
TEST_KNOBS_ENV = "CAKE_KCP_TEST_KNOBS"
# Round-1 host / plan levers (P2-R1, P2-R4; P1-R15 test knob class: reachable only through _kcp_test_knobs):
SCHEDULES = (
    "overlap",
    "wait_first",
)  # L-A: "wait_first" (shipped) = merges and tape walks after each wait; "overlap" = the round-0 R7 overlaps of FINAL_DESIGN §3.5 / §6.5 (test knob)
COLS_RULES = (
    "fixed",
    "by_chains",
)  # D1: K4 cols_per_cta of the 128-column state launches: the backend's, or 8 up to COLS_RULE_CHAINS chains else 16
COLS_RULE_CHAINS = 24  # D1 threshold (DECISIONS.md D1: <= 24 concurrent chains -> c8, otherwise c16; map launches stay c16)
P2_KNOBS = {
    "schedule": SCHEDULES,
    "fused_walks": (0, 1),
    "cols_rule": COLS_RULES,
}  # knob -> admitted values
# Shipped defaults of the round-1 pair (adoption of levers L-A + L-B together, rulings P2-R1 / P2-R4, lane LAB's keep
# rule): the "wait_first" schedule with the fused tape walks (one launch per phase).  "overlap" / fused_walks=0 stay
# selectable through the test knobs for the A/B record; cols_rule stays "fixed" (D1 in band everywhere: not adopted,
# knob kept).  The schedule and the walk grouping never change a published bit (the merges read c1_recv / the staging
# planes, the tapes read no collective buffer): G3 holds under every combination.
SCHEDULE_DEFAULT = "wait_first"
FUSED_WALKS_DEFAULT = 1

# Program names of the dispatch table (the K1-K4 programs of FINAL_DESIGN §14) and the local chunk stages
# (``cake_backend`` stages at the per-rank shape, launched through ``KdaWorkspace.run``).
KCP_PROGRAMS = (
    "kcp_fwdh_offset",  # K1 ``_offset``: walk from zero, final state -> compact tiles (B_f / b_seg)
    "kcp_fwdh_tape",  # K1 ``_tape``: walk from an indexed initial state, h / v_new tapes
    "kcp_dhu_offset",  # K2 ``_offset``: reverse walk from zero, final adjoint -> compact tiles (d_f / d_seg)
    "kcp_dhu_tape",  # K2 ``_tape``: reverse walk from an indexed final adjoint, dh tape
    "kcp_transfer_walk",  # K3: FP32-class transfer M_f (^T) (or the FP32-class offset with init_mode = 1)
    "kcp_affine_merge",  # K4: exact FP32 affine chain (state / map / transposed modes)
)
LOCAL_STAGES_FWD = ("fwd_prep", "fwd_intra", "wy", "fwd_o")
LOCAL_STAGES_BWD = ("bwd_prep", "wy", "dav", "dqkg", "glm_intra", "gate_epilogue")
COLLECTIVES = ("c1", "c2", "b1", "b2")


# ----------------------------------------------------------------------------------------------------------------
# Small helpers (metadata only)


def tile_index(h: int, slot: int, hg: int, n: int) -> int:
    """The kernels' head-group-major compact tile address (FINAL_DESIGN §7.3): tile ``((h // hg) * n + slot) * hg
    + h % hg`` of a pool whose consecutive head groups each hold ``n`` slots of ``hg`` heads."""
    return ((h // hg) * n + slot) * hg + h % hg


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _alloc(
    shape: Sequence[int], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Every per-call tensor (outputs, saved set, collective outputs) comes from here: ``torch.empty`` for buffers
    every element of which a kernel or a collective of the same call writes."""
    return torch.empty(tuple(shape), dtype=dtype, device=device)


def _zeros(
    shape: Sequence[int], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    return torch.zeros(tuple(shape), dtype=dtype, device=device)


def _squeeze_batch(x: torch.Tensor, name: str, rank: int) -> torch.Tensor:
    if x.dim() == rank + 1:
        if x.shape[0] != 1:
            raise ValueError(
                f"{name}: the batch axis must be 1, got shape {tuple(x.shape)}"
            )
        x = x.squeeze(0)
    if x.dim() != rank:
        raise ValueError(
            f"{name}: expected {rank}-D (or a leading batch axis of 1), got shape {tuple(x.shape)}"
        )
    return x


def _check(
    x: torch.Tensor,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if tuple(x.shape) != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {tuple(x.shape)}")
    if x.dtype != dtype:
        raise ValueError(f"{name}: expected {dtype}, got {x.dtype}")
    if x.numel() and not x.is_contiguous():
        raise ValueError(
            f"{name}: must be contiguous (strides {x.stride()}); the kernels take no strided operands"
        )
    if x.device != device:
        raise ValueError(f"{name}: on {x.device}, expected {device}")


def _operands(q, k, v, g, beta, ctx, *, dtypes: dict[str, torch.dtype]):
    """The operands without their optional batch axis, ``(T_r, H)`` and whether the inputs carried a batch axis.
    ``T_r`` must equal the context's local token count; ``H`` is the operands' head count (``1..64``)."""
    v = _squeeze_batch(v, "v", 3)
    T, H = int(v.shape[0]), int(v.shape[1])
    device = torch.device(ctx.device)
    batched = q.dim() == 4
    q, k, g = (_squeeze_batch(x, n, 3) for x, n in ((q, "q"), (k, "k"), (g, "g")))
    beta = _squeeze_batch(beta, "beta", 2)
    if int(ctx.T_r) != T:
        raise ValueError(
            f"operands carry {T} tokens, the KcpContext of rank {ctx.rank} holds T_r={ctx.T_r}"
        )
    if not 1 <= H <= 64:
        raise ValueError(f"H={H}: the Cake KDA kernels take 1..64 heads")
    _check(q, "q", (T, H, D), dtypes["q"], device)
    _check(k, "k", (T, H, D), dtypes["k"], device)
    _check(v, "v", (T, H, D), dtypes["v"], device)
    _check(g, "g", (T, H, D), dtypes["g"], device)
    _check(beta, "beta", (T, H), dtypes["beta"], device)
    return q, k, v, g, beta, T, H, batched


def _ranked(x: torch.Tensor, batched: bool) -> torch.Tensor:
    return x.unsqueeze(0) if batched else x


# ----------------------------------------------------------------------------------------------------------------
# Backends: the dispatch table of the KCP programs and the local chunk stages


# The contract's storage dtypes.  A backend may override them (``Backend.dtypes``): the FP64 emulation keeps every
# operand, state and output in FP64 so the orchestration can be checked against the global FP64 reference to ~1e-12.
CONTRACT_DTYPES: dict[str, torch.dtype] = {
    "q": torch.bfloat16,
    "k": torch.bfloat16,
    "v": torch.bfloat16,
    "g": torch.float32,
    "beta": torch.bfloat16,
    "out": torch.bfloat16,
    "dout": torch.bfloat16,
    "dq": torch.bfloat16,
    "dk": torch.bfloat16,
    "dv": torch.bfloat16,
    "dbeta": torch.bfloat16,
    "dg": torch.float32,
    "state": torch.float32,  # the FP32 states, summaries and adjoints of every pool and collective
}


class Backend:
    """A dispatch table for the KCP programs and the local chunk stages.

    ``run(name, args)`` launches program ``name`` (one of :data:`KCP_PROGRAMS`) with the logical arguments ``args``
    (the ABI vocabulary of FINAL_DESIGN §14: pools, index tables, ``hg`` / ``n`` addressing scalars, modes; plus
    ``_call``, the :class:`KcpCall` the launch belongs to).  ``stage(name, call, **extra)`` launches one local chunk
    stage of :mod:`.cake_backend` at the per-rank shape.  ``dtypes`` are the storage dtypes of the operands, outputs
    and states; ``local_stages`` says whether the local stages exist (the FP64 emulation folds them into its tape
    programs).  The shipped backend is :class:`KernelBackend`; tests install another with :func:`set_backend`.
    """

    name = "abstract"
    dtypes: dict[str, torch.dtype] = dict(CONTRACT_DTYPES)
    local_stages = True
    path = "fi-rendered"  # recorded with every result: fake (emulation), direct (bring-up), fi-rendered (the package)

    def run(
        self, name: str, args: dict[str, Any]
    ) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def stage(
        self, name: str, call: KcpCall, **extra: Any
    ) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def begin_call(self, call: KcpCall) -> None:
        """Hook at the start of a forward or backward call (the kernel backend builds the local stage values, the
        emulation resets its per-call scratch)."""

    def end_call(self, call: KcpCall) -> None:
        """Hook at the end of a call."""

    @staticmethod
    def transpose_tiles(
        *,
        src: torch.Tensor,
        src_tiles: torch.Tensor,
        dst: torch.Tensor,
        dst_tiles: torch.Tensor,
    ) -> None:
        """``dst`` tiles ``dst_tiles[i]`` <- the transposes of the ``src`` tiles ``src_tiles[i]`` (torch; bitwise;
        shared by every backend: the ``M_f`` -> ``M_f^T`` step of the segmented summaries)."""
        dst.view(-1, D, D)[dst_tiles] = src.view(-1, D, D)[src_tiles].transpose(1, 2)


_SPEC3_ROWS = (
    "gk",
    "kg",
    "qg",
    "w",
    "u",
)  # the prep rows SPEC-3 compares (FINAL_DESIGN §6.4)
SPEC3_ENV = "CAKE_KCP_SPEC3"  # "0": skip the debug assertion; "warn": record a failure without raising; default: assert
_PROGRAM_OF = {
    "kcp_fwdh_offset": "fwdh_offset",
    "kcp_fwdh_tape": "fwdh_tape",
    "kcp_dhu_offset": "dhu_offset",
    "kcp_dhu_tape": "dhu_tape",
    "kcp_transfer_walk": "transfer_walk",
    "kcp_affine_merge": "affine_merge",
}


class KcpPreconditionError(RuntimeError):
    """SPEC-3 failed: ``bwd_prep`` + ``wy`` did not reproduce the forward's ``gk`` / ``kg`` / ``qg`` / ``w`` / ``u`` rows
    bitwise (FINAL_DESIGN §6.4); saved mode and replay mode would differ.  Options (a) / (b) / (c) of §6.4 are a parent
    decision (``NEED_PARENT``), never a silent relaxation."""


@dataclasses.dataclass(frozen=True)
class _Block:
    """One compact indexed state block as a walk kernel addresses it: base pointer, ``index[NSEG]`` (i32 device),
    ``hg``, ``n`` and the ``tile_stride`` of the pool (1 dense plane, 2 interleaved pair)."""

    ptr: torch.Tensor
    index: torch.Tensor
    hg: int
    n: int
    stride: int


class KernelBackend(Backend):
    """The shipped backend: the KCP programs K1-K4 through a program provider (``launch(program, args, *, grid,
    device_index, precision_mode)`` with the positional ABI of FINAL_DESIGN §14 / rulings P1-R1, P1-R3) and the local
    chunk stages through :class:`cake_backend.KdaWorkspace` of the per-rank problem.  The default provider is H3's
    registry (:mod:`.cake_kcp_jit`, imported at the first launch so this module imports without it); a bring-up
    provider launching the source builds directly can be passed (``path="direct"``; never a gated row).

    Local stages: a forward call runs ``fwd_prep`` / ``fwd_intra`` / ``wy`` / ``fwd_o`` on the arena's ``fwd`` layout;
    a backward call runs on the ``bwd`` layout — saved mode ``bwd_prep`` (``copy_saved = 1`` from the saved ``Aqk`` /
    ``Akk``) + ``wy``, replay mode ``bwd_prep`` (``copy_saved = 0``) + ``fwd_intra`` (``write_int``) + ``wy`` as the
    existing replay path (``cake_backend.chunk_kda_bwd``) — then ``dav`` / ``dqkg`` / ``glm_intra`` / ``gate_epilogue``.
    The K1 / K2 / K3 walks read the layout's ``w`` / ``kg`` / ``u`` / ``gk`` (``qg`` / ``do_int`` / ``dv1``) rows and write
    its ``h`` / ``v_new`` (``dh`` / ``dv2``) tapes.

    SPEC-3 (debug assertion, once per process on the first saved-mode backward): before that backward's ``bwd_prep``
    the forward's ``fwd_prep`` + ``fwd_intra`` + ``wy`` run on the ``fwd`` layout and their five rows are kept; after
    ``bwd_prep`` + ``wy`` the ``bwd`` layout's rows must equal them bitwise on every real token row (pad rows are
    reported separately).  ``CAKE_KCP_SPEC3=0`` skips it, ``=warn`` records without raising; the result is in
    ``LAST_CALL["bwd"]["spec3"]`` and :attr:`spec3`.
    """

    name = "kernels"

    def __init__(
        self,
        programs: Any | None = None,
        *,
        cols_per_cta: int = 8,
        path: str = "fi-rendered",
    ) -> None:
        self._programs = programs
        if cols_per_cta not in (8, 16, 32) or D % cols_per_cta:
            raise ValueError(f"cols_per_cta must be 8, 16 or 32, got {cols_per_cta}")
        self.cols_per_cta = int(cols_per_cta)
        self.path = path
        self._dev: dict[tuple[Any, ...], torch.Tensor] = {}
        self._checked_chains: set[int] = set()
        self.spec3: dict[str, Any] | None = None
        self._spec3_pending = os.environ.get(SPEC3_ENV, "1") != "0"

    # -- providers
    def programs(self) -> Any:
        if self._programs is None:
            from . import cake_kcp_jit  # noqa: PLC0415 - H3's registry; optional at import time

            self._programs = _FiPrograms(cake_kcp_jit)
        return self._programs

    # -- device tables (small int tables, cached by content)
    def _dev_table(
        self,
        kind: str | tuple[Any, ...],
        values: Sequence[int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (kind, str(device), dtype, tuple(int(v) for v in values))
        t = self._dev.get(key)
        if t is None:
            t = torch.tensor(list(key[3]), dtype=dtype, device=device)
            self._dev[key] = t
        return t

    # -- local stage values
    def begin_call(self, call: KcpCall) -> None:
        if call.local is None:
            return
        ws, ctx, p = call.local, call.ctx, call.local.plan
        if (call.T_r, int(ctx.n_frag), call.H) != (p.S, p.N, p.H):
            raise ValueError(
                f"local arena planned for (S, N, H) = {(p.S, p.N, p.H)}, the call is {(call.T_r, ctx.n_frag, call.H)}"
            )
        if (p.NC_pad, p.R) != (int(ctx.NC_pad), int(ctx.R)):
            raise ValueError(
                f"local plan geometry (NC_pad, R) = {(p.NC_pad, p.R)} differs from the KCP context's {(ctx.NC_pad, ctx.R)}"
            )
        views = ws.views("bwd" if call.backward else "fwd")
        values: dict[str, Any] = dict(vars(views))
        for name in cake_backend.TABLE_NAMES[:4]:
            values[name] = ctx.tables[name]
        values["seg_chunk_start"] = call.tb.hp.device_tables["seg_chunk_start"]
        values.update(
            num_seqs=p.N,
            num_chunks_pad=p.NC_pad,
            num_heads=p.H,
            group=1,
            num_segments=p.NSEG,
            segments=p.P,
        )
        values.update(call.ops)
        values.update(h2d=views.h.view(-1, D), scale=call.scale)
        if call.backward:
            values["dh2d"] = views.dh.view(-1, D)
        call.local_values = values
        call.scratch["views"] = views

    def stage(self, name: str, call: KcpCall, **extra: Any) -> None:
        values, views = call.local_values, call.scratch["views"]
        if call.out is not None:
            values["out"] = call.out
        if call.grads is not None:
            values.update(call.grads)
        if (
            name == "fwd_prep" and call.backward
        ):  # replay: the backward layout's prep without the saved copies
            name, extra = "bwd_prep", {"saved": None}
        if name == "bwd_prep":
            saved = extra.pop("saved", None)
            if saved is None:
                extra.update(aqk_src=views.akk_int, akk_src=views.akk_int, copy_saved=0)
            else:
                if saved.Aqk is None or saved.Akk is None:
                    raise ValueError(
                        "saved: the saved set has no Aqk / Akk rows (forward without save_for_backward?)"
                    )
                _check(
                    saved.Aqk,
                    "saved.Aqk",
                    (call.T_r, call.H, BT),
                    torch.bfloat16,
                    call.device,
                )
                _check(
                    saved.Akk,
                    "saved.Akk",
                    (call.T_r, call.H, BT),
                    torch.bfloat16,
                    call.device,
                )
                extra.update(aqk_src=saved.Aqk, akk_src=saved.Akk, copy_saved=1)
                if self._spec3_pending:
                    self._spec3_forward_rows(
                        call
                    )  # before the backward's prep: the layouts share the arena
        elif name == "fwd_intra":
            if int(extra.get("write_ext", 0)):
                aqk = call.scratch["Aqk"] = _alloc(
                    (call.T_r, call.H, BT), torch.bfloat16, call.device
                )
                akk = call.scratch["Akk"] = _alloc(
                    (call.T_r, call.H, BT), torch.bfloat16, call.device
                )
                extra.update(aqk_ext=aqk, akk_ext=akk)
            else:  # not written: any BF16 tensor serves as the unused pointer (as cake_backend does)
                extra.update(aqk_ext=views.akk_int, akk_ext=views.akk_int)
        call.local.run(name, {**values, **extra})
        if name == "wy" and call.scratch.get("spec3_ref") is not None:
            self._spec3_compare(call)

    # -- SPEC-3
    def _spec3_forward_rows(self, call: KcpCall) -> None:
        self._spec3_pending = False
        ws = call.local
        a = ws.views("fwd")
        vals = {**call.local_values, **vars(a)}
        vals.update(
            h2d=a.h.view(-1, D),
            aqk_ext=a.akk_int,
            akk_ext=a.akk_int,
            out=call.grads["dv"],
        )
        ws.run("fwd_prep", vals)
        ws.run("fwd_intra", {**vals, "write_ext": 0, "write_int": 1})
        ws.run("wy", vals)
        call.scratch["spec3_ref"] = {n: getattr(a, n).clone() for n in _SPEC3_ROWS}

    def _spec3_compare(self, call: KcpCall) -> None:
        ref = call.scratch.pop("spec3_ref")
        b, ctx = call.scratch["views"], call.ctx
        R = int(ctx.R)
        rows = torch.arange(R, device=call.device)
        lens = torch.tensor(list(ctx.chunk_len), dtype=torch.int64, device=call.device)
        real = (rows % BT) < lens[rows // BT]
        result: dict[str, Any] = {
            "checked": True,
            "T_r": call.T_r,
            "H": call.H,
            "R": R,
            "rows": {},
            "equal": True,
        }
        for n in _SPEC3_ROWS:
            x, y = getattr(b, n), ref[n]
            it = torch.int32 if x.dtype == torch.float32 else torch.int16
            differ = (x.view(it) != y.view(it)).view(R, -1).any(1)
            n_all, n_real = int(differ.sum()), int((differ & real).sum())
            result["rows"][n] = {"rows_differ": n_all, "real_rows_differ": n_real}
            if n_real:
                result["equal"] = False
        self.spec3 = call.scratch["spec3"] = result
        if not result["equal"] and os.environ.get(SPEC3_ENV, "1") != "warn":
            raise KcpPreconditionError(
                f"SPEC-3: bwd_prep + wy differ from fwd_prep + wy on real rows: {result['rows']}"
            )

    # -- KCP programs (positional ABI of FINAL_DESIGN §14 with rulings P1-R1 / P1-R3)
    def abi(
        self, name: str, args: dict[str, Any]
    ) -> tuple[str, list[Any], int, dict[str, Any]]:
        """``(program, positional ABI arguments, grid x, variant)`` of one abstract launch ``args`` of program
        ``name`` (the integration contract with the registry: ``program`` in ``fwdh_offset`` / ``fwdh_tape`` /
        ``dhu_offset`` / ``dhu_tape`` / ``transfer_walk`` / ``affine_merge``; ``variant`` = the K3 test knobs)."""
        call: KcpCall = args["_call"]
        prog = _PROGRAM_OF[name]
        if prog in ("fwdh_offset", "fwdh_tape"):
            abi, grid = self._fwdh_args(call, args, tape=prog == "fwdh_tape")
        elif prog in ("dhu_offset", "dhu_tape"):
            abi, grid = self._dhu_args(call, args, tape=prog == "dhu_tape")
        elif prog == "transfer_walk":
            abi, grid = self._transfer_args(call, args)
        else:
            abi, grid = self._merge_args(call, args)
        variant = (
            {"precision_mode": call.precision_mode, "transfer_walk": call.transfer_walk}
            if prog == "transfer_walk"
            else {}
        )
        if prog == "affine_merge" and args["cols_per_cta"] is not None:
            variant = {
                "cols_per_cta": int(args["cols_per_cta"])
            }  # D1: an explicit record instead of the registry's choice
        return prog, abi, int(grid), variant

    def run(self, name: str, args: dict[str, Any]) -> None:
        call: KcpCall = args["_call"]
        prog, abi, grid, variant = self.abi(name, args)
        self.programs().launch(
            prog,
            abi,
            grid=grid,
            device_index=call.device.index,
            name=name,
            args=args,
            **variant,
        )

    def _walk_common(
        self, call: KcpCall, args: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, int, Any]:
        walk = args["walk"]
        walk_seg = self._dev_table("walk", walk, call.device, torch.int32)
        return (
            walk_seg,
            call.tb.hp.device_tables["seg_chunk_start"],
            int(call.tb.NSEG),
            call.scratch["views"],
        )

    def _block(self, call: KcpCall, ix: Indexed | None) -> _Block:
        """The kernel's view of an :class:`Indexed` pool; ``None`` (or no slot) = the zero state: every index ``-1``
        and a never-read base pointer."""
        nseg = int(call.tb.NSEG)
        if ix is None or ix.n_slots == 0:
            return _Block(
                call.buf("eye"),
                self._dev_table("neg", [-1] * nseg, call.device, torch.int32),
                call.H,
                1,
                1,
            )
        base_tile, hg, n, stride = ix.addressing(call.H)
        if len(ix.index) != nseg:
            raise ValueError(
                f"Indexed: {len(ix.index)} segment entries, the plan has NSEG = {nseg}"
            )
        ptr = ix.pool.view(-1)[base_tile * TILE :] if base_tile else ix.pool.view(-1)
        return _Block(
            ptr,
            self._dev_table("index", ix.index, call.device, torch.int32),
            int(hg),
            int(n),
            int(stride),
        )

    def _fwdh_args(
        self, call: KcpCall, args: dict[str, Any], *, tape: bool
    ) -> tuple[list[Any], int]:
        walk_seg, seg_table, nseg, v = self._walk_common(call, args)
        if tape:  # reads h0, writes the tapes; the out block is unused
            h0 = self._block(call, args.get("h0"))
            out = h0
        else:  # walks from zero, writes the indexed final states; the h0 block is unused
            out = self._block(call, args.get("out"))
            h0 = out
        abi = [
            v.w,
            v.kg,
            v.u,
            v.gk,
            call.H,
            seg_table,
            nseg,
            walk_seg,
            h0.ptr,
            h0.index,
            h0.hg,
            h0.n,
            out.ptr,
            out.index,
            out.hg,
            out.n,
            out.stride if not tape else h0.stride,
            v.h,
            v.v_new,
        ]
        return abi, len(args["walk"]) * call.H

    def _dhu_args(
        self, call: KcpCall, args: dict[str, Any], *, tape: bool
    ) -> tuple[list[Any], int]:
        walk_seg, seg_table, nseg, v = self._walk_common(call, args)
        if tape:
            dht = self._block(call, args.get("dht"))
            out = dht
        else:
            out = self._block(call, args.get("out"))
            dht = out
        abi = [
            v.kg,
            v.qg,
            v.w,
            v.do_int,
            v.dv1,
            v.gk,
            call.H,
            seg_table,
            nseg,
            float(call.scale),
            walk_seg,
            dht.ptr,
            dht.index,
            dht.hg,
            dht.n,
            out.ptr,
            out.index,
            out.hg,
            out.n,
            out.stride if not tape else dht.stride,
            v.dh,
            v.dv2,
        ]
        return abi, len(args["walk"]) * call.H

    def _transfer_args(
        self, call: KcpCall, args: dict[str, Any]
    ) -> tuple[list[Any], int]:
        walk_seg, seg_table, nseg, v = self._walk_common(call, args)
        out = self._block(call, args["out"])
        abi = [
            v.w,
            v.kg,
            v.u,
            v.gk,
            call.H,
            seg_table,
            nseg,
            walk_seg,
            out.ptr,
            out.index,
            out.hg,
            out.n,
            out.stride,
            int(args["init_mode"]),
            int(args["use_u"]),
            int(args["write_transposed"]),
        ]
        return abi, len(args["walk"]) * call.H

    def _merge_args(self, call: KcpCall, args: dict[str, Any]) -> tuple[list[Any], int]:
        ch: Chains = args["chains"]
        if id(ch) not in self._checked_chains:
            ch.check_alignment()
            self._checked_chains.add(id(ch))
        args["cols_per_cta"] = cols = merge_cols_per_cta(
            ch.ncols, ch.n_chains, call.cols_rule
        )
        call.merge_log.append(
            (str(args.get("family")), int(ch.ncols), int(ch.n_chains), cols)
        )  # D1 evidence: None = registry record
        cols = self.cols_per_cta if cols is None else cols
        if ch.ncols % cols:
            raise ValueError(
                f"ncols {ch.ncols} is not a multiple of cols_per_cta {cols}"
            )
        dev = call.device
        y = args["y"]
        m = args["m"]
        c = (
            args["c"] if args.get("c") is not None else y
        )  # never read when every elem_c is -1
        x0 = (
            args["x0"] if args.get("x0") is not None else y
        )  # never read when every x0_index is -1
        tag = ("chain", id(ch))
        abi = [
            m,
            c,
            x0,
            y,
            self._dev_table(tag + ("start",), ch.chain_start, dev, torch.int32),
            self._dev_table(tag + ("m",), ch.elem_m, dev, torch.int64),
            self._dev_table(tag + ("c",), ch.elem_c, dev, torch.int64),
            self._dev_table(tag + ("y",), ch.elem_y, dev, torch.int64),
            self._dev_table(tag + ("x0",), ch.x0_index, dev, torch.int64),
            self._dev_table(tag + ("yf",), ch.y_final, dev, torch.int64),
            int(ch.ncols),
            int(args["transpose_m"]),
            int(args["write_states"]),
            cols,
        ]
        return abi, ch.n_chains * (ch.ncols // cols)


def merge_cols_per_cta(ncols: int, n_chains: int, rule: str) -> int | None:
    """D1 (DECISIONS.md D1, ruling P2-R4): the K4 ``cols_per_cta`` a launch asks for explicitly, ``None`` for the
    registry's choice of record (``fixed``: the shipped rule, c8 state / c16 map).  ``by_chains`` keeps the 8-column
    program for 128-column state chains up to :data:`COLS_RULE_CHAINS` concurrent chains and takes the 16-column
    program beyond it; 256-column map chains stay with the registry's c16.  Bitwise per layout (K4 REPORT §3)."""
    if rule == "by_chains" and int(ncols) == D and int(n_chains) > COLS_RULE_CHAINS:
        return 16
    return None


# Kernel parameter names of the frozen ABIs (K1 a217181fa, K2 a217181fa, K3 0637ec9f4, K4 eaa97cf5e): the logical
# vocabulary of the registry records when the FI-rendered provider launches by name.
ABI_PARAMS = {
    "fwdh_offset": (
        "w_tma",
        "kg_tma",
        "u",
        "gk",
        "num_heads",
        "seq_chunk_start",
        "num_seqs",
        "walk_seg",
        "h0",
        "h0_index",
        "h0_hg",
        "h0_n",
        "out",
        "out_index",
        "out_hg",
        "out_n",
        "tile_stride",
        "h_tape",
        "v_new",
    ),
    "dhu_offset": (
        "kg_tma",
        "qg_tma",
        "w_tma",
        "do_tma",
        "dv1",
        "gk",
        "num_heads",
        "seq_chunk_start",
        "num_seqs",
        "scale",
        "walk_seg",
        "dht",
        "dht_index",
        "dht_hg",
        "dht_n",
        "out",
        "out_index",
        "out_hg",
        "out_n",
        "tile_stride",
        "dh_tape",
        "dv2",
    ),
    "transfer_walk": (
        "w_tma",
        "kg_tma",
        "u",
        "gk",
        "num_heads",
        "seq_chunk_start",
        "num_seqs",
        "walk_seg",
        "out",
        "out_index",
        "out_hg",
        "out_n",
        "tile_stride",
        "init_mode",
        "use_u",
        "write_transposed",
    ),
    "affine_merge": (
        "m",
        "c",
        "x0",
        "y",
        "chain_start",
        "elem_m",
        "elem_c",
        "elem_y",
        "x0_index",
        "y_final",
        "ncols",
        "transpose_m",
        "write_states",
        "cols_per_cta",
    ),
}
ABI_PARAMS["fwdh_tape"] = ABI_PARAMS["fwdh_offset"]
ABI_PARAMS["dhu_tape"] = ABI_PARAMS["dhu_offset"]


class _FiPrograms:
    """The FI-rendered program provider: H3's :mod:`.cake_kcp_jit` registry, through whichever entry it exports:
    ``launch_kcp(program, abi, *, grid, device_index, precision_mode, transfer_walk)`` (positional ABI), else
    ``run_program(name, args, device_index=)`` (the abstract launch ``args`` of :class:`KcpCall` — H3's 09:59Z form),
    else ``run_module(record, values, grid=(x, 1, 1), device_index=)`` with ``values`` keyed by :data:`ABI_PARAMS`
    (record names ``kcp_<program>``; the merge record by ``ncols``: ``_c8`` / ``_c16``; K3 test variants by suffix)."""

    path = "fi-rendered"

    def __init__(self, jit: Any) -> None:
        self.jit = jit

    def launch(
        self,
        program: str,
        abi: list[Any],
        *,
        grid: int,
        device_index: int,
        precision_mode: int = 0,
        transfer_walk: str = "tf32_hilo",
        name: str | None = None,
        args: dict[str, Any] | None = None,
        cols_per_cta: int | None = None,
    ) -> None:
        jit = self.jit
        if (
            program == "affine_merge" and cols_per_cta is not None
        ):  # D1 cols_rule: the explicit `_c<cols>` record
            values = dict(zip(ABI_PARAMS[program], abi, strict=True))
            jit.run_module(
                f"kcp_affine_merge_c{int(cols_per_cta)}",
                values,
                grid=(int(grid), 1, 1),
                device_index=device_index,
            )
            return
        if hasattr(jit, "launch_kcp"):
            jit.launch_kcp(
                program,
                abi,
                grid=grid,
                device_index=device_index,
                precision_mode=precision_mode,
                transfer_walk=transfer_walk,
            )
            return
        if hasattr(jit, "run_program") and name is not None and args is not None:
            jit.run_program(name, args, device_index=device_index)
            return
        record = f"kcp_{program}"
        if program == "transfer_walk":
            if transfer_walk != "tf32_hilo":
                record = "kcp_transfer_walk_bf16x3" + (
                    "_hw64" if transfer_walk == "bf16x3_hw64" else ""
                )
            elif precision_mode:
                record += {1: "_three_mma", 2: "_plain_tf32"}[int(precision_mode)]
        elif program == "affine_merge":
            record += (
                "_c8" if int(abi[ABI_PARAMS[program].index("ncols")]) == D else "_c16"
            )
        values = dict(zip(ABI_PARAMS[program], abi, strict=True))
        jit.run_module(
            record, values, grid=(int(grid), 1, 1), device_index=device_index
        )


_BACKEND: Backend = KernelBackend()


def set_backend(backend: Backend | None) -> Backend:
    """Install ``backend`` as the dispatch table of every following call (``None``: the shipped
    :class:`KernelBackend`); returns the previous one."""
    global _BACKEND
    previous = _BACKEND
    _BACKEND = KernelBackend() if backend is None else backend
    return previous


def get_backend() -> Backend:
    return _BACKEND


# ----------------------------------------------------------------------------------------------------------------
# Tables of one (ctx, H) problem, derived from the plan module's head plan (``ctx.heads(H)``)


@dataclasses.dataclass(frozen=True)
class Indexed:
    """A compact state pool as a walk kernel reads or writes it: ``pool`` (a flat tensor of ``128 x 128`` tiles),
    ``index[seg]`` (the slot of segment ``seg``; ``-1`` = no tile: the zero state on input, not written on output) and
    ``tiles[slot][h]`` (the tile number of head ``h`` of that slot inside ``pool``).  :meth:`addressing` recovers the
    kernels' two-integer form (ruling P1-R3 / P1-R4): ``tiles[slot][h] == base_tile + tile_index(h, slot, hg, n) * stride``."""

    pool: torch.Tensor
    index: tuple[int, ...]
    tiles: tuple[tuple[int, ...], ...]

    @property
    def n_slots(self) -> int:
        return len(self.tiles)

    def addressing(self, H: int) -> tuple[int, int, int, int]:
        """``(base_tile, hg, n, stride)`` of the kernels' addressing, ``ValueError`` when the table has no such form."""
        n_slots = len(self.tiles)
        if n_slots == 0:
            return 0, int(H), 0, 1
        base = self.tiles[0][0]
        for stride in (1, 2):
            for hg in range(1, H + 1):
                if hg < H:
                    span = self.tiles[0][hg] - base
                    if span <= 0 or span % (hg * stride):
                        continue
                    n = span // (hg * stride)
                    if n < n_slots:
                        continue
                else:
                    n = n_slots
                if all(
                    self.tiles[s][h] == base + tile_index(h, s, hg, n) * stride
                    for s in range(n_slots)
                    for h in range(H)
                ):
                    return base, hg, n, stride
        raise ValueError("Indexed: the tile table has no two-integer (hg, n) form")


class Chains:
    """K4 chain tables of one launch (FINAL_DESIGN §14.4; ruling P1-R1): CSR over elements ``(elem_m, elem_c, elem_y)``
    in application order (element offsets into the ``m`` / ``c`` / ``y`` pools; ``elem_c = -1`` no offset; ``elem_y`` =
    where the state ENTERING the element is published when ``write_states``, ``-1`` none), the initial state
    ``x0_index`` (``-1``: zero) and ``y_final`` (the state leaving the chain; an empty chain publishes its initial state).
    ``elem_m`` must be ``>= 0`` (the kernel has no identity element)."""

    def __init__(self, ncols: int = D) -> None:
        self.ncols = int(ncols)
        self.chain_start: list[int] = [0]
        self.elem_m: list[int] = []
        self.elem_c: list[int] = []
        self.elem_y: list[int] = []
        self.x0_index: list[int] = []
        self.y_final: list[int] = []

    @property
    def n_chains(self) -> int:
        return len(self.x0_index)

    @property
    def n_elems(self) -> int:
        return len(self.elem_m)

    def add(
        self, elems: Sequence[tuple[int, int, int]], *, x0: int = -1, y_final: int = -1
    ) -> None:
        for m, c, y in elems:
            if m < 0:
                raise ValueError(
                    "Chains: every element needs an operator tile (elem_m >= 0)"
                )
            self.elem_m.append(int(m))
            self.elem_c.append(int(c))
            self.elem_y.append(int(y))
        self.chain_start.append(len(self.elem_m))
        self.x0_index.append(int(x0))
        self.y_final.append(int(y_final))

    def check_alignment(self) -> None:
        for name in ("elem_m", "elem_c", "elem_y", "x0_index", "y_final"):
            if any(v % 4 for v in getattr(self, name) if v >= 0):
                raise ValueError(
                    f"Chains: {name} holds an offset that is not a multiple of 4 floats"
                )

    def as_dict(self) -> dict[str, Any]:
        return {
            "ncols": self.ncols,
            "n_chains": self.n_chains,
            "chain_start": list(self.chain_start),
            "elem_m": list(self.elem_m),
            "elem_c": list(self.elem_c),
            "elem_y": list(self.elem_y),
            "x0_index": list(self.x0_index),
            "y_final": list(self.y_final),
        }


def _prefix(values) -> list[int]:
    out = [0]
    for v in values:
        out.append(out[-1] + int(v))
    return out


class _Tables:
    """Everything the launch sequences read for one ``(ctx, H)``: the segment plan, walk lists, the compact tile
    tables of every pool, the collective splits and the K4 chains, derived once from ``cake_kcp_plan.KcpContext`` /
    ``KcpHeadPlan`` and cached by the workspace (FINAL_DESIGN §7.2, §10.3, §14.4; rulings P1-R1, P1-R3, P1-R4).

    Pools (tile = 16384 floats; tile numbers pool-relative): ``c1_send`` = owner-major ``(out_slot, head)`` pairs
    ``[B_f | M_f^T]`` (B tile ``2 t``, M tile ``2 t + 1`` of ``hp.out_tile[out_slot][h] = t``); ``S_in`` (the c2
    receive, ``hp.h0_tile``), ``b2_recv`` = ``dS_out`` (``hp.dht_tile``), ``b1_send`` (``hp.b1_d_tile_d`` /
    ``hp.b1_d_tile_dm`` / ``hp.b1_m_tile_dm``: the plan module's layout of the ``d`` and ``dm`` forms); at ``P > 1`` the
    staging planes ``stage_b`` (``b_seg`` / ``d_seg``) and ``stage_m`` (``M_seg^T``) ``[NSEG, H]`` and the carry plane
    ``[n_frag (P - 1), H]`` (forward carries, then the adjoint carries); ``eye`` = one identity tile.

    Operator orientation (K4): ``transpose_m = 0`` reads a tile holding ``Mop^T`` (the c1 ``M_f^T`` tiles and the
    ``M_seg^T`` staging under ``Mop = M``; the b1 ``M_f`` tiles under ``Mop = M^T``), ``transpose_m = 1`` a tile holding
    ``Mop`` (the c1 / staging ``M^T`` tiles under ``Mop = M^T``).  The segment compositions of a fragment (``P > 1``)
    therefore split into the forward family (``Mop = M_p``, segments in order, ``transpose_m = 0``: ``B_f``, ``M_f``,
    carries) and the adjoint family (``Mop = M_p^T``, segments in reverse, ``transpose_m = 1``: ``d_f``, ``M_f^T``,
    adjoint carries); each is one launch (ruling P1-R3 fixes the pair orientations ``[B_f | M_f^T]`` and ``[d_f | M_f]``).
    """

    _ring: _Ring | None = None  # the ring-exchange tables, derived on first use

    def __init__(
        self, ctx: Any, H: int, *, fused_walks: int = FUSED_WALKS_DEFAULT
    ) -> None:
        self.ctx = ctx
        self.H = H = int(H)
        self.fused_walks = int(
            fused_walks
        )  # lever L-B (P2-R1 / P2-R4 bundle): one base pointer per tape phase
        hp = self.hp = ctx.heads(H)
        self.P, self.NSEG = int(hp.P), int(hp.NSEG)
        self.Hg, self.Hg_r = int(hp.Hg), int(hp.Hg_r)
        self.n_frag, self.T_r = int(ctx.n_frag), int(ctx.T_r)
        self.n_in, self.n_out = int(ctx.n_in), int(ctx.n_out)
        self.max_hops = max(int(ctx.n_d_max) - 1, 0)
        defaults = {
            "fwd_m_off": SUMMARY_TILE,
            "fwd_b_off": SUMMARY_TILE,
            "fwd_out_off": TILE,
            "adj_d_off_d": TILE,
            "adj_d_off_dm": SUMMARY_TILE,
            "adj_m_off_b1": SUMMARY_TILE,
            "adj_m_off_c1": SUMMARY_TILE,
            "adj_out_off": TILE,
        }
        self.hg_stride = {
            **defaults,
            **{k: int(v) for k, v in (getattr(hp, "hg_stride", None) or {}).items()},
        }
        P, NSEG = self.P, self.NSEG
        self.frags = frags = list(ctx.fragments)
        in_slot, out_slot = list(ctx.in_slot), list(ctx.out_slot)
        live = [f for f in frags if f.length > 0]
        starts = [
            f for f in live if not f.crossing_in
        ]  # document-start fragments (single-fragment documents too)
        ins = [f for f in live if f.crossing_in]
        outs = [f for f in live if f.crossing_out]
        ends = [f for f in live if not f.crossing_out]  # document-last fragments

        seg_table = [int(x) for x in hp.seg_chunk_start]

        def segs(fs, ps):
            # LPT order (longest segment first; ties by segment number): the CTA issue order of a walk launch — the
            # kernels are per-CTA independent, so the order changes tail effects only, never a bit of the results
            out = [f.slot * P + p for f in fs for p in ps]
            return sorted(out, key=lambda s: (-(seg_table[s + 1] - seg_table[s]), s))

        # -- walk lists (segments)
        self.walk_offset = [
            int(s) for s in hp.walk_seg_offset
        ]  # K1 / K3 offset walks (crossing-out frags at P = 1, all at P > 1)
        self.walk_all = list(range(NSEG))
        self.walk_start = segs(
            starts, range(P)
        )  # tape under c1: zero (p = 0) / carried (p >= 1) initial states
        self.walk_in_p0 = segs(ins, [0])  # tape after c2 from S_in
        self.walk_in_p1 = segs(ins, range(1, P))  # tape after c2 from the carry plane
        self.walk_end = segs(
            ends, range(P)
        )  # adjoint tape under b2: zero (p = P - 1) / carried final adjoints
        self.walk_cross_last = segs(outs, [P - 1])  # adjoint tape after b2 from dS_out
        self.walk_cross_rest = segs(
            outs, range(P - 1)
        )  # adjoint tape after b2 from the adjoint carries
        self.walk_in = segs(ins, [0]) if P == 1 else self.walk_all  # K2 offset walk
        self.walk_m_b1 = (
            segs([f for f in ins if f.crossing_out], [0]) if P == 1 else self.walk_all
        )  # K3 M-only (b1)
        # fused_walks (L-B): one launch per tape phase -- every live segment after wait(c2) / wait(b2) (schedule
        # wait_first), or the crossing-in (crossing-out) fragments' segments when the document-start (document-last)
        # walk stays under the collective (schedule overlap)
        self.walk_live = segs(starts + ins, range(P))
        self.walk_in_fused = segs(ins, range(P))
        self.walk_cross_fused = segs(outs, range(P))
        # -- per-segment slot tables
        self.h0_index = tuple(int(i) for i in hp.h0_index)
        self.carry_index = tuple(int(i) for i in hp.carry_index)
        self.dht_index = tuple(int(i) for i in hp.dht_index)
        self.dcarry_index = tuple(int(i) for i in hp.dcarry_index)
        self.h0_index_fused = tuple(
            int(i) for i in hp.h0_index_fused
        )  # L-B: the carry plane (P > 1) / S_in (P = 1)
        self.dht_index_fused = tuple(
            int(i) for i in hp.dht_index_fused
        )  # L-B: the carry plane (P > 1) / dS_out (P = 1)
        self.out_index = tuple(int(i) for i in hp.out_index)
        self.in_index = tuple(
            in_slot[s // P] if (s % P == 0 and frags[s // P].crossing_in) else -1
            for s in range(NSEG)
        )
        self.m_b1_index = tuple(
            in_slot[s // P]
            if (s % P == 0 and frags[s // P].crossing_in and frags[s // P].crossing_out)
            else -1
            for s in range(NSEG)
        )
        self.seg_index = tuple(range(NSEG))
        # -- tile tables [slot][h]
        self.c1_b_tiles = tuple(tuple(2 * int(t) for t in row) for row in hp.out_tile)
        self.c1_m_tiles = tuple(
            tuple(2 * int(t) + 1 for t in row) for row in hp.out_tile
        )
        self.s_in_tiles = tuple(tuple(int(t) for t in row) for row in hp.h0_tile)
        self.ds_out_tiles = tuple(tuple(int(t) for t in row) for row in hp.dht_tile)
        self.b1_d_tiles = {
            "d": tuple(tuple(int(t) for t in row) for row in hp.b1_d_tile_d),
            "dm": tuple(tuple(int(t) for t in row) for row in hp.b1_d_tile_dm),
        }
        self.b1_m_tiles = tuple(tuple(int(t) for t in row) for row in hp.b1_m_tile_dm)
        self.stage_tiles = tuple(
            tuple(s * H + h for h in range(H)) for s in range(NSEG)
        )
        c1_tiles = int(hp.numel["c1_send"]) // TILE
        self.mf_tiles = (
            tuple(
                tuple(c1_tiles + o * H + h for h in range(H)) for o in range(self.n_out)
            )
            if P > 1
            else ()
        )
        n_carry = self.n_frag * (P - 1) if P > 1 else 0
        if (
            self.fused_walks and P > 1
        ):  # L-B: + one slot per crossing fragment for the S_in / dS_out copies
            n_carry += max(self.n_in, self.n_out)
        self.carry_tiles = tuple(
            tuple(c * H + h for h in range(H)) for c in range(n_carry)
        )
        # -- splits (elements) and pool sizes (elements)
        self.splits = {k: tuple(int(x) for x in v) for k, v in hp.splits.items()}
        numel = {k: int(v) for k, v in hp.numel.items()}
        self.numel = numel
        self.sizes = {
            "c1_send": numel["c1_send"] + (self.n_out * H * TILE if P > 1 else 0),
            "c1_recv": numel["c1_recv"],
            "c2_send": numel["c2_send"],
            "S_in": numel["c2_recv"],
            "b1_send": max(numel["b1_send_d"], numel["b1_send_dm"]),
            "b1_recv": max(numel["b1_recv_d"], numel["b1_recv_dm"]),
            "b2_send": numel["b2_send"],
            "b2_recv": numel["b2_recv"],
            "stage_b": NSEG * H * TILE if P > 1 else 0,
            "stage_m": NSEG * H * TILE if P > 1 else 0,
            "carry": n_carry * H * TILE,
            "eye": TILE,
            "ring_out": max(self.n_in, self.n_out) * H * TILE,
            "ring_in": max(self.n_in, self.n_out) * H * TILE,
        }
        self._check_tiles()
        self._build_chains(starts, ins, outs, ends, in_slot, out_slot)

    # K4 chain families: the pools their offsets index, the operator orientation and the state publication.
    FAMILIES: dict[str, dict[str, Any]] = {
        "fwd": {
            "m": "c1_recv",
            "c": "c1_recv",
            "x0": "c1_recv",
            "y": "c2_send",
            "tm": 0,
            "ws": 1,
        },
        "adj_b1": {
            "m": "b1_recv",
            "c": "b1_recv",
            "x0": "b1_recv",
            "y": "b2_send",
            "tm": 0,
            "ws": 1,
        },
        "adj_c1": {
            "m": "c1_recv",
            "c": "b1_recv",
            "x0": "b1_recv",
            "y": "b2_send",
            "tm": 1,
            "ws": 1,
        },
        "map": {
            "m": "stage_m",
            "c": "stage_b",
            "x0": "eye",
            "y": "c1_send",
            "tm": 0,
            "ws": 0,
        },
        "carry_start": {
            "m": "stage_m",
            "c": "stage_b",
            "x0": None,
            "y": "carry",
            "tm": 0,
            "ws": 1,
        },
        "carry_in": {
            "m": "stage_m",
            "c": "stage_b",
            "x0": "S_in",
            "y": "carry",
            "tm": 0,
            "ws": 1,
        },
        "tmap_d_d": {
            "m": "stage_m",
            "c": "stage_b",
            "x0": None,
            "y": "b1_send",
            "tm": 1,
            "ws": 0,
        },
        "tmap_d_dm": {
            "m": "stage_m",
            "c": "stage_b",
            "x0": None,
            "y": "b1_send",
            "tm": 1,
            "ws": 0,
        },
        "tmap_m": {
            "m": "stage_m",
            "c": None,
            "x0": "eye",
            "y": "b1_send",
            "tm": 0,
            "ws": 0,
        },
        "dcarry_end": {
            "m": "stage_m",
            "c": "stage_b",
            "x0": None,
            "y": "carry",
            "tm": 1,
            "ws": 1,
        },
        "dcarry_cross": {
            "m": "stage_m",
            "c": "stage_b",
            "x0": "b2_recv",
            "y": "carry",
            "tm": 1,
            "ws": 1,
        },
    }
    # the b1 pool bound of a family depends on the layout form it addresses
    _B1_BOUND = {
        "tmap_d_d": "b1_send_d",
        "tmap_d_dm": "b1_send_dm",
        "tmap_m": "b1_send_dm",
        "adj_b1": "b1_recv_dm",
        "adj_c1": "b1_recv_d",
    }

    def _check_tiles(self) -> None:
        def bound(name: str, tiles, limit_elems: int) -> None:
            top = max((t for row in tiles for t in row), default=-1)
            if (top + 1) * TILE > limit_elems:
                raise ValueError(
                    f"kcp tables: {name} addresses tile {top} beyond its pool of {limit_elems // TILE} tiles"
                )

        n = self.numel
        bound("c1 B tiles", self.c1_b_tiles, n["c1_send"])
        bound("c1 M tiles", self.c1_m_tiles, n["c1_send"])
        bound("S_in tiles", self.s_in_tiles, n["c2_recv"])
        bound("dS_out tiles", self.ds_out_tiles, n["b2_recv"])
        bound("b1 d tiles (d form)", self.b1_d_tiles["d"], n["b1_send_d"])
        bound("b1 d tiles (dm form)", self.b1_d_tiles["dm"], n["b1_send_dm"])
        bound("b1 M tiles", self.b1_m_tiles, n["b1_send_dm"])
        # the L-B tables address the fused carry plane (n_frag (P - 1) + max(n_in, n_out) slots; allocated only when
        # fused_walks is set, inert otherwise) at P > 1 and the S_in / dS_out pools at P = 1
        n_fused = self.n_frag * (self.P - 1) + max(self.n_in, self.n_out)
        fused_pool = (
            (n_fused, n_fused)
            if self.P > 1
            else (len(self.s_in_tiles), len(self.ds_out_tiles))
        )
        for name, idx, n_slots in (
            ("h0_index", self.h0_index, len(self.s_in_tiles)),
            ("dht_index", self.dht_index, len(self.ds_out_tiles)),
            ("carry_index", self.carry_index, len(self.carry_tiles)),
            ("dcarry_index", self.dcarry_index, len(self.carry_tiles)),
            ("in_index", self.in_index, len(self.b1_d_tiles["d"])),
            ("m_b1_index", self.m_b1_index, len(self.b1_m_tiles)),
            ("h0_index_fused", self.h0_index_fused, fused_pool[0]),
            ("dht_index_fused", self.dht_index_fused, fused_pool[1]),
        ):
            if len(idx) != self.NSEG or any(i >= n_slots for i in idx):
                raise ValueError(
                    f"kcp tables: {name} is not a valid [NSEG] slot table of its pool"
                )
        if self.P == 1 and any(i >= len(self.c1_b_tiles) for i in self.out_index):
            raise ValueError("kcp tables: out_index exceeds the c1 slots")
        if self.P > 1 and (
            any(i >= self.NSEG for i in self.out_index)
            or len(self.walk_offset) != self.NSEG
        ):
            raise ValueError(
                "kcp tables: at P > 1 the offset walk must visit every segment into its staging slot"
            )

    def _build_chains(self, starts, ins, outs, ends, in_slot, out_slot) -> None:
        H, P = self.H, self.P
        st = self.stage_tiles
        ch: dict[str, Chains] = {name: Chains() for name in self.FAMILIES}
        ch["fwd"] = self._owner_fwd()
        ch["adj_b1"] = self._owner_adj("adj_d_off_dm", "adj_m_off_b1")
        ch["adj_c1"] = self._owner_adj("adj_d_off_d", "adj_m_off_c1")
        if P > 1:

            def mt(s: int, h: int) -> int:
                return st[s][h] * TILE

            def carry(f, i: int, h: int) -> int:
                return self.carry_tiles[f.slot * (P - 1) + i][h] * TILE

            for f in outs:
                S = [f.slot * P + p for p in range(P)]
                for h in range(H):
                    ch["map"].add(
                        [(mt(s, h), mt(s, h), -1) for s in S],
                        y_final=self.c1_b_tiles[out_slot[f.slot]][h] * TILE,
                    )
                    ch["map"].add(
                        [(mt(s, h), -1, -1) for s in S],
                        x0=0,
                        y_final=self.mf_tiles[out_slot[f.slot]][h] * TILE,
                    )
            extra = (
                self.n_frag * (P - 1)
            )  # L-B: the extra carry slots hold the S_in (dS_out) copies of the fused form

            def copy_slot(
                f, h: int, slot: int
            ) -> (
                int
            ):  # elem_y of the first element = publish the entering state (P1-R1)
                return (
                    self.carry_tiles[extra + slot][h] * TILE if self.fused_walks else -1
                )

            for name, fs in (("carry_start", starts), ("carry_in", ins)):
                for f in fs:
                    for h in range(H):
                        elems = [
                            (
                                mt(f.slot * P + p, h),
                                mt(f.slot * P + p, h),
                                carry(f, p - 1, h) if p >= 1 else -1,
                            )
                            for p in range(P - 1)
                        ]
                        x0 = (
                            self.s_in_tiles[in_slot[f.slot]][h] * TILE
                            if name == "carry_in"
                            else -1
                        )
                        if name == "carry_in":
                            elems[0] = (
                                elems[0][0],
                                elems[0][1],
                                copy_slot(f, h, in_slot[f.slot]),
                            )
                        ch[name].add(elems, x0=x0, y_final=carry(f, P - 2, h))
            for form in ("d", "dm"):
                for f in ins:
                    S = [f.slot * P + p for p in range(P)]
                    for h in range(H):
                        ch[f"tmap_d_{form}"].add(
                            [(mt(s, h), mt(s, h), -1) for s in reversed(S)],
                            y_final=self.b1_d_tiles[form][in_slot[f.slot]][h] * TILE,
                        )
            for f in ins:
                if f.crossing_out:
                    S = [f.slot * P + p for p in range(P)]
                    for h in range(H):
                        ch["tmap_m"].add(
                            [(mt(s, h), -1, -1) for s in S],
                            x0=0,
                            y_final=self.b1_m_tiles[in_slot[f.slot]][h] * TILE,
                        )
            for name, fs in (("dcarry_end", ends), ("dcarry_cross", outs)):
                for f in fs:
                    for h in range(H):
                        elems = [
                            (
                                mt(f.slot * P + p, h),
                                mt(f.slot * P + p, h),
                                carry(f, p, h) if p <= P - 2 else -1,
                            )
                            for p in range(P - 1, 0, -1)
                        ]
                        x0 = (
                            self.ds_out_tiles[out_slot[f.slot]][h] * TILE
                            if name == "dcarry_cross"
                            else -1
                        )
                        if name == "dcarry_cross":
                            elems[0] = (
                                elems[0][0],
                                elems[0][1],
                                copy_slot(f, h, out_slot[f.slot]),
                            )
                        ch[name].add(elems, x0=x0, y_final=carry(f, 0, h))
        for name, c in ch.items():
            c.check_alignment()
            self._check_chain(name, c)
        self.chains = ch

    def _check_chain(self, name: str, c: Chains) -> None:
        spec = self.FAMILIES[name]
        limits = {
            "c1_send": self.sizes["c1_send"],
            "c1_recv": self.numel["c1_recv"],
            "c2_send": self.numel["c2_send"],
            "S_in": self.numel["c2_recv"],
            "b2_send": self.numel["b2_send"],
            "b2_recv": self.numel["b2_recv"],
            "stage_b": self.sizes["stage_b"],
            "stage_m": self.sizes["stage_m"],
            "carry": self.sizes["carry"],
            "eye": TILE,
        }
        b1 = self._B1_BOUND.get(name)
        if b1 is not None:
            limits["b1_send"] = limits["b1_recv"] = self.numel[b1]
        for key, values in (
            ("m", c.elem_m),
            ("c", c.elem_c),
            ("x0", c.x0_index),
            ("y", c.elem_y + c.y_final),
        ):
            pool = spec[key]
            top = max((v for v in values if v >= 0), default=-1)
            if top < 0:
                continue
            if pool is None or top + TILE > limits[pool]:
                raise ValueError(
                    f"kcp tables: chain family {name}: {key} offset {top} beyond pool {pool}"
                )

    def _owner_fwd(self) -> Chains:
        """The owner's forward chains (FINAL_DESIGN §3.5): per document and owned head, from ``X0 = B_{f1}`` (the
        start fragment's offset: ``S_in(f2)``) over the fragments ``f2 .. f(n-1)`` with ``Mop = M_f`` (c1 ``M_f^T``
        tiles, ``transpose_m = 0``); the state entering each element and the final state are the ``S_in`` of the
        element's fragment and of ``fn``, published into the c2 send layout."""
        hp, ch = self.hp, Chains()
        sm, sb, so = (
            self.hg_stride["fwd_m_off"],
            self.hg_stride["fwd_b_off"],
            self.hg_stride["fwd_out_off"],
        )
        starts = [int(x) for x in hp.fwd_chain_start]
        for ci in range(len(starts) - 1):
            e0, e1 = starts[ci], starts[ci + 1]
            for hg in range(self.Hg_r):
                elems = [
                    (
                        int(hp.fwd_m_off[e]) + hg * sm,
                        int(hp.fwd_b_off[e]) + hg * sb,
                        int(hp.fwd_out_off[e - 1]) + hg * so,
                    )
                    for e in range(e0 + 1, e1)
                ]
                ch.add(
                    elems,
                    x0=int(hp.fwd_b_off[e0]) + hg * sb,
                    y_final=int(hp.fwd_out_off[e1 - 1]) + hg * so,
                )
        return ch

    def _owner_adj(self, d_key: str, m_key: str) -> Chains:
        """The owner's adjoint chains (FINAL_DESIGN §6.5): per document and owned head, from ``X0 = d_{fn}`` (the
        last fragment's adjoint offset: ``dS_out(f(n-1))``) over ``f(n-1) .. f2`` with ``Mop = M_f^T``; the state
        entering each element and the final state are the ``dS_out`` of the element's predecessor, published into the
        b2 send layout.  ``d_key`` / ``m_key`` name the plan module's offset tables (``adj_d_off_dm`` +
        ``adj_m_off_b1``: the b1 pairs ``[d_f | M_f]``; ``adj_d_off_d`` + ``adj_m_off_c1``: the b1 ``d_f`` plane and the
        c1 ``M_f^T`` tiles); the head strides come from ``hp.hg_stride``."""
        hp, ch = self.hp, Chains()
        d_off, m_off = getattr(hp, d_key), getattr(hp, m_key)
        sd, sm, so = (
            self.hg_stride[d_key],
            self.hg_stride[m_key],
            self.hg_stride["adj_out_off"],
        )
        starts = [int(x) for x in hp.adj_chain_start]
        for ci in range(len(starts) - 1):
            e0, e1 = starts[ci], starts[ci + 1]
            for hg in range(self.Hg_r):
                elems = [
                    (
                        int(m_off[e]) + hg * sm,
                        int(d_off[e]) + hg * sd,
                        int(hp.adj_out_off[e - 1]) + hg * so,
                    )
                    for e in range(e0 + 1, e1)
                ]
                ch.add(
                    elems,
                    x0=int(d_off[e0]) + hg * sd,
                    y_final=int(hp.adj_out_off[e1 - 1]) + hg * so,
                )
        return ch

    def segment_tokens(self, seg: int) -> tuple[int, int]:
        """``[t0, t1)`` local token range of segment ``seg`` (``t0 == t1``: no chunk), from the chunk tables."""
        scs = self.hp.seg_chunk_start
        c0, c1 = int(scs[seg]), int(scs[seg + 1])
        if c1 <= c0:
            return 0, 0
        t0 = int(self.ctx.chunk_bos[c0])
        t1 = int(self.ctx.chunk_bos[c1 - 1]) + int(self.ctx.chunk_len[c1 - 1])
        return t0, t1

    def split_elems(self, coll: str) -> tuple[list[int], list[int]]:
        """``(input_split_sizes, output_split_sizes)`` in elements of ``c1``, ``c2``, ``b1_d``, ``b1_dm`` or ``b2``."""
        if coll.startswith("b1_"):
            form = coll[3:]
            return list(self.splits[f"b1_send_{form}"]), list(
                self.splits[f"b1_recv_{form}"]
            )
        return list(self.splits[f"{coll}_send"]), list(self.splits[f"{coll}_recv"])

    def device_tiles(self, tiles, device: torch.device) -> torch.Tensor:
        """The flat int64 device copy of a tile table ``[slot][h]`` (cached per device)."""
        key = (tuple(tuple(int(t) for t in row) for row in tiles), str(device))
        cache = self.__dict__.setdefault("_device_tiles", {})
        t = cache.get(key)
        if t is None:
            flat = [int(x) for row in tiles for x in row]
            t = torch.tensor(flat, dtype=torch.int64, device=device)
            cache[key] = t
        return t

    def pool_sizes(self, backward: bool) -> dict[str, int]:
        names = [
            "c1_send",
            "c1_recv",
            "c2_send",
            "stage_b",
            "stage_m",
            "carry",
            "eye",
            "ring_out",
            "ring_in",
        ]
        if backward:
            names += ["b1_send", "b1_recv", "b2_send", "b2_recv"]
        return {
            n: self.sizes[n] for n in names
        }  # zero-size pools included: every buffer name resolves

    def describe(self) -> dict[str, Any]:
        return {
            "P": self.P,
            "NSEG": self.NSEG,
            "Hg": self.Hg,
            "Hg_r": self.Hg_r,
            "n_frag": self.n_frag,
            "T_r": self.T_r,
            "n_in": self.n_in,
            "n_out": self.n_out,
            "max_hops": self.max_hops,
            "fused_walks": self.fused_walks,
            "walks": {
                "live": len(self.walk_live),
                "start": len(self.walk_start),
                "in_p0": len(self.walk_in_p0),
                "in_p1": len(self.walk_in_p1),
                "end": len(self.walk_end),
                "cross_last": len(self.walk_cross_last),
                "cross_rest": len(self.walk_cross_rest),
            },
            "chains": {
                k: {"n_chains": c.n_chains, "n_elems": c.n_elems}
                for k, c in self.chains.items()
            },
            "sizes": dict(self.sizes),
        }


# ----------------------------------------------------------------------------------------------------------------
# Workspace (FINAL_DESIGN §7.4): transient buffers only; saved-set tensors are fresh per call (SPEC-4)


class KcpWorkspace:
    """The transient buffers of the KCP route on one rank (FINAL_DESIGN §7.4) plus the existing backend's local
    arena (``cake_backend.KdaWorkspace``) for the token-side stages.

    Pools (flat, ``state_dtype``): ``c1_send``, ``c1_recv``, ``c2_send``; ``b1_send``, ``b1_recv``, ``b2_send``,
    ``b2_recv`` (``backward``); at ``P > 1`` the staging planes ``stage_b`` / ``stage_m`` and the ``carry`` plane;
    one identity tile ``eye``.  The saved-set tensors (``S_in`` = the c2 output, the c1 receive tensor under
    ``save_transfer``, ``M_seg`` under ``save_transfer`` at ``P > 1``) are never arena slices: the calls allocate
    them fresh (SPEC-4).  A workspace is reusable across calls and layers of the same ``(ctx, H)``; a call with a
    larger need grows it.  ``bytes`` counts the pools plus the local arena (:func:`kcp_workspace_bytes` is the
    plan module's prediction of the same quantity, CONSISTENCY-6)."""

    def __init__(
        self,
        ctx: Any | None = None,
        H: int | None = None,
        device: torch.device | str | None = None,
        *,
        backward: bool = True,
        state_dtype: torch.dtype = torch.float32,
        local: bool = True,
        save_mode: str = "light",
    ) -> None:
        if save_mode not in SAVE_MODES:
            raise ValueError(
                f"KcpWorkspace: save_mode must be one of {SAVE_MODES}, got {save_mode!r}"
            )
        self.state_dtype = state_dtype
        self.with_local = bool(local)
        self.save_mode = save_mode
        self.device: torch.device | None = None
        self._pools: dict[str, torch.Tensor] = {}
        self._tables: dict[tuple[Any, int, int], _Tables] = {}
        self.local: Any | None = None
        if ctx is not None:
            if H is None:
                raise ValueError("KcpWorkspace: H is required with ctx")
            self.ensure(ctx, H, device=device, backward=backward)

    def tables(
        self, ctx: Any, H: int, *, fused_walks: int = FUSED_WALKS_DEFAULT
    ) -> _Tables:
        key = (ctx, int(H), int(fused_walks))
        tb = self._tables.get(key)
        if tb is None:
            tb = _Tables(ctx, H, fused_walks=fused_walks)
            self._tables[key] = tb
        return tb

    def ensure(
        self,
        ctx: Any,
        H: int,
        *,
        device: torch.device | str | None = None,
        backward: bool = True,
        fused_walks: int = FUSED_WALKS_DEFAULT,
    ) -> _Tables:
        """Size the pools (and the local arena) for ``(ctx, H)``; returns the tables."""
        device = (
            torch.device(device) if device is not None else torch.device(ctx.device)
        )
        if self.device is None:
            self.device = device
        elif self.device != device:
            raise ValueError(
                f"KcpWorkspace: lives on {self.device}, the call is on {device}"
            )
        tb = self.tables(ctx, H, fused_walks=fused_walks)
        for name, n in tb.pool_sizes(backward).items():
            cur = self._pools.get(name)
            if cur is None or cur.numel() < n:
                self._pools[name] = _alloc((n,), self.state_dtype, device)
                if name == "eye":
                    self._pools[name].view(D, D).copy_(
                        torch.eye(D, dtype=self.state_dtype, device=device)
                    )
        if self.with_local and ctx.T_r > 0 and torch.device(device).type == "cuda":
            p = self.local.plan if self.local is not None else None
            if (
                p is None
                or (ctx.T_r, ctx.n_frag, int(H)) != (p.S, p.N, p.H)
                or (backward and not self.local.backward)
            ):
                self.local = cake_backend.KdaWorkspace.for_problem(
                    ctx.T_r,
                    ctx.n_frag,
                    int(H),
                    device,
                    backward=backward,
                    save_mode=self.save_mode,
                )
        return tb

    def pool(self, name: str, n: int | None = None) -> torch.Tensor:
        t = self._pools.get(name)
        if t is None:
            if n not in (None, 0) or self.device is None:
                raise KeyError(
                    f"KcpWorkspace: pool {name!r} was not sized for this call (ensure() first)"
                )
            t = self._pools[name] = _alloc((0,), self.state_dtype, self.device)
        return t if n is None else t[:n]

    def has(self, name: str) -> bool:
        return name in self._pools

    @property
    def bytes(self) -> int:
        kcp = sum(t.numel() * t.element_size() for t in self._pools.values())
        return kcp + (int(self.local.bytes) if self.local is not None else 0)

    @property
    def pool_bytes(self) -> dict[str, int]:
        return {k: t.numel() * t.element_size() for k, t in self._pools.items()}


# ----------------------------------------------------------------------------------------------------------------
# Saved set (FINAL_DESIGN §16), options (§10.1), exchange scheme (SPEC-2)


@dataclasses.dataclass(frozen=True)
class KcpSaved:
    """The saved context of one forward (``boundary_mode="saved"``, FINAL_DESIGN §16): ``S_in`` (the fresh c2
    output), the owner's c1 receive tensor (``save_transfer``), ``M_seg`` (``save_transfer`` at ``P > 1``), the
    token-row ``Aqk`` / ``Akk`` of the existing version-1 set (kernel path), and the scalars of the call.  The int32
    tables are rebuilt from ``ctx`` (droppable, §16 item 3)."""

    version: int
    S_in: torch.Tensor
    c1_recv: torch.Tensor | None
    M_seg: torch.Tensor | None
    Aqk: torch.Tensor | None
    Akk: torch.Tensor | None
    T_r: int
    H: int
    n_frag: int
    P: int
    Hg: int
    scale: float
    save_mode: str
    save_transfer: bool
    offset_class: str
    exchange: str
    precision_mode: int
    ctx: Any
    transfer_walk: str = "tf32_hilo"

    _OPTIONAL = ("c1_recv", "M_seg", "Aqk", "Akk")

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (self.S_in,) + tuple(
            t for t in (self.c1_recv, self.M_seg, self.Aqk, self.Akk) if t is not None
        )

    def flags(self) -> tuple[bool, ...]:
        return tuple(getattr(self, n) is not None for n in self._OPTIONAL)

    def meta(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "T_r": self.T_r,
            "H": self.H,
            "n_frag": self.n_frag,
            "P": self.P,
            "Hg": self.Hg,
            "scale": self.scale,
            "save_mode": self.save_mode,
            "save_transfer": self.save_transfer,
            "offset_class": self.offset_class,
            "exchange": self.exchange,
            "precision_mode": self.precision_mode,
            "transfer_walk": self.transfer_walk,
        }

    @classmethod
    def from_tensors(
        cls,
        tensors: Sequence[torch.Tensor],
        flags: Sequence[bool],
        meta: dict[str, Any],
        ctx: Any,
    ) -> KcpSaved:
        it = iter(tensors)
        S_in = next(it)
        opt = {
            n: (next(it) if f else None)
            for n, f in zip(cls._OPTIONAL, flags, strict=True)
        }
        return cls(S_in=S_in, ctx=ctx, **opt, **meta)

    @property
    def bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.tensors())


@dataclasses.dataclass
class KcpOptions:
    """The extra keywords of the entry points (FINAL_DESIGN §10.1): ``save_transfer`` (R6: keep the owner's c1
    receive tensor / ``M_seg`` for the backward; default :data:`cake_kcp_plan.SAVE_TRANSFER_DEFAULT` = ``True`` since
    the round-1 adoption, ruling P2-R3; ``False`` = the round-0 recompute form, which re-sends ``M_f^T`` in b1),
    ``exchange`` (``auto`` | ``a2a`` | ``ring``; ring only when ``n_d_max - 1 <= 2``, SPEC-2, and only with
    ``save_transfer=False``: the ring path receives no c1 tensor), ``head_groups`` (``G``; Phase 1 ships ``1``),
    ``offset_class`` (``fla``: K1 / K2 offsets; ``fp32`` is a Phase-2 knob), ``debug_sequence`` (R7: all-gather the
    collective issue order and assert it), ``precision_mode`` (K3's static program; set only through ``_kcp_test_knobs``)."""

    save_transfer: bool = (
        cake_kcp_plan.SAVE_TRANSFER_DEFAULT
    )  # P2-R3: True shipped; False = round-0 recompute form
    exchange: str = "auto"
    head_groups: int = 1
    offset_class: str = "fla"
    debug_sequence: bool = False
    precision_mode: int = 0
    transfer_walk: str = (
        "tf32_hilo"  # K3 program family (P1-R15); set only through _kcp_test_knobs
    )
    p_rule: str | None = (
        None  # plan P rule named by _kcp_test_knobs (lever L-C); None = the context's; must equal ctx.p_rule
    )
    schedule: str = SCHEDULE_DEFAULT  # L-A (P2-R1): "wait_first" (shipped) issues the merges and tape walks after each wait; "overlap" via _kcp_test_knobs
    fused_walks: int = FUSED_WALKS_DEFAULT  # L-B (shipped 1): one tape launch per phase from the carry plane (+ the S_in / dS_out copies); 0 via _kcp_test_knobs
    cols_rule: str = "fixed"  # D1 (P2-R4): "by_chains" = c8 up to COLS_RULE_CHAINS state chains, c16 beyond; _kcp_test_knobs only

    def check(self) -> KcpOptions:
        if self.exchange not in EXCHANGES:
            raise ValueError(
                f"exchange must be one of {EXCHANGES}, got {self.exchange!r}"
            )
        if self.offset_class not in OFFSET_CLASSES:
            raise ValueError(
                f"offset_class must be one of {OFFSET_CLASSES}, got {self.offset_class!r}"
            )
        if self.offset_class != "fla":
            raise NotImplementedError(
                "offset_class='fp32' (K3's offset program) is a Phase-2 knob"
            )
        if int(self.head_groups) != 1:
            raise NotImplementedError(
                "head_groups > 1 (lever L3) is not part of Phase 1"
            )
        self.save_transfer = bool(self.save_transfer)
        self.debug_sequence = bool(self.debug_sequence)
        self.precision_mode = int(self.precision_mode)
        if self.transfer_walk not in TRANSFER_WALKS:
            raise ValueError(
                f"transfer_walk must be one of {TRANSFER_WALKS}, got {self.transfer_walk!r}"
            )
        if self.precision_mode and self.transfer_walk != "tf32_hilo":
            raise ValueError(
                "precision_mode selects a program of the tf32_hilo family; it cannot combine with transfer_walk"
            )
        for key, allowed in P2_KNOBS.items():
            if getattr(self, key) not in allowed:
                raise ValueError(
                    f"{key} must be one of {allowed}, got {getattr(self, key)!r}"
                )
        return self


_OPTION_KEYS = (
    "save_transfer",
    "exchange",
    "head_groups",
    "offset_class",
    "debug_sequence",
)


def _options(kwargs: dict[str, Any]) -> KcpOptions:
    """Validate the entry points' ``**options`` (unknown keywords are a ``TypeError``); ``_kcp_test_knobs`` goes
    through the plan module's validator (``precision_mode``) and, for the P1-R15 knob ``transfer_walk`` (K3's
    ``bf16x3`` / ``bf16x3_hw64`` candidates of ``build.test.json``), through the same environment gate here, so the
    public API can never select a non-zero ``precision_mode`` or a non-shipped transfer-walk program."""
    kwargs = dict(kwargs)
    raw = kwargs.pop("_kcp_test_knobs", None)
    transfer_walk = TRANSFER_WALKS[0]
    if isinstance(raw, dict) and "transfer_walk" in raw:
        if os.environ.get(cake_kcp_plan.TEST_KNOBS_ENV) != "1":
            raise ValueError(
                f"_kcp_test_knobs is a private test hook: it is accepted only with {cake_kcp_plan.TEST_KNOBS_ENV}=1 in the environment"
            )
        raw = dict(raw)
        transfer_walk = str(raw.pop("transfer_walk"))
        if transfer_walk not in TRANSFER_WALKS:
            raise ValueError(
                f"_kcp_test_knobs: transfer_walk must be one of {TRANSFER_WALKS}, got {transfer_walk!r}"
            )
    p2: dict[str, Any] = {}
    if isinstance(raw, dict) and any(
        k in raw for k in P2_KNOBS
    ):  # round-1 levers (P2-R1 / P2-R4): the same gate
        if os.environ.get(cake_kcp_plan.TEST_KNOBS_ENV) != "1":
            raise ValueError(
                f"_kcp_test_knobs is a private test hook: it is accepted only with {cake_kcp_plan.TEST_KNOBS_ENV}=1 in the environment"
            )
        raw = dict(raw)
        for key in P2_KNOBS:
            if key in raw:
                p2[key] = (
                    int(raw.pop(key)) if key == "fused_walks" else str(raw.pop(key))
                )
    knobs = cake_kcp_plan.validate_test_knobs(raw)
    p_rule = (
        knobs["p_rule"] if isinstance(raw, dict) and "p_rule" in raw else None
    )  # lever L-C: the context carries the rule
    unknown = sorted(set(kwargs) - set(_OPTION_KEYS))
    if unknown:
        raise TypeError(
            f"unexpected keyword argument(s) {unknown}; known options: {list(_OPTION_KEYS)} and _kcp_test_knobs"
        )
    return KcpOptions(
        **kwargs,
        precision_mode=int(knobs["precision_mode"]),
        transfer_walk=transfer_walk,
        p_rule=p_rule,
        **p2,
    ).check()


def _exchange_scheme(tb: _Tables, opts: KcpOptions) -> str:
    """``a2a`` or ``ring`` (SPEC-2): ``auto`` is ``a2a`` in Phase 1 (the ring path arrives with gate 5); an explicit
    ``ring`` needs every document chain within ``RING_MAX_HOPS`` hops."""
    if opts.exchange == "ring":
        if tb.max_hops > RING_MAX_HOPS:
            raise ValueError(
                f"exchange='ring' needs n_d_max - 1 <= {RING_MAX_HOPS}, this partition has {tb.max_hops}"
            )
        if opts.save_transfer:
            raise NotImplementedError(
                "exchange='ring' receives no c1 tensor: save_transfer=True has no ring form"
            )
        return "ring"
    return "a2a"


# ----------------------------------------------------------------------------------------------------------------
# One call (forward or backward): buffers, launches, collectives


class KcpCall:
    """The state of one forward or backward call on one rank: tables, buffers (workspace pools and the call's fresh
    tensors), operands, backend, the launch / collective counters and the issue sequence (R7 debug counter)."""

    def __init__(
        self,
        *,
        ctx: Any,
        H: int,
        ws: KcpWorkspace,
        tb: _Tables,
        backend: Backend,
        opts: KcpOptions,
        scale: float,
        ops: dict[str, torch.Tensor],
        backward: bool,
    ) -> None:
        self.ctx, self.H, self.ws, self.tb, self.backend, self.opts = (
            ctx,
            int(H),
            ws,
            tb,
            backend,
            opts,
        )
        self.scale = float(scale)
        self.device = ws.device
        self.rank, self.world, self.group = (
            int(ctx.rank),
            int(ctx.world_size),
            ctx.group,
        )
        self.state_dtype = ws.state_dtype
        self.ops = ops
        self.T_r = int(ctx.T_r)
        self.backward = bool(backward)
        self.counters = {
            "launches": 0,
            "collectives": 0,
            "walks": 0,
            "merges": 0,
            "stages": 0,
        }
        self.sequence: list[str] = []
        self.merge_log: list[
            tuple[str, int, int, int | None]
        ] = []  # (family, ncols, n_chains, explicit cols_per_cta) per K4 launch
        self.scratch: dict[str, Any] = {}
        self.bufs: dict[str, torch.Tensor] = {}
        self.local = (
            ws.local if (backend.local_stages and ws.local is not None) else None
        )
        self.local_values: dict[str, Any] | None = None
        self.out: torch.Tensor | None = None
        self.grads: dict[str, torch.Tensor] | None = None
        self.exchange = _exchange_scheme(tb, opts)
        self.precision_mode = int(opts.precision_mode)
        self.transfer_walk = str(opts.transfer_walk)
        if opts.p_rule is not None and opts.p_rule != getattr(
            ctx, "p_rule", cake_kcp_plan.P_RULE_DEFAULT
        ):
            raise ValueError(
                f"_kcp_test_knobs p_rule {opts.p_rule!r} differs from the context's plan rule {getattr(ctx, 'p_rule', cake_kcp_plan.P_RULE_DEFAULT)!r}: give the knob to build_kcp_context"
            )
        self.schedule = str(opts.schedule)  # L-A
        self.cols_rule = str(opts.cols_rule)  # D1
        if int(tb.fused_walks) != int(opts.fused_walks):
            raise ValueError(
                f"tables built with fused_walks={tb.fused_walks}, the call asks for {opts.fused_walks}"
            )

    # -- buffers
    def buf(self, name: str) -> torch.Tensor:
        t = self.bufs.get(name)
        return t if t is not None else self.ws.pool(name, self.tb.sizes[name])

    def fresh(self, name: str) -> torch.Tensor:
        """A fresh per-call tensor standing in for pool ``name`` (the saved-set tensors of SPEC-4)."""
        t = _alloc((self.tb.sizes[name],), self.state_dtype, self.device)
        self.bufs[name] = t
        return t

    def indexed(
        self, pool: str, index: Sequence[int], tiles: Sequence[Sequence[int]]
    ) -> Indexed:
        return Indexed(
            self.buf(pool),
            tuple(int(i) for i in index),
            tuple(tuple(int(t) for t in row) for row in tiles),
        )

    # -- launches
    def kernel(self, name: str, **args: Any) -> None:
        self.counters["launches"] += 1
        self.sequence.append(name)
        args["_call"] = self
        self.backend.run(name, args)

    def walk(self, name: str, walk: Sequence[int], **args: Any) -> None:
        """A K1 / K2 / K3 launch over the segments ``walk`` (nothing when the list is empty)."""
        if len(walk) == 0:
            return
        self.counters["walks"] += 1
        self.kernel(name, walk=[int(s) for s in walk], **args)

    def merge(
        self,
        family: str,
        *,
        chains: Chains | None = None,
        spec: dict[str, Any] | None = None,
    ) -> None:
        """One K4 launch of chain family ``family`` (nothing when it has no chain); ``chains`` / ``spec`` replace the
        table's family (the ring exchange's one-element chains)."""
        ch = self.tb.chains[family] if chains is None else chains
        if ch.n_chains == 0:
            return
        spec = _Tables.FAMILIES[family] if spec is None else spec
        self.counters["merges"] += 1
        self.kernel(
            "kcp_affine_merge",
            family=family,
            m=self.buf(spec["m"]),
            c=self.buf(spec["c"]) if spec["c"] else None,
            x0=self.buf(spec["x0"]) if spec["x0"] else None,
            y=self.buf(spec["y"]),
            chains=ch,
            transpose_m=int(spec["tm"]),
            write_states=int(spec["ws"]),
        )

    def transpose_tiles(self, src: str, src_tiles, dst: str, dst_tiles) -> None:
        """Scatter the transposes of the tiles ``src_tiles[slot][h]`` of pool ``src`` into ``dst_tiles[slot][h]`` of
        pool ``dst`` (``M_f`` -> ``M_f^T``, bitwise); nothing when there is no tile."""
        n = sum(len(row) for row in src_tiles)
        if n == 0:
            return
        self.counters["launches"] += 1
        self.sequence.append("kcp_transpose_tiles")
        self.backend.transpose_tiles(
            src=self.buf(src),
            src_tiles=self.tb.device_tiles(src_tiles, self.device),
            dst=self.buf(dst),
            dst_tiles=self.tb.device_tiles(dst_tiles, self.device),
        )

    def stage(self, name: str, **extra: Any) -> None:
        """A token-side stage of the existing backend (skipped without a local arena or tokens)."""
        if self.local is None or self.T_r == 0:
            return
        self.counters["launches"] += 1
        self.counters["stages"] += 1
        self.sequence.append(name)
        self.backend.stage(name, self, **extra)

    # -- collectives
    def a2a(
        self,
        tag: str,
        send_name: str,
        recv_name: str,
        send_splits: Sequence[int],
        recv_splits: Sequence[int],
    ):
        """Issue ``all_to_all_single`` (async) from pool ``send_name`` into ``recv_name`` with element splits per
        peer; returns the work handle (``None`` on a single rank)."""
        send_splits, recv_splits = (
            [int(x) for x in send_splits],
            [int(x) for x in recv_splits],
        )
        if len(send_splits) != self.world or len(recv_splits) != self.world:
            raise ValueError(
                f"{tag}: split lists must have one entry per rank ({self.world})"
            )
        send = self.buf(send_name)[: sum(send_splits)]
        recv = self.buf(recv_name)[: sum(recv_splits)]
        self.counters["collectives"] += 1
        self._sequence(tag)
        if self.world == 1 or not (dist.is_available() and dist.is_initialized()):
            if send.numel() != recv.numel():
                raise ValueError(
                    f"{tag}: a single rank must send exactly what it receives"
                )
            recv.copy_(send)
            return None
        return dist.all_to_all_single(
            recv,
            send,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
            group=self.group,
            async_op=True,
        )

    @staticmethod
    def wait(work) -> None:
        if work is not None:
            work.wait()

    def _sequence(self, tag: str, kind: str = "a2a") -> None:
        self.sequence.append(f"{kind}:{tag}")
        seq = getattr(self.ctx, "sequence", None)
        n = seq.next(tag) if seq is not None else len(self.sequence)
        if (
            self.opts.debug_sequence
            and self.world > 1
            and dist.is_available()
            and dist.is_initialized()
        ):
            gathered: list[Any] = [None] * self.world
            dist.all_gather_object(gathered, (tag, n), group=self.group)
            if any(g != (tag, n) for g in gathered):
                raise RuntimeError(
                    f"debug_sequence: the collective issue order differs across ranks: {gathered}"
                )


LAST_CALL: dict[str, dict[str, Any]] = {}


def _record(kind: str, call: KcpCall) -> None:
    """The launch inventory of the last forward / backward (FINAL_DESIGN §10.5; M1 reads it)."""
    LAST_CALL[kind] = {
        **{f"n_{k}": v for k, v in call.counters.items()},
        "sequence": list(call.sequence),
        "exchange": call.exchange,
        "backend": call.backend.name,
        "P": call.tb.P,
        "NSEG": call.tb.NSEG,
        "Hg": call.tb.Hg,
        "Hg_r": call.tb.Hg_r,
        "n_in": call.tb.n_in,
        "n_out": call.tb.n_out,
        "workspace_bytes": call.ws.bytes,
        "pool_bytes": call.ws.pool_bytes,
        "fresh_bytes": {k: t.numel() * t.element_size() for k, t in call.bufs.items()},
        "precision_mode": call.precision_mode,
        "transfer_walk": call.transfer_walk,
        "schedule": call.schedule,
        "fused_walks": call.tb.fused_walks,
        "cols_rule": call.cols_rule,  # round-1 knobs (P2-R1 / P2-R4)
        "merges": list(
            call.merge_log
        ),  # D1 evidence: (family, ncols, n_chains, explicit cols_per_cta or None = registry)
        "spec3": call.scratch.get(
            "spec3"
        ),  # the SPEC-3 debug assertion's result (first saved-mode backward)
    }


# ----------------------------------------------------------------------------------------------------------------
# Launch sequences (FINAL_DESIGN §3.5 forward, §6.5 backward; one compute stream, waits at first consumption, R7;
# shipped since round 1: schedule wait_first + fused tape walks, the R7 overlaps only under the overlap knob)


# ----------------------------------------------------------------------------------------------------------------
# Ring exchange (SPEC-2; gate 5): per-document chains of point-to-point hops on ``ctx.group`` instead of the owner
# merge + c1 / c2 (b1 / b2).  Hop ``h`` of the forward: every crossing-out fragment at chain position ``h - 1`` sends
# its exit state ``S_out = M_f S_in + B_f`` (hop 1: ``B_f`` itself) to the fragment at position ``h``; the receiver
# writes it into its ``S_in`` tiles, so the tape walks after the hops read exactly what the a2a path's c2 delivers
# (bitwise: the same K4 element step on the same tile bits, or a plain copy).  The backward runs the hops from the
# document end: the fragment at position ``n_d - h`` sends ``dS_in = M_f^T dS_out + d_f`` (hop 1: ``d_f``) to its
# predecessor as that fragment's ``dS_out`` (into the ``b2_recv`` tiles).  One ``batch_isend_irecv`` per hop in
# ``(doc, slot)`` order; the document-start (document-last) tape walk overlaps hop 1 (R7).


class _Ring:
    """Per-hop send / receive lists of one rank (positions from ``ctx.chains``), built once per ``_Tables``."""

    def __init__(self, tb: _Tables) -> None:
        ctx = tb.ctx
        rank = int(ctx.rank)
        self.hops = tb.max_hops
        pos: dict[
            int, tuple[int, int]
        ] = {}  # slot -> (position in its document chain, chain length)
        for f in tb.frags:
            if f.length <= 0:
                continue
            chain = list(ctx.chains[f.doc])
            pos[f.slot] = (chain.index((rank, f.slot)), len(chain))
        self.pos = pos
        in_slot, out_slot = list(ctx.in_slot), list(ctx.out_slot)
        # forward hop h: senders at position h - 1 (crossing out), receivers at position h (crossing in)
        self.fwd_send = {
            h: [
                f
                for f in tb.frags
                if f.length > 0 and f.crossing_out and pos[f.slot][0] == h - 1
            ]
            for h in range(1, self.hops + 1)
        }
        self.fwd_recv = {
            h: [
                f
                for f in tb.frags
                if f.length > 0 and f.crossing_in and pos[f.slot][0] == h
            ]
            for h in range(1, self.hops + 1)
        }
        # backward hop h: senders at position n_d - h (crossing in), receivers at position n_d - h - 1 (crossing out)
        self.bwd_send = {
            h: [
                f
                for f in tb.frags
                if f.length > 0
                and f.crossing_in
                and pos[f.slot][0] == pos[f.slot][1] - h
            ]
            for h in range(1, self.hops + 1)
        }
        self.bwd_recv = {
            h: [
                f
                for f in tb.frags
                if f.length > 0
                and f.crossing_out
                and pos[f.slot][0] == pos[f.slot][1] - h - 1
            ]
            for h in range(1, self.hops + 1)
        }
        self.in_slot, self.out_slot = in_slot, out_slot

        def peer(f, step):
            chain = ctx.chains[f.doc]
            return int(chain[pos[f.slot][0] + step][0])

        self.next_rank = {
            f.slot: peer(f, 1) for f in tb.frags if f.length > 0 and f.crossing_out
        }
        self.prev_rank = {
            f.slot: peer(f, -1) for f in tb.frags if f.length > 0 and f.crossing_in
        }


def _ring_tables(tb: _Tables) -> _Ring:
    if tb._ring is None:
        tb._ring = _Ring(tb)
    return tb._ring


def _ring_block(pool: torch.Tensor, slot: int, H: int) -> torch.Tensor:
    """The contiguous ``[H, D, D]`` block of ``slot`` in a ring staging pool (tiles ``slot * H + h``)."""
    return pool.view(-1)[slot * H * TILE : (slot + 1) * H * TILE].view(H, D, D)


def _ring_copy_in(
    call: KcpCall,
    tb: _Tables,
    src: torch.Tensor,
    src_tiles,
    slot_src: int,
    dst: torch.Tensor,
    slot_dst: int,
) -> None:
    """``dst`` ring block ``slot_dst`` <- the ``H`` tiles ``src_tiles[slot_src][h]`` of pool ``src`` (bitwise copy)."""
    idx = tb.device_tiles([src_tiles[slot_src]], call.device)
    _ring_block(dst, slot_dst, call.H).copy_(src.view(-1, D, D)[idx.view(-1)])


def _ring_scatter(
    call: KcpCall,
    tb: _Tables,
    src: torch.Tensor,
    slot_src: int,
    dst: torch.Tensor,
    dst_tiles,
    slot_dst: int,
) -> None:
    """Pool ``dst`` tiles ``dst_tiles[slot_dst][h]`` <- the ring block ``slot_src`` of ``src`` (bitwise copy)."""
    idx = tb.device_tiles([dst_tiles[slot_dst]], call.device)
    dst.view(-1, D, D)[idx.view(-1)] = _ring_block(src, slot_src, call.H)


def _ring_p2p(
    call: KcpCall,
    hop: int,
    sends: list[tuple[torch.Tensor, int]],
    recvs: list[tuple[torch.Tensor, int]],
):
    """One ``batch_isend_irecv`` of the hop (sends then receives, each list in ``(doc, slot)`` order); returns the
    works (empty when the rank has nothing to exchange)."""
    call.counters["collectives"] += 1
    call._sequence(f"hop{hop}", kind="ring")
    if call.world == 1 or not dist.is_initialized():
        for _t, _r in sends:  # a one-rank chain cannot exist (no crossing); defensive
            raise RuntimeError("ring exchange on a single rank")
        return []
    ops = [dist.P2POp(dist.isend, t, peer, group=call.group) for t, peer in sends]
    ops += [dist.P2POp(dist.irecv, t, peer, group=call.group) for t, peer in recvs]
    return dist.batch_isend_irecv(ops) if ops else []


def _ring_wait(works) -> None:
    for w in works:
        w.wait()


def _ring_chain(
    tb: _Tables, family: str, elems: list[tuple[int, int, int, int]]
) -> Chains:
    """One-element chains per (fragment, head) from tile numbers ``(x0, m, c, y)`` (K4 state step on one fragment)."""
    ch = Chains(ncols=D)
    for (
        x0,
        m,
        c,
        y,
    ) in (
        elems
    ):  # tile numbers -> element offsets (the chain tables address pools in elements)
        ch.add([(m * TILE, c * TILE, -1)], x0=x0 * TILE, y_final=y * TILE)
    return ch


def _ring_forward(call: KcpCall) -> None:
    """Hops of the forward ring; afterwards ``S_in`` holds the crossing-in fragments' initial states (as after c2)."""
    tb, H = call.tb, call.H
    ring = _ring_tables(tb)
    out_pool, in_pool = call.buf("ring_out"), call.buf("ring_in")
    c1 = call.buf("c1_send")
    s_in = call.buf("S_in")
    for h in range(1, ring.hops + 1):
        senders, receivers = ring.fwd_send[h], ring.fwd_recv[h]
        if (
            h == 1
        ):  # S_out of a document-start fragment is its B_f: copy the c1 B tiles into the contiguous block
            for f in senders:
                _ring_copy_in(
                    call,
                    tb,
                    c1,
                    tb.c1_b_tiles,
                    ring.out_slot[f.slot],
                    out_pool,
                    ring.out_slot[f.slot],
                )
        else:  # S_out = M_f S_in + B_f: one K4 element per sender (forward family: c1 tiles hold M_f^T)
            elems = [
                (
                    tb.s_in_tiles[ring.in_slot[f.slot]][hh],
                    tb.c1_m_tiles[ring.out_slot[f.slot]][hh],
                    tb.c1_b_tiles[ring.out_slot[f.slot]][hh],
                    ring.out_slot[f.slot] * H + hh,
                )
                for f in senders
                for hh in range(H)
            ]
            if elems:
                call.merge(
                    f"ring_fwd_{h}",
                    chains=_ring_chain(tb, f"ring_fwd_{h}", elems),
                    spec={
                        "m": "c1_send",
                        "c": "c1_send",
                        "x0": "S_in",
                        "y": "ring_out",
                        "tm": 0,
                        "ws": 0,
                    },
                )
        works = _ring_p2p(
            call,
            h,
            [
                (
                    _ring_block(out_pool, ring.out_slot[f.slot], H),
                    ring.next_rank[f.slot],
                )
                for f in senders
            ],
            [
                (_ring_block(in_pool, ring.in_slot[f.slot], H), ring.prev_rank[f.slot])
                for f in receivers
            ],
        )
        if h == 1 and call.schedule == "overlap":
            carry = (
                call.indexed("carry", tb.carry_index, tb.carry_tiles)
                if tb.P > 1
                else None
            )
            call.walk(
                "kcp_fwdh_tape", tb.walk_start, h0=carry, out=call.out
            )  # document starts: under hop 1 (R7)
        _ring_wait(works)
        for f in receivers:
            _ring_scatter(
                call,
                tb,
                in_pool,
                ring.in_slot[f.slot],
                s_in,
                tb.s_in_tiles,
                ring.in_slot[f.slot],
            )


def _ring_backward(call: KcpCall) -> None:
    """Hops of the adjoint ring; afterwards ``b2_recv`` holds the crossing-out fragments' ``dS_out`` (as after b2).
    Requires the b1 ``dm`` form (``d_f`` and ``M_f`` of the crossing-in fragments in ``b1_send``)."""
    tb, H = call.tb, call.H
    ring = _ring_tables(tb)
    out_pool, in_pool = call.buf("ring_out"), call.buf("ring_in")
    b1, b2 = call.buf("b1_send"), call.buf("b2_recv")
    d_tiles, m_tiles = tb.b1_d_tiles["dm"], tb.b1_m_tiles
    for h in range(1, ring.hops + 1):
        senders, receivers = ring.bwd_send[h], ring.bwd_recv[h]
        if h == 1:  # the document-last fragment's dS_in is its d_f
            for f in senders:
                _ring_copy_in(
                    call,
                    tb,
                    b1,
                    d_tiles,
                    ring.in_slot[f.slot],
                    out_pool,
                    ring.in_slot[f.slot],
                )
        else:  # dS_in = M_f^T dS_out + d_f (adjoint step; the b1 M_f tiles hold Mop^T under Mop = M_f^T)
            elems = [
                (
                    tb.ds_out_tiles[ring.out_slot[f.slot]][hh],
                    m_tiles[ring.in_slot[f.slot]][hh],
                    d_tiles[ring.in_slot[f.slot]][hh],
                    ring.in_slot[f.slot] * H + hh,
                )
                for f in senders
                for hh in range(H)
            ]
            if elems:
                call.merge(
                    f"ring_bwd_{h}",
                    chains=_ring_chain(tb, f"ring_bwd_{h}", elems),
                    spec={
                        "m": "b1_send",
                        "c": "b1_send",
                        "x0": "b2_recv",
                        "y": "ring_out",
                        "tm": 0,
                        "ws": 0,
                    },
                )
        works = _ring_p2p(
            call,
            h,
            [
                (_ring_block(out_pool, ring.in_slot[f.slot], H), ring.prev_rank[f.slot])
                for f in senders
            ],
            [
                (_ring_block(in_pool, ring.out_slot[f.slot], H), ring.next_rank[f.slot])
                for f in receivers
            ],
        )
        if h == 1 and call.schedule == "overlap":
            dcarry = (
                call.indexed("carry", tb.dcarry_index, tb.carry_tiles)
                if tb.P > 1
                else None
            )
            if tb.P > 1:
                call.merge("dcarry_end")
            call.walk(
                "kcp_dhu_tape", tb.walk_end, dht=dcarry
            )  # document ends: under hop 1 (R7)
        _ring_wait(works)
        for f in receivers:
            _ring_scatter(
                call,
                tb,
                in_pool,
                ring.out_slot[f.slot],
                b2,
                tb.ds_out_tiles,
                ring.out_slot[f.slot],
            )


def _summaries(call: KcpCall) -> None:
    """The fragment summaries ``[B_f | M_f^T]`` into the c1 send layout: at ``P = 1`` K1 ``_offset`` and K3 write the
    crossing-out fragments' tiles directly; at ``P > 1`` both walk every segment into the staging planes and K4
    composes the crossing-out fragments' segments (forward family -> ``B_f``, adjoint family -> ``M_f^T``)."""
    tb = call.tb
    if tb.P == 1:
        call.walk(
            "kcp_fwdh_offset",
            tb.walk_offset,
            h0=None,
            out=call.indexed("c1_send", tb.out_index, tb.c1_b_tiles),
        )
        call.walk(
            "kcp_transfer_walk",
            tb.walk_offset,
            out=call.indexed("c1_send", tb.out_index, tb.c1_m_tiles),
            init_mode=0,
            use_u=0,
            write_transposed=0,
        )
        return
    call.walk(
        "kcp_fwdh_offset",
        tb.walk_offset,
        h0=None,
        out=call.indexed("stage_b", tb.seg_index, tb.stage_tiles),
    )
    call.walk(
        "kcp_transfer_walk",
        tb.walk_offset,
        out=call.indexed("stage_m", tb.seg_index, tb.stage_tiles),
        init_mode=0,
        use_u=0,
        write_transposed=0,
    )
    call.merge(
        "map"
    )  # B_f (from 0) and M_f (from I) of every crossing-out fragment, forward family
    call.transpose_tiles(
        "c1_send", tb.mf_tiles, "c1_send", tb.c1_m_tiles
    )  # M_f^T into the c1 pairs (bitwise)


def _fwd_tapes(
    call: KcpCall, *, out: torch.Tensor | None, carry: Indexed | None, wait_first: bool
) -> None:
    """The forward tape walk(s) after ``S_in`` is complete.  ``fused_walks`` (L-B): one launch from one pool -- the
    carry plane at ``P > 1`` (``h0_index_fused``: carries, the ``S_in`` copies of ``carry_in``, ``-1`` zero states),
    ``S_in`` itself at ``P = 1`` -- over every live segment (``wait_first``) or the crossing-in fragments' segments
    (the document-start walk ran under c1); otherwise the three launches of FINAL_DESIGN §3.5 (the document-start
    walk here only under ``wait_first``)."""
    tb = call.tb
    h0_in = call.indexed("S_in", tb.h0_index, tb.s_in_tiles)
    if tb.fused_walks:
        h0 = (
            call.indexed("carry", tb.h0_index_fused, tb.carry_tiles)
            if tb.P > 1
            else h0_in
        )
        call.walk(
            "kcp_fwdh_tape",
            tb.walk_live if wait_first else tb.walk_in_fused,
            h0=h0,
            out=out,
        )
        return
    if wait_first:
        call.walk(
            "kcp_fwdh_tape", tb.walk_start, h0=carry, out=out
        )  # document starts: after wait(c2) (L-A)
    call.walk("kcp_fwdh_tape", tb.walk_in_p0, h0=h0_in, out=out)
    if tb.P > 1:
        call.walk("kcp_fwdh_tape", tb.walk_in_p1, h0=carry, out=out)


def _adj_tapes(call: KcpCall, *, dcarry: Indexed | None, wait_first: bool) -> None:
    """The adjoint tape walk(s) after ``dS_out`` is complete (the mirror of :func:`_fwd_tapes`: the carry plane holds
    the adjoint carries and the ``dS_out`` copies of ``dcarry_cross`` at ``P > 1``; ``b2_recv`` itself at ``P = 1``)."""
    tb = call.tb
    dht_cross = call.indexed("b2_recv", tb.dht_index, tb.ds_out_tiles)
    if tb.fused_walks:
        dht = (
            call.indexed("carry", tb.dht_index_fused, tb.carry_tiles)
            if tb.P > 1
            else dht_cross
        )
        call.walk(
            "kcp_dhu_tape", tb.walk_live if wait_first else tb.walk_cross_fused, dht=dht
        )
        return
    if wait_first:
        call.walk(
            "kcp_dhu_tape", tb.walk_end, dht=dcarry
        )  # document ends: after wait(b2) (L-A)
    call.walk("kcp_dhu_tape", tb.walk_cross_last, dht=dht_cross)
    if tb.P > 1:
        call.walk("kcp_dhu_tape", tb.walk_cross_rest, dht=dcarry)


def _forward(call: KcpCall, *, save: bool, replay: bool) -> None:
    """FINAL_DESIGN §3.5.  ``save``: the saved set is kept (fresh ``S_in``; with ``save_transfer`` also the c1 receive
    tensor and ``M_seg``); ``replay``: the forward of a replay-mode backward (no ``fwd_o``, nothing kept).
    ``schedule="wait_first"`` (L-A, P2-R1; the shipped default since the round-1 adoption): ``c1 -> wait -> fwd merge -> c2 -> wait -> carry merges -> tape walk(s)``
    -- no merge or walk is issued while a collective is in flight (the R7 overlap of the document-start walk under c1
    is dropped); the merges read only ``c1_recv`` / the staging planes and the tapes read no collective buffer, so
    every published bit is the same."""
    tb, opts = call.tb, call.opts
    wait_first = call.schedule == "wait_first"
    keep_transfer = opts.save_transfer and save and not replay
    call.fresh("S_in")
    if keep_transfer:
        call.fresh("c1_recv")
        if tb.P > 1:
            call.fresh("stage_m")  # M_seg^T of the segmented fragments (§16 item 5b)
    call.stage("fwd_prep")
    call.stage("fwd_intra", write_ext=1 if (save and not replay) else 0, write_int=1)
    call.stage("wy")
    _summaries(call)
    carry = call.indexed("carry", tb.carry_index, tb.carry_tiles) if tb.P > 1 else None
    if (
        call.exchange == "ring"
    ):  # SPEC-2: per-document hops; S_in filled as c2 would (bitwise)
        if tb.P > 1 and not wait_first:
            call.merge("carry_start")
        _ring_forward(
            call
        )  # the document-start tape walk runs under hop 1 (overlap schedule only)
    else:
        w1 = call.a2a("c1", "c1_send", "c1_recv", *tb.split_elems("c1"))
        if not wait_first:
            if tb.P > 1:
                call.merge(
                    "carry_start"
                )  # reads the staging planes only (R7: no collective buffer)
            call.walk(
                "kcp_fwdh_tape", tb.walk_start, h0=carry, out=call.out
            )  # document starts: under c1 (R7)
        call.wait(w1)
        call.merge("fwd")  # the owner's chains: S_in slices into the c2 send layout
        w2 = call.a2a("c2", "c2_send", "S_in", *tb.split_elems("c2"))
        call.wait(w2)
    if tb.P > 1:
        if wait_first:
            call.merge("carry_start")
        call.merge(
            "carry_in"
        )  # with fused_walks also the S_in copies into the extra carry slots (P1-R1 write_states)
    _fwd_tapes(call, out=call.out, carry=carry, wait_first=wait_first)
    if not replay:
        call.stage("fwd_o")


def _backward(call: KcpCall, saved: KcpSaved | None) -> None:
    """FINAL_DESIGN §6.5.  ``saved=None`` is the replay mode (the forward sequence without ``fwd_o`` recomputes
    ``S_in``, the tapes and the owner's c1 receive tensor); otherwise the tapes come from K1 ``_tape`` over the saved
    ``S_in`` and the owner's adjoint reads ``M_f`` from b1 (``save_transfer=False``) or ``M_f^T`` from the saved c1
    receive tensor.  ``schedule="wait_first"`` (L-A; the shipped default since round 1): ``b1 -> wait -> adjoint merge -> b2 -> wait -> dcarry merges ->
    adjoint tape walk(s)`` (the R7 overlap of the document-last walk under b2 is dropped)."""
    tb = call.tb
    wait_first = call.schedule == "wait_first"
    replay = saved is None
    if replay:
        _forward(call, save=False, replay=True)
        m_source = "c1"
    else:
        call.bufs["S_in"] = saved.S_in
        m_source = "b1"
        if saved.c1_recv is not None:
            call.bufs["c1_recv"] = saved.c1_recv
            m_source = "c1"
        if saved.M_seg is not None:
            call.bufs["stage_m"] = saved.M_seg
        call.stage("bwd_prep", saved=saved)
        call.stage("wy")
        if (
            tb.P > 1
        ):  # the forward carries again: b_seg (and M_seg^T unless saved) -> K4 carries from 0 / S_in
            call.walk(
                "kcp_fwdh_offset",
                tb.walk_all,
                h0=None,
                out=call.indexed("stage_b", tb.seg_index, tb.stage_tiles),
            )
            if saved.M_seg is None:
                call.walk(
                    "kcp_transfer_walk",
                    tb.walk_all,
                    out=call.indexed("stage_m", tb.seg_index, tb.stage_tiles),
                    init_mode=0,
                    use_u=0,
                    write_transposed=0,
                )
            call.merge("carry_start")
            call.merge("carry_in")
        if (
            tb.fused_walks and tb.P > 1
        ):  # L-B: the whole tape replay from the carry plane in one launch
            call.walk(
                "kcp_fwdh_tape",
                tb.walk_live,
                h0=call.indexed("carry", tb.h0_index_fused, tb.carry_tiles),
                out=None,
            )
        else:
            p0 = [s for s in tb.walk_all if s % tb.P == 0]
            call.walk(
                "kcp_fwdh_tape",
                p0,
                h0=call.indexed("S_in", tb.h0_index, tb.s_in_tiles),
                out=None,
            )
            if tb.P > 1:
                call.walk(
                    "kcp_fwdh_tape",
                    [s for s in tb.walk_all if s % tb.P != 0],
                    h0=call.indexed("carry", tb.carry_index, tb.carry_tiles),
                    out=None,
                )
    call.stage("dav")
    ring = call.exchange == "ring"
    form = (
        "dm" if (m_source == "b1" or ring) else "d"
    )  # the ring's adjoint step needs M_f on the fragment's own rank
    if tb.P == 1:
        call.walk(
            "kcp_dhu_offset",
            tb.walk_in,
            dht=None,
            out=call.indexed("b1_send", tb.in_index, tb.b1_d_tiles[form]),
        )
        if (
            form == "dm"
        ):  # the maps travel with the offsets: M_f (untransposed) beside d_f
            call.walk(
                "kcp_transfer_walk",
                tb.walk_m_b1,
                out=call.indexed("b1_send", tb.m_b1_index, tb.b1_m_tiles),
                init_mode=0,
                use_u=0,
                write_transposed=1,
            )
    else:
        call.walk(
            "kcp_dhu_offset",
            tb.walk_all,
            dht=None,
            out=call.indexed("stage_b", tb.seg_index, tb.stage_tiles),
        )
        call.merge(
            f"tmap_d_{form}"
        )  # d_f of the crossing-in fragments (adjoint family)
        if form == "dm":
            call.merge(
                "tmap_m"
            )  # M_f of the fragments whose map travels (forward family)
    dcarry = (
        call.indexed("carry", tb.dcarry_index, tb.carry_tiles) if tb.P > 1 else None
    )
    if ring:  # SPEC-2: adjoint hops from the document end; b2_recv filled as b2 would (bitwise)
        _ring_backward(
            call
        )  # dcarry_end + the document-end tape walk run under hop 1 (overlap schedule only)
    else:
        wb1 = call.a2a("b1", "b1_send", "b1_recv", *tb.split_elems(f"b1_{form}"))
        call.wait(wb1)
        call.merge(
            "adj_b1" if m_source == "b1" else "adj_c1"
        )  # dS_out slices into the b2 send layout
        wb2 = call.a2a("b2", "b2_send", "b2_recv", *tb.split_elems("b2"))
        if not wait_first:
            if tb.P > 1:
                call.merge("dcarry_end")  # reads the staging planes only (R7)
            call.walk(
                "kcp_dhu_tape", tb.walk_end, dht=dcarry
            )  # document ends: under b2 (R7)
        call.wait(wb2)
    if tb.P > 1:
        if wait_first:
            call.merge("dcarry_end")
        call.merge(
            "dcarry_cross"
        )  # with fused_walks also the dS_out copies into the extra carry slots
    _adj_tapes(call, dcarry=dcarry, wait_first=wait_first)
    call.stage("dqkg")
    call.stage("glm_intra")
    call.stage("gate_epilogue")


# ----------------------------------------------------------------------------------------------------------------
# Entry points (GOAL §3.2 / FINAL_DESIGN §10.1)


def _prepare(
    q, k, v, g, beta, ctx, *, scale, save_mode, workspace, options, backward, dout=None
):
    backend = get_backend()
    opts = options if isinstance(options, KcpOptions) else _options(options)
    if save_mode not in SAVE_MODES:
        raise ValueError(f"save_mode must be one of {SAVE_MODES}, got {save_mode!r}")
    if save_mode != "light":
        raise NotImplementedError(
            "save_mode='tape' is not part of Phase 1 of the KCP route (light only)"
        )
    q, k, v, g, beta, T, H, batched = _operands(
        q, k, v, g, beta, ctx, dtypes=backend.dtypes
    )
    device = q.device
    if dout is not None:
        dout = _squeeze_batch(dout, "dout", 3)
        _check(dout, "dout", (T, H, D), backend.dtypes["dout"], device)
    if workspace is None:
        workspace = KcpWorkspace(
            state_dtype=backend.dtypes["state"],
            local=backend.local_stages,
            save_mode=save_mode,
        )
    elif not isinstance(workspace, KcpWorkspace):
        raise TypeError("workspace must be a KcpWorkspace")
    if workspace.state_dtype != backend.dtypes["state"]:
        raise ValueError(
            f"workspace state dtype {workspace.state_dtype} differs from the backend's {backend.dtypes['state']}"
        )
    tb = workspace.ensure(
        ctx, H, device=device, backward=backward, fused_walks=opts.fused_walks
    )
    ops = {"q": q, "k": k, "v": v, "g": g, "beta": beta}
    if dout is not None:
        ops["dout"] = dout
    call = KcpCall(
        ctx=ctx,
        H=H,
        ws=workspace,
        tb=tb,
        backend=backend,
        opts=opts,
        scale=SCALE if scale is None else float(scale),
        ops=ops,
        backward=backward,
    )
    backend.begin_call(call)
    return call, batched


def chunk_kda_kcp_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    ctx: Any,
    *,
    scale: float | None = None,
    save_for_backward: bool = True,
    save_mode: str = "light",
    boundary_mode: str = "saved",
    workspace: KcpWorkspace | None = None,
    **options: Any,
) -> tuple[torch.Tensor, KcpSaved | None]:
    """The KCP forward of this rank's fragments (GOAL §3.2): ``out`` like ``v`` (fresh) and the saved set
    (``None`` without ``save_for_backward`` or in ``boundary_mode="replay"``).  ``ctx`` is this rank's
    ``KcpContext``; the token operands are its ``T_r`` local tokens in fragment order; ``**options`` are the
    keywords of :class:`KcpOptions` (+ ``_kcp_test_knobs``)."""
    if boundary_mode not in BOUNDARY_MODES:
        raise ValueError(
            f"boundary_mode must be one of {BOUNDARY_MODES}, got {boundary_mode!r}"
        )
    call, batched = _prepare(
        q,
        k,
        v,
        g,
        beta,
        ctx,
        scale=scale,
        save_mode=save_mode,
        workspace=workspace,
        options=options,
        backward=False,
    )
    save = bool(save_for_backward) and boundary_mode == "saved"
    call.out = _alloc((call.T_r, call.H, D), call.backend.dtypes["out"], call.device)
    _forward(call, save=save, replay=False)
    saved = None
    if save:
        tb, opts = call.tb, call.opts
        saved = KcpSaved(
            version=KCP_SAVED_VERSION,
            S_in=call.bufs["S_in"],
            c1_recv=call.bufs.get("c1_recv") if opts.save_transfer else None,
            M_seg=call.bufs.get("stage_m")
            if (opts.save_transfer and tb.P > 1)
            else None,
            Aqk=call.scratch.get("Aqk"),
            Akk=call.scratch.get("Akk"),
            T_r=call.T_r,
            H=call.H,
            n_frag=tb.n_frag,
            P=tb.P,
            Hg=tb.Hg,
            scale=call.scale,
            save_mode=save_mode,
            save_transfer=opts.save_transfer,
            offset_class=opts.offset_class,
            exchange=call.exchange,
            precision_mode=call.precision_mode,
            ctx=ctx,
            transfer_walk=call.transfer_walk,
        )
    _record("fwd", call)
    call.backend.end_call(call)
    return _ranked(call.out, batched), saved


def _check_saved(saved: KcpSaved, call: KcpCall) -> None:
    tb, opts = call.tb, call.opts
    if saved.version != KCP_SAVED_VERSION:
        raise ValueError(f"saved: version {saved.version} is not {KCP_SAVED_VERSION}")
    if saved.ctx is not call.ctx:
        raise ValueError("saved: the saved set belongs to another KcpContext")
    if (saved.T_r, saved.H, saved.n_frag, saved.P) != (
        call.T_r,
        call.H,
        tb.n_frag,
        tb.P,
    ):
        raise ValueError(
            "saved: the saved set describes another problem (T_r, H, n_frag, P)"
        )
    if saved.scale != call.scale:
        raise ValueError(
            f"saved: scale {saved.scale} differs from the call's {call.scale}"
        )
    if (
        saved.save_transfer != opts.save_transfer
        or saved.exchange != call.exchange
        or saved.precision_mode != call.precision_mode
        or saved.transfer_walk != call.transfer_walk
    ):
        raise ValueError(
            "saved: save_transfer / exchange / precision_mode / transfer_walk differ from the forward's"
        )
    _check(saved.S_in, "saved.S_in", (tb.sizes["S_in"],), call.state_dtype, call.device)
    if opts.save_transfer:
        if saved.c1_recv is None:
            raise ValueError(
                "saved: save_transfer needs the forward's c1 receive tensor"
            )
        _check(
            saved.c1_recv,
            "saved.c1_recv",
            (tb.sizes["c1_recv"],),
            call.state_dtype,
            call.device,
        )
        if tb.P > 1:
            if saved.M_seg is None:
                raise ValueError("saved: save_transfer at P > 1 needs M_seg")
            _check(
                saved.M_seg,
                "saved.M_seg",
                (tb.sizes["stage_m"],),
                call.state_dtype,
                call.device,
            )


def chunk_kda_kcp_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    ctx: Any,
    dout: torch.Tensor,
    *,
    saved: KcpSaved | None = None,
    scale: float | None = None,
    save_mode: str = "light",
    workspace: KcpWorkspace | None = None,
    **options: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The KCP backward (GOAL §3.2): ``(dq, dk, dv, dbeta, dg)``, fresh.  ``saved=None`` replays the forward."""
    call, batched = _prepare(
        q,
        k,
        v,
        g,
        beta,
        ctx,
        scale=scale,
        save_mode=save_mode,
        workspace=workspace,
        options=options,
        backward=True,
        dout=dout,
    )
    if saved is not None:
        _check_saved(saved, call)
    dt = call.backend.dtypes
    T, H = call.T_r, call.H
    call.grads = {
        "dq": _alloc((T, H, D), dt["dq"], call.device),
        "dk": _alloc((T, H, D), dt["dk"], call.device),
        "dv": _alloc((T, H, D), dt["dv"], call.device),
        "dg": _alloc((T, H, D), dt["dg"], call.device),
        "dbeta": _alloc((T, H), dt["dbeta"], call.device),
    }
    _backward(call, saved)
    _record("bwd", call)
    call.backend.end_call(call)
    return tuple(
        _ranked(call.grads[n], batched) for n in ("dq", "dk", "dv", "dbeta", "dg")
    )


# ----------------------------------------------------------------------------------------------------------------
# autograd


class ChunkKDAKcpFunction(torch.autograd.Function):
    """:func:`chunk_kda_kcp` as an autograd node: the forward keeps the inputs and the saved set (saved mode) or
    the inputs only (replay mode); the backward runs :func:`chunk_kda_kcp_bwd`.  Under
    ``torch.utils.checkpoint(use_reentrant=False)`` the forward is re-run in the backward: the same program order
    on every rank, so the collectives match (R7)."""

    @staticmethod
    def forward(
        fctx, q, k, v, g, beta, ctx, scale, save_mode, boundary_mode, workspace, options
    ):
        out, saved = chunk_kda_kcp_fwd(
            q,
            k,
            v,
            g,
            beta,
            ctx,
            scale=scale,
            save_for_backward=True,
            save_mode=save_mode,
            boundary_mode=boundary_mode,
            workspace=workspace,
            **options,
        )
        tensors = saved.tensors() if saved is not None else ()
        fctx.save_for_backward(q, k, v, g, beta, *tensors)
        fctx.kcp = (
            ctx,
            scale,
            save_mode,
            workspace,
            dict(options),
            None if saved is None else (saved.flags(), saved.meta()),
        )
        return out

    @staticmethod
    def backward(fctx, dout):
        q, k, v, g, beta, *tensors = fctx.saved_tensors
        ctx, scale, save_mode, workspace, options, meta = fctx.kcp
        saved = (
            None
            if meta is None
            else KcpSaved.from_tensors(tensors, meta[0], meta[1], ctx)
        )
        if not dout.is_contiguous():
            dout = dout.contiguous()
        dq, dk, dv, dbeta, dg = chunk_kda_kcp_bwd(
            q,
            k,
            v,
            g,
            beta,
            ctx,
            dout,
            saved=saved,
            scale=scale,
            save_mode=save_mode,
            workspace=workspace,
            **options,
        )
        return dq, dk, dv, dg, dbeta, None, None, None, None, None, None


def chunk_kda_kcp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    ctx: Any,
    scale: float | None = None,
    save_mode: str = "light",
    boundary_mode: str = "saved",
    workspace: KcpWorkspace | None = None,
    **options: Any,
) -> torch.Tensor:
    """The differentiable KCP forward (GOAL §3.2 signature): ``out`` like ``v``; gradients for ``q, k, v, g, beta``."""
    _options(options)  # validate on the caller's frame
    if boundary_mode not in BOUNDARY_MODES:
        raise ValueError(
            f"boundary_mode must be one of {BOUNDARY_MODES}, got {boundary_mode!r}"
        )
    return ChunkKDAKcpFunction.apply(
        q, k, v, g, beta, ctx, scale, save_mode, boundary_mode, workspace, dict(options)
    )


# the plan module's byte queries, re-exported for the GOAL §3.2 literal forms
kcp_saved_bytes = cake_kcp_plan.kcp_saved_bytes
kcp_comm_bytes = cake_kcp_plan.kcp_comm_bytes
kcp_workspace_bytes = cake_kcp_plan.kcp_workspace_bytes

__all__ = [
    "ABI_PARAMS",
    "TRANSFER_WALKS",
    "SPEC3_ENV",
    "KcpPreconditionError",
    "BOUNDARY_MODES",
    "COLLECTIVES",
    "CONTRACT_DTYPES",
    "EXCHANGES",
    "KCP_PROGRAMS",
    "KCP_SAVED_VERSION",
    "LAST_CALL",
    "OFFSET_CLASSES",
    "RING_MAX_HOPS",
    "SUMMARY_TILE",
    "TILE",
    "Backend",
    "Chains",
    "ChunkKDAKcpFunction",
    "Indexed",
    "KcpCall",
    "KcpOptions",
    "KcpSaved",
    "KcpWorkspace",
    "KernelBackend",
    "chunk_kda_kcp",
    "chunk_kda_kcp_bwd",
    "chunk_kda_kcp_fwd",
    "get_backend",
    "kcp_comm_bytes",
    "kcp_saved_bytes",
    "kcp_workspace_bytes",
    "set_backend",
    "tile_index",
]
