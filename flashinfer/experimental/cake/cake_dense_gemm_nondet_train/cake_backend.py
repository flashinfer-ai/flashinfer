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

Cake backend of ``flashinfer.cake_dense_gemm_nondet_train``, non-deterministic: the GLM-5.2 dense projection GEMMs
for training (K1) and the FP32 router-gate GEMM (K2) on SM100 / SM103 / SM107.

The deterministic implementation is the package ``cake_dense_gemm_det_train`` (flashinfer-ai/flashinfer#5677 and
#7043; public module ``flashinfer.cake_dense_gemm_det_train``): its stream-K and split-K reductions add the partials
in a fixed order, so every result is bitwise reproducible run to run.  This package keeps that package's operations,
view contract, dtypes, accuracy budgets and host paths (prepared and convenience) and adds programs whose split-K
reductions add the partials in the order they finish.  The products, the FP32 accumulation, the rounding points (an
output is rounded once, from the complete FP32 sum), the masks, the layouts and the dynamic shapes are the
deterministic package's, and no reduction flushes a subnormal.  A result of this package carries no run-to-run
bitwise guarantee on any row, also on a row that runs a fixed-order program today.

Mode and dispatch rules
-----------------------
The module selects the mode: no entry point takes a mode argument.  Every entry point runs the routes of
:data:`ROUTES`:

* the K1 weight gradient (both operands MN-major, ``K = T`` ragged), launched as the transposed-store GEMM
  ``out[l, n, m]``: each of ``L x m_tiles x n_tiles`` tiles is cut into ``num_splits`` K ranges; every split stores
  its FP32 partial into its own workspace slot and takes a ticket, and the split that draws the last ticket adds the
  other ``num_splits - 1`` partials with IEEE round-to-nearest adds, rounds once and stores (:func:`k1_plan` reads the
  tile width and the split count from the measured verdicts of :data:`K1_SPLIT_RULES`; every other row runs the
  deterministic program).  Two splits add ``own + other``, one add of two values, so a two-split row repeats bit for
  bit; three or more add in the order the splits finish (no row of the shipped tables: only a caller's explicit
  ``splits``).  The rows of :data:`K1_ORDERED_RANGES_BY_SM` (212-SM sm_107a parts, and one 148-SM sm_100a BF16 row)
  add four splits in the fixed mirror order or run two or four chains of one split, the FP32 rows of
  :data:`K1_PARITY_RANGES_BY_SM` (sm_107a parts of 208, 212, 216 and more SMs)
  run two, four or eight chains of one split by the parity of their number of K blocks, the one-split rows store
  whole tiles, and the rows of
  :data:`K1_WGRAD_DET_RULES_BY_SM` run another fixed-order program of the deterministic package;
* the K1 weight gradient of the FP32 rows of :data:`K1_DSM_RANGES_BY_SM` (thin ``d_out`` on 148-SM and 152-SM
  ``sm_100a``, 152-SM ``sm_103a`` and 212-SM ``sm_107a`` parts, in that table's token ranges; the odd numbers of K
  blocks of :data:`K1_DSM_ODD_BLOCK_RANGES_BY_SM` take that table's 64-wide four-CTA tile), before every other K1
  table: a cluster program (:func:`k1_dsm_plan`) whose ``S`` CTAs each take one of a tile's ``S`` K ranges and meet in the cluster's
  distributed shared memory -- each CTA sends the column quads another rank owns to that rank, adds the ``S``
  partials of its own quads in cluster-rank order and stores them once (no workspace, no ticket; the same bits on
  every run);
* the K2 weight gradient: the deterministic program at the split count of its row's ``"det_s<n>"`` token window in
  the router tables (:func:`router_det_splits`; the partials added in a fixed order), else at the deterministic
  package's split count of the row.  A stream-K partition of the (tile, K block) steps into ``units`` spans
  (:func:`router_partition`) runs only for a caller's explicit split count (also of the forward), in the ``"waves"``
  windows on a part without a router table of its SM count where it gives every element at most two partials, and
  in the ``"ksum_S<n>"`` windows of the router tables, whose K-order program adds a tile's partials in K order
  through a workspace (:func:`router_ksum_template`; the same bits on every run); in the other partitions a segment that covers its tile's whole K
  range stores the deterministic program's one-split bits, any other adds its partial into ``out`` with a TMA
  reduce-add (``out`` is zeroed on the launch's stream first).  The head/tail partitions (long head spans, short tail
  spans) stay registered for an explicit table entry; no row of the shipped tables runs them.  A forward partition gives every element at most two partials,
  ``0 + a + b == 0 + b + a``, so a forward recomputed under activation checkpointing gives the same logits and the same
  top-k;
* the K2 weight and input gradients in the SIMT token windows of their rows in the router tables
  (:func:`router_rules`, :func:`router_simt`): an exact-FP32 program on the CUDA cores over the FP32 operands read in
  place (compensated in the 148-SM tables' one-token window at hidden 4096), every element of ``out`` stored once
  (other bits than the deterministic program's BF16x3 emulation, within the deterministic package's router FP64
  budget); the input gradient's every other row and token count runs the deterministic program (one split);
* the K2 forward in the SIMT token windows of its rows (:func:`router_forward_simt`): an exact-FP32 program on the CUDA
  cores whose cluster of CTAs splits K and adds the ranks' partials in rank order, every element of ``out`` stored
  once (other bits than the deterministic program's, within the deterministic package's router FP64 budget); every
  other forward row and token count runs the deterministic program.

Every other operation -- the K1 forward and input gradient, the BF16x3 split -- runs the deterministic programs: a
recomputed forward must reproduce its result, and where the deterministic plan of a K1 forward or input gradient
splits K across CTAs (its automatic two-way split), it adds the partials in a fixed order.  The K1 forward rows of
:data:`K1_FORWARD_RULES_BY_SM` (exact token counts on 148-SM ``sm_100a`` and 152-SM ``sm_100a`` / ``sm_103a`` parts)
run another plan of those programs (the single-CTA program, every tile's whole K in one CTA, or another named
program), the input-gradient rows of :data:`K1_DGRAD_RULES_BY_SM` another named program of that package and those of
:data:`K1_DGRAD_PROMO_RANGES_BY_SM` its plan with another TMA L2 promotion (the deterministic plan's bits).  The K1 forward rows of :data:`K1_FORWARD_PROGRAM_RULES_BY_SM` run this package's one-wave
forward program and the input-gradient rows of :data:`K1_DGRAD_PROGRAM_RULES_BY_SM` this package's input-gradient
program: both add each output element's whole K in one ascending FP32 chain, with the deterministic plan's bits.  The
input-gradient rows of :data:`K1_DGRAD_NVJET_RULES_BY_SM` launch this package's nvjet program on the deterministic
package's plan of one of its programs (the same kernel parameters and bits).

Under ``torch.use_deterministic_algorithms(True)`` a launch whose result can vary run to run -- the K1 split-K program
of CTA pairs at three or more splits in the arrival order, a K2 stream-K partition that gives an element more than two
partials; no default plan of the shipped tables, only a caller's explicit split count -- raises
PyTorch's alert right before it launches (:func:`_alert_order_varying`): ``RuntimeError`` (the program is not launched
and ``out`` is untouched), or with ``warn_only=True`` a ``UserWarning``, after which the launch runs.  The fixed-order
launches (the one- and two-split K1 rows, the mirror order, the K1 cluster program, the K2 SIMT and K-order programs,
a K2 partition of at most two partials per element, every deterministic program) do not alert.  The alert is checked on every launch, never at plan or
prepare time; a launch captured into a CUDA graph is checked at capture, not at replay.  The PyTorch setting never
changes which program a call runs, and no call goes to the deterministic package because of it.

The two packages
----------------
This package registers only the programs of its own schedule (``cake_jit``: programs
``cake_dense_projection_gemm_<hex>``, templates ``dense_proj_gemm_*`` / ``router_fp32_gemm_*``); every
deterministic program, and every planner rule of the deterministic package, is that package's.  A call of this
package reuses them on the rows that run the deterministic programs (above): it launches the
deterministic plan and program through a binding of this package (the same bits, fewer host operations per call),
or, where that binding cannot rebuild the call's launch views from its tensors (two operands that alias) and for a
batched forward that the deterministic package runs on its swapped small-token program, it calls the deterministic
package's entry point of the same operation (the same program, the same bits).  On such a row the ``*_plan`` /
``*_prepare`` entry points return the deterministic package's plan or prepared launch
(``CakeDenseGemmDetProjectionPlan``, ``CakeDenseGemmDetProjectionLaunch``, ``CakeDenseGemmDetRouterPlan``,
``CakeDenseGemmDetRouterLaunch``).  A row that runs a program of this package
never goes to the deterministic package.  The deterministic package is imported at run time as this package's
sibling (the top-level package ``cake_dense_gemm_det_train`` when this package is vendored as a top-level package),
never through an absolute ``flashinfer`` import; a copy under another name is bound with
:func:`use_deterministic_backend`.  The deterministic package never imports this one, and this module re-exports no
name of it.  :func:`cake_dense_gemm_nondet_train_convenience_cache_clear` clears the convenience caches and workspace arenas of both packages; a
convenience-path call that this package launches from its arenas, captured into a CUDA graph, stays valid until then
(the arenas keep every buffer a launch was bound to, through growths and cache evictions); a call it hands to the
deterministic package's entry point follows that package's rule (:func:`cake_dense_gemm_nondet_train_convenience_cache_clear`).

The tensors passed between the forward and the gradients carry no mode: recompute a checkpointed forward with the
package that made the saved one (same package only).
"""

from __future__ import annotations

import math
import re
import sys
import threading
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import cached_property
from types import FrameType, ModuleType
from typing import Any, Callable, Optional, TypeVar, Union

import torch
import tvm_ffi

from . import cake_jit

# This package's own registry and loader, re-exported (its programs only).
from .cake_jit import KERNELS as KERNELS
from .cake_jit import MODULES as MODULES
from .cake_jit import TRACKING_ISSUE as TRACKING_ISSUE
from .cake_jit import arg_plan_of as arg_plan_of
from .cake_jit import (
    load_cake_dense_projection_gemm_module as load_cake_dense_projection_gemm_module,
)
from .cake_jit import select_module as select_module

# ---------------------------------------------------------------------------
# The deterministic package
# ---------------------------------------------------------------------------

#: The deterministic package's entry points this package calls on the rows that run the deterministic programs
#: (see "The two packages" above); :func:`use_deterministic_backend` checks that a bound module has them.
_DET_ENTRIES = (
    "cake_dense_gemm_det_train_projection",
    "cake_dense_gemm_det_train_projection_plan",
    "cake_dense_gemm_det_train_router_plan",
    "cake_dense_gemm_det_train_projection_prepare",
    "cake_dense_gemm_det_train_projection_wgrad_prepare",
    "cake_dense_gemm_det_train_router_prepare",
    "cake_dense_gemm_det_train_projection_dgrad",
    "cake_dense_gemm_det_train_projection_forward",
    "cake_dense_gemm_det_train_projection_wgrad",
    "cake_dense_gemm_det_train_router",
    "split_fp32_to_bf16x3_det_",
    "cake_dense_gemm_det_train_convenience_cache_clear",
)


def use_deterministic_backend(module: ModuleType) -> None:
    """Bind the deterministic package's ``cake_backend`` module explicitly.  The deterministic package is this
    package's sibling and is resolved on first use, in a FlashInfer tree and in a vendored tree that ships
    ``cake_dense_gemm_det_train`` next to this package (both top-level packages, or both in one parent package); a
    vendored tree that holds it under another name binds its ``cake_backend`` here before the first call that
    needs it.  Binding drops this package's convenience caches and arenas (:func:`_drop_convenience_entries`): their
    entries hold the plans and programs of the package bound before.  Like
    :func:`cake_dense_gemm_nondet_train_convenience_cache_clear`, it invalidates the CUDA graphs that captured a
    convenience-path call of this package."""
    missing = [name for name in _DET_ENTRIES if not hasattr(module, name)]
    if missing:
        raise TypeError(
            f"{getattr(module, '__name__', module)!r} is not the deterministic dense projection GEMM backend: "
            f"it lacks {missing}"
        )
    cake_jit.bind_deterministic_package(module.__name__.rpartition(".")[0])
    _drop_convenience_entries()


# The module :func:`deterministic_backend` last resolved: ``(bound package, module name, module)``.
_BOUND_BACKEND: tuple[Optional[str], str, Optional[ModuleType]] = (None, "", None)


def deterministic_backend() -> ModuleType:
    """The deterministic package's ``cake_backend`` module (see :func:`use_deterministic_backend`).  Every call of
    the convenience and prepared paths reads it: the module resolved last is returned while the bound package is the
    same and ``sys.modules`` still holds it under its name (the module an import would return), so a call costs no
    import-system lookup."""
    global _BOUND_BACKEND
    package = cake_jit._det_package
    bound, name, module = _BOUND_BACKEND
    if module is not None and bound == package and sys.modules.get(name) is module:
        return module
    module = cake_jit.deterministic_module("cake_backend")
    _BOUND_BACKEND = (package, module.__name__, module)
    return module


#: The public modules of the two packages: the texts of the alert and of the errors name them.
_MODULE = "flashinfer.cake_dense_gemm_nondet_train"
_DET_MODULE = "flashinfer.cake_dense_gemm_det_train"


def _warning_stacklevel() -> int:
    """The ``stacklevel`` of a warning raised in this module that points at the first frame outside it (the caller's
    line): the depth of this module's frames between the warning and the caller."""
    level = 1
    frame: Optional[FrameType] = sys._getframe(1)
    while frame is not None and frame.f_globals.get("__name__") == __name__:
        level += 1
        frame = frame.f_back
    return level


def _alert_order_varying(launch: str) -> None:
    """PyTorch's alert for an operation without a deterministic implementation, raised right before this package
    launches ``launch``, a program whose result can vary run to run (the K1 split-K program of CTA pairs at three or
    more splits, a K2 stream-K partition that gives an element more than two partials), while PyTorch's deterministic algorithms
    are enabled: a ``RuntimeError``, or a ``UserWarning`` under ``torch.use_deterministic_algorithms(True,
    warn_only=True)``, after which the launch runs; nothing otherwise.  The alert of PyTorch's own operators (PyTorch
    exposes no public Python entry for it).  Fixed-order launches do not call it, and it never changes the program
    that runs.  A launch captured into a CUDA graph calls it at capture; a replay does not."""
    if not torch.are_deterministic_algorithms_enabled():
        return
    if torch.is_deterministic_algorithms_warn_only_enabled():
        warnings.warn(
            f"{_MODULE}: {launch} does not have a deterministic implementation, but you set "
            "'torch.use_deterministic_algorithms(True, warn_only=True)'; "
            f"{_DET_MODULE} is the deterministic module",
            UserWarning,
            stacklevel=_warning_stacklevel(),
        )
        return
    raise RuntimeError(
        f"{_MODULE}: {launch} does not have a deterministic implementation, but you set "
        f"'torch.use_deterministic_algorithms(True)'; {_DET_MODULE} is the deterministic module, or use "
        "the 'warn_only=True' option if that is acceptable for your application"
    )


def _k1_launch_text(plan: Any) -> str:
    """The launch an order-varying K1 plan (:class:`CakeDenseGemmNondetProjectionPlan`) names in :func:`_alert_order_varying`."""
    return (
        f"the K1 split-K program {plan.template} at {plan.splits} splits (the partials added in the order the "
        "splits finish)"
    )


def _k2_launch_text(plan: Any) -> str:
    """The launch an order-varying K2 plan (:class:`CakeDenseGemmNondetRouterPlan`) names in :func:`_alert_order_varying`."""
    return (
        f"the router stream-K program {plan.template} with {plan.units} units over {plan.tiles} tiles (more than "
        "two partials per element, added in the order they finish)"
    )


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

#: The program family each operation runs: ``"atomic"`` -- this package's planner and
#: programs (the planner keeps every row without a measured verdict on the deterministic program, see
#: :func:`k1_plan` / :func:`router_partition`) -- or ``"det"`` -- the deterministic package's programs (fixed order).
#: The PyTorch deterministic-algorithms setting never reads this table.  K1 operations are keyed by the operand layouts of the launched GEMM (``kk`` forward, ``kn`` input gradient, ``nn``
#: weight gradient; ``nk`` has no training use), K2 by the layout class of ``router_layout_class``.
ROUTES: dict[str, str] = {
    "k1_kk": "det",  # forward: feeds the recompute of checkpointing; whole tiles or a fixed-order two-way K split (the
    # rows of K1_FORWARD_RULES_BY_SM: another plan of the deterministic programs, whole K per tile)
    "k1_kn": "det",  # input gradient: whole tiles or a fixed-order two-way K split (the rows of
    # K1_DGRAD_RULES_BY_SM: another program of the deterministic package, of K1_DGRAD_PROMO_RANGES_BY_SM: its plan
    # with another TMA L2 promotion; the same bits)
    "k1_nk": "det",
    "k1_nn": "atomic",  # weight gradient: split-K over the ragged token count; the rows of the tables repeat bit for
    # bit (two splits, the cluster program, one split, a fixed split order) but the four-split arrival-order rows of
    # K1_ORDERED_RANGES_BY_SM and K1_DSM_ODD_BLOCK_RANGES_BY_SM, the others run the deterministic program
    "k2_kk": "atomic",  # router forward: the cluster SIMT program in its token windows; an explicit split count <= 2
    "k2_kn": "atomic",  # router input gradient: the SIMT program in its token windows, else one split
    "k2_nn_t": "atomic",  # router weight gradient: the SIMT program in its token windows, else the deterministic
    # program at the split count of the router tables (fixed order); stream-K only for an explicit split count
    "split": "det",  # the exact BF16x3 split of the router's second operand
}


def _k1_layout(A: torch.Tensor, B: torch.Tensor) -> str:
    """``k1_<a><b>``: ``k`` for a K-major operand (unit stride on the contraction axis), ``n`` for an MN-major one,
    classified as the deterministic planner's ``operand_view`` does (the contraction axis first)."""
    a = "k" if A.stride(-1) == 1 else "n"
    b = "k" if B.stride(-2) == 1 else "n"
    return f"k1_{a}{b}"


def _k2_class(a_mn: bool, b_mn: bool) -> str:
    """``k2_<class>`` of the router operand layouts (``kk``, ``kn`` or ``nn_t``; the deterministic planner refuses
    ``nk``)."""
    return (
        "k2_"
        + {
            (False, False): "kk",
            (False, True): "kn",
            (True, False): "nk",
            (True, True): "nn_t",
        }[(bool(a_mn), bool(b_mn))]
    )


def _k2_layout(A: torch.Tensor, B: torch.Tensor) -> str:
    """:func:`_k2_class` of the router operands, classified as the deterministic planner does (``A`` K-major when
    its ``k`` stride is 1, ``B`` K-major when its ``k`` stride is 1)."""
    return _k2_class(A.stride(1) != 1, B.stride(0) != 1)


def _route(key: str) -> str:
    return ROUTES.get(key, "det")


def _ceil_div(a: int, b: int) -> int:
    return -(-int(a) // int(b))


# The launch geometry of this package's programs (the generated kernels' constants).
ROUTER_BLOCK_M = 128  # rows per CTA of a router tile (256 per CTA pair)
ROUTER_BLOCK_N = 128  # columns per router tile
ROUTER_BLOCK_K = 32  # fp32 elements per router K block
K1_PAIR_ROWS = 256  # rows of a K1 tile (a CTA pair, 2 x 128)
K1_DSM_ROWS = 128  # rows of a K1 cluster-program tile (one CTA per K range)
K1_BLOCK_K = 64  # K elements per K1 K block
K1_COUNTER_WORDS = 512  # ticket words per K1 tile: 16 epilogue warp slices x 32 lanes
K1_BLOCK_N_CHOICES = (64, 128, 192, 256)
#: The tile widths of :data:`K1_BLOCK_N_CHOICES` this package runs as its row-major-store program, with the operands
#: exchanged (:attr:`CakeDenseGemmNondetProjectionPlan.out_t` false): a CTA pair's tile is 256 rows of the transposed-store launch's ``N``
#: by ``block_n`` columns of its ``M``.  The other widths run as transposed-store programs.
K1_ROW_MAJOR_BLOCK_N: tuple[int, ...] = (192,)
#: Accumulation chains of a one-split K1 program: with ``c`` chains each tile's K blocks are cut into ``c`` contiguous
#: ranges, each summed in a TMEM accumulator of its own from an ``init`` MMA, and the epilogue adds them once, in order
#: (no workspace, the same bits on every run).  This package ships the one-chain programs and, for the FP32 rows of
#: :data:`K1_CHAIN_RANGES_BY_SM`, three-chain programs (templates ``..._c3``); the two- and four-chain programs of
#: :data:`K1_ORDERED_RANGES_BY_SM` (templates ``..._c2``, ``..._c4``) are the planner's only, not a ``chains`` choice.
K1_CHAINS_CHOICES: tuple[int, ...] = (1, 3)

# ---------------------------------------------------------------------------
# K1 planner: the measured verdicts
# ---------------------------------------------------------------------------
#
# A K1 launch of this package is the transposed-store GEMM ``out[l, n, m] = sum_k A[l, k, m] B[l, k, n]``: ``M`` is
# the inner (unit-stride) extent of the stored output and ``N`` its row count.  A weight gradient
# ``dW[d_out, d_in] = G[T, d_out].T @ X[T, d_in]`` has ``A = X``, ``B = G``, ``M = d_in``, ``N = d_out``, ``K = T``.

#: Per-row verdicts of the K1 weight gradient, ``(arch, L, M, N, out_dtype) -> (block_n, num_splits)`` (this package's
#: program with that tile width and split count; a width of :data:`K1_ROW_MAJOR_BLOCK_N` runs the row-major program with
#: the operands exchanged) or ``-> None`` (the deterministic program); ``M`` / ``N`` as above, ``out_dtype`` ``"bf16"``
#: or ``"f32"``.  A row carries a program only where that program was measured faster than the deterministic package's
#: shipped plan of the row in every measurement run: a one-sided paired sign test (p < 0.05) with a median time ratio
#: below 0.99, in eager launches and in CUDA graph replays, at token counts 16172 to 16384, and no more than 3 % slower
#: than that plan next to a kernel holding 8, 16 or 32 SMs on another stream.  A ``None`` entry is a measured row that
#: missed; a row without an entry was not measured.  Both run the deterministic program.  A verdict does not hold at the
#: token counts of :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS` and, where the token count spans an odd number of K blocks, of
#: :data:`K1_SPLIT_ODD_BLOCK_EXCLUDED_TOKENS`.  At every token count a verdict holds, its output met the deterministic
#: package's error budgets against an FP64 reference of the exact operands (BF16 output ``atol = rtol = 1e-2``
#: elementwise with no element outside; FP32 output a relative Frobenius error ``<= 5e-5`` and a max error
#: ``<= 1e-3 * max|ref|``) on every measured seed, the bound the deterministic package states for its own programs; the
#: split program's FP64 error relative to the deterministic program's is reported data, not a rule.  Where both programs
#: miss that budget for some seed (BF16 output at the largest token counts, FP32 output from about 98305 tokens), the
#: budget does not choose between them, and the tables keep the routing they give there.
# fmt: off
K1_SPLIT_RULES:dict[tuple[str, int, int, int, str], Optional[tuple[int, int]]] = {
    # sm_100a: two splits of 64-wide tiles (one wave of the CTA pairs; two partials, bitwise reproducible).  d_out 128
    # at d_in 4096 (128 CTAs) loses to the deterministic plan next to a kernel holding 32 SMs
    ("sm_100a", 1, 4096, 32, "bf16"): (64, 2), ("sm_100a", 1, 4096, 32, "f32"): (64, 2),
    ("sm_100a", 1, 4096, 48, "bf16"): (64, 2), ("sm_100a", 1, 4096, 48, "f32"): (64, 2),
    ("sm_100a", 1, 4096, 64, "bf16"): (64, 2), ("sm_100a", 1, 4096, 64, "f32"): (64, 2),
    ("sm_100a", 1, 4096, 128, "bf16"): None, ("sm_100a", 1, 4096, 128, "f32"): None,
    ("sm_100a", 1, 5120, 32, "bf16"): (64, 2), ("sm_100a", 1, 5120, 32, "f32"): (64, 2),
    ("sm_100a", 1, 5120, 48, "bf16"): (64, 2), ("sm_100a", 1, 5120, 48, "f32"): (64, 2),
    ("sm_100a", 1, 5120, 64, "bf16"): (64, 2), ("sm_100a", 1, 5120, 64, "f32"): (64, 2),
    ("sm_100a", 1, 6144, 32, "bf16"): (64, 2), ("sm_100a", 1, 6144, 32, "f32"): (64, 2),
    # sm_103a: two splits of 64-wide tiles, of 128-wide tiles for d_out = 128 at d_in >= 5120 (one wave); d_out 48 at
    # d_in 4096 from 152 SMs only (K1_RULE_MIN_SM_COUNT); d_out 64 at d_in 4096 is not faster than the deterministic
    # plan in every run; d_out 128 at d_in 4096 as on sm_100a
    ("sm_103a", 1, 4096, 32, "bf16"): (64, 2), ("sm_103a", 1, 4096, 32, "f32"): (64, 2),
    ("sm_103a", 1, 4096, 48, "bf16"): (64, 2), ("sm_103a", 1, 4096, 48, "f32"): (64, 2),
    ("sm_103a", 1, 4096, 64, "bf16"): None, ("sm_103a", 1, 4096, 64, "f32"): None,
    ("sm_103a", 1, 4096, 128, "bf16"): None, ("sm_103a", 1, 4096, 128, "f32"): None,
    ("sm_103a", 1, 5120, 32, "bf16"): (64, 2), ("sm_103a", 1, 5120, 32, "f32"): (64, 2),
    ("sm_103a", 1, 5120, 48, "bf16"): (64, 2), ("sm_103a", 1, 5120, 48, "f32"): (64, 2),
    ("sm_103a", 1, 5120, 64, "bf16"): (64, 2), ("sm_103a", 1, 5120, 64, "f32"): (64, 2),
    ("sm_103a", 1, 5120, 128, "bf16"): (128, 2), ("sm_103a", 1, 5120, 128, "f32"): (128, 2),
    ("sm_103a", 1, 6144, 32, "bf16"): (64, 2), ("sm_103a", 1, 6144, 32, "f32"): (64, 2),
    ("sm_103a", 1, 6144, 128, "bf16"): (128, 2), ("sm_103a", 1, 6144, 128, "f32"): (128, 2),
    # sm_107a (216 SMs): one or two splits (the rows of three or four splits, whose partials were added in the order
    # they finish, run two splits where the two-split program was not slower than the deterministic plan on 212-SM
    # parts, else the deterministic program); the one-split rows store whole tiles (deterministic) and win where the
    # deterministic package's sm_107a program of the row is slower (the FP32 one-split rows up to 40960 tokens on
    # 216 SMs, 5120 -> 1024 to 32768, and up to 960 on more SMs: K1_SPLIT_RULE_EXCLUDED_TOKENS).  d_out 64 BF16 at
    # d_in 4096 is not faster than the deterministic plan in every run; d_in 4096 to d_out 512 BF16 (192 CTAs) and to
    # d_out 4096 FP32 (216 CTAs) need more SMs than a kernel holding 32 SMs leaves free and lose next to it
    ("sm_107a", 1, 4096, 32, "bf16"): None, ("sm_107a", 1, 4096, 32, "f32"): (64, 2),
    ("sm_107a", 1, 4096, 48, "f32"): None,
    ("sm_107a", 1, 4096, 64, "bf16"): None, ("sm_107a", 1, 4096, 64, "f32"): None,
    ("sm_107a", 1, 4096, 128, "bf16"): None, ("sm_107a", 1, 4096, 128, "f32"): None,
    ("sm_107a", 1, 4096, 512, "bf16"): None, ("sm_107a", 1, 4096, 512, "f32"): (128, 1),
    ("sm_107a", 1, 4096, 1024, "f32"): (256, 1),
    ("sm_107a", 1, 4096, 4096, "f32"): None,
    ("sm_107a", 1, 5120, 32, "bf16"): None, ("sm_107a", 1, 5120, 32, "f32"): (64, 2),
    ("sm_107a", 1, 5120, 48, "f32"): (64, 2),
    ("sm_107a", 1, 5120, 64, "f32"): None,
    ("sm_107a", 1, 5120, 512, "f32"): (128, 1),
    ("sm_107a", 1, 5120, 1024, "bf16"): (256, 1), ("sm_107a", 1, 5120, 1024, "f32"): (256, 1),
    ("sm_107a", 1, 6144, 32, "bf16"): None, ("sm_107a", 1, 6144, 32, "f32"): None,
    ("sm_107a", 1, 8192, 128, "bf16"): (128, 2), ("sm_107a", 1, 8192, 128, "f32"): (128, 2),
}
# fmt: on
#: The fewest SMs a device of the architecture needs for the verdicts of :data:`K1_SPLIT_RULES`; a device with fewer
#: takes the verdicts of :data:`K1_SPLIT_RULES_BY_SM` for its SM count (every K1 row runs the deterministic program
#: where that table has none), and a row of :data:`K1_RULE_MIN_SM_COUNT` needs more.  The sm_100a and sm_103a verdicts
#: were measured on 148-SM and 152-SM parts (each fits one wave of the 74 CTA pairs of 148 SMs); the sm_107a verdicts
#: on 216-SM parts.
K1_MIN_SM_COUNT: dict[str, int] = {"sm_100a": 148, "sm_103a": 148, "sm_107a": 216}
#: Rows of :data:`K1_SPLIT_RULES` whose verdict holds only from a higher SM count than :data:`K1_MIN_SM_COUNT` of
#: their architecture, ``key -> fewest SMs``: below it the row runs the deterministic program.  sm_103a d_in 4096 to
#: d_out 48 is faster on 152 SMs and not faster than the deterministic plan in every run on 148 SMs.
K1_RULE_MIN_SM_COUNT: dict[tuple[str, int, int, int, str], int] = {
    ("sm_103a", 1, 4096, 48, "bf16"): 152,
    ("sm_103a", 1, 4096, 48, "f32"): 152,
}
#: Token counts at which a row of :data:`K1_SPLIT_RULES` (or its entry in :data:`K1_SPLIT_OVERRIDES_BY_SM`) or of a
#: table of :data:`K1_SPLIT_RULES_BY_SM` does not take its verdict,
#: ``key -> ((sm_count, min_tokens, max_tokens), ...)``: a token count (``K`` of the weight gradient) from
#: ``min_tokens`` to ``max_tokens`` (``None``: no upper bound) runs the deterministic program on a part of at least
#: ``sm_count`` SMs where ``sm_count`` is at or above the row's SM-count floor (:data:`K1_MIN_SM_COUNT`,
#: :data:`K1_RULE_MIN_SM_COUNT`), and on a part of exactly ``sm_count`` SMs below it (the parts of
#: :data:`K1_SPLIT_RULES_BY_SM`); every other token count and part keeps the row's verdict.  The verdicts were measured
#: at 16172 to 16384 tokens.  The sm_100a / sm_103a entries come from a re-measurement at 512, 1024, 2048, 4096, 8192,
#: 16231, 32768 and 65536 tokens on two 152-SM parts per row and token count (GB200, GB300; the strict runs of 8 GB300
#: points ran on two other parts, four GB300 parts in all; the protocol is in the K1 token table of the README).  A
#: measured token count is an entry where the split program failed in at least one run (the rule above, or the backward
#: pass or the checkpointed step more than 1.3 % slower than with the deterministic program) and the deterministic
#: program leaves the backward pass and the checkpointed step no more than 1.3 % slower, or where the split program's
#: output missed the deterministic package's error budgets against an FP64 reference of the exact operands (BF16 output
#: ``atol = rtol = 1e-2`` elementwise with no element outside; FP32 output a relative Frobenius error ``<= 5e-5`` and a
#: max error ``<= 1e-3 * max|ref|``) for some seed while the deterministic program's met it.  A token count between two
#: consecutive measured entries is an entry too (not measured).  The split program's FP64 error relative to the
#: deterministic program's is reported, not a rule.  Where the deterministic program misses the budget for some seed
#: too, the budget does not choose a program, and these open-ended entries keep the deterministic program: sm_103a
#: ``4096 -> 32`` and ``4096 -> 48`` above 65536 tokens (BF16: both programs outside the budget on every seed from 65537
#: tokens; FP32: the two-split program's output is the deterministic program's bits at an even number of 64-token K
#: blocks, and at an odd number it met the budget at 65537 tokens and missed it on every seed from 98305 tokens, as the
#: deterministic program's did; the deterministic plan of these rows is the same on 148 and 152 SMs, and their entries
#: start at the token counts at which the split program failed the timing rule above), and the open-ended sm_100a BF16
#: entries of ``4096 -> 48`` and ``5120 -> 48`` from 49153 tokens, ``4096 -> 32`` from 65536 tokens and ``4096 -> 64``
#: from 65536 tokens on 152 SMs (the 148-SM entry is in :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS_BY_SM`), where the split
#: program's output missed the BF16 budget for some seed and the deterministic program's did too (10 seeds per token
#: count on a 148-SM B200 part; the deterministic plan of these rows is the same on 148 and 152 SMs).  sm_100a
#: ``5120 -> 32`` BF16 keeps its entry from 65537 tokens: measured with 10 seeds (20 runs each) at 32769 to 65025 tokens
#: on a 148-SM B200 part and two 152-SM GB200 parts (finer from 45569 to 49665 tokens), both programs met the BF16
#: budget on every seed up to 46593 tokens, and from 47105 tokens each missed it for some seed at some token counts,
#: neither more accurate at every one; the budget does not choose a program there.  The sm_103a BF16 entries of
#: ``5120 -> 32`` and ``5120 -> 64`` start at 45058 tokens: their split program's output met the BF16 budget on every
#: seed at 32768 tokens (30 seeds on two parts) and at every measured token count up to 45057 tokens on a 152-SM GB300
#: part (10 seeds, 20 runs each), and missed it for some seed at the next measured token count, 47105, where the
#: deterministic program's output missed it too; the row keeps the deterministic program from there up (the split
#: program misses it on every seed at 65536 tokens, 30 seeds on two parts).  That of ``5120 -> 128`` starts at 32768
#: tokens: its split program met the budget up to 45057 tokens as well, but in a strict timing re-measurement on that
#: part it failed the rule above at 5 of 11 token counts from 32768 to 45057 (32768, 33793, 38913, 43009 and 44033) and
#: kept it at the other 6, so no measured window holds.  That of ``5120 -> 48`` starts at 40961 tokens: on the 152-SM
#: part its split program's output met the budget on every seed up to 49152 tokens (the deterministic program's missed
#: it for one seed at 45057 and at 49152 tokens), but on a 148-SM B300 part, which runs the same program, it missed it
#: for one of 10 seeds at 40961 and at 40962 tokens while the deterministic program's met it on every seed; from 49153
#: tokens both miss it on the 152-SM part.  Unmeasured token counts keep the verdict: below 512, above 65536 (except the
#: rows with open-ended entries), between two measured token counts with different outcomes, and on parts with fewer
#: than 152 SMs (148-SM parts: their own entries in :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS_BY_SM`).  The sm_107a entries
#: are single token counts of the deterministic package's sm_107a rules for one token count (16231 and 16384 tokens;
#: 16231 only for ``4096 -> 1024``), which give the row a faster deterministic plan there, and 16231 tokens of
#: ``5120 -> 1024`` BF16 on 216 SMs (measured in the same runs; the deterministic plan there is the row's plan at every
#: token count).  The other single 216-SM token counts (512 and 65536) come from a re-measurement of chosen points at
#: 512, 8192, 32768 and 65536 tokens on two 216-SM parts per point under the rule of the sm_100a / sm_103a entries (the
#: protocol is in the sm_107a K1 token table of the README); their neighbouring token counts were not measured.  On 208,
#: 212 and 216 SMs the window entries come from a re-measurement at 16172, 16231 and 16384 tokens against those plans
#: (the protocol is in the sm_107a K1 token table of the README): a measured token count is an entry where the row's
#: program failed the rule above in some run, or left the backward pass or the checkpointed step more than 1.3 % slower
#: than the deterministic program (ratio of the medians over the runs, or median of the paired ratios, in eager launches
#: or in graph replays).  The 208-SM entry of ``4096 -> 1024`` FP32 at 16384 tokens comes from a later strict
#: re-measurement of the restored window on a 208-SM R200 part: the deterministic program's plan there (160-wide tiles)
#: was faster, and the row's program was 14.5 % slower at the operation and left the checkpointed step 6.3 % slower in
#: graph replays. The open-ended sm_107a entries come from neither rule: on parts of at least 216 SMs the FP32 rows
#: ``4096 -> 512``, ``4096 -> 1024``, ``5120 -> 512`` and ``5120 -> 1024`` run the deterministic program from 40961
#: tokens up (``5120 -> 1024`` from 32769, below) because their one-split program's FP64 relative Frobenius error
#: exceeds the deterministic package's FP32 error budget (5e-5) on every seed there.  That program adds the whole token
#: reduction of an output in one chain, and its error grows about linearly with the token count.  Measured with the
#: 216-SM plan on 212-SM R200 parts (the same program and accumulation order as on 216 SMs; the error does not depend on
#: the SM count) against an FP64 reference, each row on two parts, 5 seeds, at 512 to 65536 tokens: within the budget on
#: every seed up to 40960 tokens (1.86e-5 at 16384 tokens, 3.77e-5 at 32768) and above it on every seed from 45056
#: tokens up (7.6e-5 at 65536 tokens, where the deterministic program's is 3.8e-5).  The range has no upper bound: the
#: error keeps growing with the token count.  The token counts between 40960 and 45056 run the deterministic program too
#: (not measured).  From 961 tokens (16 K blocks) up the deterministic program of these rows cuts the token reduction
#: into units of ``ceil(blocks / 2)`` K blocks and adds the partials in a fixed order (its automatic two-way split);
#: there the one-split program's error is 2.0 to 2.4 times the deterministic program's (reported); up to 960 tokens the
#: two give the same bits.  A later strict re-measurement on a 216-SM R200 part, each failure reproduced in a second
#: run, moved two more token ranges to the deterministic program on 216-SM parts: ``5120 -> 1024`` from 32769 tokens
#: up, whose one-split program left the checkpointed step 1.6 to 5.3 % slower at 36864 and 40960 tokens (eager launches
#: or graph replays; the weight gradient itself took 0.87 to 0.92 times the deterministic program's time, and the output
#: met the FP32 budget), while it kept the rule and the 1.3 % bound at 16385 to 32768 tokens; and ``4096 -> 1024`` at
#: 16384 tokens, where its 216-SM override (:data:`K1_SPLIT_OVERRIDES_BY_SM`) failed the rule above in 4 and then 5 of
#: its 8 cells (two launch layouts in both orders, eager launches and graph replays; median time ratios 0.982 to 0.995
#: there).  The 217-SM entries keep the four rows on the deterministic program from 961 tokens up on parts of more than
#: 216 SMs, the route they took before that measurement: the one-split program's time at 961 to 40960 tokens was
#: measured on 216-SM parts only (where the ranges of :data:`K1_CHAIN_RANGES_BY_SM` come first), and no part of more
#: SMs was measured.  From 961 to 40961 tokens the ranges of :data:`K1_PARITY_RANGES_BY_SM` come first on those parts
#: (:data:`K1_PARITY_RANGES_ABOVE_SM`).  The sm_107a rows that now run the deterministic program at every token count (their three- or
#: four-split verdicts are gone, :data:`K1_SPLIT_RULES`) have no entries.
# fmt: off
K1_SPLIT_RULE_EXCLUDED_TOKENS: dict[
    tuple[str, int, int, int, str], tuple[tuple[int, int, Optional[int]], ...]
] = {
    ("sm_100a", 1, 4096, 32, "bf16"): ((148, 65536, None), (152, 512, 512)),
    ("sm_100a", 1, 4096, 48, "bf16"): ((148, 49153, None), (152, 512, 512)),
    ("sm_100a", 1, 4096, 48, "f32"): ((152, 512, 512),),
    ("sm_100a", 1, 4096, 64, "bf16"): ((152, 512, 512), (152, 65536, None)),
    ("sm_100a", 1, 4096, 64, "f32"): ((152, 512, 512),),
    ("sm_100a", 1, 5120, 32, "bf16"): ((148, 65537, None), (152, 512, 512)),
    ("sm_100a", 1, 5120, 32, "f32"): ((152, 512, 512),),
    ("sm_100a", 1, 5120, 48, "bf16"): ((148, 49153, None), (152, 512, 2048)),
    ("sm_100a", 1, 5120, 48, "f32"): ((152, 512, 2048),),
    ("sm_100a", 1, 5120, 64, "bf16"): ((152, 512, 2048), (152, 8192, 8192)),
    ("sm_100a", 1, 5120, 64, "f32"): ((152, 512, 2048), (152, 8192, 8192)),
    ("sm_100a", 1, 6144, 32, "bf16"): ((152, 1024, 1024),),
    ("sm_103a", 1, 4096, 32, "bf16"): ((152, 512, 512), (152, 65536, None)),
    ("sm_103a", 1, 4096, 32, "f32"): ((152, 512, 512), (152, 32768, None)),
    ("sm_103a", 1, 4096, 48, "bf16"): ((152, 512, 512), (152, 32768, None)),
    ("sm_103a", 1, 4096, 48, "f32"): ((152, 512, 512), (152, 32768, None)),
    ("sm_103a", 1, 5120, 32, "bf16"): ((152, 512, 512), (152, 45058, None)),
    ("sm_103a", 1, 5120, 32, "f32"): ((152, 512, 512),),
    ("sm_103a", 1, 5120, 48, "bf16"): ((152, 512, 2048), (152, 40961, None)),
    ("sm_103a", 1, 5120, 48, "f32"): ((152, 512, 2048),),
    ("sm_103a", 1, 5120, 64, "bf16"): ((152, 512, 2048), (152, 45058, None)),
    ("sm_103a", 1, 5120, 64, "f32"): ((152, 512, 2048),),
    ("sm_103a", 1, 5120, 128, "bf16"): ((152, 512, 8192), (152, 32768, None)),
    ("sm_103a", 1, 5120, 128, "f32"): ((152, 512, 4096),),
    ("sm_103a", 1, 6144, 32, "bf16"): ((152, 512, 512),),
    ("sm_103a", 1, 6144, 32, "f32"): ((152, 512, 512),),
    ("sm_103a", 1, 6144, 128, "bf16"): ((152, 512, 8192),),
    ("sm_103a", 1, 6144, 128, "f32"): ((152, 512, 8192),),
    ("sm_107a", 1, 4096, 32, "f32"): ((216, 16231, 16231),),
    ("sm_107a", 1, 4096, 512, "f32"): (
        (208, 16231, 16231), (208, 16384, 16384), (212, 16231, 16231), (212, 16384, 16384),
        (216, 16231, 16231), (216, 16384, 16384), (216, 40961, None), (217, 961, None),
    ),
    ("sm_107a", 1, 4096, 1024, "f32"): (
        (208, 16231, 16231), (208, 16384, 16384), (212, 16231, 16231), (212, 16384, 16384),
        (216, 16231, 16231), (216, 16384, 16384), (216, 40961, None), (217, 961, None),
    ),
    ("sm_107a", 1, 5120, 32, "f32"): ((216, 512, 512),),
    ("sm_107a", 1, 5120, 48, "f32"): (
        (208, 16231, 16231), (208, 16384, 16384), (212, 16231, 16231), (216, 512, 512),
        (216, 16231, 16231), (216, 16384, 16384),
    ),
    ("sm_107a", 1, 5120, 512, "f32"): (
        (212, 16231, 16231), (212, 16384, 16384), (216, 40961, None), (217, 961, None),
    ),
    ("sm_107a", 1, 5120, 1024, "bf16"): ((216, 16231, 16231),),
    ("sm_107a", 1, 5120, 1024, "f32"): ((216, 32769, None), (217, 961, None)),
    ("sm_107a", 1, 8192, 128, "bf16"): ((216, 512, 512), (216, 65536, 65536)),
    ("sm_107a", 1, 8192, 128, "f32"): ((216, 512, 512), (216, 65536, 65536)),
}
# fmt: on
#: Token counts at which a row of :data:`K1_SPLIT_RULES` runs the deterministic program on the parts of exactly one SM
#: count, ``(arch, sm_count) -> {key: ((min_tokens, max_tokens), ...)}`` (``max_tokens`` ``None``: no upper bound).  An
#: entry of :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS` at or above a row's SM-count floor holds from its SM count up; these
#: hold on exactly their SM count, and every other part keeps its routing.  The sm_100a / sm_103a 148-SM entries (B200 /
#: B300 parts) follow the rule of the 152-SM entries of :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS`, measured on 148-SM parts
#: at 512, 1024, 2048, 4096, 8192, 16231, 32768 and 65536 tokens (the protocol is in the 148-SM K1 token table of the
#: README): every point in a screen on one part, the points that failed there again on a second part (B300: every point
#: on a part of a second node too), and the FP64 error of 5 seeds on two parts at 32768 and 65536 tokens and at the
#: first and the last token count of every entry and one token either side of it (and at 98304 and 131072 tokens for the
#: entries that reach 65536 tokens).  A measured token count is an entry where the split program failed in some run and
#: the deterministic program leaves the backward pass and the checkpointed step no more than 1.3 % slower on both parts
#: (the worst of the ratio of the medians and the median of the paired ratios, eager launches and graph replays), or
#: where the split program's output missed the deterministic package's error budget against FP64 for some seed while the
#: deterministic program's met it.  Where both miss it for some seed (BF16 output at the largest token counts), the
#: budget does not choose a program and the entry keeps the deterministic program.  A token count between two
#: consecutive measured entries is an entry too.  Entries without an upper bound: sm_103a ``4096 -> 32`` and sm_100a
#: ``4096 -> 64`` (BF16 and FP32) from 65536 tokens up: there the split program's output is the deterministic program's
#: bits where the token count spans an even number of 64-token blocks (65601 to 131072 tokens), where it does not make
#: the backward pass or the checkpointed step 1 % faster (sm_100a: the backward pass up to 3.5 % slower); where it spans
#: an odd number (65537, 65538, 65600 and 65665 tokens) its BF16 output and the deterministic program's both miss the
#: budget for some seed, and its FP32 output (a relative Frobenius error 2.1 to 2.3 % above the deterministic program's:
#: reported) misses the FP32 budget on every seed from 98305 tokens, as the deterministic program's does.  sm_103a
#: ``5120 -> 48`` BF16 runs the deterministic program at 40961 to 49151 tokens and ``5120 -> 64`` BF16 at 47105 to 49152
#: tokens on 148 SMs: on a 148-SM B300 part (10 seeds, 20 runs each, at 36865 to 61441 tokens) their split program's
#: output missed the BF16 budget for one of 10 seeds at 40961 and at 40962 tokens (``5120 -> 64``: 47105) where the
#: deterministic program's met it on every seed (``5120 -> 48`` met it at 41985 to 47105 tokens inside its range), and
#: from 49152 tokens (``5120 -> 64``: 49153) both programs missed it for some seed, so the split program stays there.
#: Unmeasured token counts keep the verdict.  The 152-SM ``sm_103a`` entries keep the FP32 rows ``5120 -> 32 / 48 / 64 /
#: 128`` on the deterministic program at the token ranges where :data:`K1_DSM_RANGES_BY_SM` ran a cluster of three CTAs
#: and at 16384 tokens of ``5120 -> 32`` (the accuracy check against the reference implementation described there),
#: and ``5120 -> 48`` BF16 at 16231 tokens, where its two-split output missed that check on the mirrored-cancellation
#: inputs with the permuted split (1.104 times the reference's worst error).
# fmt: off
K1_SPLIT_RULE_EXCLUDED_TOKENS_BY_SM: dict[
    tuple[str, int],
    dict[tuple[str, int, int, int, str], tuple[tuple[int, Optional[int]], ...]],
] = {
    ("sm_100a", 148): {
        ("sm_100a", 1, 4096, 48, "bf16"): ((512, 512),),
        ("sm_100a", 1, 4096, 48, "f32"): ((512, 512),),
        ("sm_100a", 1, 4096, 64, "bf16"): ((512, 512), (65536, None)),
        ("sm_100a", 1, 4096, 64, "f32"): ((512, 512), (65536, None)),
        ("sm_100a", 1, 5120, 48, "bf16"): ((512, 2048),),
        ("sm_100a", 1, 5120, 48, "f32"): ((512, 512),),
        ("sm_100a", 1, 5120, 64, "bf16"): ((512, 2048), (8192, 8192)),
        ("sm_100a", 1, 5120, 64, "f32"): ((512, 1024), (8192, 8192)),
    },
    ("sm_103a", 152): {
        ("sm_103a", 1, 5120, 32, "f32"): ((8192, 12287), (16384, 16384)),
        ("sm_103a", 1, 5120, 48, "bf16"): ((16231, 16231),),
        ("sm_103a", 1, 5120, 48, "f32"): ((4096, 8192),),
        ("sm_103a", 1, 5120, 64, "f32"): ((4096, 6143), (12288, 24576)),
        ("sm_103a", 1, 5120, 128, "f32"): ((4097, 32768),),
    },
    ("sm_103a", 148): {
        ("sm_103a", 1, 4096, 32, "bf16"): ((65536, None),),
        ("sm_103a", 1, 4096, 32, "f32"): ((65536, None),),
        ("sm_103a", 1, 5120, 48, "bf16"): ((512, 2048), (40961, 49151)),
        ("sm_103a", 1, 5120, 48, "f32"): ((512, 2048),),
        ("sm_103a", 1, 5120, 64, "bf16"): ((512, 1024), (47105, 49152)),
        ("sm_103a", 1, 5120, 64, "f32"): ((512, 1024),),
        ("sm_103a", 1, 5120, 128, "bf16"): ((512, 4096), (32768, 65536)),
        ("sm_103a", 1, 5120, 128, "f32"): ((512, 4096), (32768, 65536)),
        ("sm_103a", 1, 6144, 128, "bf16"): ((512, 8192),),
        ("sm_103a", 1, 6144, 128, "f32"): ((512, 8192),),
    },
}
# fmt: on
#: Token ranges at which a row of :data:`K1_SPLIT_RULES` (or of a table of :data:`K1_SPLIT_RULES_BY_SM`) runs the
#: deterministic program where the token count spans an odd number of 64-element K blocks (``ceil(K / 64)`` odd),
#: ``key -> ((sm_count, min_tokens, max_tokens), ...)`` (parts and ``max_tokens`` as in
#: :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS`).  A token count with an even block count, and every token count outside the
#: ranges, keeps the routing of the tables above.  In these rows the deterministic program cuts the token reduction of
#: its tiles, taken in order, into units of ``ceil(blocks / 2)`` K blocks (at an odd block count a unit runs on into the
#: next tile) and adds the partials in a fixed order: from 961 tokens (16 K blocks) up for ``4096 -> 32``,
#: ``4096 -> 48`` and ``4096 -> 64``; for sm_100a ``5120 -> 32`` and ``5120 -> 48`` from where one wave's operands
#: outgrow the L2 (from 12289 tokens with 126 MiB of L2, between 12292 and 12337 tokens on a B200; below, whole tiles).
#: At an even block count the two-split program's output is the deterministic program's bit for bit; at an odd block
#: count it is not, and its FP32 output met the deterministic package's FP32 error budget on every seed at every
#: measured odd token count from 1025 to 65537 tokens (10 seeds each; a relative Frobenius error 1.03 to 1.15 times the
#: deterministic program's at 1025 to 49153 tokens and 1.021 to 1.024 times at 65537 for the ``4096`` rows on 148-SM
#: B200 and B300 parts, whose deterministic plan is the same on 148 and 152 SMs, and 1.025 to 1.075 times at 12337 to
#: 32769 tokens for sm_100a ``5120 -> 32`` and ``5120 -> 48`` on a B200: reported, not a rule).  From 98305 tokens up
#: neither program's FP32 output met the budget on every seed (98305 and 131073 tokens), so the ranges start above the
#: last measured odd token count at which the two-split program met it (65537; the token counts up to 98305 were not
#: measured).  The BF16 outputs of these rows had the deterministic program's relative Frobenius error and met the error
#: budget on every seed where the deterministic program met it.  The sm_103a ``4096 -> 48`` range from 14338 to 32767
#: tokens comes from the time instead: on a 152-SM GB300 part, at odd block counts the two-split program left the
#: checkpointed step 2.0 to 2.3 % slower in graph replays at 16173 tokens (two measurements), the backward pass 1.6 and
#: 2.0 % slower in eager launches at 20481 and 28673 tokens, and failed the K1 rule in one of 8 groups at 18433 tokens,
#: while it kept the rule and the 1.3 % bound at 1025, 4097, 6145, 8193, 10241, 12289 and 14337 tokens.  sm_107a
#: ``8192 -> 128`` FP32 (two splits on 212 and 216 SMs) has an entry from 217 SMs up only: at an odd block count from
#: 1025 to 8192 tokens, where its deterministic program's two-way split runs while one wave's operands fit the L2 (to
#: 7808 tokens with 126 MiB of L2, to between 4097 and 8192 tokens on a 212-SM R200), its output differs from the
#: deterministic program's and met the FP32 budget on every seed (10 seeds; an error 1.04 to 1.14 times the
#: deterministic program's at 1025 to 4097 tokens: reported), and its time there was measured on 212-SM and 216-SM parts
#: only, so parts of more SMs keep the deterministic program there (the route they took before the 216-SM measurement).
# fmt: off
K1_SPLIT_ODD_BLOCK_EXCLUDED_TOKENS: dict[
    tuple[str, int, int, int, str], tuple[tuple[int, int, Optional[int]], ...]
] = {
    ("sm_100a", 1, 4096, 32, "f32"): ((148, 65538, None),),
    ("sm_100a", 1, 4096, 48, "f32"): ((148, 65538, None),),
    ("sm_100a", 1, 4096, 64, "f32"): ((148, 65538, None),),
    ("sm_100a", 1, 5120, 32, "f32"): ((148, 65538, None),),
    ("sm_100a", 1, 5120, 48, "f32"): ((148, 65538, None),),
    ("sm_103a", 1, 4096, 32, "f32"): ((148, 65538, None),),
    ("sm_103a", 1, 4096, 48, "f32"): ((152, 14338, 32767), (152, 65538, None)),
    ("sm_107a", 1, 8192, 128, "f32"): ((217, 1025, 8192),),
}
# fmt: on
#: Per-SM-count verdicts of the parts of an architecture with fewer SMs than its :data:`K1_MIN_SM_COUNT`,
#: ``(arch, sm_count) -> {key: (block_n, num_splits) or None}`` (keys and values as in :data:`K1_SPLIT_RULES`, measured
#: on parts of exactly that SM count under the same rule, in two jobs on different parts, at 16172 to 16384 tokens).
#: Such a part takes the table of its own SM count only, and only at the token counts of
#: :data:`K1_SPLIT_RULES_BY_SM_TOKENS` and of the row's ranges in :data:`K1_SPLIT_RULES_BY_SM_TOKEN_RANGES` for the
#: part; a part without one, and every other token count, runs every K1 row on the deterministic program.  The
#: sm_107a tables were measured on 208-SM and 212-SM R200 parts.  Rows whose deterministic
#: program is a stream-K launch that did not complete next to a kernel holding SMs on another stream have no measured
#: verdict and stay on it, on 208 and on 212 SMs: ``4096 -> 32`` and ``4096 -> 64`` BF16, ``5120 -> 32`` BF16 and
#: ``6144 -> 32``.  ``4096 -> 4096`` FP32 runs the deterministic program on 212 SMs too: the deterministic package's
#: sm_107a rule for the row (160-wide tiles at every token count) is faster than this package's 256-wide tiles at one
#: split there in every run.  Not faster than the deterministic plan in every run on 208 SMs (one of the parts runs at
#: a lower SM clock): ``8192 -> 128``, ``4096 -> 4096`` FP32 and ``5120 -> 1024`` BF16; the last also runs the
#: deterministic program on 212 SMs at 16172 to 16384 tokens, where it is 1.3 to 2.5 % faster (its 212-SM range in
#: :data:`K1_SPLIT_RANGE_OVERRIDES_BY_SM` runs this package's program).  On 212 SMs ``4096 -> 1024`` FP32 takes the
#: 192-wide row-major program at one split (88 tiles of 256 ``d_out`` rows by 192 ``d_in`` columns, 176 CTAs, one wave
#: next to a kernel holding 32 SMs), which keeps more SMs busy than the 128 CTAs of the 256-wide transposed tile.  The
#: token counts of :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS` for a part's SM count run the deterministic program.  The FP32
#: rows ``4096 / 5120 -> 512 / 1024`` take the ranges of :data:`K1_PARITY_RANGES_BY_SM` first (on 208 and 212 SMs).
# fmt: off
K1_SPLIT_RULES_BY_SM: dict[
    tuple[str, int], dict[tuple[str, int, int, int, str], Optional[tuple[int, int]]]
] = {
    ("sm_107a", 208): {
        ("sm_107a", 1, 4096, 32, "bf16"): None, ("sm_107a", 1, 4096, 32, "f32"): (64, 2),
        ("sm_107a", 1, 4096, 48, "f32"): None,
        ("sm_107a", 1, 4096, 64, "bf16"): None, ("sm_107a", 1, 4096, 64, "f32"): None,
        ("sm_107a", 1, 4096, 512, "bf16"): None, ("sm_107a", 1, 4096, 512, "f32"): (128, 1),
        ("sm_107a", 1, 4096, 1024, "f32"): (256, 1),
        ("sm_107a", 1, 4096, 4096, "f32"): None,
        ("sm_107a", 1, 5120, 32, "bf16"): None, ("sm_107a", 1, 5120, 32, "f32"): (64, 2),
        ("sm_107a", 1, 5120, 48, "f32"): (64, 2),
        ("sm_107a", 1, 5120, 64, "f32"): None,
        ("sm_107a", 1, 5120, 512, "f32"): (128, 1),
        ("sm_107a", 1, 5120, 1024, "bf16"): None, ("sm_107a", 1, 5120, 1024, "f32"): (256, 1),
        ("sm_107a", 1, 6144, 32, "bf16"): None, ("sm_107a", 1, 6144, 32, "f32"): None,
        ("sm_107a", 1, 8192, 128, "bf16"): None, ("sm_107a", 1, 8192, 128, "f32"): None,
    },
    ("sm_107a", 212): {
        ("sm_107a", 1, 4096, 32, "bf16"): None, ("sm_107a", 1, 4096, 32, "f32"): (64, 2),
        ("sm_107a", 1, 4096, 48, "f32"): None,
        ("sm_107a", 1, 4096, 64, "bf16"): None, ("sm_107a", 1, 4096, 64, "f32"): None,
        ("sm_107a", 1, 4096, 512, "bf16"): None, ("sm_107a", 1, 4096, 512, "f32"): (128, 1),
        ("sm_107a", 1, 4096, 1024, "f32"): (192, 1),
        ("sm_107a", 1, 4096, 4096, "f32"): None,
        ("sm_107a", 1, 5120, 32, "bf16"): None, ("sm_107a", 1, 5120, 32, "f32"): (64, 2),
        ("sm_107a", 1, 5120, 48, "f32"): (64, 2),
        ("sm_107a", 1, 5120, 64, "f32"): None,
        ("sm_107a", 1, 5120, 512, "f32"): (128, 1),
        ("sm_107a", 1, 5120, 1024, "bf16"): None, ("sm_107a", 1, 5120, 1024, "f32"): (256, 1),
        ("sm_107a", 1, 6144, 32, "bf16"): None, ("sm_107a", 1, 6144, 32, "f32"): None,
        ("sm_107a", 1, 8192, 128, "bf16"): (128, 2), ("sm_107a", 1, 8192, 128, "f32"): (128, 2),
    },
}
# fmt: on
#: The token counts (``K`` of the weight gradient, ``(min_tokens, max_tokens)``) the tables of
#: :data:`K1_SPLIT_RULES_BY_SM` and :data:`K1_SPLIT_OVERRIDES_BY_SM` hold for: they were measured at 16172 to 16384
#: tokens only.  Below the SM-count floor every other token count runs the deterministic program, except the ranges of
#: :data:`K1_SPLIT_RULES_BY_SM_TOKEN_RANGES`; an override leaves every other token count to :data:`K1_SPLIT_RULES`.
K1_SPLIT_RULES_BY_SM_TOKENS: tuple[int, int] = (16172, 16384)
#: Further token counts at which a row of a table of :data:`K1_SPLIT_RULES_BY_SM` takes its verdict, ``(arch,
#: sm_count) -> {key: ((min_tokens, max_tokens), ...)}`` (keys as in :data:`K1_SPLIT_RULES`, rows of the part's table
#: with a program): a token count from ``min_tokens`` to ``max_tokens`` of one of the row's ranges takes the verdict as
#: one of :data:`K1_SPLIT_RULES_BY_SM_TOKENS` does, except the token counts of :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS`
#: for the part.  Every other row, part and token count keeps the window alone.  Measured on 212-SM R200 parts at 512,
#: 1024, 2048, 4096, 8192, 32768 and 65536 tokens against the deterministic package's plan of the row, in two nodes of
#: four parts each, with the protocol of the window (the sm_107a K1 token table of the README), and the FP64 error of 5
#: seeds on two parts.  A measured token count joins a range where the row's program passed the K1 rule above in every
#: run, left the backward pass and the checkpointed step no more than 1.3 % slower than the deterministic program (ratio
#: of the medians and median of the paired ratios, eager launches and graph replays), and met the deterministic
#: package's FP64 error budget for every seed on both parts.  A range runs from such a measured token count to
#: the window's first token count, or from the window's last one to such a measured token count, and holds no measured
#: token count that failed; the token counts between two consecutive measured ones inside it take the verdict too (not
#: measured).  The token counts below a range's first measured token count, above its last, and those between a
#: measured token count that passed and one that failed run the deterministic program.  On 212 SMs:
#: ``8192 -> 128`` from 1024 to 32768 tokens (at 512 slower, at 65536 not faster in every run), and
#: the FP32 rows ``4096 / 5120 -> 512 / 1024`` at 512 tokens only (from 1024 tokens up their one-split program was not
#: taken for its FP64 error, about twice the deterministic program's, which is reported data and not a rule; there
#: these rows run the programs of :data:`K1_CHAIN_RANGES_BY_SM` and :data:`K1_TWO_SPLIT_RANGES_BY_SM`).  The FP32 rows
#: ``4096 -> 32, 48, 64`` and ``5120 -> 32, 48, 64`` held three splits from 1024 to 65536 tokens; three partials added
#: in the order they finish vary from run to run and missed the accuracy check of :data:`K1_DSM_RANGES_BY_SM`, so these
#: rows now take their two-split or deterministic verdict in the window only (two splits were timed there alone).
# fmt: off
K1_SPLIT_RULES_BY_SM_TOKEN_RANGES: dict[
    tuple[str, int], dict[tuple[str, int, int, int, str], tuple[tuple[int, int], ...]]
] = {
    ("sm_107a", 212): {
        ("sm_107a", 1, 4096, 512, "f32"): ((512, 512),),
        ("sm_107a", 1, 4096, 1024, "f32"): ((512, 512),),
        ("sm_107a", 1, 5120, 512, "f32"): ((512, 512),),
        ("sm_107a", 1, 5120, 1024, "f32"): ((512, 512),),
        ("sm_107a", 1, 8192, 128, "bf16"): ((1024, 16172), (16384, 32768)),
        ("sm_107a", 1, 8192, 128, "f32"): ((1024, 16172), (16384, 32768)),
    },
}
# fmt: on
#: Verdicts that replace a row's verdict of a table of :data:`K1_SPLIT_RULES_BY_SM` at further token ranges, ``(arch,
#: sm_count) -> {key: ((min_tokens, max_tokens, (block_n, num_splits)), ...)}`` (keys and values as in
#: :data:`K1_SPLIT_RULES`): on a part of exactly ``sm_count`` SMs below the row's SM-count floor a token count from
#: ``min_tokens`` to ``max_tokens`` runs that range's program, except the token counts of
#: :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS` for the part.  A range lies outside :data:`K1_SPLIT_RULES_BY_SM_TOKENS` and
#: outside the row's ranges in :data:`K1_SPLIT_RULES_BY_SM_TOKEN_RANGES`; every other token count, part and row keeps
#: those tables.  Measured as those ranges (the protocol and the rule are in the 212-SM K1 token-range table of the
#: README).  On 212 SMs ``5120 -> 1024`` BF16 runs the 256-wide program at one split (its :data:`K1_SPLIT_RULES`
#: verdict) from 512 to 12288 tokens; it stays on the deterministic program at 16172 to 16384 tokens, where the
#: deterministic plan is faster, and from 32768 tokens up: at 32768 tokens it failed the K1 rule in every run (0.1 to
#: 0.6 % faster than the deterministic plan alone), and its checkpointed step measured 2.5 to 4.3 % slower, as did a
#: second deterministic launch against the first there (1.9 to 4.4 %).
# fmt: off
K1_SPLIT_RANGE_OVERRIDES_BY_SM: dict[
    tuple[str, int],
    dict[tuple[str, int, int, int, str], tuple[tuple[int, int, tuple[int, int]], ...]],
] = {
    ("sm_107a", 212): {
        ("sm_107a", 1, 5120, 1024, "bf16"): ((512, 12288, (256, 1)),),
    },
}
# fmt: on
#: Verdicts of rows on a part of exactly one SM count at further token ranges,
#: ``(arch, sm_count) -> {key: ((min_tokens, max_tokens, (block_n, num_splits)), ...)}`` (keys and values as in
#: :data:`K1_SPLIT_RULES`): on a part of exactly ``sm_count`` SMs, below the row's SM-count floor or at or above it, a
#: token count from ``min_tokens`` to ``max_tokens`` runs that range's program, at an even and at an odd number of
#: 64-element K blocks, except the token counts of :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS` for the part and the token
#: counts of the row's ranges in :data:`K1_CHAIN_RANGES_BY_SM` for the part, which come first.  Every other token count,
#: part and row keeps the routing of the tables above.  A range lies outside :data:`K1_SPLIT_RULES_BY_SM_TOKENS` and
#: outside the row's ranges in :data:`K1_SPLIT_RULES_BY_SM_TOKEN_RANGES` and :data:`K1_SPLIT_RANGE_OVERRIDES_BY_SM`.
#: Every program here is two splits, each adding half of an output's K blocks: at an even block count its output is the
#: deterministic program's bit for bit; at an odd count it is not, and its FP32 output met the deterministic package's
#: FP32 error budget on every measured seed (its FP64 error relative to the deterministic program's is reported, not a
#: rule).  On 152-SM sm_100a and sm_103a parts the FP32 row ``4096 -> 128`` (no verdict in :data:`K1_SPLIT_RULES`) runs
#: two splits of 64-wide tiles: its output was the deterministic program's bit for bit at an even block count (5 seeds
#: at 1024 to 8192 tokens); at an odd count (1088 and 2112 tokens) it was not, with an FP64 relative Frobenius error
#: 1.107 and 1.142 times the deterministic program's.  It passed the K1 rule (the README) at 1024, 2048 and 4096 tokens
#: on two parts of each GPU (0.70 to 0.79 times the deterministic program's time in CUDA graph replays), and at 6144
#: tokens on two sm_103a parts; on the sm_100a part measured at 6144 tokens its time next to a kernel holding SMs on
#: another stream was 1.099 times the deterministic program's, and at 512 and 8192 tokens it failed on every part.
#: 148-SM parts keep the deterministic program.  On 212-SM sm_107a parts the same row (no verdict in the part's table of
#: :data:`K1_SPLIT_RULES_BY_SM` either) runs that program from 1024 to 7680 tokens: its output was the deterministic
#: program's bit for bit at an even block count (5 seeds at 1024, 2048, 4096, 6144, 6656, 7168 and 7680 tokens), at an
#: odd count (1088 to 5000 tokens) its FP64 error was 1.11 to 1.16 times the deterministic program's; it passed the K1
#: rule there on two nodes of each of two R200 systems (0.70 to 0.76 times the deterministic program's time; 0.48 to
#: 0.51 times at 1088 tokens), and failed at 512, 768 and 896 tokens; from 8000 tokens the deterministic plan changes
#: and the two-split error was 1.54 times its error, and at 12288 and 16384 tokens it was 5 to 7 % slower.  On 212 SMs
#: the FP32 rows ``4096 / 5120 -> 512 / 1024`` run two splits of their 212-SM tile width (128, 192 for ``4096 -> 1024``,
#: 256 for ``5120 -> 1024``).  From 1024 tokens up, at an even block count, their output was the deterministic program's
#: bit for bit (5 seeds on two parts at 1024, 2048, 4096, 8192, 16231, 16384, 32768 and 65536 tokens, on one part at
#: 6000, 12000 and 48000); at an odd count from 17 blocks up it was not, with an FP64 relative Frobenius error 1.01 to
#: 1.17 times the deterministic program's (5 seeds, 1025 to 65537 tokens; at most 3.8e-5 at 65537 tokens).  Measured as
#: the ranges of :data:`K1_SPLIT_RULES_BY_SM_TOKEN_RANGES`, at 512, 1024, 2048, 4096, 8192, 16231, 16384, 32768 and
#: 65536 tokens (the protocol and the rule are in the 212-SM K1 token-range table of the README): a range runs between
#: measured token counts at which the two-split program passed, and holds none that failed; the token counts between two
#: measured ones inside it take its verdict at either block parity (not measured).  At 512 tokens two splits are not
#: faster than the deterministic plan, ``4096 -> 1024`` is not faster in every run at 1024 tokens, and ``5120 -> 1024``
#: failed the rule at every measured token count from 16231 up.  The ranges of these four rows below 16172 tokens and
#: from 32768 to 40961 tokens run the three-chain program of :data:`K1_CHAIN_RANGES_BY_SM` instead.
# fmt: off
K1_TWO_SPLIT_RANGES_BY_SM: dict[
    tuple[str, int],
    dict[tuple[str, int, int, int, str], tuple[tuple[int, int, tuple[int, int]], ...]],
] = {
    ("sm_100a", 152): {
        ("sm_100a", 1, 4096, 128, "f32"): ((1024, 4096, (64, 2)),),
    },
    ("sm_103a", 152): {
        ("sm_103a", 1, 4096, 128, "f32"): ((1024, 6144, (64, 2)),),
    },
    ("sm_107a", 212): {
        ("sm_107a", 1, 4096, 128, "f32"): ((1024, 7680, (64, 2)),),
        ("sm_107a", 1, 4096, 512, "f32"): ((1024, 8192, (128, 2)), (32768, 65536, (128, 2))),
        ("sm_107a", 1, 4096, 1024, "f32"): ((2048, 8192, (192, 2)), (32768, 65536, (192, 2))),
        ("sm_107a", 1, 5120, 512, "f32"): ((1024, 8192, (128, 2)), (32768, 65536, (128, 2))),
        ("sm_107a", 1, 5120, 1024, "f32"): ((1024, 8192, (256, 2)),),
    },
}
# fmt: on
#: Verdicts that replace a row's :data:`K1_SPLIT_RULES` verdict on the parts of exactly one SM count from the row's
#: SM-count floor up, ``(arch, sm_count) -> {key: (block_n, num_splits) or None}`` (keys and values as in
#: :data:`K1_SPLIT_RULES`, measured on parts of exactly that SM count under the same rule, in two jobs on different
#: parts), at the token counts of :data:`K1_SPLIT_RULES_BY_SM_TOKENS` only, except the token counts of
#: :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS` for the part.  Every other token count, every other SM count and every row
#: without an entry keep the verdict of :data:`K1_SPLIT_RULES`.  On 216 SMs ``4096 -> 1024`` FP32 takes the 192-wide
#: row-major program at one split (88 tiles of 256 ``d_out`` rows by 192 ``d_in`` columns, 176 CTAs, one wave next to a
#: kernel holding 32 SMs), which keeps more SMs busy than the 128 CTAs of the 256-wide transposed tile; at 16231 and
#: 16384 tokens the row runs the deterministic program (its entries in :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS`; 16384: a
#: strict re-measurement on a 216-SM part), and at 16172 tokens the three-chain program of
#: :data:`K1_CHAIN_RANGES_BY_SM`.
# fmt: off
K1_SPLIT_OVERRIDES_BY_SM: dict[
    tuple[str, int], dict[tuple[str, int, int, int, str], Optional[tuple[int, int]]]
] = {
    ("sm_107a", 216): {
        ("sm_107a", 1, 4096, 1024, "f32"): (192, 1),
    },
}
# fmt: on
#: Token ranges at which a row runs this package's one-split program with several accumulation chains on the parts of
#: exactly one SM count, ``(arch, sm_count) -> {key: ((min_tokens, max_tokens, (block_n, chains)), ...)}`` (keys as in
#: :data:`K1_SPLIT_RULES`): a token count from ``min_tokens`` to ``max_tokens`` runs the program of that tile width at
#: one split with that many chains (:func:`k1_chain_plan`) in place of the routing of the tables above (the one-chain
#: program of :data:`K1_SPLIT_RULES` and of the 212-SM table of :data:`K1_SPLIT_RULES_BY_SM`, and the two splits of
#: :data:`K1_TWO_SPLIT_RANGES_BY_SM`): at the even block counts of the two-split ranges both programs were timed at on
#: 212 SMs (1024 to 8192 and 32768 to 40960 tokens), three chains ran the weight gradient 7 to 52 % and the checkpointed
#: step 2 to 21 % faster than two splits (medians, eager and graph), with an FP64 error below the deterministic
#: program's, which two splits match bit for bit.  The FP32 rows ``4096 / 5120 -> 512 / 1024``: from 961 tokens up their
#: one-chain program adds an output's whole token reduction in one chain, about twice the deterministic program's FP64
#: error (reported); three chains of a third of the K blocks each had an FP64 relative Frobenius error 0.31 to 0.73
#: times the deterministic program's on every seed (5 seeds on a 212-SM R200 part at 960, 961, 1024, 2048, 4096, 8192,
#: 16172, 16231, 16384, 32768 and 40961 tokens, even and odd block counts; the error does not depend on the SM count)
#: and ran within 5 % of the one-chain program.  Measured as the ranges of :data:`K1_SPLIT_RULES_BY_SM_TOKEN_RANGES`
#: (the rule, the bound on the backward pass and the checkpointed step, and the construction of a range are that
#: table's): on 212 SMs in two jobs on four parts of one node each, at 960, 4096, 16172, 16384, 32768 and 40961 tokens
#: and at 960, 1024, 2048, 4096, 4097, 8192, 16172, 16384, 32768, 32769, 40960 and 40961 tokens; on 216 SMs at 960, 961,
#: 1024, 2048, 4096, 8192, 16172, 16231, 16384, 32768 and 40961 tokens on one part.  Up to 960 tokens on 216 SMs the
#: one-chain program keeps its :data:`K1_SPLIT_RULES` verdict (the deterministic program adds one chain there too).
#: ``4096 -> 512`` and ``5120 -> 512`` are not faster than the deterministic plan in every run at 16384 tokens on 212
#: SMs, nor are ``4096 -> 512`` and ``4096 -> 1024`` at 16231 and 16384 tokens on 216 SMs; ``4096 -> 1024`` at 16384
#: tokens on 212 SMs (in one of the two jobs) and ``5120 -> 1024`` at 32768 tokens on 216 SMs left the checkpointed step
#: more than 1.3 % slower than the deterministic program.  Outside these ranges the rows keep the routing of the tables
#: above (on 216 SMs the one-chain program up to 40960 tokens, on more SMs up to 960 tokens, except the token counts
#: of :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS`).  From 961 tokens the ranges of :data:`K1_PARITY_RANGES_BY_SM` come first
#: for these rows on both SM counts (three chains missed the accuracy check on mirrored cancelling inputs): the three
#: chains of this table run only at 960 tokens on 212 SMs.
# fmt: off
K1_CHAIN_RANGES_BY_SM: dict[
    tuple[str, int],
    dict[tuple[str, int, int, int, str], tuple[tuple[int, int, tuple[int, int]], ...]],
] = {
    ("sm_107a", 212): {
        ("sm_107a", 1, 4096, 512, "f32"): ((960, 16172, (128, 3)), (32768, 40961, (128, 3))),
        ("sm_107a", 1, 4096, 1024, "f32"): ((960, 16172, (192, 3)), (32768, 40961, (192, 3))),
        ("sm_107a", 1, 5120, 512, "f32"): ((960, 16172, (128, 3)), (32768, 40961, (128, 3))),
        ("sm_107a", 1, 5120, 1024, "f32"): ((960, 40961, (256, 3)),),
    },
    ("sm_107a", 216): {
        ("sm_107a", 1, 4096, 512, "f32"): ((961, 16172, (128, 3)), (32768, 40961, (128, 3))),
        ("sm_107a", 1, 4096, 1024, "f32"): ((961, 16172, (192, 3)), (32768, 40961, (192, 3))),
        ("sm_107a", 1, 5120, 512, "f32"): ((961, 40961, (128, 3)),),
        ("sm_107a", 1, 5120, 1024, "f32"): ((961, 16384, (256, 3)), (40961, 40961, (256, 3))),
    },
}
# fmt: on
#: Token ranges at which a row runs this package's one-split program of one accumulation chain on the parts of exactly
#: one SM count, ``(arch, sm_count) -> {key: ((min_tokens, max_tokens, block_n), ...)}`` (keys as in
#: :data:`K1_SPLIT_RULES`): a token count from ``min_tokens`` to ``max_tokens`` runs the transposed-store program of
#: that tile width at one split, at an even and at an odd number of 64-element K blocks alike, in place of the routing
#: of every table above except :data:`K1_CHAIN_RANGES_BY_SM` (the row's verdict, its range of
#: :data:`K1_TWO_SPLIT_RANGES_BY_SM` and its entries in :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS` and
#: :data:`K1_SPLIT_ODD_BLOCK_EXCLUDED_TOKENS`); every other token count, part and row keeps that routing.  Each CTA
#: pair adds its tile's whole token reduction in one accumulator and stores the tile once (no workspace, no ticket).
#: On 152-SM sm_100a and sm_103a parts the FP32 rows ``4096 -> 32`` and ``4096 -> 128`` run 64-wide tiles at one split
#: (16 and 32 tiles) where that routing runs two splits of 64-wide tiles or the deterministic program.  The ranges were
#: measured against that routing (and against the deterministic program at the odd block counts from 1025 tokens) on
#: a GB200 and a GB300 part, at 256 to 4096 tokens, under the rule of :data:`K1_SPLIT_RULES` with cold
#: L2 and the GPU span of the operation (CUPTI): in every run faster in eager launches and in CUDA graph replays, in
#: both launch orders, in one process for all token counts and in a fresh process per token count, no more than 3 %
#: slower next to a kernel holding 8, 16 or 32 SMs, and the backward pass and the checkpointed step no more than 1.3 %
#: slower.  A range runs from its first measured token count to the last one at which the program passed, and holds no
#: measured token count that failed; the token counts between two consecutive measured ones take it too (not
#: measured).  Below 256 tokens (not measured), from the first failing token count up (``4096 -> 32``: 2304 on both
#: GPUs, and 2049 against the two-split program; ``4096 -> 128``: 3584 on sm_100a, 3072 on sm_103a) and on parts of
#: any other SM count the routing of the tables above holds.  The output met the deterministic package's FP32 budget
#: against FP64 (relative Frobenius error at most 5e-5, max error at most 1e-3 times the largest reference element) in
#: every one of 20 runs of 10 seeds at every measured token count of the ranges (worst relative Frobenius error
#: 3.54e-6); from 1024 tokens up, where the deterministic program adds the halves of an output's token reduction in two
#: partials, that error was 2.10 to 2.38 times the deterministic program's (reported, not a rule), and at 256 to 800
#: tokens the same.  The protocol, the measured token counts and the times are in the 152-SM one-split table of the
#: README.  The 148-SM ``sm_100a`` row ``4096 -> 1024`` BF16 runs one split of two chains of 256-wide tiles there
#: instead (:data:`K1_ORDERED_RANGES_BY_SM`): one chain missed the accuracy check on mirrored cancelling inputs.
# fmt: off
K1_ONE_SPLIT_RANGES_BY_SM: dict[
    tuple[str, int], dict[tuple[str, int, int, int, str], tuple[tuple[int, int, int], ...]]
] = {
    ("sm_100a", 152): {
        ("sm_100a", 1, 4096, 32, "f32"): ((256, 2048, 64),),
        ("sm_100a", 1, 4096, 128, "f32"): ((256, 3328, 64),),
    },
    ("sm_103a", 152): {
        ("sm_103a", 1, 4096, 32, "f32"): ((256, 2048, 64),),
        ("sm_103a", 1, 4096, 128, "f32"): ((256, 2816, 64),),
    },
}
# fmt: on
#: Token ranges at which a row runs this package's cluster program on the parts of exactly one SM count,
#: ``(arch, sm_count) -> {key: ((min_tokens, max_tokens, (block_n, splits)), ...)}`` (keys as in
#: :data:`K1_SPLIT_RULES`; FP32 rows only): a token count from ``min_tokens`` to ``max_tokens`` runs the program
#: :func:`dsm_k1_template` names -- ``L x m_tiles x n_tiles`` tiles of 128 ``d_in`` columns by ``block_n`` ``d_out``
#: rows, each tile one cluster of ``splits`` CTAs, CTA ``s`` taking the ``s``-th of ``splits`` contiguous runs of the
#: tile's 64-element K blocks -- in place of the routing of every other K1 table (:func:`k1_dsm_plan`, before
#: :func:`k1_chain_plan` and the one-split ranges); every other token count, part and row keeps that routing.  The
#: CTAs of a cluster meet in its distributed shared memory: each sends the 4-column quads of its partial tile that
#: another rank owns to that rank (``st.async``, counted on the owner's barrier), adds the ``splits`` partials of its
#: own quads in cluster-rank order and stores them once, and no CTA leaves before every CTA of the cluster received
#: its quads.  No workspace, no ticket, no atomic: the same bits on every run.  On 152-SM sm_100a parts the FP32 rows
#: ``4096`` / ``5120 -> 32``, ``48``, ``64`` and ``128`` run it in the ranges below, where the other tables run one or
#: two splits of 64-wide tiles of the CTA-pair program or the deterministic program.  A measured token count's program
#: is the fastest of the tile widths and cluster sizes that passed a first screen there (two GB200 parts), and a token
#: count between two consecutive screened ones takes the lower one's program where that program passed at both, else
#: the upper one's.  The ranges were then measured with this table, on two GB200 parts, at 1024, 2048, 3072, 4096,
#: 6144, 8192, 12288, 16384, 24576 and 32768 tokens against that routing (the plan this module gives the part without
#: the table), with cold L2 and the GPU span (CUPTI), in one process per GPU pinned to its own cores, both launch
#: orders and two runs per launch order on each part.  A measured token count passed where, on both parts, the
#: operation alone was faster in every run (paired median below 0.99, sign test), no more than 3 % slower than that
#: routing next to a kernel holding 8, 16 or 32 SMs on another stream, and the training step (forward, input and
#: weight gradient) and the step with the forward recomputed were, in CUDA graphs and launched eagerly, each no slower
#: than that routing (median of the runs at most 1.013 times, no run slower than the slowest run of the routing
#: against itself).  A row's ranges run from a passing measured token count to a passing measured token count and
#: hold no measured token count that failed; every other token count, part and row keeps the routing of the other
#: tables (``4096 -> 32`` from 2049 to 6143 tokens, ``4096 -> 48`` from 2049 to 4095 and from 8193 to 16171 tokens,
#: ``4096 -> 128`` below 4096 and above 8192 tokens, ``5120 -> 64`` from 2049 to 4095 and above 24576 tokens,
#: ``5120 -> 48`` above 24576 tokens, ``5120 -> 128`` above 6144 tokens, every row below 1024 and above 32768 tokens).
#: On 152-SM ``sm_103a`` parts (two GB300 parts) and 148-SM ``sm_100a`` parts (two B200 parts) the same rows run it in
#: the ranges of their tables.  There a measured token count's program is the fastest of the six programs on a first
#: screen of both parts at 1024, 2048, 4096, 6144, 8192, 12288, 16384, 24576 and 32768 tokens, and the ranges were
#: measured with the table at those token counts with the protocol above, except that the time next to a kernel holding
#: SMs was recorded, not required.  Where two consecutive measured token counts of a range take different programs, the
#: lower one's runs up to the token count before the upper one.  Not in a range (the routing of the other tables
#: holds): on ``sm_103a`` ``4096 -> 48`` from 6145 to 12287 tokens, ``4096 -> 128`` above 12288 tokens, ``5120 -> 32``
#: from 2049 to 6143 tokens, ``5120 -> 48`` above 8192 tokens and ``5120 -> 64`` above 24576 tokens; on 148-SM
#: ``sm_100a`` ``4096 -> 48`` from 16385 to 32767 tokens, ``4096 -> 64`` from 6145 to 12287 tokens and ``4096 -> 128``
#: above 12288 tokens; every row below 1024 and above 32768 tokens.  On every part the output met the deterministic
#: package's FP32 budget against FP64 (relative Frobenius error at most 5e-5, max error at most 1e-3 times the largest
#: reference element) in every one of 20 runs of 10 seeds at every measured token count.  The protocol, the measured
#: token counts and the times are in the cluster tables of the README.
#:
#: Accuracy against the reference implementation (FlashInfer commit ``2e989556``): a cluster of three CTAs sums a
#: tile's token reduction in three partials in rank order, so it repeats bit for bit, but on the rows whose reference
#: adds the two halves of that reduction (``4096 -> 32 / 48 / 64 / 128`` and ``5120 -> 32 / 48``) it does not cancel a
#: mirrored input exactly: its relative error on such an input was 5.8 to 8.7 times the reference's median at 16231
#: and 16384 tokens on GB200, GB300 and B200 parts.  A cluster of two CTAs (the halves) stayed within 1.1 times the
#: reference's median error on all six input kinds of that check (random normal, long sums, mirrored and permuted
#: cancellation, concentrated values, values near the largest magnitude) there, 0.89 to 1.00 times the deterministic
#: plan's time.  So those rows run two CTAs in every range where they ran three.  On 152-SM ``sm_103a`` parts the
#: reference cuts the ``5120 -> x`` reduction at 135 / 136 of 256 K blocks, not at the half, and a two-CTA split
#: failed the check at 16384 tokens (``5120 -> 64``: largest error 1.14 times on random normal inputs; ``5120 -> 32``:
#: 1.15 times on values near the largest magnitude): their ranges of three CTAs and that token count run the
#: deterministic program (:data:`K1_SPLIT_RULE_EXCLUDED_TOKENS_BY_SM`).  The 148-SM ``sm_100a`` ranges of three CTAs of
#: ``5120 -> 64`` and ``5120 -> 128`` stay: there the reference sums the reduction in one chain, and ``5120 -> 128`` on
#: three CTAs met the check on all six kinds at 16231 and 16384 tokens (0.36 to 0.40 times the reference's error).
#:
#: On 148-SM and 152-SM ``sm_100a`` parts and 152-SM ``sm_103a`` parts ``4096 -> 48`` and ``4096 -> 64`` run two CTAs of
#: 32-wide tiles from 16172 to 16384 tokens.  At an even number of 64-token K blocks (16231 and 16384 tokens) they sum
#: the same two halves as the reference (its error exactly, on all six kinds) and were the fastest program that met the
#: check: 0.98 to 0.99 times the time of two CTAs of 64-wide tiles on a B200 part and 0.94 times on a GB200 and a GB300
#: part.  At an odd block count every two-way split missed the check; those token counts run the program of
#: :data:`K1_DSM_ODD_BLOCK_RANGES_BY_SM`.
#:
#: On 212-SM ``sm_107a`` parts ``4096 -> 48`` and ``4096 -> 64`` run four CTAs of 64-wide tiles from 16172 to 16384
#: tokens, at an even and at an odd block count: on a R200 part it met the check on all six kinds at 16172, 16231,
#: 16300 and 16384 tokens (its worst relative error 0.55 to 0.58 times the reference's median), the same bits on every
#: run, at 0.84 to 0.87 times the time of four splits of the 64-wide CTA-pair program (the program these rows ran
#: before, :data:`K1_ORDERED_RANGES_BY_SM`), 0.91 times that of two CTAs of 32-wide tiles at an even block count and
#: 0.68 to 0.69 times the deterministic program's at an odd one.
# fmt: off
K1_DSM_RANGES_BY_SM: dict[
    tuple[str, int],
    dict[tuple[str, int, int, int, str], tuple[tuple[int, int, tuple[int, int]], ...]],
] = {
    ("sm_100a", 148): {
        ("sm_100a", 1, 4096, 32, "f32"): ((1024, 32768, (32, 2)),),
        ("sm_100a", 1, 4096, 48, "f32"): (
            (1024, 4095, (32, 2)), (4096, 16171, (64, 2)), (16172, 16384, (32, 2)), (32768, 32768, (64, 2)),
        ),
        ("sm_100a", 1, 4096, 64, "f32"): (
            (1024, 4095, (32, 2)), (4096, 6144, (64, 2)), (12288, 16171, (64, 2)), (16172, 16384, (32, 2)),
            (16385, 32768, (64, 2)),
        ),
        ("sm_100a", 1, 4096, 128, "f32"): ((1024, 12288, (64, 2)),),
        ("sm_100a", 1, 5120, 32, "f32"): ((1024, 32768, (32, 2)),),
        ("sm_100a", 1, 5120, 48, "f32"): ((1024, 32768, (64, 2)),),
        ("sm_100a", 1, 5120, 64, "f32"): ((1024, 2047, (64, 3)), (2048, 32768, (64, 2))),
        ("sm_100a", 1, 5120, 128, "f32"): ((1024, 4095, (128, 2)), (4096, 32768, (128, 3))),
    },
    ("sm_100a", 152): {
        ("sm_100a", 1, 4096, 32, "f32"): ((1024, 2048, (32, 2)), (6144, 32768, (32, 2))),
        ("sm_100a", 1, 4096, 48, "f32"): (
            (1024, 2048, (64, 2)), (4096, 8192, (64, 2)), (16172, 16384, (32, 2)), (16385, 32768, (64, 2)),
        ),
        ("sm_100a", 1, 4096, 64, "f32"): ((1024, 16171, (64, 2)), (16172, 16384, (32, 2)), (16385, 32768, (64, 2))),
        ("sm_100a", 1, 4096, 128, "f32"): ((4096, 6144, (64, 2)), (6145, 8192, (128, 2))),
        ("sm_100a", 1, 5120, 32, "f32"): ((1024, 32768, (32, 2)),),
        ("sm_100a", 1, 5120, 48, "f32"): ((1024, 24576, (64, 2)),),
        ("sm_100a", 1, 5120, 64, "f32"): ((1024, 2048, (64, 2)), (4096, 24576, (64, 2))),
        ("sm_100a", 1, 5120, 128, "f32"): ((1024, 6144, (128, 2)),),
    },
    ("sm_103a", 152): {
        ("sm_103a", 1, 4096, 32, "f32"): ((1024, 32768, (32, 2)),),
        ("sm_103a", 1, 4096, 48, "f32"): (
            (1024, 4095, (32, 2)), (4096, 6144, (64, 2)), (12288, 16171, (64, 2)), (16172, 16384, (32, 2)),
            (16385, 32768, (64, 2)),
        ),
        ("sm_103a", 1, 4096, 64, "f32"): (
            (1024, 2047, (32, 2)), (2048, 8191, (64, 2)), (8192, 12287, (32, 2)), (12288, 16171, (64, 2)),
            (16172, 16384, (32, 2)), (16385, 32768, (64, 2)),
        ),
        ("sm_103a", 1, 4096, 128, "f32"): ((1024, 12288, (64, 2)),),
        ("sm_103a", 1, 5120, 32, "f32"): ((1024, 2048, (32, 2)), (6144, 8191, (32, 2)), (12288, 16383, (32, 2)), (16385, 32768, (32, 2))),
        ("sm_103a", 1, 5120, 48, "f32"): ((2048, 4095, (64, 2)),),
        ("sm_103a", 1, 5120, 64, "f32"): ((1024, 4095, (64, 2)), (6144, 12287, (64, 2))),
        ("sm_103a", 1, 5120, 128, "f32"): ((1024, 4095, (128, 2)),),
    },
    ("sm_107a", 212): {
        ("sm_107a", 1, 4096, 48, "f32"): ((16172, 16384, (64, 4)),),
        ("sm_107a", 1, 4096, 64, "f32"): ((16172, 16384, (64, 4)),),
    },
}
# fmt: on
#: The programs that rows of :data:`K1_DSM_RANGES_BY_SM` run in place of their cluster program at a token count that
#: spans an odd number of 64-element K blocks (``ceil(K / 64)`` odd) inside a range of this table, on the parts of
#: exactly one SM count, ``(arch, sm_count) -> {key: ((min_tokens, max_tokens, program), ...)}`` (keys as in
#: :data:`K1_SPLIT_RULES`; ``program`` ``None`` for the deterministic program, ``(block_n, splits)`` for the cluster
#: program of that tile as in :data:`K1_DSM_RANGES_BY_SM`, or ``(block_n, splits, chains, order)`` of this package's
#: CTA-pair program as in :data:`K1_ORDERED_RANGES_BY_SM`).  An even block count, a token count outside the ranges and
#: a token count outside the row's cluster ranges keep the routing of the other tables.  The deterministic program of
#: these rows cuts the token reduction of the tiles, taken in order, into units of ``ceil(blocks / 2)`` K blocks.  At an
#: even block count a tile's two units are the halves that two CTAs of the cluster program sum; at an odd one a unit
#: runs on into the next tile, so 31 of the 32 tiles sum three shorter partials.  There every two-way split (two CTAs
#: of 32-wide or 64-wide tiles, two splits of the CTA-pair program) missed the check of :data:`K1_DSM_RANGES_BY_SM`:
#: its worst relative error was 1.16 times the reference's median at 16130, 16172, 16300 and 16320 tokens on a B200
#: part (at 16172 tokens also on a GB200 and a GB300 part), and three CTAs of 64-wide tiles missed it on mirrored
#: cancelling inputs.  A cluster of four CTAs of 64-wide tiles (its four partials added in cluster-rank order in shared
#: memory, so the same bits on every run) met the check on all six kinds at 16172 and 16300 tokens on a B200, a GB200
#: and a GB300 part (its worst relative error 0.60 to 0.64 times the reference's median) and was the fastest program
#: that met it: 0.89 to 0.93 times the time of four splits of the 64-wide CTA-pair program and 0.83 to 0.93 times the
#: deterministic program's there.  Both rows run it from 16172 to 16384 tokens.
# fmt: off
K1_DSM_ODD_BLOCK_RANGES_BY_SM: dict[
    tuple[str, int],
    dict[
        tuple[str, int, int, int, str],
        tuple[tuple[int, int, Optional[Union[tuple[int, int], tuple[int, int, int, str]]]], ...],
    ],
] = {
    ("sm_100a", 148): {
        ("sm_100a", 1, 4096, 48, "f32"): ((16172, 16384, (64, 4)),),
        ("sm_100a", 1, 4096, 64, "f32"): ((16172, 16384, (64, 4)),),
    },
    ("sm_100a", 152): {
        ("sm_100a", 1, 4096, 48, "f32"): ((16172, 16384, (64, 4)),),
        ("sm_100a", 1, 4096, 64, "f32"): ((16172, 16384, (64, 4)),),
    },
    ("sm_103a", 152): {
        ("sm_103a", 1, 4096, 48, "f32"): ((16172, 16384, (64, 4)),),
        ("sm_103a", 1, 4096, 64, "f32"): ((16172, 16384, (64, 4)),),
    },
}
# fmt: on


#: The orders in which the last split of a CTA-pair K1 program adds the ``num_splits`` partial rows of a tile:
#: ``"arrival"`` its own first, then the others from the split after its own (the order the splits finish, so with three
#: or more splits the bits change from run to run), ``"mirror"`` each split ``j`` of the first half with its mirror
#: ``j + num_splits // 2`` first and then the pair sums in split order (``((p_0 + p_h) + (p_1 + p_{h+1})) + ...``,
#: the last split of an odd count at the end): the same bits on every run.
K1_SPLIT_ORDERS: tuple[str, ...] = ("arrival", "mirror")
#: Token ranges at which a row runs a given program of this package's CTA pairs, split order included, on the parts of
#: exactly one SM count, ``(arch, sm_count) -> {key: ((min_tokens, max_tokens, (block_n, splits, chains, order)),
#: ...)}`` (keys as in :data:`K1_SPLIT_RULES`; ``order`` one of :data:`K1_SPLIT_ORDERS`; more than one chain at one
#: split only): a token count from ``min_tokens`` to ``max_tokens`` runs that program (:func:`k1_ordered_plan`) in place
#: of the routing of every other K1 table except :data:`K1_DSM_RANGES_BY_SM`; every other token count, part and row
#: keeps that routing.  These rows' programs of the other tables were either slower, or missed the accuracy check of
#: :data:`K1_DSM_RANGES_BY_SM` (within 1.1 times the reference's median error on six input kinds) on mirrored cancelling
#: inputs, and these programs met it on all six kinds (on a 212-SM R200 part; the 148-SM row below on a B200 part).  On
#: 212-SM sm_107a parts the thin FP32 weight gradient ``4096 -> 32`` (16172 to 16384 tokens) runs four splits of 64-wide
#: tiles in the mirror order (the same bits on every repeat; three splits in the arrival order and three CTAs of the
#: cluster program missed the check there): its worst kind at most 0.70 times the reference's error, about 2 % slower
#: than four splits in the arrival order (whose bits vary run to run) and 0.97 to 0.98 times the deterministic program's
#: time.  ``4096 -> 48`` and ``4096 -> 64`` take the cluster program of :data:`K1_DSM_RANGES_BY_SM` there.  The FP32
#: weight gradients ``4096 -> 1024`` (the 192-wide row-major program) and ``5120 -> 1024`` (the 256-wide
#: transposed-store program) run one split with the chain count of the parity of their 64-token K blocks from 16172 to
#: 16384 tokens: four chains at an odd count (253 and 255 blocks), two at an even count (254 and 256 blocks; two chains
#: are bit for bit two splits of one chain); ``5120 -> 512`` (the 128-wide transposed-store program) runs two chains at
#: the even counts there.  At an odd count most of the reference's tiles add three K segments: two partials lost about
#: 15 % on the long sums and three equal partials (the three-chain program of :data:`K1_CHAIN_RANGES_BY_SM`) missed the
#: mirrored cancelling inputs, and four chains met the check.  Measured on a 212-SM R200 part at 16172, 16231 and 16384
#: tokens: every kind within the check (the worst kind of four chains 0.79 times the reference's error at
#: ``4096 -> 1024`` and 0.60 times at ``5120 -> 1024``), the same bits on 200 repeats; four chains took 1.005 to 1.010
#: times the three-chain program's time, two chains 0.999 to 1.000 times the time of the former route of their row
#: (three chains, or the deterministic program at ``5120 -> 512``).  On 148-SM sm_100a parts the BF16 weight gradient
#: ``4096 -> 1024`` runs one split of two chains of 256-wide tiles from 16172 to 16384 tokens: 64 tiles, one wave of 128
#: CTAs, and the bits of two splits of one chain (the same two K ranges, one addition), the same on every run.  Its
#: former route, one split of one chain, missed the check on mirrored cancelling inputs at every token count of that
#: window (1.87 to 1.95 times the reference's error, and 1.38 to 1.39 times on the permuted ones); two splits met all
#: six kinds there (at most 1.08 times the reference's error; the reference's own error at the even block counts) but
#: took 1.108 to 1.116 times one split's time, in a second wave of CTAs.  Measured on a B200 part at 16172, 16231,
#: 16256, 16300 and 16384 tokens in one process: two chains had the bits of two splits on all six kinds and repeated
#: bit for bit; they took 86.5 to 88.9 us, 1.003 to 1.005 times one chain's time, 0.897 to 0.906 times two splits'
#: (96.0 to 98.4 us) and 0.74 to 0.84 times the deterministic program's (103.9 to 120.0 us), in eager launches and in
#: CUDA graph replays.  From 961 to 16171 tokens (16 K blocks up), where the row ran the deterministic program, it runs
#: the same two chains: on a B200 part at 961 to 16171 tokens (10 token counts) every kind within the check (at most
#: 1.006 times the reference's error), the same bits on 20 and 200 repeats and, at an even number of K blocks, the
#: deterministic program's bits; on four B200 parts, 17 paired groups in eager launches and in CUDA graph replays each,
#: at 1024 to 16171 tokens (8 token counts), 12.1 to 87.3 us, 0.42 to 0.84 times the deterministic program's time.  The
#: 212-SM ranges of the FP32 rows ``4096 -> 1024``, ``5120 -> 512`` and ``5120 -> 1024`` lie inside ranges of
#: :data:`K1_PARITY_RANGES_BY_SM`, which comes first and runs the same programs there, except eight chains at the odd
#: counts of ``5120 -> 1024`` (and four chains at the odd counts of ``5120 -> 512``).
# fmt: off
K1_ORDERED_RANGES_BY_SM: dict[
    tuple[str, int],
    dict[tuple[str, int, int, int, str], tuple[tuple[int, int, tuple[int, int, int, str]], ...]],
] = {
    ("sm_100a", 148): {
        ("sm_100a", 1, 4096, 1024, "bf16"): ((961, 16384, (256, 1, 2, "arrival")),),
    },
    ("sm_107a", 212): {
        ("sm_107a", 1, 4096, 32, "f32"): ((16172, 16384, (64, 4, 1, "mirror")),),
        ("sm_107a", 1, 4096, 1024, "f32"): (
            (16172, 16192, (192, 1, 4, "arrival")), (16193, 16256, (192, 1, 2, "arrival")),
            (16257, 16320, (192, 1, 4, "arrival")), (16321, 16384, (192, 1, 2, "arrival")),
        ),
        ("sm_107a", 1, 5120, 512, "f32"): (
            (16193, 16256, (128, 1, 2, "arrival")), (16321, 16384, (128, 1, 2, "arrival")),
        ),
        ("sm_107a", 1, 5120, 1024, "f32"): (
            (16172, 16192, (256, 1, 4, "arrival")), (16193, 16256, (256, 1, 2, "arrival")),
            (16257, 16320, (256, 1, 4, "arrival")), (16321, 16384, (256, 1, 2, "arrival")),
        ),
    },
}
# fmt: on


def k1_ordered_plan(
    *,
    arch: str,
    L: int,
    M: int,
    N: int,
    K: int,
    out_f32: bool,
    sm_count: int,
) -> Optional[tuple[int, int, int, str]]:
    """``(block_n, splits, chains, order)`` of the program the row runs at ``K`` on a device of exactly
    ``sm_count`` SMs (its range in :data:`K1_ORDERED_RANGES_BY_SM`), or ``None``: the row keeps the routing of
    :func:`k1_plan` -- outside those ranges, or at a ``K`` too short for the splits or chains (:func:`k1_splits_fit`,
    :func:`k1_chains_fit`).  ``M``, ``N`` and ``K`` as in :func:`k1_plan`."""
    key = (arch, int(L), int(M), int(N), "f32" if out_f32 else "bf16")
    for lo, hi, (block_n, splits, chains, order) in K1_ORDERED_RANGES_BY_SM.get(
        (arch, int(sm_count)), {}
    ).get(key, ()):
        if lo <= int(K) <= hi:
            fits = k1_splits_fit(splits, K) and k1_chains_fit(chains, K)
            return (
                (int(block_n), int(splits), int(chains), str(order)) if fits else None
            )
    return None


#: Token ranges at which a row runs a program chosen by the parity of its number of 64-token K blocks, on the parts of
#: exactly one SM count, ``(arch, sm_count) -> {key: ((min_tokens, max_tokens, even, odd), ...)}`` (keys as in
#: :data:`K1_SPLIT_RULES`): a token count from ``min_tokens`` to ``max_tokens`` runs ``even`` at an even number of K
#: blocks and ``odd`` at an odd number (:func:`k1_parity_plan`), each a ``(block_n, splits, chains, order)`` of this
#: package's CTA-pair program as in :data:`K1_ORDERED_RANGES_BY_SM`, ``()`` for the deterministic program, or ``None``
#: for the routing of the other tables.  A part of more SMs than every SM count of the table for its architecture takes
#: the ranges of the SM count of :data:`K1_PARITY_RANGES_ABOVE_SM`.  It comes before every K1 table except the cluster
#: tables of :data:`K1_DSM_RANGES_BY_SM`: on 212-SM and 216-SM ``sm_107a`` parts the ranges of
#: :data:`K1_ORDERED_RANGES_BY_SM` and :data:`K1_CHAIN_RANGES_BY_SM` of these rows hold only where this table has none
#: (960 tokens on 212 SMs), and on 208-SM parts the window verdicts of :data:`K1_SPLIT_RULES_BY_SM` (one chain) hold
#: nowhere for these rows.
#: The FP32 weight gradients ``4096 / 5120 -> 512 / 1024`` from 961 tokens: two chains of one split at an even count
#: (the bits of two splits of one chain), four chains at an odd count (eight for ``5120 -> 1024``), where three chains
#: (equal thirds of the K blocks) missed the accuracy check on mirrored cancelling inputs and one chain (an output's
#: whole token reduction in one accumulator) missed it on all six input kinds; the 512-feature rows on the 128-wide
#: transposed-store program, ``4096 -> 1024`` on the 192-wide row-major one and ``5120 -> 1024`` on the 256-wide
#: transposed-store one.  Four chains of ``5120 -> 1024`` missed the check by up to 1.27 times on one input kind at 255
#: and 257 K blocks (16300 and 16385 tokens); eight chains meet it at every measured token count (at most 1.00 times,
#: 0.63 and 0.65 times there) at 1.002 to 1.007 times the time of four.
#: ``4096 -> 512`` runs the deterministic program at its even counts from 12001 tokens and at every count from 32768 to
#: 40961 tokens, faster there and within the check; ``5120 -> 512`` runs it from 40962 to 65536 tokens on 208 SMs, where
#: two splits missed the check at the odd counts (212 SMs keep two splits there, :data:`_K1_PARITY_SM107A_212`), and
#: ``4096 -> 1024`` four chains at the odd counts there.  The check is
#: the one of :data:`K1_DSM_RANGES_BY_SM` (at most 1.1 times the reference's median error on each of six input kinds,
#: five runs each), measured on a 212-SM R200 part at the ends of every range, one token on each side of them and the
#: token counts between (22 to 27 per row), the 216-SM ranges as the same plans forced on that part (the accuracy and
#: the bits are the plan's; no 216-SM time is claimed); every program repeated its bits on 20 and 200 runs.  Measured
#: against the routes they replace in the same process (17 paired groups, eager launches and CUDA graph replays): two
#: chains at 1.00 to 1.01 times their time, four chains at 1.00 to 1.005 times at 1024 features.  On 208-SM parts the
#: rows take the 212-SM ranges, except, from 16172 to 16384 tokens (the window of their one-chain verdicts), the token
#: counts at which the two ``4096 ->`` rows ran the deterministic program there (16231 and 16384 tokens), which they
#: keep, and the even counts of ``4096 -> 512``, which run two chains; outside that window the deterministic plan of
#: 208 SMs missed the check (``4096 -> 1024`` at 16171 and 16385 tokens up to 1.38 times the reference's error,
#: ``4096 -> 512`` at 16385 tokens 1.15 times).  Parts of 216 and more SMs take the 216-SM ranges and the 212-SM range
#: of ``4096 -> 1024`` past 40961 tokens (four chains at the odd counts, where the deterministic plan of 224 SMs missed
#: the check, 1.29 times at 40962 tokens); parts of more than 216 SMs ran the deterministic program from 961 tokens up
#: (the 217-SM entries of :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS`), whose ``sm_107a`` program missed the check at some of
#: these token counts on the 212-SM part (``5120 -> 512`` at 20000 and 32000 tokens up to 12 times the reference's
#: error).  The 208-SM and the larger plans were measured forced on the 212-SM part (one split, the chains of one CTA
#: pair: the accumulation order does not depend on the SM count); no time is claimed for them.
_C2_N128, _C4_N128 = (128, 1, 2, "arrival"), (128, 1, 4, "arrival")
_C2_N192, _C4_N192 = (192, 1, 2, "arrival"), (192, 1, 4, "arrival")
_C2_N256, _C8_N256 = (256, 1, 2, "arrival"), (256, 1, 8, "arrival")
_C4_N256 = (256, 1, 4, "arrival")
# fmt: off
_K1_PARITY_SM107A: dict[tuple[int, int], tuple[tuple[int, int, Any, Any], ...]] = {
    (4096, 512): ((961, 12000, _C2_N128, _C4_N128), (12001, 32767, (), _C4_N128), (32768, 40961, (), ())),
    (5120, 512): ((961, 40961, _C2_N128, _C4_N128),),
    (4096, 1024): ((961, 40961, _C2_N192, _C4_N192),),
    (5120, 1024): ((961, 40961, _C2_N256, _C8_N256),),
}
#: The ranges past 40961 tokens (to 65536, where the 212-SM two-split ranges of :data:`K1_TWO_SPLIT_RANGES_BY_SM` end):
#: on 208 SMs both rows, on 216 and more SMs ``4096 -> 1024``; 212 SMs take the ``4096 -> 1024`` one in
#: :data:`_K1_PARITY_SM107A_212`.
_K1_PARITY_SM107A_TAIL: dict[tuple[int, int], tuple[tuple[int, int, Any, Any], ...]] = {
    (5120, 512): ((40962, 65536, (), ()),),
    (4096, 1024): ((40962, 65536, None, _C4_N192),),
}
#: The two ``4096 ->`` rows on 208 SMs: the 212-SM ranges, but from 16172 to 16384 tokens (the window of their
#: one-chain verdicts) the deterministic program at the token counts that ran it there and two chains at every even
#: count of ``4096 -> 512``.
_K1_PARITY_SM107A_208: dict[tuple[int, int], tuple[tuple[int, int, Any, Any], ...]] = {
    (4096, 512): (
        (961, 12000, _C2_N128, _C4_N128), (12001, 16171, (), _C4_N128), (16172, 16230, _C2_N128, _C4_N128),
        (16231, 16231, (), ()), (16232, 16383, _C2_N128, _C4_N128), (16384, 16384, (), ()),
        (16385, 32767, (), _C4_N128), (32768, 40961, (), ()),
    ),
    (4096, 1024): (
        (961, 16230, _C2_N192, _C4_N192), (16231, 16231, (), ()), (16232, 16383, _C2_N192, _C4_N192),
        (16384, 16384, (), ()), (16385, 40961, _C2_N192, _C4_N192),
    ),
}
#: The 212-SM ranges: the ranges above and the ones past 40961 tokens, rechecked where they ran slower than the routing
#: they replaced (this package's A/B against FI main on a 212-SM R200 part) per range and K-block parity against every
#: token count measured on that part under the same accuracy check (FI main's program where it met the check at every
#: measured count of the range, else the fastest program that did; two ranges keep FI main's program although it
#: missed the check at some counts: ``4096 -> 512`` at the odd counts from 24577 to 32767 tokens and ``5120 -> 512`` at
#: the odd counts from 40962 tokens).  ``4096 -> 512`` runs the deterministic program at the odd counts from 24577
#: tokens, FI main's program up to 32767 tokens: it met the check at 24577, 28673, 30721 and 32641 tokens and missed it
#: at 25601 (1.12 times); below 24577 four chains stay (it missed the check at 16385 and 20001 tokens, up to 1.15
#: times).  ``4096 -> 1024`` runs the deterministic program at the even counts from 24576 tokens (FI main's program
#: up to 32767 tokens, where it met the check at every measured even count; FI main's three chains from 32768 tokens
#: missed it; 0.94 to 0.98 times the time of two chains).  ``5120 -> 1024`` runs four chains from 16170 to 16172 tokens
#: (FI main's program at 16172 tokens; they met the check at all three counts and missed it at 16300 and 16385).
#: ``5120 -> 512`` keeps FI main's routing from 40962 tokens (two splits: the bits of the deterministic program at the
#: even counts; at the odd counts they missed the check at 40962 and 57345 tokens, up to 1.83 times, and met it at 49153
#: and 65409).
_K1_PARITY_SM107A_212: dict[tuple[int, int], tuple[tuple[int, int, Any, Any], ...]] = {
    (4096, 512): ((961, 12000, _C2_N128, _C4_N128), (12001, 24576, (), _C4_N128), (24577, 40961, (), ())),
    (5120, 512): ((961, 40961, _C2_N128, _C4_N128),),
    (4096, 1024): ((961, 24575, _C2_N192, _C4_N192), (24576, 40961, (), _C4_N192), (40962, 65536, None, _C4_N192)),
    (5120, 1024): (
        (961, 16169, _C2_N256, _C8_N256), (16170, 16172, _C2_N256, _C4_N256), (16173, 40961, _C2_N256, _C8_N256),
    ),
}
K1_PARITY_RANGES_BY_SM: dict[
    tuple[str, int], dict[tuple[str, int, int, int, str], tuple[tuple[int, int, Any, Any], ...]]
] = {
    ("sm_107a", 208): {
        ("sm_107a", 1, M, N, "f32"): _K1_PARITY_SM107A_208.get((M, N), ranges) + _K1_PARITY_SM107A_TAIL.get((M, N), ())
        for (M, N), ranges in _K1_PARITY_SM107A.items()
    },
    ("sm_107a", 212): {("sm_107a", 1, M, N, "f32"): ranges for (M, N), ranges in _K1_PARITY_SM107A_212.items()},
    ("sm_107a", 216): {
        ("sm_107a", 1, M, N, "f32"): ranges + (_K1_PARITY_SM107A_TAIL[(M, N)] if (M, N) == (4096, 1024) else ())
        for (M, N), ranges in _K1_PARITY_SM107A.items()
    },
}
# fmt: on
#: The SM count whose ranges of :data:`K1_PARITY_RANGES_BY_SM` a part of more SMs takes, ``arch -> sm_count`` (the
#: largest SM count of that architecture in the table).
K1_PARITY_RANGES_ABOVE_SM: dict[str, int] = {"sm_107a": 216}


def k1_parity_plan(
    *,
    arch: str,
    L: int,
    M: int,
    N: int,
    K: int,
    out_f32: bool,
    sm_count: int,
) -> Union[None, tuple[()], tuple[int, int, int, str]]:
    """The program the row runs at ``K`` on a device of ``sm_count`` SMs by the parity of ``K``'s 64-element K blocks
    (its range in :data:`K1_PARITY_RANGES_BY_SM` for exactly ``sm_count``, or, above every SM count of ``arch`` there,
    for :data:`K1_PARITY_RANGES_ABOVE_SM`): ``(block_n, splits, chains, order)``, ``()`` for the deterministic program,
    or ``None``: the row keeps the routing of the other tables -- outside those ranges, at a parity whose entry is
    ``None``, or at a ``K`` too short for the splits or chains (:func:`k1_splits_fit`, :func:`k1_chains_fit`).  ``M``,
    ``N`` and ``K`` as in :func:`k1_plan`."""
    key = (arch, int(L), int(M), int(N), "f32" if out_f32 else "bf16")
    part = K1_PARITY_RANGES_BY_SM.get((arch, int(sm_count)))
    top = K1_PARITY_RANGES_ABOVE_SM.get(arch)
    if part is None and top is not None and int(sm_count) > top:
        part = K1_PARITY_RANGES_BY_SM.get((arch, top))
    for lo, hi, even, odd in (part or {}).get(key, ()):
        if lo <= int(K) <= hi:
            program = odd if _ceil_div(int(K), K1_BLOCK_K) % 2 else even
            if program is None or program == ():
                return program
            block_n, splits, chains, order = program
            if not (k1_splits_fit(splits, K) and k1_chains_fit(chains, K)):
                return None
            return int(block_n), int(splits), int(chains), str(order)
    return None


def k1_splits_fit(splits: int, K: int) -> bool:
    """Whether ``splits`` K ranges each get at least one 64-element K block of ``K`` (``splits * 64 <= K + 63``,
    the program's requirement)."""
    return int(splits) >= 1 and int(splits) * K1_BLOCK_K <= int(K) + K1_BLOCK_K - 1


def k1_chains_fit(chains: int, K: int) -> bool:
    """Whether ``chains`` accumulation chains each get at least one 64-element K block of ``K`` (``chains * 64 <= K +
    63``, the program's requirement)."""
    return int(chains) >= 1 and int(chains) * K1_BLOCK_K <= int(K) + K1_BLOCK_K - 1


def k1_dsm_plan(
    *,
    arch: str,
    L: int,
    M: int,
    N: int,
    K: int,
    out_f32: bool,
    sm_count: int,
) -> Optional[tuple[int, int]]:
    """``(block_n, splits)`` of the cluster program the row runs at ``K`` on a device of exactly ``sm_count`` SMs
    (its range in :data:`K1_DSM_RANGES_BY_SM`), or ``None``: the row keeps the routing of :func:`k1_plan` -- outside
    those ranges, or at a ``K`` with fewer 64-element K blocks than CTAs in the cluster (:func:`k1_splits_fit`).
    ``M``, ``N`` and ``K`` as in :func:`k1_plan`; :func:`plan_atomic_gemm` asks this function before
    :func:`k1_plan`."""
    key = (arch, int(L), int(M), int(N), "f32" if out_f32 else "bf16")
    part = (arch, int(sm_count))
    for lo, hi, (block_n, splits) in K1_DSM_RANGES_BY_SM.get(part, {}).get(key, ()):
        if lo <= int(K) <= hi:
            return (int(block_n), int(splits)) if k1_splits_fit(splits, K) else None
    return None


def k1_dsm_odd_block_plan(
    *,
    arch: str,
    L: int,
    M: int,
    N: int,
    K: int,
    out_f32: bool,
    sm_count: int,
) -> Optional[tuple]:
    """The program a row runs in place of its cluster program (:func:`k1_dsm_plan`) at a ``K`` of an odd number of
    64-element K blocks on a device of exactly ``sm_count`` SMs (its range in :data:`K1_DSM_ODD_BLOCK_RANGES_BY_SM`):
    ``()`` for the deterministic program, ``(block_n, splits)`` for the cluster program of that tile, else ``(block_n,
    splits, chains, order)`` of this package's CTA-pair program.  ``None`` at an even block count, outside those
    ranges, or at a ``K`` too short for the splits or chains (:func:`k1_splits_fit`, :func:`k1_chains_fit`).
    :func:`plan_atomic_gemm` asks it only where the row has a cluster range.  ``M``, ``N`` and ``K`` as in
    :func:`k1_plan`."""
    if _ceil_div(int(K), K1_BLOCK_K) % 2 == 0:
        return None
    key = (arch, int(L), int(M), int(N), "f32" if out_f32 else "bf16")
    for lo, hi, program in K1_DSM_ODD_BLOCK_RANGES_BY_SM.get(
        (arch, int(sm_count)), {}
    ).get(key, ()):
        if lo <= int(K) <= hi:
            if program is None:
                return ()
            if len(program) == 2:
                block_n, splits = program
                return (int(block_n), int(splits)) if k1_splits_fit(splits, K) else None
            block_n, splits, chains, order = program
            fits = k1_splits_fit(splits, K) and k1_chains_fit(chains, K)
            return (
                (int(block_n), int(splits), int(chains), str(order)) if fits else None
            )
    return None


def k1_chain_plan(
    *,
    arch: str,
    L: int,
    M: int,
    N: int,
    K: int,
    out_f32: bool,
    sm_count: int,
) -> Optional[tuple[int, int]]:
    """``(block_n, chains)`` of the one-split program of several accumulation chains the row runs at ``K`` on a device
    of exactly ``sm_count`` SMs (its range in :data:`K1_CHAIN_RANGES_BY_SM`), or ``None``: the row keeps the routing
    of :func:`k1_plan`'s other tables -- outside those ranges, or at a ``K`` too short for the chains
    (:func:`k1_chains_fit`).  ``M``, ``N`` and ``K`` as in :func:`k1_plan`."""
    key = (arch, int(L), int(M), int(N), "f32" if out_f32 else "bf16")
    part = (arch, int(sm_count))
    for lo, hi, (block_n, chains) in K1_CHAIN_RANGES_BY_SM.get(part, {}).get(key, ()):
        if lo <= int(K) <= hi:
            return (int(block_n), int(chains)) if k1_chains_fit(chains, K) else None
    return None


def k1_plan(
    *,
    arch: str,
    L: int,
    M: int,
    N: int,
    K: int,
    out_f32: bool,
    sm_count: int,
) -> Optional[tuple[int, int]]:
    """``(block_n, num_splits)`` of this package's K1 program for the transposed-store launch ``out[l, n, m]`` (``M``,
    ``N`` as in the comment above; ``K`` the token count) on a device of ``sm_count`` SMs, or ``None``: the row runs the
    deterministic program.  The row's verdict in :data:`K1_SPLIT_RULES` from the SM-count floor of the row
    (:data:`K1_MIN_SM_COUNT`, :data:`K1_RULE_MIN_SM_COUNT`) up, replaced at a ``K`` of
    :data:`K1_SPLIT_RULES_BY_SM_TOKENS` by the row's entry in the table of :data:`K1_SPLIT_OVERRIDES_BY_SM` for exactly
    ``sm_count``, if any, and at a ``K`` of the row's range in :data:`K1_TWO_SPLIT_RANGES_BY_SM` for exactly
    ``sm_count`` by that range's verdict; below the floor the row's verdict in the table of :data:`K1_SPLIT_RULES_BY_SM`
    for exactly ``sm_count``, only at a ``K`` of :data:`K1_SPLIT_RULES_BY_SM_TOKENS` or of the row's ranges in
    :data:`K1_SPLIT_RULES_BY_SM_TOKEN_RANGES` for that SM count, and the verdict of the row's range in
    :data:`K1_SPLIT_RANGE_OVERRIDES_BY_SM` for that SM count at a ``K`` of that range, else the verdict of the row's
    range in :data:`K1_TWO_SPLIT_RANGES_BY_SM` for that SM count at a ``K`` of that range; either except at a ``K`` of
    the row's entries in :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS` for this part (from the entry's SM count up where that
    count is at or above the floor, exactly that SM count below it), in :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS_BY_SM` for
    exactly ``sm_count``, or, at a ``K`` that spans an odd number of 64-element K blocks, in
    :data:`K1_SPLIT_ODD_BLOCK_EXCLUDED_TOKENS` for this part (as the first table).  A row without a verdict, a ``K`` of
    those entries, every other ``K`` below the floor and a ``K`` too short for the split count (:func:`k1_splits_fit`)
    run the deterministic program.  A ``K`` of the row's ranges in :data:`K1_CHAIN_RANGES_BY_SM` for exactly
    ``sm_count`` (:func:`k1_chain_plan`) returns that range's tile width at one split before any of these tables: the
    program of :func:`k1_chain_plan`'s chain count; next, a ``K`` of the row's ranges in
    :data:`K1_ONE_SPLIT_RANGES_BY_SM` for exactly ``sm_count`` returns that range's tile width at one split (one
    chain), before the other tables."""
    chained = k1_chain_plan(
        arch=arch, L=L, M=M, N=N, K=K, out_f32=out_f32, sm_count=sm_count
    )
    if chained is not None:
        return chained[0], 1
    key = (arch, int(L), int(M), int(N), "f32" if out_f32 else "bf16")
    one_split = K1_ONE_SPLIT_RANGES_BY_SM.get((arch, int(sm_count)), {})
    for lo, hi, block_n in one_split.get(key, ()):
        if lo <= int(K) <= hi:
            return int(block_n), 1
    floor = max(K1_MIN_SM_COUNT.get(arch, 0), K1_RULE_MIN_SM_COUNT.get(key, 0))
    first, last = K1_SPLIT_RULES_BY_SM_TOKENS
    in_window = first <= int(K) <= last
    if int(sm_count) >= floor:
        choice = K1_SPLIT_RULES.get(key)
        if in_window:
            overrides = K1_SPLIT_OVERRIDES_BY_SM.get((arch, int(sm_count)), {})
            choice = overrides.get(key, choice)
        two_part = K1_TWO_SPLIT_RANGES_BY_SM.get((arch, int(sm_count)), {})
        own = [v for lo, hi, v in two_part.get(key, ()) if lo <= int(K) <= hi]
        if own:
            choice = own[0]
    else:
        part = (arch, int(sm_count))
        ranges = K1_SPLIT_RULES_BY_SM_TOKEN_RANGES.get(part, {}).get(key, ())
        own = [
            v
            for lo, hi, v in K1_SPLIT_RANGE_OVERRIDES_BY_SM.get(part, {}).get(key, ())
            if lo <= int(K) <= hi
        ]
        if not own:
            two = K1_TWO_SPLIT_RANGES_BY_SM.get(part, {}).get(key, ())
            own = [v for lo, hi, v in two if lo <= int(K) <= hi]
        if own:
            choice = own[0]
        elif in_window or any(lo <= int(K) <= hi for lo, hi in ranges):
            choice = K1_SPLIT_RULES_BY_SM.get(part, {}).get(key)
        else:
            choice = None
    excluded = K1_SPLIT_RULE_EXCLUDED_TOKENS.get(key, ())
    if _ceil_div(K, K1_BLOCK_K) % 2:
        excluded += K1_SPLIT_ODD_BLOCK_EXCLUDED_TOKENS.get(key, ())
    for sms, lo, hi in excluded:
        on_part = int(sm_count) >= sms if sms >= floor else int(sm_count) == sms
        if on_part and lo <= int(K) and (hi is None or int(K) <= hi):
            choice = None
    exact = K1_SPLIT_RULE_EXCLUDED_TOKENS_BY_SM.get((arch, int(sm_count)), {})
    for lo, hi in exact.get(key, ()):
        if lo <= int(K) and (hi is None or int(K) <= hi):
            choice = None
    if choice is None:
        return None
    block_n, splits = (int(v) for v in choice)
    if not k1_splits_fit(splits, K):
        return None
    return block_n, splits


def k1_block_n(N: int) -> int:
    """Column tile of a K1 launch with an explicit split count and no tile width: the transposed-store choice of
    :data:`K1_BLOCK_N_CHOICES` that pads ``N`` least (the wider tile on a tie)."""
    widths = [bn for bn in K1_BLOCK_N_CHOICES if bn not in K1_ROW_MAJOR_BLOCK_N]
    return min(widths, key=lambda bn: (_ceil_div(N, bn) * bn, -bn))


#: The deterministic package's planner knobs of its single-CTA program for a forward row: one 128-row tile per CTA
#: (``cta1``, no CTA pair), 128-wide column tiles and no TMA cache hints -- every output tile over the whole K, no
#: split of K (the program ``kk_n128_c1`` of that package).
K1_FORWARD_SINGLE_CTA_KNOBS: dict[str, Any] = {
    "cta1": True,
    "block_n": 128,
    "hints": ("none", "none"),
}
#: The K1 forward rows of this package that run another plan of the deterministic package's programs than
#: its own plan of the row, ``(arch, sm_count) -> {(K, N, T): knobs or program}``: on a part of exactly ``sm_count`` SMs,
#: :func:`cake_dense_gemm_nondet_train_projection_forward` of ``X[T, K]`` and ``W[N, K]`` (both K-major) into a BF16 ``out`` of exactly these ``K``,
#: ``N`` and ``T`` runs the program the deterministic planner gives the row with these knobs (:func:`k1_forward_knobs`);
#: every other part, row, token count, layout and output dtype runs the deterministic plan.  On 152-SM ``sm_100a`` and
#: ``sm_103a`` parts the deterministic plan of the knob rows (1024 to 4096 tokens) splits K two ways over CTA pairs of 256-wide tiles (stream-K,
#: the partials added in a fixed order); the single-CTA program (:data:`K1_FORWARD_SINGLE_CTA_KNOBS`) keeps every tile's
#: whole K in one CTA, so a forward still repeats bit for bit and a recomputed forward reproduces its result, with other
#: bits than the deterministic package's plan (within the deterministic package's BF16 budget against FP64).  Measured at exactly
#: these token counts (no other token count, SM count or row): the single op faster than the deterministic plan in every
#: run (a one-sided paired sign test, p < 0.05, median ratio below 0.99, eager and CUDA graph), no more than 3 % slower
#: than it next to a kernel holding 8, 16 or 32 SMs on another stream, and the training steps that contain the forward
#: (forward + backward, 2 x forward + backward with an FP32 and with a BF16 weight gradient) no slower than with the
#: deterministic plan: the median over every run of the paired step-time ratio at most 1.013, eager and CUDA graph
#: separately, and no run above the largest ratio of the deterministic plan against itself in the same runs.
#: Not in the table: ``4096 -> 512`` at 4096 tokens (next to a kernel holding 32 SMs it was up to 4.7 % slower), and
#: ``4096 -> 512`` at 3072 tokens on both parts and ``5120 -> 512`` at 4096 tokens on ``sm_103a`` (in one run a training
#: step with eager launches was further above the deterministic plan than that plan against itself).
#:
#: A row whose entry is a program symbol of the deterministic package (its template without the
#: ``dense_proj_det_gemm_`` prefix; 16172 to 16384 tokens) runs that program, planned with the knobs that name it
#: (:func:`k1_program_knobs`): ``kk_n256_ht_cg_pro15`` on 148-SM ``sm_100a`` and 152-SM ``sm_100a`` / ``sm_103a``
#: parts, where the deterministic plan of the row is ``kk_n256_pro7``.  Measured at exactly these rows and token counts
#: on one part of each SM count, against the deterministic plan of the row in the same process: faster in eager
#: launches and in CUDA graph replays in at least 4 of 5 paired groups of a screen, the output equal to the
#: deterministic plan's bit for bit, and the same bits on 20 repeats.  The other rows of that screen (``5120 -> 512``
#: at 16172 and 16231 tokens on 152-SM ``sm_100a`` parts: faster in 3 of 5 groups in graph replays) keep the
#: deterministic plan.  On 212-SM ``sm_107a`` parts ``kk_n128_hen_pro15`` took 1.013 to 1.035 times the time of the
#: deterministic plan ``kk_n256_ht_cg_pro15`` of these rows in every paired group of three runs of this package's A/B
#: against it, eager and in CUDA graph replays; there they run ``kk_n192_m256_hen_s5_pro15`` (0.934 to 0.960 times the
#: time of the deterministic plan, 17 paired groups, eager and graph; ``4096 -> 512`` at 16384 tokens, at 1.007 times
#: its time in eager launches in this package's A/B against FI main, keeps it), and ``5120 -> 32`` / ``48`` / ``64`` and
#: ``8192 -> 128`` run ``kk_n128_hen_pro15`` instead of the deterministic plan's ``kk_n128_hen_pro7`` (0.979 to 0.989
#: times its time, 17 of 17 groups, on two R200 parts), at 16172, 16231 and 16384 tokens; the same bits as that plan.
# fmt: off
K1_FORWARD_RULES_BY_SM: dict[
    tuple[str, int], dict[tuple[int, int, int], Union[dict[str, Any], str]]
] = {
    ("sm_100a", 148): {
        (4096, 512, 16172): "kk_n256_ht_cg_pro15", (5120, 512, 16172): "kk_n256_ht_cg_pro15",
        (5120, 512, 16231): "kk_n256_ht_cg_pro15", (5120, 512, 16384): "kk_n256_ht_cg_pro15",
    },
    ("sm_100a", 152): {
        (4096, 512, 2048): K1_FORWARD_SINGLE_CTA_KNOBS,
        (5120, 512, 1024): K1_FORWARD_SINGLE_CTA_KNOBS,
        (5120, 512, 4096): K1_FORWARD_SINGLE_CTA_KNOBS,
        (4096, 512, 16172): "kk_n256_ht_cg_pro15", (5120, 512, 16384): "kk_n256_ht_cg_pro15",
    },
    ("sm_103a", 152): {
        (4096, 512, 2048): K1_FORWARD_SINGLE_CTA_KNOBS,
        (5120, 512, 1024): K1_FORWARD_SINGLE_CTA_KNOBS,
        (4096, 512, 16172): "kk_n256_ht_cg_pro15", (5120, 512, 16172): "kk_n256_ht_cg_pro15",
    },
    ("sm_107a", 212): {
        **{(k, n, t): "kk_n128_hen_pro15" for k, n in ((5120, 32), (5120, 48), (5120, 64), (8192, 128))
           for t in (16172, 16231, 16384)},
        **{(k, 512, t): "kk_n192_m256_hen_s5_pro15" for k in (4096, 5120) for t in (16172, 16231, 16384)
           if (k, t) != (4096, 16384)},
    },
}
# fmt: on


def k1_forward_knobs(*, arch: str, sm_count: int, K: int, N: int, T: int) -> tuple:
    """The deterministic package's planner knobs of the forward ``out[T, N] = X[T, K] @ W[N, K].T`` (both operands
    K-major, BF16 ``out``) of this package on a device of exactly ``sm_count`` SMs: the row's entry in
    the table of :data:`K1_FORWARD_RULES_BY_SM` for ``(arch, sm_count)`` at exactly ``T`` tokens, as sorted
    ``(name, value)`` pairs (a program symbol: :func:`k1_program_knobs`), or ``()`` -- the deterministic plan."""
    knobs = K1_FORWARD_RULES_BY_SM.get((arch, int(sm_count)), {}).get(
        (int(K), int(N), int(T))
    )
    if isinstance(knobs, str):
        return k1_program_knobs(knobs)
    return tuple(sorted(knobs.items())) if knobs else ()


def _k1_forward_knobs_of(X: torch.Tensor, W: torch.Tensor, out: torch.Tensor) -> tuple:
    """:func:`k1_forward_knobs` of the forward call ``(X, W, out)`` on ``out``'s device; ``()`` (the deterministic
    plan) without a row of :data:`K1_FORWARD_RULES_BY_SM` of the call's shape, for an operand that is not K-major and
    for an ``out`` that is not BF16.  The device is read only for a shape of the table."""
    if X.dim() != 2 or W.dim() != 2:
        return ()
    K, N, T = int(X.shape[1]), int(W.shape[0]), int(X.shape[0])
    if not any((K, N, T) in rows for rows in K1_FORWARD_RULES_BY_SM.values()):
        return ()
    if X.stride(1) != 1 or W.stride(1) != 1 or out.dtype != torch.bfloat16:
        return ()
    facts = deterministic_backend().device_facts(out.device)
    if facts.arch is None:
        return ()
    return k1_forward_knobs(arch=facts.arch, sm_count=facts.sm_count, K=K, N=N, T=T)


def k1_program_knobs(program: str) -> tuple:
    """The deterministic package's planner knobs that name its program ``program`` (a template symbol without the
    ``dense_proj_det_gemm_`` prefix, e.g. ``kn_n128_f32_tma2_s4_c1``), as sorted ``(name, value)`` pairs: that
    package's inverse of its symbol (``_template_knobs``).  ``ValueError`` for a symbol that names no planner knobs.
    The serpentine K walk ``kser`` of that inverse is not a planner keyword: it is dropped when 0, and a ``_ks<P>``
    program (``kser`` P > 0), which that package plans from its own rules only, names no planner knobs."""
    det = deterministic_backend()
    knobs = det._template_knobs(det._TEMPLATE_PREFIX + program)
    if knobs is None or knobs.pop("kser", 0):
        raise ValueError(
            f"cake_dense_gemm_nondet_train: {program!r} names no program of the deterministic planner"
        )
    return tuple(sorted(knobs.items()))


#: The K1 input-gradient rows of this package that run another program of the deterministic package than its own
#: plan of the row, ``(arch, sm_count) -> {(K, N, T, out_dtype): program}``: on a part of exactly ``sm_count`` SMs,
#: :func:`cake_dense_gemm_nondet_train_projection_dgrad` of ``G[T, N]`` and ``W[N, K]`` (both with a unit stride on
#: their last axis) into an ``out[T, K]`` of ``out_dtype`` (``"bf16"`` or ``"f32"``) at exactly these ``K``, ``N``
#: and ``T`` runs the deterministic package's program ``program`` (its template without the ``dense_proj_det_gemm_``
#: prefix), planned with the knobs that name it (:func:`k1_program_knobs`); every other part, row, token count, layout
#: and output dtype runs the deterministic plan.  The table's programs (``kn_n128_f32_tma2_s4_c1``,
#: ``kn_n256_tma1_w8`` / ``_w8r`` / ``_w8rs``, ``kn_n256_f32_v8_ef_pro7``; on 212-SM ``sm_107a`` parts
#: ``kn_n128_f32_tma1_pd2_c1`` and ``kn_n256_tma2_pd2_lw_pro15``) were measured at exactly these rows and token counts on
#: one part of each SM count, against the deterministic plan of the row in the same process: faster in eager launches
#: and in CUDA graph replays in at least 4 of 5 paired groups of a screen (``sm_107a``: 5 of 5), the output equal to
#: the deterministic plan's bit for bit, and the same bits on 20 repeats; the rows and token counts of that screen that
#: missed keep the deterministic plan.
# fmt: off
K1_DGRAD_RULES_BY_SM: dict[tuple[str, int], dict[tuple[int, int, int, str], str]] = {
    ("sm_100a", 148): {
        (4096, 32, 16172, "bf16"): "kn_n256_tma1_w8r", (4096, 32, 16172, "f32"): "kn_n128_f32_tma2_s4_c1",
        (4096, 32, 16231, "bf16"): "kn_n256_tma1_w8r", (4096, 32, 16231, "f32"): "kn_n128_f32_tma2_s4_c1",
        (4096, 32, 16384, "bf16"): "kn_n256_tma1_w8r", (4096, 32, 16384, "f32"): "kn_n128_f32_tma2_s4_c1",
        (4096, 48, 16172, "bf16"): "kn_n256_tma1_w8rs", (4096, 48, 16172, "f32"): "kn_n128_f32_tma2_s4_c1",
        (4096, 48, 16231, "f32"): "kn_n128_f32_tma2_s4_c1", (4096, 48, 16384, "f32"): "kn_n128_f32_tma2_s4_c1",
        (4096, 64, 16172, "bf16"): "kn_n256_tma1_w8rs", (4096, 64, 16172, "f32"): "kn_n128_f32_tma2_s4_c1",
        (4096, 64, 16231, "f32"): "kn_n128_f32_tma2_s4_c1", (4096, 64, 16384, "bf16"): "kn_n256_tma1_w8rs",
        (4096, 64, 16384, "f32"): "kn_n128_f32_tma2_s4_c1", (4096, 128, 16172, "bf16"): "kn_n256_tma1_w8r",
        (4096, 128, 16172, "f32"): "kn_n128_f32_tma2_s4_c1", (4096, 128, 16231, "bf16"): "kn_n256_tma1_w8r",
        (4096, 128, 16231, "f32"): "kn_n128_f32_tma2_s4_c1", (4096, 128, 16384, "bf16"): "kn_n256_tma1_w8rs",
        (4096, 128, 16384, "f32"): "kn_n128_f32_tma2_s4_c1", (4096, 1024, 16172, "f32"): "kn_n256_f32_v8_ef_pro7",
        (4096, 1024, 16231, "f32"): "kn_n256_f32_v8_ef_pro7", (4096, 1024, 16384, "f32"): "kn_n256_f32_v8_ef_pro7",
        (5120, 32, 16172, "bf16"): "kn_n256_tma1_w8r", (5120, 32, 16172, "f32"): "kn_n128_f32_tma2_s4_c1",
        (5120, 32, 16231, "bf16"): "kn_n256_tma1_w8rs", (5120, 32, 16231, "f32"): "kn_n128_f32_tma2_s4_c1",
        (5120, 32, 16384, "bf16"): "kn_n256_tma1_w8rs", (5120, 32, 16384, "f32"): "kn_n128_f32_tma2_s4_c1",
        (5120, 48, 16172, "bf16"): "kn_n256_tma1_w8rs", (5120, 48, 16172, "f32"): "kn_n128_f32_tma2_s4_c1",
        (5120, 48, 16231, "bf16"): "kn_n256_tma1_w8r", (5120, 48, 16231, "f32"): "kn_n128_f32_tma2_s4_c1",
        (5120, 48, 16384, "bf16"): "kn_n256_tma1_w8r", (5120, 48, 16384, "f32"): "kn_n128_f32_tma2_s4_c1",
        (5120, 64, 16172, "f32"): "kn_n128_f32_tma2_s4_c1", (5120, 64, 16231, "f32"): "kn_n128_f32_tma2_s4_c1",
        (5120, 64, 16384, "f32"): "kn_n128_f32_tma2_s4_c1", (5120, 128, 16172, "bf16"): "kn_n256_tma1_w8r",
        (5120, 128, 16172, "f32"): "kn_n128_f32_tma2_s4_c1", (5120, 128, 16231, "bf16"): "kn_n256_tma1_w8rs",
        (5120, 128, 16231, "f32"): "kn_n128_f32_tma2_s4_c1", (5120, 128, 16384, "bf16"): "kn_n256_tma1_w8r",
        (5120, 1024, 16172, "f32"): "kn_n256_f32_v8_ef_pro7", (5120, 1024, 16231, "f32"): "kn_n256_f32_v8_ef_pro7",
        (5120, 1024, 16384, "f32"): "kn_n256_f32_v8_ef_pro7", (6144, 32, 16172, "bf16"): "kn_n256_tma1_w8r",
        (6144, 32, 16231, "bf16"): "kn_n256_tma1_w8r", (6144, 32, 16384, "bf16"): "kn_n256_tma1_w8rs",
        (6144, 48, 16172, "bf16"): "kn_n256_tma1_w8rs", (6144, 48, 16172, "f32"): "kn_n128_f32_tma2_s4_c1",
        (6144, 48, 16231, "bf16"): "kn_n256_tma1_w8rs", (6144, 48, 16231, "f32"): "kn_n128_f32_tma2_s4_c1",
        (6144, 48, 16384, "bf16"): "kn_n256_tma1_w8r", (6144, 48, 16384, "f32"): "kn_n128_f32_tma2_s4_c1",
        (6144, 64, 16172, "bf16"): "kn_n256_tma1_w8r", (6144, 64, 16172, "f32"): "kn_n128_f32_tma2_s4_c1",
        (6144, 64, 16231, "bf16"): "kn_n256_tma1_w8r", (6144, 64, 16231, "f32"): "kn_n128_f32_tma2_s4_c1",
        (6144, 64, 16384, "bf16"): "kn_n256_tma1_w8r", (6144, 64, 16384, "f32"): "kn_n128_f32_tma2_s4_c1",
        (6144, 128, 16172, "bf16"): "kn_n256_tma1_w8r", (6144, 128, 16172, "f32"): "kn_n128_f32_tma2_s4_c1",
        (6144, 128, 16231, "bf16"): "kn_n256_tma1_w8rs", (6144, 128, 16231, "f32"): "kn_n128_f32_tma2_s4_c1",
        (6144, 128, 16384, "bf16"): "kn_n256_tma1_w8rs", (6144, 128, 16384, "f32"): "kn_n128_f32_tma2_s4_c1",
        (8192, 128, 16172, "bf16"): "kn_n256_tma1_w8r", (8192, 128, 16172, "f32"): "kn_n128_f32_tma2_s4_c1",
        (8192, 128, 16231, "bf16"): "kn_n256_tma1_w8rs", (8192, 128, 16231, "f32"): "kn_n128_f32_tma2_s4_c1",
        (8192, 128, 16384, "bf16"): "kn_n256_tma1_w8r", (8192, 128, 16384, "f32"): "kn_n128_f32_tma2_s4_c1",
    },
    ("sm_100a", 152): {
        (4096, 32, 16172, "bf16"): "kn_n256_tma1_w8r", (4096, 32, 16172, "f32"): "kn_n128_f32_tma2_s4_c1",
        (4096, 32, 16231, "bf16"): "kn_n256_tma1_w8rs", (4096, 32, 16231, "f32"): "kn_n128_f32_tma2_s4_c1",
        (4096, 32, 16384, "bf16"): "kn_n256_tma1_w8", (4096, 32, 16384, "f32"): "kn_n128_f32_tma2_s4_c1",
        (4096, 48, 16172, "bf16"): "kn_n256_tma1_w8r", (4096, 48, 16172, "f32"): "kn_n128_f32_tma2_s4_c1",
        (4096, 48, 16231, "f32"): "kn_n128_f32_tma2_s4_c1", (4096, 48, 16384, "f32"): "kn_n128_f32_tma2_s4_c1",
        (4096, 64, 16172, "f32"): "kn_n128_f32_tma2_s4_c1", (4096, 64, 16231, "f32"): "kn_n128_f32_tma2_s4_c1",
        (4096, 64, 16384, "bf16"): "kn_n256_tma1_w8rs", (4096, 64, 16384, "f32"): "kn_n128_f32_tma2_s4_c1",
        (4096, 128, 16172, "bf16"): "kn_n256_tma1_w8rs", (4096, 128, 16172, "f32"): "kn_n128_f32_tma2_s4_c1",
        (4096, 128, 16231, "bf16"): "kn_n256_tma1_w8r", (4096, 128, 16231, "f32"): "kn_n128_f32_tma2_s4_c1",
        (4096, 128, 16384, "bf16"): "kn_n256_tma1_w8r", (4096, 128, 16384, "f32"): "kn_n128_f32_tma2_s4_c1",
        (5120, 32, 16172, "bf16"): "kn_n256_tma1_w8", (5120, 32, 16172, "f32"): "kn_n128_f32_tma2_s4_c1",
        (5120, 32, 16231, "bf16"): "kn_n256_tma1_w8r", (5120, 32, 16231, "f32"): "kn_n128_f32_tma2_s4_c1",
        (5120, 32, 16384, "bf16"): "kn_n256_tma1_w8rs", (5120, 32, 16384, "f32"): "kn_n128_f32_tma2_s4_c1",
        (5120, 48, 16172, "bf16"): "kn_n256_tma1_w8r", (5120, 48, 16172, "f32"): "kn_n128_f32_tma2_s4_c1",
        (5120, 48, 16231, "bf16"): "kn_n256_tma1_w8", (5120, 48, 16231, "f32"): "kn_n128_f32_tma2_s4_c1",
        (5120, 48, 16384, "bf16"): "kn_n256_tma1_w8r", (5120, 48, 16384, "f32"): "kn_n128_f32_tma2_s4_c1",
        (5120, 64, 16172, "f32"): "kn_n128_f32_tma2_s4_c1", (5120, 64, 16231, "f32"): "kn_n128_f32_tma2_s4_c1",
        (5120, 64, 16384, "f32"): "kn_n128_f32_tma2_s4_c1", (5120, 128, 16172, "bf16"): "kn_n256_tma1_w8rs",
        (5120, 128, 16172, "f32"): "kn_n128_f32_tma2_s4_c1", (5120, 128, 16231, "bf16"): "kn_n256_tma1_w8r",
        (5120, 128, 16384, "bf16"): "kn_n256_tma1_w8r", (5120, 128, 16384, "f32"): "kn_n128_f32_tma2_s4_c1",
        (6144, 32, 16172, "bf16"): "kn_n256_tma1_w8rs", (6144, 32, 16231, "bf16"): "kn_n256_tma1_w8rs",
        (6144, 32, 16384, "bf16"): "kn_n256_tma1_w8rs", (6144, 48, 16172, "bf16"): "kn_n256_tma1_w8rs",
        (6144, 48, 16172, "f32"): "kn_n128_f32_tma2_s4_c1", (6144, 48, 16231, "bf16"): "kn_n256_tma1_w8rs",
        (6144, 48, 16231, "f32"): "kn_n128_f32_tma2_s4_c1", (6144, 48, 16384, "bf16"): "kn_n256_tma1_w8rs",
        (6144, 48, 16384, "f32"): "kn_n128_f32_tma2_s4_c1", (6144, 64, 16172, "bf16"): "kn_n256_tma1_w8r",
        (6144, 64, 16172, "f32"): "kn_n128_f32_tma2_s4_c1", (6144, 64, 16231, "bf16"): "kn_n256_tma1_w8r",
        (6144, 64, 16231, "f32"): "kn_n128_f32_tma2_s4_c1", (6144, 64, 16384, "bf16"): "kn_n256_tma1_w8r",
        (6144, 64, 16384, "f32"): "kn_n128_f32_tma2_s4_c1", (6144, 128, 16172, "bf16"): "kn_n256_tma1_w8rs",
        (6144, 128, 16172, "f32"): "kn_n128_f32_tma2_s4_c1", (6144, 128, 16231, "bf16"): "kn_n256_tma1_w8r",
        (6144, 128, 16231, "f32"): "kn_n128_f32_tma2_s4_c1", (6144, 128, 16384, "bf16"): "kn_n256_tma1_w8r",
        (6144, 128, 16384, "f32"): "kn_n128_f32_tma2_s4_c1", (8192, 128, 16172, "bf16"): "kn_n256_tma1_w8rs",
        (8192, 128, 16172, "f32"): "kn_n128_f32_tma2_s4_c1", (8192, 128, 16231, "bf16"): "kn_n256_tma1_w8rs",
        (8192, 128, 16384, "bf16"): "kn_n256_tma1_w8r", (8192, 128, 16384, "f32"): "kn_n128_f32_tma2_s4_c1",
    },
    ("sm_103a", 152): {
        (4096, 32, 16172, "bf16"): "kn_n256_tma1_w8r", (4096, 32, 16172, "f32"): "kn_n128_f32_tma2_s4_c1",
        (4096, 32, 16231, "bf16"): "kn_n256_tma1_w8rs", (4096, 32, 16231, "f32"): "kn_n128_f32_tma2_s4_c1",
        (4096, 32, 16384, "bf16"): "kn_n256_tma1_w8r", (4096, 32, 16384, "f32"): "kn_n128_f32_tma2_s4_c1",
        (4096, 48, 16172, "f32"): "kn_n128_f32_tma2_s4_c1", (4096, 48, 16231, "f32"): "kn_n128_f32_tma2_s4_c1",
        (4096, 48, 16384, "f32"): "kn_n128_f32_tma2_s4_c1", (4096, 64, 16172, "bf16"): "kn_n256_tma1_w8rs",
        (4096, 64, 16172, "f32"): "kn_n128_f32_tma2_s4_c1", (4096, 64, 16231, "bf16"): "kn_n256_tma1_w8rs",
        (4096, 64, 16231, "f32"): "kn_n128_f32_tma2_s4_c1", (4096, 64, 16384, "bf16"): "kn_n256_tma1_w8r",
        (4096, 64, 16384, "f32"): "kn_n128_f32_tma2_s4_c1", (4096, 128, 16172, "bf16"): "kn_n256_tma1_w8rs",
        (4096, 128, 16172, "f32"): "kn_n128_f32_tma2_s4_c1", (4096, 128, 16231, "bf16"): "kn_n256_tma1_w8rs",
        (4096, 128, 16231, "f32"): "kn_n128_f32_tma2_s4_c1", (4096, 128, 16384, "f32"): "kn_n128_f32_tma2_s4_c1",
        (4096, 1024, 16172, "f32"): "kn_n256_f32_v8_ef_pro7", (4096, 1024, 16231, "f32"): "kn_n256_f32_v8_ef_pro7",
        (4096, 1024, 16384, "f32"): "kn_n256_f32_v8_ef_pro7", (5120, 32, 16172, "bf16"): "kn_n256_tma1_w8r",
        (5120, 32, 16172, "f32"): "kn_n128_f32_tma2_s4_c1", (5120, 32, 16231, "bf16"): "kn_n256_tma1_w8r",
        (5120, 32, 16231, "f32"): "kn_n128_f32_tma2_s4_c1", (5120, 32, 16384, "f32"): "kn_n128_f32_tma2_s4_c1",
        (5120, 48, 16172, "bf16"): "kn_n256_tma1_w8r", (5120, 48, 16172, "f32"): "kn_n128_f32_tma2_s4_c1",
        (5120, 48, 16231, "bf16"): "kn_n256_tma1_w8r", (5120, 48, 16231, "f32"): "kn_n128_f32_tma2_s4_c1",
        (5120, 48, 16384, "bf16"): "kn_n256_tma1_w8r", (5120, 48, 16384, "f32"): "kn_n128_f32_tma2_s4_c1",
        (5120, 64, 16172, "f32"): "kn_n128_f32_tma2_s4_c1", (5120, 64, 16231, "f32"): "kn_n128_f32_tma2_s4_c1",
        (5120, 64, 16384, "f32"): "kn_n128_f32_tma2_s4_c1", (5120, 128, 16172, "bf16"): "kn_n256_tma1_w8rs",
        (5120, 128, 16172, "f32"): "kn_n128_f32_tma2_s4_c1", (5120, 128, 16231, "f32"): "kn_n128_f32_tma2_s4_c1",
        (5120, 128, 16384, "f32"): "kn_n128_f32_tma2_s4_c1", (5120, 1024, 16231, "f32"): "kn_n256_f32_v8_ef_pro7",
        (5120, 1024, 16384, "f32"): "kn_n256_f32_v8_ef_pro7", (6144, 32, 16172, "bf16"): "kn_n256_tma1_w8rs",
        (6144, 32, 16231, "bf16"): "kn_n256_tma1_w8rs", (6144, 32, 16384, "bf16"): "kn_n256_tma1_w8rs",
        (6144, 48, 16172, "bf16"): "kn_n256_tma1_w8r", (6144, 48, 16172, "f32"): "kn_n128_f32_tma2_s4_c1",
        (6144, 48, 16231, "bf16"): "kn_n256_tma1_w8rs", (6144, 48, 16231, "f32"): "kn_n128_f32_tma2_s4_c1",
        (6144, 48, 16384, "bf16"): "kn_n256_tma1_w8r", (6144, 48, 16384, "f32"): "kn_n128_f32_tma2_s4_c1",
        (6144, 64, 16172, "bf16"): "kn_n256_tma1_w8r", (6144, 64, 16172, "f32"): "kn_n128_f32_tma2_s4_c1",
        (6144, 64, 16231, "bf16"): "kn_n256_tma1_w8r", (6144, 64, 16231, "f32"): "kn_n128_f32_tma2_s4_c1",
        (6144, 64, 16384, "bf16"): "kn_n256_tma1_w8r", (6144, 64, 16384, "f32"): "kn_n128_f32_tma2_s4_c1",
        (6144, 128, 16172, "bf16"): "kn_n256_tma1_w8r", (6144, 128, 16172, "f32"): "kn_n128_f32_tma2_s4_c1",
        (6144, 128, 16231, "bf16"): "kn_n256_tma1_w8r", (6144, 128, 16231, "f32"): "kn_n128_f32_tma2_s4_c1",
        (6144, 128, 16384, "bf16"): "kn_n256_tma1_w8rs", (6144, 128, 16384, "f32"): "kn_n128_f32_tma2_s4_c1",
        (8192, 128, 16172, "f32"): "kn_n128_f32_tma2_s4_c1", (8192, 128, 16231, "f32"): "kn_n128_f32_tma2_s4_c1",
        (8192, 128, 16384, "f32"): "kn_n128_f32_tma2_s4_c1",
    },
    ("sm_107a", 212): {
        (4096, 128, 16172, "f32"): "kn_n128_f32_tma1_pd2_c1", (4096, 128, 16231, "f32"): "kn_n128_f32_tma1_pd2_c1",
        (4096, 128, 16384, "f32"): "kn_n128_f32_tma1_pd2_c1", (5120, 128, 16172, "f32"): "kn_n128_f32_tma1_pd2_c1",
        (5120, 128, 16231, "f32"): "kn_n128_f32_tma1_pd2_c1", (5120, 128, 16384, "f32"): "kn_n128_f32_tma1_pd2_c1",
        (6144, 32, 16172, "bf16"): "kn_n256_tma2_pd2_lw_pro15", (6144, 32, 16231, "bf16"): "kn_n256_tma2_pd2_lw_pro15",
        (6144, 32, 16384, "bf16"): "kn_n256_tma2_pd2_lw_pro15",
    },
}
# fmt: on

#: The K1 input-gradient rows of this package that run the deterministic plan of the row with another TMA L2
#: promotion of its operands, ``(arch, sm_count) -> {(K, N, out_dtype): ((lo, hi, promo), ...)}``: on a part of
#: exactly ``sm_count`` SMs, :func:`cake_dense_gemm_nondet_train_projection_dgrad` of ``G[T, N]`` and ``W[N, K]``
#: (both with a unit stride on their last axis) into an ``out[T, K]`` of ``out_dtype`` at ``lo <= T <= hi`` tokens
#: runs the deterministic planner's plan of the row with the planner knob ``promo`` (:func:`k1_dgrad_knobs`), a launch
#: parameter of that plan: the same program, grid and raster, so the deterministic plan's bits.  A row of
#: :data:`K1_DGRAD_RULES_BY_SM` comes first; every other part, row, token count, layout and output dtype keeps the
#: deterministic plan.  On 148-SM ``sm_100a`` parts the BF16 rows ``4096 / 5120 -> 512 / 1024`` and ``6144 -> 576``
#: from 16172 to 16384 tokens promote 256 bytes (the deterministic plan of these rows promotes none).  Measured on one
#: B200 part at 16172, 16231 and 16384 tokens, 17 paired groups in eager launches and in CUDA graph replays, against
#: the deterministic plan in the same process: 0.979 to 0.997 times its time (``4096 -> 512`` 0.980 to 0.987,
#: ``5120 -> 512`` 0.978 to 0.983, the other three rows 0.994 to 0.997), faster in 15 to 17 of 17 groups, the same
#: kernel and grid, the same bits on 20 launches; at 16172 and 16384 tokens on a second GPU of the part 0.976 to 0.998
#: times its time, faster in 15 to 17 of 17 groups.
# fmt: off
K1_DGRAD_PROMO_RANGES_BY_SM: dict[tuple[str, int], dict[tuple[int, int, str], tuple[tuple[int, int, str], ...]]] = {
    ("sm_100a", 148): {
        (K, N, "bf16"): ((16172, 16384, "l2_256b"),)
        for K, N in ((4096, 512), (4096, 1024), (5120, 512), (5120, 1024), (6144, 576))
    },
}
# fmt: on


def k1_dgrad_promo(
    *, arch: str, sm_count: int, K: int, N: int, T: int, out_f32: bool
) -> Optional[str]:
    """The TMA L2 promotion of the input gradient ``out[T, K] = G[T, N] @ W[N, K]`` on a device of exactly
    ``sm_count`` SMs (``out`` FP32 when ``out_f32``, else BF16): that of the row's range in the table of
    :data:`K1_DGRAD_PROMO_RANGES_BY_SM` for ``(arch, sm_count)`` that holds ``T``, or ``None``."""
    ranges = K1_DGRAD_PROMO_RANGES_BY_SM.get((arch, int(sm_count)), {}).get(
        (int(K), int(N), "f32" if out_f32 else "bf16"), ()
    )
    for lo, hi, promo in ranges:
        if lo <= int(T) <= hi:
            return promo
    return None


def k1_dgrad_knobs(
    *, arch: str, sm_count: int, K: int, N: int, T: int, out_f32: bool
) -> tuple:
    """The deterministic package's planner knobs of the input gradient ``out[T, K] = G[T, N] @ W[N, K]`` of this
    package on a device of exactly ``sm_count`` SMs (``out`` FP32 when ``out_f32``, else BF16): those of the program of
    the row's entry in the table of :data:`K1_DGRAD_RULES_BY_SM` for ``(arch, sm_count)`` at exactly ``T`` tokens
    (:func:`k1_program_knobs`), else the ``promo`` of the row's range of :data:`K1_DGRAD_PROMO_RANGES_BY_SM`
    (:func:`k1_dgrad_promo`), or ``()`` -- the deterministic plan."""
    program = K1_DGRAD_RULES_BY_SM.get((arch, int(sm_count)), {}).get(
        (int(K), int(N), int(T), "f32" if out_f32 else "bf16")
    )
    if program:
        return k1_program_knobs(program)
    promo = k1_dgrad_promo(arch=arch, sm_count=sm_count, K=K, N=N, T=T, out_f32=out_f32)
    return (("promo", promo),) if promo else ()


def _k1_dgrad_knobs_of(G: torch.Tensor, W: torch.Tensor, out: torch.Tensor) -> tuple:
    """:func:`k1_dgrad_knobs` of the input-gradient call ``(G, W, out)`` on ``out``'s device; ``()`` (the
    deterministic plan) without a row of :data:`K1_DGRAD_RULES_BY_SM` or :data:`K1_DGRAD_PROMO_RANGES_BY_SM` of the
    call's shape and output dtype, and for an operand without a unit stride on its last axis.  The device is read only
    for a shape of the tables."""
    if G.dim() != 2 or W.dim() != 2 or out.dtype not in (torch.bfloat16, torch.float32):
        return ()
    K, N, T = int(W.shape[1]), int(W.shape[0]), int(G.shape[0])
    dt = "f32" if out.dtype == torch.float32 else "bf16"
    if not any(
        (K, N, T, dt) in rows for rows in K1_DGRAD_RULES_BY_SM.values()
    ) and not any(
        lo <= T <= hi
        for rows in K1_DGRAD_PROMO_RANGES_BY_SM.values()
        for lo, hi, _ in rows.get((K, N, dt), ())
    ):
        return ()
    if G.stride(1) != 1 or W.stride(1) != 1:
        return ()
    facts = deterministic_backend().device_facts(out.device)
    if facts.arch is None:
        return ()
    return k1_dgrad_knobs(
        arch=facts.arch,
        sm_count=facts.sm_count,
        K=K,
        N=N,
        T=T,
        out_f32=out.dtype == torch.float32,
    )


#: The registry template of this package's input-gradient program of 256-wide output tiles with FP32 output
#: (:data:`K1_DGRAD_PROGRAM_RULES_BY_SM`): row stores, or TMA stores of the tile from shared memory (``tma_store``),
#: both loading the output gradient with the L2 evict-first hint; ``lean_free``: TMA stores without that hint, the
#: start path of cuBLAS's nvjet kernels (each role waiting on the cluster barrier only before its first operation that
#: needs it) and the epilogue freeing the tensor memory (``tma_store`` implied).
def dgrad_program_template(tma_store: bool, lean_free: bool = False) -> str:
    if lean_free:
        return f"{cake_jit.TEMPLATE_PREFIXES[9]}kn_n256_f32_tma_lean_free"
    return f"{cake_jit.TEMPLATE_PREFIXES[9]}kn_n256_f32_ef" + (
        "_tma" if tma_store else ""
    )


_WAVE_LEAN_N224_PF2_ST = "dense_proj_gemm_wave_lean_kk_n224_ef_e8_pf2_st"
_WAVE_LEAN_N224_PF2_ST_LH = "dense_proj_gemm_wave_lean_kk_n224_ef_e8_pf2_st_lh"
_DGRAD_N = "dense_proj_gemm_dgrad_kn_n256_f32_ef"
_DGRAD_N_TMA = "dense_proj_gemm_dgrad_kn_n256_f32_ef_tma"
_DGRAD_N_TMA_LEAN_FREE = "dense_proj_gemm_dgrad_kn_n256_f32_tma_lean_free"
#: The token counts of the 148-SM ``sm_100a`` and 152-SM ``sm_103a`` input-gradient windows of
#: :data:`K1_DGRAD_PROGRAM_RULES_BY_SM`.
_DGRAD_WINDOW_TOKENS = range(16172, 16385)

#: The K1 forward rows that run a program of this package instead of the deterministic package's,
#: ``(arch, sm_count) -> {(K, N, T): template}``: on a part of exactly ``sm_count`` SMs,
#: :func:`cake_dense_gemm_nondet_train_projection_forward` of ``X[T, K]`` and ``W[N, K]`` (both K-major) into a BF16
#: ``out[T, N]`` (unit inner stride) at exactly these extents and token count runs the forward program ``template``
#: (``dense_proj_gemm_wave_lean_kk_n224_ef_e8_pf2_st``): one CTA pair per work item of 512 output features by 224
#: tokens over the whole K (``ceil(N / 512) * ceil(T / 224)`` items: 74 at 512 features, one wave of the SM pairs; 146
#: or 148 at 1024 features, at most two per CTA pair), each output element one ascending FP32 chain over K, no split of
#: K, no workspace: the deterministic package's bits, the same on every run.  It comes before
#: :data:`K1_FORWARD_RULES_BY_SM`; every other part, row, token count, layout and output dtype keeps that table and the
#: deterministic plan.  The program is the one-wave forward (the same items, ring, MMAs and order of the sums as its
#: plain form, so the same bits) with the start path of cuBLAS's nvjet kernels (each role waiting on the cluster barrier
#: only before its first operation that needs it), the epilogue freeing the tensor memory, eight epilogue warps, an L2
#: prefetch of the X box two ring stages ahead, and each item's last K stage run by token half, so that the first half
#: of the tile is stored while the MMAs of the second half run.  Measured at exactly these rows and token counts on one
#: part of each SM count against the former program of the row in the same process, eager launches and CUDA graph
#: replays, 5 paired groups, the output bits equal to the former program's on every row checked: at 512 features the
#: same program without the prefetch and the token halves took 1.007 to 1.040 times its time on a B200 part, 1.018 to
#: 1.055 times on a GB200 part and 1.012 to 1.026 times on a GB300 part, slower in 0 of 5 groups on every row; at 1024
#: features the deterministic plan's program took 1.032 to 1.057 times its time on a GB200 part and 1.010 to 1.033
#: times on a GB300 part, slower in at most 1 of 5 groups.  The other 1024-feature rows of the 152-SM parts
#: (``4096 -> 1024`` and ``5120 -> 1024`` at 16231 tokens on ``sm_100a``, ``4096 -> 1024`` at 16231 and
#: ``5120 -> 1024`` at 16384 tokens on ``sm_103a``), 17 paired groups in eager launches and in CUDA graph replays each:
#: the deterministic plan's program took 1.017 to 1.036 times its time in two processes on a GB200 part and 1.011 to
#: 1.033 times on a GB300 part, slower in at most 2 of 17 groups, the same bits on 20 repeats.  The 1024-feature rows of
#: the 148-SM parts keep the deterministic plan.  ``4096 -> 512`` and ``5120 -> 512`` at 16172 and 16231 tokens run the
#: same program with L2 eviction priorities on its loads and stores (``dense_proj_gemm_wave_lean_kk_n224_ef_e8_pf2_st_lh``:
#: the same items, MMAs and bits): on a GB300 part 0.965 to 0.975 times the time of the program without them, below 1 in
#: both modes on all four rows; on a B200 part below 1 in 8 of 8 row-modes; on a GB200 part a median below 1 in 16 of 16
#: row-modes over two processes; the same bits on 20 repeats.
# fmt: off
K1_FORWARD_PROGRAM_RULES_BY_SM: dict[tuple[str, int], dict[tuple[int, int, int], str]] = {
    ("sm_100a", 148): {
        (4096, 512, 16172): _WAVE_LEAN_N224_PF2_ST_LH, (4096, 512, 16231): _WAVE_LEAN_N224_PF2_ST_LH,
        (4096, 512, 16384): _WAVE_LEAN_N224_PF2_ST, (5120, 512, 16172): _WAVE_LEAN_N224_PF2_ST_LH,
        (5120, 512, 16231): _WAVE_LEAN_N224_PF2_ST_LH, (5120, 512, 16384): _WAVE_LEAN_N224_PF2_ST,
    },
    ("sm_100a", 152): {
        (4096, 512, 16172): _WAVE_LEAN_N224_PF2_ST_LH, (4096, 512, 16231): _WAVE_LEAN_N224_PF2_ST_LH,
        (4096, 512, 16384): _WAVE_LEAN_N224_PF2_ST, (5120, 512, 16172): _WAVE_LEAN_N224_PF2_ST_LH,
        (5120, 512, 16231): _WAVE_LEAN_N224_PF2_ST_LH, (5120, 512, 16384): _WAVE_LEAN_N224_PF2_ST,
        (4096, 1024, 16172): _WAVE_LEAN_N224_PF2_ST, (4096, 1024, 16231): _WAVE_LEAN_N224_PF2_ST,
        (4096, 1024, 16384): _WAVE_LEAN_N224_PF2_ST, (5120, 1024, 16172): _WAVE_LEAN_N224_PF2_ST,
        (5120, 1024, 16231): _WAVE_LEAN_N224_PF2_ST, (5120, 1024, 16384): _WAVE_LEAN_N224_PF2_ST,
    },
    ("sm_103a", 152): {
        (4096, 512, 16172): _WAVE_LEAN_N224_PF2_ST_LH, (4096, 512, 16231): _WAVE_LEAN_N224_PF2_ST_LH,
        (4096, 512, 16384): _WAVE_LEAN_N224_PF2_ST, (5120, 512, 16172): _WAVE_LEAN_N224_PF2_ST_LH,
        (5120, 512, 16231): _WAVE_LEAN_N224_PF2_ST_LH, (5120, 512, 16384): _WAVE_LEAN_N224_PF2_ST,
        (4096, 1024, 16172): _WAVE_LEAN_N224_PF2_ST, (4096, 1024, 16231): _WAVE_LEAN_N224_PF2_ST,
        (4096, 1024, 16384): _WAVE_LEAN_N224_PF2_ST, (5120, 1024, 16172): _WAVE_LEAN_N224_PF2_ST,
        (5120, 1024, 16231): _WAVE_LEAN_N224_PF2_ST, (5120, 1024, 16384): _WAVE_LEAN_N224_PF2_ST,
    },
}
# fmt: on

#: The K1 input-gradient rows that run a program of this package instead of the deterministic package's,
#: ``(arch, sm_count) -> {(K, N, T, out_dtype): template}``: on a part of exactly ``sm_count`` SMs,
#: :func:`cake_dense_gemm_nondet_train_projection_dgrad` of ``G[T, N]`` and ``W[N, K]`` (both with a unit stride on
#: their last axis) into an ``out[T, K]`` of ``out_dtype`` (unit inner stride) at exactly these extents and token count
#: runs the input-gradient program ``template`` (:func:`dgrad_program_template`): a persistent CTA pair per SM pair
#: over ``256 x 256`` tiles of ``out``, the whole ``N`` per tile, each output element one ascending FP32 chain, no split,
#: no workspace: the deterministic package's bits, the same on every run.  It comes before
#: :data:`K1_DGRAD_RULES_BY_SM`; every other part, row, token count, layout and output dtype keeps that table and the
#: deterministic plan.  Measured at exactly these rows and token counts on one part of each SM count against this
#: package's program of the row in the same process: faster in eager launches and in CUDA graph replays in 5 of 5 paired
#: groups, the output equal to the deterministic plan's bit for bit, and the same bits on 20 repeats.  On 148-SM
#: ``sm_100a`` parts ``4096 -> 512``, ``5120 -> 512`` and ``6144 -> 576`` run the program without the evict-first hint
#: (``dense_proj_gemm_dgrad_kn_n256_f32_tma_lean_free``) at every token count from 16172 to 16384
#: (:data:`_DGRAD_WINDOW_TOKENS`).  On one B200 part, two GPUs, 17 paired groups in eager launches and in CUDA graph
#: replays each, the same bits as the deterministic plan and on 20 repeats: at 16172 and 16384 tokens the 512-feature
#: rows took 0.95 to 0.97 times the time of the evict-first TMA form they ran before; at 16231 tokens the form without
#: the hint took 0.93 to 0.95 times the time of that form in 5 of 5 paired groups.  ``6144 -> 576`` on one B200 part,
#: with its weight placed at 24 random addresses at each of 16172, 16231, 16300 and 16384 tokens: 0.944 to 0.949 times
#: the deterministic plan's time (medians over the placements), slower than it at none.  A form holding the tile row's
#: output gradient in shared memory with the evict-first hint took 0.95 times that plan's time at most placements but
#: 1.07 to 1.08 times at some (the time depended on the memory pages holding the weight); this form's did not.  On 152-SM ``sm_103a`` parts ``4096 -> 512``, ``5120 -> 512`` and
#: ``6144 -> 576`` run the program without the evict-first hint at every token count of the same window.  On one GB300
#: part, two GPUs, 17 paired groups in eager launches and in CUDA graph replays each, at 16172, 16231 and 16384 tokens,
#: the same bits as the deterministic plan and on 20 repeats: the 512-feature rows took 0.95 to 0.96 times the time of
#: the evict-first TMA form they ran before and ``6144 -> 576`` 0.94 to 0.95 times that of the deterministic plan.
# fmt: off
K1_DGRAD_PROGRAM_RULES_BY_SM: dict[tuple[str, int], dict[tuple[int, int, int, str], str]] = {
    ("sm_100a", 148): {
        (4096, 1024, 16172, "f32"): _DGRAD_N, (4096, 1024, 16231, "f32"): _DGRAD_N, (4096, 1024, 16384, "f32"): _DGRAD_N,
        (5120, 1024, 16172, "f32"): _DGRAD_N, (5120, 1024, 16231, "f32"): _DGRAD_N, (5120, 1024, 16384, "f32"): _DGRAD_N,
        **{(K, N, T, "f32"): _DGRAD_N_TMA_LEAN_FREE for K, N in ((4096, 512), (5120, 512), (6144, 576))
           for T in _DGRAD_WINDOW_TOKENS},
    },
    ("sm_100a", 152): {
        (4096, 1024, 16172, "f32"): _DGRAD_N_TMA, (4096, 1024, 16231, "f32"): _DGRAD_N_TMA,
        (4096, 1024, 16384, "f32"): _DGRAD_N_TMA, (5120, 1024, 16172, "f32"): _DGRAD_N_TMA,
        (5120, 1024, 16231, "f32"): _DGRAD_N_TMA, (5120, 1024, 16384, "f32"): _DGRAD_N_TMA,
        (4096, 512, 16172, "f32"): _DGRAD_N_TMA, (4096, 512, 16231, "f32"): _DGRAD_N_TMA,
        (4096, 512, 16384, "f32"): _DGRAD_N_TMA, (5120, 512, 16172, "f32"): _DGRAD_N_TMA,
        (5120, 512, 16231, "f32"): _DGRAD_N_TMA, (5120, 512, 16384, "f32"): _DGRAD_N_TMA,
    },
    ("sm_103a", 152): {
        (4096, 1024, 16172, "f32"): _DGRAD_N, (4096, 1024, 16231, "f32"): _DGRAD_N, (4096, 1024, 16384, "f32"): _DGRAD_N,
        (5120, 1024, 16172, "f32"): _DGRAD_N, (5120, 1024, 16231, "f32"): _DGRAD_N, (5120, 1024, 16384, "f32"): _DGRAD_N,
        **{(K, N, T, "f32"): _DGRAD_N_TMA_LEAN_FREE for K, N in ((4096, 512), (5120, 512), (6144, 576))
           for T in _DGRAD_WINDOW_TOKENS},
    },
}
# fmt: on

_NVJET_EARLY_LEAN = "dense_proj_gemm_nvjet_early_kn_n256_lean"
_NVJET_EARLY_LEAN_FREE = "dense_proj_gemm_nvjet_early_kn_n256_lean_free"
#: The deterministic package's program on whose plan the rows of :data:`K1_DGRAD_NVJET_RULES_BY_SM` run this
#: package's program (a template symbol without the ``dense_proj_det_gemm_`` prefix).
K1_DGRAD_NVJET_DET_PROGRAM = "kn_n256_tma1_w8r"
#: The ``(K, N, T)`` rows (BF16 out) measured for :data:`K1_DGRAD_NVJET_RULES_BY_SM`.
# fmt: off
_NVJET_DGRAD_ROWS: tuple[tuple[int, int, int], ...] = (
    *((4096, n, t) for n in (32, 64) for t in (512, 2048, 4864, 9472, 9728, 14592, 16172, 16231, 16384)),
    *((4096, n, 16231) for n in (16, 48, 96, 128)),
    *((k, n, 16231) for k, n in ((5120, 32), (5120, 48), (5120, 64), (5120, 128), (6144, 32), (6144, 128), (8192, 128))),
)
#: The rows of :data:`_NVJET_DGRAD_ROWS` that 152-SM ``sm_103a`` parts, and 148-SM and 152-SM ``sm_100a`` parts,
#: leave to the other tables.
_NVJET_DGRAD_SM103_KEPT = frozenset({(4096, 32, 2048)})
_NVJET_DGRAD_SM100_KEPT = _NVJET_DGRAD_SM103_KEPT | {(4096, 32, 14592), (4096, 64, 4864)}
_GLM3 = (16172, 16231, 16384)
#: The ``(K, N, T)`` rows (BF16 out) per part that run ``dense_proj_gemm_nvjet_early_kn_n256_lean_free`` in
#: :data:`K1_DGRAD_NVJET_RULES_BY_SM`.
_NVJET_FREE_ROWS: dict[tuple[str, int], tuple[tuple[int, int, int], ...]] = {
    ("sm_100a", 148): (
        *((k, n, t) for k, n in ((4096, 32), (4096, 48), (4096, 128), (5120, 32), (6144, 32), (6144, 128)) for t in _GLM3),
        *((k, n, t) for k, n in ((5120, 48), (5120, 64), (5120, 128), (8192, 128)) for t in (16172, 16384)),
    ),
    ("sm_103a", 152): (
        *((k, n, t) for k, n in ((4096, 128), (5120, 128), (6144, 32), (6144, 128)) for t in _GLM3),
        *((k, n, t) for k, n in ((4096, 48), (5120, 32), (5120, 48), (5120, 64), (8192, 128)) for t in (16172, 16384)),
    ),
    ("sm_107a", 212): tuple(
        (k, n, t)
        for k, n in ((4096, 32), (4096, 48), (4096, 64), (4096, 128), (5120, 32), (5120, 48), (5120, 64), (5120, 128),
                     (6144, 32), (6144, 128), (8192, 128))
        for t in _GLM3
    ),
}
# fmt: on
#: The K1 input-gradient rows that run this package's nvjet program on the deterministic package's plan of
#: :data:`K1_DGRAD_NVJET_DET_PROGRAM`, ``(arch, sm_count) -> {(K, N, T, out_dtype): template}``: on a part of exactly
#: ``sm_count`` SMs, :func:`cake_dense_gemm_nondet_train_projection_dgrad` of ``G[T, N]`` and ``W[N, K]`` (both with a
#: unit stride on their last axis) into an ``out[T, K]`` of ``out_dtype`` at exactly these extents and token count
#: plans that program of the deterministic package (the knobs that name it, :func:`k1_program_knobs`) and launches this
#: package's program ``template`` with the plan's arguments: the same kernel parameters, host plan, shared memory,
#: cluster and grid, and the same bits as the deterministic plan.  The program (``nvjet_early_kn_n256_lean``) loads the
#: first item's first K step and prefetches the boxes after it into the L2 in the prologue, and takes the start path of
#: cuBLAS's nvjet kernels (a relaxed cluster arrive, each role's wait just before its first operation that needs the
#: peer CTA, the pair's tensor-memory allocation at the constant base 0).  It comes after
#: :data:`K1_DGRAD_PROGRAM_RULES_BY_SM` and before :data:`K1_DGRAD_RULES_BY_SM`; every other part, row, token count and
#: output dtype keeps those tables and the deterministic plan.  Measured at exactly these rows and token counts against
#: the deterministic program it replaces in the same process, on one B200, one GB200 and one GB300 part: faster in eager
#: launches and in CUDA graph replays in 5 of 5 paired groups (that program's time 1.00 to 1.17 times this one's; on the
#: GB300 part ``4096 -> 32`` at 14592 tokens in 9 of 10 eager groups), the same bits on every repeat, equal to the
#: deterministic plan's.  ``4096 -> 32`` at 2048 tokens was slower in eager launches on all three parts (that
#: program's time 0.96 to 0.99 times this one's) and keeps the other tables on every part; on ``sm_100a``
#: ``4096 -> 32`` at 14592 tokens and ``4096 -> 64`` at 4864 tokens were not faster in every group on the GB200 part and
#: keep the other tables too.
#:
#: The rows that run the program with the epilogue freeing the tensor memory under the last store drain
#: (``dense_proj_gemm_nvjet_early_kn_n256_lean_free``, :data:`_NVJET_FREE_ROWS`; the same items, MMAs and bits) on
#: 148-SM ``sm_100a``, 152-SM ``sm_103a`` and 212-SM ``sm_107a`` parts: measured at exactly these rows and token counts
#: against the package's former route of the row in the same process, 17 paired groups in eager launches and in CUDA
#: graph replays, faster in 17 of 17 groups in both: on a B200 part that route took 1.006 to 1.043 times its time, on a
#: GB300 part 1.005 to 1.053 times, on an R200 part 1.036 to 1.108 times (and 1.07 to 1.11 times on two more cards for
#: 8 rows).  The other rows of those screens (within 0.998 to 1.014 times) keep their routes.
# fmt: off
K1_DGRAD_NVJET_RULES_BY_SM: dict[tuple[str, int], dict[tuple[int, int, int, str], str]] = {
    ("sm_100a", 148): {
        **{(*row, "bf16"): _NVJET_EARLY_LEAN for row in _NVJET_DGRAD_ROWS if row not in _NVJET_DGRAD_SM100_KEPT},
        **{(*row, "bf16"): _NVJET_EARLY_LEAN_FREE for row in _NVJET_FREE_ROWS[("sm_100a", 148)]},
    },
    ("sm_100a", 152): {
        (*row, "bf16"): _NVJET_EARLY_LEAN for row in _NVJET_DGRAD_ROWS if row not in _NVJET_DGRAD_SM100_KEPT
    },
    ("sm_103a", 152): {
        **{(*row, "bf16"): _NVJET_EARLY_LEAN for row in _NVJET_DGRAD_ROWS if row not in _NVJET_DGRAD_SM103_KEPT},
        **{(*row, "bf16"): _NVJET_EARLY_LEAN_FREE for row in _NVJET_FREE_ROWS[("sm_103a", 152)]},
    },
    ("sm_107a", 212): {(*row, "bf16"): _NVJET_EARLY_LEAN_FREE for row in _NVJET_FREE_ROWS[("sm_107a", 212)]},
}
# fmt: on

#: Rows of the one-wave forward program per work item (512 output features), and the input-gradient program's tile.
_WAVE_ITEM_ROWS = 512
_DGRAD_ITEM_ROWS = 256


def _k1_program_module(table: dict, key: tuple, out: torch.Tensor) -> Optional[str]:
    """The registered module of this package that ``table`` (a table of :data:`K1_FORWARD_PROGRAM_RULES_BY_SM` /
    :data:`K1_DGRAD_PROGRAM_RULES_BY_SM` form) names for ``key`` on ``out``'s device, or ``None`` (no entry for the
    part, or a template this checkout does not register for its architecture)."""
    facts = deterministic_backend().device_facts(out.device)
    if facts.arch is None:
        return None
    template = table.get((facts.arch, int(facts.sm_count)), {}).get(key)
    if template is None:
        return None
    return cake_jit.KERNELS.get(facts.arch, {}).get(template)


def _k1_forward_program_of(
    X: torch.Tensor, W: torch.Tensor, out: torch.Tensor
) -> Optional[str]:
    """The module of this package the forward call ``(X, W, out)`` runs (:data:`K1_FORWARD_PROGRAM_RULES_BY_SM`), or
    ``None``: a shape without a row of the table, operands that are not BF16 and K-major on one CUDA device, an
    ``out`` that is not BF16 with a unit inner stride, and a part without the row keep the deterministic routing.  The
    device is read only for a shape of the table."""
    if X.dim() != 2 or W.dim() != 2 or out.dim() != 2:
        return None
    key = (int(X.shape[1]), int(W.shape[0]), int(X.shape[0]))
    if not any(key in rows for rows in K1_FORWARD_PROGRAM_RULES_BY_SM.values()):
        return None
    if (
        X.dtype != torch.bfloat16
        or W.dtype != torch.bfloat16
        or out.dtype != torch.bfloat16
    ):
        return None
    if (
        X.stride(1) != 1
        or W.stride(1) != 1
        or out.stride(1) != 1
        or int(W.shape[1]) != key[0]
    ):
        return None
    if tuple(out.shape) != (key[2], key[1]) or _k1_index(X, W, out) < 0:
        return None
    return _k1_program_module(K1_FORWARD_PROGRAM_RULES_BY_SM, key, out)


def _k1_dgrad_program_of(
    G: torch.Tensor, W: torch.Tensor, out: torch.Tensor
) -> Optional[str]:
    """The module of this package the input-gradient call ``(G, W, out)`` runs (:data:`K1_DGRAD_PROGRAM_RULES_BY_SM`),
    or ``None``: a shape and output dtype without a row of the table, operands that are not BF16 with a unit stride on
    their last axis on one CUDA device, an ``out`` without a unit inner stride, and a part without the row keep the
    deterministic routing.  The device is read only for a shape of the table."""
    if (
        G.dim() != 2
        or W.dim() != 2
        or out.dim() != 2
        or out.dtype not in (torch.bfloat16, torch.float32)
    ):
        return None
    key = (
        int(W.shape[1]),
        int(W.shape[0]),
        int(G.shape[0]),
        "f32" if out.dtype == torch.float32 else "bf16",
    )
    if not any(key in rows for rows in K1_DGRAD_PROGRAM_RULES_BY_SM.values()):
        return None
    if (
        G.dtype != torch.bfloat16
        or W.dtype != torch.bfloat16
        or int(G.shape[1]) != key[1]
    ):
        return None
    if (
        G.stride(1) != 1
        or W.stride(1) != 1
        or out.stride(1) != 1
        or tuple(out.shape) != (key[2], key[0])
    ):
        return None
    if _k1_index(G, W, out) < 0:
        return None
    return _k1_program_module(K1_DGRAD_PROGRAM_RULES_BY_SM, key, out)


def _k1_dgrad_nvjet_of(
    G: torch.Tensor, W: torch.Tensor, out: torch.Tensor
) -> Optional[str]:
    """The module of this package the input-gradient call ``(G, W, out)`` runs on the deterministic package's plan of
    :data:`K1_DGRAD_NVJET_DET_PROGRAM` (:data:`K1_DGRAD_NVJET_RULES_BY_SM`), or ``None``: a shape and output dtype
    without a row of the table, an operand without a unit stride on its last axis, tensors not on one CUDA device and a
    part without the row keep the other tables.  The device is read only for a shape of the table."""
    if (
        G.dim() != 2
        or W.dim() != 2
        or out.dim() != 2
        or out.dtype not in (torch.bfloat16, torch.float32)
    ):
        return None
    key = (
        int(W.shape[1]),
        int(W.shape[0]),
        int(G.shape[0]),
        "f32" if out.dtype == torch.float32 else "bf16",
    )
    if not any(key in rows for rows in K1_DGRAD_NVJET_RULES_BY_SM.values()):
        return None
    if G.stride(1) != 1 or W.stride(1) != 1 or _k1_index(G, W, out) < 0:
        return None
    return _k1_program_module(K1_DGRAD_NVJET_RULES_BY_SM, key, out)


def _launch_k1_program(
    module: str,
    values: dict[str, Any],
    items: int,
    out: torch.Tensor,
    enable_pdl: Optional[bool],
    units: Optional[int] = None,
    sources: Optional[tuple] = None,
    head: Optional[tuple] = None,
) -> None:
    """Launch this package's program ``module`` (a forward / input-gradient program of the tables above, one CTA pair
    per SM pair up to ``items`` work items) with ``values`` in its argument order, the PDL flag of ``enable_pdl`` for
    ``out``'s device, on the current stream of that device.  ``units``: the stream-K program's unit count, one unit
    per CTA pair; a grid of another number of CTA pairs raises ``ValueError`` (a pair running two units could wait on
    a higher pair that is not resident).  With an entry point's ``head`` flags and its tensors ``sources`` (the last
    is ``out``) the launch is bound and cached in :data:`_K1_PROGRAM_CACHE` under the entry point's key
    (:func:`_k1_cached_launch` launches it on a repeat call) and launched through that binding; a view the call's
    tensors do not rebuild launches unbound, as without them."""
    det = deterministic_backend()
    index = _k1_index(out, out, out)
    facts = det.device_facts(out.device)
    stream = det._current_raw_stream(index)
    grid = (2 * max(1, min(int(facts.sm_count) // 2, int(items))), 1, 1)
    if units is not None and grid[0] != 2 * int(units):
        raise ValueError(
            f"stream-K launch of {units} units on {grid[0] // 2} CTA pairs: the grid runs one unit per CTA pair"
        )
    fn, plan = _atomic_entry(module, grid, facts.arch)
    pdl = int(det.pdl_enabled(enable_pdl, index))
    values = dict(values, enable_pdl=pdl)
    if head is not None and sources is not None:
        bound = _k1_program_binding(module, plan, values, grid, sources)
        if bound is not None:
            key = _k1_key(index, stream, (*head, pdl, *_K1_NOT_STATIC), *sources)
            with _ATOMIC_PLAN_LOCK:
                _cache_insert(
                    _K1_PROGRAM_CACHE,
                    key,
                    _CachedProgramK1(module_name=module, entry=fn, k1=bound),
                )
            _launch_k1_binding(det, fn, bound, None, sources, index, stream)
            return
    arguments = _order_atomic_arguments(module, plan, values, grid)
    with _k1_stream_context(det, index, stream):
        fn(*arguments)


def _k1_program_binding(
    module: str,
    arg_plan: list,
    values: dict[str, Any],
    grid: tuple[int, int, int],
    sources: tuple,
) -> Optional["_K1Binding"]:
    """The :class:`_K1Binding` of this package's program ``module`` for a call of the tensors ``sources``: ``values``
    in the program's order, every tensor argument that starts at the data pointer of one of ``sources`` rebuilt per
    call (a ``buffer`` argument as a pointer), the other tensors (the deterministic package's per-device dummies)
    converted once; ``None`` when a view is not rebuilt from ``sources`` (:func:`_k1_view_recipe`)."""
    pointers = {t.data_ptr() for t in sources}
    marked, views = dict(values), {}
    for kind, name in arg_plan:
        value = values.get(name)
        if kind in ("tma_buffer", "buffer") and isinstance(value, torch.Tensor):
            if value.data_ptr() in pointers:
                marked[name] = _Arg(name)
                views[name] = (value, kind == "buffer")
    template = _order_atomic_arguments(module, arg_plan, marked, grid)
    return _k1_binding(template, views, sources)


def _launch_wave_forward(
    module: str,
    X: torch.Tensor,
    W: torch.Tensor,
    out: torch.Tensor,
    enable_pdl: Optional[bool],
    head: Optional[tuple] = None,
) -> torch.Tensor:
    """``out[T, N] = X[T, K] @ W[N, K].T`` on the one-wave forward program ``module``: ``W`` / ``X`` / ``out`` as
    batches of one (``[1, N, K]``, ``[1, T, K]``, ``[1, T, N]``), ``M = N`` features and ``N = T`` tokens of the
    program; with the entry point's ``head`` flags bound and cached (:func:`_launch_k1_program`)."""
    tile_n = int(MODULES[module]["build"]["kwargs"]["tile_n"])
    F, K, T = int(W.shape[0]), int(W.shape[1]), int(X.shape[0])
    items = _ceil_div(F, _WAVE_ITEM_ROWS) * _ceil_div(T, tile_n)
    values = dict(
        A=W.unsqueeze(0), B=X.unsqueeze(0), C=out.unsqueeze(0), M=F, N=T, K=K, L=1
    )
    _launch_k1_program(
        module, values, items, out, enable_pdl, sources=(X, W, out), head=head
    )
    return out


def _launch_dgrad_program(
    module: str,
    G: torch.Tensor,
    W: torch.Tensor,
    out: torch.Tensor,
    enable_pdl: Optional[bool],
    head: Optional[tuple] = None,
) -> torch.Tensor:
    """``out[T, K] = G[T, N] @ W[N, K]`` on the input-gradient program ``module``: the program's ``M = T`` rows,
    ``N = K`` columns and ``K = N`` reduction, ``W`` as ``[1, N, K]``; ``out`` stored through its own pointer and row
    stride (a padded ``out`` as the contiguous 1-D alias of its span, ``flat_alias``: the binding checks the
    pointer's contiguity), and through ``[1, T, K]`` TMA maps (the map of the other dtype a per-device dummy the
    program never touches).  With the entry point's ``head`` flags bound and cached (:func:`_launch_k1_program`)."""
    det = deterministic_backend()
    T, N, K = int(G.shape[0]), int(W.shape[0]), int(W.shape[1])
    f32 = out.dtype == torch.float32
    other = torch.bfloat16 if f32 else torch.float32
    spare, flat = det._dummy_map(out.device, other), det._dummy(out.device, other)
    ptr = out if out.is_contiguous() else det.flat_alias(out)
    items = _ceil_div(T, _DGRAD_ITEM_ROWS) * _ceil_div(K, _DGRAD_ITEM_ROWS)
    values = dict(
        A=G,
        B=W.unsqueeze(0),
        out=flat if f32 else ptr,
        out32=ptr if f32 else flat,
        C16=spare if f32 else out.unsqueeze(0),
        C32=out.unsqueeze(0) if f32 else spare,
        M=T,
        N=K,
        K=N,
        ldo=int(out.stride(0)),
    )
    _launch_k1_program(
        module, values, items, out, enable_pdl, sources=(G, W, out), head=head
    )
    return out


#: The registry template of this package's stream-K weight-gradient program of ``block_n``-wide transposed-store tiles
#: (:data:`K1_WGRAD_SK_RULES_BY_SM`).
def wgrad_sk_template(block_n: int, out_f32: bool) -> str:
    return (
        f"{cake_jit.TEMPLATE_PREFIXES[10]}nn_n{int(block_n)}"
        + ("_f32" if out_f32 else "")
        + "_t"
    )


_SK_N256_T = "dense_proj_gemm_sk_nn_n256_t"
_SK_N256_F32_T = "dense_proj_gemm_sk_nn_n256_f32_t"
#: Flag words of one stream-K unit (one per epilogue warp slice and lane), zero before and after every launch.
_SK_FLAG_WORDS = 512

#: The K1 weight-gradient rows that run this package's stream-K program, ``(arch, sm_count) -> {(d_in, d_out, T,
#: out_dtype): template}``: on a part of exactly ``sm_count`` SMs, :func:`cake_dense_gemm_nondet_train_projection_wgrad`
#: of ``G[T, d_out]`` and ``X[T, d_in]`` (both with a unit stride on their last axis) into an ``out[d_out, d_in]`` of
#: ``out_dtype`` (unit inner stride) at exactly these extents and token count, with the default ``sk`` and
#: ``cta_rows``, runs the stream-K program ``template`` (:func:`wgrad_sk_template`): the ``ceil(d_in / 256) x
#: ceil(d_out / block_n)`` output tiles laid end to end (``d_out`` tiles fastest) on one line of 64-token K blocks, cut
#: into one unit per CTA pair (``sm_count / 2`` units, the grid one CTA pair per unit); each pair runs its unit in
#: increasing order, and a tile split across units is summed by the unit that holds its last K block, which adds the
#: other units' FP32 partials in unit order and rounds once: no atomic, the same bits on every run (other bits than the
#: deterministic program's).  That unit waits only for the partials of the lower units that hold the tile's earlier K
#: blocks, and the other units wait for none, so a launch whose CTA pairs are not all resident at once still completes
#: when the hardware starts lower CTA pairs no later than higher ones (observed on these parts; CUDA does not guarantee
#: that order).  It comes before every other weight-gradient table; every other part, row, token count and knob keeps
#: them.  Measured at exactly these rows and token counts on two 148-SM parts against the fastest non-Cake program of
#: the row (cuBLAS, cuBLASLt with split K, DeepGEMM, QuACK) and against this package's program of the row, in the same
#: process: 5.0 to 7.6 % faster than the former and 4.3 to 7.8 % faster than the latter in eager launches and in CUDA
#: graph replays in every paired group, the same bits on 20 repeats, within 1.1 times the reference's median error on
#: the six input kinds of :data:`K1_DSM_RANGES_BY_SM`'s check.  On a 152-SM ``sm_103a`` part (76 units) the FP32
#: program at ``5120 -> 1024`` and 16384 tokens, measured the same way on one GB300 node: 6.0 to 6.7 % faster than the
#: fastest non-Cake program and 3.3 to 4.2 % faster than this package's program of the row, the same bits on 20
#: repeats and within the same check.  The BF16 program at 16172 tokens and the FP32 program at 16231 tokens, on two
#: GB300 nodes: this package's program of the row took 1.018 to 1.047 times their median time in eager launches and in
#: CUDA graph replays, slower in at most 2 of 15 paired groups on one node and 1 of 7 on the other on every GPU, the
#: same bits on 20 repeats and within the same check.  The FP32 program at 16100, 16172 and 16300 tokens, on one GB300
#: node (four GPUs, 17 paired groups each): the deterministic program the row ran before took 1.030 to 1.070 times its
#: median time in eager launches and in CUDA graph replays on every GPU.  The other BF16 token counts keep this
#: package's program on ``sm_103a``: the stream-K program missed that group count there.  On a 212-SM ``sm_107a`` part
#: (106 units) the BF16 program at ``4096 -> 1024`` and 16172 and 16384 tokens, on two parts: this package's program of
#: the row took 1.017 to 1.042 times its median time in eager launches and in CUDA graph replays, slower in 0 of 7
#: paired groups, the same bits on 20 repeats and within the same check; 16231 tokens keep this package's program.
# fmt: off
K1_WGRAD_SK_RULES_BY_SM: dict[tuple[str, int], dict[tuple[int, int, int, str], str]] = {
    ("sm_100a", 148): {
        (5120, 1024, 16172, "bf16"): _SK_N256_T, (5120, 1024, 16231, "bf16"): _SK_N256_T,
        (5120, 1024, 16384, "bf16"): _SK_N256_T, (5120, 1024, 16172, "f32"): _SK_N256_F32_T,
        (5120, 1024, 16231, "f32"): _SK_N256_F32_T, (5120, 1024, 16384, "f32"): _SK_N256_F32_T,
    },
    ("sm_103a", 152): {
        (5120, 1024, 16172, "bf16"): _SK_N256_T, (5120, 1024, 16231, "f32"): _SK_N256_F32_T,
        (5120, 1024, 16384, "f32"): _SK_N256_F32_T, (5120, 1024, 16100, "f32"): _SK_N256_F32_T,
        (5120, 1024, 16172, "f32"): _SK_N256_F32_T, (5120, 1024, 16300, "f32"): _SK_N256_F32_T,
    },
    ("sm_107a", 212): {
        (4096, 1024, 16172, "bf16"): _SK_N256_T, (4096, 1024, 16384, "bf16"): _SK_N256_T,
    },
}
# fmt: on


def _k1_wgrad_sk_program_of(
    G: torch.Tensor,
    X: torch.Tensor,
    out: torch.Tensor,
    sk: Any,
    cta_rows: Optional[int],
) -> Optional[str]:
    """The stream-K module of this package the weight-gradient call ``(G, X, out)`` runs
    (:data:`K1_WGRAD_SK_RULES_BY_SM`), or ``None``: a call with a non-default ``sk`` / ``cta_rows``, a shape and output
    dtype without a row of the table, operands that are not BF16 with a unit stride on their last axis on one CUDA
    device, an ``out`` of another shape or without a unit inner stride, and a part without the row keep the other
    tables.  The device is read only for a shape of the table."""
    if (
        sk != "auto"
        or cta_rows is not None
        or G.dim() != 2
        or X.dim() != 2
        or out.dim() != 2
    ):
        return None
    if out.dtype not in (torch.bfloat16, torch.float32):
        return None
    T, d_out, d_in = int(G.shape[0]), int(G.shape[1]), int(X.shape[1])
    key = (d_in, d_out, T, "f32" if out.dtype == torch.float32 else "bf16")
    if not any(key in rows for rows in K1_WGRAD_SK_RULES_BY_SM.values()):
        return None
    if G.dtype != torch.bfloat16 or X.dtype != torch.bfloat16 or int(X.shape[0]) != T:
        return None
    if (
        G.stride(1) != 1
        or X.stride(1) != 1
        or out.stride(1) != 1
        or tuple(out.shape) != (d_out, d_in)
    ):
        return None
    if _k1_index(X, G, out) < 0:
        return None
    return _k1_program_module(K1_WGRAD_SK_RULES_BY_SM, key, out)


def _launch_wgrad_sk(
    module: str,
    G: torch.Tensor,
    X: torch.Tensor,
    out: torch.Tensor,
    enable_pdl: Optional[bool],
) -> torch.Tensor:
    """``out[d_out, d_in] = G[T, d_out].T @ X[T, d_in]`` on the stream-K program ``module`` (transposed store: ``A = X``,
    ``B = G`` as batches of one, ``M = d_in``, ``N = d_out``, ``K = T``): one unit per CTA pair of the part (at most
    one per 64-token K block of the line), the grid one CTA pair per unit (checked: :func:`_launch_k1_program`), the
    unit length the line's K blocks over the units, rounded up; the units' FP32 partial slots and flag words from the
    arena of the (device, stream) (``k1_slots`` / ``k1_tickets``: the flag words are zero before and after every
    complete launch).  ``out`` is stored through its pointer and row stride (a padded ``out`` as the contiguous 1-D
    alias of its span, ``flat_alias``)."""
    det = deterministic_backend()
    index = _k1_index(out, out, out)
    stream = det._current_raw_stream(index)
    facts = det.device_facts(out.device)
    block_n = int(MODULES[module]["build"]["kwargs"]["block_n"])
    T, d_out, d_in = int(G.shape[0]), int(G.shape[1]), int(X.shape[1])
    line = (
        _ceil_div(d_in, K1_PAIR_ROWS)
        * _ceil_div(d_out, block_n)
        * _ceil_div(T, K1_BLOCK_K)
    )
    units = max(1, min(int(facts.sm_count) // 2, line))
    arena = _atomic_arena_for(index, stream, out.device)
    with _ATOMIC_PLAN_LOCK:
        ws = arena.require("k1_slots", units * K1_PAIR_ROWS * block_n)
        flags = arena.require("k1_tickets", units * _SK_FLAG_WORDS)
    values = dict(
        A=X.unsqueeze(0),
        B=G.unsqueeze(0),
        out=out if out.is_contiguous() else det.flat_alias(out),
        ws=ws,
        flags=flags,
        M=d_in,
        N=d_out,
        K=T,
        num_l=1,
        num_units=units,
        unit_len=_ceil_div(line, units),
        ldo=int(out.stride(0)),
        out_l=d_out * int(out.stride(0)),
    )
    _launch_k1_program(module, values, units, out, enable_pdl, units=units)
    return out


#: The K1 weight-gradient rows this package runs on another program of the deterministic package than its own plan of
#: the row, where :func:`plan_atomic_gemm` keeps the row on the deterministic program, ``(arch, sm_count) -> {(d_in,
#: d_out, T, out_dtype): program}``: on a part of exactly ``sm_count`` SMs,
#: :func:`cake_dense_gemm_nondet_train_projection_wgrad` of ``G[T, d_out]`` and ``X[T, d_in]`` into an ``out[d_out,
#: d_in]`` of ``out_dtype`` at exactly these extents and token count, with the default ``sk`` and ``cta_rows``, runs the
#: deterministic package's program ``program`` (its template without the ``dense_proj_det_gemm_`` prefix) in that
#: package's orientation of the row, planned with the knobs that name it (:func:`k1_program_knobs`); every other part,
#: row, token count and knob runs the deterministic plan.  These programs split a tile's token reduction in two ranges
#: and add the two partials in a fixed order (``_sks_sb2``): the same bits on every run, other bits than the
#: deterministic plan (one chain over the tokens).  Measured on a 148-SM part at exactly these rows and token counts,
#: against the deterministic plan in the same process: faster in eager launches and in CUDA graph replays in 5 of 5
#: paired groups of a screen (1.1 to 3.2 %), within the deterministic package's error budgets against FP64, and the same
#: bits on 20 repeats.
# fmt: off
K1_WGRAD_DET_RULES_BY_SM: dict[tuple[str, int], dict[tuple[int, int, int, str], str]] = {
    ("sm_100a", 148): {
        (5120, 1024, 16172, "bf16"): "nn_n192_bz64_hne_sks_sb2_pro15",
        (5120, 1024, 16231, "bf16"): "nn_n192_bz64_hne_sks_sb2_pro15",
        (5120, 1024, 16384, "bf16"): "nn_n192_bz64_hne_sks_sb2_pro15",
        (5120, 1024, 16172, "f32"): "nn_n192_bz64_hne_f32_v8_ef_sks_sb2_pro15",
        (5120, 1024, 16231, "f32"): "nn_n192_bz64_hne_f32_v8_ef_sks_sb2_pro15",
        (5120, 1024, 16384, "f32"): "nn_n192_bz64_hne_f32_v8_ef_sks_sb2_pro15",
    },
}
# fmt: on


def k1_wgrad_det_knobs(
    *, arch: str, sm_count: int, d_in: int, d_out: int, T: int, out_f32: bool
) -> tuple:
    """The deterministic package's planner knobs of this package's deterministic weight gradient ``out[d_out, d_in] =
    G[T, d_out].T @ X[T, d_in]`` on a device of exactly ``sm_count`` SMs: those of the program of the row's entry in the
    table of :data:`K1_WGRAD_DET_RULES_BY_SM` for ``(arch, sm_count)`` at exactly ``T`` tokens
    (:func:`k1_program_knobs`), or ``()`` -- the deterministic plan."""
    program = K1_WGRAD_DET_RULES_BY_SM.get((arch, int(sm_count)), {}).get(
        (int(d_in), int(d_out), int(T), "f32" if out_f32 else "bf16")
    )
    return k1_program_knobs(program) if program else ()


def atomic_k1_template(
    block_n: int, out_f32: bool, chains: int = 1, split_order: str = "arrival"
) -> str:
    """The registry template of this package's K1 program (both operands MN-major): the transposed-store program of
    the tile width, or the row-major-store one for a width of :data:`K1_ROW_MAJOR_BLOCK_N`; ``_c<chains>`` for more
    than one accumulation chain; ``_<split_order>`` for a split order of :data:`K1_SPLIT_ORDERS` other than
    ``"arrival"``."""
    if split_order not in K1_SPLIT_ORDERS:
        raise ValueError(
            f"split_order is one of {K1_SPLIT_ORDERS}, got {split_order!r}"
        )
    return (
        f"{cake_jit.TEMPLATE_PREFIXES[0]}nn_n{int(block_n)}"
        + ("_f32" if out_f32 else "")
        + ("" if int(block_n) in K1_ROW_MAJOR_BLOCK_N else "_t")
        + (f"_c{int(chains)}" if int(chains) > 1 else "")
        + ("" if split_order == "arrival" else f"_{split_order}")
    )


def dsm_k1_template(block_n: int, splits: int) -> str:
    """The registry template of this package's K1 cluster program (both operands MN-major, FP32 out, transposed
    store): ``block_n`` ``d_out`` rows per tile and ``splits`` CTAs per cluster, one K range each
    (:data:`K1_DSM_RANGES_BY_SM`)."""
    return f"{cake_jit.TEMPLATE_PREFIXES[7]}nn_n{int(block_n)}_f32_t_s{int(splits)}"


# ---------------------------------------------------------------------------
# K1 planning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CakeDenseGemmNondetProjectionPlan:
    """The host plan of one K1 launch of this package: the transposed-store GEMM ``out[l, n, m] = sum_k A[l, k, m]
    B[l, k, n]`` (``out_t``), or the row-major-store GEMM ``out[l, m, n]`` (``out_t`` false: the widths of
    :data:`K1_ROW_MAJOR_BLOCK_N`), over the launch's ``[L, K, M]`` / ``[L, K, N]`` operand views, ``L x m_tiles x
    n_tiles`` tiles of ``256 x block_n`` each cut into ``splits`` K ranges (or, at one split, its K blocks into
    ``chains`` accumulation chains), walked persistently by at most ``sm_pairs`` CTA pairs.  With ``dsm``, the
    cluster program instead (:func:`k1_dsm_plan`; transposed store, FP32 out): ``L x m_tiles x n_tiles`` tiles of
    ``128 x block_n``, each one cluster of ``splits`` CTAs (a K range each, the partials summed in the cluster's
    shared memory), ``splits`` CTAs per tile and no workspace.  ``M`` and ``N`` are the launch's.
    ``transposed_out`` is the caller's: the launch takes the caller's
    operands exchanged (``swapped``) when its store form differs from the caller's output layout."""

    L: int
    M: int
    N: int
    K: int
    out_f32: bool
    transposed_out: bool
    block_n: int
    splits: int
    m_tiles: int
    n_tiles: int
    sm_pairs: int
    template: str
    out_t: bool = True
    chains: int = 1
    dsm: bool = False
    split_order: str = "arrival"

    @property
    def swapped(self) -> bool:
        return self.transposed_out != self.out_t

    @property
    def tiles(self) -> int:
        return self.L * self.m_tiles * self.n_tiles

    @property
    def items(self) -> int:
        return self.tiles * self.splits

    @property
    def reproducible(self) -> bool:
        """Whether the result repeats bit for bit: one split stores whole tiles (its chains added in order), two add
        ``own + other`` (one add of two values), the cluster program (``dsm``) adds its splits in cluster-rank order,
        and a fixed split order (``split_order`` other than ``"arrival"``) adds them in one order; three or more splits
        of the CTA-pair program in the arrival order add the partials in the order the splits finish."""
        return self.dsm or self.splits <= 2 or self.split_order != "arrival"

    @property
    def grid(self) -> tuple[int, int, int]:
        if self.dsm:
            return (self.items, 1, 1)
        return (2 * min(self.items, self.sm_pairs), 1, 1)

    @property
    def workspace(self) -> bool:
        """Whether the launch uses the split workspace and the tickets: the CTA-pair program at more than one split
        (the cluster program sums its splits in shared memory)."""
        return self.splits > 1 and not self.dsm

    @property
    def ws_f32_elems(self) -> int:
        """FP32 elements of the split workspace: one ``256 x block_n`` slot per tile and split (none with one
        split, nor for the cluster program).  A launch writes every slot it reads except the padding columns of a
        partial tile (``N`` not a multiple of ``block_n``), which the last split reads and discards (they never reach
        ``out``); the workspace is zeroed when allocated, so no read is of uninitialised memory, and needs no memset
        per launch."""
        return self.items * K1_PAIR_ROWS * self.block_n if self.workspace else 0

    @property
    def counters_u32(self) -> int:
        """Ticket words: one per tile, epilogue warp slice and lane (none with one split, nor for the cluster
        program).  Zero when allocated; every complete launch leaves them zero."""
        return self.tiles * K1_COUNTER_WORDS if self.workspace else 0


def _k1_out_strides(out3: torch.Tensor) -> tuple[int, int]:
    """``(ldo, out_l)`` of the batched output view (the deterministic package's convention: a batch of one whose
    batch stride does not span the matrix passes the matrix span)."""
    ldo, batch = int(out3.stride(1)), int(out3.stride(0))
    span = int(out3.shape[1]) * ldo
    if int(out3.shape[0]) == 1 and batch < span:
        batch = span
    return ldo, batch


_K1_INT_MAX = 1 << 30


def plan_atomic_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    *,
    transposed_out: bool,
    arch: str,
    sm_count: int,
    splits: Optional[int] = None,
    block_n: Optional[int] = None,
    chains: Optional[int] = None,
) -> tuple[Optional[CakeDenseGemmNondetProjectionPlan], Any, Any, Any, str]:
    """Plan this package's K1 GEMM ``out = A @ B`` (``A [L, M, K]`` with unit stride on ``m``, ``B [L, K, N]`` with
    unit stride on ``n``; ``out [L, M, N]``, or ``[L, N, M]`` with ``transposed_out``).

    Returns ``(plan, a_desc, b_desc, out3, reason)``: ``a_desc`` / ``b_desc`` are the launch's ``[L, K, M]`` /
    ``[L, K, N]`` operand views (the caller's ``B`` and ``A`` when the plan is swapped) and ``out3`` the batched
    output.  ``plan`` is ``None`` when the row runs the deterministic program, and ``reason`` says why -- a view this
    package's program does not take (dtypes, a K-major operand, extents not multiples of 8, an output row or batch
    stride not a multiple of 8 elements, an output not 16-byte aligned), a template this checkout does not register
    for ``arch``, or the planner (:func:`k1_plan`).  An explicit ``splits`` / ``block_n`` replaces the planner's
    choice (the other one from the planner, else ``1`` split / :func:`k1_block_n`) and runs this package's program of
    that width (the row-major-store program for a width of :data:`K1_ROW_MAJOR_BLOCK_N`) whenever the view and the
    registry allow it.  The planner's choice is first :func:`k1_dsm_plan`'s cluster program where the row has a
    range of :data:`K1_DSM_RANGES_BY_SM` (a plan with ``dsm``; ``None`` when this checkout does not register its
    template for ``arch``; at an odd number of K blocks in a range of :data:`K1_DSM_ODD_BLOCK_RANGES_BY_SM` that
    table's program instead, :func:`k1_dsm_odd_block_plan`), then :func:`k1_parity_plan`'s program or the
    deterministic program where the row has a range of :data:`K1_PARITY_RANGES_BY_SM`, then
    :func:`k1_ordered_plan`'s program where the row has a range of :data:`K1_ORDERED_RANGES_BY_SM`, then
    :func:`k1_plan`'s, with :func:`k1_chain_plan`'s
    accumulation chains where it has them; an explicit choice runs one chain unless ``chains`` (one of :data:`K1_CHAINS_CHOICES`) names
    more, at one split, in the arrival order, and never the cluster program.
    Raises ``ValueError`` for inconsistent shapes (as the deterministic planner does) and for an explicit choice the
    program cannot run."""
    det = deterministic_backend()
    A3, B3, O3 = (
        det.as_batched(A, "A"),
        det.as_batched(B, "B"),
        det.as_batched(out, "out"),
    )
    L, M, K = (int(s) for s in A3.shape)
    Lb, Kb, N = (int(s) for s in B3.shape)
    if Lb != L or Kb != K:
        raise ValueError(
            f"cake_dense_gemm_nondet_train_projection: A is [L={L}, M={M}, K={K}] but B is [{Lb}, {Kb}, {N}]"
        )
    want = (L, N, M) if transposed_out else (L, M, N)
    if tuple(int(s) for s in O3.shape) != want:
        raise ValueError(
            f"cake_dense_gemm_nondet_train_projection: out must be {list(want)}, got {list(O3.shape)}"
        )
    none = (None, None, None, None)
    if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
        return (*none, "operands are not bf16")
    if out.dtype not in (torch.bfloat16, torch.float32):
        return (*none, f"out dtype {out.dtype}")
    if min(L, M, N, K) < 1:
        return (*none, "an empty extent")
    try:
        a_mn, a_desc = det.operand_view(A3, "A", k_axis=2)
        b_mn, b_desc = det.operand_view(B3, "B", k_axis=1)
    except ValueError as exc:
        return (*none, str(exc))
    if not (a_mn and b_mn):
        return (*none, "a K-major operand (not the nn layout)")
    # The transposed-store launch: M_ is the inner extent of the stored output, N_ its rows.
    if transposed_out:
        a_launch, b_launch, M_, N_ = a_desc, b_desc, M, N
    else:
        a_launch, b_launch, M_, N_ = b_desc, a_desc, N, M
    if M_ % 8 or N_ % 8:
        return (*none, "M or N not a multiple of 8")
    if max(M_, N_, K) > _K1_INT_MAX:
        return (*none, "an extent beyond 2**30")
    ldo, out_l = _k1_out_strides(O3)
    if O3.stride(2) != 1 or ldo % 8 or out_l % 8 or ldo < M_ or out.data_ptr() % 16:
        return (
            *none,
            "output view (unit inner stride, row and batch strides multiples of 8, 16-byte aligned)",
        )
    out_f32 = out.dtype == torch.float32
    c = 1 if chains is None else int(chains)
    order = "arrival"
    if splits is None and block_n is None and chains is None:
        dsm = k1_dsm_plan(
            arch=arch, L=L, M=M_, N=N_, K=K, out_f32=out_f32, sm_count=sm_count
        )
        odd = None
        if dsm is not None:
            odd = k1_dsm_odd_block_plan(
                arch=arch, L=L, M=M_, N=N_, K=K, out_f32=out_f32, sm_count=sm_count
            )
            if odd == ():
                return (*none, "the planner keeps the row on the deterministic program")
            if odd is not None and len(odd) == 2:
                # the cluster program of the odd-block table's tile
                dsm, odd = odd, None
        if dsm is not None and odd is None:
            bn, p = dsm
            template = dsm_k1_template(bn, p)
            if template not in cake_jit.KERNELS.get(arch, {}):
                return (*none, f"{template} is not registered for {arch}")
            plan = CakeDenseGemmNondetProjectionPlan(
                L=L,
                M=M_,
                N=N_,
                K=K,
                out_f32=out_f32,
                transposed_out=bool(transposed_out),
                block_n=bn,
                splits=p,
                m_tiles=_ceil_div(M_, K1_DSM_ROWS),
                n_tiles=_ceil_div(N_, bn),
                sm_pairs=max(1, int(sm_count) // 2),
                template=template,
                dsm=True,
            )
            return plan, a_launch, b_launch, O3, ""
        parity = None
        if odd is None:
            parity = k1_parity_plan(
                arch=arch, L=L, M=M_, N=N_, K=K, out_f32=out_f32, sm_count=sm_count
            )
            if parity == ():
                return (*none, "the planner keeps the row on the deterministic program")
        ordered = (
            odd
            or parity
            or k1_ordered_plan(
                arch=arch, L=L, M=M_, N=N_, K=K, out_f32=out_f32, sm_count=sm_count
            )
        )
        if ordered is not None:
            bn, p, c, order = ordered
        else:
            choice = k1_plan(
                arch=arch, L=L, M=M_, N=N_, K=K, out_f32=out_f32, sm_count=sm_count
            )
            if choice is None:
                return (*none, "the planner keeps the row on the deterministic program")
            bn, p = choice
            chained = k1_chain_plan(
                arch=arch, L=L, M=M_, N=N_, K=K, out_f32=out_f32, sm_count=sm_count
            )
            if chained is not None:
                c = chained[1]
    else:
        planned = k1_plan(
            arch=arch, L=L, M=M_, N=N_, K=K, out_f32=out_f32, sm_count=sm_count
        )
        if block_n is not None:
            bn = int(block_n)
        else:
            bn = planned[0] if planned is not None else k1_block_n(N_)
        if splits is not None:
            p = int(splits)
        else:
            p = 1 if c > 1 else planned[1] if planned is not None else 1
        if c not in K1_CHAINS_CHOICES:
            raise ValueError(f"chains is one of {K1_CHAINS_CHOICES}, got {chains}")
        if c > 1 and (p != 1 or not k1_chains_fit(c, K)):
            raise ValueError(
                f"cake_dense_gemm_nondet_train_projection: {c} chains run at one split (got {p}) with a 64-element K block each "
                f"(K={K})"
            )
        if bn not in K1_BLOCK_N_CHOICES:
            raise ValueError(f"block_n is one of {K1_BLOCK_N_CHOICES}, got {block_n}")
        if not k1_splits_fit(p, K) or p > 65536:
            raise ValueError(
                f"cake_dense_gemm_nondet_train_projection: splits must be in [1, {_ceil_div(K, K1_BLOCK_K)}] (the 64-element K blocks "
                f"of K={K}), got {p}"
            )
    template = atomic_k1_template(bn, out_f32, c, order)
    if template not in cake_jit.KERNELS.get(arch, {}):
        return (*none, f"{template} is not registered for {arch}")
    out_t = bn not in K1_ROW_MAJOR_BLOCK_N
    if not out_t:
        # the row-major-store launch: the transposed-store launch with its operands and extents exchanged
        a_launch, b_launch, M_, N_ = b_launch, a_launch, N_, M_
    plan = CakeDenseGemmNondetProjectionPlan(
        L=L,
        M=M_,
        N=N_,
        K=K,
        out_f32=out_f32,
        transposed_out=bool(transposed_out),
        block_n=bn,
        splits=p,
        m_tiles=_ceil_div(M_, K1_PAIR_ROWS),
        n_tiles=_ceil_div(N_, bn),
        sm_pairs=max(1, int(sm_count) // 2),
        template=template,
        out_t=out_t,
        chains=c,
        split_order=order,
    )
    return plan, a_launch, b_launch, O3, ""


def _k1_launch_views(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    plan: CakeDenseGemmNondetProjectionPlan,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The launch's ``[L, K, M]`` / ``[L, K, N]`` operand views and the batched output of a planned K1 launch,
    rebuilt from the caller's tensors (the geometry is the plan's, so its checks hold)."""
    det = deterministic_backend()
    _, a_desc = det.operand_view(det.as_batched(A, "A"), "A", k_axis=2)
    _, b_desc = det.operand_view(det.as_batched(B, "B"), "B", k_axis=1)
    out3 = det.as_batched(out, "out")
    return (b_desc, a_desc, out3) if plan.swapped else (a_desc, b_desc, out3)


def _k1_values(
    plan: CakeDenseGemmNondetProjectionPlan,
    *,
    A: Any,
    B: Any,
    out: Any,
    ws: Any,
    counters: Any,
    ldo: int,
    out_l: int,
    enable_pdl: int,
) -> dict[str, Any]:
    """The keyword arguments of a K1 launch of this package (ordered by the program's argument plan); ``enable_pdl``
    is the launch's PDL flag (0 or 1)."""
    return dict(
        A=A,
        B=B,
        out=out,
        ws=ws,
        counters=counters,
        M=plan.M,
        N=plan.N,
        K=plan.K,
        num_l=plan.L,
        num_splits=plan.splits,
        ldo=ldo,
        out_l=out_l,
        enable_pdl=enable_pdl,
    )


# ---------------------------------------------------------------------------
# K2 planner: the stream-K partition and the measured verdicts
# ---------------------------------------------------------------------------

#: The change count of the router verdict tables (:class:`_VerdictTable`): every change of one adds one, so what the
#: convenience path keeps from the tables (:func:`_router_memo`) is derived again after any change.
_VERDICT_GENERATION = 0


def _verdicts_changed() -> None:
    global _VERDICT_GENERATION
    _VERDICT_GENERATION += 1


class _VerdictTable(dict):
    """A ``dict`` of measured verdicts that counts its changes in :data:`_VERDICT_GENERATION` (an item set or deleted,
    the table updated, popped or cleared).  :data:`ROUTER_RULES`, :data:`ROUTER_RULES_BY_SM` and each table it holds
    are such tables, so the convenience path keeps what it reads from them (:func:`_router_memo`) and the K2 launches
    it plans under them (:data:`_K2_ATOMIC_CACHE`, :data:`_K2_ENTRY_CACHE`) until one changes; a table bound in their
    place that is not one is read on every call.  The other planner tables (the K1 tables, :data:`ROUTER_SM_COUNT`,
    the head/tail templates, the deterministic package's) are read when a key is planned: a cached key keeps its plan
    until :func:`cake_dense_gemm_nondet_train_convenience_cache_clear`.  A copy (``dict(table)``) is a plain ``dict``."""

    __slots__ = ()

    def __setitem__(self, key: Any, value: Any) -> None:
        super().__setitem__(key, value)
        _verdicts_changed()

    def __delitem__(self, key: Any) -> None:
        super().__delitem__(key)
        _verdicts_changed()

    def __ior__(self, other: Any) -> Any:  # type: ignore[misc]
        try:
            return super().__ior__(other)
        finally:
            _verdicts_changed()

    def clear(self) -> None:
        super().clear()
        _verdicts_changed()

    def pop(self, *args: Any) -> Any:
        try:
            return super().pop(*args)
        finally:
            _verdicts_changed()

    def popitem(self) -> Any:
        try:
            return super().popitem()
        finally:
            _verdicts_changed()

    def setdefault(self, *args: Any) -> Any:
        try:
            return super().setdefault(*args)
        finally:
            _verdicts_changed()

    def update(self, *args: Any, **kwargs: Any) -> None:
        try:
            super().update(*args, **kwargs)
        finally:
            _verdicts_changed()


#: Per-row verdicts of the router, ``(arch, layout, hidden, experts) -> rule``: ``layout`` ``"kk"`` (the forward,
#: ``hidden = K``), ``"nn_t"`` (the swapped weight gradient, ``hidden`` = the launch's ``M``) or ``"kn"`` (the input
#: gradient, ``hidden`` = the launch's ``N``), ``experts`` = the launch's ``N`` (``K`` of the input gradient).  A
#: partition spec -- ``"waves"`` (:func:`router_waves`), a head/tail spec ``"H<f>x<k>"`` or ``"H<f>x<k>p<h>"`` (the
#: weight gradient's head/tail program, :func:`router_partition_of`) or one of :func:`router_units_of` (``"forward"``,
#: ``"pairs"``, ``"tiles"``, ``"S<n>"`` or a unit count) -- runs this package's stream-K program, a SIMT spec
#: ``"simt_n<tile_n>m<tile_m>s<slices>[<form>]"`` the exact-FP32 program (plain, or compensated with a form) of the
#: row's layout on the CUDA cores (:func:`router_simt`); ``None`` runs the deterministic program (the split count
#: that package plans for the row, its ``router_splits``).  A rule is ``None``, token windows
#: ``((min_tokens, max_tokens, spec), ...)`` -- the spec of the window that holds the launch's token count (``K`` of the weight gradient, ``M`` of the forward and of the input gradient);
#: every other token count runs the deterministic program -- or a spec alone, at every token count (the plan a test or a
#: measurement forces; no row ships one).  A window carries a spec only where that partition was faster than the
#: deterministic package's shipped plan in every measurement run at the token counts measured in it (one long-lived
#: process and a fresh process per token count): a one-sided paired sign test (p < 0.05) with a median time ratio below
#: 0.99, in eager launches and in CUDA graph replays, and no more than 3 % slower than that plan next to a kernel
#: holding 8, 16 or 32 SMs on another stream -- in the same launch order, and also with this partition launched before
#: that kernel against the deterministic plan launched after it.  A window ends at a measured token count; one that
#: starts at a single token keeps the plan below its smallest measured count (64 tokens) unmeasured.  A token count
#: outside the windows was not measured or missed, a ``None`` entry is a measured row that missed, a row without an
#: entry was not measured: all of them run the deterministic program, and so does every window but a ``"waves"`` one on
#: a device whose SM count is not the measured one, :data:`ROUTER_SM_COUNT`, unless :data:`ROUTER_RULES_BY_SM` has a
#: table for that SM count.  A SIMT window comes first and may overlap the windows after it: on a device where the
#: table's windows do not hold, the token count takes the next window that holds it (the ``"waves"`` window of hidden
#: 5120 below 129 tokens, as before the SIMT windows).  A ``"waves"`` partition runs only where it gives every element
#: at most two partials (:func:`router_at_most_two_partials`; the same bits on every run), else the deterministic
#: program: on the shipped rows (hidden 5120, 256 experts: 40 tiles) only at up to 64 tokens, one or two 32-element
#: K blocks per tile, and only on an SM count whose wave count caps the units at one or two per tile.
#:
#: A ``"det_s<n>"`` window runs the weight gradient on the deterministic program at ``n`` splits
#: (:func:`router_det_splits`): the partials added in a fixed order, so the result repeats bit for bit.  Its accuracy
#: was checked against the reference implementation (FlashInfer commit ``2e989556``) on six input kinds (random
#: normal, long sums, mirrored and permuted cancellation, concentrated values, values near the largest magnitude):
#: within 1.1 times the reference's median error on every kind.  Hidden 6144 takes six splits from 16129 to 16384
#: tokens, where the deterministic package's three splits missed that check on every kind (up to 2.0 times) and six,
#: the reference's count, met it at 1 to 3 % more time.  Hidden 5120 keeps the deterministic package's nine splits
#: there, and hidden 4096 its six: 16 splits also met the check but took 1.00 to 1.02 times the time of six through
#: this package's entry on GB200 and GB300 parts (CUDA graph replays, 16172 to 16384 tokens).  The
#: head/tail partitions these rows ran before (hidden 5120 ``"H0.5x3"`` on both architectures, ``sm_103a`` hidden 4096
#: ``"H1x4p44"`` and 6144 ``"H1x5p44"``, ``sm_107a`` hidden 5120 ``"H0.5x4"``) were 3 to 23 % faster than the
#: deterministic program but varied run to run and missed the check (up to 1.94 times the reference's error), so the
#: rows run the deterministic program there.
# fmt: off
ROUTER_RULES: dict[
    tuple[str, str, int, int], Optional[Union[str, int, tuple[tuple[int, int, Union[str, int]], ...]]]
] = _VerdictTable({
    # the forward: no stream-K partition replay-safe under recompute (at most two partials per element) holds its
    # margin next to a resident kernel.  Few tokens on 152-SM GB200 (sm_100a) / GB300 (sm_103a) parts: the cluster
    # SIMT program (router_forward_simt), timed on two nodes of each part at 1, 4, 5, 8, 9, 12, 16, 17, 32, 64, 96,
    # 97, 112, 128, 129, 160, 176 and 192 tokens: the 8 x 4 tile in the compensated form up to 8 tokens (16 on
    # sm_103a), on sm_100a the 8 x 16 tile from 9 to 16, the 16 x 32 FFMA tile from 17 to 96 and from 129 to 192 and
    # the 16 x 64 FFMA tile from 97 to 128.  Hidden 4096 on sm_100a ends at 160 tokens: at 176 one run was 3.0 % and
    # at 192 an earlier measurement 3.1 and 3.4 % slower than the deterministic plan in CUDA graphs next to a resident
    # kernel.  From 193 tokens, and on 216-SM sm_107a parts, the deterministic plan (not measured under the rule)
    ("sm_100a", "kk", 4096, 256): (
        (1, 8, "simt_c8_kg4_l32_e8t4"), (9, 16, "simt_c8_kg4_l32_e8t16"), (17, 96, "simt_c8_ffma_l16_e16t32"),
        (97, 128, "simt_c8_ffma_l16_e16t64"), (129, 160, "simt_c8_ffma_l16_e16t32"),
    ),
    ("sm_100a", "kk", 5120, 256): (
        (1, 8, "simt_c8_kg4_l32_e8t4"), (9, 16, "simt_c8_kg4_l32_e8t16"), (17, 96, "simt_c8_ffma_l16_e16t32"),
        (97, 128, "simt_c8_ffma_l16_e16t64"), (129, 192, "simt_c8_ffma_l16_e16t32"),
    ),
    ("sm_100a", "kk", 6144, 256): (
        (1, 8, "simt_c8_kg4_l32_e8t4"), (9, 16, "simt_c8_kg4_l32_e8t16"), (17, 96, "simt_c8_ffma_l16_e16t32"),
        (97, 128, "simt_c8_ffma_l16_e16t64"), (129, 192, "simt_c8_ffma_l16_e16t32"),
    ),
    ("sm_103a", "kk", 4096, 256): (
        (1, 16, "simt_c8_kg4_l32_e8t4"), (17, 96, "simt_c8_ffma_l16_e16t32"), (97, 128, "simt_c8_ffma_l16_e16t64"),
        (129, 192, "simt_c8_ffma_l16_e16t32"),
    ),
    ("sm_103a", "kk", 5120, 256): (
        (1, 16, "simt_c8_kg4_l32_e8t4"), (17, 96, "simt_c8_ffma_l16_e16t32"), (97, 128, "simt_c8_ffma_l16_e16t64"),
        (129, 192, "simt_c8_ffma_l16_e16t32"),
    ),
    ("sm_103a", "kk", 6144, 256): (
        (1, 16, "simt_c8_kg4_l32_e8t4"), (17, 96, "simt_c8_ffma_l16_e16t32"), (97, 128, "simt_c8_ffma_l16_e16t64"),
        (129, 192, "simt_c8_ffma_l16_e16t32"),
    ),
    ("sm_107a", "kk", 4096, 256): None, ("sm_107a", "kk", 5120, 256): None, ("sm_107a", "kk", 6144, 256): None,
    # the weight gradient, measured on 152-SM GB200 (sm_100a) / GB300 (sm_103a) and 216-SM R200 (sm_107a) parts.  A
    # window keeps the deterministic package's router budget: for every seed (10 seeds per measured token count, the
    # router FP64 error table of the README) a relative Frobenius error against FP64 of the exact operands, worst of
    # 20 runs, at most 2e-6.  Its ratio to the error of the plan the row takes without the window is reported data,
    # not a gate.
    # Few tokens on sm_100a: the SIMT program (measured at 64, 128, 192, 224 and 256 tokens, alone, in the backward
    # pass and in the checkpointed step), the 128 x 64 tile from 1 to 224 tokens at hidden 4096 (128 CTAs of 256
    # threads, one wave on 152 SMs; at 256 tokens it was 6 to 12 % slower than the deterministic plan in CUDA graphs
    # next to a resident kernel holding 32 SMs, which leaves 120 SMs for its 128 CTAs), the 64 x 64 tile from 1 to 256
    # tokens at hidden 5120 and 6144 (320 and 384 CTAs of 128 threads; the 128 x 64 tile gives 160 and 192 CTAs
    # there, more than one per SM, and was 20 to 26 % slower alone).  Few tokens on 152-SM sm_103a parts: the SIMT
    # program with the 64 x 64 tile in two slices at every hidden size (measured from 1 to 320 tokens at hidden 4096
    # and 6144 and from 1 to 192 at hidden 5120, against the deterministic plan and at hidden 5120 against "waves"),
    # from 1 to 288 tokens at hidden 4096, 1 to 192 at hidden 5120 and 1 to 256 at hidden 6144; one step more (320
    # tokens at hidden 4096, 288 at hidden 6144) it was more than 3 % slower than the deterministic plan in CUDA
    # graphs next to a resident kernel.
    # Hidden 5120: "waves" at few tokens (faster than the deterministic plan at 64 and 128 tokens, and on sm_103a at
    # 192; more than 3 % slower than it next to a resident kernel at every measured count from 224 to 10240 tokens;
    # on 152-SM sm_100a and sm_103a parts the SIMT window before it takes these token counts),
    # then the head/tail program with one head unit per CTA pair over half of the steps and three tail units per pair
    # ("H0.5x3", 76 head and 228 tail units on 152 SMs; windows measured at 10240 and 12288 (sm_100a), 16172, 16231,
    # 16384 and 32768 tokens; more than 3 % slower next to a resident kernel at 8192 tokens on sm_100a, at 10240 and
    # 12288 on sm_103a; a later sm_103a measurement passed at 10240 on two parts, between misses at 8192 and 12288, so
    # 10240 was not added).  On sm_100a the window ended at 16384 tokens: measured again against six splits, it was
    # up to 4.5 % slower than that plan next to a resident kernel at 24576 tokens and up to 6.9 % at 32768 (16385 to
    # 24575 tokens not measured); its windows are 10240 to 16384 and 16400 to 28672 tokens now (below).
    # Hidden 4096 on sm_100a keeps the deterministic plan above its SIMT window: the head/tail partition with 44 head
    # units and five tail units per CTA pair ("H1x5p44"), measured again against the current deterministic plan at
    # 10240 to 16384 tokens, left the checkpointed step (two forwards and the backward pass) up to 2.7 % slower in
    # eager launches at 10240 and 12288 tokens.
    # Hidden 4096 on sm_103a: "H1x4p44" (44 head units, then four tail units per CTA pair) from 20480 to 49152 tokens
    # (measured at 20480, 24576, 32768 and 49152 tokens, each on four or five parts); up to 7 % slower than the
    # deterministic plan next to a resident kernel at 8192 to 16384 tokens, and at 65536 the checkpointed step (two
    # forwards and the backward pass) was up to 1.4 % slower.  Hidden 6144 on sm_103a: "H1x5p44" from 16385 to 20480
    # tokens (first from 16172, measured at 16172, 16231, 16384 and 20480 tokens, each on four parts; 16172 to 16384
    # below); up to 7.8 % slower next to a resident kernel at 8192 to 12288 tokens, the backward pass (input and weight
    # gradient) no faster at 24576 and up to 2.3 % slower at 32768.  Hidden 6144 on sm_100a keeps the deterministic
    # plan above its SIMT window: no partition was faster in every run in CUDA graphs.
    # sm_107a: hidden 5120 "H0.5x4" (four tail units, 108 head and 432 tail units on 216 SMs; measured on two parts at
    # 16172, 16231 and 16384 tokens); hidden 4096 and 6144 keep the deterministic plan: at 4096 "H0.5x4" was up to
    # 7 % slower than that plan when a kernel holding 16 SMs on another stream started during it (the kernel takes SMs
    # the head spans wait for), at 6144 it left the backward pass no faster.
    # The deterministic plan of the measurements above is that package's six splits.  From 16129 to 16384 tokens on
    # 148- and 152-SM sm_100a / sm_103a parts it plans nine splits at hidden 5120 and three at hidden 6144 (its
    # ROUTER_SPLIT_RULES, router_splits).  The 152-SM head/tail windows at and next to those token counts were measured
    # again on two parts against the plan the row runs without them (the rule's count from 16129 to 16384 tokens, six
    # elsewhere), alone, in the backward pass and in the checkpointed step, in eager launches and in CUDA graphs.  A
    # token count stays in a window where, on both parts, the partition was faster alone in every run (a one-sided
    # paired sign test, p < 0.05, with a median time ratio below 0.99) and the backward pass and the checkpointed step
    # were no slower (pooled median ratio at most 1.013, no run slower than the slowest run of that plan against
    # itself); next to a resident kernel it was up to 4.5 % slower at some kept token counts (data, not a condition
    # here).  A window ends at kept token counts and holds no token count that missed.  The comparison is what a caller
    # gets: the window through this package's entry against the deterministic plan through it; alone in eager launches
    # the entry adds about 14 us to the head/tail program and about 6 us to the deterministic plan, while a direct call
    # of the deterministic program's binding adds under 1 us, so a miss alone in eager launches against that binding
    # alone does not remove a window:
    # - sm_100a hidden 5120 "H0.5x3": main's window (10240 to 16384 tokens) measured again with these tables on two
    #   parts: faster than the deterministic plan through this package's entry at 12289, 16128 and 16384 tokens on
    #   both parts, alone, in the backward pass and in the checkpointed step, eager and in CUDA graphs (at 16129 on one
    #   part; on the other one eager checkpointed-step run was 3.3 % slower), and at 10240, 11264 and 12288 faster than
    #   the deterministic program in CUDA graphs, in the backward pass and in the checkpointed step (alone in eager
    #   launches it missed against that program's binding only); kept at 16400, 18432, 20480, 24576 and 28672 (against
    #   six), missed at 16385 (one eager checkpointed-step run on one part) and at 32768 (the checkpointed step in CUDA
    #   graphs on one part): the windows are 10240 to 16384 and 16400 to 28672 tokens.
    # - sm_103a hidden 5120 "H0.5x3": main's window (16172 to 32768 tokens) measured again with these tables on two
    #   parts: faster than the deterministic plan at 16172 and 16231 tokens (nine splits) on both parts, alone, in the
    #   backward pass and in the checkpointed step, eager and in CUDA graphs (through the entry against the plan
    #   through it on one part: 3.9 to 5.1 % faster alone in eager launches, 9.6 to 9.9 % in CUDA graphs), kept at
    #   16232 (nine splits), 16385, 20480, 24576, 28672 and 32768 (six), and at 16384 (nine) faster except alone in
    #   eager launches against the deterministic program's binding: the window stays 16172 to 32768 tokens.
    # - sm_103a hidden 6144 "H1x5p44": main's window (16172 to 20480 tokens) measured again: at 16172, 16231 and 16384
    #   tokens (three splits) the window through this package's entry was 2.9 to 3.7 % slower alone in eager launches
    #   than the deterministic plan through it, in every run on one part (no slower in the backward pass and the
    #   checkpointed step, 1.4 to 2.0 % faster alone in CUDA graphs); at 16385 and 20480 tokens (six splits) it was
    #   no slower than that plan: the window is 16385 to 20480 tokens.
    ("sm_100a", "nn_t", 4096, 256): ((1, 224, "simt_n128m64s2"),),
    ("sm_100a", "nn_t", 5120, 256): ((1, 256, "simt_n64m64s2"), (1, 128, "waves")),
    ("sm_100a", "nn_t", 6144, 256): ((1, 256, "simt_n64m64s2"), (16129, 16384, "det_s6")),
    ("sm_103a", "nn_t", 4096, 256): ((1, 288, "simt_n64m64s2"),),
    ("sm_103a", "nn_t", 5120, 256): ((1, 192, "simt_n64m64s2"), (1, 192, "waves")),
    ("sm_103a", "nn_t", 6144, 256): ((1, 256, "simt_n64m64s2"), (16129, 16384, "det_s6")),
    ("sm_107a", "nn_t", 4096, 256): None,
    ("sm_107a", "nn_t", 5120, 256): None,
    ("sm_107a", "nn_t", 6144, 256): None,
    # the input gradient, measured on 152-SM GB200 (sm_100a) parts at 1, 17, 32, 64, 96 and 128 tokens at hidden 4096
    # and 5120 and at 1, 17, 32 and 64 tokens at hidden 6144 (outside the windows: at 192 tokens, at 224 and 256 at
    # hidden 4096 and at 128 at hidden 6144): the SIMT program with the 32 x 64 tile in four slices (128 threads, one
    # CTA per 32 tokens and 64 hidden columns) up to 128 tokens at hidden 4096 and 5120 and up to 64 at hidden 6144;
    # next to a resident kernel it was more than 3 % slower than the deterministic program from 192 tokens (from 128 at
    # hidden 6144).  152-SM GB300 (sm_103a) parts take the same tile and windows (measured there at the same token
    # counts inside the windows, on two parts).  216-SM sm_107a parts keep the deterministic program (not measured)
    ("sm_100a", "kn", 4096, 256): ((1, 128, "simt_n64m32s4"),),
    ("sm_100a", "kn", 5120, 256): ((1, 128, "simt_n64m32s4"),),
    ("sm_100a", "kn", 6144, 256): ((1, 64, "simt_n64m32s4"),),
    ("sm_103a", "kn", 4096, 256): ((1, 128, "simt_n64m32s4"),),
    ("sm_103a", "kn", 5120, 256): ((1, 128, "simt_n64m32s4"),),
    ("sm_103a", "kn", 6144, 256): ((1, 64, "simt_n64m32s4"),),
})
# fmt: on
#: The SM count the verdicts of :data:`ROUTER_RULES` were measured on, per architecture (152-SM GB200 / GB300, 216-SM
#: R200): on a device with any other SM count a router row runs the deterministic program, except in a ``"waves"``
#: window, which keeps the condition of :func:`router_waves` on every SM count (the plan of these rows before the
#: windows) and runs its partition only where every element takes at most two partials.  An explicit ``splits`` is not
#: affected.
ROUTER_SM_COUNT: dict[str, int] = {"sm_100a": 152, "sm_103a": 152, "sm_107a": 216}
#: Per-SM-count router verdicts of the parts of an architecture whose SM count is not :data:`ROUTER_SM_COUNT`,
#: ``(arch, sm_count) -> {key: rule}`` (keys and rules as in :data:`ROUTER_RULES`, measured on parts of exactly that SM
#: count under the same rule, in two jobs on different parts).  Such a part takes the table of its own SM count only:
#: a row without an entry in it, and every token count outside its windows, runs the deterministic program.
#: The sm_107a tables were measured on 208-SM and 212-SM R200 parts at 16172 to 16384 tokens: hidden 5120 takes
#: "H0.5x4" there (104 head and 416 tail units on 208 SMs, 106 and 424 on 212).  On 212 SMs hidden 6144 takes "H0.5x4"
#: too, a window that holds only with the head/tail program of :data:`ROUTER_HEAD_TAIL_TEMPLATES_BY_SM` (with
#: ``router_fp32_gemm_atomic_nn_t_ht_c3_nreg`` it was up to 9.2 % slower than the deterministic plan next to a resident
#: kernel).  Hidden 4096 keeps the deterministic plan on both SM counts (on 212 SMs "H0.5x4" with the tensor-memory
#: program was up to 6.2 % slower next to a resident kernel), as do hidden 6144 on 208 SMs and on 208 SMs the forward,
#: as on 216 SMs.  On 212 SMs every hidden size takes the SIMT program with the 128 x 64 tile, from 1 to 256 tokens at
#: hidden 4096 and 5120 and from 1 to 224 tokens at hidden 6144 (measured at 64, 128, 192 and 256 tokens on 212-SM
#: R200 parts, at hidden 6144 also at 192, 224 and 256 tokens in a second measurement on two parts; 128, 160 and 192
#: CTAs of 256 threads, one wave; at hidden 6144 and 256 tokens it was up to 12 % slower than the deterministic plan
#: next to a resident kernel holding 32 SMs, which leaves 180 SMs for its 192 CTAs, and up to 9.4 % slower in the
#: second measurement).
#: A part's table may also hold rows :data:`ROUTER_RULES` has no entry for, measured on that SM count only: on 212 SMs
#: the input gradient takes the SIMT program with the 32 x 64 tile in eight slices up to 64 tokens and the 64 x 64
#: tile in four slices from 65 to 128 tokens at hidden 4096 and 5120 (256 threads each; measured on 212-SM R200 parts
#: at 1, 17, 32, 64, 96 and 128 tokens at hidden 4096 and 5120 and at 1, 17, 32 and 64 tokens at hidden 6144, and at
#: 192 tokens: from 192 tokens, from 128 at hidden 6144, more than 3 % slower than the deterministic program next to a
#: resident kernel), up to 64 tokens at hidden 6144.
#: On 212 SMs the forward takes the cluster SIMT program (:func:`router_forward_simt`) at hidden 4096, 5120 and 6144:
#: the 8 x 4 tile in the compensated form from 1 to 8 tokens and the 16 x 32 FFMA tile from 9 to 192 tokens, measured
#: under the rule of :data:`ROUTER_RULES` on two 212-SM R200 nodes at 1, 4, 5, 8, 9, 12, 16, 17, 32, 64, 96, 97, 112,
#: 128, 129, 160, 176 and 192 tokens (every measured token count held); from 193 tokens the deterministic plan (not
#: measured under the rule).
#: The 148-SM B200 (sm_100a) and B300 (sm_103a) tables hold SIMT windows measured on one node of each part under the
#: rule of :data:`ROUTER_RULES` against the plan the token count took before (both launch orders, eager and CUDA
#: graphs, next to a kernel holding 8, 16 or 32 SMs, the backward pass and the checkpointed step, in a long-lived
#: process and a fresh process per token count; the whole-step ratio also no worse than the same plan measured against
#: itself in the same runs).  A window starts and ends at measured token counts and holds at least one more timed and
#: two more accuracy-checked token counts inside; the token counts between them take its program.
#: The weight gradient takes the plain 64 x 64 tile in two slices from 1 to 256 tokens at hidden 5120 and 6144, and at
#: hidden 4096 the plain 128 x 64 tile from 2 to 224 tokens; at one token hidden 4096 keeps the 64 x 64 tile in the
#: compensated Dot2 form (the plain 128 x 64 tile left the backward pass up to 1.9 % slower on B200 and 3.8 % on B300
#: there), and at 256 tokens the 128 x 64 tile was up to 11 % (B200) and 13 % (B300) slower next to a resident kernel.
#: The input gradient takes the plain 32 x 64 tile in eight slices up to 64 tokens at hidden 4096 and up to 32 at
#: hidden 5120 and 6144, and on 148-SM sm_100a parts the 32 x 64 tile in four slices from 64 to 128 tokens at hidden
#: 5120 (on B300 the checkpointed step at 128 tokens was 0.09 % and 0.18 % slower than before in eager launches, more
#: than the plan measured against itself).  Every window's program keeps the router budget: a relative Frobenius error
#: against FP64 of at most 2.1e-7 at the 124 (B200) and 120 (B300) measured token counts of the windows (10 seeds,
#: worst of 20 runs; 0.95 to 2.66 times the deterministic program's).  Every other row and token count of these parts
#: runs the deterministic program.
#: The ``"det_s<n>"`` windows of these tables follow the check of :data:`ROUTER_RULES`: hidden 6144 takes six splits
#: from 16129 to 16384 tokens on 148-SM parts (the deterministic package's three missed the check on B200; B300 runs the
#: same arithmetic, its time not measured), B200 hidden 4096 16 splits at 16384 tokens (2.4 % faster than six there,
#: not at 16172 or 16231 tokens), and 212-SM ``sm_107a`` parts hidden 5120 12 splits at 16172 to 16384 tokens (0.90 to
#: 0.92 times the time of six, the deterministic package's count there; hidden 4096 and 6144 keep six, which met the
#: check).  The 208- and 212-SM ``sm_107a`` head/tail windows ``"H0.5x4"`` missed the check (2 to 6 kinds, up to 1.63
#: times) and are gone: those rows run the deterministic program.
#: A ``"ksum_S<n>"`` window runs the weight gradient on the K-order program of the part
#: (:data:`ROUTER_KSUM_TEMPLATES_BY_SM`, :func:`router_ksum_template`) at ``n`` equal spans per tile: each span stores
#: its FP32 partial of the tile into a workspace slot and takes a ticket, and the span that draws the tile's last ticket
#: adds the slots in K order and stores the tile -- the deterministic program's ``n`` split partials summed in K order,
#: so the result repeats bit for bit (no fill of ``out``, the tickets zero before and after every complete launch).  On
#: 212-SM ``sm_107a`` parts hidden 6144 takes six spans from 16172 to 16231 tokens: on a R200 part it met the check of
#: :data:`ROUTER_RULES` on all six kinds at 16172 and 16231 tokens (its worst error 1.09 times the reference's, on
#: mirrored cancellation) at 0.93 times the time of the deterministic program at six splits there, in eager launches,
#: on a card without a power cap; the head/tail program of :data:`ROUTER_HEAD_TAIL_TEMPLATES_BY_SM` was 3 % faster but
#: missed the check (four and one of six kinds).
# fmt: off
ROUTER_RULES_BY_SM: dict[
    tuple[str, int],
    dict[tuple[str, str, int, int], Optional[Union[str, int, tuple[tuple[int, int, Union[str, int]], ...]]]],
] = _VerdictTable({
    ("sm_100a", 148): _VerdictTable({
        ("sm_100a", "kk", 4096, 256): None, ("sm_100a", "kk", 5120, 256): None, ("sm_100a", "kk", 6144, 256): None,
        ("sm_100a", "nn_t", 4096, 256): (
            (1, 1, "simt_n64m64s2dot2"), (2, 224, "simt_n128m64s2"), (16384, 16384, "det_s16"),
        ),
        ("sm_100a", "nn_t", 5120, 256): ((1, 256, "simt_n64m64s2"),),
        ("sm_100a", "nn_t", 6144, 256): ((1, 256, "simt_n64m64s2"), (16129, 16384, "det_s6")),
        ("sm_100a", "kn", 4096, 256): ((1, 64, "simt_n64m32s8"),),
        ("sm_100a", "kn", 5120, 256): ((1, 32, "simt_n64m32s8"), (64, 128, "simt_n64m32s4")),
        ("sm_100a", "kn", 6144, 256): ((1, 32, "simt_n64m32s8"),),
    }),
    ("sm_103a", 148): _VerdictTable({
        ("sm_103a", "kk", 4096, 256): None, ("sm_103a", "kk", 5120, 256): None, ("sm_103a", "kk", 6144, 256): None,
        ("sm_103a", "nn_t", 4096, 256): ((1, 1, "simt_n64m64s2dot2"), (2, 224, "simt_n128m64s2")),
        ("sm_103a", "nn_t", 5120, 256): ((1, 256, "simt_n64m64s2"),),
        ("sm_103a", "nn_t", 6144, 256): ((1, 256, "simt_n64m64s2"), (16129, 16384, "det_s6")),
        ("sm_103a", "kn", 4096, 256): ((1, 64, "simt_n64m32s8"),),
        ("sm_103a", "kn", 5120, 256): ((1, 32, "simt_n64m32s8"),),
        ("sm_103a", "kn", 6144, 256): ((1, 32, "simt_n64m32s8"),),
    }),
    ("sm_107a", 208): _VerdictTable({
        ("sm_107a", "kk", 4096, 256): None, ("sm_107a", "kk", 5120, 256): None, ("sm_107a", "kk", 6144, 256): None,
        ("sm_107a", "nn_t", 4096, 256): None,
        ("sm_107a", "nn_t", 5120, 256): None,
        ("sm_107a", "nn_t", 6144, 256): None,
    }),
    ("sm_107a", 212): _VerdictTable({
        ("sm_107a", "kk", 4096, 256): ((1, 8, "simt_c8_kg4_l32_e8t4"), (9, 192, "simt_c8_ffma_l16_e16t32")),
        ("sm_107a", "kk", 5120, 256): ((1, 8, "simt_c8_kg4_l32_e8t4"), (9, 192, "simt_c8_ffma_l16_e16t32")),
        ("sm_107a", "kk", 6144, 256): ((1, 8, "simt_c8_kg4_l32_e8t4"), (9, 192, "simt_c8_ffma_l16_e16t32")),
        ("sm_107a", "nn_t", 4096, 256): ((1, 256, "simt_n128m64s2"),),
        ("sm_107a", "nn_t", 5120, 256): ((1, 256, "simt_n128m64s2"), (16172, 16384, "det_s12")),
        ("sm_107a", "nn_t", 6144, 256): ((1, 224, "simt_n128m64s2"), (16172, 16231, "ksum_S6")),
        ("sm_107a", "kn", 4096, 256): ((1, 64, "simt_n64m32s8"), (65, 128, "simt_n64m64s4")),
        ("sm_107a", "kn", 5120, 256): ((1, 64, "simt_n64m32s8"), (65, 128, "simt_n64m64s4")),
        ("sm_107a", "kn", 6144, 256): ((1, 64, "simt_n64m32s8"),),
    }),
})
# fmt: on
#: The template suffix of the weight gradient's head/tail program (``router_fp32_gemm_atomic_nn_t`` + suffix): the
#: head/tail partition on three TMEM chunk buffers, with register reallocation between its warpgroups.
ROUTER_HEAD_TAIL_SUFFIX = "_ht_c3_nreg"
#: The weight gradient's head/tail program of an architecture that runs another one than
#: ``router_fp32_gemm_atomic_nn_t`` + :data:`ROUTER_HEAD_TAIL_SUFFIX`: on ``sm_103a``, the head/tail program with the
#: BF16 parts of ``A`` in tensor memory (the same products, accumulators, partition and part segments; two TMEM chunk
#: buffers, six slots of raw ``A`` tiles and six of ``B`` parts in shared memory).  ``sm_103a`` takes a head/tail
#: partition only in the windows of :data:`ROUTER_RULES` on 152 SMs, where it was measured.
ROUTER_HEAD_TAIL_TEMPLATES: dict[str, str] = {
    "sm_103a": "router_fp32_gemm_ts_r6p6_atomic_nn_t_ht_c2_nreg",
}
#: The weight gradient's head/tail program of the parts of an architecture with exactly ``sm_count`` SMs where it is
#: not the architecture's (:data:`ROUTER_HEAD_TAIL_TEMPLATES`), ``(arch, sm_count) -> template``: on 212-SM ``sm_107a``
#: parts, the head/tail program with the BF16 parts of ``A`` in tensor memory, which serves the hidden 5120 and 6144
#: windows of that SM count's table in :data:`ROUTER_RULES_BY_SM` (measured on 212-SM R200 parts at 16172, 16231 and
#: 16384 tokens, four parts in two nodes).  208-SM and 216-SM ``sm_107a`` parts keep
#: ``router_fp32_gemm_atomic_nn_t`` + :data:`ROUTER_HEAD_TAIL_SUFFIX` (the other program was not measured there).
ROUTER_HEAD_TAIL_TEMPLATES_BY_SM: dict[tuple[str, int], str] = {
    ("sm_107a", 212): "router_fp32_gemm_ts_r6p6_atomic_nn_t_ht_c2_nreg",
}
#: The weight gradient's K-order program of the parts of an architecture with exactly ``sm_count`` SMs, ``(arch,
#: sm_count) -> template``: the program of a ``"ksum_S<n>"`` window of that SM count's table in
#: :data:`ROUTER_RULES_BY_SM` (the head/tail program with the BF16 parts of ``A`` in tensor memory, its part segments
#: summed in K order through workspace slots and per-tile tickets instead of added into ``out`` by TMA reductions).  A
#: part without an entry runs the deterministic program in such a window.
ROUTER_KSUM_TEMPLATES_BY_SM: dict[tuple[str, int], str] = {
    ("sm_107a", 212): "router_fp32_gemm_ts_r6p6_atomic_nn_t_ksum_ht_c2_nreg",
}
#: A K-order window spec ``"ksum_S<n>"``: ``n`` equal spans per tile, summed in K order (:data:`ROUTER_RULES_BY_SM`).
_ROUTER_KSUM_SPEC = re.compile(r"ksum_S(\d+)")
#: FP32 elements of one K-order workspace slot (a ``256 x 128`` tile of one span) and ticket words of one tile.
ROUTER_KSUM_SLOT_ELEMS = 256 * 128
ROUTER_KSUM_TILE_WORDS = 512
_ROUTER_HEAD_TAIL_SPEC = re.compile(r"H(\d+(?:\.\d+)?)x(\d+)(?:p(\d+))?")
#: A SIMT window spec ``"simt_n<tile_n>m<tile_m>s<slices>[<form>]"``: an exact-FP32 program on the CUDA cores with one
#: CTA per ``tile_n x tile_m`` tile of ``out`` over the whole K range: ``slices`` slices of the CTA each take every
#: ``slices``-th K block (16 elements; 8 in the Dot2 form) and add in slice order.  No BF16 parts, no split of K across
#: CTAs: every element of ``out`` is stored once, its bits fixed by ``K`` and the program.  Without a form the plain
#: program (one FFMA per product: ``router_fp32_gemm_simt_n<tile_n>m<tile_m>s<slices>`` on a weight-gradient row,
#: ``nn_t``, and ``router_fp32_dgrad_simt_m<tile_m>n<tile_n>s<slices>`` on an input-gradient row, ``kn``); with one a
#: compensated program, each slice keeping a sum and its compensation and the slices merged in slice order by TwoSum
#: with their compensations (every FP32 instruction rounded to nearest without flush):
#:
#: * ``g<group>``: per output and group of ``group`` consecutive K steps of a slice one FMA chain from the negated
#:   compensation and one Kahan add (``(group + 3) / group`` FP32 instructions per product), the program
#:   ``router_fp32_gemm_simt_kahan_n<tile_n>m<tile_m>s<slices>g<group>`` /
#:   ``router_fp32_dgrad_simt_kahan_m<tile_m>n<tile_n>s<slices>g<group>`` -- for ``g1``, one Kahan step per product,
#:   without the ``g1`` suffix;
#: * ``dot2`` (weight gradient): each product rounded once with its exact error kept, a TwoSum into the sum and both
#:   errors into the compensation, in blocks of 8 rows without the rows past ``K`` (10 FP32 instructions per product),
#:   the program ``router_fp32_gemm_simt_kahan_dot2_n<tile_n>m<tile_m>s<slices>``.
#:
#: The weight gradient runs ``out[N, M] = B[K, N].T @ A[K, M]`` of the swapped launch (``A`` the activations, ``B`` the
#: output gradient), the input gradient ``out[M, N] = A[M, K] @ B[K, N]`` (``A`` the output gradient, ``B`` the router
#: weight; ``tile_m`` token rows by ``tile_n`` hidden columns).
_ROUTER_SIMT_SPEC = re.compile(r"simt_n(\d+)m(\d+)s(\d+)(g\d+|dot2)?")
#: The K rows of a SIMT program's block (its ``k_iters`` count them).
ROUTER_SIMT_BLOCK_K = 16
#: A forward SIMT window spec ``"simt_c8_<kind>_l<lanes>_e<experts>t<tokens>"`` (forward rows, ``kk``): the
#: exact-FP32 program ``router_fp32_fwd_simt_c8_<kind>_l<lanes>_e<experts>t<tokens>`` on the CUDA cores,
#: ``out[T, N] = X[T, K] @ W[N, K].T`` with the FP32 activations and router weight read in place.  A cluster of
#: :data:`ROUTER_FWD_SIMT_CLUSTER` CTAs (256 threads each) owns one tile of ``experts`` experts by ``tokens`` tokens,
#: rank ``r`` its ``r``-th eighth of ``K``; the ``lanes`` threads of a lane group split the eighth's columns and
#: combine their chains by an ``xor`` butterfly with ascending lane offsets, and rank ``r`` adds the eight ranks'
#: partials of its eighth of the tile's experts in rank order through distributed shared memory, storing every
#: element once.  ``kind`` ``"ffma"``: one FFMA chain per lane and output; ``"kg4"``: per lane and output a sum and
#: its compensation, an FMA chain over four columns from the negated compensation and one Kahan add, the lane
#: tree and the ranks adding the (sum, compensation) pairs by TwoSum.  Every FP32 instruction rounds to nearest
#: without flush.
_ROUTER_FWD_SIMT_SPEC = re.compile(r"simt_c8_(ffma|kg4)_l(\d+)_e(\d+)t(\d+)")
#: The CTAs of the forward SIMT program's cluster (the eighths of ``K``) and the K columns of its chunk: ``K`` must
#: be a multiple of their product.
ROUTER_FWD_SIMT_CLUSTER = 8
ROUTER_FWD_SIMT_BLOCK_K = 128
#: The forward's largest explicit split count: an element of the forward takes at most two partials, so a forward
#: recomputed under activation checkpointing repeats bit for bit (``0 + a + b == 0 + b + a``).
ROUTER_FORWARD_MAX_SPLITS = 2
#: The deterministic package's default split count of the router weight gradient (its ``ROUTER_SPLITS["nn_t"]``):
#: the plan the ``"waves"`` partition is measured against, and the count of :func:`router_waves` without one.
#: :func:`router_partition` passes the count that package plans for the row (its ``router_splits``), which differs
#: from this one only on the rows of its ``ROUTER_SPLIT_RULES``; none of them meets a ``"waves"`` window.
ROUTER_DET_WGRAD_SPLITS = 6
#: The ``"waves"`` partition: about this many K blocks (32 elements each) of one tile per unit ...
ROUTER_WAVE_K_BLOCKS = 68
#: ... and only where the deterministic plan fills its last wave of CTA pairs below this fraction
#: (``w / ceil(w)``, ``w = splits * tiles / pairs``, ``splits`` that plan's split count).
ROUTER_WAVE_MAX_DET_FILL = 0.8


def router_rules(
    arch: str, sm_count: int
) -> tuple[dict[tuple[str, str, int, int], Any], bool]:
    """``(rules, measured)``: the router verdicts a device of ``sm_count`` SMs takes -- the table of
    :data:`ROUTER_RULES_BY_SM` for exactly ``(arch, sm_count)``, else :data:`ROUTER_RULES` -- and whether their
    windows other than ``"waves"`` hold on it (always for a table of its own; for :data:`ROUTER_RULES` only on
    :data:`ROUTER_SM_COUNT` SMs)."""
    table = ROUTER_RULES_BY_SM.get((arch, int(sm_count)))
    if table is not None:
        return table, True
    return ROUTER_RULES, int(sm_count) == ROUTER_SM_COUNT.get(arch)


def router_waves(
    m_pairs: int,
    n_tiles: int,
    k_iters: int,
    sm_count: int,
    det_splits: int = ROUTER_DET_WGRAD_SPLITS,
) -> Optional[int]:
    """The ``"waves"`` unit count on a device of ``sm_count`` SMs (CTA pairs ``sm_count // 2``): whole waves of the
    CTA pairs, ``c * pairs`` units with ``c = max(2, round(tiles * k_iters / (ROUTER_WAVE_K_BLOCKS * pairs))) + 1``
    (spans of about ``ROUTER_WAVE_K_BLOCKS`` K blocks of a tile, one wave more, so a kernel holding SMs on another
    stream delays a fine last wave rather than a coarse one), or ``None`` (the deterministic program) where the
    deterministic plan's last wave is at least :data:`ROUTER_WAVE_MAX_DET_FILL` full.  ``det_splits`` is that plan's
    split count (default :data:`ROUTER_DET_WGRAD_SPLITS`; :func:`router_partition` passes the deterministic
    package's count of the row)."""
    tiles, pairs = m_pairs * n_tiles, max(1, int(sm_count) // 2)
    det_waves = int(det_splits) * tiles / pairs
    if det_waves / math.ceil(det_waves) >= ROUTER_WAVE_MAX_DET_FILL:
        return None
    return (max(2, round(tiles * k_iters / (ROUTER_WAVE_K_BLOCKS * pairs))) + 1) * pairs


def router_units_of(
    spec: Union[str, int], m_pairs: int, n_tiles: int, k_iters: int, sm_count: int
) -> int:
    """The unit count of a partition spec on a device of ``sm_count`` SMs (CTA pairs ``sm_count // 2``): ``"tiles"``
    (one unit per tile), ``"pairs"`` (one per CTA pair), ``"forward"`` (one per CTA pair, at most one per tile),
    ``"S<n>"`` (``n`` units per tile) or a count, rounded down to a multiple of the column tile count (the units of a
    span run its column tiles), at least ``n_tiles`` and at most one unit per (tile, K block) step; a head/tail spec
    ``"H<f>x<k>"`` / ``"H<f>x<k>p<h>"``: the units of :func:`router_partition_of`."""
    if router_head_tail(spec) is not None:
        return router_partition_of(spec, m_pairs, n_tiles, k_iters, sm_count)[0]
    tiles, pairs = m_pairs * n_tiles, int(sm_count) // 2
    if isinstance(spec, int):
        count = spec
    elif spec == "tiles":
        count = tiles
    elif spec == "pairs":
        count = pairs
    elif spec == "forward":
        count = min(tiles, pairs)
    elif isinstance(spec, str) and spec.startswith("S") and spec[1:].isdigit():
        count = int(spec[1:]) * tiles
    else:
        raise ValueError(f"unknown router partition spec {spec!r}")
    return max(n_tiles, min(count, tiles * k_iters) // n_tiles * n_tiles)


def router_head_tail(spec: Union[str, int, None]) -> Optional[tuple[float, int]]:
    """``(f, k)`` of a head/tail partition spec ``"H<f>x<k>"`` or ``"H<f>x<k>p<h>"``, else ``None``."""
    if isinstance(spec, str):
        match = _ROUTER_HEAD_TAIL_SPEC.fullmatch(spec)
        if match:
            return float(match.group(1)), int(match.group(2))
    return None


def router_head_pairs(spec: Union[str, int, None]) -> Optional[int]:
    """``h`` of a head/tail partition spec ``"H<f>x<k>p<h>"`` (the head units), else ``None`` (one head unit per CTA
    pair)."""
    if isinstance(spec, str):
        match = _ROUTER_HEAD_TAIL_SPEC.fullmatch(spec)
        if match and match.group(3) is not None:
            return int(match.group(3))
    return None


def router_simt_tile(
    spec: Union[str, int, None],
) -> Optional[tuple[Union[int, str], ...]]:
    """The tile of a SIMT window spec: ``(tile_n, tile_m, slices)`` of a plain spec
    ``"simt_n<tile_n>m<tile_m>s<slices>"``, ``(tile_n, tile_m, slices, form)`` of a compensated one (``form``
    ``"g<group>"`` or ``"dot2"``); else ``None``."""
    if isinstance(spec, str):
        match = _ROUTER_SIMT_SPEC.fullmatch(spec)
        if match:
            tile_n, tile_m, slices, form = match.groups()
            tile = (int(tile_n), int(tile_m), int(slices))
            return tile if form is None else (*tile, form)
    return None


def router_forward_simt_tile(
    spec: Union[str, int, None],
) -> Optional[tuple[str, int, int, int]]:
    """The tile ``(kind, lanes, experts, tokens)`` of a forward SIMT window spec
    ``"simt_c8_<kind>_l<lanes>_e<experts>t<tokens>"``, else ``None``."""
    if isinstance(spec, str):
        match = _ROUTER_FWD_SIMT_SPEC.fullmatch(spec)
        if match:
            kind, lanes, experts, tokens = match.groups()
            return kind, int(lanes), int(experts), int(tokens)
    return None


def router_forward_simt_template(tile: tuple[str, int, int, int]) -> str:
    """The template of the forward's SIMT program of the tile ``(kind, lanes, experts, tokens)``
    (:func:`router_forward_simt_tile`): ``router_fp32_fwd_simt_c8_<kind>_l<lanes>_e<experts>t<tokens>``."""
    if len(tile) != 4 or tile[0] not in ("ffma", "kg4"):
        raise ValueError(
            f"router_forward_simt_template: {tile!r} is not a forward SIMT tile"
        )
    kind, lanes, experts, tokens = tile
    return f"{cake_jit.TEMPLATE_PREFIXES[6]}c8_{kind}_l{int(lanes)}_e{int(experts)}t{int(tokens)}"


def _router_is_simt(spec: Union[str, int, None]) -> bool:
    """Whether a window spec runs a SIMT program (:func:`router_simt_tile` / :func:`router_forward_simt_tile`)."""
    return (
        router_simt_tile(spec) is not None or router_forward_simt_tile(spec) is not None
    )


_ROUTER_DET_SPEC = re.compile(r"det_s(\d+)")


def router_det_spec_splits(spec: Union[str, int, None]) -> Optional[int]:
    """The split count of a ``"det_s<n>"`` window spec (the deterministic program at ``n`` splits), else ``None``."""
    if not isinstance(spec, str):
        return None
    m = _ROUTER_DET_SPEC.fullmatch(spec)
    return int(m[1]) if m else None


def router_det_splits(
    layout: str, *, arch: str, M: int, N: int, K: int, sm_count: int
) -> Optional[int]:
    """The split count the deterministic program runs a weight-gradient launch ``[M, N] = [M, K] @ [K, N]`` (layout
    class ``nn_t``, the swapped GEMM: ``M`` hidden, ``N`` experts, ``K`` tokens) with on a device of ``sm_count`` SMs:
    ``n`` of the ``"det_s<n>"`` spec of the row's token window that holds ``K`` (:data:`ROUTER_RULES`, or the table of
    the device's own SM count in :data:`ROUTER_RULES_BY_SM`; :func:`router_rules`), on a device where the table's
    windows hold.  ``None``: no such window -- another layout class, a token or SM count outside those windows -- and
    a launch without a caller's split count takes the deterministic package's split count of the row
    (``router_splits``).  The deterministic program adds a tile's ``n`` partials in a fixed order (its ``torch.sum``,
    or this package's partial sum where it serves the launch), so the result repeats bit for bit."""
    if layout != "nn_t" or not router_rules(arch, sm_count)[1]:
        return None
    return router_det_spec_splits(
        _router_rule_spec(layout, arch=arch, sm_count=sm_count, M=M, N=N, K=K)
    )


def router_ksum_spec_splits(spec: Union[str, int, None]) -> Optional[int]:
    """The span count of a ``"ksum_S<n>"`` window spec (the K-order program at ``n`` equal spans per tile), else
    ``None``."""
    if not isinstance(spec, str):
        return None
    m = _ROUTER_KSUM_SPEC.fullmatch(spec)
    return int(m.group(1)) if m else None


def router_ksum_template(arch: str, sm_count: int) -> Optional[str]:
    """The weight gradient's K-order program on a device of ``arch`` with ``sm_count`` SMs
    (:data:`ROUTER_KSUM_TEMPLATES_BY_SM`), or ``None``: the part has none."""
    return ROUTER_KSUM_TEMPLATES_BY_SM.get((arch, int(sm_count)))


def router_ksum_workspace(m_pairs: int, n_tiles: int, units: int) -> tuple[int, int]:
    """``(slot elements, ticket words)`` of a K-order launch of ``units`` spans over ``m_pairs x n_tiles`` tiles:
    slot ``(span + pair) * n_tiles + column`` of :data:`ROUTER_KSUM_SLOT_ELEMS` FP32 elements (a span's segments of
    consecutive row-tile pairs take consecutive slots), and :data:`ROUTER_KSUM_TILE_WORDS` words per tile."""
    spans = int(units) // int(n_tiles)
    return (
        (spans + int(m_pairs) - 1) * int(n_tiles) * ROUTER_KSUM_SLOT_ELEMS,
        int(m_pairs) * int(n_tiles) * ROUTER_KSUM_TILE_WORDS,
    )


def router_simt_template(
    tile: tuple[Union[int, str], ...], layout: str = "nn_t"
) -> str:
    """The template of the SIMT program of the tile ``(tile_n, tile_m, slices)`` (plain) or ``(tile_n, tile_m, slices,
    form)`` (compensated, :func:`router_simt_tile`) of the layout class ``layout``: the weight gradient's
    ``router_fp32_gemm_simt_n<tile_n>m<tile_m>s<slices>`` / ``router_fp32_gemm_simt_kahan_n<tile_n>m<tile_m>s<slices>
    g<group>`` (``g1``: no suffix) / ``router_fp32_gemm_simt_kahan_dot2_n<tile_n>m<tile_m>s<slices>`` (``nn_t``), the
    input gradient's ``router_fp32_dgrad_simt_m<tile_m>n<tile_n>s<slices>`` /
    ``router_fp32_dgrad_simt_kahan_m<tile_m>n<tile_n>s<slices>g<group>`` (``kn``)."""
    if len(tile) not in (3, 4):
        raise ValueError(f"router_simt_template: {tile!r} is not a SIMT tile")
    tile_n, tile_m, slices = (int(t) for t in tile[:3])
    form = str(tile[3]) if len(tile) == 4 else None
    if form is not None and not re.fullmatch(r"g[1-9]\d*|dot2", form):
        raise ValueError(f"router_simt_template: {tile!r} is not a SIMT tile")
    if layout not in ("nn_t", "kn"):
        raise ValueError(
            f"router_simt_template: no SIMT program of the {layout!r} class"
        )
    if form == "dot2" and layout == "kn":
        raise ValueError(
            f"router_simt_template: {tile!r}: the Dot2 form is the weight gradient's"
        )
    kahan = "" if form is None else "kahan_dot2_" if form == "dot2" else "kahan_"
    suffix = "" if form in (None, "dot2", "g1") else form
    if layout == "kn":
        return (
            f"{cake_jit.TEMPLATE_PREFIXES[4]}{kahan}m{tile_m}n{tile_n}s{slices}{suffix}"
        )
    return f"{cake_jit.TEMPLATE_PREFIXES[3]}{kahan}n{tile_n}m{tile_m}s{slices}{suffix}"


def router_head_tail_template(arch: str, sm_count: Optional[int] = None) -> str:
    """The template of the weight gradient's head/tail program on ``arch`` (on a device of ``sm_count`` SMs):
    :data:`ROUTER_HEAD_TAIL_TEMPLATES_BY_SM` for exactly ``(arch, sm_count)``, else
    :data:`ROUTER_HEAD_TAIL_TEMPLATES`, else ``router_fp32_gemm_atomic_nn_t`` + :data:`ROUTER_HEAD_TAIL_SUFFIX`
    (``sm_count`` ``None``: the architecture's program)."""
    if sm_count is not None:
        template = ROUTER_HEAD_TAIL_TEMPLATES_BY_SM.get((arch, int(sm_count)))
        if template is not None:
            return template
    return ROUTER_HEAD_TAIL_TEMPLATES.get(
        arch, cake_jit.TEMPLATE_PREFIXES[1] + "nn_t" + ROUTER_HEAD_TAIL_SUFFIX
    )


def router_partition_of(
    spec: Union[str, int], m_pairs: int, n_tiles: int, k_iters: int, sm_count: int
) -> tuple[int, int, int]:
    """``(units, head_spans, head_steps)`` of a partition spec on a device of ``sm_count`` SMs (CTA pairs
    ``sm_count // 2``).  A head/tail spec ``"H<f>x<k>"``: one head unit per CTA pair, then ``k`` tail units per CTA
    pair (each count rounded down to a multiple of the column tile count, at least ``n_tiles``); the ``head_spans``
    head spans share the first ``head_steps`` of the ``m_pairs * k_iters`` (row-tile pair, K block) steps, about the
    fraction ``f`` (at least one step per head span, at least one per tail span left), the tail spans the rest.  Every
    pair starts a long head span at once and the short tail spans go to the pairs that finish first, so a kernel
    holding SMs on another stream delays a short span, not a long one.  ``"H<f>x<k>p<h>"``: ``h`` head units instead
    (at most one per CTA pair, rounded down to a multiple of the column tile count, at least ``n_tiles``), each span
    about ``f`` of a pair's even share of the steps (the heads share about ``f * h / pairs`` of them); the other pairs
    start on the tail spans, so a kernel holding the SMs of up to ``pairs - h`` CTA pairs on another stream delays no
    head span.  Any other spec: :func:`router_units_of`'s units with every span alike (``head_spans = head_steps =
    0``)."""
    head_tail = router_head_tail(spec)
    if head_tail is None:
        return router_units_of(spec, m_pairs, n_tiles, k_iters, sm_count), 0, 0
    fraction, per_pair = head_tail
    pairs, steps = int(sm_count) // 2, m_pairs * k_iters
    all_pairs = max(n_tiles, pairs // n_tiles * n_tiles)
    head_pairs = router_head_pairs(spec)
    if head_pairs is None:
        head_units, head_share = all_pairs, round(fraction * steps)
    else:
        head_units = max(n_tiles, min(head_pairs, pairs) // n_tiles * n_tiles)
        head_share = round(fraction * steps * head_units / all_pairs)
    units = head_units + max(n_tiles, per_pair * pairs // n_tiles * n_tiles)
    head_spans, spans = head_units // n_tiles, units // n_tiles
    head_steps = min(max(head_share, head_spans), steps - (spans - head_spans))
    return units, head_spans, head_steps


def router_partition_fits(
    m_pairs: int,
    n_tiles: int,
    k_iters: int,
    units: int,
    head_spans: int,
    head_steps: int,
) -> bool:
    """Whether the program can run a partition: at most one unit per (tile, K block) step and, for a head/tail
    partition, at least one step per span."""
    steps, spans = m_pairs * k_iters, units // n_tiles
    if units > m_pairs * n_tiles * k_iters:
        return False
    return head_spans == 0 or (
        head_spans <= head_steps and steps - head_steps >= spans - head_spans
    )


def _router_span_first(
    span: int, steps: int, spans: int, head_spans: int, head_steps: int
) -> int:
    """The first (row-tile pair, K block) step of span ``span``: the head spans share the first ``head_steps`` steps
    evenly, the others the rest (every span alike when ``head_spans`` is 0), as the program cuts them."""
    if span < head_spans:
        return span * head_steps // head_spans
    return head_steps + (span - head_spans) * (steps - head_steps) // (
        spans - head_spans
    )


def router_has_part_tiles(
    m_pairs: int,
    n_tiles: int,
    k_iters: int,
    units: int,
    head_spans: int = 0,
    head_steps: int = 0,
) -> bool:
    """Whether a launch adds into ``out`` anywhere (then ``out`` is zeroed before the program): some span of the
    partition starts inside a row-tile pair."""
    steps, spans = m_pairs * k_iters, units // n_tiles
    return any(
        _router_span_first(span, steps, spans, head_spans, head_steps) % k_iters
        for span in range(1, spans)
    )


def router_at_most_two_partials(m_pairs: int, n_tiles: int, units: int) -> bool:
    """Whether every element of the launch takes at most two partials (then the result repeats bit for bit): at most
    one unit per tile (a span of at least a pair's K blocks meets a pair at most twice), or exactly two."""
    tiles = m_pairs * n_tiles
    return units <= tiles or units == 2 * tiles


def _router_grid(M: int, N: int, K: int) -> tuple[int, int, int]:
    """``(m_tiles, n_tiles, k_iters)`` of a router launch: an even count of 128-row tiles (a CTA pair owns two
    adjacent ones), the 128-column tiles and the 32-element K blocks."""
    m_tiles = _ceil_div(M, ROUTER_BLOCK_M)
    m_tiles += m_tiles % 2
    return m_tiles, _ceil_div(N, ROUTER_BLOCK_N), _ceil_div(K, ROUTER_BLOCK_K)


def _router_rule_spec(
    layout: str, *, arch: str, sm_count: int, M: int, N: int, K: int
) -> Optional[Union[str, int]]:
    """The verdict for the launched GEMM ``[M, N] = [M, K] @ [K, N]`` of the layout class ``layout`` (``kk`` /
    ``kn`` / ``nn_t``) in the router table of a device of ``arch`` with ``sm_count`` SMs (:func:`router_rules`: the
    table of its own in :data:`ROUTER_RULES_BY_SM`, else :data:`ROUTER_RULES`): the row's spec, or the spec of its
    token window that holds the token count (``K`` of the weight gradient, ``M`` of the forward and of the input
    gradient; a SIMT window only on a device where the table's windows hold, else the next window that holds it).
    ``None``: no verdict -- a launch without a caller's split count runs the deterministic program
    (:func:`router_partition`)."""
    if layout == "kk":
        hidden, experts, tokens = K, N, M
    elif layout == "kn":
        hidden, experts, tokens = N, K, M
    else:
        hidden, experts, tokens = M, N, K
    rules, holds = router_rules(arch, sm_count)
    rule = rules.get((arch, layout, int(hidden), int(experts)))
    if isinstance(rule, tuple):  # token windows (min_tokens, max_tokens, spec)
        # a SIMT window only where the table's windows hold; elsewhere the next window that holds the token count
        return next(
            (
                s
                for lo, hi, s in rule
                if lo <= int(tokens) <= hi and (holds or not _router_is_simt(s))
            ),
            None,
        )
    return rule


def router_partition(
    layout: str,
    *,
    arch: str,
    M: int,
    N: int,
    K: int,
    sm_count: int,
    splits: Optional[int] = None,
) -> Optional[tuple[int, int, int]]:
    """``(units, head_spans, head_steps)`` of this package's K2 launch of the layout class ``layout`` (``kk``,
    ``kn``, ``nn_t``) for ``[M, N] = [M, K] @ [K, N]`` (the launched, possibly swapped GEMM) on a device of
    ``sm_count`` SMs (:func:`router_partition_of`; ``head_spans`` 0: every span alike), or ``None``: the launch runs
    the deterministic program.

    * ``kn`` (input gradient): ``None`` (no stream-K program; its SIMT windows are :func:`router_simt`'s).
    * a caller's ``splits``: ``splits`` units per tile (``1``: every tile stored whole, the deterministic program's
      one-split bits -- the control of a measurement); the forward takes at most
      :data:`ROUTER_FORWARD_MAX_SPLITS`.
    * otherwise the row's verdict in :data:`ROUTER_RULES` (on a device with a table of its own in
      :data:`ROUTER_RULES_BY_SM`, that table; :func:`router_rules`): the spec of its token window that holds the
      launch's token count (``K`` of the weight gradient, ``M`` of the forward), on a device of
      :data:`ROUTER_SM_COUNT` SMs or one with a table of its own (a ``"waves"`` window on every device where
      :func:`router_waves` gives a partition against the deterministic package's split count of the row, its
      ``router_splits``, that gives every element at most two partials, :func:`router_at_most_two_partials`); a row,
      token count or SM count without one, or a head/tail partition the token count cannot hold
      (:func:`router_partition_fits`), runs the deterministic program.  A SIMT window is no partition (``None``):
      the launch runs the SIMT program there (:func:`router_simt`); nor is a ``"det_s<n>"`` window: the launch runs the
      deterministic program at ``n`` splits (:func:`router_det_splits`).  A ``"ksum_S<n>"`` window of the weight
      gradient is the partition ``"S<n>"`` (``n`` units per tile), which :func:`plan_atomic_router` runs on the
      K-order program of the part.

    A forward partition must give every element at most two partials (:func:`router_at_most_two_partials`) and
    cannot be a head/tail partition: ``ValueError`` otherwise."""
    if layout not in ("kk", "nn_t"):
        return None
    m_tiles, n_tiles, k_iters = _router_grid(M, N, K)
    m_pairs = m_tiles // 2
    spec: Optional[Union[str, int]]
    if splits is not None:
        s = int(splits)
        if not 1 <= s <= k_iters:
            raise ValueError(
                f"cake_dense_gemm_nondet_train_router: splits must be in [1, {k_iters}] (the 32-element K blocks of K={K}), got {s}"
            )
        if layout == "kk" and s > ROUTER_FORWARD_MAX_SPLITS:
            raise ValueError(
                f"cake_dense_gemm_nondet_train_router: the forward takes at most {ROUTER_FORWARD_MAX_SPLITS} "
                f"splits (more partials per element do not repeat bit for bit under recompute), got {s}; "
                f"{_DET_MODULE}.cake_dense_gemm_det_train_router takes more"
            )
        spec = f"S{s}"
    else:
        spec = _router_rule_spec(layout, arch=arch, sm_count=sm_count, M=M, N=N, K=K)
        if spec == "waves":  # its own SM-count condition, on every SM count
            # the deterministic plan the row runs without the partition: that package's split count of the row
            det_splits = deterministic_backend().router_splits(
                layout, M, N, K, arch, sm_count
            )
            spec = router_waves(m_pairs, n_tiles, k_iters, sm_count, det_splits)
            if spec is not None and not router_at_most_two_partials(
                m_pairs,
                n_tiles,
                router_units_of(spec, m_pairs, n_tiles, k_iters, sm_count),
            ):
                # its bits would vary run to run: the deterministic program
                return None
        elif not router_rules(arch, sm_count)[1]:
            return None
        if (
            spec is None
            or _router_is_simt(spec)
            or router_det_spec_splits(spec) is not None
        ):
            return None
        ksum = router_ksum_spec_splits(spec)
        if ksum is not None:
            if layout != "nn_t":
                return None
            spec = f"S{ksum}"
    if layout == "kk" and router_head_tail(spec) is not None:
        raise ValueError(
            f"cake_dense_gemm_nondet_train_router: the head/tail partition {spec!r} is the weight gradient's; "
            "the forward takes at most two partials per element"
        )
    units, head_spans, head_steps = router_partition_of(
        spec, m_pairs, n_tiles, k_iters, sm_count
    )
    if layout == "kk" and not router_at_most_two_partials(m_pairs, n_tiles, units):
        raise ValueError(
            f"cake_dense_gemm_nondet_train_router: the forward partition {spec!r} ({units} units over "
            f"{m_pairs * n_tiles} tiles) gives an element more than two partials"
        )
    if not router_partition_fits(
        m_pairs, n_tiles, k_iters, units, head_spans, head_steps
    ):
        return None
    return units, head_spans, head_steps


def router_units(
    layout: str,
    *,
    arch: str,
    M: int,
    N: int,
    K: int,
    sm_count: int,
    splits: Optional[int] = None,
) -> Optional[int]:
    """The unit count of :func:`router_partition` (same arguments), or ``None``: the launch runs the deterministic
    program."""
    partition = router_partition(
        layout, arch=arch, M=M, N=N, K=K, sm_count=sm_count, splits=splits
    )
    return None if partition is None else partition[0]


def router_simt(
    layout: str,
    *,
    arch: str,
    M: int,
    N: int,
    K: int,
    sm_count: int,
    splits: Optional[int] = None,
) -> Optional[tuple[Union[int, str], ...]]:
    """The tile (:func:`router_simt_tile`) of the SIMT program a launch ``[M, N] = [M, K] @ [K, N]`` runs on a device of
    ``sm_count`` SMs: the weight gradient's (the swapped GEMM: ``M`` hidden, ``N`` experts, ``K`` tokens; layout class
    ``nn_t``) when the tile divides ``M`` and ``N``, the input gradient's (``M`` tokens, ``N`` hidden, ``K`` experts;
    layout class ``kn``) when the tile divides ``N`` (the last row tile takes the rows left).  The SIMT spec of the
    row's token window that holds the token count (:data:`ROUTER_RULES`, or the table of the device's own SM count in
    :data:`ROUTER_RULES_BY_SM`; :func:`router_rules`), on a device where the table's windows hold.  ``None``: no SIMT
    window -- the forward, a caller's ``splits`` (the stream-K program of the weight gradient, the deterministic
    program of the input gradient), a token or SM count outside the SIMT windows -- and the launch takes the planner
    of its class (:func:`router_partition`; the input gradient's deterministic program)."""
    if layout not in ("nn_t", "kn") or splits is not None:
        return None
    if not router_rules(arch, sm_count)[1]:
        return None
    tile = router_simt_tile(
        _router_rule_spec(layout, arch=arch, sm_count=sm_count, M=M, N=N, K=K)
    )
    if (
        tile is None
        or int(N) % int(tile[0])
        or (layout == "nn_t" and int(M) % int(tile[1]))
    ):
        return None
    return tile


def router_forward_simt(
    *,
    arch: str,
    M: int,
    N: int,
    K: int,
    sm_count: int,
    splits: Optional[int] = None,
) -> Optional[tuple[str, int, int, int]]:
    """The tile (:func:`router_forward_simt_tile`) of the forward's SIMT program a launch ``out[M, N] = X[M, K] @
    W[N, K].T`` (``M`` tokens, ``N`` experts, ``K`` hidden; layout class ``kk``) runs on a device of ``sm_count`` SMs:
    the spec of the row's token window that holds ``M`` (:data:`ROUTER_RULES`, or the table of the device's own SM
    count in :data:`ROUTER_RULES_BY_SM`; :func:`router_rules`) on a device where the table's windows hold, when the
    tile's experts divide ``N`` and ``K`` is a multiple of :data:`ROUTER_FWD_SIMT_CLUSTER` x
    :data:`ROUTER_FWD_SIMT_BLOCK_K`.  ``None``: a caller's ``splits`` (this package's stream-K program), a token or SM
    count outside the windows -- the deterministic program (:func:`router_partition`)."""
    if splits is not None or not router_rules(arch, sm_count)[1]:
        return None
    tile = router_forward_simt_tile(
        _router_rule_spec("kk", arch=arch, sm_count=sm_count, M=M, N=N, K=K)
    )
    if (
        tile is None
        or int(N) % tile[2]
        or int(K) % (ROUTER_FWD_SIMT_CLUSTER * ROUTER_FWD_SIMT_BLOCK_K)
    ):
        return None
    return tile


@dataclass(frozen=True)
class CakeDenseGemmNondetRouterPlan:
    """The host plan of one K2 launch of this package (the launched, possibly swapped GEMM ``[M, N] = [M, K] @
    [K, N]``): ``m_tiles / 2 x n_tiles`` tiles of ``256 x 128``, their ``k_iters`` K blocks cut into ``units``
    stream-K spans (``2 * units`` CTAs; the first ``head_spans`` spans share the first ``head_steps`` steps on the
    head/tail program, every span alike when ``head_spans`` is 0); with part tiles the launch zeroes ``out`` before
    the program.  With ``ksum`` (a ``"ksum_S<n>"`` window, :func:`router_ksum_template`) the K-order program instead:
    the spans' partials of a tile summed in K order through workspace slots and tickets (:attr:`ksum_workspace`), no
    fill of ``out``.  With ``simt`` (a SIMT program, its tile, :func:`router_simt`): one CTA per
    ``tile_n x tile_m`` tile of ``out`` over ``k_iters`` K blocks of :data:`ROUTER_SIMT_BLOCK_K` elements, the
    operands' row strides ``lda`` / ``ldb``; ``units`` 0, no part tiles.  The weight gradient's (``nn_t``) launches
    ``m_tiles x n_tiles`` CTAs over the transposed ``out`` (``A`` the activations, ``B`` the output gradient), the
    input gradient's (``kn``) ``n_tiles x m_tiles`` CTAs, ``m_tiles`` row tiles of ``tile_m`` tokens, the last one
    partial (``A`` the output gradient, ``B`` the router weight).  The forward's (``kk``, ``simt`` a tile of
    :func:`router_forward_simt`) launches ``ROUTER_FWD_SIMT_CLUSTER * n_tiles x m_tiles`` CTAs in clusters of
    :data:`ROUTER_FWD_SIMT_CLUSTER` (``n_tiles`` tiles of ``experts`` experts, ``m_tiles`` of ``tokens`` tokens, the
    last one partial; ``A`` the activations ``[T, K]``, ``ldb`` the row stride of the router weight ``[N, K]``) and
    ``k_iters`` is the cluster size."""

    M: int
    N: int
    K: int
    a_mn: bool
    b_mn: bool
    swapped: bool
    layout: str
    m_tiles: int
    n_tiles: int
    k_iters: int
    units: int
    template: str
    head_spans: int = 0
    head_steps: int = 0
    simt: Optional[tuple[Union[int, str], ...]] = None
    lda: int = 0
    ldb: int = 0
    ksum: bool = False

    @property
    def m_pairs(self) -> int:
        return self.m_tiles // 2

    @property
    def ksum_workspace(self) -> tuple[int, int]:
        """``(slot elements, ticket words)`` of the K-order program (:func:`router_ksum_workspace`; ``(0, 0)`` for
        any other program)."""
        if not self.ksum:
            return (0, 0)
        return router_ksum_workspace(self.m_pairs, self.n_tiles, self.units)

    @property
    def tiles(self) -> int:
        return self.m_pairs * self.n_tiles

    @property
    def grid(self) -> tuple[int, int, int]:
        if self.simt is not None:
            if self.layout == "kn":
                return (self.n_tiles, self.m_tiles, 1)
            if self.layout == "kk":
                return (ROUTER_FWD_SIMT_CLUSTER * self.n_tiles, self.m_tiles, 1)
            return (self.m_tiles, self.n_tiles, 1)
        return (2 * self.units, 1, 1)

    @cached_property
    def part_tiles(self) -> bool:
        """Whether some segment adds into ``out`` (the launch zeroes ``out`` before its program runs); computed once
        per plan.  The K-order program (``ksum``) stores every element once."""
        if self.simt is not None or self.ksum:
            return False
        return router_has_part_tiles(
            self.m_pairs,
            self.n_tiles,
            self.k_iters,
            self.units,
            self.head_spans,
            self.head_steps,
        )

    @property
    def reproducible(self) -> bool:
        """Whether every element takes at most two partials, so the result repeats bit for bit (the SIMT program
        stores every element once; the K-order program adds the partials in K order)."""
        if self.simt is not None or self.ksum:
            return True
        return router_at_most_two_partials(self.m_pairs, self.n_tiles, self.units)


def _router_a_mn(A: torch.Tensor) -> Optional[bool]:
    if A.stride(1) == 1:
        return False
    if A.stride(0) == 1:
        return True
    return None


def _router_b_mn(B: torch.Tensor) -> Optional[bool]:
    if B.stride(0) == 1:
        return False
    if B.stride(1) == 1:
        return True
    return None


def plan_atomic_router(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    *,
    arch: str,
    sm_count: int,
    splits: Optional[int] = None,
) -> tuple[Optional[CakeDenseGemmNondetRouterPlan], str]:
    """Plan this package's K2 GEMM ``out[M, N] = A[M, K] @ B[K, N]`` (fp32 views, the deterministic package's view
    contract and swap rule) of the ``kk`` / ``nn_t`` classes, and of the ``kn`` class in its SIMT windows.  Returns
    ``(plan, reason)``: ``plan`` is ``None`` when the launch runs the deterministic program, and ``reason`` says why --
    another layout class, a ``kn`` launch outside the SIMT windows, a view the reduce-add
    program cannot address (an ``out`` off a 16-byte boundary or with a row stride not a multiple of 4 elements runs
    the deterministic path, which serves or refuses it as that package does), a template this checkout does not
    register for ``arch``, or the planner (:func:`router_partition`).  An explicit ``splits`` (``1`` included) runs
    this package's program when the view and the registry allow it; a head/tail partition runs the head/tail program
    of the architecture on a device of ``sm_count`` SMs (:func:`router_head_tail_template`), a ``"ksum_S<n>"`` window
    the K-order program of the part (:func:`router_ksum_template`; the deterministic program where the part has none
    registered); a SIMT window
    (:func:`router_simt`) runs the SIMT program of its tile when the operands' rows are whole rows of a multiple of 4
    elements (else the deterministic program): the weight gradient's over the swapped launch, the input gradient's
    over ``A`` (the output gradient, K-major) and ``B`` (the router weight, N-major) as they are."""
    if (
        A.dtype != torch.float32
        or B.dtype != torch.float32
        or out.dtype != torch.float32
    ):
        raise ValueError(
            "cake_dense_gemm_nondet_train_router: A, B and out must be fp32"
        )
    if A.dim() != 2 or B.dim() != 2 or out.dim() != 2:
        raise ValueError(
            "cake_dense_gemm_nondet_train_router: A, B and out must be 2-D views"
        )
    M, K = (int(s) for s in A.shape)
    Kb, N = (int(s) for s in B.shape)
    if Kb != K or tuple(out.shape) != (M, N):
        raise ValueError(
            f"cake_dense_gemm_nondet_train_router: A [{M}, {K}], B [{Kb}, {N}] and out {list(out.shape)} do not agree"
        )
    if min(M, K, N) < 1:
        raise ValueError(
            f"cake_dense_gemm_nondet_train_router: M, K and N must be >= 1 (M={M}, N={N}, K={K})"
        )
    a_mn, b_mn = _router_a_mn(A), _router_b_mn(B)
    if a_mn is None or b_mn is None:
        return None, "an operand without a unit stride"
    layout = _k2_class(a_mn, b_mn)[3:]
    if layout not in ("kk", "kn", "nn_t"):
        return None, f"the {layout} class runs the deterministic program"
    swapped = layout == "nn_t"
    if swapped:
        A_eff, B_eff, M_eff, N_eff = B.t(), A.t(), N, M
    else:
        A_eff, B_eff, M_eff, N_eff = A, B, M, N
    a_eff_mn, b_eff_mn = bool(_router_a_mn(A_eff)), bool(_router_b_mn(B_eff))
    lda = int(A_eff.stride(1)) if a_eff_mn else int(A_eff.stride(0))
    if lda % 4 or A_eff.data_ptr() % 16 or B_eff.data_ptr() % 16:
        return None, "an operand off a 16-byte boundary"
    ldo = int(out.stride(0))
    if out.stride(1) != 1 or out.data_ptr() % 16 or ldo % 4:
        return (
            None,
            "out off a 16-byte boundary, or its rows not a multiple of 4 elements (the reduce-add map)",
        )
    if N_eff % 8:
        return None, "N not a multiple of 8"
    if ldo < (M_eff if swapped else N_eff):
        return None, "out rows shorter than the output"
    forward = (
        router_forward_simt(
            arch=arch, M=M_eff, N=N_eff, K=K, sm_count=sm_count, splits=splits
        )
        if layout == "kk"
        else None
    )
    if forward is not None:
        # out[M, N] = X[M, K] @ W[N, K].T: the activations' rows (lda) and the router weight's rows (ldb), K-major
        ldb = int(B_eff.stride(1))
        if K % 4 or lda < K or ldb % 4 or ldb < K:
            return (
                None,
                "the forward's SIMT program reads the activations' and router weight's rows whole, multiples of 4 "
                "elements",
            )
        template = router_forward_simt_template(forward)
        if template not in cake_jit.KERNELS.get(arch, {}):
            return None, f"{template} is not registered for {arch}"
        _, _, experts, tokens = forward
        return (
            CakeDenseGemmNondetRouterPlan(
                M=M_eff,
                N=N_eff,
                K=K,
                a_mn=a_eff_mn,
                b_mn=b_eff_mn,
                swapped=False,
                layout=layout,
                m_tiles=_ceil_div(M_eff, tokens),
                n_tiles=N_eff // experts,
                k_iters=ROUTER_FWD_SIMT_CLUSTER,
                units=0,
                template=template,
                simt=forward,
                lda=lda,
                ldb=ldb,
            ),
            "",
        )
    simt = router_simt(
        layout, arch=arch, M=M_eff, N=N_eff, K=K, sm_count=sm_count, splits=splits
    )
    if simt is not None:
        ldb = int(B_eff.stride(0))
        if swapped:
            # out[N, M] = B_eff.T @ A_eff.T: the activations' rows (lda) and the output gradient's rows (ldb), K-major
            if not (a_eff_mn and b_eff_mn) or lda < M_eff:
                return None, "the SIMT program reads the activations' rows whole"
            if ldb % 4 or ldb < N_eff:
                return (
                    None,
                    "the SIMT program reads the output gradient's rows whole, a multiple of 4 elements",
                )
        else:
            # out[M, N] = A @ B: the output gradient's rows (lda, K-major) and the router weight's rows (ldb, N-major)
            if K % 4 or lda < K:
                return (
                    None,
                    "the SIMT program reads the output gradient's rows whole, K a multiple of 4 elements",
                )
            if ldb % 4 or ldb < N_eff:
                return (
                    None,
                    "the SIMT program reads the router weight's rows whole, a multiple of 4 elements",
                )
        template = router_simt_template(simt, layout)
        if template not in cake_jit.KERNELS.get(arch, {}):
            return None, f"{template} is not registered for {arch}"
        tile_n, tile_m = int(simt[0]), int(simt[1])
        return (
            CakeDenseGemmNondetRouterPlan(
                M=M_eff,
                N=N_eff,
                K=K,
                a_mn=a_eff_mn,
                b_mn=b_eff_mn,
                swapped=swapped,
                layout=layout,
                m_tiles=_ceil_div(M_eff, tile_m),
                n_tiles=N_eff // tile_n,
                k_iters=_ceil_div(K, ROUTER_SIMT_BLOCK_K),
                units=0,
                template=template,
                simt=simt,
                lda=lda,
                ldb=ldb,
            ),
            "",
        )
    if layout == "kn":
        return (
            None,
            "the kn class runs the deterministic program outside its SIMT windows",
        )
    template = cake_jit.TEMPLATE_PREFIXES[1] + ("nn_t" if swapped else "kk")
    if template not in cake_jit.KERNELS.get(arch, {}):
        return None, f"{template} is not registered for {arch}"
    partition = router_partition(
        layout, arch=arch, M=M_eff, N=N_eff, K=K, sm_count=sm_count, splits=splits
    )
    if partition is None:
        return None, "the planner keeps the row on the deterministic program"
    units, head_spans, head_steps = partition
    ksum = (
        splits is None
        and layout == "nn_t"
        and router_ksum_spec_splits(
            _router_rule_spec(
                layout, arch=arch, sm_count=sm_count, M=M_eff, N=N_eff, K=K
            )
        )
        is not None
    )
    if ksum:
        template = router_ksum_template(arch, sm_count)
        if template is None or template not in cake_jit.KERNELS.get(arch, {}):
            return (
                None,
                f"no K-order program is registered for {arch} on {sm_count} SMs",
            )
    elif head_spans:
        template = router_head_tail_template(arch, sm_count)
        if template not in cake_jit.KERNELS.get(arch, {}):
            return None, f"{template} is not registered for {arch}"
    m_tiles, n_tiles, k_iters = _router_grid(M_eff, N_eff, K)
    plan = CakeDenseGemmNondetRouterPlan(
        M=M_eff,
        N=N_eff,
        K=K,
        a_mn=a_eff_mn,
        b_mn=b_eff_mn,
        swapped=swapped,
        layout=layout,
        m_tiles=m_tiles,
        n_tiles=n_tiles,
        k_iters=k_iters,
        units=units,
        template=template,
        head_spans=head_spans,
        head_steps=head_steps,
        ksum=ksum,
    )
    return plan, ""


#: The geometry of a K2 call the convenience path reads once (one call per tensor attribute):
#: ``(A.shape, B.shape, out.shape, A.stride(), B.stride(), (A.dtype, B.dtype, out.dtype))``.
_K2Geometry = tuple


def _k2_geometry(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor) -> _K2Geometry:
    return (
        A.shape,
        B.shape,
        out.shape,
        A.stride(),
        B.stride(),
        (A.dtype, B.dtype, out.dtype),
    )


def _k2_geometry_without_verdict(
    geometry: _K2Geometry, arch: str, sm_count: int
) -> bool:
    """:func:`_k2_row_without_verdict` of a call's :func:`_k2_geometry`."""
    sa, sb, so, a_st, b_st, dtypes = geometry
    f32 = torch.float32
    if (
        dtypes[0] != f32
        or dtypes[1] != f32
        or dtypes[2] != f32
        or len(sa) != 2
        or len(sb) != 2
        or len(so) != 2
    ):
        return False
    M, K = sa
    Kb, N = sb
    if Kb != K or so[0] != M or so[1] != N or min(M, K, N) < 1:
        return False
    # the planner's operand classes (_router_a_mn / _router_b_mn) from the strides
    a_mn = False if a_st[1] == 1 else (True if a_st[0] == 1 else None)
    b_mn = False if b_st[0] == 1 else (True if b_st[1] == 1 else None)
    if a_mn is None or b_mn is None or (a_mn and not b_mn):
        return False
    if a_mn:  # nn_t: the swapped launch [N, M] = [N, K] @ [K, M]
        spec = _router_rule_spec("nn_t", arch=arch, sm_count=sm_count, M=N, N=M, K=K)
    elif b_mn:  # kn: the input gradient
        spec = _router_rule_spec("kn", arch=arch, sm_count=sm_count, M=M, N=N, K=K)
    else:
        spec = _router_rule_spec("kk", arch=arch, sm_count=sm_count, M=M, N=N, K=K)
    return spec is None


def _k2_row_without_verdict(
    A: torch.Tensor, B: torch.Tensor, out: torch.Tensor, arch: str, sm_count: int
) -> bool:
    """Whether :func:`plan_atomic_router` keeps a launch without a caller's split count on the deterministic program
    for want of a verdict: valid 2-D fp32 views of the ``kk`` / ``kn`` / ``nn_t`` class whose row has no spec on a
    device of ``arch`` with ``sm_count`` SMs (:func:`_router_rule_spec` of the launched GEMM), when no other check of
    the planner can give a plan.  ``False`` leaves the decision (and every refusal of an invalid binding) to the
    planner."""
    return _k2_geometry_without_verdict(_k2_geometry(A, B, out), arch, sm_count)


def _k2_det_splits(
    A: torch.Tensor, B: torch.Tensor, arch: str, sm_count: int
) -> Optional[int]:
    """:func:`router_det_splits` of a call ``out = A @ B`` of the weight-gradient class (both operands MN-major, the
    swapped launch ``[N, M] = [N, K] @ [K, M]``) on a device of ``arch`` with ``sm_count`` SMs; ``None`` for another
    class or view (the deterministic package's split count of the row)."""
    if A.dim() != 2 or B.dim() != 2 or not (_router_a_mn(A) and _router_b_mn(B)):
        return None
    M, K = (int(s) for s in A.shape)
    N = int(B.shape[1])
    return router_det_splits("nn_t", arch=arch, M=N, N=M, K=K, sm_count=sm_count)


@dataclass(slots=True)
class _RouterMemo:
    """What the convenience path keeps from the router tables bound at ``generation`` (:data:`_VERDICT_GENERATION`,
    the tables ``rules`` / ``by_sm`` as :data:`ROUTER_RULES` / :data:`ROUTER_RULES_BY_SM`): whether some table gives
    the forward (``kk``) a verdict, and the conclusions of :func:`_k2_geometry_without_verdict` per (device index,
    :func:`_k2_geometry`) -- a device's architecture and SM count are fixed for the process.  ``tracked`` is false
    when some table in use is not a :class:`_VerdictTable`: such a memo is derived again on every call."""

    generation: int
    rules: Any
    by_sm: Any
    tracked: bool
    forward_verdicts: bool
    without_verdict: dict = field(default_factory=dict)


#: The most conclusions a :class:`_RouterMemo` keeps (it starts over when full: a geometry is one token count).
_ROUTER_MEMO_SIZE = 1024
_ROUTER_MEMO = _RouterMemo(-1, None, None, False, True)


def _router_memo() -> _RouterMemo:
    """The :class:`_RouterMemo` of the router tables bound now: the kept one while no table changed or was rebound,
    else a new one (the generation is read before the tables, so a change while it is derived renews it again)."""
    global _ROUTER_MEMO
    memo = _ROUTER_MEMO
    if (
        memo.generation == _VERDICT_GENERATION
        and memo.tracked
        and memo.rules is ROUTER_RULES
        and memo.by_sm is ROUTER_RULES_BY_SM
    ):
        return memo
    generation, rules, by_sm = _VERDICT_GENERATION, ROUTER_RULES, ROUTER_RULES_BY_SM
    tables = (rules, *by_sm.values())
    tracked = isinstance(by_sm, _VerdictTable) and all(
        isinstance(t, _VerdictTable) for t in tables
    )
    forward = any(
        rule is not None for t in tables for key, rule in t.items() if key[1] == "kk"
    )
    memo = _RouterMemo(generation, rules, by_sm, tracked, forward)
    _ROUTER_MEMO = memo
    return memo


def _k2_runs_without_verdict(
    geometry: _K2Geometry, index: int, memo: Optional[_RouterMemo] = None
) -> bool:
    """:func:`_k2_geometry_without_verdict` of a call's :func:`_k2_geometry` on CUDA device ``index`` (its
    architecture and SM count), kept in ``memo`` (default: the :class:`_RouterMemo` of the tables bound now)."""
    if memo is None:
        memo = _router_memo()
    key = (index, geometry)
    without = memo.without_verdict.get(key)
    if without is None:
        facts = deterministic_backend().device_facts(index)
        without = facts.arch is not None and _k2_geometry_without_verdict(
            geometry, facts.arch, facts.sm_count
        )
        if len(memo.without_verdict) >= _ROUTER_MEMO_SIZE:
            memo.without_verdict.clear()
        memo.without_verdict[key] = without
    return without


def _k2_key(
    index: int,
    stream: int,
    split_pdl: tuple[Optional[int], int],
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    geometry: _K2Geometry,
    out_ptr: int,
) -> tuple:
    """The convenience key of a K2 call (the deterministic package's ``convenience_key(index, stream, ("k2",
    (splits, pdl)), A, B, out)``) from its :func:`_k2_geometry` and ``out``'s data pointer, read once; the PDL
    flag is part of the key because it is bound into the argument lists."""
    sa, sb, so, a_st, b_st, (da, db, do) = geometry
    return (
        int(index),
        int(stream),
        ("k2", split_pdl),
        (tuple(sa), a_st, da, A.storage_offset(), A.data_ptr() & 15),
        (tuple(sb), b_st, db, B.storage_offset(), B.data_ptr() & 15),
        (tuple(so), out.stride(), do, out.storage_offset(), out_ptr & 15),
    )


def _k2_views(
    A: torch.Tensor, B: torch.Tensor, plan: CakeDenseGemmNondetRouterPlan
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(a_view, split_src)``: the streamed operand as its ``[1, M, K]`` / ``[1, K, M]`` TMA span and the source of
    the BF16x3 split of the second operand (the deterministic package's convention), after the swap.  The swapped
    operands are the transposes ``A_eff = B.t()``, ``B_eff = A.t()``, so ``A_eff.t()`` is ``B`` and ``B_eff.t()`` is
    ``A``: each view is taken in at most two view operations of the caller's tensor (the same sizes, strides, storage
    offset and data pointer as the views of the swapped operands)."""
    if plan.swapped:
        split_src = A.t() if plan.b_mn else A
        a_view = (B if plan.a_mn else B.t()).unsqueeze(0)
    else:
        split_src = B if plan.b_mn else B.t()
        a_view = (A.t() if plan.a_mn else A).unsqueeze(0)
    return a_view, split_src


def _k2_out32(out: torch.Tensor) -> torch.Tensor:
    return out if out.is_contiguous() else deterministic_backend().flat_alias(out)


def _k2_values(
    plan: CakeDenseGemmNondetRouterPlan,
    *,
    A: Any,
    parts: Any,
    out: torch.Tensor,
    enable_pdl: int = 0,
    ws: Any = None,
    counters: Any = None,
) -> dict[str, Any]:
    """The keyword arguments of a K2 launch of this package (ordered by the program's argument plan): ``out`` is
    passed twice, as the store pointer and as the reduce-add map ``D``; the K-order program takes its workspace slots
    ``ws`` and tickets ``counters`` instead of ``D``; ``head_spans`` / ``head_steps`` are read by the head/tail
    programs only; ``enable_pdl`` is the launch's PDL flag (0 or 1)."""
    return dict(
        A=A,
        B=parts,
        out32=_k2_out32(out),
        D=out,
        ws=ws,
        counters=counters,
        M=plan.M,
        N=plan.N,
        m_tiles=plan.m_tiles,
        n_tiles=plan.n_tiles,
        k_iters=plan.k_iters,
        ldo=int(out.stride(0)),
        units=plan.units,
        head_spans=plan.head_spans,
        head_steps=plan.head_steps,
        enable_pdl=enable_pdl,
    )


def _simt_values(
    plan: CakeDenseGemmNondetRouterPlan,
    *,
    A: Any,
    B: Any,
    out: Any,
    ldo: int,
    enable_pdl: int = 0,
) -> dict[str, Any]:
    """The keyword arguments of a launch of a SIMT program (ordered by the program's argument plan): ``A`` the streamed
    operand's span and ``B`` the split source of the stream-K programs -- the weight gradient's activations
    ``[1, K, M]`` and output gradient ``[K, N]``, the input gradient's output gradient ``[1, M, K]`` and router weight
    ``[K, N]`` -- and ``out`` (the transposed product ``[N, M]``, the product ``[M, N]``), read in place through their
    row strides ``lda`` / ``ldb`` / ``ldo``.  A SIMT program of a gradient never launches as a programmatic dependent
    launch, whatever the call's PDL flag (``enable_pdl`` is ignored there): its CTAs, launched early next to a running
    predecessor, would be placed only on the SMs that predecessor leaves free, several per SM, and finish later than
    spread over every SM once it ends (on 152-SM GB200 parts the backward pass, input and weight gradient, at hidden
    5120 and 6144 and 192 and 256 tokens was 12 to 16 % slower in CUDA graphs than with the weight gradient's program
    launched after its predecessor; the input gradient's program was not faster as a dependent launch).  The forward's
    cluster program (layout ``kk``) takes ``X`` (``A``, the activations' span ``[1, T, K]``), ``W`` (``B``, the router
    weight ``[N, K]``), ``T``, ``ldx``, ``ldw`` and launches with the call's PDL flag, as it was measured."""
    if plan.layout == "kk":
        return dict(
            X=A,
            W=B,
            out=out,
            T=plan.M,
            N=plan.N,
            K=plan.K,
            ldx=plan.lda,
            ldw=plan.ldb,
            ldo=int(ldo),
            enable_pdl=int(enable_pdl),
        )
    return dict(
        A=A,
        B=B,
        out=out,
        M=plan.M,
        N=plan.N,
        K=plan.K,
        lda=plan.lda,
        ldb=plan.ldb,
        ldo=int(ldo),
        enable_pdl=0,
    )


# ---------------------------------------------------------------------------
# Binding and the prepared launches
# ---------------------------------------------------------------------------


@dataclass
class _AtomicLaunch:
    module: str
    entry: Callable[..., Any] = field(repr=False)
    arguments: tuple = field(repr=False)
    grid: tuple[int, int, int]

    def __call__(self) -> None:
        self.entry(*self.arguments)


def _ffi_tensors(arguments: Any) -> tuple:
    """``arguments`` with every torch tensor converted to a ``tvm_ffi.Tensor`` (which keeps its torch tensor alive):
    converting a torch tensor is the larger part of the host time of an FFI call, so the launches convert their
    arguments before the first launch of a call (or once, when prepared)."""
    return tuple(
        tvm_ffi.from_dlpack(a, require_contiguous=False)
        if isinstance(a, torch.Tensor)
        else a
        for a in arguments
    )


def _converted_launch(launch: Any) -> _AtomicLaunch:
    """A bound launch (this package's or the deterministic package's) whose tensor arguments are converted to
    ``tvm_ffi.Tensor`` once (:func:`_ffi_tensors`): a prepared launch binds fixed tensors."""
    return _AtomicLaunch(
        launch.module, launch.entry, _ffi_tensors(launch.arguments), launch.grid
    )


# ``cuda.bindings.driver`` once imported (``False``: unavailable, the K2 fill of ``out`` then runs ``out.zero_()``).
_CU_DRIVER: Any = None


def _cu_driver() -> Any:
    global _CU_DRIVER
    if _CU_DRIVER is None:
        try:
            from cuda.bindings import driver as cu_driver
        except ImportError:
            cu_driver = False
        _CU_DRIVER = cu_driver
    return _CU_DRIVER


@dataclass(frozen=True)
class _OutFill:
    """The zero fill of a K2 ``out`` with part tiles (row-major fp32 rows of ``pitch`` bytes, unit inner stride) on a
    raw stream of device ``index``: one driver memset -- ``cuMemsetD32Async`` over contiguous rows,
    ``cuMemsetD2D32Async`` over padded ones; a memset node under graph capture -- in a fraction of the host time of
    ``out.zero_()``.  The memset runs in the thread's current context, so it is issued only when torch's current
    device is ``index`` (the FFI call before it has then bound that device's context to the thread); otherwise,
    without ``cuda.bindings``, or when the driver refuses the call, ``out.zero_()`` fills ``out`` on the same
    stream."""

    out: torch.Tensor = field(repr=False)
    index: int
    ptr: int
    rows: int
    cols: int
    pitch: int

    def __call__(self, stream: int) -> None:
        _fill_rows(
            self.out, self.index, self.ptr, self.rows, self.cols, self.pitch, stream
        )


def _fill_rows(
    out: torch.Tensor,
    index: int,
    ptr: int,
    rows: int,
    cols: int,
    pitch: int,
    stream: int,
) -> None:
    """The fill of :class:`_OutFill` with its fields given (the convenience path binds the geometry once per cached
    launch and passes the caller's pointer on every call)."""
    cu = _CU_DRIVER
    if cu and torch.cuda.current_device() == index:
        if rows == 1 or pitch == 4 * cols:
            (status,) = cu.cuMemsetD32Async(ptr, 0, rows * cols, stream)
        else:
            (status,) = cu.cuMemsetD2D32Async(ptr, pitch, 0, cols, rows, stream)
        if status == cu.CUresult.CUDA_SUCCESS:
            return
    out.zero_()


def _out_fill(out: torch.Tensor, index: int) -> _OutFill:
    """The :class:`_OutFill` of a K2 ``out`` (a planned view: fp32, unit inner stride, row stride a multiple of 4
    elements) on device ``index``."""
    _cu_driver()
    return _OutFill(
        out,
        index,
        int(out.data_ptr()),
        int(out.shape[0]),
        int(out.shape[1]),
        int(out.stride(0)) * out.element_size(),
    )


def _order_atomic_arguments(
    module_name: str, plan, values: dict[str, Any], grid: tuple[int, int, int]
) -> list:
    """``values`` in the order of the argument plan ``plan`` (``[kind, name]`` pairs); the ``grid`` entries come
    from ``grid``.  Fails closed: a keyword the kernel expects that the host does not provide raises ``KeyError``
    naming both sides."""
    grid_values = dict(zip(("grid_x", "grid_y", "grid_z"), grid, strict=True))
    arguments = []
    for kind, name in plan:
        if kind == "grid":
            arguments.append(grid_values[name])
        elif name in values and values[name] is not None:
            arguments.append(values[name])
        else:
            raise KeyError(
                f"generated module {module_name!r} expects argument {name!r} ({kind}); "
                f"the host binding provides {sorted(k for k, v in values.items() if v is not None)}"
            )
    return arguments


def _atomic_entry(
    module_name: str, grid: tuple[int, int, int], arch: str
) -> tuple[Callable[..., Any], list]:
    """``(entry, argument plan)`` of this package's program ``module_name`` (loaded once per program and
    architecture).  The grid must be a multiple of the cluster shape baked into the module."""
    record = MODULES[module_name]
    cluster = record.get("launch", {}).get("cluster")
    if cluster and any(g % c for g, c in zip(grid, cluster, strict=True)):
        raise ValueError(
            f"{module_name}: grid {grid} is not a multiple of the cluster shape {tuple(cluster)} baked into the module"
        )
    module = load_cake_dense_projection_gemm_module(module_name, arch)
    return getattr(module, record["ffi_entry"]), arg_plan_of(record)


def _atomic_bind(
    module_name: str, values: dict[str, Any], grid: tuple[int, int, int], arch: str
) -> tuple[Callable[..., Any], tuple]:
    """``(entry, arguments)`` of this package's program ``module_name``: the loaded FFI entry and ``values`` ordered
    by the program's argument plan."""
    entry, plan = _atomic_entry(module_name, grid, arch)
    return entry, tuple(_order_atomic_arguments(module_name, plan, values, grid))


def bind_launch(
    module_name: str, values: dict[str, Any], grid: tuple[int, int, int], arch: str
) -> Any:
    """Order ``values`` by the generated argument plan of ``module_name``, a program of this package, and load its
    entry (the deterministic package's programs bind through that package's ``bind_launch``).  Fails closed on a
    keyword the kernel expects that the host does not provide.  Takes no mode keyword."""
    entry, arguments = _atomic_bind(module_name, values, grid, arch)
    return _AtomicLaunch(module_name, entry, arguments, grid)


def _k1_dummies(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """The workspace of a one-split K1 launch, which never reads or writes it: the deterministic package's
    per-device 16-element dummies."""
    det = deterministic_backend()
    return det._dummy(device, torch.float32), det._dummy(device, torch.uint32)


@dataclass
class CakeDenseGemmNondetProjectionLaunch:
    """One prepared K1 launch of this package: the bound program and its workspace (the FP32 split slots, zeroed
    when prepared because the last split of a partial tile reads padding columns that no split writes and discards
    them; the tickets, zero when prepared and zero again after every complete launch).  ``launch()`` writes ``out``
    and returns it without allocating; it captures into a CUDA graph as one kernel.  A prepared launch must not run
    concurrently with itself (two streams would share its workspace); after a launch that did not complete (an
    aborted process step, a device error), :meth:`reset_counters` re-zeroes the tickets on the current stream."""

    module_name: str
    arch: str
    plan: CakeDenseGemmNondetProjectionPlan
    out: torch.Tensor
    tensors: dict[str, torch.Tensor] = field(repr=False)
    device_index: int = 0
    _launch: Optional[_AtomicLaunch] = field(default=None, repr=False)

    @property
    def template(self) -> str:
        return self.plan.template

    @property
    def grid(self) -> tuple[int, int, int]:
        return self.plan.grid

    @property
    def splits(self) -> int:
        return self.plan.splits

    def counters_are_zero(self) -> bool:
        """Whether the tickets hold zeros (a device-to-host read: tests and checks only)."""
        if not self.plan.workspace:
            return True
        return not bool(self.tensors["counters"].view(torch.int32).any())

    def reset_counters(self) -> None:
        """Zero the tickets on the current stream (after a launch that did not complete)."""
        if self.plan.workspace:
            self.tensors["counters"].view(torch.int32).zero_()

    def launch(self) -> torch.Tensor:
        assert self._launch is not None
        if not self.plan.reproducible:
            _alert_order_varying(_k1_launch_text(self.plan))
        with deterministic_backend()._ffi_stream_context(self.device_index):
            self._launch()
        return self.out

    __call__ = launch


@dataclass
class CakeDenseGemmNondetRouterLaunch:
    """One prepared K2 launch of this package: the retained BF16x3 stack of the second operand (re-split from the
    bound fp32 view on every ``launch()`` by the deterministic package's split stage), then ``out`` zeroed on the
    stream (one memset, :class:`_OutFill`) when a segment adds into it, then the stream-K program.  Everything but
    the current stream is bound when prepared (the FFI arguments converted once, the fill's pointer and pitch), so a
    call issues its launches back to back.  ``launch()`` writes ``out`` and returns it without allocating; it
    captures into a CUDA graph (split, memset, kernel).  The SIMT program (``plan.simt``) reads the bound fp32
    operands in place: no stack, no fill, one kernel."""

    module_name: str
    arch: str
    plan: CakeDenseGemmNondetRouterPlan
    out: torch.Tensor
    tensors: dict[str, torch.Tensor] = field(repr=False)
    device_index: int = 0
    _launch: Optional[_AtomicLaunch] = field(default=None, repr=False)
    _split: Optional[Any] = field(default=None, repr=False)
    _fill: Optional[_OutFill] = field(default=None, repr=False)

    @property
    def template(self) -> str:
        return self.plan.template

    @property
    def grid(self) -> tuple[int, int, int]:
        return self.plan.grid

    @property
    def units(self) -> int:
        return self.plan.units

    @property
    def fused_split(self) -> bool:
        return self._split is not None

    def launch(self) -> torch.Tensor:
        assert self._launch is not None
        if not self.plan.reproducible:
            # before the split, the fill of ``out`` and the program: a raise leaves ``out`` untouched
            _alert_order_varying(_k2_launch_text(self.plan))
        det = deterministic_backend()
        index = self.device_index
        stream = det._current_raw_stream(index)
        if self._split is None and self.plan.simt is None:
            t = self.tensors
            det.split_fp32_to_bf16x3_det_(t["split_src"], t["parts"], t["resid"])
        with det._ffi_stream_context(index, stream):
            if self._split is not None:
                self._split()
            if self._fill is not None:
                self._fill(stream)
            self._launch()
        return self.out

    __call__ = launch


@dataclass
class CakeDenseGemmNondetProgramLaunch:
    """One prepared K1 launch of a row that a program table of the device routes (:func:`_k1_table_call`): the call
    of the convenience entry point of the row's operation, with the views and the PDL flag given when prepared, so it
    runs that entry point's program and gives its bits -- this package's one-wave forward, input-gradient or stream-K
    weight-gradient program, its nvjet program on the deterministic package's plan, or the deterministic package's
    program the table names.  ``table`` names the table that routes the row, ``template`` the program the row names
    (this package's when ``own``, else the deterministic package's; ``None`` for knobs that name no program).
    ``launch()`` writes ``out`` and returns it; every program of these tables repeats bit for bit (no alert).  The
    first launch plans and binds the row as the entry point does (the workspace of a program that has one is the arena
    of the device and stream); launch it once before capturing it into a CUDA graph."""

    table: str
    template: Optional[str]
    own: bool
    out: torch.Tensor
    _launch: Callable[[], Any] = field(repr=False)

    def launch(self) -> torch.Tensor:
        self._launch()
        return self.out

    __call__ = launch


def _device_and_arch(A, B, out) -> tuple[torch.device, str, int]:
    det = deterministic_backend()
    device = det._same_cuda_device(A, B, out)
    return device, det.require_arch(device), det.device_facts(device).sm_count


#: The K1 knobs of this package's planner (the others are the deterministic planner's, applied when a row runs the
#: deterministic program).
_K1_OWN_KNOBS = ("splits", "block_n", "chains")


def _split_knobs(knobs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    knobs = dict(knobs)
    own = {k: knobs.pop(k) for k in _K1_OWN_KNOBS if k in knobs}
    return own, knobs


def _prepare_atomic_k1(
    plan: CakeDenseGemmNondetProjectionPlan,
    a_desc,
    b_desc,
    out3,
    out: torch.Tensor,
    arch: str,
    enable_pdl: Optional[bool],
) -> CakeDenseGemmNondetProjectionLaunch:
    det = deterministic_backend()
    device = out.device
    if plan.workspace:
        # zeroed once: the last split reads the never-written padding columns of a partial tile (and discards them)
        ws = torch.zeros(plan.ws_f32_elems, device=device, dtype=torch.float32)
        counters = torch.zeros(plan.counters_u32, device=device, dtype=torch.uint32)
    else:
        ws, counters = _k1_dummies(device)
    ldo, out_l = _k1_out_strides(out3)
    o_alias = det.flat_alias(out3)
    module_name = cake_jit.select_module(arch, plan.template)
    values = _k1_values(
        plan,
        A=a_desc,
        B=b_desc,
        out=o_alias,
        ws=ws,
        counters=counters,
        ldo=ldo,
        out_l=out_l,
        enable_pdl=int(det.pdl_enabled(enable_pdl, device)),
    )
    launch = bind_launch(module_name, values, plan.grid, arch)
    return CakeDenseGemmNondetProjectionLaunch(
        module_name=module_name,
        arch=arch,
        plan=plan,
        out=out,
        tensors=dict(A=a_desc, B=b_desc, out=o_alias, ws=ws, counters=counters),
        device_index=det._device_index(device),
        _launch=launch,
    )


def _k1_arch_and_sms(
    A, B, out, arch: Optional[str], sm_count: Optional[int]
) -> tuple[str, int]:
    """The architecture and SM count a K1 plan targets: the caller's ``arch`` / ``sm_count`` knobs, else the
    device's."""
    if arch is None or sm_count is None:
        _, device_arch, device_sms = _device_and_arch(A, B, out)
        arch = arch or device_arch
        sm_count = int(sm_count or device_sms)
    return arch, int(sm_count)


def _atomic_k1(
    op: str,
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    *,
    transposed_out: bool,
    own: dict[str, Any],
    det_call: Callable[[], Any],
    arch: Optional[str] = None,
    sm_count: Optional[int] = None,
    det_args: Optional[Callable[[], tuple]] = None,
    det_knobs: tuple = (),
    enable_pdl: Optional[bool] = None,
    weights_static: bool = False,
) -> Any:
    """This package's K1 GEMM ``out = A @ B`` (both MN-major): ``op`` is ``"plan"`` (returns the plan),
    ``"prepare"`` (the prepared launch) or ``"launch"`` (the convenience path, returns ``out``); ``own`` holds this
    package's knobs (``splits``, ``block_n``, ``chains``).  A plan or a prepared launch of a row
    :func:`plan_atomic_gemm` keeps on the deterministic program returns ``det_call()``: the deterministic package's
    object of the entry point and ``op``, with its knobs; a launch of such a row runs the deterministic package's
    convenience call ``det_args()`` through this package's binding (:func:`_launch_cached_atomic_k1`; ``det_knobs``:
    that call's knobs).
    ``enable_pdl`` (default: the device supports it) launches the program as a programmatic dependent launch;
    ``weights_static`` is the deterministic call's (:func:`cake_dense_gemm_nondet_train_projection`)."""
    if op == "launch":
        return _launch_cached_atomic_k1(
            A,
            B,
            out,
            transposed_out=transposed_out,
            own=own,
            det_call=det_call,
            det_args=det_args,
            det_knobs=det_knobs,
            enable_pdl=enable_pdl,
            weights_static=weights_static,
        )
    arch, sm_count = _k1_arch_and_sms(A, B, out, arch, sm_count)
    plan, a_desc, b_desc, out3, _ = plan_atomic_gemm(
        A, B, out, transposed_out=transposed_out, arch=arch, sm_count=sm_count, **own
    )
    if plan is None:
        return det_call()
    if op == "plan":
        return plan
    return _prepare_atomic_k1(plan, a_desc, b_desc, out3, out, arch, enable_pdl)


def _prepare_atomic_k2(
    plan: CakeDenseGemmNondetRouterPlan,
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    arch: str,
    enable_pdl: Optional[bool] = None,
) -> CakeDenseGemmNondetRouterLaunch:
    det = deterministic_backend()
    device = out.device
    pdl = int(det.pdl_enabled(enable_pdl, device))
    a_view, split_src = _k2_views(A, B, plan)
    if plan.simt is not None:
        module_name = cake_jit.select_module(arch, plan.template)
        values = _simt_values(
            plan, A=a_view, B=split_src, out=out, ldo=out.stride(0), enable_pdl=pdl
        )
        return CakeDenseGemmNondetRouterLaunch(
            module_name=module_name,
            arch=arch,
            plan=plan,
            out=out,
            tensors=dict(A=a_view, B=split_src),
            device_index=det._device_index(device),
            _launch=_converted_launch(
                bind_launch(module_name, values, plan.grid, arch)
            ),
        )
    split_module = det._fused_split_for(arch, split_src)
    parts = torch.empty((3, *split_src.shape), device=device, dtype=torch.bfloat16)
    tensors = dict(A=a_view, split_src=split_src, parts=parts)
    if split_module is None:
        tensors["resid"] = torch.empty(
            split_src.shape, device=device, dtype=torch.float32
        )
    module_name = cake_jit.select_module(arch, plan.template)
    ws = counters = None
    if plan.ksum:
        slots, words = plan.ksum_workspace
        ws = tensors["ws"] = torch.empty(slots, device=device, dtype=torch.float32)
        counters = tensors["counters"] = torch.zeros(
            words, device=device, dtype=torch.uint32
        )
    values = _k2_values(
        plan,
        A=a_view,
        parts=parts,
        out=out,
        enable_pdl=pdl,
        ws=ws,
        counters=counters,
    )
    launch = _converted_launch(bind_launch(module_name, values, plan.grid, arch))
    split_launch = None
    if split_module is not None:
        n = split_src.numel()
        split_launch = _converted_launch(
            det.bind_launch(
                split_module,
                det._split_values(split_src, parts, n, pdl),
                det._split_grid(n),
                arch,
            )
        )
    index = det._device_index(device)
    return CakeDenseGemmNondetRouterLaunch(
        module_name=module_name,
        arch=arch,
        plan=plan,
        out=out,
        tensors=tensors,
        device_index=index,
        _launch=launch,
        _split=split_launch,
        _fill=_out_fill(out, index) if plan.part_tiles else None,
    )


def _atomic_k2(
    op: str,
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    *,
    splits: Optional[int],
    enable_pdl: Optional[bool] = None,
    views: Optional[tuple] = None,
) -> Any:
    """This package's K2 GEMM ``out = A @ B`` (FP32 through the BF16x3 emulation, or a SIMT program) of the layout
    classes ``kk`` / ``kn`` / ``nn_t``: ``op`` is ``"plan"``, ``"prepare"`` or ``"launch"`` as for
    :func:`_atomic_k1`.  A plan or a prepared launch :func:`plan_atomic_router` keeps on the deterministic program is
    the deterministic package's object of ``op`` with the caller's ``splits`` (``None``: the row's split count in the
    router tables, :func:`_k2_det_splits`, else the deterministic package's, ``router_splits``); a launch runs that program through this package's binding (:func:`_launch_cached_atomic_k2`;
    ``views``: an entry point's own launch views).
    ``enable_pdl`` (default: the device supports it) launches the split and the program as programmatic dependent
    launches."""
    if op == "launch":
        return _launch_cached_atomic_k2(
            A, B, out, splits=splits, enable_pdl=enable_pdl, views=views
        )
    _, arch, sm_count = _device_and_arch(A, B, out)
    plan, _ = plan_atomic_router(A, B, out, arch=arch, sm_count=sm_count, splits=splits)
    if plan is None:
        det = deterministic_backend()
        if splits is None:
            splits = _k2_det_splits(A, B, arch, sm_count)
        if op == "plan":
            return det.cake_dense_gemm_det_train_router_plan(A, B, out, splits=splits)
        return det.cake_dense_gemm_det_train_router_prepare(
            A, B, out, splits=splits, **_pdl_kw(enable_pdl)
        )
    if op == "plan":
        return plan
    return _prepare_atomic_k2(plan, A, B, out, arch, enable_pdl)


# --- convenience-path state of this package's programs -----------------------------------------------------------
#
# The deterministic package's design (one least-recently-used cache per kind, keyed by device, stream and view
# geometry; one growable workspace arena per (device, raw stream) shared by every cached entry), with caches and
# arenas of their own: a buffer of this package is never a buffer of the deterministic package.  The cache
# bound is the deterministic package's ``CONVENIENCE_CACHE_SIZE``, read on every insertion, so one setting bounds
# both packages.  A cached entry holds its plan and the resolved program entry, never a caller's tensor: the views
# are rebuilt from the caller's tensors on every call.  The launches of one stream share its arena; launches on
# different streams never do.  The K1 tickets are zero when allocated or grown (``torch.zeros`` on the arena's
# stream) and zero again after every complete launch, so a launch needs no memset; the K1 split slots are zeroed when
# allocated or grown (the last split reads the never-written padding columns of a partial tile and discards them) and
# need no memset per launch either.  A K1 call of this package that runs the deterministic program -- the
# forward, the input gradient, a layout routed ``"det"`` (cache ``_K1_DET_CACHE``, keyed by the entry point's own
# tensors), a weight-gradient row the planner keeps on it (cached as such in ``_K1_ATOMIC_CACHE``) -- runs it through a
# binding of this package (:func:`_plan_det_k1`): the deterministic package's plan and program, so its bits, with this
# package's arena for its stream-K slabs and slice counters (the deterministic package's zero contract: every complete
# launch leaves the counters it uses zero); a batched forward the deterministic package runs on its swapped
# small-token program (with its PDL flag and ``weights_static``) stays on that package's call.  A K1 entry binds its
# argument list once (:class:`_K1Binding`: the plan's values in the program's order with the PDL flag -- a key per
# flag --, the constant dummies converted once, the workspace converted once per arena growth),
# so a call rebuilds only its views (each the caller's tensor, its ``unsqueeze(0)`` or one strided view, checked
# against the planner's views when bound; passed as torch tensors, each converted in the FFI call) and enters its
# thread's stream context of the (device, stream), made once; the projection entry points go from their key straight
# to the bound launch when it is cached (:func:`_k1_cached_launch`).  A forward or input-gradient row of
# ``K1_FORWARD_PROGRAM_RULES_BY_SM`` / ``K1_DGRAD_PROGRAM_RULES_BY_SM`` binds this package's program the same way
# (cache ``_K1_PROGRAM_CACHE``, under the entry point's key; :func:`_k1_program_binding`): a repeat call skips the
# table lookup, the argument values and their order.  A K2 call of this package that runs the
# deterministic program -- a row the planner keeps on it, a call without a split count whose row has no verdict
# (:func:`_k2_row_without_verdict`, kept per device and view geometry until a router table changes,
# :func:`_router_memo`; not planned here), a forward without a split count while no router table gives the forward a
# verdict, the input gradient outside its SIMT windows -- runs it through a binding of this package
# (:func:`_launch_cached_splitk_k2`, cache ``_K2_SPLITK_CACHE``): the deterministic package's plan and programs, and
# the split-K partials reduced in the order of that package's ``torch.sum`` (by this package's partial sum where it
# serves the launch, :func:`_partials_sum_serves`), so its bits, with this package's arena for the stack and the
# partials.  A K2 entry binds its argument lists once
# (:class:`_K2Binding`: the plan's values in the program's order, the arena's BF16x3 stack -- and the deterministic
# program's partials -- carved and converted once per arena growth), so a call passes only the caller's views (an
# entry point's own tensors when they are the launch views, :func:`_launch_views`; torch tensors, each converted in
# its FFI call) and fills ``out`` with the bound geometry; the K2 launches of an arena enter its stream context (made
# once) under its lock.  The router entry points' calls (``cake_dense_gemm_nondet_train_router_forward`` / ``cake_dense_gemm_nondet_train_router_dgrad`` / ``cake_dense_gemm_nondet_train_router_wgrad``)
# are cached once more by the entry point's own tensors (``_K2_ENTRY_CACHE``, :func:`_k2_entry`): each holds the
# cached launch of ``_K2_SPLITK_CACHE`` / ``_K2_ATOMIC_CACHE`` that the entry point's path takes for the key and the
# router-table memo and route it was routed under, so a call goes from its key to the bound launch.  After a launch
# that did not complete, :func:`cake_dense_gemm_nondet_train_convenience_cache_clear` drops the arenas (fresh tickets on the next call).
#
# CUDA graphs.  A cached launch captured into a CUDA graph replays the workspace pointers it was bound with, so no
# buffer of an arena is freed before :func:`cake_dense_gemm_nondet_train_convenience_cache_clear` (or :func:`use_deterministic_backend`) drops the
# arenas: a growth retires the buffer it replaces into the arena (:class:`_AtomicArena`) instead of releasing it, and
# an entry evicted from the caches leaves its arena in ``_ATOMIC_ARENAS``.  A captured graph therefore stays valid
# until that call, which invalidates it.  The calls this path hands to the deterministic package's convenience path
# keep that package's contract.

#: workspace kind -> (dtype, zeroed when allocated)
_ATOMIC_ARENA_KINDS: dict[str, tuple[torch.dtype, bool]] = {
    # K1 split partial slots, zeroed: the padding columns of a partial tile are read (and discarded), never written
    "k1_slots": (torch.float32, True),
    # K1 per-(tile, warp slice, lane) arrival tickets, zero between launches
    "k1_tickets": (torch.uint32, True),
    # K2 BF16x3 stack of the split operand ``[3, *split_src.shape]``
    "k2_stack": (torch.bfloat16, False),
    # K2 fp32 scratch of the torch-chain split
    "k2_split_scratch": (torch.float32, False),
    # the split-K partials ``[splits, rows, cols]`` of the deterministic K2 program (:func:`_launch_cached_splitk_k2`),
    # written by the program and reduced right after it on the same stream (:func:`_launch_splitk_k2_binding`)
    "k2_splitk_partials": (torch.float32, False),
    # the stream-K partial slabs of the deterministic K1 program (:func:`_plan_det_k1`; no state between launches)
    "k1_det_ws": (torch.float32, False),
    # its stream-K slice counters, zeroed: every complete launch leaves the counters it uses zero
    "k1_det_counters": (torch.uint32, True),
    # the K-order K2 program's workspace slots (every slot it reads is written by the same launch first)
    "k2_ksum_slots": (torch.float32, False),
    # its per-tile tickets, zeroed: every complete launch leaves them zero
    "k2_ksum_tickets": (torch.uint32, True),
}
_ATOMIC_ARENA_MIN_ELEMS = 16


@dataclass
class _AtomicArena:
    """The workspace of every cached launch of this package on one (device index, raw stream): one buffer per kind
    of ``_ATOMIC_ARENA_KINDS``, grown and never shrunk, as the deterministic package's ``_Arena`` (a zeroed kind is
    allocated with ``torch.zeros`` on the current stream).

    Unlike that arena, a growth retires the buffer it replaces (``retired``) instead of releasing it: a launch captured
    into a CUDA graph replays the pointer it was bound with, and a launch in flight keeps using it, so every buffer of
    the arena stays allocated until :func:`cake_dense_gemm_nondet_train_convenience_cache_clear` drops the arena.  A retired buffer keeps its zero
    contract: the launches bound to it leave its tickets and counters zero.  A growth at least doubles the buffer, so
    the buffers a kind retires hold fewer bytes than its live one, and a kind grows at most
    ``ceil(log2(largest requirement / _ATOMIC_ARENA_MIN_ELEMS))`` times."""

    device: torch.device
    key: tuple[int, int]
    buffers: dict[str, torch.Tensor] = field(default_factory=dict, repr=False)
    # the buffers growths replaced, kept for the launches (CUDA graphs) bound to them
    retired: list[torch.Tensor] = field(default_factory=list, repr=False)
    version: int = 0
    lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )
    # the tvm-ffi stream context of the (device, stream), made by the first K2 launch and entered by every K2 launch
    # under ``lock`` (:func:`_k2_stream_context`)
    stream_context: Any = field(default=None, repr=False, compare=False)

    def require(self, kind: str, numel: int) -> torch.Tensor:
        """The buffer of ``kind`` with at least ``numel`` elements, grown to at least twice the present one's when
        that is smaller (called when an entry is planned, under ``_ATOMIC_PLAN_LOCK``; the replaced buffer is
        retired); monotone."""
        buf = self.buffers.get(kind)
        numel = max(int(numel), _ATOMIC_ARENA_MIN_ELEMS)
        if buf is None or buf.numel() < numel:
            if buf is not None:
                numel = max(numel, 2 * buf.numel())
            dtype, zeroed = _ATOMIC_ARENA_KINDS[kind]
            new = (torch.zeros if zeroed else torch.empty)(
                numel, device=self.device, dtype=dtype
            )
            buf = self.buffers.get(kind)
            if buf is None or buf.numel() < new.numel():
                if buf is not None:
                    self.retired.append(buf)
                buf = self.buffers[kind] = new
            self.version += 1
        return buf

    def nbytes(self) -> int:
        """Bytes the arena holds (every kind, its live and its retired buffers)."""
        return sum(
            int(b.numel()) * b.element_size()
            for b in (*self.buffers.values(), *self.retired)
        )


_ATOMIC_ARENAS: dict[tuple[int, int], _AtomicArena] = {}
_ATOMIC_PLAN_LOCK = threading.Lock()
_K1_ATOMIC_CACHE: "OrderedDict[tuple, Any]" = OrderedDict()
_K2_ATOMIC_CACHE: "OrderedDict[tuple, Any]" = OrderedDict()


def _atomic_arena_for(index: int, stream: int, device: torch.device) -> _AtomicArena:
    """The arena of (device ``index``, raw ``stream``), created on first use (under ``_ATOMIC_PLAN_LOCK``)."""
    key = (int(index), int(stream))
    arena = _ATOMIC_ARENAS.get(key)
    if arena is None:
        arena = _ATOMIC_ARENAS[key] = _AtomicArena(device, key)
    return arena


class _Arg:
    """A per-call argument of a cached launch's argument template (a view of the caller's tensors, or the arena's
    BF16x3 stack), replaced by its value when the launch is bound."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"_Arg({self.name!r})"


#: The per-call arguments of a cached K2 launch, in the order of the views a call converts: the streamed operand's
#: span, ``out`` as the store pointer and as the reduce-add map, the fused split's source.
_K2_CALL_ARGS = (_Arg("A"), _Arg("out32"), _Arg("D"), _Arg("src"))
_K2_PARTS_ARG = _Arg("parts")
#: The split-K partials of the deterministic K2 program, bound with the stack (:meth:`_K2Binding.carve`).
_K2_PARTIALS_ARG = _Arg("partials")
#: The K-order K2 program's workspace slots and tickets, bound with the stack (:meth:`_K2Binding.carve`).
_K2_KSUM_WS_ARG = _Arg("ksum_ws")
_K2_KSUM_TICKETS_ARG = _Arg("ksum_tickets")

#: The template of this package's partial sum: ``out[r, c] = sum_s partials[s, r, c]`` over the deterministic K2
#: program's split-K partials ``[splits, rows, cols]`` (contiguous) into an FP32 ``out [rows, cols]`` with unit inner
#: stride and row stride ``ldo``, in one fixed order: four FP32 accumulators start at ``+0``, partial ``s`` is added to
#: accumulator ``s % 4`` in ascending ``s``, and the result is ``((a0 + a1) + a2) + a3`` (each add one
#: round-to-nearest FP32 add; for at most four splits ``((w0 + w1) + w2) + w3`` and then ``+ 0.0``).  This is the order
#: of the deterministic package's ``torch.sum(partials, 0, out=out)`` on these launches: PyTorch's CUDA reduction over
#: the outer dimension of a contiguous tensor gives each thread whole outputs and adds their values into four
#: accumulators combined left to right, and spreads one output's values over several threads only from sixteen values
#: per thread on.  Both agree bit for bit -- signed zeros, NaN, infinities, subnormals and overflow included -- and the
#: sum is one more launch in the bound launch chain instead of a PyTorch operator.  The tests check the program against
#: ``torch.sum`` and against the written order on every split count it serves.
PARTIALS_SUM_TEMPLATE = cake_jit.TEMPLATE_PREFIXES[5]
#: The split counts the partial sum serves (its kernel requires two to eight).  The deterministic router planner issues
#: four for the forward and six for the weight gradient, except on the rows of its ``ROUTER_SPLIT_RULES``: three
#: (served) and nine (``torch.sum``) for the weight gradient.
PARTIALS_SUM_SPLITS = range(2, 9)
#: Threads per CTA of the partial sum and the ``out`` columns each thread sums (a flat grid over the rows' column
#: quads).
_PARTIALS_SUM_THREADS, _PARTIALS_SUM_COLUMNS = 256, 4


def _partials_sum_serves(partials_shape: tuple[int, ...], out: torch.Tensor) -> bool:
    """Whether this package's partial sum reduces the deterministic K2 program's split-K partials ``partials_shape``
    (``[splits, rows, cols]``) into ``out``: a split count of :data:`PARTIALS_SUM_SPLITS`, two outputs or more (a
    single output is a full reduction, which PyTorch orders otherwise) and ``out`` an FP32 ``[rows, cols]`` view with
    unit inner stride and a row stride of ``cols`` or more.  Any other launch keeps the deterministic package's
    ``torch.sum``: the choice is made when the launch is planned, and a failing launch raises its error."""
    splits, rows, cols = (int(d) for d in partials_shape)
    return (
        splits in PARTIALS_SUM_SPLITS
        and rows * cols >= 2
        and out.dtype == torch.float32
        and tuple(out.shape) == (rows, cols)
        and out.stride(1) == 1
        and out.stride(0) >= cols
    )


def _partials_sum_binding(
    arch: str, partials_shape: tuple[int, ...], out: torch.Tensor, pdl: int
) -> tuple[Callable[..., Any], list]:
    """``(entry, argument template)`` of the partial sum of a deterministic K2 launch whose partials are
    ``partials_shape`` (:func:`_partials_sum_serves`): this package's program of :data:`PARTIALS_SUM_TEMPLATE` on
    ``arch`` with the PDL flag ``pdl`` (it waits for the program that writes the partials before its first load), the
    partials as :data:`_K2_PARTIALS_ARG` (bound with the stack) and ``out`` per call (the slot of ``out32``)."""
    splits, rows, cols = (int(d) for d in partials_shape)
    module_name = cake_jit.select_module(arch, PARTIALS_SUM_TEMPLATE)
    quads = -(-cols // _PARTIALS_SUM_COLUMNS)
    grid = (-(-(rows * quads) // _PARTIALS_SUM_THREADS), 1, 1)
    values = dict(
        ws=_K2_PARTIALS_ARG,
        out=_K2_CALL_ARGS[1],
        splits=splits,
        rows=rows,
        cols=cols,
        ldo=int(out.stride(0)),
        enable_pdl=int(pdl),
    )
    entry, arg_plan = _atomic_entry(module_name, grid, arch)
    return entry, _order_atomic_arguments(module_name, arg_plan, values, grid)


def _arg_slots(template: list) -> tuple[tuple[int, int], ...]:
    """``(position, index into _K2_CALL_ARGS)`` of every per-call argument of a bound template."""
    return tuple(
        (i, _K2_CALL_ARGS.index(a)) for i, a in enumerate(template) if type(a) is _Arg
    )


@dataclass
class _K2Binding:
    """The bound arguments of a cached K2 launch: the kernel's and the fused split's argument lists in the order of
    their argument plans (fixed by the plan and the view geometry of the cache key), with the arena's BF16x3 stack --
    and, for the deterministic program with more than one split, its partials ``[splits, rows, cols]``
    (``partials_shape``, carved as ``workspace``), and for the K-order program its slots and tickets
    (``ksum_workspace``) -- carved and converted once per arena growth (``version``) and the
    caller's views left as ``slots``; the geometry of the zero fill of ``out`` (rows, columns, row pitch in bytes;
    ``None`` without part tiles).  Without ``stack`` (the SIMT program, which reads the fp32 operands in place) the
    kernel's list is bound with no stack and no split.  ``sum_template`` is the argument list of the partial sum that
    reduces the partials into ``out`` (:func:`_partials_sum_binding`; ``None``: ``torch.sum`` does), bound with the
    partials."""

    kernel_template: list
    split_template: Optional[list]
    split_shape: tuple[int, ...]
    fill: Optional[tuple[int, int, int]]
    out_contiguous: bool
    partials_shape: Optional[tuple[int, ...]] = None
    stack: bool = True
    version: int = -1
    parts: Optional[torch.Tensor] = None
    resid: Optional[torch.Tensor] = None
    workspace: Optional[torch.Tensor] = None
    kernel_args: list = field(default_factory=list)
    kernel_slots: tuple[tuple[int, int], ...] = ()
    split_args: list = field(default_factory=list)
    split_slots: tuple[tuple[int, int], ...] = ()
    sum_template: Optional[list] = None
    sum_args: list = field(default_factory=list)
    sum_slots: tuple[tuple[int, int], ...] = ()
    ksum_workspace: Optional[tuple[int, int]] = None

    def carve(self, arena: "_AtomicArena", torch_chain: bool) -> None:
        """Carve the BF16x3 stack (and the torch chain's fp32 scratch) out of ``arena`` and bind the stack into the
        argument lists (called under the arena's lock when the arena grew since the last carve)."""
        # the version is read before carving: a growth that interleaves leaves the binding stale
        version = arena.version
        if not self.stack:
            self.kernel_args = list(self.kernel_template)
            self.kernel_slots = _arg_slots(self.kernel_args)
            self.version = version
            return
        n = math.prod(self.split_shape)
        self.parts = arena.buffers["k2_stack"][: 3 * n].view(3, *self.split_shape)
        self.resid = (
            arena.buffers["k2_split_scratch"][:n].view(self.split_shape)
            if torch_chain
            else None
        )
        parts = tvm_ffi.from_dlpack(self.parts, require_contiguous=False)
        args = [parts if a is _K2_PARTS_ARG else a for a in self.kernel_template]
        if self.partials_shape is not None:
            n_partials = math.prod(self.partials_shape)
            self.workspace = arena.buffers["k2_splitk_partials"][:n_partials].view(
                self.partials_shape
            )
            partials = tvm_ffi.from_dlpack(self.workspace, require_contiguous=False)
            args = [partials if a is _K2_PARTIALS_ARG else a for a in args]
            if self.sum_template is not None:
                self.sum_args = [
                    partials if a is _K2_PARTIALS_ARG else a for a in self.sum_template
                ]
                self.sum_slots = _arg_slots(self.sum_args)
        if self.ksum_workspace is not None:
            slots, words = self.ksum_workspace
            ws = tvm_ffi.from_dlpack(
                arena.buffers["k2_ksum_slots"][:slots], require_contiguous=False
            )
            tickets = tvm_ffi.from_dlpack(
                arena.buffers["k2_ksum_tickets"][:words], require_contiguous=False
            )
            args = [
                ws
                if a is _K2_KSUM_WS_ARG
                else tickets
                if a is _K2_KSUM_TICKETS_ARG
                else a
                for a in args
            ]
        self.kernel_args = args
        self.kernel_slots = _arg_slots(self.kernel_args)
        if self.split_template is not None:
            self.split_args = [
                parts if a is _K2_PARTS_ARG else a for a in self.split_template
            ]
            self.split_slots = _arg_slots(self.split_args)
        self.version = version

    def arguments(self, views: tuple) -> tuple[list, Optional[list]]:
        """The kernel's and the fused split's argument lists of one call: the bound lists with the call's converted
        views (in the order of :data:`_K2_CALL_ARGS`) in place (``None``: no fused split)."""
        kernel = self.kernel_args.copy()
        for i, k in self.kernel_slots:
            kernel[i] = views[k]
        if self.split_template is None:
            return kernel, None
        split = self.split_args.copy()
        for i, k in self.split_slots:
            split[i] = views[k]
        return kernel, split

    def sum_arguments(self, views: tuple) -> list:
        """The partial sum's argument list of one call: the bound list with the call's ``out`` (the ``out32`` view of
        ``views``) in place."""
        partial_sum = self.sum_args.copy()
        for i, k in self.sum_slots:
            partial_sum[i] = views[k]
        return partial_sum


#: How a per-call view of a bound K1 launch is rebuilt from the call's tensor (:class:`_K1Binding`): the tensor
#: itself, its ``unsqueeze(0)``, or its ``as_strided`` view of the bound shape and strides.
_K1_VIEW_SELF, _K1_VIEW_UNSQUEEZE, _K1_VIEW_STRIDED = 0, 1, 2


@dataclass
class _K1Binding:
    """The bound arguments of a cached K1 launch (a program of this package or of the deterministic package): the
    program's argument list in the order of its argument plan (fixed by the plan and the view geometry of the cache
    key), with the constant tensors among them (the deterministic package's dummies) converted when bound and the
    arena's workspace (``workspace``: ``(position, arena kind)``) converted once per arena growth (``version``).
    ``views`` are the per-call views ``(position, source, recipe, shape, strides)``: the view is the call's tensor
    ``source`` (an index into the call's tensors) as is, its ``unsqueeze(0)`` or its ``as_strided(shape, strides)``
    (:data:`_K1_VIEW_SELF` / ...), each checked against the planner's view when bound (:func:`_k1_view_recipe`)."""

    template: list
    views: tuple
    workspace: tuple = ()
    version: int = -1
    args: list = field(default_factory=list)

    def carve(self, arena: Optional["_AtomicArena"]) -> None:
        """Bind the arena's workspace buffers into the argument list (called when bound without an arena, and
        under the arena's lock when the arena grew since the last carve)."""
        # the version is read before carving: a growth that interleaves leaves the binding stale
        version = -1 if arena is None else arena.version
        args = list(self.template)
        for i, kind in self.workspace:
            assert arena is not None
            args[i] = tvm_ffi.from_dlpack(arena.buffers[kind], require_contiguous=False)
        self.args = args
        self.version = version


@dataclass
class _CachedAtomic:
    """A planned launch of the convenience path: the plan (``None``: the row runs the deterministic program), the
    program entry and argument plan, the arena of its (device, stream) and, for K2, the fused split's module with its
    entry and argument plan, and the bound arguments (:class:`_K2Binding`; K1: :class:`_K1Binding`, of the
    deterministic program -- its plan ``det_plan`` -- when ``plan`` is ``None``).  ``memo``: the router-table memo
    (:func:`_router_memo`) a K2 launch without a caller's split count was planned under, ``None`` with one (its plan
    reads no router table); a call under another memo plans the key again."""

    plan: Any
    arch: str = ""
    module_name: str = ""
    entry: Optional[Callable[..., Any]] = field(default=None, repr=False)
    arg_plan: Optional[list] = field(default=None, repr=False)
    arena: Optional[_AtomicArena] = field(default=None, repr=False)
    split_module: Optional[str] = None
    split_entry: Optional[Callable[..., Any]] = field(default=None, repr=False)
    split_arg_plan: Optional[list] = field(default=None, repr=False)
    k2: Optional[_K2Binding] = field(default=None, repr=False)
    k1: Optional[_K1Binding] = field(default=None, repr=False)
    det_plan: Any = field(default=None, repr=False)
    memo: Any = field(default=None, repr=False)
    # the plan's launch can vary run to run (``not plan.reproducible``): PyTorch's alert before every launch
    order_varying: bool = False
    # K2 without a plan and without a caller's split count: the deterministic program's split count of the row in the
    # router tables (:func:`router_det_splits`), ``None``: the deterministic package's own
    det_splits: Optional[int] = None


#: An entry of the convenience caches (:data:`_K1_ATOMIC_CACHE`, :data:`_K1_DET_CACHE`, :data:`_K1_PROGRAM_CACHE`,
#: :data:`_K2_ATOMIC_CACHE`, :data:`_K2_SPLITK_CACHE`, :data:`_K2_ENTRY_CACHE`).
_CacheEntry = TypeVar(
    "_CacheEntry",
    "_CachedAtomic",
    "_CachedDetK1",
    "_CachedProgramK1",
    "_CachedSplitK2",
    "_CachedK2Entry",
)


def _cache_insert(
    cache: "OrderedDict[tuple, Any]", key: tuple, entry: _CacheEntry
) -> _CacheEntry:
    """Insert ``entry`` under ``key`` (the deterministic package's ``_cache_put`` with its ``CONVENIENCE_CACHE_SIZE``
    bound).  An evicted entry leaves its arena in ``_ATOMIC_ARENAS``: a CUDA graph may still replay its launch."""
    det = deterministic_backend()
    return det._cache_put(cache, key, entry, det.CONVENIENCE_CACHE_SIZE)


def _k1_key(
    index: int,
    stream: int,
    flags: tuple,
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
) -> tuple:
    """The convenience key of a K1 call of the tensors ``A``, ``B``, ``out`` (the deterministic package's
    ``convenience_key``: the device, the raw stream, the entry point's flags and the geometry of every view -- shape,
    strides, dtype, storage offset and 16-byte alignment residue, never an address), built inline."""
    return (
        index,
        stream,
        flags,
        (A.shape, A.stride(), A.dtype, A.storage_offset(), A.data_ptr() & 15),
        (B.shape, B.stride(), B.dtype, B.storage_offset(), B.data_ptr() & 15),
        (out.shape, out.stride(), out.dtype, out.storage_offset(), out.data_ptr() & 15),
    )


def _k1_index(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor) -> int:
    """The CUDA device index of a K1 call of the tensors ``A``, ``B``, ``out`` (``out``'s), or ``-1`` when they are
    not all on that CUDA device."""
    index = out.get_device()
    if index < 0 or A.get_device() != index or B.get_device() != index:
        return -1
    return index


def _k1_view_recipe(
    view: torch.Tensor, sources: tuple, *, pointer: bool
) -> Optional[tuple[int, int, tuple[int, ...], tuple[int, ...]]]:
    """``(source, recipe, shape, strides)`` rebuilding the launch view ``view`` from one of the call's tensors
    ``sources`` (:class:`_K1Binding`), checked against ``view`` (its data pointer, shape, strides and dtype), or
    ``None`` when none does.  A ``pointer`` argument (a ``buffer`` of the argument plan: the program reads its data
    pointer, and the binding checks its dtype, device and contiguity) takes a contiguous source of the same data
    pointer and dtype as is.  The source is the one call tensor that starts at the view's data pointer: with several
    (tensors of the call that alias, which the key does not record) there is no recipe."""
    det = deterministic_backend()
    ptr, shape, strides = view.data_ptr(), tuple(view.shape), tuple(view.stride())
    found = [s for s, t in enumerate(sources) if t.data_ptr() == ptr]
    if len(found) != 1:
        return None
    s = found[0]
    t = sources[s]
    if t.dtype != view.dtype:
        return None
    if (pointer and t.is_contiguous()) or det._same_view(t, view):
        return s, _K1_VIEW_SELF, shape, strides
    if t.dim() == 2 and det._same_view(t.unsqueeze(0), view):
        return s, _K1_VIEW_UNSQUEEZE, shape, strides
    try:
        strided = t.as_strided(shape, strides)
    except RuntimeError:  # the view leaves this tensor's storage
        return None
    if det._same_view(strided, view):
        return s, _K1_VIEW_STRIDED, shape, strides
    return None


def _k1_binding(
    template: Any, views: dict[str, tuple[torch.Tensor, bool]], sources: tuple
) -> Optional[_K1Binding]:
    """The :class:`_K1Binding` of a program's argument list ``template`` (ordered by its argument plan) whose
    per-call views and workspace are :class:`_Arg` markers: ``views`` maps a view marker's name to the planner's
    view and whether the program takes it as a pointer, the markers ``ws`` / ``counters`` name the arena kinds of
    :data:`_K1_WORKSPACE_KINDS`; the other tensors are converted here.  ``None`` when a planner's view is not rebuilt
    from the call's tensors ``sources`` (:func:`_k1_view_recipe`)."""
    args, recipes, workspace = list(template), [], []
    for i, a in enumerate(args):
        if isinstance(a, torch.Tensor):
            args[i] = tvm_ffi.from_dlpack(a, require_contiguous=False)
        elif type(a) is _Arg:
            if a.name in _K1_WORKSPACE_KINDS:
                workspace.append((i, _K1_WORKSPACE_KINDS[a.name]))
                continue
            view, pointer = views[a.name]
            recipe = _k1_view_recipe(view, sources, pointer=pointer)
            if recipe is None:
                return None
            recipes.append((i, *recipe))
    bound = _K1Binding(template=args, views=tuple(recipes), workspace=tuple(workspace))
    if not workspace:
        bound.carve(None)
    return bound


def _launch_k1_binding(
    det: ModuleType,
    fn: Callable[..., Any],
    bound: _K1Binding,
    arena: Optional[_AtomicArena],
    sources: tuple,
    index: int,
    stream: int,
) -> None:
    """Launch a bound K1 program on the raw ``stream`` of device ``index`` (``det``: the deterministic package's
    ``cake_backend``, for its stream context): the bound argument list with the views of the call's tensors
    ``sources`` in place (rebuilt here; torch tensors, each converted in the FFI call), after binding the arena's
    workspace again if the arena grew, in the thread's stream context of the (device, stream)
    (:func:`_k1_stream_context`)."""
    if arena is not None and bound.version != arena.version:
        with arena.lock:
            if bound.version != arena.version:
                bound.carve(arena)
    args = bound.args.copy()
    for i, s, recipe, shape, strides in bound.views:
        t = sources[s]
        if recipe == _K1_VIEW_UNSQUEEZE:
            t = t.unsqueeze(0)
        elif recipe == _K1_VIEW_STRIDED:
            t = t.as_strided(shape, strides)
        args[i] = t
    with _k1_stream_context(det, index, stream):
        fn(*args)


#: The tvm-ffi stream context of every (thread, device index, raw stream) a K1 launch of this package ran on, made by
#: the first launch: a context keeps the stream it replaced while it is entered, so a thread enters only its own.
_K1_STREAM_CONTEXTS: dict[tuple[int, int, int], Any] = {}


def _k1_stream_context(det: ModuleType, index: int, stream: int) -> Any:
    """The calling thread's tvm-ffi stream context of (device ``index``, raw ``stream``)
    (:data:`_K1_STREAM_CONTEXTS`)."""
    key = (threading.get_ident(), index, stream)
    context = _K1_STREAM_CONTEXTS.get(key)
    if context is None:
        context = _K1_STREAM_CONTEXTS[key] = det._ffi_stream_context(index, stream)
    return context


def _k1_cached_launch(
    cache: "OrderedDict[tuple, Any]",
    head: tuple,
    tail: tuple,
    sources: tuple,
    enable_pdl: Optional[bool],
    programs: Optional["OrderedDict[tuple, Any]"] = None,
) -> bool:
    """A projection entry point's call when its key is cached and bound: the key the entry
    point's own path builds for the tensors ``sources`` (``(A, B, out)`` of that path) in ``cache``, then (a miss
    there) in ``programs`` (:data:`_K1_PROGRAM_CACHE`, the same key: a row of a program table) --
    :data:`_K1_DET_CACHE` (:func:`_det_k1`, flags ``(*head, pdl, *tail)`` = ``(flags, pdl, weights_static)``) or
    :data:`_K1_ATOMIC_CACHE` (:func:`_launch_cached_atomic_k1`) -- and its bound launch, without the path's call
    layers.  ``False`` (nothing launched) when the tensors are not on one CUDA device or the key is not cached and
    bound: the entry point's own path then plans, launches or raises."""
    index = _k1_index(*sources)
    if index < 0:
        return False
    det = deterministic_backend()
    stream = det._current_raw_stream(index)
    pdl = int(det.pdl_enabled(enable_pdl, index))
    key = _k1_key(index, stream, (*head, pdl, *tail), *sources)
    entry = det._cache_get(cache, key)
    if entry is None and programs is not None:
        entry = det._cache_get(programs, key)
    if entry is None or entry.k1 is None or entry.entry is None:
        return False
    if entry.order_varying:
        _alert_order_varying(_k1_launch_text(entry.plan))
    _launch_k1_binding(det, entry.entry, entry.k1, entry.arena, sources, index, stream)
    return True


#: The arena kinds of the workspace markers of a bound K1 launch: this package's split slots and tickets, and the
#: deterministic program's stream-K slabs and slice counters (:data:`_ATOMIC_ARENA_KINDS`).
_K1_WORKSPACE_KINDS = {
    "ws": "k1_slots",
    "counters": "k1_tickets",
    "det_ws": "k1_det_ws",
    "det_counters": "k1_det_counters",
}


def _bind_atomic_k1(
    plan: CakeDenseGemmNondetProjectionPlan,
    module_name: str,
    arg_plan: list,
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    arena: Optional[_AtomicArena],
    pdl: int,
) -> Optional[_K1Binding]:
    """The :class:`_K1Binding` of this package's planned K1 program for the call ``(A, B, out)``: its values in the
    program's order (the split slots and tickets from the arena with more than one split, else the deterministic
    package's dummies, and the PDL flag ``pdl``, 0 or 1), the operand views and ``out``'s pointer rebuilt per
    call."""
    a_desc, b_desc, out3 = _k1_launch_views(A, B, out, plan)
    ldo, out_l = _k1_out_strides(out3)
    if arena is not None:
        ws, counters = _Arg("ws"), _Arg("counters")
    else:
        ws, counters = _k1_dummies(out.device)
    values = _k1_values(
        plan,
        A=_Arg("A"),
        B=_Arg("B"),
        out=_Arg("out"),
        ws=ws,
        counters=counters,
        ldo=ldo,
        out_l=out_l,
        enable_pdl=pdl,
    )
    template = _order_atomic_arguments(module_name, arg_plan, values, plan.grid)
    views = dict(
        A=(a_desc, False),
        B=(b_desc, False),
        out=(deterministic_backend().flat_alias(out3), True),
    )
    return _k1_binding(template, views, (A, B, out))


#: The cached bindings of the deterministic K1 program on this package's convenience path (keys of :func:`_k1_key`
#: over the entry point's tensors).
_K1_DET_CACHE: "OrderedDict[tuple, Any]" = OrderedDict()
#: The cached bindings of this package's programs of :data:`K1_FORWARD_PROGRAM_RULES_BY_SM` and
#: :data:`K1_DGRAD_PROGRAM_RULES_BY_SM` on the convenience path, under the keys of :data:`_K1_DET_CACHE` (the entry
#: point's flags, the PDL flag and the geometry of its tensors): a repeat call goes from its key to the bound launch,
#: without the table lookup, the argument values and their order (:func:`_k1_cached_launch`).
_K1_PROGRAM_CACHE: "OrderedDict[tuple, Any]" = OrderedDict()
#: The leading and trailing key flags of :func:`_det_k1` for the projection forward and input gradient (the PDL flag
#: goes between them), as :func:`_k1_cached_launch` reads them.
_K1_FWD_HEAD = (("fwd",),)
_K1_DGRAD_HEAD = (("dgrad",),)
_K1_NOT_STATIC = (False,)


@dataclass
class _CachedProgramK1:
    """A bound launch of this package's forward / input-gradient program ``module_name`` (:data:`_K1_PROGRAM_CACHE`):
    its entry and bound arguments; no workspace, so no arena; a fixed-order program, so no alert."""

    module_name: str
    entry: Optional[Callable[..., Any]] = field(default=None, repr=False)
    k1: Optional[_K1Binding] = field(default=None, repr=False)
    arena: None = None
    order_varying: bool = False


@dataclass
class _CachedDetK1:
    """A planned launch of the deterministic K1 program on this package's convenience path: the deterministic
    package's plan, its program's entry, the arena of the (device, stream) and the bound arguments (``k1``;
    ``None``: the call goes to the deterministic package's convenience path)."""

    plan: Any
    entry: Optional[Callable[..., Any]] = field(default=None, repr=False)
    arena: Optional[_AtomicArena] = field(default=None, repr=False)
    k1: Optional[_K1Binding] = field(default=None, repr=False)
    # a deterministic program is fixed-order: never alerts (read by :func:`_k1_cached_launch`)
    order_varying: bool = False
    # this package's program bound on the plan in place of the plan's own (K1_DGRAD_NVJET_RULES_BY_SM), else None
    module_name: Optional[str] = None


def _plan_det_k1(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    transposed_out: bool,
    knobs: tuple,
    sources: tuple,
    index: int,
    stream: int,
    pdl: int,
    weights_static: bool = False,
    module: Optional[str] = None,
) -> _CachedDetK1:
    """Plan the deterministic K1 program for the deterministic package's convenience call ``(A, B, out,
    transposed_out, **knobs)`` (under ``_ATOMIC_PLAN_LOCK``): that package's planner (its errors), its program bound
    in its argument order with the PDL flag ``pdl`` (0 or 1; :func:`_k1_binding`; the per-call views rebuilt from the
    call's tensors ``sources``, which hold the data pointers of ``A`` / ``B`` / ``out``), the stream-K slabs and slice
    counters in this package's arena.  With ``module`` (this package's program of :data:`K1_DGRAD_NVJET_RULES_BY_SM`)
    a plan of :data:`K1_DGRAD_NVJET_DET_PROGRAM` binds that program in place of the plan's own, in its own argument
    order (the deterministic package need not register its program on the part: on ``sm_107a`` it does not); a plan
    of another program binds its own.  A planner's view the call's tensors do not rebuild leaves the
    entry unbound (the deterministic package's convenience path), as does a call without knobs that the
    deterministic package runs on its swapped small-token forward (``_swap_ab_launch`` with the call's
    ``weights_static`` under PDL: its program, not the planner's)."""
    det = deterministic_backend()
    if not knobs:
        # the deterministic call's swapped small-token forward: its program, not the planner's
        static = bool(weights_static) and bool(pdl)
        if det._swap_ab_launch(A, B, out, transposed_out, static) is not None:
            return _CachedDetK1(plan=None)
    if module is None:
        plan, a_desc, b_desc, O3, arch, module_name = det._k1_plan(
            A, B, out, transposed_out=transposed_out, knobs=dict(knobs)
        )
    else:
        # the deterministic planner's plan without its registry lookup (``det._k1_plan`` less ``select_module``)
        device = det._same_cuda_device(A, B, out)
        facts = det.device_facts(device)
        arch = det.require_arch(device)
        plan, a_desc, b_desc, O3 = det.cake_dense_gemm_det_train_projection_plan(
            A,
            B,
            out,
            sm_count=facts.sm_count,
            l2_bytes=facts.l2_bytes,
            transposed_out=transposed_out,
            arch=arch,
            **dict(knobs),
        )
        module_name = None
    ldo, out_l = det._out_strides(O3)
    values = det._k1_values(
        plan,
        out.device,
        A=_Arg("A"),
        B=_Arg("B"),
        out3=_Arg("out3"),
        o_alias=_Arg("o_alias"),
        ws=_Arg("det_ws"),
        counters=_Arg("det_counters"),
        ldo=ldo,
        out_l=out_l,
        enable_pdl=pdl,
    )
    nvjet = module is not None and plan.template == (
        det._TEMPLATE_PREFIX + K1_DGRAD_NVJET_DET_PROGRAM
    )
    if nvjet:
        fn, template = _atomic_bind(module, values, plan.grid, arch)
    else:
        if module_name is None:
            module_name = det.select_module(arch, plan.template)
        fn, template = det._bind(module_name, values, plan.grid, arch)
    views = dict(
        A=(a_desc, False),
        B=(b_desc, False),
        out3=(O3, False),
        o_alias=(det.flat_alias(O3), True),
    )
    bound = _k1_binding(template, views, sources)
    if bound is None:
        return _CachedDetK1(plan=plan)
    # the workspace is a marker: grow the arena only once the program and its views are bound
    arena = _atomic_arena_for(index, stream, out.device)
    arena.require("k1_det_ws", plan.ws_f32_elems)
    arena.require("k1_det_counters", plan.counters_alloc_u32)
    return _CachedDetK1(
        plan=plan,
        entry=fn,
        arena=arena,
        k1=bound,
        module_name=module if nvjet else None,
    )


def _det_k1(
    sources: tuple,
    flags: tuple,
    det_args: Callable[[], tuple],
    det_call: Callable[[], Any],
    enable_pdl: Optional[bool],
    weights_static: bool = False,
    module: Optional[str] = None,
) -> Any:
    """The deterministic package's K1 convenience call -- ``det_args()`` gives its ``(A, B, out, transposed_out,
    knobs)``, ``det_call()`` makes it with ``enable_pdl`` / ``weights_static`` -- on this package's convenience path
    (``module``: this package's program bound on the plan, :func:`_plan_det_k1`):
    planned once per key of the entry point's tensors ``sources`` (the last is ``out``), ``flags`` and the PDL flag
    (``enable_pdl`` resolved for the device, bound into the argument list; :func:`_plan_det_k1`: the deterministic
    package's plan, program and so its bits), then the bound launch (:class:`_K1Binding`).  Tensors that are not on
    one CUDA device, and a key whose views are not bound, make the deterministic package's call."""
    index = _k1_index(*sources)
    if index < 0:
        return det_call()
    det = deterministic_backend()
    stream = det._current_raw_stream(index)
    pdl = int(det.pdl_enabled(enable_pdl, index))
    key = _k1_key(index, stream, (flags, pdl, bool(weights_static)), *sources)
    entry = det._cache_get(_K1_DET_CACHE, key)
    if entry is None:
        with _ATOMIC_PLAN_LOCK:
            entry = det._cache_get(_K1_DET_CACHE, key)
            if entry is None:
                A, B, out, transposed_out, knobs = det_args()
                entry = _cache_insert(
                    _K1_DET_CACHE,
                    key,
                    _plan_det_k1(
                        A,
                        B,
                        out,
                        transposed_out,
                        knobs,
                        sources,
                        index,
                        stream,
                        pdl,
                        weights_static,
                        module,
                    ),
                )
    bound, fn = entry.k1, entry.entry
    if bound is None or fn is None:
        return det_call()
    _launch_k1_binding(det, fn, bound, entry.arena, sources, index, stream)
    return sources[-1]


def _launch_cached_atomic_k1(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    *,
    transposed_out: bool,
    own: dict[str, Any],
    det_call: Callable[[], Any],
    det_args: Optional[Callable[[], tuple]] = None,
    det_knobs: tuple = (),
    enable_pdl: Optional[bool] = None,
    weights_static: bool = False,
) -> Any:
    """The convenience path of this package's K1 GEMM: planned once per key (:func:`plan_atomic_gemm`), then the
    bound launch (:class:`_K1Binding`).  A row the planner keeps on the deterministic program runs that program
    through this package's binding (:func:`_plan_det_k1`) of the deterministic package's convenience call
    ``det_args()`` -- ``(A, B, out, transposed_out, knobs)``; ``det_knobs`` are its knobs, part of the key -- or,
    without ``det_args`` or a binding, makes that call (``det_call()``).  The PDL flag (``enable_pdl`` resolved for
    the device) is bound into the argument list: a key per flag and per ``weights_static``."""
    det = deterministic_backend()
    index = _k1_index(A, B, out)
    if index < 0:
        # A, B and out are not on one CUDA device: raises
        index = det._device_index(det._same_cuda_device(A, B, out))
    stream = det._current_raw_stream(index)
    pdl = int(det.pdl_enabled(enable_pdl, index))
    flags = (
        "k1",
        bool(transposed_out),
        tuple(sorted(own.items())),
        det_knobs,
        pdl,
        bool(weights_static),
    )
    key = _k1_key(index, stream, flags, A, B, out)
    entry = det._cache_get(_K1_ATOMIC_CACHE, key)
    if entry is None:
        with _ATOMIC_PLAN_LOCK:
            entry = det._cache_get(_K1_ATOMIC_CACHE, key)
            if entry is None:
                device = out.device
                arch = det.require_arch(device)
                plan, _, _, _, _ = plan_atomic_gemm(
                    A,
                    B,
                    out,
                    transposed_out=transposed_out,
                    arch=arch,
                    sm_count=det.device_facts(device).sm_count,
                    **own,
                )
                if plan is None:
                    entry = _CachedAtomic(plan=None, arch=arch)
                    if det_args is not None:
                        dA, dB, dout, dtransposed, dknobs = det_args()
                        bound = _plan_det_k1(
                            dA,
                            dB,
                            dout,
                            dtransposed,
                            dknobs,
                            (A, B, out),
                            index,
                            stream,
                            pdl,
                            weights_static,
                        )
                        entry.det_plan, entry.entry = bound.plan, bound.entry
                        entry.arena, entry.k1 = bound.arena, bound.k1
                else:
                    arena = None
                    if plan.workspace:
                        arena = _atomic_arena_for(index, stream, device)
                        arena.require("k1_slots", plan.ws_f32_elems)
                        arena.require("k1_tickets", plan.counters_u32)
                    module_name = cake_jit.select_module(arch, plan.template)
                    fn, arg_plan = _atomic_entry(module_name, plan.grid, arch)
                    entry = _CachedAtomic(
                        plan=plan,
                        arch=arch,
                        module_name=module_name,
                        entry=fn,
                        arg_plan=arg_plan,
                        arena=arena,
                        k1=_bind_atomic_k1(
                            plan, module_name, arg_plan, A, B, out, arena, pdl
                        ),
                        order_varying=not plan.reproducible,
                    )
                entry = _cache_insert(_K1_ATOMIC_CACHE, key, entry)
    bound, fn = entry.k1, entry.entry
    if bound is None or fn is None:
        if entry.plan is None:
            # a row on the deterministic program whose views the binding cannot rebuild: the same program through
            # the deterministic package's entry point
            return det_call()
        if entry.order_varying:
            _alert_order_varying(_k1_launch_text(entry.plan))
        return _launch_atomic_k1_unbound(entry, A, B, out, index, stream, pdl)
    if entry.order_varying:
        _alert_order_varying(_k1_launch_text(entry.plan))
    _launch_k1_binding(det, fn, bound, entry.arena, (A, B, out), index, stream)
    return out


def _launch_atomic_k1_unbound(
    entry: _CachedAtomic,
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    index: int,
    stream: int,
    pdl: int,
) -> torch.Tensor:
    """This package's planned K1 program for a key whose views are not bound (:func:`_bind_atomic_k1`): the
    operand views and values built for the call (``pdl``: the launch's PDL flag, 0 or 1)."""
    det = deterministic_backend()
    plan, fn = entry.plan, entry.entry
    assert fn is not None
    a_desc, b_desc, out3 = _k1_launch_views(A, B, out, plan)
    ldo, out_l = _k1_out_strides(out3)
    if entry.arena is not None:
        ws, counters = (
            entry.arena.buffers["k1_slots"],
            entry.arena.buffers["k1_tickets"],
        )
    else:
        ws, counters = _k1_dummies(out.device)
    values = _k1_values(
        plan,
        A=a_desc,
        B=b_desc,
        out=det.flat_alias(out3),
        ws=ws,
        counters=counters,
        ldo=ldo,
        out_l=out_l,
        enable_pdl=pdl,
    )
    arguments = _order_atomic_arguments(
        entry.module_name, entry.arg_plan, values, plan.grid
    )
    with det._ffi_stream_context(index, stream):
        fn(*arguments)
    return out


def _k2_binding(
    plan: CakeDenseGemmNondetRouterPlan,
    out: torch.Tensor,
    split_src: torch.Tensor,
    module_name: str,
    arg_plan: list,
    split_module: Optional[str],
    split_arg_plan: Optional[list],
    pdl: int,
) -> _K2Binding:
    """The :class:`_K2Binding` of a planned K2 launch (its argument lists ordered once, fail-closed as
    :func:`_order_atomic_arguments`: the values a call of the key passes, with the per-call views as
    :data:`_K2_CALL_ARGS`)."""
    a_arg, out32_arg, d_arg, src_arg = _K2_CALL_ARGS
    if plan.simt is not None:
        # a SIMT program: the streamed operand's span, the split source and out, per call
        values = _simt_values(
            plan, A=a_arg, B=src_arg, out=d_arg, ldo=out.stride(0), enable_pdl=pdl
        )
        return _K2Binding(
            kernel_template=_order_atomic_arguments(
                module_name, arg_plan, values, plan.grid
            ),
            split_template=None,
            split_shape=(),
            fill=None,
            out_contiguous=out.is_contiguous(),
            stack=False,
        )
    values = _k2_values(
        plan,
        A=a_arg,
        parts=_K2_PARTS_ARG,
        out=out,
        enable_pdl=pdl,
        ws=_K2_KSUM_WS_ARG if plan.ksum else None,
        counters=_K2_KSUM_TICKETS_ARG if plan.ksum else None,
    )
    values.update(out32=out32_arg, D=d_arg)
    kernel_template = _order_atomic_arguments(module_name, arg_plan, values, plan.grid)
    split_template = None
    n = split_src.numel()
    if split_module is not None:
        det = deterministic_backend()
        split_template = _order_atomic_arguments(
            split_module,
            split_arg_plan,
            det._split_values(src_arg, _K2_PARTS_ARG, n, pdl),
            det._split_grid(n),
        )
    fill = None
    if plan.part_tiles:
        fill = (
            int(out.shape[0]),
            int(out.shape[1]),
            int(out.stride(0)) * out.element_size(),
        )
    return _K2Binding(
        kernel_template=kernel_template,
        split_template=split_template,
        split_shape=tuple(int(s) for s in split_src.shape),
        fill=fill,
        out_contiguous=out.is_contiguous(),
        ksum_workspace=plan.ksum_workspace if plan.ksum else None,
    )


def _k2_index(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor) -> int:
    """The CUDA device index of a K2 call (``out``'s), after checking that ``A`` and ``B`` are on it (else the
    deterministic package's error is raised)."""
    index = out.get_device()
    if index < 0 or A.get_device() != index or B.get_device() != index:
        # A, B and out are not on one CUDA device: raises
        deterministic_backend()._same_cuda_device(A, B, out)
    return index


#: The plan flags ``(swapped, a_mn, b_mn)`` under which an entry point's own tensors are the launch views of
#: :func:`_k2_views` (the streamed span ``P[None]`` and the split source ``Q``), so the call takes no transposes:
#: the forward ``X @ W.T`` (``X[None]``, ``W``), the input gradient ``G @ W`` (``G[None]``, ``W``) and the weight
#: gradient ``G.T @ X`` swapped (``X[None]``, ``G``).
_FORWARD_VIEWS = (False, False, False)
_DGRAD_VIEWS = (False, False, True)
_WGRAD_VIEWS = (True, True, True)


def _launch_views(
    A: torch.Tensor, B: torch.Tensor, plan: Any, views: Optional[tuple]
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(a_view, split_src)`` of a launch: an entry point's own views ``(a_view, split_src, flags)`` when ``plan``
    has those flags (``(plan.swapped, plan.a_mn, plan.b_mn)``, one of :data:`_FORWARD_VIEWS` / ...), else
    :func:`_k2_views` of the caller's operands."""
    if views is not None and views[2] == (plan.swapped, plan.a_mn, plan.b_mn):
        return views[0], views[1]
    return _k2_views(A, B, plan)


def _plan_atomic_k2(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    splits: Optional[int],
    index: int,
    stream: int,
    pdl: int,
    memo: Optional[_RouterMemo] = None,
) -> _CachedAtomic:
    """Plan this package's K2 program for a key of :func:`_launch_cached_atomic_k2` (under ``_ATOMIC_PLAN_LOCK``):
    :func:`plan_atomic_router` on the device's architecture and SM count (``plan=None``: the planner keeps the row on
    the deterministic program, at the split count of :func:`_k2_det_splits` without a caller's), the program's and
    fused split's entries, the BF16x3 stack in the arena of (device
    ``index``, raw ``stream``) and the argument lists bound with the PDL flag ``pdl`` (0 or 1); ``memo`` is the
    router-table memo the call read (``None`` with a caller's split count)."""
    det = deterministic_backend()
    device = out.device
    arch = det.require_arch(device)
    sm_count = det.device_facts(device).sm_count
    plan, _ = plan_atomic_router(
        A,
        B,
        out,
        arch=arch,
        sm_count=sm_count,
        splits=splits,
    )
    if plan is None:
        det_n = None if splits is not None else _k2_det_splits(A, B, arch, sm_count)
        return _CachedAtomic(plan=None, arch=arch, memo=memo, det_splits=det_n)
    _, split_src = _k2_views(A, B, plan)
    if plan.simt is not None:
        # the SIMT program: no stack, no split; the arena holds the stream context its launches enter
        module_name = cake_jit.select_module(arch, plan.template)
        fn, arg_plan = _atomic_entry(module_name, plan.grid, arch)
        return _CachedAtomic(
            plan=plan,
            arch=arch,
            module_name=module_name,
            entry=fn,
            arg_plan=arg_plan,
            arena=_atomic_arena_for(index, stream, device),
            k2=_k2_binding(
                plan, out, split_src, module_name, arg_plan, None, None, pdl
            ),
            memo=memo,
        )
    split_module = det._fused_split_for(arch, split_src)
    arena = _atomic_arena_for(index, stream, device)
    arena.require("k2_stack", 3 * split_src.numel())
    if split_module is None:
        arena.require("k2_split_scratch", split_src.numel())
    if plan.ksum:
        slots, words = plan.ksum_workspace
        arena.require("k2_ksum_slots", slots)
        arena.require("k2_ksum_tickets", words)
    module_name = cake_jit.select_module(arch, plan.template)
    fn, arg_plan = _atomic_entry(module_name, plan.grid, arch)
    split_fn, split_arg_plan = None, None
    if split_module is not None:
        record = det.MODULES[split_module]
        split_fn = getattr(
            det.load_cake_dense_projection_det_gemm_module(split_module, arch),
            record["ffi_entry"],
        )
        split_arg_plan = det.arg_plan_of(record)
    if plan.part_tiles:
        _cu_driver()
    binding = _k2_binding(
        plan,
        out,
        split_src,
        module_name,
        arg_plan,
        split_module,
        split_arg_plan,
        pdl,
    )
    return _CachedAtomic(
        plan=plan,
        arch=arch,
        module_name=module_name,
        entry=fn,
        arg_plan=arg_plan,
        arena=arena,
        split_module=split_module,
        split_entry=split_fn,
        split_arg_plan=split_arg_plan,
        k2=binding,
        memo=memo,
        order_varying=not plan.reproducible,
    )


def _k2_stream_context(
    det: ModuleType, arena: _AtomicArena, index: int, stream: int
) -> Any:
    """The tvm-ffi stream context of ``arena``'s (device ``index``, raw ``stream``), made once and kept in the arena:
    a K2 launch enters it under the arena's lock, so one thread at a time does."""
    context = arena.stream_context
    if context is None:
        context = arena.stream_context = det._ffi_stream_context(index, stream)
    return context


def _launch_atomic_k2_binding(
    det: ModuleType,
    entry: _CachedAtomic,
    out: torch.Tensor,
    a_view: torch.Tensor,
    split_src: torch.Tensor,
    index: int,
    stream: int,
    out_ptr: int,
) -> None:
    """Launch a planned K2 program of this package (:func:`_plan_atomic_k2`) into ``out`` (data pointer ``out_ptr``)
    on the raw ``stream`` of device ``index`` with the call's launch views ``a_view`` / ``split_src``: the split, the
    zero fill of ``out`` (part tiles) and the program back to back under the arena's lock, the stack carved again if
    the arena grew.  ``out`` is converted once for both of its arguments; the other views go to the FFI calls as
    torch tensors (each converted there, once).  A SIMT program is one kernel over ``a_view``, ``split_src`` (the
    weight gradient's output gradient, the input gradient's router weight) and ``out``."""
    arena, fn, bound = entry.arena, entry.entry, entry.k2
    assert arena is not None and fn is not None and bound is not None
    if entry.order_varying:
        # before the split, the fill of ``out`` and the program: a raise leaves ``out`` untouched
        _alert_order_varying(_k2_launch_text(entry.plan))
    split_fn = entry.split_entry
    # the split, the memset and the kernel share the arena's BF16x3 stack
    with arena.lock:
        if bound.version != arena.version:
            bound.carve(arena, split_fn is None)
        if not bound.stack:
            arguments, _ = bound.arguments((a_view, None, out, split_src))
            with _k2_stream_context(det, arena, index, stream):
                fn(*arguments)
            return
        if split_fn is None:
            det.split_fp32_to_bf16x3_det_(split_src, bound.parts, bound.resid)
        out_ffi = tvm_ffi.from_dlpack(out, require_contiguous=False)
        views = (
            a_view,
            out_ffi if bound.out_contiguous else det.flat_alias(out),
            out_ffi,
            None if split_fn is None else split_src,
        )
        arguments, split_arguments = bound.arguments(views)
        fill = bound.fill
        if fill is not None and _CU_DRIVER is None:
            _cu_driver()
        with _k2_stream_context(det, arena, index, stream):
            if split_fn is not None and split_arguments is not None:
                split_fn(*split_arguments)
            if fill is not None:
                rows, cols, pitch = fill
                _fill_rows(out, index, out_ptr, rows, cols, pitch, stream)
            fn(*arguments)


def _launch_cached_atomic_k2(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    *,
    splits: Optional[int],
    enable_pdl: Optional[bool] = None,
    views: Optional[tuple] = None,
) -> torch.Tensor:
    """The convenience path of this package's K2 GEMM: a row without a verdict (no ``splits``) and a binding the
    planner keeps on the deterministic program run that program (:func:`_launch_cached_splitk_k2`); ``views`` are an
    entry point's own launch views (:func:`_launch_views`).  A key without ``splits`` is planned again after a router
    table changed (its launch keeps the :func:`_router_memo` it was planned under)."""
    det = deterministic_backend()
    # the call's geometry is read once (one call per tensor attribute) for the checks and the key below
    index = _k2_index(A, B, out)
    geometry = _k2_geometry(A, B, out)
    pdl = int(det.pdl_enabled(enable_pdl, index))
    # a call without a split count reads the router tables (their memo); with one, its plan reads none
    memo = None if splits is not None else _router_memo()
    if memo is not None and _k2_runs_without_verdict(geometry, index, memo):
        # a row without a verdict runs the deterministic program, without planning (or caching) the key here
        return _launch_cached_splitk_k2(A, B, out, None, index, geometry, pdl, views)
    stream = det._current_raw_stream(index)
    out_ptr = out.data_ptr()
    # the PDL flag is bound into the argument lists: a key per flag
    key = _k2_key(index, stream, (splits, pdl), A, B, out, geometry, out_ptr)
    entry = det._cache_get(_K2_ATOMIC_CACHE, key)
    if entry is None or entry.memo is not memo:
        with _ATOMIC_PLAN_LOCK:
            entry = det._cache_get(_K2_ATOMIC_CACHE, key)
            if entry is None or entry.memo is not memo:
                entry = _cache_insert(
                    _K2_ATOMIC_CACHE,
                    key,
                    _plan_atomic_k2(A, B, out, splits, index, stream, pdl, memo),
                )
    plan = entry.plan
    if plan is None:
        det_n = splits if splits is not None else entry.det_splits
        return _launch_cached_splitk_k2(A, B, out, det_n, index, geometry, pdl, views)
    a_view, split_src = _launch_views(A, B, plan, views)
    _launch_atomic_k2_binding(
        det, entry, out, a_view, split_src, index, stream, out_ptr
    )
    return out


#: The cached bindings of the deterministic K2 program on this package's convenience path (keys as
#: :data:`_K2_ATOMIC_CACHE`'s, :func:`_k2_key`).
_K2_SPLITK_CACHE: "OrderedDict[tuple, Any]" = OrderedDict()


@dataclass
class _CachedSplitK2:
    """A planned launch of the deterministic K2 program on this package's convenience path: the deterministic
    package's plan, its program's and fused split's entries, the arena of the (device, stream), the bound arguments
    (``k2``; ``None``: the call goes to the deterministic package's convenience path) and the entry of the partial sum
    (``sum_entry``; ``None``: one split, or ``torch.sum`` reduces the partials)."""

    plan: Any
    entry: Optional[Callable[..., Any]] = field(default=None, repr=False)
    arena: Optional[_AtomicArena] = field(default=None, repr=False)
    k2: Optional[_K2Binding] = field(default=None, repr=False)
    split_entry: Optional[Callable[..., Any]] = field(default=None, repr=False)
    sum_entry: Optional[Callable[..., Any]] = field(default=None, repr=False)


def _plan_splitk_k2(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    splits: Optional[int],
    index: int,
    stream: int,
    pdl: int,
) -> _CachedSplitK2:
    """Plan the deterministic K2 program for a key of :func:`_launch_cached_splitk_k2` (under ``_ATOMIC_PLAN_LOCK``):
    the deterministic package's planner and fail-closed view check (its errors), its program and fused split bound in
    the deterministic package's argument order with the PDL flag ``pdl`` (0 or 1), the BF16x3 stack, the torch
    chain's scratch and the split-K partials in this package's arena, and the partial sum of several splits where it
    serves the launch (:func:`_partials_sum_serves`, :func:`_partials_sum_binding`).  The views :func:`_k2_views` rebuilds per call must be the deterministic package's (else
    the entry calls that package's convenience path)."""
    det = deterministic_backend()
    device = out.device
    arch = det.require_arch(device)
    plan, A_eff, B_eff = det.cake_dense_gemm_det_train_router_plan(
        A, B, out, splits=splits
    )
    a_view, split_src = det._k2_checked_views(A, B, A_eff, B_eff, plan)
    mine = _k2_views(A, B, plan)
    if not (det._same_view(mine[0], a_view) and det._same_view(mine[1], split_src)):
        return _CachedSplitK2(plan=plan)
    n = split_src.numel()
    split_module = det._fused_split_for(arch, split_src)
    a_arg, out32_arg, _, src_arg = _K2_CALL_ARGS
    partials_shape: Optional[tuple[int, ...]] = None
    if plan.splits > 1:
        rows, cols = (plan.N, plan.M) if plan.transposed_out else (plan.M, plan.N)
        partials_shape = (int(plan.splits), int(rows), int(cols))
        out32, ldo, out_l = _K2_PARTIALS_ARG, cols, rows * cols
    else:
        out32, ldo, out_l = out32_arg, int(out.stride(0)), 0
    values = det._k2_values(
        plan,
        A=a_arg,
        parts=_K2_PARTS_ARG,
        out32=out32,
        ldo=ldo,
        out_l=out_l,
        enable_pdl=pdl,
    )
    fn, template = det._bind(
        det.select_module(arch, plan.template), values, plan.grid, arch
    )
    split_fn: Optional[Callable[..., Any]] = None
    split_template: Optional[tuple] = None
    if split_module is not None:
        split_fn, split_template = det._bind(
            split_module,
            det._split_values(src_arg, _K2_PARTS_ARG, n, pdl),
            det._split_grid(n),
            arch,
        )
    arena = _atomic_arena_for(index, stream, device)
    arena.require("k2_stack", 3 * n)
    if split_module is None:
        arena.require("k2_split_scratch", n)
    sum_fn: Optional[Callable[..., Any]] = None
    sum_template: Optional[list] = None
    if partials_shape is not None:
        arena.require("k2_splitk_partials", math.prod(partials_shape))
        if _partials_sum_serves(partials_shape, out):
            sum_fn, sum_template = _partials_sum_binding(arch, partials_shape, out, pdl)
    binding = _K2Binding(
        kernel_template=list(template),
        split_template=None if split_template is None else list(split_template),
        split_shape=tuple(int(s) for s in split_src.shape),
        fill=None,
        out_contiguous=out.is_contiguous(),
        partials_shape=partials_shape,
        sum_template=sum_template,
    )
    return _CachedSplitK2(
        plan=plan,
        entry=fn,
        arena=arena,
        k2=binding,
        split_entry=split_fn,
        sum_entry=sum_fn,
    )


def _launch_cached_splitk_k2(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    splits: Optional[int],
    index: int,
    geometry: _K2Geometry,
    pdl: int,
    views: Optional[tuple] = None,
) -> torch.Tensor:
    """The deterministic package's K2 GEMM ``out = A @ B`` (its plan and programs, and the split-K partials summed in
    the order of its ``torch.sum``: its bits) on this package's convenience path: planned once per key (:func:`_plan_splitk_k2`), then
    the bound launch (:func:`_launch_splitk_k2_binding`) -- the caller's views passed per call, the stack and the
    partials converted once per arena growth (:class:`_K2Binding`).  The split, the program and the reduction share
    the arena and hold its lock, as the deterministic package's convenience path does.  ``index`` / ``geometry`` are
    the call's (:func:`_k2_index`, :func:`_k2_geometry`); ``pdl`` the launch's PDL flag (0 or 1: the split and the
    program launch as programmatic dependent launches); ``views`` an entry point's own launch views
    (:func:`_launch_views`)."""
    det = deterministic_backend()
    stream = det._current_raw_stream(index)
    # the PDL flag is bound into the argument lists: a key per flag
    key = _k2_key(index, stream, (splits, pdl), A, B, out, geometry, out.data_ptr())
    entry = det._cache_get(_K2_SPLITK_CACHE, key)
    if entry is None:
        with _ATOMIC_PLAN_LOCK:
            entry = det._cache_get(_K2_SPLITK_CACHE, key)
            if entry is None:
                entry = _cache_insert(
                    _K2_SPLITK_CACHE,
                    key,
                    _plan_splitk_k2(A, B, out, splits, index, stream, pdl),
                )
    if entry.k2 is None or entry.entry is None or entry.arena is None:
        return det.cake_dense_gemm_det_train_router(
            A, B, out, splits=splits, enable_pdl=bool(pdl)
        )
    a_view, split_src = _launch_views(A, B, entry.plan, views)
    _launch_splitk_k2_binding(det, entry, out, a_view, split_src, index, stream)
    return out


def _launch_splitk_k2_binding(
    det: ModuleType,
    entry: _CachedSplitK2,
    out: torch.Tensor,
    a_view: torch.Tensor,
    split_src: torch.Tensor,
    index: int,
    stream: int,
) -> None:
    """Launch a bound deterministic K2 program (:func:`_plan_splitk_k2`) into ``out`` on the raw ``stream`` of device
    ``index`` with the call's launch views ``a_view`` / ``split_src``: the split, the program and the reduction of
    its split-K partials under the arena's lock, the stack and the partials carved again if the arena grew.  The
    partials are reduced by this package's partial sum right after the program, in the same launch chain, where it
    serves the launch (``entry.sum_entry``), else by the deterministic package's ``torch.sum``: the same order, so the
    same bits.  The call's views go to the FFI calls as torch tensors (each converted there, once)."""
    bound, fn, arena = entry.k2, entry.entry, entry.arena
    assert bound is not None and fn is not None and arena is not None
    split_fn, sum_fn = entry.split_entry, entry.sum_entry
    with arena.lock:
        if bound.version != arena.version:
            bound.carve(arena, split_fn is None)
        if split_fn is None:
            det.split_fp32_to_bf16x3_det_(split_src, bound.parts, bound.resid)
        out32: Any = None
        if bound.partials_shape is None or sum_fn is not None:
            # one split: the program stores into ``out``; several: the partial sum does
            out32 = out if bound.out_contiguous else det.flat_alias(out)
        views = (a_view, out32, None, None if split_fn is None else split_src)
        arguments, split_arguments = bound.arguments(views)
        with _k2_stream_context(det, arena, index, stream):
            if split_fn is not None and split_arguments is not None:
                split_fn(*split_arguments)
            fn(*arguments)
            if sum_fn is not None:
                sum_fn(*bound.sum_arguments(views))
        if sum_fn is None and bound.workspace is not None:
            # the deterministic package's reduction of the partials (its overload and kernel)
            torch.sum(bound.workspace, 0, out=out)


def _splitk_k2(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    splits: Optional[int],
    views: Optional[tuple] = None,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    """:func:`_launch_cached_splitk_k2` of a call (its device checked, its geometry read once, ``enable_pdl``
    resolved: default, the device supports it)."""
    index = _k2_index(A, B, out)
    pdl = int(deterministic_backend().pdl_enabled(enable_pdl, index))
    return _launch_cached_splitk_k2(
        A, B, out, splits, index, _k2_geometry(A, B, out), pdl, views
    )


#: The cached calls of the router entry points (``cake_dense_gemm_nondet_train_router_forward``, ``cake_dense_gemm_nondet_train_router_dgrad``,
#: ``cake_dense_gemm_nondet_train_router_wgrad``), keyed by the entry point's own tensors (:func:`_k1_key`), the operation, its split count and the
#: PDL flag (:class:`_CachedK2Entry`).
_K2_ENTRY_CACHE: "OrderedDict[tuple, Any]" = OrderedDict()


@dataclass
class _CachedK2Entry:
    """The launch a router entry point's call of a key of :data:`_K2_ENTRY_CACHE` takes (:func:`_plan_k2_entry`): an
    entry of :data:`_K2_SPLITK_CACHE` (the deterministic program) or, with ``atomic``, of :data:`_K2_ATOMIC_CACHE` (this
    package's program), each planned and bound there as the entry point's own path plans it, with the entry point's
    tensors as its launch views; ``None``: the call takes the entry point's own path.  ``memo`` is the router-table
    memo (:func:`_router_memo`) and ``route`` the :data:`ROUTES` value of the layout class ``layout`` (``None``: a
    route that does not read it) the call was routed under; ``arena`` the launch's (the cache's reference to it)."""

    memo: Any
    layout: Optional[str] = None
    route: Optional[str] = None
    launch: Optional[Union[_CachedSplitK2, _CachedAtomic]] = field(
        default=None, repr=False
    )
    atomic: bool = False
    arena: Optional[_AtomicArena] = field(default=None, repr=False)


def _k2_entry_operands(
    op: str, P: torch.Tensor, Q: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, tuple[bool, bool, bool]]:
    """``(A, B, flags)`` of a router entry point's tensors ``P``, ``Q`` (``"fwd"``: ``X``, ``W``; ``"dgrad"``: ``G``,
    ``W``; ``"wgrad"``: ``G``, ``X``): the launched GEMM ``A @ B`` and the plan flags under which the entry point's own
    tensors are the launch views (:data:`_FORWARD_VIEWS` / :data:`_DGRAD_VIEWS` / :data:`_WGRAD_VIEWS`)."""
    if op == "fwd":
        return P, Q.t(), _FORWARD_VIEWS
    if op == "dgrad":
        return P, Q, _DGRAD_VIEWS
    return P.t(), Q, _WGRAD_VIEWS


def _plan_k2_entry(
    op: str,
    P: torch.Tensor,
    Q: torch.Tensor,
    out: torch.Tensor,
    splits: Optional[int],
    index: int,
    stream: int,
    pdl: int,
    memo: _RouterMemo,
) -> _CachedK2Entry:
    """The :class:`_CachedK2Entry` of a router entry point's call (under ``_ATOMIC_PLAN_LOCK``): the route the entry
    point's own path takes under the router tables of ``memo`` -- the forward without a split count while no router
    table gives the forward a verdict, a layout routed ``"det"``, a call without a split count whose row has no
    verdict (:func:`_k2_runs_without_verdict`) and a row the planner keeps on the deterministic program run the
    deterministic program (the input gradient: one split), the others this package's program -- and that path's
    cached launch of the route (looked up in, or planned into, :data:`_K2_SPLITK_CACHE` / :data:`_K2_ATOMIC_CACHE`
    under the same key; a launch of this package without a split count planned under other router tables is planned
    again).  A launch that is not bound, or whose plan does not take the entry point's tensors as its views, leaves
    the call to the entry point's own path."""
    det = deterministic_backend()
    A, B, flags = _k2_entry_operands(op, P, Q)
    geometry = _k2_geometry(A, B, out)
    key = _k2_key(index, stream, (splits, pdl), A, B, out, geometry, out.data_ptr())
    launch: Optional[Union[_CachedSplitK2, _CachedAtomic]] = None
    atomic = False
    # the deterministic program's split count: the caller's, else the row's in the router tables (the planner's)
    det_n = splits
    # the forward without a split count while no table gives it a verdict does not read ROUTES
    layout = route = None
    if splits is not None or op != "fwd" or memo.forward_verdicts:
        layout = _k2_layout(A, B)
        route = _route(layout)
    if (
        route is not None
        and route != "det"
        and not (splits is None and _k2_runs_without_verdict(geometry, index, memo))
    ):
        # the memo of the launch's plan, as _launch_cached_atomic_k2 keeps it (none with a split count)
        planned = memo if splits is None else None
        cached = det._cache_get(_K2_ATOMIC_CACHE, key)
        if cached is None or cached.memo is not planned:
            cached = _cache_insert(
                _K2_ATOMIC_CACHE,
                key,
                _plan_atomic_k2(A, B, out, splits, index, stream, pdl, planned),
            )
        if cached.plan is not None:
            launch, atomic = cached, True
        elif splits is None:
            det_n = cached.det_splits
    if launch is None:
        if det_n != splits:
            key = _k2_key(
                index, stream, (det_n, pdl), A, B, out, geometry, out.data_ptr()
            )
        launch = det._cache_get(_K2_SPLITK_CACHE, key)
        if launch is None:
            launch = _cache_insert(
                _K2_SPLITK_CACHE,
                key,
                _plan_splitk_k2(A, B, out, det_n, index, stream, pdl),
            )
    plan = launch.plan
    if (
        launch.k2 is None
        or launch.entry is None
        or launch.arena is None
        or (plan.swapped, plan.a_mn, plan.b_mn) != flags
    ):
        return _CachedK2Entry(memo=memo, layout=layout, route=route)
    return _CachedK2Entry(
        memo=memo,
        layout=layout,
        route=route,
        launch=launch,
        atomic=atomic,
        arena=launch.arena,
    )


def _k2_entry_stale(entry: Optional[_CachedK2Entry], memo: _RouterMemo) -> bool:
    """Whether a call plans its key of :data:`_K2_ENTRY_CACHE` (again): no entry, or one routed under other router
    tables (``memo``) or another :data:`ROUTES` value of its layout class."""
    return (
        entry is None
        or entry.memo is not memo
        or (entry.layout is not None and _route(entry.layout) != entry.route)
    )


def _k2_entry(
    op: str,
    P: torch.Tensor,
    Q: torch.Tensor,
    out: torch.Tensor,
    splits: Optional[int],
    enable_pdl: Optional[bool],
) -> Optional[torch.Tensor]:
    """A router entry point's call (``op`` and its tensors ``P``, ``Q`` as for
    :func:`_k2_entry_operands`, into ``out``) through :data:`_K2_ENTRY_CACHE`: planned once per key of the entry
    point's own tensors, the operation, ``splits`` and the PDL flag (``enable_pdl`` resolved for the device), and again
    after a router table or the route of its layout class changed (:func:`_plan_k2_entry`), then the bound launch with
    the entry point's tensors as the launch views (``X[None]`` / ``G[None]`` and the split source), so a call builds
    no transposed operand, takes no route and reads no second key.  Returns ``out``, or ``None`` when the call takes
    the entry point's own path: tensors not on one CUDA device, router tables the memo does not track, a launch not
    bound for the key."""
    index = _k1_index(P, Q, out)
    if index < 0:
        return None
    det = deterministic_backend()
    stream = det._current_raw_stream(index)
    pdl = int(det.pdl_enabled(enable_pdl, index))
    key = _k1_key(index, stream, (op, splits, pdl), P, Q, out)
    memo = _router_memo()
    entry = det._cache_get(_K2_ENTRY_CACHE, key)
    if _k2_entry_stale(entry, memo):
        if not memo.tracked:
            return None
        with _ATOMIC_PLAN_LOCK:
            entry = det._cache_get(_K2_ENTRY_CACHE, key)
            if _k2_entry_stale(entry, memo):
                entry = _cache_insert(
                    _K2_ENTRY_CACHE,
                    key,
                    _plan_k2_entry(op, P, Q, out, splits, index, stream, pdl, memo),
                )
    launch = entry.launch
    if launch is None:
        return None
    if op == "wgrad":
        a_view, split_src = Q.unsqueeze(0), P
    else:
        a_view, split_src = P.unsqueeze(0), Q
    if entry.atomic:
        _launch_atomic_k2_binding(
            det, launch, out, a_view, split_src, index, stream, out.data_ptr()
        )
    else:
        _launch_splitk_k2_binding(det, launch, out, a_view, split_src, index, stream)
    return out


def _drop_convenience_entries() -> None:
    """Drop this package's cached convenience-path launches, what the convenience path keeps from the router tables
    (:func:`_router_memo`) and this package's workspace arenas with their retired buffers: a CUDA graph that captured
    a convenience-path launch of this package must not replay afterwards."""
    global _ROUTER_MEMO
    with _ATOMIC_PLAN_LOCK:
        _K1_ATOMIC_CACHE.clear()
        _K1_DET_CACHE.clear()
        _K1_PROGRAM_CACHE.clear()
        _K2_ATOMIC_CACHE.clear()
        _K2_SPLITK_CACHE.clear()
        _K2_ENTRY_CACHE.clear()
        _ATOMIC_ARENAS.clear()
        _K1_STREAM_CONTEXTS.clear()
        _ROUTER_MEMO = _RouterMemo(-1, None, None, False, True)


def cake_dense_gemm_nondet_train_convenience_cache_clear() -> None:
    """Drop every cached convenience-path launch, what the convenience path keeps from the router tables
    (:func:`_router_memo`) and every workspace arena of both packages (this package's launches and the deterministic
    package's).  This invalidates every CUDA graph that captured a convenience-path call: its replay would use freed
    workspace, so capture such graphs again after it.  A captured convenience-path call of this package that
    this package launches from its arenas (its programs, and the deterministic programs it launches through its
    binding) stays valid until then: its arena keeps every buffer a launch was bound to, through growths and cache
    evictions.  A call on a row that runs the deterministic program, which this package hands to the deterministic
    package's entry point of the same operation (launch views its binding cannot rebuild from the call's tensors, such
    as two operands that alias; a batched forward on that package's swapped small-token program), follows that
    package's rule: a CUDA graph captures its prepared launch, not its convenience path.  The graphs captured on one stream share
    its arena: replay them one at a time, and not concurrently with calls on that stream."""
    _drop_convenience_entries()
    deterministic_backend().cake_dense_gemm_det_train_convenience_cache_clear()


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def is_cake_dense_gemm_nondet_train_supported(device: torch.device) -> bool:
    """Whether ``device`` is a CUDA device of compute capability 10.0, 10.3 or 10.7 (``sm_100a`` / ``sm_103a`` /
    ``sm_107a``), the devices of the deterministic package whose programs this package reuses (its ``arch_for``).
    Shape-level support is checked per call: an unsupported view, layout or size raises ``ValueError``, and a geometry
    without a registered program raises ``NotImplementedError``."""
    return deterministic_backend().arch_for(device) is not None


def _weights_kw(weights_static: bool) -> dict[str, Any]:
    """``weights_static`` as the keyword the deterministic entry point receives: passed only when set."""
    return {"weights_static": True} if weights_static else {}


def _pdl_kw(enable_pdl: Optional[bool]) -> dict[str, Any]:
    """``enable_pdl`` as the keyword a deterministic entry point receives: passed only when the caller set it, so the
    default is the deterministic package's (the device's support) and a call without it is the same call."""
    return {} if enable_pdl is None else {"enable_pdl": enable_pdl}


def _splits_kw(splits: Optional[int]) -> dict[str, Any]:
    """The ``splits`` keyword for a deterministic call: absent when the caller gave none, so the deterministic
    package's own split count of the row (its ``router_splits``) applies."""
    return {} if splits is None else {"splits": splits}


def router_symbol(a_mn: bool, b_mn: bool, out_t: bool) -> str:
    """Kernel symbol / registry template of this package's router stream-K program for the layout class of
    ``(a_mn, b_mn)``: ``router_fp32_gemm_atomic_<a><b>[_t]`` for the classes this package reduces with it (``kk``,
    ``nn_t``).  Takes no mode keyword.  ``kn`` has no stream-K program (its SIMT programs are named per tile,
    :func:`router_simt_template`) and the deterministic planner refuses ``nk``: ``ValueError``."""
    layout = _k2_class(a_mn, b_mn)
    if layout not in ("k2_kk", "k2_nn_t"):
        raise ValueError(
            f"router_symbol: the layout class {layout[3:]!r} has no stream-K program of this package (the input "
            "gradient's SIMT programs are router_simt_template(tile, 'kn'))"
        )
    return (
        cake_jit.TEMPLATE_PREFIXES[1]
        + ("n" if a_mn else "k")
        + ("n" if b_mn else "k")
        + ("_t" if out_t else "")
    )


class _Unset:
    """The default of a planner knob the caller did not give: such a knob is not passed on (:func:`_given_knobs`)."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<unset>"


#: The default of every planner knob of :func:`cake_dense_gemm_nondet_train_projection_plan` and
#: :func:`cake_dense_gemm_nondet_train_projection_prepare`.
_UNSET: Any = _Unset()


def _given_knobs(values: dict[str, Any]) -> dict[str, Any]:
    """The knobs of an entry point's call that the caller gave (``values``: the entry point's ``locals()`` on entry,
    without its tensors): only these reach the deterministic planner, so it receives the same keywords as a direct call
    of it, and its own defaults apply to the others."""
    return {
        name: value
        for name, value in values.items()
        if value is not _UNSET and name not in ("A", "B", "out")
    }


#: The program tables whose rows run a program of this package (the other tables of :func:`_k1_table_call` name a
#: program of the deterministic package).
_K1_OWN_PROGRAM_TABLES = (
    "K1_FORWARD_PROGRAM_RULES_BY_SM",
    "K1_DGRAD_PROGRAM_RULES_BY_SM",
    "K1_DGRAD_NVJET_RULES_BY_SM",
    "K1_WGRAD_SK_RULES_BY_SM",
)


def _k1_table_call(
    A: torch.Tensor, B: torch.Tensor, out: torch.Tensor, transposed_out: bool
) -> Optional[tuple[str, tuple, Callable[[Optional[bool]], torch.Tensor]]]:
    """``(table, key, call)``: the program table of the device that routes the K1 call ``out = A @ B`` of the generic
    entry points (2-D views), the row's key in it and the convenience entry point of its operation that runs it with a
    PDL flag; or ``None``.  ``A`` and ``B`` K-major into ``out`` -- the forward ``(X, W) = (A, B.T)`` -- on a row of
    :data:`K1_FORWARD_PROGRAM_RULES_BY_SM` or :data:`K1_FORWARD_RULES_BY_SM`; ``A`` K-major and ``B`` MN-major into
    ``out`` -- the input gradient ``(G, W) = (A, B)`` -- on a row of :data:`K1_DGRAD_PROGRAM_RULES_BY_SM`,
    :data:`K1_DGRAD_NVJET_RULES_BY_SM`, :data:`K1_DGRAD_RULES_BY_SM` or :data:`K1_DGRAD_PROMO_RANGES_BY_SM`; both MN-major into the transposed ``out`` --
    the weight gradient ``(G, X) = (B, A.T)`` -- on a row of :data:`K1_WGRAD_SK_RULES_BY_SM`.  The call runs the
    program and gives the bits of :func:`cake_dense_gemm_nondet_train_projection_forward` / ``_dgrad`` / ``_wgrad``.
    The device is read only for a shape of a table."""
    if A.dim() != 2 or B.dim() != 2 or out.dim() != 2:
        return None
    layout = _k1_layout(A, B)
    if layout == "k1_kk" and not transposed_out:
        X, W = A, B.t()
        key: tuple = (int(X.shape[1]), int(W.shape[0]), int(X.shape[0]))
        if _k1_forward_program_of(X, W, out) is not None:
            table = "K1_FORWARD_PROGRAM_RULES_BY_SM"
        elif _k1_forward_knobs_of(X, W, out):
            table = "K1_FORWARD_RULES_BY_SM"
        else:
            return None
        return (
            table,
            key,
            lambda pdl: cake_dense_gemm_nondet_train_projection_forward(
                X, W, out, enable_pdl=pdl
            ),
        )
    if layout == "k1_kn" and not transposed_out:
        G, W = A, B
        dt = "f32" if out.dtype == torch.float32 else "bf16"
        key = (int(W.shape[1]), int(W.shape[0]), int(G.shape[0]), dt)
        if _k1_dgrad_program_of(G, W, out) is not None:
            table = "K1_DGRAD_PROGRAM_RULES_BY_SM"
        elif _k1_dgrad_nvjet_of(G, W, out) is not None:
            table = "K1_DGRAD_NVJET_RULES_BY_SM"
        elif _k1_dgrad_knobs_of(G, W, out):
            facts = deterministic_backend().device_facts(out.device)
            rules = K1_DGRAD_RULES_BY_SM.get((facts.arch, int(facts.sm_count)), {})
            table = (
                "K1_DGRAD_RULES_BY_SM"
                if key in rules
                else "K1_DGRAD_PROMO_RANGES_BY_SM"
            )
        else:
            return None
        return (
            table,
            key,
            lambda pdl: cake_dense_gemm_nondet_train_projection_dgrad(
                G, W, out, enable_pdl=pdl
            ),
        )
    if layout == "k1_nn" and transposed_out:
        G, X = B, A.t()
        if _k1_wgrad_sk_program_of(G, X, out, "auto", None) is None:
            return None
        return (
            "K1_WGRAD_SK_RULES_BY_SM",
            _k1_wgrad_key(G, X, out),
            lambda pdl: cake_dense_gemm_nondet_train_projection_wgrad(
                G, X, out, enable_pdl=pdl
            ),
        )
    return None


def _k1_wgrad_key(G: torch.Tensor, X: torch.Tensor, out: torch.Tensor) -> tuple:
    """``(d_in, d_out, T, out_dtype)``: the key of the weight-gradient call ``(G, X, out)`` in its tables."""
    dt = "f32" if out.dtype == torch.float32 else "bf16"
    return (int(X.shape[1]), int(G.shape[1]), int(G.shape[0]), dt)


def _k1_program_launch(
    table: str, key: tuple, out: torch.Tensor, launch: Callable[[], Any]
) -> CakeDenseGemmNondetProgramLaunch:
    """The prepared launch ``launch`` of the row ``key`` of the program table named ``table`` on ``out``'s device, with
    the template the row names: this package's, or the deterministic package's for a program symbol of its tables
    (``None`` for the knobs of a row of :data:`K1_FORWARD_RULES_BY_SM` that name no program and for a row of
    :data:`K1_DGRAD_PROMO_RANGES_BY_SM`)."""
    det = deterministic_backend()
    facts = det.device_facts(out.device)
    rows = globals()[table].get((facts.arch, int(facts.sm_count)), {})
    template, own = rows.get(key), table in _K1_OWN_PROGRAM_TABLES
    if not own:
        template = (
            det._TEMPLATE_PREFIX + template if isinstance(template, str) else None
        )
    return CakeDenseGemmNondetProgramLaunch(
        table=table, template=template, own=own, out=out, _launch=launch
    )


def _k1_program_prepare(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    transposed_out: bool,
    enable_pdl: Optional[bool],
) -> Optional[CakeDenseGemmNondetProgramLaunch]:
    """The prepared launch of a generic K1 call on a row of a program table of the device (:func:`_k1_table_call`),
    else ``None``."""
    call = _k1_table_call(A, B, out, transposed_out)
    if call is None:
        return None
    table, key, run = call
    return _k1_program_launch(table, key, out, lambda: run(enable_pdl))


def _wgrad_det_prepare(
    G: torch.Tensor,
    X: torch.Tensor,
    out: torch.Tensor,
    sk: bool | str,
    cta_rows: Optional[int],
    enable_pdl: Optional[bool],
) -> Any:
    """The prepared launch of a weight-gradient row this package's planner keeps on the deterministic program: on a
    row of :data:`K1_WGRAD_DET_RULES_BY_SM` for the device (the default ``sk`` and ``cta_rows``) the call of
    :func:`cake_dense_gemm_nondet_train_projection_wgrad`, which runs the table's program; else the deterministic
    package's ``cake_dense_gemm_det_train_projection_wgrad_prepare``."""
    if _k1_wgrad_det_row(G, X, out, sk, cta_rows):
        return _k1_program_launch(
            "K1_WGRAD_DET_RULES_BY_SM",
            _k1_wgrad_key(G, X, out),
            out,
            lambda: cake_dense_gemm_nondet_train_projection_wgrad(
                G, X, out, sk=sk, cta_rows=cta_rows, enable_pdl=enable_pdl
            ),
        )
    return deterministic_backend().cake_dense_gemm_det_train_projection_wgrad_prepare(
        G, X, out, sk=sk, cta_rows=cta_rows, **_pdl_kw(enable_pdl)
    )


def cake_dense_gemm_nondet_train_projection_plan(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    *,
    splits: Optional[int] = _UNSET,
    chains: Optional[int] = _UNSET,
    sm_count: int = _UNSET,
    l2_bytes: int = _UNSET,
    transposed_out: bool = _UNSET,
    sk: bool | str = _UNSET,
    block_n: Optional[int] = _UNSET,
    stages: Optional[int] = _UNSET,
    epi: Optional[str] = _UNSET,
    slots: Optional[int] = _UNSET,
    cta_rows: Optional[int] = _UNSET,
    pf: Optional[int] = _UNSET,
    sk_max_units: Optional[int] = _UNSET,
    sk_parts: Optional[int] = _UNSET,
    promo: Optional[str] = _UNSET,
    hints: Optional[tuple] = _UNSET,
    group_m: Optional[int] = _UNSET,
    f32_v8: Optional[bool] = _UNSET,
    quad_store: Optional[bool] = _UNSET,
    park: Optional[bool] = _UNSET,
    ovl: Optional[bool] = _UNSET,
    htail: Optional[bool] = _UNSET,
    b_swz: Optional[int] = _UNSET,
    sk_exact: Optional[int] = _UNSET,
    batch_group: Optional[int] = _UNSET,
    store_ef: Optional[bool] = _UNSET,
    store_hint: Optional[str] = _UNSET,
    pd: Optional[int] = _UNSET,
    sh: Optional[int] = _UNSET,
    sk_sync: Optional[bool] = _UNSET,
    sk_sync_m: Optional[int] = _UNSET,
    sk_slab: Optional[int] = _UNSET,
    cta1: Optional[bool] = _UNSET,
    cgrp: Optional[bool] = _UNSET,
    ovr: Optional[bool] = _UNSET,
    stt: Optional[bool] = _UNSET,
    pro: Optional[int] = _UNSET,
    sk_first: Optional[bool] = _UNSET,
    lw: Optional[bool] = _UNSET,
    mg: Optional[bool] = _UNSET,
    sw: Optional[int] = _UNSET,
    form: Optional[str] = _UNSET,
    arch: str = _UNSET,
) -> Any:
    """Validate the views and plan the K1 launch: a weight-gradient layout (both operands MN-major) is planned for
    this package's split-K program (:class:`CakeDenseGemmNondetProjectionPlan`; ``splits``, ``block_n`` and ``chains``
    are this package's knobs).  A row the planner keeps on the deterministic program, and every other layout, returns
    the deterministic package's plan (``cake_dense_gemm_det_train_projection_plan``, a
    ``CakeDenseGemmDetProjectionPlan``).  The other keyword-only knobs are the deterministic planner's public ones
    (``sm_count`` and ``l2_bytes`` included): a knob the caller gives reaches that planner as given, and
    ``transposed_out``, ``arch`` and ``sm_count`` also apply to this package's plan; a knob the caller does not give
    takes the deterministic planner's default."""
    # must stay the first statement: ``locals()`` holds only the parameters here
    knobs = _given_knobs(locals())
    if _route(_k1_layout(A, B)) == "det":
        return deterministic_backend().cake_dense_gemm_det_train_projection_plan(
            A, B, out, **knobs
        )
    own, det_knobs = _split_knobs(knobs)
    return _atomic_k1(
        "plan",
        A,
        B,
        out,
        transposed_out=bool(det_knobs.get("transposed_out", False)),
        own=own,
        arch=det_knobs.get("arch"),
        sm_count=det_knobs.get("sm_count"),
        det_call=lambda: deterministic_backend().cake_dense_gemm_det_train_projection_plan(
            A, B, out, **det_knobs
        ),
    )


def cake_dense_gemm_nondet_train_projection_prepare(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    *,
    splits: Optional[int] = _UNSET,
    chains: Optional[int] = _UNSET,
    transposed_out: bool = _UNSET,
    sk: bool | str = _UNSET,
    block_n: Optional[int] = _UNSET,
    stages: Optional[int] = _UNSET,
    epi: Optional[str] = _UNSET,
    slots: Optional[int] = _UNSET,
    cta_rows: Optional[int] = _UNSET,
    pf: Optional[int] = _UNSET,
    sk_max_units: Optional[int] = _UNSET,
    sk_parts: Optional[int] = _UNSET,
    promo: Optional[str] = _UNSET,
    hints: Optional[tuple] = _UNSET,
    group_m: Optional[int] = _UNSET,
    f32_v8: Optional[bool] = _UNSET,
    quad_store: Optional[bool] = _UNSET,
    park: Optional[bool] = _UNSET,
    ovl: Optional[bool] = _UNSET,
    htail: Optional[bool] = _UNSET,
    b_swz: Optional[int] = _UNSET,
    sk_exact: Optional[int] = _UNSET,
    batch_group: Optional[int] = _UNSET,
    store_ef: Optional[bool] = _UNSET,
    store_hint: Optional[str] = _UNSET,
    pd: Optional[int] = _UNSET,
    sh: Optional[int] = _UNSET,
    sk_sync: Optional[bool] = _UNSET,
    sk_sync_m: Optional[int] = _UNSET,
    sk_slab: Optional[int] = _UNSET,
    cta1: Optional[bool] = _UNSET,
    cgrp: Optional[bool] = _UNSET,
    ovr: Optional[bool] = _UNSET,
    stt: Optional[bool] = _UNSET,
    pro: Optional[int] = _UNSET,
    form: Optional[str] = _UNSET,
    enable_pdl: Optional[bool] = _UNSET,
) -> Any:
    """Validate one K1 binding, plan it for the device and prepare its allocation-free launch: a weight-gradient
    layout prepares this package's split-K program (:class:`CakeDenseGemmNondetProjectionLaunch`; its workspace is
    owned by the prepared object; ``splits``, ``block_n`` and ``chains`` as for
    :func:`cake_dense_gemm_nondet_train_projection_plan`).  Without other knobs than ``transposed_out`` and
    ``enable_pdl``, a row of a program table of the device -- the forward, input-gradient and stream-K
    weight-gradient tables (:func:`_k1_table_call`) -- prepares the program the convenience entry point of its
    operation runs (:class:`CakeDenseGemmNondetProgramLaunch`, the same program and bits).  A row the planner keeps on
    the deterministic program, and every other layout, prepares the deterministic package's
    ``cake_dense_gemm_det_train_projection_prepare`` (a ``CakeDenseGemmDetProjectionLaunch``).  The other keyword-only knobs are that entry point's public ones, passed
    on only when the caller gives them; ``enable_pdl`` (default: the device supports it) makes either launch a
    programmatic dependent launch."""
    # must stay the first statement: ``locals()`` holds only the parameters here
    knobs = _given_knobs(locals())
    if not set(knobs) - {"transposed_out", "enable_pdl"}:
        program = _k1_program_prepare(
            A,
            B,
            out,
            bool(knobs.get("transposed_out", False)),
            knobs.get("enable_pdl"),
        )
        if program is not None:
            return program
    if _route(_k1_layout(A, B)) == "det":
        return deterministic_backend().cake_dense_gemm_det_train_projection_prepare(
            A, B, out, **knobs
        )
    own, det_knobs = _split_knobs(knobs)
    return _atomic_k1(
        "prepare",
        A,
        B,
        out,
        transposed_out=bool(det_knobs.get("transposed_out", False)),
        own=own,
        det_call=lambda: deterministic_backend().cake_dense_gemm_det_train_projection_prepare(
            A, B, out, **det_knobs
        ),
        enable_pdl=det_knobs.get("enable_pdl"),
    )


def cake_dense_gemm_nondet_train_projection(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    *,
    transposed_out: bool = False,
    enable_pdl: Optional[bool] = None,
    weights_static: bool = False,
) -> torch.Tensor:
    """``out[l, m, n] = sum_k A[l, m, k] B[l, k, n]`` for strided views (the cached convenience path; see the
    deterministic package for the view contract).  A row of a program table of the device -- the forward,
    input-gradient and stream-K weight-gradient tables (:func:`_k1_table_call`) -- runs the program and gives the bits
    of the convenience entry point of its operation; a weight-gradient layout (both operands MN-major) runs this
    package's split-K program where the planner takes the row; every other layout and row runs the deterministic
    program.  ``enable_pdl`` (default: the device supports it)
    launches either program as a programmatic dependent launch; each waits for the preceding kernel before its first
    global access, and ``weights_static=True`` lets the deterministic swapped small-token forward start its loads of
    ``B`` before the wait (see the deterministic package's ``cake_dense_gemm_det_train_projection``)."""

    def det_call() -> Any:
        return deterministic_backend().cake_dense_gemm_det_train_projection(
            A,
            B,
            out,
            transposed_out=transposed_out,
            enable_pdl=enable_pdl,
            **_weights_kw(weights_static),
        )

    def det_args() -> tuple:
        return A, B, out, transposed_out, ()

    table = _k1_table_call(A, B, out, transposed_out)
    if table is not None:
        # a row of a program table: the program of its operation's entry point
        return table[2](enable_pdl)
    if _route(_k1_layout(A, B)) == "det":
        # the deterministic program through this package's binding (its plan, program and bits)
        return _det_k1(
            (A, B, out),
            ("gemm", bool(transposed_out)),
            det_args,
            det_call,
            enable_pdl,
            weights_static,
        )
    return _atomic_k1(
        "launch",
        A,
        B,
        out,
        transposed_out=transposed_out,
        own={},
        det_call=det_call,
        det_args=det_args,
        enable_pdl=enable_pdl,
        weights_static=weights_static,
    )


def cake_dense_gemm_nondet_train_projection_forward(
    X: torch.Tensor,
    W: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    """``out[T, N] = X[T, K] @ W[N, K].T`` (bf16 out by default).  The deterministic package's programs (a
    recomputed forward must reproduce its result; where the deterministic plan splits K across CTAs, it adds the
    partials in a fixed order) through this package's binding of them (:func:`_det_k1`): the deterministic plan,
    program and bits, except on a row of :data:`K1_FORWARD_PROGRAM_RULES_BY_SM` for the device, which runs this
    package's one-wave forward program (the same bits), and on a row of :data:`K1_FORWARD_RULES_BY_SM` for the device,
    which runs the program the deterministic planner gives it with the table's knobs (:func:`k1_forward_knobs`; other
    bits, still the same on every run).  ``enable_pdl`` as for :func:`cake_dense_gemm_nondet_train_projection`."""
    if out is None:
        out = torch.empty(X.shape[0], W.shape[0], device=X.device, dtype=torch.bfloat16)
    o = out
    if _k1_cached_launch(
        _K1_DET_CACHE,
        _K1_FWD_HEAD,
        _K1_NOT_STATIC,
        (X, W, o),
        enable_pdl,
        _K1_PROGRAM_CACHE,
    ):
        return o
    module = _k1_forward_program_of(X, W, o)
    if module is not None:
        return _launch_wave_forward(module, X, W, o, enable_pdl, _K1_FWD_HEAD)
    return _det_k1(
        (X, W, o),
        ("fwd",),
        lambda: (X, W.t(), o, False, _k1_forward_knobs_of(X, W, o)),
        lambda: deterministic_backend().cake_dense_gemm_det_train_projection_forward(
            X, W, o, **_pdl_kw(enable_pdl)
        ),
        enable_pdl,
    )


def cake_dense_gemm_nondet_train_projection_dgrad(
    G: torch.Tensor,
    W: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    out_dtype=None,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    """``out[T, K] = G[T, N] @ W[N, K]``.  The deterministic program (whole output tiles) through this package's
    binding of it (:func:`_det_k1`: the same plan, program and bits), except on a row of
    :data:`K1_DGRAD_PROGRAM_RULES_BY_SM` for the device, which runs this package's input-gradient program (the same
    bits), on a row of :data:`K1_DGRAD_NVJET_RULES_BY_SM`, which runs this package's nvjet program on the deterministic
    package's plan of :data:`K1_DGRAD_NVJET_DET_PROGRAM` (the same bits), and on a row of
    :data:`K1_DGRAD_RULES_BY_SM` for the device, which runs the table's program of the deterministic package, and in a
    range of :data:`K1_DGRAD_PROMO_RANGES_BY_SM`, which runs the deterministic plan with the range's TMA L2 promotion
    (:func:`k1_dgrad_knobs`; the same bits as the deterministic plan, every run).  ``enable_pdl`` as for
    :func:`cake_dense_gemm_nondet_train_projection`."""
    if out is None:
        out = torch.empty(
            G.shape[0],
            W.shape[1],
            device=G.device,
            dtype=out_dtype or torch.bfloat16,
        )
    o = out
    if _k1_cached_launch(
        _K1_DET_CACHE,
        _K1_DGRAD_HEAD,
        _K1_NOT_STATIC,
        (G, W, o),
        enable_pdl,
        _K1_PROGRAM_CACHE,
    ):
        return o
    module = _k1_dgrad_program_of(G, W, o)
    if module is not None:
        return _launch_dgrad_program(module, G, W, o, enable_pdl, _K1_DGRAD_HEAD)
    nvjet = _k1_dgrad_nvjet_of(G, W, o)
    return _det_k1(
        (G, W, o),
        ("dgrad",),
        lambda: (
            G,
            W,
            o,
            False,
            k1_program_knobs(K1_DGRAD_NVJET_DET_PROGRAM)
            if nvjet
            else _k1_dgrad_knobs_of(G, W, o),
        ),
        lambda: deterministic_backend().cake_dense_gemm_det_train_projection_dgrad(
            G, W, o, out_dtype=out_dtype, **_pdl_kw(enable_pdl)
        ),
        enable_pdl,
        module=nvjet,
    )


def cake_dense_gemm_nondet_train_projection_wgrad_prepare(
    G: torch.Tensor,
    X: torch.Tensor,
    out: torch.Tensor,
    *,
    sk: bool | str = "auto",
    cta_rows: Optional[int] = None,
    enable_pdl: Optional[bool] = None,
) -> Any:
    """Prepared ``out[N, K] = G[T, N].T @ X[T, K]``, the program of
    :func:`cake_dense_gemm_nondet_train_projection_wgrad`: on a row of :data:`K1_WGRAD_SK_RULES_BY_SM` for the device
    this package's stream-K program (:class:`CakeDenseGemmNondetProgramLaunch`), else this package's split-K program as
    the transposed-store GEMM ``X.T @ G`` (``M = K``, ``N = N``) where the planner takes the row; a row it does not
    take runs the program of its row of :data:`K1_WGRAD_DET_RULES_BY_SM` (a
    :class:`CakeDenseGemmNondetProgramLaunch`) or prepares the deterministic package's
    ``cake_dense_gemm_det_train_projection_wgrad_prepare`` (its orientation, ``sk``, ``cta_rows``).  ``enable_pdl``
    as for :func:`cake_dense_gemm_nondet_train_projection_prepare`."""
    if _k1_wgrad_sk_program_of(G, X, out, sk, cta_rows) is not None:
        return _k1_program_launch(
            "K1_WGRAD_SK_RULES_BY_SM",
            _k1_wgrad_key(G, X, out),
            out,
            lambda: cake_dense_gemm_nondet_train_projection_wgrad(
                G, X, out, enable_pdl=enable_pdl
            ),
        )
    return _atomic_k1(
        "prepare",
        X.t(),
        G,
        out,
        transposed_out=True,
        own={},
        det_call=lambda: _wgrad_det_prepare(G, X, out, sk, cta_rows, enable_pdl),
        enable_pdl=enable_pdl,
    )


def cake_dense_gemm_nondet_train_projection_wgrad(
    G: torch.Tensor,
    X: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    out_dtype=None,
    sk: bool | str = "auto",
    cta_rows: Optional[int] = None,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    """``out[N, K] = G[T, N].T @ X[T, K]`` (ragged reduction length ``T``): on a row of
    :data:`K1_WGRAD_SK_RULES_BY_SM` for the device this package's stream-K program (a split tile summed by one fixed
    reducer in unit order), else this
    package's split-K program where the planner takes the row (the split partials summed in the order they finish,
    the output rounded once), else the deterministic package's ``cake_dense_gemm_det_train_projection_wgrad`` program (its orientation,
    ``sk`` and ``cta_rows``) through this package's binding of it (:func:`_launch_cached_atomic_k1`); ``enable_pdl``
    as for :func:`cake_dense_gemm_nondet_train_projection`."""
    if out is None:
        out = torch.empty(
            int(G.shape[1]),
            int(X.shape[1]),
            device=G.device,
            dtype=out_dtype or torch.bfloat16,
        )
    A = X.t()
    # the key flags of _launch_cached_atomic_k1 for this call: ("k1", transposed_out, own items, det_knobs) + pdl +
    # (weights_static,)
    if _k1_cached_launch(
        _K1_ATOMIC_CACHE,
        ("k1", True, (), (sk, cta_rows)),
        _K1_NOT_STATIC,
        (A, G, out),
        enable_pdl,
    ):
        return out
    module = _k1_wgrad_sk_program_of(G, X, out, sk, cta_rows)
    if module is not None:
        return _launch_wgrad_sk(module, G, X, out, enable_pdl)
    return _atomic_k1(
        "launch",
        A,
        G,
        out,
        transposed_out=True,
        own={},
        det_call=lambda: deterministic_backend().cake_dense_gemm_det_train_projection_wgrad(
            G,
            X,
            out,
            out_dtype=out_dtype,
            sk=sk,
            cta_rows=cta_rows,
            **_pdl_kw(enable_pdl),
        ),
        det_args=lambda: _wgrad_det_args(G, X, out, sk, cta_rows),
        det_knobs=(sk, cta_rows),
        enable_pdl=enable_pdl,
    )


def _wgrad_det_args(
    G: torch.Tensor,
    X: torch.Tensor,
    out: torch.Tensor,
    sk: bool | str,
    cta_rows: Optional[int],
) -> tuple:
    """The deterministic package's convenience call of ``cake_dense_gemm_det_train_projection_wgrad(G, X, out, sk=sk,
    cta_rows=cta_rows)``: ``(A, B, out, transposed_out, knobs)`` in its orientation (``wgrad_views``); with the
    default ``sk`` and ``cta_rows``, a row of :data:`K1_WGRAD_DET_RULES_BY_SM` for the device takes the knobs of the
    table's program instead (:func:`k1_wgrad_det_knobs`)."""
    det = deterministic_backend()
    out_f32 = out.dtype == torch.float32
    A, B, transposed = det.wgrad_views(
        G, X, arch=det.arch_for(out.device), out_f32=out_f32
    )
    knobs = _k1_wgrad_det_row(G, X, out, sk, cta_rows) or (
        ("cta_rows", cta_rows),
        ("sk", sk),
    )
    return A, B, out, transposed, knobs


def _k1_wgrad_det_row(
    G: torch.Tensor,
    X: torch.Tensor,
    out: torch.Tensor,
    sk: bool | str,
    cta_rows: Optional[int],
) -> tuple:
    """:func:`k1_wgrad_det_knobs` of the weight-gradient call ``(G, X, out)`` with the default ``sk`` and
    ``cta_rows`` on ``out``'s device, else ``()``.  The device is read only for a shape of
    :data:`K1_WGRAD_DET_RULES_BY_SM`."""
    if sk != "auto" or cta_rows is not None or G.dim() != 2 or X.dim() != 2:
        return ()
    out_f32 = out.dtype == torch.float32
    T, d_out, d_in = int(G.shape[0]), int(G.shape[1]), int(X.shape[1])
    if not any(
        (d_in, d_out, T, "f32" if out_f32 else "bf16") in rows
        for rows in K1_WGRAD_DET_RULES_BY_SM.values()
    ):
        return ()
    facts = deterministic_backend().device_facts(out.device)
    if facts.arch is None:
        return ()
    return k1_wgrad_det_knobs(
        arch=facts.arch,
        sm_count=facts.sm_count,
        d_in=d_in,
        d_out=d_out,
        T=T,
        out_f32=out_f32,
    )


def cake_dense_gemm_nondet_train_router_plan(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    *,
    splits: Optional[int] = None,
) -> Any:
    """Classify the fp32 views and plan the K2 launch: this package's stream-K program
    for the ``kk`` / ``nn_t`` classes and the SIMT programs in their windows (:class:`CakeDenseGemmNondetRouterPlan`; ``splits``
    sets ``splits`` units per tile, see :func:`router_units`); a row the planner keeps on the deterministic program,
    and a layout class routed ``"det"``, returns the deterministic package's plan (``cake_dense_gemm_det_train_router_plan``)."""
    if _route(_k2_layout(A, B)) == "det":
        return deterministic_backend().cake_dense_gemm_det_train_router_plan(
            A, B, out, splits=splits
        )
    return _atomic_k2("plan", A, B, out, splits=splits)


def cake_dense_gemm_nondet_train_router_prepare(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    *,
    splits: Optional[int] = None,
    enable_pdl: Optional[bool] = None,
) -> Any:
    """Validate one fp32 binding, plan it and prepare its allocation-free launch: this
    package's stream-K program for the ``kk`` / ``nn_t`` classes (its launch splits the second operand, zeroes
    ``out`` on the stream when a segment adds into it, and runs the program: no partial buffers), and a SIMT program
    in its windows (the ``nn_t`` and ``kn`` classes: one kernel over the fp32 operands); a row the planner keeps on the
    deterministic program prepares the deterministic package's ``cake_dense_gemm_det_train_router_prepare``.  ``enable_pdl``
    (default: the device supports it) launches the split and the stream-K program as programmatic dependent
    launches."""
    if _route(_k2_layout(A, B)) == "det":
        return deterministic_backend().cake_dense_gemm_det_train_router_prepare(
            A, B, out, splits=splits, **_pdl_kw(enable_pdl)
        )
    return _atomic_k2("prepare", A, B, out, splits=splits, enable_pdl=enable_pdl)


def _router_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    splits: Optional[int],
    enable_pdl: Optional[bool] = None,
    views: Optional[tuple] = None,
) -> torch.Tensor:
    """K2 on the convenience path: a layout class routed ``"det"`` runs the deterministic program through this
    package's binding (:func:`_splitk_k2`), the others this package's K2 path (``views``: an entry point's own launch
    views, :func:`_launch_views`).  ``enable_pdl`` reaches every route."""
    if _route(_k2_layout(A, B)) == "det":
        return _splitk_k2(A, B, out, splits, views, enable_pdl)
    return _atomic_k2(
        "launch", A, B, out, splits=splits, enable_pdl=enable_pdl, views=views
    )


def cake_dense_gemm_nondet_train_router(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    *,
    splits: Optional[int] = None,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    """``out[m, n] = sum_k A[m, k] B[k, n]`` for fp32 views through the BF16x3 emulation (the cached convenience
    path): the ``kk`` / ``nn_t`` rows the planner takes (:func:`router_units`) on this
    package's stream-K program, whose part tiles add into ``out`` in the order they finish, instead of the
    fixed-order split-K sum, the ``nn_t`` / ``kn`` token counts of a SIMT window on its exact-FP32 program
    (:func:`router_simt`) and the ``kk`` token counts of a forward SIMT window on the forward's cluster program
    (:func:`router_forward_simt`); every other row runs the deterministic program.  ``enable_pdl`` as for
    :func:`cake_dense_gemm_nondet_train_router_prepare`."""
    return _router_gemm(A, B, out, splits, enable_pdl)


def cake_dense_gemm_nondet_train_router_forward(
    X: torch.Tensor,
    W: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    splits: Optional[int] = None,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    """``out[T, N] = X[T, K] @ W[N, K].T`` in FP32 through the emulation.  A row runs
    this package's program only with an explicit ``splits`` (at most two partials per element: the logits repeat bit
    for bit) or a verdict of :data:`ROUTER_RULES` (the token windows of the forward's SIMT program,
    :func:`router_forward_simt`), else the deterministic program (its plan for ``cake_dense_gemm_nondet_train_router(X, W.T, out)``,
    through this package's binding, :func:`_launch_cached_splitk_k2`): without a split count, while no router table
    gives the forward a verdict, without the per-call check of the row."""
    if out is None:
        out = torch.empty(X.shape[0], W.shape[0], device=X.device, dtype=torch.float32)
    if _k2_entry("fwd", X, W, out, splits, enable_pdl) is not None:
        return out
    views = (X.unsqueeze(0), W, _FORWARD_VIEWS)
    if splits is None and not _router_memo().forward_verdicts:
        # no router table gives the forward a verdict: the deterministic program (the plan the planner would keep),
        # without the per-call check of the row
        return _splitk_k2(X, W.t(), out, None, views, enable_pdl)
    return _router_gemm(X, W.t(), out, splits, enable_pdl, views)


def cake_dense_gemm_nondet_train_router_dgrad(
    G: torch.Tensor,
    W: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    """``out[T, K] = G[T, N] @ W[N, K]`` in FP32: in the token windows of the row's
    input-gradient verdict in :data:`ROUTER_RULES` (or, on a part with a table of its own, :data:`ROUTER_RULES_BY_SM`)
    the exact-FP32 SIMT program (:func:`router_simt`: compensated sums of the products of ``G`` and ``W`` read in
    place, every element of ``out`` stored once), and at every other row and token count the deterministic program
    (one split: nothing to reduce) through this package's binding of it (:func:`_launch_cached_splitk_k2`: the same
    plan, program and bits).  ``enable_pdl`` as for :func:`cake_dense_gemm_nondet_train_router_prepare`."""
    if out is None:
        out = torch.empty(G.shape[0], W.shape[1], device=G.device, dtype=torch.float32)
    if _k2_entry("dgrad", G, W, out, None, enable_pdl) is not None:
        return out
    return _router_gemm(G, W, out, None, enable_pdl, (G.unsqueeze(0), W, _DGRAD_VIEWS))


def cake_dense_gemm_nondet_train_router_wgrad(
    G: torch.Tensor,
    X: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    splits: Optional[int] = None,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    """``out[N, K] = G[T, N].T @ X[T, K]`` in FP32: the swapped GEMM ``X.T @ G`` with a
    transposed store; ``splits`` defaults to the partition of the row's verdict in :data:`ROUTER_RULES` or, on a part
    with a table of its own, in :data:`ROUTER_RULES_BY_SM` (:func:`router_partition`; a row or token count without one
    runs the deterministic program at that package's split count of the row -- its ``router_splits``: six; nine at
    hidden 5120 and three at hidden 6144 from 16129 to 16384 tokens on 148- and 152-SM ``sm_100a`` / ``sm_103a``
    parts, its ``ROUTER_SPLIT_RULES`` -- through this package's binding of it without planning the row here,
    :func:`_launch_cached_splitk_k2`)."""
    if out is None:
        out = torch.empty(G.shape[1], X.shape[1], device=G.device, dtype=torch.float32)
    if _k2_entry("wgrad", G, X, out, splits, enable_pdl) is not None:
        return out
    # a row without a verdict (kept per device and geometry, :func:`_router_memo`) runs the deterministic program
    views = (X.unsqueeze(0), G, _WGRAD_VIEWS)
    return _router_gemm(G.t(), X, out, splits, enable_pdl, views)
