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

Cake backend with an explicit numerical mode: the GLM-5.2 dense projection GEMMs for training (K1) and the FP32
router-gate GEMM (K2) on SM100 / SM103 / SM107.

The deterministic implementation is the package ``dense_projection_det_gemm`` (flashinfer-ai/flashinfer#5677 and
#7043; named so by #8046): its stream-K and split-K reductions add the partials in a fixed order, so every result is
bitwise reproducible run to run.  This package keeps that package's operations, view contract, dtypes, accuracy
budgets and host paths (prepared and convenience) and adds a second numerical mode whose split-K reductions add the
partials in the order they finish.  Only run-to-run bitwise reproducibility is relaxed: the products, the FP32
accumulation, the rounding points (an output is rounded once, from the complete FP32 sum), the masks, the layouts
and the dynamic shapes are the deterministic package's, and no reduction flushes a subnormal.

Numerical mode
--------------
Every entry point takes the keyword-only argument ``deterministic: bool = True``:

* ``deterministic=True`` (the default) calls the deterministic package's canonical object with the same arguments
  (:data:`DETERMINISTIC_NAMES`: ``projection_forward`` -> ``projection_det_forward``, ...): the same planner, the
  same programs, the same bits and the same errors.  The names this import path had before #8046 keep their behavior.
* ``deterministic=False`` runs the schedule of this package for an operation :data:`ROUTES` marks ``"atomic"``:

  - the K1 weight gradient (both operands MN-major, ``K = T`` ragged), launched as the transposed-store GEMM
    ``out[l, n, m]``: each of ``L x m_tiles x n_tiles`` tiles is cut into ``num_splits`` K ranges; every split stores
    its FP32 partial into its own workspace slot and takes a ticket, and the split that draws the last ticket adds
    the other ``num_splits - 1`` partials with IEEE round-to-nearest adds, rounds once and stores (:func:`k1_plan`
    chooses the tile width and the split count; a row it does not take runs the deterministic program).  Two splits
    add ``own + other``, one add of two values, so a two-split row repeats bit for bit; three or more add in the
    order the splits finish;
  - the K2 forward and weight gradient: a stream-K partition of the (tile, K block) steps into ``units`` spans
    (:func:`router_units`); a segment that covers its tile's whole K range stores the deterministic program's
    one-split bits, any other adds its partial into ``out`` with a TMA reduce-add (``out`` is zeroed on the launch's
    stream first).  The forward's partition gives every element at most two partials, ``0 + a + b == 0 + b + a``, so a
    forward recomputed under activation checkpointing gives the same logits and the same top-k.

  Every other operation -- the K1 forward and input gradient, the K2 input gradient, the BF16x3 split -- runs the
  deterministic programs in both modes: it has no cross-CTA reduction to relax.
* With ``torch.use_deterministic_algorithms(True)`` an explicit ``deterministic=False`` raises ``RuntimeError``; with
  ``warn_only=True`` it warns and runs the deterministic path (:func:`_resolve_deterministic`).  The default does not
  consult the PyTorch setting: it is deterministic already.

The two packages
----------------
This package registers only the programs of its own schedule (``cake_jit``: programs
``cake_dense_projection_gemm_<hex>``, templates ``dense_proj_gemm_atomic_*`` / ``router_fp32_gemm_atomic_*``); every
deterministic program, and every planner rule of the deterministic package, is that package's (resolved as this
package's sibling on first use, never through an absolute ``flashinfer`` import; a vendored copy binds it with
:func:`use_deterministic_backend`).  The deterministic package never imports this one.  The module-level names this
path had before #8046 that are not entry points (``ROW_RULES``, ``stream_k_plan``, ``wgrad_views``, the plan and
prepared types ``GemmPlan`` / ``PreparedGemm`` / ``RouterPlan`` / ``PreparedRouterGemm``, ...) read through to the
deterministic package (module ``__getattr__``); assigning one here does not change the deterministic package's.
:func:`clear_convenience_caches` clears the convenience caches and workspace arenas of both packages.
"""

from __future__ import annotations

import re
import threading
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import cached_property
from types import ModuleType
from typing import Any, Callable, Optional, Union

import torch
import tvm_ffi

from . import cake_jit

# The registry names this module had before #8046, re-exported from this package's own registry.
from .cake_jit import KERNELS as KERNELS
from .cake_jit import MODULES as MODULES
from .cake_jit import TRACKING_ISSUE as TRACKING_ISSUE
from .cake_jit import arg_plan_of as arg_plan_of
from .cake_jit import (
    load_cake_dense_projection_gemm_module as load_cake_dense_projection_gemm_module,
)
from .cake_jit import select_module as select_module

# ---------------------------------------------------------------------------
# The deterministic package and the numerical mode
# ---------------------------------------------------------------------------

#: Each name of this module's entry points and types -> its canonical object in the deterministic package's
#: ``cake_backend`` (the former names #8046 kept there as aliases).  ``deterministic=True`` calls that object.
DETERMINISTIC_NAMES: dict[str, str] = {
    "GemmPlan": "DetGemmPlan",
    "PreparedGemm": "PreparedDetGemm",
    "PreparedRouterGemm": "PreparedRouterDetGemm",
    "RouterPlan": "RouterDetPlan",
    "dense_projection_gemm": "dense_projection_det_gemm",
    "plan_dense_projection_gemm": "plan_dense_projection_det_gemm",
    "plan_router_fp32_gemm": "plan_router_fp32_det_gemm",
    "prepare_dense_projection_gemm": "prepare_dense_projection_det_gemm",
    "prepare_projection_wgrad": "prepare_projection_det_wgrad",
    "prepare_router_fp32_gemm": "prepare_router_fp32_det_gemm",
    "projection_dgrad": "projection_det_dgrad",
    "projection_forward": "projection_det_forward",
    "projection_wgrad": "projection_det_wgrad",
    "router_dgrad": "router_det_dgrad",
    "router_forward": "router_det_forward",
    "router_fp32_gemm": "router_fp32_det_gemm",
    "router_wgrad": "router_det_wgrad",
    "split_fp32_to_bf16x3_": "split_fp32_to_bf16x3_det_",
    "instance_symbol": "instance_symbol",
    "router_symbol": "router_symbol",
}
#: The types among :data:`DETERMINISTIC_NAMES`: read through to the deterministic package's classes.
_DET_TYPES = ("GemmPlan", "PreparedGemm", "PreparedRouterGemm", "RouterPlan")
#: Names spelling the ``det`` token live in the deterministic package only; this module does not read them through.
_DET_SPELLED = re.compile(r"(?:^|_)det(?:_|$)|Det")


def use_deterministic_backend(module: ModuleType) -> None:
    """Bind the deterministic package's ``cake_backend`` module explicitly.  In a FlashInfer tree the deterministic
    package is this package's sibling and is resolved on first use; a vendored copy of this package (imported
    under another name, without the ``flashinfer`` package around it) vendors the deterministic package too and
    binds its ``cake_backend`` here before the first call that needs it."""
    missing = [
        name for key, name in DETERMINISTIC_NAMES.items() if not hasattr(module, name)
    ]
    if missing:
        raise TypeError(
            f"{getattr(module, '__name__', module)!r} is not the deterministic dense projection GEMM backend: "
            f"it lacks {missing}"
        )
    cake_jit.bind_deterministic_package(module.__name__.rpartition(".")[0])


def deterministic_backend() -> ModuleType:
    """The deterministic package's ``cake_backend`` module (see :func:`use_deterministic_backend`)."""
    return cake_jit.deterministic_module("cake_backend")


def _det_entry(name: str) -> Callable[..., Any]:
    return getattr(deterministic_backend(), DETERMINISTIC_NAMES[name])


def _resolve_deterministic(deterministic: bool, api: str) -> bool:
    """Whether a call of ``api`` runs the deterministic path.

    ``deterministic=True`` does.  ``deterministic=False`` runs this package's schedule unless
    ``torch.use_deterministic_algorithms(True)`` is in effect: then the call raises ``RuntimeError``, or, with
    ``warn_only=True``, warns (``UserWarning``) and runs the deterministic path.  A call that asked for determinism
    never runs a program of this package.
    """
    if deterministic:
        return True
    if not torch.are_deterministic_algorithms_enabled():
        return False
    message = (
        f"{__name__}.{api}(deterministic=False) adds split-K partials in the order they finish, but "
        "torch.use_deterministic_algorithms(True) is in effect; call it with deterministic=True or use "
        f"{cake_jit.DETERMINISTIC_PACKAGE}.cake_backend.{DETERMINISTIC_NAMES.get(api, api)}"
    )
    if torch.is_deterministic_algorithms_warn_only_enabled():
        warnings.warn(
            f"{message}; running the deterministic path", UserWarning, stacklevel=3
        )
        return True
    raise RuntimeError(message)


# ---------------------------------------------------------------------------
# Routing of the deterministic=False mode
# ---------------------------------------------------------------------------

#: The program family each operation runs with ``deterministic=False``: ``"atomic"`` -- this package's planner and
#: programs (the planner may still keep a row on the deterministic program, see :func:`k1_plan` /
#: :func:`router_units`) -- or ``"det"`` -- the deterministic package's programs in both modes.  K1 operations are
#: keyed by the operand layouts of the launched GEMM (``kk`` forward, ``kn`` input gradient, ``nn`` weight gradient;
#: ``nk`` has no training use), K2 by the layout class of ``router_layout_class``.
ROUTES: dict[str, str] = {
    "k1_kk": "det",  # forward: no cross-CTA reduction (whole output tiles); feeds the recompute of checkpointing
    "k1_kn": "det",  # input gradient: whole output tiles
    "k1_nk": "det",
    "k1_nn": "atomic",  # weight gradient: split-K over the ragged token count
    "k2_kk": "atomic",  # router forward: at most two partials per element (reproducible)
    "k2_kn": "det",  # router input gradient: one split
    "k2_nn_t": "atomic",  # router weight gradient: stream-K over the token count
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
K1_BLOCK_K = 64  # K elements per K1 K block
K1_COUNTER_WORDS = 512  # ticket words per K1 tile: 16 epilogue warp slices x 32 lanes
K1_BLOCK_N_CHOICES = (64, 128, 256)

# ---------------------------------------------------------------------------
# K1 planner: the rule table and the per-architecture planners
# ---------------------------------------------------------------------------
#
# A K1 launch of this package is the transposed-store GEMM ``out[l, n, m] = sum_k A[l, k, m] B[l, k, n]``: ``M`` is
# the inner (unit-stride) extent of the stored output and ``N`` its row count.  A weight gradient
# ``dW[d_out, d_in] = G[T, d_out].T @ X[T, d_in]`` has ``A = X``, ``B = G``, ``M = d_in``, ``N = d_out``, ``K = T``.

#: Per-row verdicts of the K1 weight gradient, ``(arch, L, M, N, out_dtype) -> (block_n, num_splits)`` (this
#: package's program with that tile width and split count) or ``-> None`` (the deterministic program); ``M`` / ``N``
#: as above, ``out_dtype`` ``"bf16"`` or ``"f32"``.  A verdict replaces the architecture's planner
#: (:data:`K1_PLANNERS`) for its row; a row without one takes the planner.  The table ships empty: the planners hold
#: the measured choices.
K1_SPLIT_RULES: dict[tuple[str, int, int, int, str], Optional[tuple[int, int]]] = {}
#: The fewest SMs a device of the architecture needs for this package's K1 programs (the verdicts and the planner);
#: a device with fewer runs every K1 row on the deterministic program.  The sm_107a table was measured on 216-SM parts.
K1_MIN_SM_COUNT: dict[str, int] = {"sm_107a": 216}
#: The sm_107a planner, ``(M, N, out_dtype) -> (block_n, num_splits)`` for ``L == 1``: the measured winners at
#: ``T`` near 16k.  The ``num_splits == 1`` rows store whole tiles (one rounding, deterministic): they win on the FP32
#: rows where the deterministic package's sm_107a program is slower.
K1_SM107A_TABLE: dict[tuple[int, int, str], tuple[int, int]] = {
    (4096, 32, "bf16"): (64, 4),
    (4096, 64, "bf16"): (64, 4),
    (4096, 128, "bf16"): (64, 3),
    (5120, 32, "bf16"): (64, 3),
    (6144, 32, "bf16"): (64, 3),
    (8192, 128, "bf16"): (128, 2),
    (4096, 512, "bf16"): (256, 3),
    (5120, 1024, "bf16"): (256, 1),
    (4096, 32, "f32"): (64, 3),
    (4096, 48, "f32"): (64, 3),
    (4096, 64, "f32"): (64, 3),
    (4096, 128, "f32"): (64, 3),
    (5120, 32, "f32"): (64, 3),
    (5120, 48, "f32"): (64, 3),
    (5120, 64, "f32"): (64, 3),
    (6144, 32, "f32"): (64, 3),
    (8192, 128, "f32"): (128, 2),
    (4096, 512, "f32"): (128, 1),
    (5120, 512, "f32"): (128, 1),
    (4096, 1024, "f32"): (256, 1),
    (5120, 1024, "f32"): (256, 1),
    (4096, 4096, "f32"): (128, 1),
}


def _k1_tiles(L: int, M: int, N: int, block_n: int) -> int:
    return int(L) * _ceil_div(M, K1_PAIR_ROWS) * _ceil_div(N, block_n)


def _k1_planner_sm_100a(
    *, L: int, M: int, N: int, out_f32: bool, sm_count: int
) -> Optional[tuple[int, int]]:
    """sm_100a: the thin weight gradients (``N <= 128``, ``M <= 6144``) whose two splits of 64-wide tiles fit one
    wave of the CTA pairs run ``(64, 2)``; every other row runs the deterministic program."""
    if L != 1 or N > 128 or M > 6144:
        return None
    if 2 * _k1_tiles(L, M, N, 64) > sm_count // 2:
        return None
    return 64, 2


def _k1_planner_sm_103a(
    *, L: int, M: int, N: int, out_f32: bool, sm_count: int
) -> Optional[tuple[int, int]]:
    """sm_103a: the thin weight gradients (``N <= 128``, ``M <= 6144``) run two splits, of 64-wide tiles when
    ``N <= 64`` or when two splits of them fit one wave of the CTA pairs, else of 128-wide tiles, as long as two
    splits of the chosen tiles fit one wave; every other row runs the deterministic program."""
    if L != 1 or N > 128 or M > 6144:
        return None
    pairs = sm_count // 2
    block_n = 64 if (N <= 64 or 2 * _k1_tiles(L, M, N, 64) <= pairs) else 128
    if 2 * _k1_tiles(L, M, N, block_n) > pairs:
        return None
    return block_n, 2


def _k1_planner_sm_107a(
    *, L: int, M: int, N: int, out_f32: bool, sm_count: int
) -> Optional[tuple[int, int]]:
    """sm_107a: the rows of :data:`K1_SM107A_TABLE`; every other row runs the deterministic program."""
    if L != 1:
        return None
    return K1_SM107A_TABLE.get((int(M), int(N), "f32" if out_f32 else "bf16"))


#: The K1 planner of each architecture: ``planner(L=, M=, N=, out_f32=, sm_count=) -> (block_n, num_splits)`` or
#: ``None`` (the deterministic program).  The planners read the device's CTA pairs as ``sm_count // 2``.
K1_PLANNERS: dict[str, Callable[..., Optional[tuple[int, int]]]] = {
    "sm_100a": _k1_planner_sm_100a,
    "sm_103a": _k1_planner_sm_103a,
    "sm_107a": _k1_planner_sm_107a,
}


def k1_splits_fit(splits: int, K: int) -> bool:
    """Whether ``splits`` K ranges each get at least one 64-element K block of ``K`` (``splits * 64 <= K + 63``,
    the program's requirement)."""
    return int(splits) >= 1 and int(splits) * K1_BLOCK_K <= int(K) + K1_BLOCK_K - 1


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
    """``(block_n, num_splits)`` of this package's K1 program for the transposed-store launch ``out[l, n, m]``
    (``M``, ``N`` as in the comment above; ``K`` the token count) on a device of ``sm_count`` SMs, or ``None``: the
    row runs the deterministic program.  The row's verdict in :data:`K1_SPLIT_RULES` first, else the architecture's
    planner (:data:`K1_PLANNERS`); a device below :data:`K1_MIN_SM_COUNT` and a ``K`` too short for the split count
    (:func:`k1_splits_fit`) run the deterministic program."""
    if int(sm_count) < K1_MIN_SM_COUNT.get(arch, 0):
        return None
    key = (arch, int(L), int(M), int(N), "f32" if out_f32 else "bf16")
    if key in K1_SPLIT_RULES:
        choice = K1_SPLIT_RULES[key]
    else:
        planner = K1_PLANNERS.get(arch)
        choice = (
            None
            if planner is None
            else planner(
                L=int(L),
                M=int(M),
                N=int(N),
                out_f32=bool(out_f32),
                sm_count=int(sm_count),
            )
        )
    if choice is None:
        return None
    block_n, splits = (int(v) for v in choice)
    if not k1_splits_fit(splits, K):
        return None
    return block_n, splits


def k1_block_n(N: int) -> int:
    """Column tile of a K1 launch with an explicit split count and no tile width: the choice of
    :data:`K1_BLOCK_N_CHOICES` that pads ``N`` least (the wider tile on a tie)."""
    return min(K1_BLOCK_N_CHOICES, key=lambda bn: (_ceil_div(N, bn) * bn, -bn))


def atomic_k1_template(block_n: int, out_f32: bool) -> str:
    """The registry template of this package's K1 program (both operands MN-major, transposed store)."""
    return (
        f"{cake_jit.TEMPLATE_PREFIXES[0]}nn_n{int(block_n)}"
        + ("_f32" if out_f32 else "")
        + "_t"
    )


# ---------------------------------------------------------------------------
# K1 planning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AtomicGemmPlan:
    """The host plan of one K1 launch of this package: the transposed-store GEMM ``out[l, n, m] = sum_k A[l, k, m]
    B[l, k, n]`` over the ``[L, K, M]`` / ``[L, K, N]`` operand views, ``L x m_tiles x n_tiles`` tiles of
    ``256 x block_n`` each cut into ``splits`` K ranges, walked persistently by at most ``sm_pairs`` CTA pairs.
    ``transposed_out`` is the caller's: without it the caller's ``out[l, m, n]`` is this launch with the caller's
    operands exchanged (``swapped``)."""

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

    @property
    def swapped(self) -> bool:
        return not self.transposed_out

    @property
    def tiles(self) -> int:
        return self.L * self.m_tiles * self.n_tiles

    @property
    def items(self) -> int:
        return self.tiles * self.splits

    @property
    def grid(self) -> tuple[int, int, int]:
        return (2 * min(self.items, self.sm_pairs), 1, 1)

    @property
    def ws_f32_elems(self) -> int:
        """FP32 elements of the split workspace: one ``256 x block_n`` slot per tile and split (none with one
        split).  Every launch writes the slots it reads: the workspace needs no initialisation."""
        return self.items * K1_PAIR_ROWS * self.block_n if self.splits > 1 else 0

    @property
    def counters_u32(self) -> int:
        """Ticket words: one per tile, epilogue warp slice and lane (none with one split).  Zero when allocated;
        every complete launch leaves them zero."""
        return self.tiles * K1_COUNTER_WORDS if self.splits > 1 else 0


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
) -> tuple[Optional[AtomicGemmPlan], Any, Any, Any, str]:
    """Plan this package's K1 GEMM ``out = A @ B`` (``A [L, M, K]`` with unit stride on ``m``, ``B [L, K, N]`` with
    unit stride on ``n``; ``out [L, M, N]``, or ``[L, N, M]`` with ``transposed_out``).

    Returns ``(plan, a_desc, b_desc, out3, reason)``: ``a_desc`` / ``b_desc`` are the launch's ``[L, K, M]`` /
    ``[L, K, N]`` operand views (the caller's ``B`` and ``A`` when the plan is swapped) and ``out3`` the batched
    output.  ``plan`` is ``None`` when the row runs the deterministic program, and ``reason`` says why -- a view this
    package's program does not take (dtypes, a K-major operand, extents not multiples of 8, an output row or batch
    stride not a multiple of 8 elements, an output not 16-byte aligned), a template this checkout does not register
    for ``arch``, or the planner (:func:`k1_plan`).  An explicit ``splits`` / ``block_n`` replaces the planner's
    choice (the other one from the planner, else ``1`` split / :func:`k1_block_n`) and runs this package's program
    whenever the view and the registry allow it.  Raises ``ValueError`` for inconsistent shapes (as the
    deterministic planner does) and for an explicit choice the program cannot run."""
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
            f"dense_projection_gemm: A is [L={L}, M={M}, K={K}] but B is [{Lb}, {Kb}, {N}]"
        )
    want = (L, N, M) if transposed_out else (L, M, N)
    if tuple(int(s) for s in O3.shape) != want:
        raise ValueError(
            f"dense_projection_gemm: out must be {list(want)}, got {list(O3.shape)}"
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
    if splits is None and block_n is None:
        choice = k1_plan(
            arch=arch, L=L, M=M_, N=N_, K=K, out_f32=out_f32, sm_count=sm_count
        )
        if choice is None:
            return (*none, "the planner keeps the row on the deterministic program")
        bn, p = choice
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
            p = planned[1] if planned is not None else 1
        if bn not in K1_BLOCK_N_CHOICES:
            raise ValueError(f"block_n is one of {K1_BLOCK_N_CHOICES}, got {block_n}")
        if not k1_splits_fit(p, K) or p > 65536:
            raise ValueError(
                f"dense_projection_gemm: splits must be in [1, {_ceil_div(K, K1_BLOCK_K)}] (the 64-element K blocks "
                f"of K={K}), got {p}"
            )
    template = atomic_k1_template(bn, out_f32)
    if template not in cake_jit.KERNELS.get(arch, {}):
        return (*none, f"{template} is not registered for {arch}")
    plan = AtomicGemmPlan(
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
    )
    return plan, a_launch, b_launch, O3, ""


def _k1_launch_views(
    A: torch.Tensor, B: torch.Tensor, out: torch.Tensor, plan: AtomicGemmPlan
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The launch's ``[L, K, M]`` / ``[L, K, N]`` operand views and the batched output of a planned K1 launch,
    rebuilt from the caller's tensors (the geometry is the plan's, so its checks hold)."""
    det = deterministic_backend()
    _, a_desc = det.operand_view(det.as_batched(A, "A"), "A", k_axis=2)
    _, b_desc = det.operand_view(det.as_batched(B, "B"), "B", k_axis=1)
    out3 = det.as_batched(out, "out")
    return (b_desc, a_desc, out3) if plan.swapped else (a_desc, b_desc, out3)


def _k1_values(
    plan: AtomicGemmPlan,
    *,
    A: Any,
    B: Any,
    out: Any,
    ws: Any,
    counters: Any,
    ldo: int,
    out_l: int,
) -> dict[str, Any]:
    """The keyword arguments of a K1 launch of this package (ordered by the program's argument plan)."""
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
    )


# ---------------------------------------------------------------------------
# K2 planner: the stream-K partition and the rule table
# ---------------------------------------------------------------------------

#: Per-row verdicts of the router, ``(arch, layout, hidden, experts) -> spec``: ``layout`` ``"kk"`` (the forward,
#: ``hidden = K``) or ``"nn_t"`` (the swapped weight gradient, ``hidden`` = the launch's ``M``), ``experts`` = the
#: launch's ``N``.  A partition spec of :func:`router_units_of` (``"forward"``, ``"pairs"``, ``"tiles"``, ``"S<n>"``
#: or a unit count) runs this package's program; ``None`` runs the deterministic program (its production split
#: count).  A verdict replaces :func:`router_default_spec` for its row; the table ships empty.
ROUTER_RULES: dict[tuple[str, str, int, int], Optional[Union[str, int]]] = {}
#: The router weight gradient of these architectures runs the deterministic program from this hidden size up (its
#: unit count matches the deterministic program's split work there, and the measured margin is within noise).
ROUTER_WGRAD_DET_FROM: dict[str, int] = {"sm_100a": 6144}
#: The forward's largest explicit split count: an element of the forward takes at most two partials, so a forward
#: recomputed under activation checkpointing repeats bit for bit (``0 + a + b == 0 + b + a``).
ROUTER_FORWARD_MAX_SPLITS = 2


def router_units_of(
    spec: Union[str, int], m_pairs: int, n_tiles: int, k_iters: int, sm_count: int
) -> int:
    """The unit count of a partition spec on a device of ``sm_count`` SMs (CTA pairs ``sm_count // 2``): ``"tiles"``
    (one unit per tile), ``"pairs"`` (one per CTA pair), ``"forward"`` (one per CTA pair, at most one per tile),
    ``"S<n>"`` (``n`` units per tile) or a count, rounded down to a multiple of the column tile count (the units of a
    span run its column tiles), at least ``n_tiles`` and at most one unit per (tile, K block) step."""
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


def router_has_part_tiles(m_pairs: int, n_tiles: int, k_iters: int, units: int) -> bool:
    """Whether a launch adds into ``out`` anywhere (then ``out`` is zeroed before the program): some span of the
    partition starts inside a row-tile pair."""
    steps, spans = m_pairs * k_iters, units // n_tiles
    return any(span * steps // spans % k_iters for span in range(1, spans))


def router_at_most_two_partials(m_pairs: int, n_tiles: int, units: int) -> bool:
    """Whether every element of the launch takes at most two partials (then the result repeats bit for bit): at most
    one unit per tile (a span of at least a pair's K blocks meets a pair at most twice), or exactly two."""
    tiles = m_pairs * n_tiles
    return units <= tiles or units == 2 * tiles


def router_default_spec(arch: str, layout: str, hidden: int) -> Optional[str]:
    """The partition of a router row without a verdict: the forward (``kk``) ``"forward"``, the weight gradient
    (``nn_t``) ``"pairs"``, or ``None`` (the deterministic program) from :data:`ROUTER_WGRAD_DET_FROM` up."""
    if layout == "kk":
        return "forward"
    if layout == "nn_t":
        if int(hidden) >= ROUTER_WGRAD_DET_FROM.get(arch, 1 << 62):
            return None
        return "pairs"
    return None


def _router_grid(M: int, N: int, K: int) -> tuple[int, int, int]:
    """``(m_tiles, n_tiles, k_iters)`` of a router launch: an even count of 128-row tiles (a CTA pair owns two
    adjacent ones), the 128-column tiles and the 32-element K blocks."""
    m_tiles = _ceil_div(M, ROUTER_BLOCK_M)
    m_tiles += m_tiles % 2
    return m_tiles, _ceil_div(N, ROUTER_BLOCK_N), _ceil_div(K, ROUTER_BLOCK_K)


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
    """Unit count of this package's K2 launch of the layout class ``layout`` (``kk``, ``kn``, ``nn_t``) for
    ``[M, N] = [M, K] @ [K, N]`` (the launched, possibly swapped GEMM) on a device of ``sm_count`` SMs, or ``None``:
    the launch runs the deterministic program.

    * ``kn`` (input gradient): ``None``.
    * a caller's ``splits``: ``splits`` units per tile (``1``: every tile stored whole, the deterministic program's
      one-split bits -- the control of a measurement); the forward takes at most
      :data:`ROUTER_FORWARD_MAX_SPLITS`.
    * otherwise the row's verdict in :data:`ROUTER_RULES`, else :func:`router_default_spec`.

    A forward partition must give every element at most two partials (:func:`router_at_most_two_partials`):
    ``ValueError`` otherwise."""
    if layout not in ("kk", "nn_t"):
        return None
    m_tiles, n_tiles, k_iters = _router_grid(M, N, K)
    m_pairs = m_tiles // 2
    if splits is not None:
        s = int(splits)
        if not 1 <= s <= k_iters:
            raise ValueError(
                f"router_fp32_gemm: splits must be in [1, {k_iters}] (the 32-element K blocks of K={K}), got {s}"
            )
        if layout == "kk" and s > ROUTER_FORWARD_MAX_SPLITS:
            raise ValueError(
                f"router_fp32_gemm(deterministic=False): the forward takes at most {ROUTER_FORWARD_MAX_SPLITS} "
                f"splits (more partials per element do not repeat bit for bit under recompute), got {s}; use "
                "deterministic=True for more"
            )
        spec: Optional[Union[str, int]] = f"S{s}"
    else:
        hidden = K if layout == "kk" else M
        key = (arch, layout, int(hidden), int(N))
        spec = (
            ROUTER_RULES[key]
            if key in ROUTER_RULES
            else router_default_spec(arch, layout, hidden)
        )
        if spec is None:
            return None
    units = router_units_of(spec, m_pairs, n_tiles, k_iters, sm_count)
    if layout == "kk" and not router_at_most_two_partials(m_pairs, n_tiles, units):
        raise ValueError(
            f"router_fp32_gemm(deterministic=False): the forward partition {spec!r} ({units} units over "
            f"{m_pairs * n_tiles} tiles) gives an element more than two partials"
        )
    return units


@dataclass(frozen=True)
class AtomicRouterPlan:
    """The host plan of one K2 launch of this package (the launched, possibly swapped GEMM ``[M, N] = [M, K] @
    [K, N]``): ``m_tiles / 2 x n_tiles`` tiles of ``256 x 128``, their ``k_iters`` K blocks cut into ``units``
    stream-K spans (``2 * units`` CTAs); with part tiles the launch zeroes ``out`` before the program."""

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

    @property
    def m_pairs(self) -> int:
        return self.m_tiles // 2

    @property
    def tiles(self) -> int:
        return self.m_pairs * self.n_tiles

    @property
    def grid(self) -> tuple[int, int, int]:
        return (2 * self.units, 1, 1)

    @cached_property
    def part_tiles(self) -> bool:
        """Whether some segment adds into ``out`` (the launch zeroes ``out`` before its program runs); computed once
        per plan."""
        return router_has_part_tiles(
            self.m_pairs, self.n_tiles, self.k_iters, self.units
        )

    @property
    def reproducible(self) -> bool:
        """Whether every element takes at most two partials, so the result repeats bit for bit."""
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
) -> tuple[Optional[AtomicRouterPlan], str]:
    """Plan this package's K2 GEMM ``out[M, N] = A[M, K] @ B[K, N]`` (fp32 views, the deterministic package's view
    contract and swap rule) of the ``kk`` / ``nn_t`` classes.  Returns ``(plan, reason)``: ``plan`` is ``None`` when
    the launch runs the deterministic program, and ``reason`` says why -- another layout class, a view the reduce-add
    program cannot address (an ``out`` off a 16-byte boundary or with a row stride not a multiple of 4 elements runs
    the deterministic path, which serves or refuses it as that package does), a template this checkout does not
    register for ``arch``, or the planner (:func:`router_units`).  An explicit ``splits`` (``1`` included) runs
    this package's program when the view and the registry allow it."""
    if (
        A.dtype != torch.float32
        or B.dtype != torch.float32
        or out.dtype != torch.float32
    ):
        raise ValueError("router_fp32_gemm: A, B and out must be fp32")
    if A.dim() != 2 or B.dim() != 2 or out.dim() != 2:
        raise ValueError("router_fp32_gemm: A, B and out must be 2-D views")
    M, K = (int(s) for s in A.shape)
    Kb, N = (int(s) for s in B.shape)
    if Kb != K or tuple(out.shape) != (M, N):
        raise ValueError(
            f"router_fp32_gemm: A [{M}, {K}], B [{Kb}, {N}] and out {list(out.shape)} do not agree"
        )
    if min(M, K, N) < 1:
        raise ValueError(
            f"router_fp32_gemm: M, K and N must be >= 1 (M={M}, N={N}, K={K})"
        )
    a_mn, b_mn = _router_a_mn(A), _router_b_mn(B)
    if a_mn is None or b_mn is None:
        return None, "an operand without a unit stride"
    layout = _k2_class(a_mn, b_mn)[3:]
    if layout not in ("kk", "nn_t"):
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
    template = cake_jit.TEMPLATE_PREFIXES[1] + ("nn_t" if swapped else "kk")
    if template not in cake_jit.KERNELS.get(arch, {}):
        return None, f"{template} is not registered for {arch}"
    units = router_units(
        layout, arch=arch, M=M_eff, N=N_eff, K=K, sm_count=sm_count, splits=splits
    )
    if units is None:
        return None, "the planner keeps the row on the deterministic program"
    m_tiles, n_tiles, k_iters = _router_grid(M_eff, N_eff, K)
    plan = AtomicRouterPlan(
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
    )
    return plan, ""


def _k2_views(
    A: torch.Tensor, B: torch.Tensor, plan: AtomicRouterPlan
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(a_view, split_src)``: the streamed operand as its ``[1, M, K]`` / ``[1, K, M]`` TMA span and the source of
    the BF16x3 split of the second operand (the deterministic package's convention), after the swap."""
    A_eff, B_eff = (B.t(), A.t()) if plan.swapped else (A, B)
    split_src = B_eff if plan.b_mn else B_eff.t()
    a_view = A_eff.t().unsqueeze(0) if plan.a_mn else A_eff.unsqueeze(0)
    return a_view, split_src


def _k2_out32(out: torch.Tensor) -> torch.Tensor:
    return out if out.is_contiguous() else deterministic_backend().flat_alias(out)


def _k2_values(
    plan: AtomicRouterPlan, *, A: Any, parts: Any, out: torch.Tensor
) -> dict[str, Any]:
    """The keyword arguments of a K2 launch of this package (ordered by the program's argument plan): ``out`` is
    passed twice, as the store pointer and as the reduce-add map ``D``."""
    return dict(
        A=A,
        B=parts,
        out32=_k2_out32(out),
        D=out,
        M=plan.M,
        N=plan.N,
        m_tiles=plan.m_tiles,
        n_tiles=plan.n_tiles,
        k_iters=plan.k_iters,
        ldo=int(out.stride(0)),
        units=plan.units,
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
        cu = _CU_DRIVER
        if cu and torch.cuda.current_device() == self.index:
            if self.rows == 1 or self.pitch == 4 * self.cols:
                (status,) = cu.cuMemsetD32Async(
                    self.ptr, 0, self.rows * self.cols, stream
                )
            else:
                (status,) = cu.cuMemsetD2D32Async(
                    self.ptr, self.pitch, 0, self.cols, self.rows, stream
                )
            if status == cu.CUresult.CUDA_SUCCESS:
                return
        self.out.zero_()


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
    """Order ``values`` by the generated argument plan of ``module_name`` and load its entry: a program of the
    deterministic package binds through that package (its ``bind_launch``), one of this package here.  Fails
    closed on a keyword the kernel expects that the host does not provide."""
    if cake_jit.is_det_program(module_name):
        return deterministic_backend().bind_launch(module_name, values, grid, arch)
    entry, arguments = _atomic_bind(module_name, values, grid, arch)
    return _AtomicLaunch(module_name, entry, arguments, grid)


def _k1_dummies(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """The workspace of a one-split K1 launch, which never reads or writes it: the deterministic package's
    per-device 16-element dummies."""
    det = deterministic_backend()
    return det._dummy(device, torch.float32), det._dummy(device, torch.uint32)


@dataclass
class PreparedAtomicGemm:
    """One prepared K1 launch of this package: the bound program and its workspace (the FP32 split slots, written
    before they are read, and the tickets, zero when prepared and zero again after every complete launch).
    ``launch()`` writes ``out`` and returns it without allocating; it captures into a CUDA graph as one kernel.  A
    prepared launch must not run concurrently with itself (two streams would share its workspace); after a launch
    that did not complete (an aborted process step, a device error), :meth:`reset_counters` re-zeroes the tickets
    on the current stream."""

    module_name: str
    arch: str
    plan: AtomicGemmPlan
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
        if self.plan.splits == 1:
            return True
        return not bool(self.tensors["counters"].view(torch.int32).any())

    def reset_counters(self) -> None:
        """Zero the tickets on the current stream (after a launch that did not complete)."""
        if self.plan.splits > 1:
            self.tensors["counters"].view(torch.int32).zero_()

    def launch(self) -> torch.Tensor:
        assert self._launch is not None
        with deterministic_backend()._ffi_stream_context(self.device_index):
            self._launch()
        return self.out

    __call__ = launch


@dataclass
class PreparedAtomicRouterGemm:
    """One prepared K2 launch of this package: the retained BF16x3 stack of the second operand (re-split from the
    bound fp32 view on every ``launch()`` by the deterministic package's split stage), then ``out`` zeroed on the
    stream (one memset, :class:`_OutFill`) when a segment adds into it, then the stream-K program.  Everything but
    the current stream is bound when prepared (the FFI arguments converted once, the fill's pointer and pitch), so a
    call issues its launches back to back.  ``launch()`` writes ``out`` and returns it without allocating; it
    captures into a CUDA graph (split, memset, kernel)."""

    module_name: str
    arch: str
    plan: AtomicRouterPlan
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
        det = deterministic_backend()
        index = self.device_index
        stream = det._current_raw_stream(index)
        if self._split is None:
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


def _device_and_arch(A, B, out) -> tuple[torch.device, str, int]:
    det = deterministic_backend()
    device = det._same_cuda_device(A, B, out)
    return device, det.require_arch(device), det.device_facts(device).sm_count


#: The K1 knobs of this package's planner (the others are the deterministic planner's, applied when a row runs the
#: deterministic program).
_K1_OWN_KNOBS = ("splits", "block_n")


def _split_knobs(knobs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    knobs = dict(knobs)
    own = {k: knobs.pop(k) for k in _K1_OWN_KNOBS if k in knobs}
    return own, knobs


def _prepare_atomic_k1(
    plan: AtomicGemmPlan, a_desc, b_desc, out3, out: torch.Tensor, arch: str
) -> PreparedAtomicGemm:
    det = deterministic_backend()
    device = out.device
    if plan.splits > 1:
        ws = torch.empty(plan.ws_f32_elems, device=device, dtype=torch.float32)
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
    )
    launch = bind_launch(module_name, values, plan.grid, arch)
    return PreparedAtomicGemm(
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
) -> Any:
    """This package's K1 GEMM ``out = A @ B`` (both MN-major): ``op`` is ``"plan"`` (returns the plan),
    ``"prepare"`` (the prepared launch) or ``"launch"`` (the convenience path, returns ``out``); ``own`` holds this
    package's knobs (``splits``, ``block_n``).  A row :func:`plan_atomic_gemm` keeps on the deterministic program
    returns ``det_call()``: the deterministic package's object of the entry point and ``op``, with its knobs."""
    if op == "launch":
        return _launch_cached_atomic_k1(
            A, B, out, transposed_out=transposed_out, own=own, det_call=det_call
        )
    arch, sm_count = _k1_arch_and_sms(A, B, out, arch, sm_count)
    plan, a_desc, b_desc, out3, _ = plan_atomic_gemm(
        A, B, out, transposed_out=transposed_out, arch=arch, sm_count=sm_count, **own
    )
    if plan is None:
        return det_call()
    if op == "plan":
        return plan
    return _prepare_atomic_k1(plan, a_desc, b_desc, out3, out, arch)


def _prepare_atomic_k2(
    plan: AtomicRouterPlan,
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    arch: str,
) -> PreparedAtomicRouterGemm:
    det = deterministic_backend()
    device = out.device
    a_view, split_src = _k2_views(A, B, plan)
    split_module = det._fused_split_for(arch, split_src)
    parts = torch.empty((3, *split_src.shape), device=device, dtype=torch.bfloat16)
    tensors = dict(A=a_view, split_src=split_src, parts=parts)
    if split_module is None:
        tensors["resid"] = torch.empty(
            split_src.shape, device=device, dtype=torch.float32
        )
    module_name = cake_jit.select_module(arch, plan.template)
    values = _k2_values(plan, A=a_view, parts=parts, out=out)
    launch = _converted_launch(bind_launch(module_name, values, plan.grid, arch))
    split_launch = None
    if split_module is not None:
        n = split_src.numel()
        split_launch = _converted_launch(
            det.bind_launch(
                split_module,
                det._split_values(split_src, parts, n),
                det._split_grid(n),
                arch,
            )
        )
    index = det._device_index(device)
    return PreparedAtomicRouterGemm(
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
) -> Any:
    """This package's K2 GEMM ``out = A @ B`` (FP32 through the BF16x3 emulation) of the layout classes ``kk`` /
    ``nn_t``: ``op`` is ``"plan"``, ``"prepare"`` or ``"launch"`` as for :func:`_atomic_k1`.  A launch
    :func:`plan_atomic_router` keeps on the deterministic program runs the deterministic package's object of ``op``
    with the caller's ``splits`` (``None``: its production split count)."""
    if op == "launch":
        return _launch_cached_atomic_k2(A, B, out, splits=splits)
    _, arch, sm_count = _device_and_arch(A, B, out)
    plan, _ = plan_atomic_router(A, B, out, arch=arch, sm_count=sm_count, splits=splits)
    if plan is None:
        name = "plan_router_fp32_gemm" if op == "plan" else "prepare_router_fp32_gemm"
        return _det_entry(name)(A, B, out, splits=splits)
    if op == "plan":
        return plan
    return _prepare_atomic_k2(plan, A, B, out, arch)


# --- convenience-path state of this package's programs -----------------------------------------------------------
#
# The deterministic package's design (one least-recently-used cache per kind, keyed by device, stream and view
# geometry; one growable workspace arena per (device, raw stream) shared by every cached entry), with caches and
# arenas of their own: a buffer of this package is never a buffer of the deterministic package, and the names below
# are not names this path had before #8046 (those read through to the deterministic package's state).  The cache
# bound is the deterministic package's ``CONVENIENCE_CACHE_SIZE``, read on every insertion, so one setting bounds
# both packages.  A cached entry holds its plan and the resolved program entry, never a caller's tensor: the views
# are rebuilt from the caller's tensors on every call.  The launches of one stream share its arena; launches on
# different streams never do.  The K1 tickets are zero when allocated or grown (``torch.zeros`` on the arena's
# stream) and zero again after every complete launch, so a launch needs no memset; the K1 split slots are written
# before they are read and are never initialised.  A row the planner keeps on the deterministic program is cached
# as such and calls the deterministic package's convenience path.  After a launch that did not complete,
# :func:`clear_convenience_caches` drops the arenas (fresh tickets on the next call).

#: workspace kind -> (dtype, zeroed when allocated)
_ATOMIC_ARENA_KINDS: dict[str, tuple[torch.dtype, bool]] = {
    # K1 split partial slots: written before they are read
    "k1_slots": (torch.float32, False),
    # K1 per-(tile, warp slice, lane) arrival tickets, zero between launches
    "k1_tickets": (torch.uint32, True),
    # K2 BF16x3 stack of the split operand ``[3, *split_src.shape]``
    "k2_stack": (torch.bfloat16, False),
    # K2 fp32 scratch of the torch-chain split
    "k2_split_scratch": (torch.float32, False),
}
_ATOMIC_ARENA_MIN_ELEMS = 16


@dataclass
class _AtomicArena:
    """The workspace of every cached launch of this package on one (device index, raw stream): one buffer per kind
    of ``_ATOMIC_ARENA_KINDS``, grown to the largest requirement seen and never shrunk (the deterministic package's
    ``_Arena`` contract; a zeroed kind is allocated with ``torch.zeros`` on the current stream)."""

    device: torch.device
    key: tuple[int, int]
    buffers: dict[str, torch.Tensor] = field(default_factory=dict, repr=False)
    version: int = 0
    refs: int = 0
    lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    def require(self, kind: str, numel: int) -> torch.Tensor:
        """The buffer of ``kind`` with at least ``numel`` elements, grown when the present one is smaller (called
        when an entry is planned, under ``_ATOMIC_PLAN_LOCK``); monotone."""
        buf = self.buffers.get(kind)
        numel = max(int(numel), _ATOMIC_ARENA_MIN_ELEMS)
        if buf is None or buf.numel() < numel:
            dtype, zeroed = _ATOMIC_ARENA_KINDS[kind]
            new = (torch.zeros if zeroed else torch.empty)(
                numel, device=self.device, dtype=dtype
            )
            buf = self.buffers.get(kind)
            if buf is None or buf.numel() < new.numel():
                buf = self.buffers[kind] = new
            self.version += 1
        return buf

    def nbytes(self) -> int:
        """Bytes the arena holds (every kind)."""
        return sum(int(b.numel()) * b.element_size() for b in self.buffers.values())


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


@dataclass
class _CachedAtomic:
    """A planned launch of the convenience path: the plan (``None``: the row runs the deterministic program), the
    program entry and argument plan, the arena of its (device, stream) and, for K2, the fused split's module with its
    entry and argument plan."""

    plan: Any
    arch: str = ""
    module_name: str = ""
    entry: Optional[Callable[..., Any]] = field(default=None, repr=False)
    arg_plan: Optional[list] = field(default=None, repr=False)
    arena: Optional[_AtomicArena] = field(default=None, repr=False)
    split_module: Optional[str] = None
    split_entry: Optional[Callable[..., Any]] = field(default=None, repr=False)
    split_arg_plan: Optional[list] = field(default=None, repr=False)


def _release_atomic_entry(entry: _CachedAtomic) -> None:
    """``on_evict`` of the convenience caches: the arena of an evicted entry loses a reference and is released when
    no cached entry uses it any more."""
    arena = entry.arena
    if arena is None:
        return
    arena.refs -= 1
    if arena.refs <= 0 and _ATOMIC_ARENAS.get(arena.key) is arena:
        del _ATOMIC_ARENAS[arena.key]


def _cache_insert(
    cache: "OrderedDict[tuple, Any]", key: tuple, entry: _CachedAtomic
) -> _CachedAtomic:
    det = deterministic_backend()
    if entry.arena is not None:
        entry.arena.refs += 1
    return det._cache_put(
        cache, key, entry, det.CONVENIENCE_CACHE_SIZE, _release_atomic_entry
    )


def _launch_cached_atomic_k1(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    *,
    transposed_out: bool,
    own: dict[str, Any],
    det_call: Callable[[], Any],
) -> Any:
    det = deterministic_backend()
    device = det._same_cuda_device(A, B, out)
    index = det._device_index(device)
    stream = det._current_raw_stream(index)
    flags = ("k1", bool(transposed_out), tuple(sorted(own.items())))
    key = det.convenience_key(index, stream, flags, A, B, out)
    entry = det._cache_get(_K1_ATOMIC_CACHE, key)
    if entry is None:
        with _ATOMIC_PLAN_LOCK:
            entry = det._cache_get(_K1_ATOMIC_CACHE, key)
            if entry is None:
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
                else:
                    arena = None
                    if plan.splits > 1:
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
                    )
                entry = _cache_insert(_K1_ATOMIC_CACHE, key, entry)
    plan = entry.plan
    if plan is None:
        return det_call()
    fn = entry.entry
    assert fn is not None
    a_desc, b_desc, out3 = _k1_launch_views(A, B, out, plan)
    ldo, out_l = _k1_out_strides(out3)
    if entry.arena is not None:
        ws, counters = (
            entry.arena.buffers["k1_slots"],
            entry.arena.buffers["k1_tickets"],
        )
    else:
        ws, counters = _k1_dummies(device)
    values = _k1_values(
        plan,
        A=a_desc,
        B=b_desc,
        out=det.flat_alias(out3),
        ws=ws,
        counters=counters,
        ldo=ldo,
        out_l=out_l,
    )
    arguments = _order_atomic_arguments(
        entry.module_name, entry.arg_plan, values, plan.grid
    )
    with det._ffi_stream_context(index, stream):
        fn(*arguments)
    return out


def _launch_cached_atomic_k2(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    *,
    splits: Optional[int],
) -> torch.Tensor:
    det = deterministic_backend()
    device = det._same_cuda_device(A, B, out)
    index = det._device_index(device)
    stream = det._current_raw_stream(index)
    key = det.convenience_key(index, stream, ("k2", splits), A, B, out)
    entry = det._cache_get(_K2_ATOMIC_CACHE, key)
    if entry is None:
        with _ATOMIC_PLAN_LOCK:
            entry = det._cache_get(_K2_ATOMIC_CACHE, key)
            if entry is None:
                arch = det.require_arch(device)
                plan, _ = plan_atomic_router(
                    A,
                    B,
                    out,
                    arch=arch,
                    sm_count=det.device_facts(device).sm_count,
                    splits=splits,
                )
                if plan is None:
                    entry = _CachedAtomic(plan=None, arch=arch)
                else:
                    _, split_src = _k2_views(A, B, plan)
                    split_module = det._fused_split_for(arch, split_src)
                    arena = _atomic_arena_for(index, stream, device)
                    arena.require("k2_stack", 3 * split_src.numel())
                    if split_module is None:
                        arena.require("k2_split_scratch", split_src.numel())
                    module_name = cake_jit.select_module(arch, plan.template)
                    fn, arg_plan = _atomic_entry(module_name, plan.grid, arch)
                    split_fn, split_arg_plan = None, None
                    if split_module is not None:
                        record = det.MODULES[split_module]
                        split_fn = getattr(
                            det.load_cake_dense_projection_det_gemm_module(
                                split_module, arch
                            ),
                            record["ffi_entry"],
                        )
                        split_arg_plan = det.arg_plan_of(record)
                    entry = _CachedAtomic(
                        plan=plan,
                        arch=arch,
                        module_name=module_name,
                        entry=fn,
                        arg_plan=arg_plan,
                        arena=arena,
                        split_module=split_module,
                        split_entry=split_fn,
                        split_arg_plan=split_arg_plan,
                    )
                entry = _cache_insert(_K2_ATOMIC_CACHE, key, entry)
    plan = entry.plan
    if plan is None:
        return _det_entry("router_fp32_gemm")(A, B, out, splits=splits)
    a_view, split_src = _k2_views(A, B, plan)
    arena, fn = entry.arena, entry.entry
    assert arena is not None and fn is not None
    # the split, the memset and the kernel share the arena's BF16x3 stack
    with arena.lock:
        n = split_src.numel()
        parts = arena.buffers["k2_stack"][: 3 * n].view(3, *split_src.shape)
        if entry.split_module is None:
            resid = arena.buffers["k2_split_scratch"][:n].view(split_src.shape)
            det.split_fp32_to_bf16x3_det_(split_src, parts, resid)
        # the arguments are converted and the fill bound before the first launch: the launches go back to back
        values = _k2_values(plan, A=a_view, parts=parts, out=out)
        arguments = _ffi_tensors(
            _order_atomic_arguments(
                entry.module_name, entry.arg_plan, values, plan.grid
            )
        )
        split_arguments = None
        if entry.split_entry is not None:
            split_arguments = _ffi_tensors(
                _order_atomic_arguments(
                    entry.split_module,
                    entry.split_arg_plan,
                    det._split_values(split_src, parts, n),
                    det._split_grid(n),
                )
            )
        fill = _out_fill(out, index) if plan.part_tiles else None
        with det._ffi_stream_context(index, stream):
            if split_arguments is not None:
                entry.split_entry(*split_arguments)
            if fill is not None:
                fill(stream)
            fn(*arguments)
    return out


def clear_convenience_caches() -> None:
    """Drop every cached convenience-path launch and release every workspace arena of both packages (this
    package's launches and the deterministic package's)."""
    with _ATOMIC_PLAN_LOCK:
        _K1_ATOMIC_CACHE.clear()
        _K2_ATOMIC_CACHE.clear()
        _ATOMIC_ARENAS.clear()
    deterministic_backend().clear_convenience_caches()


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def _splits_kw(splits: Optional[int]) -> dict[str, Any]:
    """The ``splits`` keyword for a deterministic call: absent when the caller gave none, so the deterministic
    package's own default (its production split count) applies."""
    return {} if splits is None else {"splits": splits}


def instance_symbol(key: tuple, *, deterministic: bool = True) -> str:
    """The instance template of a K1 instance key (``deterministic_backend().instance_key``): the deterministic
    package's symbol in both modes.  This package's K1 programs are not keyed by the deterministic instance keys;
    their templates are :func:`atomic_k1_template` (``dense_proj_gemm_atomic_nn_n<block_n>[_f32]_t``)."""
    _resolve_deterministic(deterministic, "instance_symbol")
    return _det_entry("instance_symbol")(key)


def router_symbol(
    a_mn: bool, b_mn: bool, out_t: bool, *, deterministic: bool = True
) -> str:
    """Kernel symbol / registry template of a router instance: ``router_fp32_det_gemm_<a><b>[_t]`` of the
    deterministic package, or with ``deterministic=False`` ``router_fp32_gemm_atomic_<a><b>[_t]`` for the layout
    classes this package reduces (``kk``, ``nn_t``) and the deterministic symbol for ``kn``."""
    det = _resolve_deterministic(deterministic, "router_symbol")
    det_symbol = _det_entry("router_symbol")(a_mn, b_mn, out_t)
    if det or _route(_k2_class(a_mn, b_mn)) == "det":
        return det_symbol
    return (
        cake_jit.TEMPLATE_PREFIXES[1]
        + ("n" if a_mn else "k")
        + ("n" if b_mn else "k")
        + ("_t" if out_t else "")
    )


def plan_dense_projection_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    *,
    deterministic: bool = True,
    **kwargs: Any,
) -> Any:
    """Validate the views and plan the K1 launch: the deterministic package's ``plan_dense_projection_det_gemm``
    with the same keywords (``sm_count``, ``l2_bytes``, ``transposed_out``, the planner knobs, ``arch``); with
    ``deterministic=False`` a weight-gradient layout (both operands MN-major) is planned for this package's split-K
    program (:class:`AtomicGemmPlan`; the knobs ``splits`` / ``block_n`` are this package's, the others apply when
    the row runs the deterministic program)."""
    if (
        _resolve_deterministic(deterministic, "plan_dense_projection_gemm")
        or _route(_k1_layout(A, B)) == "det"
    ):
        return _det_entry("plan_dense_projection_gemm")(A, B, out, **kwargs)
    own, det_kwargs = _split_knobs(kwargs)
    return _atomic_k1(
        "plan",
        A,
        B,
        out,
        transposed_out=bool(det_kwargs.get("transposed_out", False)),
        own=own,
        arch=det_kwargs.get("arch"),
        sm_count=det_kwargs.get("sm_count"),
        det_call=lambda: _det_entry("plan_dense_projection_gemm")(
            A, B, out, **det_kwargs
        ),
    )


def prepare_dense_projection_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    *,
    deterministic: bool = True,
    **kwargs: Any,
) -> Any:
    """Validate one K1 binding, plan it for the device and prepare its allocation-free launch: the deterministic
    package's ``prepare_dense_projection_det_gemm`` with the same keywords (``transposed_out`` and the planner
    knobs); with ``deterministic=False`` a weight-gradient layout prepares this package's split-K program (its
    workspace is owned by the prepared object; ``splits`` / ``block_n`` as for :func:`plan_dense_projection_gemm`)."""
    if (
        _resolve_deterministic(deterministic, "prepare_dense_projection_gemm")
        or _route(_k1_layout(A, B)) == "det"
    ):
        return _det_entry("prepare_dense_projection_gemm")(A, B, out, **kwargs)
    own, det_kwargs = _split_knobs(kwargs)
    return _atomic_k1(
        "prepare",
        A,
        B,
        out,
        transposed_out=bool(det_kwargs.get("transposed_out", False)),
        own=own,
        det_call=lambda: _det_entry("prepare_dense_projection_gemm")(
            A, B, out, **det_kwargs
        ),
    )


def dense_projection_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    *,
    transposed_out: bool = False,
    deterministic: bool = True,
) -> torch.Tensor:
    """``out[l, m, n] = sum_k A[l, m, k] B[l, k, n]`` for strided views (the cached convenience path; see the
    deterministic package for the view contract).  ``deterministic=False`` runs a weight-gradient layout (both
    operands MN-major) on this package's split-K program where the planner takes the row; every other layout and
    row runs the deterministic program."""
    if _resolve_deterministic(deterministic, "dense_projection_gemm") or (
        _route(_k1_layout(A, B)) == "det"
    ):
        return _det_entry("dense_projection_gemm")(
            A, B, out, transposed_out=transposed_out
        )
    return _atomic_k1(
        "launch",
        A,
        B,
        out,
        transposed_out=transposed_out,
        own={},
        det_call=lambda: _det_entry("dense_projection_gemm")(
            A, B, out, transposed_out=transposed_out
        ),
    )


def projection_forward(
    X: torch.Tensor,
    W: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    deterministic: bool = True,
) -> torch.Tensor:
    """``out[T, N] = X[T, K] @ W[N, K].T`` (bf16 out by default).  The deterministic program in both modes (a
    forward has no cross-CTA reduction, and a recomputed forward must reproduce its result)."""
    _resolve_deterministic(deterministic, "projection_forward")
    return _det_entry("projection_forward")(X, W, out)


def projection_dgrad(
    G: torch.Tensor,
    W: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    out_dtype=None,
    deterministic: bool = True,
) -> torch.Tensor:
    """``out[T, K] = G[T, N] @ W[N, K]``.  The deterministic program in both modes (whole output tiles)."""
    _resolve_deterministic(deterministic, "projection_dgrad")
    return _det_entry("projection_dgrad")(G, W, out, out_dtype=out_dtype)


def prepare_projection_wgrad(
    G: torch.Tensor,
    X: torch.Tensor,
    out: torch.Tensor,
    *,
    sk: bool | str = "auto",
    cta_rows: Optional[int] = None,
    deterministic: bool = True,
) -> Any:
    """Prepared ``out[N, K] = G[T, N].T @ X[T, K]``.  ``deterministic=False`` prepares this package's split-K
    program as the transposed-store GEMM ``X.T @ G`` (``M = K``, ``N = N``) where the planner takes the row; a row it
    does not take prepares the deterministic package's ``prepare_projection_det_wgrad`` (its orientation, ``sk``,
    ``cta_rows``)."""
    if _resolve_deterministic(deterministic, "prepare_projection_wgrad"):
        return _det_entry("prepare_projection_wgrad")(
            G, X, out, sk=sk, cta_rows=cta_rows
        )
    return _atomic_k1(
        "prepare",
        X.t(),
        G,
        out,
        transposed_out=True,
        own={},
        det_call=lambda: _det_entry("prepare_projection_wgrad")(
            G, X, out, sk=sk, cta_rows=cta_rows
        ),
    )


def projection_wgrad(
    G: torch.Tensor,
    X: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    out_dtype=None,
    sk: bool | str = "auto",
    cta_rows: Optional[int] = None,
    deterministic: bool = True,
) -> torch.Tensor:
    """``out[N, K] = G[T, N].T @ X[T, K]`` (ragged reduction length ``T``).  ``deterministic=False`` runs this
    package's split-K program where the planner takes the row (the split partials summed in the order they finish,
    the output rounded once), else the deterministic package's ``projection_det_wgrad``."""
    if _resolve_deterministic(deterministic, "projection_wgrad"):
        return _det_entry("projection_wgrad")(
            G, X, out, out_dtype=out_dtype, sk=sk, cta_rows=cta_rows
        )
    if out is None:
        out = torch.empty(
            int(G.shape[1]),
            int(X.shape[1]),
            device=G.device,
            dtype=out_dtype or torch.bfloat16,
        )
    return _atomic_k1(
        "launch",
        X.t(),
        G,
        out,
        transposed_out=True,
        own={},
        det_call=lambda: _det_entry("projection_wgrad")(
            G, X, out, out_dtype=out_dtype, sk=sk, cta_rows=cta_rows
        ),
    )


def plan_router_fp32_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    *,
    splits: Optional[int] = None,
    deterministic: bool = True,
) -> Any:
    """Classify the fp32 views and plan the K2 launch (the deterministic package's ``plan_router_fp32_det_gemm``);
    ``deterministic=False`` plans this package's stream-K program for the ``kk`` / ``nn_t`` classes
    (:class:`AtomicRouterPlan`; ``splits`` sets ``splits`` units per tile, see :func:`router_units`)."""
    if _resolve_deterministic(deterministic, "plan_router_fp32_gemm") or (
        _route(_k2_layout(A, B)) == "det"
    ):
        return _det_entry("plan_router_fp32_gemm")(A, B, out, splits=splits)
    return _atomic_k2("plan", A, B, out, splits=splits)


def prepare_router_fp32_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    *,
    splits: Optional[int] = None,
    deterministic: bool = True,
) -> Any:
    """Validate one fp32 binding, plan it and prepare its allocation-free launch (the deterministic package's
    ``prepare_router_fp32_det_gemm``); ``deterministic=False`` prepares this package's stream-K program for the
    ``kk`` / ``nn_t`` classes (its launch splits the second operand, zeroes ``out`` on the stream when a segment adds
    into it, and runs the program: no partial buffers)."""
    if _resolve_deterministic(deterministic, "prepare_router_fp32_gemm") or (
        _route(_k2_layout(A, B)) == "det"
    ):
        return _det_entry("prepare_router_fp32_gemm")(A, B, out, splits=splits)
    return _atomic_k2("prepare", A, B, out, splits=splits)


def _router_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    splits: Optional[int],
    det: bool,
) -> torch.Tensor:
    if det or _route(_k2_layout(A, B)) == "det":
        return _det_entry("router_fp32_gemm")(A, B, out, splits=splits)
    return _atomic_k2("launch", A, B, out, splits=splits)


def router_fp32_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    *,
    splits: Optional[int] = None,
    deterministic: bool = True,
) -> torch.Tensor:
    """``out[m, n] = sum_k A[m, k] B[k, n]`` for fp32 views through the BF16x3 emulation (the cached convenience
    path).  ``deterministic=False`` runs the ``kk`` / ``nn_t`` classes on this package's stream-K program, whose
    part tiles add into ``out`` in the order they finish, instead of the fixed-order split-K sum."""
    det = _resolve_deterministic(deterministic, "router_fp32_gemm")
    return _router_gemm(A, B, out, splits, det)


def router_forward(
    X: torch.Tensor,
    W: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    splits: Optional[int] = None,
    deterministic: bool = True,
) -> torch.Tensor:
    """``out[T, N] = X[T, K] @ W[N, K].T`` in FP32 through the emulation; ``splits`` defaults to the deterministic
    package's production count (4), or with ``deterministic=False`` to the forward partition of
    :func:`router_units` (at most two partials per element: the logits repeat bit for bit)."""
    if _resolve_deterministic(deterministic, "router_forward"):
        return _det_entry("router_forward")(X, W, out, **_splits_kw(splits))
    if out is None:
        out = torch.empty(X.shape[0], W.shape[0], device=X.device, dtype=torch.float32)
    return _router_gemm(X, W.t(), out, splits, False)


def router_dgrad(
    G: torch.Tensor,
    W: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    deterministic: bool = True,
) -> torch.Tensor:
    """``out[T, K] = G[T, N] @ W[N, K]`` in FP32 through the emulation.  The deterministic program in both modes
    (one split: nothing to reduce)."""
    _resolve_deterministic(deterministic, "router_dgrad")
    return _det_entry("router_dgrad")(G, W, out)


def router_wgrad(
    G: torch.Tensor,
    X: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    splits: Optional[int] = None,
    deterministic: bool = True,
) -> torch.Tensor:
    """``out[N, K] = G[T, N].T @ X[T, K]`` in FP32: the swapped GEMM ``X.T @ G`` with a transposed store;
    ``splits`` defaults to the deterministic package's production count (6), or with ``deterministic=False`` to the
    weight-gradient partition of :func:`router_units`."""
    if _resolve_deterministic(deterministic, "router_wgrad"):
        return _det_entry("router_wgrad")(G, X, out, **_splits_kw(splits))
    if out is None:
        out = torch.empty(G.shape[1], X.shape[1], device=G.device, dtype=torch.float32)
    return _router_gemm(G.t(), X, out, splits, False)


def split_fp32_to_bf16x3_(
    src: torch.Tensor,
    parts: torch.Tensor,
    resid: torch.Tensor,
    *,
    deterministic: bool = True,
) -> torch.Tensor:
    """Exact three-way BF16 split ``src = p1 + p2 + p3`` into the retained ``parts [3, *src.shape]`` stack with the
    fp32 scratch ``resid``.  The same deterministic stage in both modes."""
    _resolve_deterministic(deterministic, "split_fp32_to_bf16x3_")
    return _det_entry("split_fp32_to_bf16x3_")(src, parts, resid)


def __getattr__(name: str) -> Any:
    """Read the other module-level names of this path through to the deterministic package (``ROW_RULES``,
    ``stream_k_plan``, ``wgrad_views``, the plan and prepared types, ...).  Names spelling the ``det`` token are not
    read through: they live in the deterministic package only."""
    if (name.startswith("__") and name.endswith("__")) or _DET_SPELLED.search(name):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        return getattr(deterministic_backend(), DETERMINISTIC_NAMES.get(name, name))
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
