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
    reads the tile width and the split count from the measured verdicts of :data:`K1_SPLIT_RULES`; every other row
    runs the deterministic program).  Two splits add ``own + other``, one add of two values, so a two-split row
    repeats bit for bit; three or more add in the order the splits finish;
  - the K2 weight gradient (and a forward given an explicit split count): a stream-K partition of the (tile, K
    block) steps into ``units`` spans, alike or a head/tail partition of long head spans and short tail spans
    (:func:`router_partition` reads the measured verdicts and token windows of :data:`ROUTER_RULES`; every other row
    and token count runs the deterministic program); a segment that covers its tile's whole K range stores the
    deterministic program's one-split bits, any other adds its partial into ``out`` with a TMA reduce-add (``out`` is
    zeroed on the launch's stream first).  A forward partition gives every element at most two partials,
    ``0 + a + b == 0 + b + a``, so a forward recomputed under activation checkpointing gives the same logits and the
    same top-k;
  - the K2 weight and input gradients in the SIMT token windows of their rows in the router tables
    (:func:`router_rules`, :func:`router_simt`): a compensated exact-FP32 program on the CUDA cores over the FP32
    operands read in place, every element of ``out`` stored once (other bits than the deterministic program's BF16x3
    emulation, within the deterministic package's router FP64 budget); the input gradient's every other row and token
    count runs the deterministic program (one split).

  Every other operation -- the K1 forward and input gradient, the BF16x3 split -- runs the deterministic programs in
  both modes: a recomputed forward must reproduce its result, and where the deterministic plan of a K1 forward or input
  gradient splits K across CTAs (its automatic two-way split), it adds the partials in a fixed order.  On the
  convenience path a call of ``deterministic=False`` that
  runs a deterministic program (the K1 forward and input gradient, a K2 row without a verdict) launches it through a
  binding of this package (the deterministic package's plan and programs, and the reduction of its split-K partials in
  the deterministic package's order by this package's program: the same bits, fewer host operations per call).
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

import math
import re
import sys
import threading
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import cached_property
from types import ModuleType
from typing import Any, Callable, Optional, TypeVar, Union

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
    binds its ``cake_backend`` here before the first call that needs it.  Binding drops this package's convenience
    caches and arenas (:func:`_drop_convenience_entries`): their entries hold the plans and programs of the package
    bound before."""
    missing = [
        name for key, name in DETERMINISTIC_NAMES.items() if not hasattr(module, name)
    ]
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
#: programs (the planner keeps every row without a measured verdict on the deterministic program, see
#: :func:`k1_plan` / :func:`router_partition`) -- or ``"det"`` -- the deterministic package's programs in both modes.  K1
#: operations are keyed by the operand layouts of the launched GEMM (``kk`` forward, ``kn`` input gradient, ``nn``
#: weight gradient; ``nk`` has no training use), K2 by the layout class of ``router_layout_class``.
ROUTES: dict[str, str] = {
    "k1_kk": "det",  # forward: feeds the recompute of checkpointing; whole tiles or a fixed-order two-way K split
    "k1_kn": "det",  # input gradient: whole tiles or a fixed-order two-way K split
    "k1_nk": "det",
    "k1_nn": "atomic",  # weight gradient: split-K over the ragged token count
    "k2_kk": "atomic",  # router forward: no verdict ships (deterministic plan); an explicit split count <= 2 runs it
    "k2_kn": "atomic",  # router input gradient: the SIMT program in its token windows, else one split
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
K1_BLOCK_N_CHOICES = (64, 128, 192, 256)
#: The tile widths of :data:`K1_BLOCK_N_CHOICES` this package runs as its row-major-store program, with the operands
#: exchanged (:attr:`AtomicGemmPlan.out_t` false): a CTA pair's tile is 256 rows of the transposed-store launch's ``N``
#: by ``block_n`` columns of its ``M``.  The other widths run as transposed-store programs.
K1_ROW_MAJOR_BLOCK_N: tuple[int, ...] = (192,)
#: Accumulation chains of a one-split K1 program: with ``c`` chains each tile's K blocks are cut into ``c`` contiguous
#: ranges, each summed in a TMEM accumulator of its own from an ``init`` MMA, and the epilogue adds them once, in order
#: (no workspace, the same bits on every run).  This package ships the one-chain programs and, for the FP32 rows of
#: :data:`K1_CHAIN_RANGES_BY_SM`, three-chain programs (templates ``..._c3``).
K1_CHAINS_CHOICES: tuple[int, ...] = (1, 3)

# ---------------------------------------------------------------------------
# K1 planner: the measured verdicts
# ---------------------------------------------------------------------------
#
# A K1 launch of this package is the transposed-store GEMM ``out[l, n, m] = sum_k A[l, k, m] B[l, k, n]``: ``M`` is
# the inner (unit-stride) extent of the stored output and ``N`` its row count.  A weight gradient
# ``dW[d_out, d_in] = G[T, d_out].T @ X[T, d_in]`` has ``A = X``, ``B = G``, ``M = d_in``, ``N = d_out``, ``K = T``.

#: Per-row verdicts of the K1 weight gradient, ``(arch, L, M, N, out_dtype) -> (block_n, num_splits)`` (this
#: package's program with that tile width and split count; a width of :data:`K1_ROW_MAJOR_BLOCK_N` runs the row-major
#: program with the operands exchanged) or ``-> None`` (the deterministic program); ``M`` / ``N`` as above,
#: ``out_dtype`` ``"bf16"`` or ``"f32"``.  A row carries a program only where that program was measured
#: faster than the deterministic package's shipped plan of the row in every measurement run: a one-sided paired sign
#: test (p < 0.05) with a median time ratio below 0.99, in eager launches and in CUDA graph replays, at token counts
#: 16172 to 16384, and no more than 3 % slower than that plan next to a kernel holding 8, 16 or 32 SMs on another
#: stream.  A ``None`` entry is a measured row that missed; a row without an entry was not measured.  Both run the
#: deterministic program.  A verdict does not hold at the token counts of :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS` and,
#: where the token count spans an odd number of K blocks, of :data:`K1_SPLIT_ODD_BLOCK_EXCLUDED_TOKENS`.
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
    # sm_107a (216 SMs): two to four splits; the one-split rows store whole tiles (deterministic) and win where the
    # deterministic package's sm_107a program of the row is slower (the FP32 one-split rows up to 960 tokens only:
    # K1_SPLIT_RULE_EXCLUDED_TOKENS).  d_out 64 BF16 at d_in 4096 is not faster than the
    # deterministic plan in every run; d_in 4096 to d_out 512 BF16 (192 CTAs) and to d_out 4096 FP32 (216 CTAs) need
    # more SMs than a kernel holding 32 SMs leaves free and lose next to it
    ("sm_107a", 1, 4096, 32, "bf16"): (64, 4), ("sm_107a", 1, 4096, 32, "f32"): (64, 3),
    ("sm_107a", 1, 4096, 48, "f32"): (64, 3),
    ("sm_107a", 1, 4096, 64, "bf16"): None, ("sm_107a", 1, 4096, 64, "f32"): (64, 3),
    ("sm_107a", 1, 4096, 128, "bf16"): None, ("sm_107a", 1, 4096, 128, "f32"): None,
    ("sm_107a", 1, 4096, 512, "bf16"): None, ("sm_107a", 1, 4096, 512, "f32"): (128, 1),
    ("sm_107a", 1, 4096, 1024, "f32"): (256, 1),
    ("sm_107a", 1, 4096, 4096, "f32"): None,
    ("sm_107a", 1, 5120, 32, "bf16"): (64, 3), ("sm_107a", 1, 5120, 32, "f32"): (64, 3),
    ("sm_107a", 1, 5120, 48, "f32"): (64, 3),
    ("sm_107a", 1, 5120, 64, "f32"): (64, 3),
    ("sm_107a", 1, 5120, 512, "f32"): (128, 1),
    ("sm_107a", 1, 5120, 1024, "bf16"): (256, 1), ("sm_107a", 1, 5120, 1024, "f32"): (256, 1),
    ("sm_107a", 1, 6144, 32, "bf16"): (64, 3), ("sm_107a", 1, 6144, 32, "f32"): (64, 3),
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
#: table of :data:`K1_SPLIT_RULES_BY_SM` does not take its verdict, ``key -> ((sm_count, min_tokens, max_tokens),
#: ...)``: a token count (``K`` of the weight gradient) from
#: ``min_tokens`` to ``max_tokens`` (``None``: no upper bound) runs the deterministic program on a part of at least
#: ``sm_count`` SMs where ``sm_count`` is at or above the row's SM-count floor (:data:`K1_MIN_SM_COUNT`,
#: :data:`K1_RULE_MIN_SM_COUNT`), and on a part of exactly ``sm_count`` SMs below it (the parts of
#: :data:`K1_SPLIT_RULES_BY_SM`); every other token count and part keeps the row's verdict.  The verdicts were measured
#: at 16172 to 16384 tokens.  The sm_100a / sm_103a entries come from a re-measurement at 512, 1024, 2048, 4096, 8192,
#: 16231, 32768 and 65536 tokens on two 152-SM parts per row and token count (GB200, GB300; the strict runs of 8 GB300
#: points ran on two other parts, four GB300 parts in all; the protocol is in the K1 token table of the README).  A
#: measured token count is an entry where the split program failed in at least one run (the rule above; the backward
#: pass or the checkpointed step more than 1.3 % slower than with the deterministic program; or the deterministic
#: package's FP64 error budget missed with an error above the deterministic program's) and the deterministic program
#: leaves the backward pass and the checkpointed step no more than 1.3 % slower, or where the split program's FP64 error
#: (relative Frobenius error or BF16 error-budget ratio) exceeded the deterministic program's by more than 2 % for some
#: seed on both parts.  A token count between two consecutive measured entries is an entry too (not measured), and so is
#: every token count above 65536 of a row whose split program failed that FP64 comparison at 65536 tokens, and of
#: sm_103a ``4096 -> 32`` and ``4096 -> 48`` (BF16 and FP32): above 65536 tokens their two-split program's output is the
#: deterministic program's bits at an even number of 64-token K blocks and fails that comparison at an odd number (the
#: reason of their 148-SM entries in :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS_BY_SM`; the deterministic plan of these rows
#: is the same on 148 and 152 SMs; on a 148-SM part, 10 seeds: an FP32 relative Frobenius error 1.021 to 1.024 times
#: the deterministic program's at 65537 tokens; where the deterministic program misses the BF16 budget, a budget ratio
#: above 1.30 times the deterministic program's for some seed: ``4096 -> 32`` 1.305 at 65600 tokens, ``4096 -> 48`` up
#: to 1.379 at 65537 tokens).  The open-ended sm_100a BF16 entries come from that rule where the deterministic program
#: misses the BF16 budget for some seed (the rule of :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS_BY_SM`; 10 seeds per token
#: count on a 148-SM B200 part, and the deterministic plan of these rows is the same on 148 and 152 SMs): a budget ratio
#: above 1.30 times the deterministic program's for some seed from the first token count listed -- ``4096 -> 48`` and
#: ``5120 -> 48`` 1.473 and 1.320 at 49153 tokens, ``5120 -> 32`` 1.328 at 65537 tokens, ``4096 -> 32`` 1.305 at
#: 65600 tokens (up to 1.727 at 106497; the entry starts at 65536 as on sm_103a) -- and ``4096 -> 64`` a median ratio
#: of 1.041 over its odd token counts from 65537 tokens (152 SMs; the 148-SM entry is in
#: :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS_BY_SM`).  Unmeasured token counts keep the verdict: below 512, above 65536
#: (except the rows with open-ended entries), between two measured token counts with different outcomes, and on parts
#: with fewer than 152 SMs (148-SM parts: their own entries in :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS_BY_SM`).  The
#: sm_107a entries are single token counts of the deterministic package's sm_107a rules for one token count (16231 and
#: 16384 tokens; 16231 only for ``4096 -> 1024``), which give the row a faster deterministic plan there, and 16231
#: tokens of ``5120 -> 1024`` BF16 on 216 SMs (measured in the same runs; the deterministic plan there is the row's
#: plan at every token count).  The other single 216-SM
#: token counts (512 and 65536) come from a re-measurement of chosen points at 512, 8192, 32768 and 65536 tokens
#: on two 216-SM parts per point under the rule of the sm_100a / sm_103a entries (the protocol is in the sm_107a K1
#: token table of the README); their neighbouring token counts were not measured.  On 208, 212 and 216 SMs the window
#: entries come from a re-measurement at 16172, 16231 and 16384 tokens against those plans (the protocol is in the
#: sm_107a K1 token table of the README): a measured token count is an entry where the row's program failed the rule
#: above in some run, or left the backward pass or the checkpointed step more than 1.3 % slower than the deterministic
#: program (ratio of the medians over the runs, or median of the paired ratios, in eager launches or in graph
#: replays).  The open-ended sm_107a entries, and the 208-SM and 212-SM entries from 16172 to 16384 tokens, of the FP32
#: rows ``4096 -> 512``, ``4096 -> 1024``, ``5120 -> 512`` and ``5120 -> 1024`` come from neither rule: their one-split
#: program adds the whole token reduction of an output in one chain, while from 961 tokens (16 K blocks) up the
#: deterministic program of these rows cuts it into units of ``ceil(blocks / 2)`` K blocks (the two halves of each
#: output at an even block count) and adds the partials in a fixed order (its automatic two-way split).  From 961
#: tokens up the one-split program's FP64 relative Frobenius error is 2.0 to 2.4 times the deterministic program's on
#: every seed, above the 2 % the rule allows; up to 960 tokens the deterministic program adds one chain as well and the
#: two errors are equal.  Measured with the 216-SM plan on 212-SM R200 parts (the same program and accumulation order as
#: on 216 SMs, and the deterministic plan of these rows is the same on 208, 212 and 216 SMs; the error does not depend
#: on the SM count) against an FP64 reference: each row on two parts, 5 seeds, at 512 to 65536 tokens, and in a later
#: run with 10 seeds at 704 to 1024 tokens (the deterministic program's bits up to 960 tokens, 2.09 to 2.11 times its
#: error at 961 and 1024 tokens).  So these rows run the deterministic program from 961 tokens up on parts of at least
#: 216 SMs, and from 16172 to 16384 tokens on 208 and 212 SMs (the 212-SM two-split ranges of
#: :data:`K1_SPLIT_EVEN_BLOCK_RANGES_BY_SM` lie outside those tokens).
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
    ("sm_103a", 1, 5120, 32, "bf16"): ((152, 512, 512), (152, 32768, None)),
    ("sm_103a", 1, 5120, 32, "f32"): ((152, 512, 512),),
    ("sm_103a", 1, 5120, 48, "bf16"): ((152, 512, 2048), (152, 32768, None)),
    ("sm_103a", 1, 5120, 48, "f32"): ((152, 512, 2048),),
    ("sm_103a", 1, 5120, 64, "bf16"): ((152, 512, 2048), (152, 32768, None)),
    ("sm_103a", 1, 5120, 64, "f32"): ((152, 512, 2048),),
    ("sm_103a", 1, 5120, 128, "bf16"): ((152, 512, 8192), (152, 32768, None)),
    ("sm_103a", 1, 5120, 128, "f32"): ((152, 512, 4096),),
    ("sm_103a", 1, 6144, 32, "bf16"): ((152, 512, 512),),
    ("sm_103a", 1, 6144, 32, "f32"): ((152, 512, 512),),
    ("sm_103a", 1, 6144, 128, "bf16"): ((152, 512, 8192),),
    ("sm_103a", 1, 6144, 128, "f32"): ((152, 512, 8192),),
    ("sm_107a", 1, 4096, 32, "bf16"): ((216, 512, 512),),
    ("sm_107a", 1, 4096, 32, "f32"): ((216, 16231, 16231),),
    ("sm_107a", 1, 4096, 48, "f32"): (
        (208, 16231, 16231), (208, 16384, 16384), (216, 512, 512), (216, 16231, 16231),
        (216, 16384, 16384),
    ),
    ("sm_107a", 1, 4096, 64, "f32"): (
        (208, 16231, 16231), (208, 16384, 16384), (216, 512, 512), (216, 16231, 16231),
        (216, 16384, 16384),
    ),
    ("sm_107a", 1, 4096, 512, "f32"): ((208, 16172, 16384), (212, 16172, 16384), (216, 961, None)),
    ("sm_107a", 1, 4096, 1024, "f32"): ((208, 16172, 16384), (212, 16172, 16384), (216, 961, None)),
    ("sm_107a", 1, 5120, 32, "bf16"): ((216, 512, 512),),
    ("sm_107a", 1, 5120, 32, "f32"): ((216, 512, 512),),
    ("sm_107a", 1, 5120, 48, "f32"): (
        (208, 16231, 16231), (208, 16384, 16384), (212, 16231, 16231), (216, 512, 512),
        (216, 16231, 16231), (216, 16384, 16384),
    ),
    ("sm_107a", 1, 5120, 64, "f32"): (
        (208, 16231, 16231), (212, 16231, 16231), (216, 512, 512), (216, 16231, 16231),
        (216, 16384, 16384),
    ),
    ("sm_107a", 1, 5120, 512, "f32"): ((208, 16172, 16384), (212, 16172, 16384), (216, 961, None)),
    ("sm_107a", 1, 5120, 1024, "bf16"): ((216, 16231, 16231),),
    ("sm_107a", 1, 5120, 1024, "f32"): ((208, 16172, 16384), (212, 16172, 16384), (216, 961, None)),
    ("sm_107a", 1, 6144, 32, "bf16"): ((216, 512, 512),),
    ("sm_107a", 1, 8192, 128, "bf16"): ((216, 512, 512), (216, 65536, 65536)),
    ("sm_107a", 1, 8192, 128, "f32"): ((216, 512, 512), (216, 65536, 65536)),
}
# fmt: on
#: Token counts at which a row of :data:`K1_SPLIT_RULES` runs the deterministic program on the parts of exactly one SM
#: count, ``(arch, sm_count) -> {key: ((min_tokens, max_tokens), ...)}`` (``max_tokens`` ``None``: no upper bound).  An
#: entry of :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS` at or above a row's SM-count floor holds from its SM count up; these
#: hold on exactly their SM count, and every other part keeps its routing.  The sm_100a / sm_103a 148-SM entries (B200
#: / B300 parts) follow the rule of the 152-SM entries of :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS`, measured on 148-SM
#: parts at 512, 1024, 2048, 4096, 8192, 16231, 32768 and 65536 tokens (the protocol is in the 148-SM K1 token table of
#: the README): every point in a screen on one part, the points that failed there again on a second part (B300: every
#: point on a part of a second node too), and the FP64 error of 5 seeds on two parts at 32768 and 65536 tokens and at
#: the first and the last token count of every entry and one token either side of it (and at 98304 and 131072 tokens
#: for the entries that reach 65536 tokens).  A measured token count is an entry where the split program failed in some
#: run and the deterministic program leaves the backward pass and the checkpointed step no more than 1.3 % slower on
#: both parts (the worst of the ratio of the medians and the median of the paired ratios, eager launches and graph
#: replays), or where the split program failed the FP64 comparison: for some seed, a relative Frobenius error more than
#: 2 % above the deterministic program's, or, where the deterministic program meets the deterministic package's error
#: budget on every seed, an output outside it.  Where the deterministic program misses the budget for some seed (BF16
#: output at the largest token counts), the BF16 error-budget ratio, a maximum over the elements, moves by up to about
#: 25 % either way between the two programs at the same relative Frobenius error; there the split program fails where
#: the median over a row's measured token counts and seeds (10 seeds per token count) of its ratio to the deterministic
#: program's exceeds 1.02, or where one seed's ratio exceeds 1.30.  A token count between two consecutive measured
#: entries is an entry too.  Entries without an upper bound: sm_103a ``4096 -> 32`` and sm_100a ``4096 -> 64`` (BF16
#: and FP32) from 65536 tokens up: above 65536 tokens the split program's output differs from the deterministic
#: program's where the token count spans an odd number of 64-token blocks (65537, 65538, 65600 and 65665 tokens: an FP32
#: relative Frobenius error 2.1 to 2.3 % above on every seed, a BF16 budget ratio 1.305 times the deterministic
#: program's for one seed; sm_100a BF16 at 12 odd counts from 65600 to 98240 tokens, 10 seeds each: up to 1.65 times)
#: and is the deterministic program's bits where it spans an even number (65601 to 131072 tokens), where it does not
#: make the backward pass or the checkpointed step 1 % faster (sm_100a: the backward pass up to 3.5 % slower).
#: Unmeasured token counts keep the verdict.
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
    ("sm_103a", 148): {
        ("sm_103a", 1, 4096, 32, "bf16"): ((65536, None),),
        ("sm_103a", 1, 4096, 32, "f32"): ((65536, None),),
        ("sm_103a", 1, 5120, 48, "bf16"): ((512, 2048),),
        ("sm_103a", 1, 5120, 48, "f32"): ((512, 2048),),
        ("sm_103a", 1, 5120, 64, "bf16"): ((512, 1024),),
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
#: ranges, keeps the routing of the tables above.  In these ranges the deterministic program of the row cuts the token
#: reduction of its tiles, taken in order, into units of ``ceil(blocks / 2)`` K blocks (at an odd block count a unit
#: runs on into the next tile) and adds the partials in a fixed order: from 961 tokens (16 K blocks) up for
#: ``4096 -> 32``, ``4096 -> 48`` and ``4096 -> 64``; for sm_100a ``5120 -> 32`` and ``5120 -> 48`` from where one
#: wave's operands outgrow the L2 (from 12289 tokens with 126 MiB of L2, between 12292 and 12337 tokens on a B200;
#: below, whole tiles); for sm_107a ``8192 -> 128`` while they fit the L2 (to 7808 tokens with 126 MiB of L2, to
#: between 4097 and 8192 tokens on a 212-SM R200; above, a synchronised two-way split that the two-split program
#: matches).  At an even block count the two-split program's output is the deterministic program's bit for bit; at an
#: odd block count it is not, and its FP32 output's FP64 relative Frobenius error was more than the 2 % of the rule
#: above over the deterministic program's on every seed at every measured odd token count in the ranges from 1025 to
#: 65537 tokens (10 seeds each): sm_103a ``4096 -> 32`` and ``4096 -> 48`` and sm_100a ``4096 -> 32``, ``4096 -> 48``
#: and ``4096 -> 64`` 1.03 to 1.15 times at 1025 to 49153 tokens and 1.021 to 1.024 times at 65537 tokens (148-SM B300
#: and B200 parts; the deterministic plan of these rows is the same on 148 and 152 SMs); sm_100a ``5120 -> 32`` and
#: ``5120 -> 48`` 1.025 to 1.075 times at 12337 to 32769 tokens (B200); sm_107a ``8192 -> 128`` 1.04 to 1.14 times at
#: 1025 to 4097 tokens (212-SM R200, the 212-SM and the 216-SM plan; the deterministic plan of the row is the same on
#: 212 and 216 SMs).  The excess falls with the token count (``5120 -> 32`` and ``5120 -> 48``: 1.7 to 1.96 % at 49153
#: tokens, 0.7 to 1.5 % at 65537 to 131073 tokens; the ``4096`` rows: 1.2 to 1.7 % at 98305 and 131073 tokens); their
#: ranges stay open because the two-split program is not more accurate than the deterministic program at any odd block
#: count in them.  The BF16 outputs of these rows had the deterministic program's relative Frobenius error and met the
#: error budget on every seed where the deterministic program met it.
# fmt: off
K1_SPLIT_ODD_BLOCK_EXCLUDED_TOKENS: dict[
    tuple[str, int, int, int, str], tuple[tuple[int, int, Optional[int]], ...]
] = {
    ("sm_100a", 1, 4096, 32, "f32"): ((148, 1025, None),),
    ("sm_100a", 1, 4096, 48, "f32"): ((148, 1025, None),),
    ("sm_100a", 1, 4096, 64, "f32"): ((148, 1025, None),),
    ("sm_100a", 1, 5120, 32, "f32"): ((148, 12289, None),),
    ("sm_100a", 1, 5120, 48, "f32"): ((148, 12289, None),),
    ("sm_103a", 1, 4096, 32, "f32"): ((148, 1025, None),),
    ("sm_103a", 1, 4096, 48, "f32"): ((152, 1025, None),),
    ("sm_107a", 1, 8192, 128, "f32"): ((212, 1025, 8192), (216, 1025, 8192)),
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
#: token counts of :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS` for a part's SM count run the deterministic program.
# fmt: off
K1_SPLIT_RULES_BY_SM: dict[
    tuple[str, int], dict[tuple[str, int, int, int, str], Optional[tuple[int, int]]]
] = {
    ("sm_107a", 208): {
        ("sm_107a", 1, 4096, 32, "bf16"): None, ("sm_107a", 1, 4096, 32, "f32"): (64, 3),
        ("sm_107a", 1, 4096, 48, "f32"): (64, 3),
        ("sm_107a", 1, 4096, 64, "bf16"): None, ("sm_107a", 1, 4096, 64, "f32"): (64, 3),
        ("sm_107a", 1, 4096, 512, "bf16"): None, ("sm_107a", 1, 4096, 512, "f32"): (128, 1),
        ("sm_107a", 1, 4096, 1024, "f32"): (256, 1),
        ("sm_107a", 1, 4096, 4096, "f32"): None,
        ("sm_107a", 1, 5120, 32, "bf16"): None, ("sm_107a", 1, 5120, 32, "f32"): (64, 3),
        ("sm_107a", 1, 5120, 48, "f32"): (64, 3),
        ("sm_107a", 1, 5120, 64, "f32"): (64, 3),
        ("sm_107a", 1, 5120, 512, "f32"): (128, 1),
        ("sm_107a", 1, 5120, 1024, "bf16"): None, ("sm_107a", 1, 5120, 1024, "f32"): (256, 1),
        ("sm_107a", 1, 6144, 32, "bf16"): None, ("sm_107a", 1, 6144, 32, "f32"): None,
        ("sm_107a", 1, 8192, 128, "bf16"): None, ("sm_107a", 1, 8192, 128, "f32"): None,
    },
    ("sm_107a", 212): {
        ("sm_107a", 1, 4096, 32, "bf16"): None, ("sm_107a", 1, 4096, 32, "f32"): (64, 3),
        ("sm_107a", 1, 4096, 48, "f32"): (64, 3),
        ("sm_107a", 1, 4096, 64, "bf16"): None, ("sm_107a", 1, 4096, 64, "f32"): (64, 3),
        ("sm_107a", 1, 4096, 512, "bf16"): None, ("sm_107a", 1, 4096, 512, "f32"): (128, 1),
        ("sm_107a", 1, 4096, 1024, "f32"): (192, 1),
        ("sm_107a", 1, 4096, 4096, "f32"): None,
        ("sm_107a", 1, 5120, 32, "bf16"): None, ("sm_107a", 1, 5120, 32, "f32"): (64, 3),
        ("sm_107a", 1, 5120, 48, "f32"): (64, 3),
        ("sm_107a", 1, 5120, 64, "f32"): (64, 3),
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
#: package's FP64 error budget with an error (relative Frobenius error or BF16 error-budget ratio) no more than 2 %
#: above the deterministic program's for every seed on both parts.  A range runs from such a measured token count to
#: the window's first token count, or from the window's last one to such a measured token count, and holds no measured
#: token count that failed; the token counts between two consecutive measured ones inside it take the verdict too (not
#: measured).  The token counts below a range's first measured token count, above its last, and those between a
#: measured token count that passed and one that failed run the deterministic program.  On 212 SMs: the FP32 rows
#: ``4096 -> 32, 48, 64`` and ``5120 -> 32, 48, 64`` from 1024 to 65536 tokens (at 512 not faster in every run),
#: ``8192 -> 128`` from 1024 to 32768 tokens (at 512 slower, at 65536 not faster in every run), and
#: the FP32 rows ``4096 / 5120 -> 512 / 1024`` at 512 tokens only: from 1024 tokens up their one-split program, which
#: adds an output's whole token reduction in one chain, has about twice the deterministic program's FP64 error (their
#: two-split program runs at the ranges of :data:`K1_SPLIT_EVEN_BLOCK_RANGES_BY_SM`).
# fmt: off
K1_SPLIT_RULES_BY_SM_TOKEN_RANGES: dict[
    tuple[str, int], dict[tuple[str, int, int, int, str], tuple[tuple[int, int], ...]]
] = {
    ("sm_107a", 212): {
        ("sm_107a", 1, 4096, 32, "f32"): ((1024, 16172), (16384, 65536)),
        ("sm_107a", 1, 4096, 48, "f32"): ((1024, 16172), (16384, 65536)),
        ("sm_107a", 1, 4096, 64, "f32"): ((1024, 16172), (16384, 65536)),
        ("sm_107a", 1, 4096, 512, "f32"): ((512, 512),),
        ("sm_107a", 1, 4096, 1024, "f32"): ((512, 512),),
        ("sm_107a", 1, 5120, 32, "f32"): ((1024, 16172), (16384, 65536)),
        ("sm_107a", 1, 5120, 48, "f32"): ((1024, 16172), (16384, 65536)),
        ("sm_107a", 1, 5120, 64, "f32"): ((1024, 16172), (16384, 65536)),
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
#: Verdicts of rows on a part of exactly one SM count at further token ranges that hold only where the token count spans
#: an even number of 64-element K blocks (``ceil(K / 64)`` even),
#: ``(arch, sm_count) -> {key: ((min_tokens, max_tokens, (block_n, num_splits)), ...)}`` (keys and values as in
#: :data:`K1_SPLIT_RULES`): on a part of exactly ``sm_count`` SMs, below the row's SM-count floor or at or above it, a
#: token count from ``min_tokens`` to ``max_tokens`` with an even block count runs that range's program, except the
#: token counts of :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS` for the part and the token counts of the row's ranges in
#: :data:`K1_CHAIN_RANGES_BY_SM` for the part, which come first.  A token count with an odd block count, and every
#: other token count, part and row, keeps the routing of the tables above.  A range lies outside
#: :data:`K1_SPLIT_RULES_BY_SM_TOKENS` and outside the row's ranges in :data:`K1_SPLIT_RULES_BY_SM_TOKEN_RANGES` and
#: :data:`K1_SPLIT_RANGE_OVERRIDES_BY_SM`.  On 152-SM sm_100a and sm_103a parts the FP32 row ``4096 -> 128`` (no verdict
#: in :data:`K1_SPLIT_RULES`) runs two splits of 64-wide tiles, each adding half of an output's K blocks: at an even
#: block count its output was the deterministic program's bit for bit (5 seeds at 1024 to 8192 tokens), at an odd count
#: (1088 and 2112 tokens) it was not, with an FP64 relative Frobenius error 1.107 and 1.142 times the deterministic
#: program's.  It passed the K1 rule (the README) at 1024, 2048 and 4096 tokens on two parts of each GPU (0.70 to 0.79
#: times the deterministic program's time in CUDA graph replays), and at 6144 tokens on two sm_103a parts; on the
#: sm_100a part measured at 6144 tokens its time next to a kernel holding SMs on another stream was 1.099 times the
#: deterministic program's, and at 512 and 8192 tokens it failed on every part.  148-SM parts keep the deterministic
#: program.  On 212-SM sm_107a parts the same row (no verdict in the part's table of :data:`K1_SPLIT_RULES_BY_SM`
#: either) runs that program from 1024 to 7680 tokens at an even block count: its output was the deterministic program's
#: bit for bit (5 seeds at 1024, 2048, 4096, 6144, 6656, 7168 and 7680 tokens), at an odd count (1088 to 5000 tokens)
#: its FP64 error was 1.11 to 1.16 times the deterministic program's; it passed the K1 rule there on two nodes of each
#: of two R200 systems (0.70 to 0.76 times the deterministic program's time), and failed at 512, 768 and 896 tokens;
#: from 8000 tokens the deterministic plan changes and the two-split error was 1.54 times its error, and at 12288 and
#: 16384 tokens it was 5 to 7 % slower.  On 212 SMs the FP32 rows ``4096 / 5120 -> 512 / 1024`` run two splits of their
#: 212-SM tile width (128, 192 for ``4096 -> 1024``, 256 for ``5120 -> 1024``), each adding half of an output's K
#: blocks.  From 1024 tokens up, at an even block count, their output was the deterministic program's bit for bit (5
#: seeds on two parts at 1024, 2048, 4096, 8192, 16231, 16384, 32768 and 65536 tokens, on one part at 6000, 12000 and
#: 48000); at an odd count from 17 blocks up it was not, with an FP64 relative Frobenius error 1.01 to 1.17 times the
#: deterministic program's (5 seeds, 1025 to 65537 tokens).  Measured as the ranges of
#: :data:`K1_SPLIT_RULES_BY_SM_TOKEN_RANGES`, at 512, 1024, 2048, 4096, 8192, 16231, 16384, 32768 and 65536 tokens (the
#: protocol and the rule are in the 212-SM K1 token-range table of the README): a range runs between measured token
#: counts at which the two-split program passed, and holds none that failed.  At 512 tokens two splits are not faster
#: than the deterministic plan, ``4096 -> 1024`` is not faster in every run at 1024 tokens, and ``5120 -> 1024`` failed
#: the rule at every measured token count from 16231 up.
# fmt: off
K1_SPLIT_EVEN_BLOCK_RANGES_BY_SM: dict[
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
#: without an entry keep the verdict of :data:`K1_SPLIT_RULES`.  No part has an entry: the one measured verdict, the
#: 192-wide one-split program of ``4096 -> 1024`` FP32 on 216 SMs, fails the FP64 comparison of the sm_107a FP32
#: one-split rows from 961 tokens up (:data:`K1_SPLIT_RULE_EXCLUDED_TOKENS`), and the row runs the deterministic program
#: at these token counts.
K1_SPLIT_OVERRIDES_BY_SM: dict[
    tuple[str, int], dict[tuple[str, int, int, int, str], Optional[tuple[int, int]]]
] = {}
#: Token ranges at which a row runs this package's one-split program with several accumulation chains on the parts of
#: exactly one SM count, ``(arch, sm_count) -> {key: ((min_tokens, max_tokens, (block_n, chains)), ...)}`` (keys as in
#: :data:`K1_SPLIT_RULES`): a token count from ``min_tokens`` to ``max_tokens`` runs the program of that tile width at
#: one split with that many chains (:func:`k1_chain_plan`) in place of the routing of the tables above, whose entries
#: for these rows keep their one-chain program off these token counts on accuracy, and in place of the two splits of
#: :data:`K1_SPLIT_EVEN_BLOCK_RANGES_BY_SM` at an even number of 64-element K blocks: at the even block counts of those
#: ranges both programs were timed at on 212 SMs (1024 to 8192 and 32768 to 40960 tokens), three chains ran the weight
#: gradient 7 to 52 % and the checkpointed step 2 to 21 % faster than two splits (medians, eager and graph), with an
#: FP64 error below the deterministic program's, which two splits match bit for bit.  The FP32 rows
#: ``4096 / 5120 -> 512 / 1024``: from 961 tokens up their one-chain program adds an output's whole token reduction in
#: one chain, about twice the deterministic program's FP64 error; three chains of a third of the K blocks each had an
#: FP64 relative Frobenius error 0.31 to 0.73 times the deterministic program's on every seed (5 seeds on a 212-SM R200
#: part at 960, 961, 1024, 2048, 4096, 8192, 16172, 16231, 16384, 32768 and 40961 tokens, even and odd block counts; the
#: error does not depend on the SM count) and ran within 5 % of the one-chain program.  Measured as the ranges of
#: :data:`K1_SPLIT_RULES_BY_SM_TOKEN_RANGES` (the rule, the bound on the backward pass and the checkpointed step, and
#: the construction of a range are that table's): on 212 SMs in two jobs on four parts of one node each, at 960, 4096,
#: 16172, 16384, 32768 and 40961 tokens and at 960, 1024, 2048, 4096, 4097, 8192, 16172, 16384, 32768, 32769, 40960 and
#: 40961 tokens; on 216 SMs at 960, 961, 1024, 2048, 4096, 8192, 16172, 16231, 16384, 32768 and 40961 tokens on one
#: part.  Up to 960 tokens on 216 SMs the one-chain program keeps its :data:`K1_SPLIT_RULES` verdict (the deterministic
#: program adds one chain there too).  ``4096 -> 512`` and ``5120 -> 512`` are not faster than the deterministic plan in
#: every run at 16384 tokens on 212 SMs, nor are ``4096 -> 512`` and ``4096 -> 1024`` at 16231 and 16384 tokens on 216
#: SMs; ``4096 -> 1024`` at 16384 tokens on 212 SMs (in one of the two jobs) and ``5120 -> 1024`` at 32768 tokens on 216
#: SMs left the checkpointed step more than 1.3 % slower than the deterministic program.
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


def k1_splits_fit(splits: int, K: int) -> bool:
    """Whether ``splits`` K ranges each get at least one 64-element K block of ``K`` (``splits * 64 <= K + 63``,
    the program's requirement)."""
    return int(splits) >= 1 and int(splits) * K1_BLOCK_K <= int(K) + K1_BLOCK_K - 1


def k1_chains_fit(chains: int, K: int) -> bool:
    """Whether ``chains`` accumulation chains each get at least one 64-element K block of ``K`` (``chains * 64 <= K +
    63``, the program's requirement)."""
    return int(chains) >= 1 and int(chains) * K1_BLOCK_K <= int(K) + K1_BLOCK_K - 1


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
    """``(block_n, num_splits)`` of this package's K1 program for the transposed-store launch ``out[l, n, m]``
    (``M``, ``N`` as in the comment above; ``K`` the token count) on a device of ``sm_count`` SMs, or ``None``: the
    row runs the deterministic program.  The row's verdict in :data:`K1_SPLIT_RULES` from the SM-count floor of the
    row (:data:`K1_MIN_SM_COUNT`, :data:`K1_RULE_MIN_SM_COUNT`) up, replaced at a ``K`` of
    :data:`K1_SPLIT_RULES_BY_SM_TOKENS` by the row's entry in the table of :data:`K1_SPLIT_OVERRIDES_BY_SM` for exactly
    ``sm_count``, if any, and at a ``K`` of the row's range in :data:`K1_SPLIT_EVEN_BLOCK_RANGES_BY_SM` for exactly
    ``sm_count`` that spans an even number of 64-element K blocks by that range's verdict; below the floor the row's
    verdict in the table of :data:`K1_SPLIT_RULES_BY_SM` for exactly ``sm_count``, only at a ``K`` of
    :data:`K1_SPLIT_RULES_BY_SM_TOKENS` or of the row's ranges in :data:`K1_SPLIT_RULES_BY_SM_TOKEN_RANGES` for that
    SM count, and the verdict of the row's range in :data:`K1_SPLIT_RANGE_OVERRIDES_BY_SM` for that SM count at a
    ``K`` of that range, else the verdict of the row's range in :data:`K1_SPLIT_EVEN_BLOCK_RANGES_BY_SM` for that SM
    count at a ``K`` of that range that spans an even number of 64-element K blocks; either except at a ``K`` of the
    row's entries in :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS` for this part (from the entry's SM count up where that count
    is at or above the floor, exactly that SM count below it), in :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS_BY_SM` for
    exactly ``sm_count``, or, at a ``K`` that spans an odd number of 64-element K blocks, in
    :data:`K1_SPLIT_ODD_BLOCK_EXCLUDED_TOKENS` for this part (as the first table).  A row without a verdict, a ``K`` of
    those entries, every other ``K`` below the floor and a ``K`` too short for the split count (:func:`k1_splits_fit`)
    run the deterministic program.  A ``K`` of the row's ranges in :data:`K1_CHAIN_RANGES_BY_SM` for exactly
    ``sm_count`` (:func:`k1_chain_plan`) returns that range's tile width at one split before any of these tables: the
    program of :func:`k1_chain_plan`'s chain count."""
    chained = k1_chain_plan(
        arch=arch, L=L, M=M, N=N, K=K, out_f32=out_f32, sm_count=sm_count
    )
    if chained is not None:
        return chained[0], 1
    key = (arch, int(L), int(M), int(N), "f32" if out_f32 else "bf16")
    floor = max(K1_MIN_SM_COUNT.get(arch, 0), K1_RULE_MIN_SM_COUNT.get(key, 0))
    first, last = K1_SPLIT_RULES_BY_SM_TOKENS
    in_window = first <= int(K) <= last
    if int(sm_count) >= floor:
        choice = K1_SPLIT_RULES.get(key)
        if in_window:
            overrides = K1_SPLIT_OVERRIDES_BY_SM.get((arch, int(sm_count)), {})
            choice = overrides.get(key, choice)
        if _ceil_div(K, K1_BLOCK_K) % 2 == 0:
            even_part = K1_SPLIT_EVEN_BLOCK_RANGES_BY_SM.get((arch, int(sm_count)), {})
            own = [v for lo, hi, v in even_part.get(key, ()) if lo <= int(K) <= hi]
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
        if not own and _ceil_div(K, K1_BLOCK_K) % 2 == 0:
            even = K1_SPLIT_EVEN_BLOCK_RANGES_BY_SM.get(part, {}).get(key, ())
            own = [v for lo, hi, v in even if lo <= int(K) <= hi]
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


def atomic_k1_template(block_n: int, out_f32: bool, chains: int = 1) -> str:
    """The registry template of this package's K1 program (both operands MN-major): the transposed-store program of
    the tile width, or the row-major-store one for a width of :data:`K1_ROW_MAJOR_BLOCK_N`; ``_c<chains>`` for more
    than one accumulation chain."""
    return (
        f"{cake_jit.TEMPLATE_PREFIXES[0]}nn_n{int(block_n)}"
        + ("_f32" if out_f32 else "")
        + ("" if int(block_n) in K1_ROW_MAJOR_BLOCK_N else "_t")
        + (f"_c{int(chains)}" if int(chains) > 1 else "")
    )


# ---------------------------------------------------------------------------
# K1 planning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AtomicGemmPlan:
    """The host plan of one K1 launch of this package: the transposed-store GEMM ``out[l, n, m] = sum_k A[l, k, m]
    B[l, k, n]`` (``out_t``), or the row-major-store GEMM ``out[l, m, n]`` (``out_t`` false: the widths of
    :data:`K1_ROW_MAJOR_BLOCK_N`), over the launch's ``[L, K, M]`` / ``[L, K, N]`` operand views, ``L x m_tiles x
    n_tiles`` tiles of ``256 x block_n`` each cut into ``splits`` K ranges (or, at one split, its K blocks into
    ``chains`` accumulation chains), walked persistently by at most ``sm_pairs`` CTA pairs.  ``M`` and ``N`` are the
    launch's.  ``transposed_out`` is the caller's: the launch takes the caller's
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
    def grid(self) -> tuple[int, int, int]:
        return (2 * min(self.items, self.sm_pairs), 1, 1)

    @property
    def ws_f32_elems(self) -> int:
        """FP32 elements of the split workspace: one ``256 x block_n`` slot per tile and split (none with one
        split).  A launch writes every slot it reads except the padding columns of a partial tile (``N`` not a
        multiple of ``block_n``), which the last split reads and discards (they never reach ``out``); the workspace
        is zeroed when allocated, so no read is of uninitialised memory, and needs no memset per launch."""
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
    chains: Optional[int] = None,
) -> tuple[Optional[AtomicGemmPlan], Any, Any, Any, str]:
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
    registry allow it.  The planner's choice runs :func:`k1_chain_plan`'s accumulation chains where it has them; an
    explicit choice runs one chain unless ``chains`` (one of :data:`K1_CHAINS_CHOICES`) names more, at one split.
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
    c = 1 if chains is None else int(chains)
    if splits is None and block_n is None and chains is None:
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
                f"dense_projection_gemm: {c} chains run at one split (got {p}) with a 64-element K block each "
                f"(K={K})"
            )
        if bn not in K1_BLOCK_N_CHOICES:
            raise ValueError(f"block_n is one of {K1_BLOCK_N_CHOICES}, got {block_n}")
        if not k1_splits_fit(p, K) or p > 65536:
            raise ValueError(
                f"dense_projection_gemm: splits must be in [1, {_ceil_div(K, K1_BLOCK_K)}] (the 64-element K blocks "
                f"of K={K}), got {p}"
            )
    template = atomic_k1_template(bn, out_f32, c)
    if template not in cake_jit.KERNELS.get(arch, {}):
        return (*none, f"{template} is not registered for {arch}")
    out_t = bn not in K1_ROW_MAJOR_BLOCK_N
    if not out_t:
        # the row-major-store launch: the transposed-store launch with its operands and extents exchanged
        a_launch, b_launch, M_, N_ = b_launch, a_launch, N_, M_
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
        out_t=out_t,
        chains=c,
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
    until :func:`clear_convenience_caches`.  A copy (``dict(table)``) is a plain ``dict``."""

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
#: row's layout on the CUDA cores (:func:`router_simt`); ``None`` runs the deterministic program (its production
#: split count).  A rule is ``None``, token windows ``((min_tokens, max_tokens, spec), ...)`` -- the spec of the window
#: that holds the launch's token count (``K`` of the weight gradient, ``M`` of the forward and of the input gradient);
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
#: table's windows do not hold, the token count takes the next window that holds it.
# fmt: off
ROUTER_RULES: dict[
    tuple[str, str, int, int], Optional[Union[str, int, tuple[tuple[int, int, Union[str, int]], ...]]]
] = _VerdictTable({
    # the forward: no partition replay-safe under recompute (at most two partials per element) holds its margin next
    # to a resident kernel; the deterministic plan (four splits) on every architecture
    ("sm_100a", "kk", 4096, 256): None, ("sm_100a", "kk", 5120, 256): None, ("sm_100a", "kk", 6144, 256): None,
    ("sm_103a", "kk", 4096, 256): None, ("sm_103a", "kk", 5120, 256): None, ("sm_103a", "kk", 6144, 256): None,
    ("sm_107a", "kk", 4096, 256): None, ("sm_107a", "kk", 5120, 256): None, ("sm_107a", "kk", 6144, 256): None,
    # the weight gradient, measured on 152-SM GB200 (sm_100a) / GB300 (sm_103a) and 216-SM R200 (sm_107a) parts.  A
    # window keeps the deterministic package's router budget: for every seed (10 seeds per token count, at the first,
    # the last and inner token counts of the window) a relative Frobenius error against FP64 of the exact operands,
    # worst of 20 runs, at most 2e-6.  Its ratio to the error of the plan the row takes without the window is reported
    # data, not a gate; the ratios below are the reasons these rows were left out under the earlier 1.02 rule.
    # Hidden 5120 on sm_100a / sm_103a: "waves" at few tokens (faster than the deterministic plan at 64 and 128 tokens,
    # and on sm_103a at 192; more than 3 % slower than it next to a resident kernel at every measured count from 224
    # to 10240 tokens), on sm_103a up to 160 tokens: from 161 tokens (six K blocks, more (tile, K block) steps than
    # units) its error was 1.026 to 1.040 times the deterministic plan's.
    # Faster than the deterministic plan but with more error, so these rows run the deterministic plan: the SIMT
    # program on 152-SM sm_100a parts (hidden 4096 from 1 to 224 tokens, hidden 5120 and 6144 from 1 to 256: 2.23 to
    # 2.54 times the error of the plan without the window at 113 to 256 tokens, the FP32 products of each slice summed
    # in one FFMA chain along the tokens); the head/tail program at hidden 4096 on sm_100a ("H1x5p44", 10240 to 16384
    # tokens: 1.18 to 1.20 times) and on sm_103a ("H1x4p44", 20480 to 49152 tokens: 1.20 to 1.29 times), hidden 5120
    # ("H0.5x3", sm_100a 10240 to 32768 tokens: up to 1.027 times at 10240 and 16384; sm_103a 16172 to 32768 tokens:
    # up to 1.021 times at 24576) and hidden 6144 on sm_103a ("H1x5p44", 16172 to 20480 tokens: 1.35 to 1.40 times).
    # sm_107a: hidden 5120 "H0.5x4" (four tail units, 108 head and 432 tail units on 216 SMs; measured on two parts at
    # 16172, 16231 and 16384 tokens); hidden 4096 and 6144 keep the deterministic plan: at 4096 "H0.5x4" was up to
    # 7 % slower than that plan when a kernel holding 16 SMs on another stream started during it (the kernel takes SMs
    # the head spans wait for), at 6144 it left the backward pass no faster
    ("sm_100a", "nn_t", 4096, 256): None,
    ("sm_100a", "nn_t", 5120, 256): ((1, 128, "waves"),),
    ("sm_100a", "nn_t", 6144, 256): None,
    ("sm_103a", "nn_t", 4096, 256): None,
    ("sm_103a", "nn_t", 5120, 256): ((1, 160, "waves"),),
    ("sm_103a", "nn_t", 6144, 256): None,
    ("sm_107a", "nn_t", 4096, 256): None,
    ("sm_107a", "nn_t", 5120, 256): ((16172, 16384, "H0.5x4"),),
    ("sm_107a", "nn_t", 6144, 256): None,
    # the input gradient, measured on 152-SM GB200 (sm_100a) parts at 1, 17, 32, 64, 96 and 128 tokens at hidden 4096
    # and 5120 and at 1, 17, 32 and 64 tokens at hidden 6144: the SIMT program with the 32 x 64 tile in four slices was
    # faster than the deterministic program up to 128 tokens at hidden 4096 and 5120 and up to 64 at hidden 6144, but
    # its relative Frobenius error against FP64 was 1.66 to 2.07 times the deterministic program's (10 seeds per token
    # count; the FP32 products of each slice summed in one FFMA chain), so these rows run the deterministic
    # program.  sm_103a and 216-SM sm_107a parts keep the deterministic program (not measured)
    ("sm_100a", "kn", 4096, 256): None,
    ("sm_100a", "kn", 5120, 256): None,
    ("sm_100a", "kn", 6144, 256): None,
})
# fmt: on
#: The SM count the verdicts of :data:`ROUTER_RULES` were measured on, per architecture (152-SM GB200 / GB300, 216-SM
#: R200): on a device with any other SM count a router row runs the deterministic program, except in a ``"waves"``
#: window, which keeps the condition of :func:`router_waves` on every SM count (the plan of these rows before the
#: windows).  An explicit ``splits`` is not affected.
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
#: program was up to 6.2 % slower next to a resident kernel), as do hidden 6144 on 208 SMs and the forward, as on
#: 216 SMs.  The "H0.5x4" windows of 212 SMs keep the FP64 error rule of :data:`ROUTER_RULES` (0.94 to 0.98 times the
#: deterministic plan's error at 16172, 16231 and 16384 tokens).  A part's table may also hold rows
#: :data:`ROUTER_RULES` has no entry for, measured on that SM count only.  Faster than the deterministic plan but with
#: more error, so these rows run the deterministic plan on 212 SMs: the weight gradient's SIMT program with the
#: 128 x 64 tile (hidden 4096 and 5120 from 1 to 256 tokens, hidden 6144 from 1 to 192: 1.99 to 2.66 times the
#: deterministic plan's error at 97 to 256 tokens) and the input gradient's SIMT programs (the 32 x 64 tile in eight
#: slices up to 64 tokens and, at hidden 4096 and 5120, the 64 x 64 tile in four slices from 65 to 128 tokens: 1.30 to
#: 1.86 times the deterministic program's error).
#: The 148-SM B200 (sm_100a) and B300 (sm_103a) tables hold the compensated SIMT windows (specs
#: ``"simt_n<tile_n>m<tile_m>s<slices>g<group>"`` / ``"...dot2"``, :func:`router_simt_template`), measured on one node
#: of each part under the rule of :data:`ROUTER_RULES` (both launch orders, eager and CUDA graphs, next to a kernel
#: holding 8, 16 or 32 SMs, the backward pass and the checkpointed step, each window at its first, last and inner token
#: counts).
#: The weight gradient takes the 64 x 64 tile in two slices in the Dot2 form from 1 to 17 tokens and with one Kahan step
#: per product from 18 to 32; from 33 tokens Kahan groups of four K steps, the 64 x 64 tile to 160 tokens at hidden 5120
#: and 6144, and at hidden 4096 the 128 x 64 tile from 33 to 48 and 65 to 112 tokens and the 64 x 64 tile from 49 to 64
#: and 113 to 160.  The input gradient takes the 32 x 64 tile in four slices with Kahan groups of four K steps up to 32
#: tokens.  Every window's program keeps the router budget: a relative Frobenius error against FP64 of at most 5.2e-8
#: at the 105 measured token counts of each part's windows (10 seeds, worst of 20 runs; 0.55 to 1.009 times the
#: deterministic program's).  Past the windows the programs missed the rule: the input gradient at 33 tokens (3.3 %
#: slower next to a resident kernel at hidden 4096; the backward pass or the checkpointed step 1.3 to 2.7 % slower at
#: hidden 5120 and 6144), the weight gradient at 176 tokens (3.6 to 6.0 % slower next to a resident kernel at hidden
#: 4096 and 6144; the backward pass or the checkpointed step 2.4 to 3.5 % slower at hidden 5120, 4.0 % at 192 tokens).
#: Every other row and token count of these parts runs the deterministic program.
# fmt: off
ROUTER_RULES_BY_SM: dict[
    tuple[str, int],
    dict[tuple[str, str, int, int], Optional[Union[str, int, tuple[tuple[int, int, Union[str, int]], ...]]]],
] = _VerdictTable({
    ("sm_100a", 148): _VerdictTable({
        ("sm_100a", "kk", 4096, 256): None, ("sm_100a", "kk", 5120, 256): None, ("sm_100a", "kk", 6144, 256): None,
        ("sm_100a", "nn_t", 4096, 256): (
            (1, 17, "simt_n64m64s2dot2"), (18, 32, "simt_n64m64s2g1"), (33, 48, "simt_n128m64s2g4"),
            (49, 64, "simt_n64m64s2g4"), (65, 112, "simt_n128m64s2g4"), (113, 160, "simt_n64m64s2g4"),
        ),
        ("sm_100a", "nn_t", 5120, 256): (
            (1, 17, "simt_n64m64s2dot2"), (18, 32, "simt_n64m64s2g1"), (33, 160, "simt_n64m64s2g4"),
        ),
        ("sm_100a", "nn_t", 6144, 256): (
            (1, 17, "simt_n64m64s2dot2"), (18, 32, "simt_n64m64s2g1"), (33, 160, "simt_n64m64s2g4"),
        ),
        ("sm_100a", "kn", 4096, 256): ((1, 32, "simt_n64m32s4g4"),),
        ("sm_100a", "kn", 5120, 256): ((1, 32, "simt_n64m32s4g4"),),
        ("sm_100a", "kn", 6144, 256): ((1, 32, "simt_n64m32s4g4"),),
    }),
    ("sm_103a", 148): _VerdictTable({
        ("sm_103a", "kk", 4096, 256): None, ("sm_103a", "kk", 5120, 256): None, ("sm_103a", "kk", 6144, 256): None,
        ("sm_103a", "nn_t", 4096, 256): (
            (1, 17, "simt_n64m64s2dot2"), (18, 32, "simt_n64m64s2g1"), (33, 48, "simt_n128m64s2g4"),
            (49, 64, "simt_n64m64s2g4"), (65, 112, "simt_n128m64s2g4"), (113, 160, "simt_n64m64s2g4"),
        ),
        ("sm_103a", "nn_t", 5120, 256): (
            (1, 17, "simt_n64m64s2dot2"), (18, 32, "simt_n64m64s2g1"), (33, 160, "simt_n64m64s2g4"),
        ),
        ("sm_103a", "nn_t", 6144, 256): (
            (1, 17, "simt_n64m64s2dot2"), (18, 32, "simt_n64m64s2g1"), (33, 160, "simt_n64m64s2g4"),
        ),
        ("sm_103a", "kn", 4096, 256): ((1, 32, "simt_n64m32s4g4"),),
        ("sm_103a", "kn", 5120, 256): ((1, 32, "simt_n64m32s4g4"),),
        ("sm_103a", "kn", 6144, 256): ((1, 32, "simt_n64m32s4g4"),),
    }),
    ("sm_107a", 208): _VerdictTable({
        ("sm_107a", "kk", 4096, 256): None, ("sm_107a", "kk", 5120, 256): None, ("sm_107a", "kk", 6144, 256): None,
        ("sm_107a", "nn_t", 4096, 256): None,
        ("sm_107a", "nn_t", 5120, 256): ((16172, 16384, "H0.5x4"),),
        ("sm_107a", "nn_t", 6144, 256): None,
    }),
    ("sm_107a", 212): _VerdictTable({
        ("sm_107a", "kk", 4096, 256): None, ("sm_107a", "kk", 5120, 256): None, ("sm_107a", "kk", 6144, 256): None,
        ("sm_107a", "nn_t", 4096, 256): None,
        ("sm_107a", "nn_t", 5120, 256): ((16172, 16384, "H0.5x4"),),
        ("sm_107a", "nn_t", 6144, 256): ((16172, 16384, "H0.5x4"),),
        ("sm_107a", "kn", 4096, 256): None,
        ("sm_107a", "kn", 5120, 256): None,
        ("sm_107a", "kn", 6144, 256): None,
    }),
})
# fmt: on
#: The template suffix of the weight gradient's head/tail program (``router_fp32_gemm_atomic_nn_t`` + suffix): the
#: head/tail partition on three TMEM chunk buffers, with register reallocation between its warpgroups.
ROUTER_HEAD_TAIL_SUFFIX = "_ht_c3_nreg"
#: The weight gradient's head/tail program of an architecture that runs another one than
#: ``router_fp32_gemm_atomic_nn_t`` + :data:`ROUTER_HEAD_TAIL_SUFFIX`: on ``sm_103a``, the head/tail program with the
#: BF16 parts of ``A`` in tensor memory (the same products, accumulators, partition and part segments; two TMEM chunk
#: buffers, six slots of raw ``A`` tiles and six of ``B`` parts in shared memory).  No window of :data:`ROUTER_RULES`
#: gives ``sm_103a`` a head/tail partition (its windows on 152 SMs missed the accuracy rule there); a forced
#: head/tail spec runs this program.
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
#: The forward's largest explicit split count: an element of the forward takes at most two partials, so a forward
#: recomputed under activation checkpointing repeats bit for bit (``0 + a + b == 0 + b + a``).
ROUTER_FORWARD_MAX_SPLITS = 2
#: The deterministic package's production split count of the router weight gradient (its ``ROUTER_SPLITS["nn_t"]``):
#: the plan the ``"waves"`` partition is measured against.
ROUTER_DET_WGRAD_SPLITS = 6
#: The ``"waves"`` partition: about this many K blocks (32 elements each) of one tile per unit ...
ROUTER_WAVE_K_BLOCKS = 68
#: ... and only where the deterministic plan fills its last wave of CTA pairs below this fraction
#: (``w / ceil(w)``, ``w = ROUTER_DET_WGRAD_SPLITS * tiles / pairs``).
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
    m_pairs: int, n_tiles: int, k_iters: int, sm_count: int
) -> Optional[int]:
    """The ``"waves"`` unit count on a device of ``sm_count`` SMs (CTA pairs ``sm_count // 2``): whole waves of the
    CTA pairs, ``c * pairs`` units with ``c = max(2, round(tiles * k_iters / (ROUTER_WAVE_K_BLOCKS * pairs))) + 1``
    (spans of about ``ROUTER_WAVE_K_BLOCKS`` K blocks of a tile, one wave more, so a kernel holding SMs on another
    stream delays a fine last wave rather than a coarse one), or ``None`` (the deterministic program) where the
    deterministic plan's last wave is at least :data:`ROUTER_WAVE_MAX_DET_FILL` full."""
    tiles, pairs = m_pairs * n_tiles, max(1, int(sm_count) // 2)
    det_waves = ROUTER_DET_WGRAD_SPLITS * tiles / pairs
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
                if lo <= int(tokens) <= hi and (holds or router_simt_tile(s) is None)
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
      :func:`router_waves` gives a partition); a row, token count or SM count without one, or a head/tail partition
      the token count cannot hold (:func:`router_partition_fits`), runs the deterministic program.  A SIMT window is
      no partition (``None``): the launch runs the SIMT program there (:func:`router_simt`).

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
                f"router_fp32_gemm: splits must be in [1, {k_iters}] (the 32-element K blocks of K={K}), got {s}"
            )
        if layout == "kk" and s > ROUTER_FORWARD_MAX_SPLITS:
            raise ValueError(
                f"router_fp32_gemm(deterministic=False): the forward takes at most {ROUTER_FORWARD_MAX_SPLITS} "
                f"splits (more partials per element do not repeat bit for bit under recompute), got {s}; use "
                "deterministic=True for more"
            )
        spec = f"S{s}"
    else:
        spec = _router_rule_spec(layout, arch=arch, sm_count=sm_count, M=M, N=N, K=K)
        if spec == "waves":  # its own SM-count condition, on every SM count
            spec = router_waves(m_pairs, n_tiles, k_iters, sm_count)
        elif not router_rules(arch, sm_count)[1]:
            return None
        if spec is None or router_simt_tile(spec) is not None:
            return None
    if layout == "kk" and router_head_tail(spec) is not None:
        raise ValueError(
            f"router_fp32_gemm(deterministic=False): the head/tail partition {spec!r} is the weight gradient's; "
            "the forward takes at most two partials per element"
        )
    units, head_spans, head_steps = router_partition_of(
        spec, m_pairs, n_tiles, k_iters, sm_count
    )
    if layout == "kk" and not router_at_most_two_partials(m_pairs, n_tiles, units):
        raise ValueError(
            f"router_fp32_gemm(deterministic=False): the forward partition {spec!r} ({units} units over "
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


@dataclass(frozen=True)
class AtomicRouterPlan:
    """The host plan of one K2 launch of this package (the launched, possibly swapped GEMM ``[M, N] = [M, K] @
    [K, N]``): ``m_tiles / 2 x n_tiles`` tiles of ``256 x 128``, their ``k_iters`` K blocks cut into ``units``
    stream-K spans (``2 * units`` CTAs; the first ``head_spans`` spans share the first ``head_steps`` steps on the
    head/tail program, every span alike when ``head_spans`` is 0); with part tiles the launch zeroes ``out`` before
    the program.  With ``simt`` (a SIMT program, its tile, :func:`router_simt`): one CTA per
    ``tile_n x tile_m`` tile of ``out`` over ``k_iters`` K blocks of :data:`ROUTER_SIMT_BLOCK_K` elements, the
    operands' row strides ``lda`` / ``ldb``; ``units`` 0, no part tiles.  The weight gradient's (``nn_t``) launches
    ``m_tiles x n_tiles`` CTAs over the transposed ``out`` (``A`` the activations, ``B`` the output gradient), the
    input gradient's (``kn``) ``n_tiles x m_tiles`` CTAs, ``m_tiles`` row tiles of ``tile_m`` tokens, the last one
    partial (``A`` the output gradient, ``B`` the router weight)."""

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

    @property
    def m_pairs(self) -> int:
        return self.m_tiles // 2

    @property
    def tiles(self) -> int:
        return self.m_pairs * self.n_tiles

    @property
    def grid(self) -> tuple[int, int, int]:
        if self.simt is not None:
            if self.layout == "kn":
                return (self.n_tiles, self.m_tiles, 1)
            return (self.m_tiles, self.n_tiles, 1)
        return (2 * self.units, 1, 1)

    @cached_property
    def part_tiles(self) -> bool:
        """Whether some segment adds into ``out`` (the launch zeroes ``out`` before its program runs); computed once
        per plan."""
        if self.simt is not None:
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
        stores every element once)."""
        if self.simt is not None:
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
) -> tuple[Optional[AtomicRouterPlan], str]:
    """Plan this package's K2 GEMM ``out[M, N] = A[M, K] @ B[K, N]`` (fp32 views, the deterministic package's view
    contract and swap rule) of the ``kk`` / ``nn_t`` classes, and of the ``kn`` class in its SIMT windows.  Returns
    ``(plan, reason)``: ``plan`` is ``None`` when the launch runs the deterministic program, and ``reason`` says why --
    another layout class, a ``kn`` launch outside the SIMT windows, a view the reduce-add
    program cannot address (an ``out`` off a 16-byte boundary or with a row stride not a multiple of 4 elements runs
    the deterministic path, which serves or refuses it as that package does), a template this checkout does not
    register for ``arch``, or the planner (:func:`router_partition`).  An explicit ``splits`` (``1`` included) runs
    this package's program when the view and the registry allow it; a head/tail partition runs the head/tail program
    of the architecture on a device of ``sm_count`` SMs (:func:`router_head_tail_template`); a SIMT window
    (:func:`router_simt`) runs the SIMT program of its tile when the operands' rows are whole rows of a multiple of 4
    elements (else the deterministic program): the weight gradient's over the swapped launch, the input gradient's
    over ``A`` (the output gradient, K-major) and ``B`` (the router weight, N-major) as they are."""
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
            AtomicRouterPlan(
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
    if head_spans:
        template = router_head_tail_template(arch, sm_count)
        if template not in cake_jit.KERNELS.get(arch, {}):
            return None, f"{template} is not registered for {arch}"
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
        head_spans=head_spans,
        head_steps=head_steps,
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
    A: torch.Tensor, B: torch.Tensor, plan: AtomicRouterPlan
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
    plan: AtomicRouterPlan,
    *,
    A: Any,
    parts: Any,
    out: torch.Tensor,
    enable_pdl: int = 0,
) -> dict[str, Any]:
    """The keyword arguments of a K2 launch of this package (ordered by the program's argument plan): ``out`` is
    passed twice, as the store pointer and as the reduce-add map ``D``; ``head_spans`` / ``head_steps`` are read by the
    head/tail programs only; ``enable_pdl`` is the launch's PDL flag (0 or 1)."""
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
        head_spans=plan.head_spans,
        head_steps=plan.head_steps,
        enable_pdl=enable_pdl,
    )


def _simt_values(
    plan: AtomicRouterPlan,
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
    row strides ``lda`` / ``ldb`` / ``ldo``.  A SIMT program never launches as a programmatic dependent launch,
    whatever the call's PDL flag (``enable_pdl`` is accepted and ignored): its CTAs, launched early next to a running
    predecessor, would be placed only on the SMs that predecessor leaves free, several per SM, and finish later than
    spread over every SM once it ends (on 152-SM GB200 parts the backward pass, input and weight gradient, at hidden
    5120 and 6144 and 192 and 256 tokens was 12 to 16 % slower in CUDA graphs than with the weight gradient's program
    launched after its predecessor; the input gradient's program was not faster as a dependent launch)."""
    del enable_pdl
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
    """One prepared K1 launch of this package: the bound program and its workspace (the FP32 split slots, zeroed
    when prepared because the last split of a partial tile reads padding columns that no split writes and discards
    them; the tickets, zero when prepared and zero again after every complete launch).  ``launch()`` writes ``out``
    and returns it without allocating; it captures into a CUDA graph as one kernel.  A prepared launch must not run
    concurrently with itself (two streams would share its workspace); after a launch that did not complete (an
    aborted process step, a device error), :meth:`reset_counters` re-zeroes the tickets on the current stream."""

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
    captures into a CUDA graph (split, memset, kernel).  The SIMT program (``plan.simt``) reads the bound fp32
    operands in place: no stack, no fill, one kernel."""

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
    plan: AtomicGemmPlan,
    a_desc,
    b_desc,
    out3,
    out: torch.Tensor,
    arch: str,
    enable_pdl: Optional[bool],
) -> PreparedAtomicGemm:
    det = deterministic_backend()
    device = out.device
    if plan.splits > 1:
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
    ``weights_static`` is the deterministic call's (:func:`dense_projection_gemm`)."""
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
    plan: AtomicRouterPlan,
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    arch: str,
    enable_pdl: Optional[bool] = None,
) -> PreparedAtomicRouterGemm:
    det = deterministic_backend()
    device = out.device
    pdl = int(det.pdl_enabled(enable_pdl, device))
    a_view, split_src = _k2_views(A, B, plan)
    if plan.simt is not None:
        module_name = cake_jit.select_module(arch, plan.template)
        values = _simt_values(
            plan, A=a_view, B=split_src, out=out, ldo=out.stride(0), enable_pdl=pdl
        )
        return PreparedAtomicRouterGemm(
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
    values = _k2_values(plan, A=a_view, parts=parts, out=out, enable_pdl=pdl)
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
    enable_pdl: Optional[bool] = None,
    views: Optional[tuple] = None,
) -> Any:
    """This package's K2 GEMM ``out = A @ B`` (FP32 through the BF16x3 emulation, or a SIMT program) of the layout
    classes ``kk`` / ``kn`` / ``nn_t``: ``op`` is ``"plan"``, ``"prepare"`` or ``"launch"`` as for
    :func:`_atomic_k1`.  A plan or a prepared launch :func:`plan_atomic_router` keeps on the deterministic program is
    the deterministic package's object of ``op`` with the caller's ``splits`` (``None``: its production split count);
    a launch runs that program through this package's binding (:func:`_launch_cached_atomic_k2`; ``views``: an entry
    point's own launch views).
    ``enable_pdl`` (default: the device supports it) launches the split and the program as programmatic dependent
    launches."""
    if op == "launch":
        return _launch_cached_atomic_k2(
            A, B, out, splits=splits, enable_pdl=enable_pdl, views=views
        )
    _, arch, sm_count = _device_and_arch(A, B, out)
    plan, _ = plan_atomic_router(A, B, out, arch=arch, sm_count=sm_count, splits=splits)
    if plan is None:
        if op == "plan":
            return _det_entry("plan_router_fp32_gemm")(A, B, out, splits=splits)
        return _det_entry("prepare_router_fp32_gemm")(
            A, B, out, splits=splits, **_pdl_kw(enable_pdl)
        )
    if op == "plan":
        return plan
    return _prepare_atomic_k2(plan, A, B, out, arch, enable_pdl)


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
# stream) and zero again after every complete launch, so a launch needs no memset; the K1 split slots are zeroed when
# allocated or grown (the last split reads the never-written padding columns of a partial tile and discards them) and
# need no memset per launch either.  A K1 call of ``deterministic=False`` that runs the deterministic program -- the
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
# to the bound launch when it is cached (:func:`_k1_cached_launch`).  A K2 call of ``deterministic=False`` that runs the
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
# once) under its lock.  The router entry points' calls (``router_forward`` / ``router_dgrad`` / ``router_wgrad``)
# are cached once more by the entry point's own tensors (``_K2_ENTRY_CACHE``, :func:`_k2_entry`): each holds the
# cached launch of ``_K2_SPLITK_CACHE`` / ``_K2_ATOMIC_CACHE`` that the entry point's path takes for the key and the
# router-table memo and route it was routed under, so a call goes from its key to the bound launch.  After a launch
# that did not complete, :func:`clear_convenience_caches` drops the arenas (fresh tickets on the next call).

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
    # the tvm-ffi stream context of the (device, stream), made by the first K2 launch and entered by every K2 launch
    # under ``lock`` (:func:`_k2_stream_context`)
    stream_context: Any = field(default=None, repr=False, compare=False)

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
#: The split counts the partial sum serves (its kernel requires two to eight; the deterministic router planner issues
#: four for the forward and six for the weight gradient).
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
    (``partials_shape``, carved as ``workspace``) -- carved and converted once per arena growth (``version``) and the
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


def _release_atomic_entry(
    entry: Union[_CachedAtomic, _CachedSplitK2, _CachedDetK1, _CachedK2Entry],
) -> None:
    """``on_evict`` of the convenience caches: the arena of an evicted entry loses a reference and is released when
    no cached entry uses it any more."""
    arena = entry.arena
    if arena is None:
        return
    arena.refs -= 1
    if arena.refs <= 0 and _ATOMIC_ARENAS.get(arena.key) is arena:
        del _ATOMIC_ARENAS[arena.key]


#: An entry of the convenience caches (:data:`_K1_ATOMIC_CACHE`, :data:`_K1_DET_CACHE`, :data:`_K2_ATOMIC_CACHE`,
#: :data:`_K2_SPLITK_CACHE`, :data:`_K2_ENTRY_CACHE`).
_CacheEntry = TypeVar(
    "_CacheEntry", "_CachedAtomic", "_CachedDetK1", "_CachedSplitK2", "_CachedK2Entry"
)


def _cache_insert(
    cache: "OrderedDict[tuple, Any]", key: tuple, entry: _CacheEntry
) -> _CacheEntry:
    det = deterministic_backend()
    if entry.arena is not None:
        entry.arena.refs += 1
    return det._cache_put(
        cache, key, entry, det.CONVENIENCE_CACHE_SIZE, _release_atomic_entry
    )


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
) -> bool:
    """A projection entry point's call of ``deterministic=False`` when its key is cached and bound: the key the entry
    point's own path builds for the tensors ``sources`` (``(A, B, out)`` of that path) in ``cache`` --
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
    entry = det._cache_get(cache, _k1_key(index, stream, (*head, pdl, *tail), *sources))
    if entry is None or entry.k1 is None or entry.entry is None:
        return False
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
    plan: AtomicGemmPlan,
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
#: The leading and trailing key flags of :func:`_det_k1` for the projection forward and input gradient (the PDL flag
#: goes between them), as :func:`_k1_cached_launch` reads them.
_K1_FWD_HEAD = (("fwd",),)
_K1_DGRAD_HEAD = (("dgrad",),)
_K1_NOT_STATIC = (False,)


@dataclass
class _CachedDetK1:
    """A planned launch of the deterministic K1 program on this package's convenience path: the deterministic
    package's plan, its program's entry, the arena of the (device, stream) and the bound arguments (``k1``;
    ``None``: the call goes to the deterministic package's convenience path)."""

    plan: Any
    entry: Optional[Callable[..., Any]] = field(default=None, repr=False)
    arena: Optional[_AtomicArena] = field(default=None, repr=False)
    k1: Optional[_K1Binding] = field(default=None, repr=False)


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
) -> _CachedDetK1:
    """Plan the deterministic K1 program for the deterministic package's convenience call ``(A, B, out,
    transposed_out, **knobs)`` (under ``_ATOMIC_PLAN_LOCK``): that package's planner (its errors), its program bound
    in its argument order with the PDL flag ``pdl`` (0 or 1; :func:`_k1_binding`; the per-call views rebuilt from the
    call's tensors ``sources``, which hold the data pointers of ``A`` / ``B`` / ``out``), the stream-K slabs and slice
    counters in this package's arena.  A planner's view the call's tensors do not rebuild leaves the entry unbound
    (the deterministic package's convenience path), as does a call without knobs that the deterministic package runs
    on its swapped small-token forward (``_swap_ab_launch`` with the call's ``weights_static`` under PDL: its
    program, not the planner's)."""
    det = deterministic_backend()
    if not knobs:
        # the deterministic call's swapped small-token forward: its program, not the planner's
        static = bool(weights_static) and bool(pdl)
        if det._swap_ab_launch(A, B, out, transposed_out, static) is not None:
            return _CachedDetK1(plan=None)
    plan, a_desc, b_desc, O3, arch, module_name = det._k1_plan(
        A, B, out, transposed_out=transposed_out, knobs=dict(knobs)
    )
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
    return _CachedDetK1(plan=plan, entry=fn, arena=arena, k1=bound)


def _det_k1(
    sources: tuple,
    flags: tuple,
    det_args: Callable[[], tuple],
    det_call: Callable[[], Any],
    enable_pdl: Optional[bool],
    weights_static: bool = False,
) -> Any:
    """The deterministic package's K1 convenience call -- ``det_args()`` gives its ``(A, B, out, transposed_out,
    knobs)``, ``det_call()`` makes it with ``enable_pdl`` / ``weights_static`` -- on this package's convenience path:
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
                        k1=_bind_atomic_k1(
                            plan, module_name, arg_plan, A, B, out, arena, pdl
                        ),
                    )
                entry = _cache_insert(_K1_ATOMIC_CACHE, key, entry)
    bound, fn = entry.k1, entry.entry
    if bound is None or fn is None:
        if entry.plan is None:
            return det_call()
        return _launch_atomic_k1_unbound(entry, A, B, out, index, stream, pdl)
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
    plan: AtomicRouterPlan,
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
    values = _k2_values(plan, A=a_arg, parts=_K2_PARTS_ARG, out=out, enable_pdl=pdl)
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
    the deterministic program), the program's and fused split's entries, the BF16x3 stack in the arena of (device
    ``index``, raw ``stream``) and the argument lists bound with the PDL flag ``pdl`` (0 or 1); ``memo`` is the
    router-table memo the call read (``None`` with a caller's split count)."""
    det = deterministic_backend()
    device = out.device
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
        return _CachedAtomic(plan=None, arch=arch, memo=memo)
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
        return _launch_cached_splitk_k2(A, B, out, splits, index, geometry, pdl, views)
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
    plan, A_eff, B_eff = det.plan_router_fp32_det_gemm(A, B, out, splits=splits)
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
        return getattr(det, DETERMINISTIC_NAMES["router_fp32_gemm"])(
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


#: The cached calls of the router entry points with ``deterministic=False`` (``router_forward``, ``router_dgrad``,
#: ``router_wgrad``), keyed by the entry point's own tensors (:func:`_k1_key`), the operation, its split count and the
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
    if launch is None:
        launch = det._cache_get(_K2_SPLITK_CACHE, key)
        if launch is None:
            launch = _cache_insert(
                _K2_SPLITK_CACHE,
                key,
                _plan_splitk_k2(A, B, out, splits, index, stream, pdl),
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
    """A router entry point's call of ``deterministic=False`` (``op`` and its tensors ``P``, ``Q`` as for
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
    (:func:`_router_memo`) and this package's workspace arenas."""
    global _ROUTER_MEMO
    with _ATOMIC_PLAN_LOCK:
        _K1_ATOMIC_CACHE.clear()
        _K1_DET_CACHE.clear()
        _K2_ATOMIC_CACHE.clear()
        _K2_SPLITK_CACHE.clear()
        _K2_ENTRY_CACHE.clear()
        _ATOMIC_ARENAS.clear()
        _K1_STREAM_CONTEXTS.clear()
        _ROUTER_MEMO = _RouterMemo(-1, None, None, False, True)


def clear_convenience_caches() -> None:
    """Drop every cached convenience-path launch, what the convenience path keeps from the router tables
    (:func:`_router_memo`) and every workspace arena of both packages (this package's launches and the deterministic
    package's)."""
    _drop_convenience_entries()
    deterministic_backend().clear_convenience_caches()


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def _weights_kw(weights_static: bool) -> dict[str, Any]:
    """``weights_static`` as the keyword the deterministic entry point receives: passed only when set."""
    return {"weights_static": True} if weights_static else {}


def _pdl_kw(enable_pdl: Optional[bool]) -> dict[str, Any]:
    """``enable_pdl`` as the keyword a deterministic entry point receives: passed only when the caller set it, so the
    default is the deterministic package's (the device's support) and a call without it is the same call."""
    return {} if enable_pdl is None else {"enable_pdl": enable_pdl}


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
    classes this package reduces (``kk``, ``nn_t``) and the deterministic symbol for ``kn`` (no stream-K program; its
    SIMT programs are named per tile, :func:`router_simt_template`)."""
    det = _resolve_deterministic(deterministic, "router_symbol")
    det_symbol = _det_entry("router_symbol")(a_mn, b_mn, out_t)
    layout = _k2_class(a_mn, b_mn)
    if det or layout == "k2_kn" or _route(layout) == "det":
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
    program (:class:`AtomicGemmPlan`; the knobs ``splits`` / ``block_n`` / ``chains`` are this package's, the others
    apply when the row runs the deterministic program)."""
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
    workspace is owned by the prepared object; ``splits`` / ``block_n`` / ``chains`` as for
    :func:`plan_dense_projection_gemm`).
    ``enable_pdl`` (default: the device supports it) makes either launch a programmatic dependent launch."""
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
        enable_pdl=det_kwargs.get("enable_pdl"),
    )


def dense_projection_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    *,
    transposed_out: bool = False,
    deterministic: bool = True,
    enable_pdl: Optional[bool] = None,
    weights_static: bool = False,
) -> torch.Tensor:
    """``out[l, m, n] = sum_k A[l, m, k] B[l, k, n]`` for strided views (the cached convenience path; see the
    deterministic package for the view contract).  ``deterministic=False`` runs a weight-gradient layout (both
    operands MN-major) on this package's split-K program where the planner takes the row; every other layout and
    row runs the deterministic program.  ``enable_pdl`` (default: the device supports it) launches either program
    as a programmatic dependent launch; each waits for the preceding kernel before its first global access, and
    ``weights_static=True`` lets the deterministic swapped small-token forward start its loads of ``B`` before the
    wait (see the deterministic package's ``dense_projection_det_gemm``)."""
    if _resolve_deterministic(deterministic, "dense_projection_gemm"):
        return _det_entry("dense_projection_gemm")(
            A,
            B,
            out,
            transposed_out=transposed_out,
            enable_pdl=enable_pdl,
            **_weights_kw(weights_static),
        )

    def det_call() -> Any:
        return _det_entry("dense_projection_gemm")(
            A,
            B,
            out,
            transposed_out=transposed_out,
            enable_pdl=enable_pdl,
            **_weights_kw(weights_static),
        )

    def det_args() -> tuple:
        return A, B, out, transposed_out, ()

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


def projection_forward(
    X: torch.Tensor,
    W: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    deterministic: bool = True,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    """``out[T, N] = X[T, K] @ W[N, K].T`` (bf16 out by default).  The deterministic program in both modes (a
    recomputed forward must reproduce its result; where the deterministic plan splits K across CTAs, it adds the
    partials in a fixed order); with ``deterministic=False`` through this package's binding of it (:func:`_det_k1`: the
    same plan, program and bits).
    ``enable_pdl`` as for :func:`dense_projection_gemm`."""
    if _resolve_deterministic(deterministic, "projection_forward"):
        return _det_entry("projection_forward")(X, W, out, **_pdl_kw(enable_pdl))
    if out is None:
        out = torch.empty(X.shape[0], W.shape[0], device=X.device, dtype=torch.bfloat16)
    o = out
    if _k1_cached_launch(
        _K1_DET_CACHE, _K1_FWD_HEAD, _K1_NOT_STATIC, (X, W, o), enable_pdl
    ):
        return o
    return _det_k1(
        (X, W, o),
        ("fwd",),
        lambda: (X, W.t(), o, False, ()),
        lambda: _det_entry("projection_forward")(X, W, o, **_pdl_kw(enable_pdl)),
        enable_pdl,
    )


def projection_dgrad(
    G: torch.Tensor,
    W: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    out_dtype=None,
    deterministic: bool = True,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    """``out[T, K] = G[T, N] @ W[N, K]``.  The deterministic program in both modes (whole output tiles); with
    ``deterministic=False`` through this package's binding of it (:func:`_det_k1`: the same plan, program and bits).
    ``enable_pdl`` as for :func:`dense_projection_gemm`."""
    if _resolve_deterministic(deterministic, "projection_dgrad"):
        return _det_entry("projection_dgrad")(
            G, W, out, out_dtype=out_dtype, **_pdl_kw(enable_pdl)
        )
    if out is None:
        out = torch.empty(
            G.shape[0],
            W.shape[1],
            device=G.device,
            dtype=out_dtype or torch.bfloat16,
        )
    o = out
    if _k1_cached_launch(
        _K1_DET_CACHE, _K1_DGRAD_HEAD, _K1_NOT_STATIC, (G, W, o), enable_pdl
    ):
        return o
    return _det_k1(
        (G, W, o),
        ("dgrad",),
        lambda: (G, W, o, False, ()),
        lambda: _det_entry("projection_dgrad")(
            G, W, o, out_dtype=out_dtype, **_pdl_kw(enable_pdl)
        ),
        enable_pdl,
    )


def prepare_projection_wgrad(
    G: torch.Tensor,
    X: torch.Tensor,
    out: torch.Tensor,
    *,
    sk: bool | str = "auto",
    cta_rows: Optional[int] = None,
    deterministic: bool = True,
    enable_pdl: Optional[bool] = None,
) -> Any:
    """Prepared ``out[N, K] = G[T, N].T @ X[T, K]``.  ``deterministic=False`` prepares this package's split-K
    program as the transposed-store GEMM ``X.T @ G`` (``M = K``, ``N = N``) where the planner takes the row; a row it
    does not take prepares the deterministic package's ``prepare_projection_det_wgrad`` (its orientation, ``sk``,
    ``cta_rows``).  ``enable_pdl`` as for :func:`prepare_dense_projection_gemm`."""
    if _resolve_deterministic(deterministic, "prepare_projection_wgrad"):
        return _det_entry("prepare_projection_wgrad")(
            G, X, out, sk=sk, cta_rows=cta_rows, **_pdl_kw(enable_pdl)
        )
    return _atomic_k1(
        "prepare",
        X.t(),
        G,
        out,
        transposed_out=True,
        own={},
        det_call=lambda: _det_entry("prepare_projection_wgrad")(
            G, X, out, sk=sk, cta_rows=cta_rows, **_pdl_kw(enable_pdl)
        ),
        enable_pdl=enable_pdl,
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
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    """``out[N, K] = G[T, N].T @ X[T, K]`` (ragged reduction length ``T``).  ``deterministic=False`` runs this
    package's split-K program where the planner takes the row (the split partials summed in the order they finish,
    the output rounded once), else the deterministic package's ``projection_det_wgrad`` program (its orientation,
    ``sk`` and ``cta_rows``) through this package's binding of it (:func:`_launch_cached_atomic_k1`); ``enable_pdl``
    as for :func:`dense_projection_gemm`."""
    if _resolve_deterministic(deterministic, "projection_wgrad"):
        return _det_entry("projection_wgrad")(
            G,
            X,
            out,
            out_dtype=out_dtype,
            sk=sk,
            cta_rows=cta_rows,
            **_pdl_kw(enable_pdl),
        )
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
    return _atomic_k1(
        "launch",
        A,
        G,
        out,
        transposed_out=True,
        own={},
        det_call=lambda: _det_entry("projection_wgrad")(
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
    """The deterministic package's convenience call of ``projection_det_wgrad(G, X, out, sk=sk,
    cta_rows=cta_rows)``: ``(A, B, out, transposed_out, knobs)`` in its orientation (``wgrad_views``)."""
    det = deterministic_backend()
    A, B, transposed = det.wgrad_views(
        G, X, arch=det.arch_for(out.device), out_f32=out.dtype == torch.float32
    )
    return A, B, out, transposed, (("cta_rows", cta_rows), ("sk", sk))


def plan_router_fp32_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    *,
    splits: Optional[int] = None,
    deterministic: bool = True,
) -> Any:
    """Classify the fp32 views and plan the K2 launch (the deterministic package's ``plan_router_fp32_det_gemm``);
    ``deterministic=False`` plans this package's stream-K program for the ``kk`` / ``nn_t`` classes and the SIMT
    programs in their windows (:class:`AtomicRouterPlan`; ``splits`` sets ``splits`` units per tile, see
    :func:`router_units`)."""
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
    enable_pdl: Optional[bool] = None,
) -> Any:
    """Validate one fp32 binding, plan it and prepare its allocation-free launch (the deterministic package's
    ``prepare_router_fp32_det_gemm``); ``deterministic=False`` prepares this package's stream-K program for the
    ``kk`` / ``nn_t`` classes (its launch splits the second operand, zeroes ``out`` on the stream when a segment adds
    into it, and runs the program: no partial buffers), and a SIMT program in its windows (the ``nn_t`` and ``kn``
    classes: one kernel over the fp32 operands).  ``enable_pdl`` (default: the device supports it) launches the split
    and the stream-K program as programmatic dependent launches."""
    if _resolve_deterministic(deterministic, "prepare_router_fp32_gemm") or (
        _route(_k2_layout(A, B)) == "det"
    ):
        return _det_entry("prepare_router_fp32_gemm")(
            A, B, out, splits=splits, **_pdl_kw(enable_pdl)
        )
    return _atomic_k2("prepare", A, B, out, splits=splits, enable_pdl=enable_pdl)


def _router_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    splits: Optional[int],
    det: bool,
    enable_pdl: Optional[bool] = None,
    views: Optional[tuple] = None,
) -> torch.Tensor:
    """K2 on the convenience path: ``det`` calls the deterministic package's; with ``deterministic=False`` a layout
    class routed ``"det"`` runs the deterministic program through this package's binding (:func:`_splitk_k2`), the
    others this package's K2 path (``views``: an entry point's own launch views, :func:`_launch_views`).
    ``enable_pdl`` reaches every route."""
    if det:
        return _det_entry("router_fp32_gemm")(
            A, B, out, splits=splits, **_pdl_kw(enable_pdl)
        )
    if _route(_k2_layout(A, B)) == "det":
        return _splitk_k2(A, B, out, splits, views, enable_pdl)
    return _atomic_k2(
        "launch", A, B, out, splits=splits, enable_pdl=enable_pdl, views=views
    )


def router_fp32_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    *,
    splits: Optional[int] = None,
    deterministic: bool = True,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    """``out[m, n] = sum_k A[m, k] B[k, n]`` for fp32 views through the BF16x3 emulation (the cached convenience
    path).  ``deterministic=False`` runs the ``kk`` / ``nn_t`` rows the planner takes (:func:`router_units`) on this
    package's stream-K program, whose part tiles add into ``out`` in the order they finish, instead of the
    fixed-order split-K sum, and the ``nn_t`` / ``kn`` token counts of a SIMT window on its exact-FP32 program
    (:func:`router_simt`); every other row runs the deterministic program.  ``enable_pdl`` as for
    :func:`prepare_router_fp32_gemm`."""
    det = _resolve_deterministic(deterministic, "router_fp32_gemm")
    return _router_gemm(A, B, out, splits, det, enable_pdl)


def router_forward(
    X: torch.Tensor,
    W: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    splits: Optional[int] = None,
    deterministic: bool = True,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    """``out[T, N] = X[T, K] @ W[N, K].T`` in FP32 through the emulation; ``splits`` defaults to the deterministic
    package's production count (4).  With ``deterministic=False`` a row runs this package's program only with an
    explicit ``splits`` (at most two partials per element: the logits repeat bit for bit) or a verdict of
    :data:`ROUTER_RULES` (none ships for the forward), else the deterministic program (its plan for
    ``router_fp32_gemm(X, W.T, out)``, through this package's binding, :func:`_launch_cached_splitk_k2`): without a
    split count, while no router table gives the forward a verdict, without the per-call check of the row."""
    if _resolve_deterministic(deterministic, "router_forward"):
        return _det_entry("router_forward")(
            X, W, out, **_splits_kw(splits), **_pdl_kw(enable_pdl)
        )
    if out is None:
        out = torch.empty(X.shape[0], W.shape[0], device=X.device, dtype=torch.float32)
    if _k2_entry("fwd", X, W, out, splits, enable_pdl) is not None:
        return out
    views = (X.unsqueeze(0), W, _FORWARD_VIEWS)
    if splits is None and not _router_memo().forward_verdicts:
        # no router table gives the forward a verdict: the deterministic program (the plan the planner would keep),
        # without the per-call check of the row
        return _splitk_k2(X, W.t(), out, None, views, enable_pdl)
    return _router_gemm(X, W.t(), out, splits, False, enable_pdl, views)


def router_dgrad(
    G: torch.Tensor,
    W: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    deterministic: bool = True,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    """``out[T, K] = G[T, N] @ W[N, K]`` in FP32: the deterministic program (one split: nothing to reduce) through the
    emulation; with ``deterministic=False`` in the token windows of the row's input-gradient verdict in
    :data:`ROUTER_RULES` (or, on a part with a table of its own, :data:`ROUTER_RULES_BY_SM`) the exact-FP32 SIMT
    program (:func:`router_simt`: compensated sums of the products of ``G`` and ``W`` read in place, every element of
    ``out`` stored once), and at every other row and token count the deterministic program through this package's
    binding of it (:func:`_launch_cached_splitk_k2`: the same plan, program and bits).  ``enable_pdl`` as for
    :func:`prepare_router_fp32_gemm`."""
    if _resolve_deterministic(deterministic, "router_dgrad"):
        return _det_entry("router_dgrad")(G, W, out, **_pdl_kw(enable_pdl))
    if out is None:
        out = torch.empty(G.shape[0], W.shape[1], device=G.device, dtype=torch.float32)
    if _k2_entry("dgrad", G, W, out, None, enable_pdl) is not None:
        return out
    return _router_gemm(
        G, W, out, None, False, enable_pdl, (G.unsqueeze(0), W, _DGRAD_VIEWS)
    )


def router_wgrad(
    G: torch.Tensor,
    X: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    splits: Optional[int] = None,
    deterministic: bool = True,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    """``out[N, K] = G[T, N].T @ X[T, K]`` in FP32: the swapped GEMM ``X.T @ G`` with a transposed store;
    ``splits`` defaults to the deterministic package's production count (6), or with ``deterministic=False`` to the
    partition of the row's verdict in :data:`ROUTER_RULES` or, on a part with a table of its own, in
    :data:`ROUTER_RULES_BY_SM` (:func:`router_partition`; a row or token count without one runs the deterministic
    program, through this package's binding of it without planning the row here, :func:`_launch_cached_splitk_k2`)."""
    if _resolve_deterministic(deterministic, "router_wgrad"):
        return _det_entry("router_wgrad")(
            G, X, out, **_splits_kw(splits), **_pdl_kw(enable_pdl)
        )
    if out is None:
        out = torch.empty(G.shape[1], X.shape[1], device=G.device, dtype=torch.float32)
    if _k2_entry("wgrad", G, X, out, splits, enable_pdl) is not None:
        return out
    # a row without a verdict (kept per device and geometry, :func:`_router_memo`) runs the deterministic program
    views = (X.unsqueeze(0), G, _WGRAD_VIEWS)
    return _router_gemm(G.t(), X, out, splits, False, enable_pdl, views)


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
