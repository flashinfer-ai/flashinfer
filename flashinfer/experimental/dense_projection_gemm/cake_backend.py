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
router-gate GEMM (K2) on SM100 / SM103 / SM107 (flashinfer-ai/flashinfer#8069).

The deterministic implementation is the package ``dense_projection_det_gemm`` (flashinfer-ai/flashinfer#5677 and
#7043; named so by #8046): its stream-K and split-K reductions add the partials in a fixed order, so every result is
bitwise reproducible run to run.  This package keeps that package's operations, view contract, dtypes, accuracy
budgets and host paths (prepared and convenience) and adds a second numerical mode in which those reductions add the
partials with unordered FP32 reductions (``red.global.add.f32``).  Only run-to-run bitwise reproducibility is
relaxed: the products, the FP32 accumulation, the rounding points (a BF16 output is rounded once, from the complete
FP32 sum), the masks, the layouts and the dynamic shapes are the deterministic package's.

Numerical mode
--------------
Every entry point takes the keyword-only argument ``deterministic: bool = True``:

* ``deterministic=True`` (the default) calls the deterministic package's canonical object with the same arguments
  (:data:`DETERMINISTIC_NAMES`: ``projection_forward`` -> ``projection_det_forward``, ...): the same planner, the
  same programs, the same bits and the same errors.  The names this import path had before #8046 keep their behavior.
* ``deterministic=False`` runs the non-deterministic schedule of an operation :data:`ROUTES` marks ``"atomic"``:

  - the K1 weight gradient (both operands MN-major, ``K = T`` ragged): split-K over the token count, the parts
    reduce-add FP32 into a zero-invariant workspace and the last arriving part rounds once, stores and re-zeroes
    (:func:`k1_split_parts` chooses the split; a row it keeps whole runs the deterministic program);
  - the K2 forward and weight gradient: split-K partials reduce-added into ``out`` (:func:`router_splits`; the forward
    takes at most two splits, ``0 + a + b == 0 + b + a``, so a forward recomputed under activation checkpointing
    gives the same logits and the same top-k).

  Every other operation -- the K1 forward and input gradient, the K2 input gradient, the BF16x3 split -- runs the
  deterministic programs in both modes: it has no cross-CTA reduction to relax.
* With ``torch.use_deterministic_algorithms(True)`` an explicit ``deterministic=False`` raises ``RuntimeError``; with
  ``warn_only=True`` it warns and runs the deterministic path (:func:`_resolve_deterministic`).  The default does not
  consult the PyTorch setting: it is deterministic already.

The two packages
----------------
This package registers only the programs of the non-deterministic schedule (``cake_jit``: programs
``cake_dense_projection_gemm_<hex>``, templates ``dense_proj_gemm_atomic_*`` / ``router_fp32_gemm_atomic_*``); every
deterministic program, and every planner rule, is the deterministic package's (resolved as this package's sibling on
first use, never through an absolute ``flashinfer`` import; a vendored copy binds it with
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
from types import ModuleType
from typing import Any, Callable, Optional

import torch

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

    ``deterministic=True`` does.  ``deterministic=False`` runs this package's non-deterministic schedule unless
    ``torch.use_deterministic_algorithms(True)`` is in effect: then the call raises ``RuntimeError``, or, with
    ``warn_only=True``, warns (``UserWarning``) and runs the deterministic path.  A call that asked for determinism
    never runs a non-deterministic program.
    """
    if deterministic:
        return True
    if not torch.are_deterministic_algorithms_enabled():
        return False
    message = (
        f"{__name__}.{api}(deterministic=False) adds split-K partials with unordered FP32 reductions, but "
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
# Routing of the non-deterministic mode
# ---------------------------------------------------------------------------

#: The program family each operation runs with ``deterministic=False``: ``"atomic"`` -- this package's planner and
#: programs (unordered FP32 reduce-add of split-K partials; the planner may still keep a row on the deterministic
#: program, see :func:`k1_split_parts` / :func:`router_splits`) -- or ``"det"`` -- the deterministic package's
#: programs in both modes.  K1 operations are keyed by the operand layouts of the launched GEMM (``kk`` forward,
#: ``kn`` input gradient, ``nn`` weight gradient; ``nk`` has no training use), K2 by the layout class of
#: ``router_layout_class``.
ROUTES: dict[str, str] = {
    "k1_kk": "det",  # forward: no cross-CTA reduction (whole output tiles); feeds the recompute of checkpointing
    "k1_kn": "det",  # input gradient: whole output tiles
    "k1_nk": "det",
    "k1_nn": "atomic",  # weight gradient: split-K over the ragged token count
    "k2_kk": "atomic",  # router forward: at most two splits (reproducible)
    "k2_kn": "det",  # router input gradient: one split
    "k2_nn_t": "atomic",  # router weight gradient: free split count
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


# ---------------------------------------------------------------------------
# Planner hooks of the non-deterministic programs
# ---------------------------------------------------------------------------

#: Measured split counts of the non-deterministic K2 programs, ``(arch, layout class) -> S`` (the per-row rules of
#: the router).  An entry of ``1`` keeps the layout on the deterministic program.
ROUTER_SPLIT_RULES: dict[tuple[str, str], int] = {}
#: The router forward's split bound in this mode: with at most two partials the unordered sum is reproducible
#: (``0 + a + b == 0 + b + a``, also with subnormals flushed), so a recomputed forward gives the same logits.
ROUTER_FORWARD_MAX_SPLITS = 2
#: Measured split counts of the non-deterministic K1 weight-gradient programs, keyed like the weight-gradient rows
#: of the deterministic ``ROW_RULES`` without the ragged token count: ``(arch, out_f32, transposed_out, L, M, N)``
#: -> ``p`` (``1``: whole tiles, the deterministic program).
K1_SPLIT_RULES: dict[tuple[str, bool, bool, int, int, int], int] = {}


def router_splits(
    layout: str,
    *,
    arch: str,
    M: int,
    N: int,
    K: int,
    sm_count: int,
    splits: Optional[int] = None,
) -> int:
    """Split count ``S`` of a non-deterministic K2 launch of the layout class ``layout`` (``kk``, ``kn``, ``nn_t``)
    for ``[M, N] = [M, K] @ [K, N]`` on a device of ``sm_count`` SMs.  ``S == 1`` runs the deterministic program (one
    split stores its tile; nothing is reduced); ``S > 1`` runs this package's program.

    * ``kn`` (input gradient): always ``1``.
    * a caller's ``splits``: taken as is, except that the forward (``kk``) refuses more than
      :data:`ROUTER_FORWARD_MAX_SPLITS` (its result must be reproducible, see the module docstring).
    * otherwise the measured rule :data:`ROUTER_SPLIT_RULES`, else the split model of the architecture.
    """
    if layout == "kn":
        return 1
    if splits is not None:
        s = int(splits)
        if s < 1:
            raise ValueError("router_fp32_gemm: splits must be >= 1")
        if layout == "kk" and s > ROUTER_FORWARD_MAX_SPLITS:
            raise ValueError(
                f"router_fp32_gemm(deterministic=False): the forward takes at most {ROUTER_FORWARD_MAX_SPLITS} "
                f"splits (an unordered sum of more partials is not reproducible under recompute), got {s}; use "
                "deterministic=True for more"
            )
        return s
    rule = ROUTER_SPLIT_RULES.get((arch, layout))
    if rule is not None:
        return rule
    return _router_split_model(layout, arch=arch, M=M, N=N, K=K, sm_count=sm_count)


def _router_split_model(
    layout: str, *, arch: str, M: int, N: int, K: int, sm_count: int
) -> int:
    # KERNEL-LANE STUB (K2): the wave model S = argmin R(S) * (k_iters(S) + o_arch) + lambda_arch * S * out_bytes,
    # calibrated per architecture (U3 3.4), the forward bounded by ROUTER_FORWARD_MAX_SPLITS.
    raise NotImplementedError(
        f"router_fp32_gemm(deterministic=False): no split rule for {layout} on {arch} yet (see {TRACKING_ISSUE})"
    )


def k1_split_parts(
    *,
    arch: str,
    L: int,
    M: int,
    N: int,
    K: int,
    out_f32: bool,
    transposed_out: bool,
    sm_count: int,
) -> int:
    """Split count ``p`` of a non-deterministic K1 weight gradient ``out[L, M, N]`` reduced over ``K`` (the ragged
    token count) on a device of ``sm_count`` SMs: ``1`` keeps whole tiles (the deterministic program serves the row,
    the same bits); ``p > 1`` splits K into ``p`` parts of this package's program.  The measured rule
    :data:`K1_SPLIT_RULES` first, else the split model of the architecture."""
    rule = K1_SPLIT_RULES.get((arch, bool(out_f32), bool(transposed_out), L, M, N))
    if rule is not None:
        return rule
    return _k1_split_model(
        arch=arch,
        L=L,
        M=M,
        N=N,
        K=K,
        out_f32=out_f32,
        transposed_out=transposed_out,
        sm_count=sm_count,
    )


def _k1_split_model(
    *,
    arch: str,
    L: int,
    M: int,
    N: int,
    K: int,
    out_f32: bool,
    transposed_out: bool,
    sm_count: int,
) -> int:
    # KERNEL-LANE STUB (K1): fill the CTA pairs with L x pair tiles x p work items (each split >= SK_MIN_ITERS K
    # steps), p = 1 when the whole-tile grid already fills them.
    raise NotImplementedError(
        f"projection_wgrad(deterministic=False): no split rule for [{L}, {M}, {N}] on {arch} yet "
        f"(see {TRACKING_ISSUE})"
    )


# ---------------------------------------------------------------------------
# The non-deterministic launches (planner, binding, prepared and convenience paths)
# ---------------------------------------------------------------------------


def _atomic_k1(
    op: str,
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    *,
    transposed_out: bool,
    knobs: dict[str, Any],
) -> Any:
    """The non-deterministic K1 weight gradient ``out = A @ B`` (both MN-major): ``op`` is ``"plan"`` (returns the
    plan), ``"prepare"`` (the prepared launch) or ``"launch"`` (the convenience path, returns ``out``).  A split of
    ``p == 1`` (:func:`k1_split_parts`) runs the deterministic program."""
    # KERNEL-LANE STUB (K1): plan the split-K work items (L x pair tiles x p), bind the program of
    # dense_proj_gemm_atomic_* with the zero-invariant workspace (arena kinds "k1_acc" / "k1_tickets") and launch.
    raise NotImplementedError(
        "dense_projection_gemm(deterministic=False): the split-K weight-gradient programs are not registered in "
        f"this checkout (see {TRACKING_ISSUE})"
    )


def _atomic_k2(
    op: str,
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    *,
    splits: Optional[int],
) -> Any:
    """The non-deterministic K2 GEMM ``out = A @ B`` (FP32 through the BF16x3 emulation) of the layout classes
    ``kk`` / ``nn_t``: ``op`` is ``"plan"``, ``"prepare"`` or ``"launch"`` as for :func:`_atomic_k1`.  A split count
    of ``1`` (:func:`router_splits`) runs the deterministic program."""
    # KERNEL-LANE STUB (K2): plan S, zero ``out`` on the stream when S > 1, split B into the BF16x3 stack (the
    # deterministic split stage), launch router_fp32_gemm_atomic_<class> with accumulate = (S > 1).
    raise NotImplementedError(
        "router_fp32_gemm(deterministic=False): the split-K reduce-add programs are not registered in this "
        f"checkout (see {TRACKING_ISSUE})"
    )


@dataclass
class _AtomicLaunch:
    module: str
    entry: Callable[..., Any] = field(repr=False)
    arguments: tuple = field(repr=False)
    grid: tuple[int, int, int]

    def __call__(self) -> None:
        self.entry(*self.arguments)


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


def _atomic_bind(
    module_name: str, values: dict[str, Any], grid: tuple[int, int, int], arch: str
) -> tuple[Callable[..., Any], tuple]:
    """``(entry, arguments)`` of this package's program ``module_name``: the loaded FFI entry and ``values`` ordered
    by the program's argument plan.  The grid must be a multiple of the cluster shape baked into the module."""
    record = MODULES[module_name]
    cluster = record.get("launch", {}).get("cluster")
    if cluster and any(g % c for g, c in zip(grid, cluster, strict=True)):
        raise ValueError(
            f"{module_name}: grid {grid} is not a multiple of the cluster shape {tuple(cluster)} baked into the module"
        )
    arguments = _order_atomic_arguments(module_name, arg_plan_of(record), values, grid)
    module = load_cake_dense_projection_gemm_module(module_name, arch)
    return getattr(module, record["ffi_entry"]), tuple(arguments)


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


# --- convenience-path state of the non-deterministic programs --------------------------------------------------
#
# The deterministic package's design (one least-recently-used cache per kind, keyed by device, stream and view
# geometry; one growable workspace arena per (device, raw stream) shared by every cached entry), with caches and
# arenas of their own: a buffer of this package is never a buffer of the deterministic package, and the names below
# are not names this path had before #8046 (those read through to the deterministic package's state).  The cache
# bound is the deterministic package's ``CONVENIENCE_CACHE_SIZE``, read on every insertion, so one setting bounds
# both packages.  The kinds marked zeroed hold the zero invariant of the split-K reduction: zero when allocated or
# grown (``torch.zeros`` on the arena's stream) and zero again after every launch (the last arriving part of a tile
# re-zeroes its accumulator and resets its ticket), so a launch needs no memset.

#: workspace kind -> (dtype, zeroed when allocated)
_ATOMIC_ARENA_KINDS: dict[str, tuple[torch.dtype, bool]] = {
    "k1_acc": (
        torch.float32,
        True,
    ),  # K1 split-K accumulators: one FP32 slab per split output tile
    "k1_tickets": (
        torch.uint32,
        True,
    ),  # K1 per-tile arrival tickets (acq_rel), reset by the last arriver
    "k2_parts": (
        torch.bfloat16,
        False,
    ),  # K2 BF16x3 stack of the split operand ``[3, *split_src.shape]``
    "k2_resid": (torch.float32, False),  # K2 fp32 scratch of the torch-chain split
}
_ATOMIC_ARENA_MIN_ELEMS = 16


@dataclass
class _AtomicArena:
    """The workspace of every cached non-deterministic launch of one (device index, raw stream): one buffer per
    kind of ``_ATOMIC_ARENA_KINDS``, grown to the largest requirement seen and never shrunk (the deterministic package's
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


def clear_convenience_caches() -> None:
    """Drop every cached convenience-path launch and release every workspace arena of both packages (this
    package's non-deterministic launches and the deterministic package's)."""
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
    package's symbol, or with ``deterministic=False`` the template of this package's split-K program."""
    if _resolve_deterministic(deterministic, "instance_symbol"):
        return _det_entry("instance_symbol")(key)
    # KERNEL-LANE STUB (K1): the dense_proj_gemm_atomic_* template of a split-K instance key.
    raise NotImplementedError(
        f"instance_symbol(deterministic=False): the split-K templates are not defined in this checkout (see "
        f"{TRACKING_ISSUE})"
    )


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
    program."""
    if (
        _resolve_deterministic(deterministic, "plan_dense_projection_gemm")
        or _route(_k1_layout(A, B)) == "det"
    ):
        return _det_entry("plan_dense_projection_gemm")(A, B, out, **kwargs)
    transposed_out = bool(kwargs.pop("transposed_out", False))
    return _atomic_k1("plan", A, B, out, transposed_out=transposed_out, knobs=kwargs)


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
    zero-invariant workspace is owned by the prepared object)."""
    if (
        _resolve_deterministic(deterministic, "prepare_dense_projection_gemm")
        or _route(_k1_layout(A, B)) == "det"
    ):
        return _det_entry("prepare_dense_projection_gemm")(A, B, out, **kwargs)
    transposed_out = bool(kwargs.pop("transposed_out", False))
    return _atomic_k1("prepare", A, B, out, transposed_out=transposed_out, knobs=kwargs)


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
    operands MN-major) on this package's split-K program; every other layout runs the deterministic program."""
    if _resolve_deterministic(deterministic, "dense_projection_gemm") or (
        _route(_k1_layout(A, B)) == "det"
    ):
        return _det_entry("dense_projection_gemm")(
            A, B, out, transposed_out=transposed_out
        )
    return _atomic_k1("launch", A, B, out, transposed_out=transposed_out, knobs={})


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
    """Prepared ``out[N, K] = G[T, N].T @ X[T, K]`` (orientation by the deterministic package's ``wgrad_views``);
    ``deterministic=False`` prepares this package's split-K program (``sk`` / ``cta_rows`` are knobs of the
    deterministic planner, applied when the split rule keeps the row whole)."""
    if _resolve_deterministic(deterministic, "prepare_projection_wgrad"):
        return _det_entry("prepare_projection_wgrad")(
            G, X, out, sk=sk, cta_rows=cta_rows
        )
    A, B, transposed = _wgrad_views(G, X, out)
    return _atomic_k1(
        "prepare",
        A,
        B,
        out,
        transposed_out=transposed,
        knobs=dict(sk=sk, cta_rows=cta_rows),
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
    """``out[N, K] = G[T, N].T @ X[T, K]`` (ragged reduction length ``T``).  ``deterministic=False`` reduces the
    split-K parts of this package's program with unordered FP32 adds (BF16 output rounded once)."""
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
    A, B, transposed = _wgrad_views(G, X, out)
    return _atomic_k1(
        "launch",
        A,
        B,
        out,
        transposed_out=transposed,
        knobs=dict(sk=sk, cta_rows=cta_rows),
    )


def _wgrad_views(
    G: torch.Tensor, X: torch.Tensor, out: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """The deterministic package's weight-gradient orientation for ``out``'s device and dtype."""
    det = deterministic_backend()
    return det.wgrad_views(
        G, X, arch=det.arch_for(out.device), out_f32=out.dtype == torch.float32
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
    ``deterministic=False`` plans this package's reduce-add program for the ``kk`` / ``nn_t`` classes
    (``splits`` overrides :func:`router_splits`)."""
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
    ``prepare_router_fp32_det_gemm``); ``deterministic=False`` prepares this package's reduce-add program for the
    ``kk`` / ``nn_t`` classes (its launch zeroes ``out`` on the stream, then splits and launches: no partials)."""
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
    path).  ``deterministic=False`` reduces the split-K partials of the ``kk`` / ``nn_t`` classes into ``out`` with
    unordered FP32 adds instead of the fixed-order sum."""
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
    package's production count (4), or with ``deterministic=False`` to :func:`router_splits` (at most 2)."""
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
    ``splits`` defaults to the deterministic package's production count (6), or with ``deterministic=False`` to
    :func:`router_splits`."""
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
