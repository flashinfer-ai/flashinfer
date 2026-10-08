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

Compute-capability 9.0 (Hopper) Cake backend for MiniMax Sparse Attention.

The Cake-generated programs behind this module are registered in
``flashinfer.jit.cake_hopper_msa``: ``ROUTES`` maps a logical route
(``<route key>:<stage>``) to the program that serves it and ``MODULES``
carries each program's physical argument order.  This module owns the public
semantics only: argument validation, the host-side plans (sparse decode: head
fold, split count, ring depth, L2 policy and grid form; proxy score: CTA order
and L2 policy of the decode regime, key-split count of the prefill regime;
top-k select: columns per CTA and tile groups per column; sparse prefill: the
v1 / v2 kernel by pages per sequence), the scratch contract of the split
decode, and the CUDA-graph rules.  Dispatch reads host-known scalars and
tensor metadata only; nothing here synchronizes the stream.  Plans, route
checks and the per-geometry launch templates (program, grid, kernel scalars,
retained buffers) are resolved once per host-known key (``functools.lru_cache``;
every key is a tuple of ints / bools / dtypes / the device index, never a
tensor) and never invalidated: they are pure functions of the key and of the
import-time program registry (host round 1).

Routing on SM90 stays in ``flashinfer.msa_ops._sm90_dispatch``: it keeps the
surface's raise rules (paged KV only, K and V interleaved in one allocation,
fp8 e4m3 KV, bf16 q and output, head_dim 128, block 128, topk 16, no LSE,
no per-tensor K/V scales) and calls the functions below for the operation /
shape classes a Cake program serves (``*_route_available``); a launcher asked
for a class without a program raises instead of running another body.

Graph capture: every program launches on the current torch stream through
TVM-FFI and allocates nothing.  The split-decode scratch (partials, merge
state, monotonic per-item counters) is allocated once per
``(device, items, splits, head fold)`` and retained for the process, so a
captured decode keeps valid device pointers after later, larger captures;
capturing a decode whose scratch has not been warmed by an eager call raises.
The persistent decode grid, the proxy-score, top-k and sparse-prefill
programs take no scratch at all.

Program loading: a program's module is JIT-built (FlashInfer, one file lock
per module) and loaded through TVM-FFI at the first eager use of its route;
a program first needed while the current stream is capturing a CUDA graph
raises a RuntimeError naming the program and the remedies instead of
building inside the capture.  A per-geometry eager warm-up before capture
(what the split-decode scratch requires anyway, and what vLLM's capture
loop does) therefore loads every program a capture needs; frameworks that
capture geometries never run eagerly, or that want the build cost at model
load, call ``preload_programs`` (every delivered program of the given op
kinds, read from the route table) once per process (host round 1, phase 3).
"""

from __future__ import annotations

import functools
import heapq
import math
import threading
from dataclasses import dataclass, replace
from typing import Any, Iterable, Optional

import torch
import tvm_ffi

from ..jit.cake_hopper_msa import MODULES, ROUTES, load_hopper_msa_module, route_program
from ..utils import get_compute_capability, get_device_sm_count

_BLOCK_SIZE = 128
_HEAD_DIM = 128
_TOPK = 16
_SUPPORTED_COMPUTE_CAPABILITIES = {(9, 0)}
_LOG2E = 1.4426950408889634
_GRID_AXES = {"grid_x": 0, "grid_y": 1, "grid_z": 2}

# Decode program geometry (fixed by the schedule family).
_KT = 64  # keys per WGMMA M half of a page tile
_HALF_BYTES = _KT * _HEAD_DIM  # one fp8 K (or V) half page
_PAGE_BYTES = 4 * _HALF_BYTES  # one ring stage: K half 0, K half 1, V half 0, V half 1
_HSTAGE_BYTES = (
    2 * _HALF_BYTES
)  # one 64-key ring stage of the HALF option: K half, V half (16 KiB)
# Planner rules mirrored from the Cake source planner (``plan_decode_sm90_v5``, decode rounds 2-4): the persistent
# grid for one-split classes with more items than resident CTAs, and the 64-key (HALF) ring for the two q-len-1
# classes the same-session ABBA proved (GQA 16x4 q1 b128: HALF STAGES 2 + VSEQ; b16 q1: HALF STAGES 4).  Both
# forms are physical variants of their own: ``plan_sparse_decode`` serves them only when the delivered program set
# has their route, otherwise the 128-key variant the latency model picked (``_route_available``).
_PERSIST_RULE = True
_HALF_RULE = True
_HALF_RULE_NC16 = True  # planner rule C (decode round 4): NC 16 one-split class beyond the resident CTAs -> HALF STAGES 2 VSEQ
_NC16_LEAN = (
    "qk",
    "v",
    "vlate",
)  # register-lean levels of rule C (a different binary from the plain HALF VSEQ form)
_NC16_MINB = None  # launch-bound override of rule C (None = SMEM-derived residency: four CTAs per SM)
_IDENT_BYTES = 64 * _HEAD_DIM  # fp8 identity tile
_WARPGROUP = 128  # threads per CTA
_SMEM_PER_SM = 228 * 1024
_HINT_REUSE_THRESHOLD = 1.5
_HINT_REUSE_MULTIWAVE = 0.5  # MTP grids beyond one 128-key wave (items > resident) use the default policy from this reuse on
_HINT_REUSE_FULLWAVE = 1.0  # MTP grids of more than one CTA per SM (grid > SM count) use the default policy from this reuse on
_HINT_STREAM = "evict_first"
_HINT_DEFAULT = "none"
# Proxy-score decode program coordinates.
_PROXY_HEADS = (1, 2, 4)
_PROXY_QLENS = (1, 2, 3, 4)
# Every program family declares a trace carrier parameter (``TRACE=0`` compiles the
# stamp stores out, the parameter stays in the kernel signature); one never-written
# 16-word buffer per device satisfies it (the largest stamp count of the families is 14).
_TRACE_WORDS = 16


def is_hopper_msa_device(device: torch.device | str) -> bool:
    """Return whether ``device`` is a compute-capability 9.0 MSA target."""

    normalized = torch.device(device)
    return (
        normalized.type == "cuda"
        and get_compute_capability(normalized) in _SUPPORTED_COMPUTE_CAPABILITIES
    )


# ---------------------------------------------------------------------------
# Device facts and program launch
# ---------------------------------------------------------------------------


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


@functools.cache
def _num_sms(device_index: int) -> int:
    """Multiprocessor count of one device, resolved once."""

    device = torch.device("cuda", device_index)
    compute_capability = get_compute_capability(device)
    if compute_capability not in _SUPPORTED_COMPUTE_CAPABILITIES:
        raise RuntimeError(
            "the Hopper MSA backend requires compute capability 9.0; "
            f"got {compute_capability[0]}.{compute_capability[1]}"
        )
    return int(get_device_sm_count(device))


# Stream of a launch: ``tvm_ffi.use_torch_stream()`` builds a ``torch.cuda.Stream`` object, formats its device and
# resolves a tvm_ffi device on every call; the raw handle of the current stream on the current device is the same
# value read directly (one tvm_ffi device object per CUDA device, the context object per launch).  Falls back to
# ``use_torch_stream`` when either entry point is absent.  Host round 1 of the Hopper MSA port.
_raw_stream = getattr(torch._C, "_cuda_getCurrentRawStream", None)
try:
    from tvm_ffi.stream import StreamContext as _FFIStreamContext
except Exception:  # noqa: BLE001  (older tvm_ffi layouts)
    _FFIStreamContext = None
_ffi_devices: dict[int, Any] = {}


def _ffi_stream_context():
    """Context running the FFI call on the current torch stream of the current device."""

    if _raw_stream is None or _FFIStreamContext is None:
        return tvm_ffi.use_torch_stream()
    index = torch.cuda.current_device()
    device = _ffi_devices.get(index)
    if device is None:
        device = tvm_ffi.device(f"cuda:{index}")
        _ffi_devices[index] = device
    return _FFIStreamContext(device, _raw_stream(index))


class _Program:
    """One loaded program: its FFI entry and physical argument order."""

    __slots__ = ("entry", "plan", "name", "slots")

    def __init__(
        self, name: str, entry: Any, plan: tuple[tuple[str, str], ...]
    ) -> None:
        self.name = name
        self.entry = entry
        self.plan = plan
        # The argument order bound once: (True, grid axis) for the grid scalars, (False, argument name) otherwise.
        self.slots = tuple(
            (kind == "grid", _GRID_AXES[argument] if kind == "grid" else argument)
            for kind, argument in plan
        )

    def launch(
        self,
        grid: tuple[int, int, int],
        constants: Optional[dict[str, Any]] = None,
        **arguments: Any,
    ) -> None:
        """Launch on the current torch stream with the generated argument order.

        ``constants`` holds the arguments of a launch template resolved once per geometry (kernel scalars, retained
        buffers); ``arguments`` the per-call tensors.  A name present in both is taken from ``arguments``.
        """

        if constants:
            values = [
                int(grid[key])
                if is_grid
                else (arguments[key] if key in arguments else constants[key])
                for is_grid, key in self.slots
            ]
        else:
            values = [
                int(grid[key]) if is_grid else arguments[key]
                for is_grid, key in self.slots
            ]
        with _ffi_stream_context():
            self.entry(*values)


# ---------------------------------------------------------------------------
# Program loading
# ---------------------------------------------------------------------------

# Op kind -> the route-key families (``<family>:<coordinates>:<stage>``) of its delivered programs.
_KIND_FAMILIES: dict[str, tuple[str, ...]] = {
    "sparse_decode": ("decode_v5",),
    "proxy_decode": ("proxy_decode",),
    "proxy_prefill": ("proxy_prefill",),
    "topk_select": ("topk_select",),
    "sparse_prefill": ("prefill_v1", "prefill_v2"),
}
# Op kind -> the launcher below whose eager call loads a program of the kind (named in the capture error).
_KIND_LAUNCHERS: dict[str, str] = {
    "sparse_decode": "hopper_msa_sparse_decode_attention",
    "proxy_decode": "hopper_msa_proxy_score_decode",
    "proxy_prefill": "hopper_msa_proxy_score_prefill",
    "topk_select": "hopper_msa_topk_select",
    "sparse_prefill": "hopper_msa_sparse_attention",
}
PROGRAM_KINDS: tuple[str, ...] = tuple(_KIND_FAMILIES)


def _route_family(route: str) -> str:
    return route.split(":", 1)[0]


def _check_route_families() -> None:
    """Every delivered route belongs to an op kind, or ``preload_programs`` could not reach its program."""

    known = {family for families in _KIND_FAMILIES.values() for family in families}
    unassigned = sorted({_route_family(route) for route in ROUTES} - known)
    if unassigned:
        raise RuntimeError(
            f"Hopper MSA route families without an op kind: {unassigned} (extend _KIND_FAMILIES)"
        )


_check_route_families()


@functools.cache
def _programs_of_kind(kind: str) -> tuple[str, ...]:
    """The delivered programs of one op kind (distinct, sorted), read from the route table."""

    families = _KIND_FAMILIES[kind]
    return tuple(
        sorted(
            {name for route, name in ROUTES.items() if _route_family(route) in families}
        )
    )


def _kind_of_route(route: str) -> str:
    family = _route_family(route)
    return next(kind for kind, families in _KIND_FAMILIES.items() if family in families)


# program name -> its loaded module (this process)
_loaded_modules: dict[str, Any] = {}


def _capture_error(route: str) -> RuntimeError:
    kind = _kind_of_route(route)
    return RuntimeError(
        f"Hopper MSA program {route_program(route)} (route {route!r}) is not loaded and the current stream is "
        f"capturing a CUDA graph: run {_KIND_LAUNCHERS[kind]} eagerly with this geometry once per process before "
        f"capture, or call preload_programs(({kind!r},)) at model load"
    )


def _loaded_module(route: str) -> Any:
    """The module of one route, built (FlashInfer JIT, one file lock per module) and loaded at its first use --
    never under CUDA graph capture: a program first needed there raises instead."""

    name = route_program(route)
    module = _loaded_modules.get(name)
    if module is None:
        if torch.cuda.is_current_stream_capturing():
            raise _capture_error(route)
        module = load_hopper_msa_module(name)
        _loaded_modules[name] = module
    return module


def preload_programs(kinds: Optional[Iterable[str]] = None) -> dict[str, int]:
    """Build and load every delivered program of the op kinds ``kinds`` (default: all of ``PROGRAM_KINDS``) now,
    outside CUDA graph capture, so that a geometry first met under capture finds its program loaded.

    Explicit opt-in.  By default a program is built and loaded at the first eager use of its route -- nothing up
    front, and what a per-geometry eager warm-up before capture (vLLM's capture loop) relies on.  Preloading a kind
    costs one nvcc build per program not yet in the JIT cache (about 4 s each: 22 / 24 / 23 / 4 / 6 programs for
    sparse decode / proxy decode / top-k select / proxy prefill / sparse prefill) and about 10 ms per prebuilt
    program.  Returns the number of programs this call loaded, per kind.  Raises under capture and for an unknown
    kind.
    """

    selected = PROGRAM_KINDS if kinds is None else tuple(kinds)
    unknown = [kind for kind in selected if kind not in _KIND_FAMILIES]
    if unknown:
        raise ValueError(
            f"unknown Hopper MSA program kinds {unknown}; known kinds: {PROGRAM_KINDS}"
        )
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "preload_programs builds and loads programs and must not run under CUDA graph capture"
        )
    loaded: dict[str, int] = {}
    for kind in selected:
        count = 0
        for name in _programs_of_kind(kind):
            if name not in _loaded_modules:
                _loaded_modules[name] = load_hopper_msa_module(name)
                count += 1
        loaded[kind] = count
    return loaded


@functools.cache
def _program(route: str) -> _Program:
    name = route_program(route)
    record = MODULES[name]
    module = _loaded_module(route)
    plan = tuple((str(kind), str(argument)) for kind, argument in record["arg_plan"])
    return _Program(name, getattr(module, record["ffi_entry"]), plan)


def _route_available(route: str) -> bool:
    return route in ROUTES


_trace_carriers: dict[int, torch.Tensor] = {}
_trace_carriers_lock = threading.Lock()


def _trace_carrier(device: torch.device) -> torch.Tensor:
    """The never-written trace buffer every program takes, allocated once per device."""

    index = _device_index(device)
    with _trace_carriers_lock:
        tensor = _trace_carriers.get(index)
        if tensor is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Hopper MSA programs must run eagerly once per device before CUDA graph capture"
                )
            tensor = torch.zeros((_TRACE_WORDS,), dtype=torch.uint64, device=device)
            _trace_carriers[index] = tensor
    return tensor


_int32_dummies: dict[int, torch.Tensor] = {}
_int32_dummies_lock = threading.Lock()


def _int32_dummy(device: torch.device) -> torch.Tensor:
    """A never-read int32 buffer for the optional tensor inputs a program declares (allocated once per device)."""

    index = _device_index(device)
    with _int32_dummies_lock:
        tensor = _int32_dummies.get(index)
        if tensor is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Hopper MSA programs must run eagerly once per device before CUDA graph capture"
                )
            tensor = torch.zeros((4,), dtype=torch.int32, device=device)
            _int32_dummies[index] = tensor
    return tensor


# ---------------------------------------------------------------------------
# Sparse decode: plan, scratch, launch
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HopperDecodePlan:
    """Physical decode variant: head fold, split count, ring depth, prologue issue, L2 policy, grid form.

    ``persist``: one CTA per residency slot walks the items (S = 1; launch grid ``(ctas, 1, 1)``, the item count is a
    kernel scalar, no split scratch).  ``half``: 64-key (16 KiB) ring stages, ``stages`` then counts 64-key stages
    (2 = one page in flight, four CTAs per SM; 4 = two pages); ``vseq`` is its register-lean form (V widen of half 1
    after the P^T publish).  The HALF forms keep the 128-key kernel's argument plan; the route key carries the form
    as trailing segments so the 128-key routes keep their names.
    """

    nc: int
    chunks: int
    splits: int
    stages: int
    pre: int
    hint: str
    persist: bool = False
    half: bool = False
    vseq: bool = False
    lean: tuple[str, ...] = ()
    minb: Optional[int] = None
    ctas: int = 0

    @property
    def route(self) -> str:
        form = ""
        if self.persist:
            form = ":persist"
        elif self.half:
            form = ":half:vseq" if self.vseq else ":half"
        if self.lean:
            form += ":lean-" + "-".join(self.lean)
        if self.minb is not None:
            form += f":minb{int(self.minb)}"
        return f"decode_v5:nc{self.nc}:s{self.splits}:st{self.stages}:{self.hint}{form}:main"

    def grid(self, *, total_q: int, num_kv_heads: int) -> tuple[int, int, int]:
        if self.persist:
            return (self.ctas, 1, 1)
        return (num_kv_heads * self.chunks * self.splits, total_q, 1)


def _decode_smem_bytes(nc: int, stages: int, half: bool = False) -> int:
    off_ident = stages * (_HSTAGE_BYTES if half else _PAGE_BYTES)
    off_q16 = off_ident + _IDENT_BYTES
    off_p16 = off_q16 + nc * 2 * _HEAD_DIM
    off_red = off_p16 + 2 * 2 * nc * 2 * _KT
    off_lred = off_red + nc * 4 * 4
    off_flag = off_lred + nc * 4 * 4
    off_meta = off_flag + 16
    total = off_meta + 2 * _TOPK * 4
    return -(-total // 128) * 128


def _decode_smem_bytes_persistent(nc: int, stages: int) -> int:
    """The persistent CTA's map: two Q^T buffers (items alternate) and two (block, page head) tables."""
    off_ident = stages * _PAGE_BYTES
    off_q16 = off_ident + _IDENT_BYTES
    off_p16 = off_q16 + 2 * nc * 2 * _HEAD_DIM
    off_red = off_p16 + 2 * 2 * nc * 2 * _KT
    off_lred = off_red + nc * 4 * 4
    off_meta = off_lred + nc * 4 * 4
    total = off_meta + 2 * 2 * _TOPK * 4
    return -(-total // 128) * 128


def _resident_ctas(num_sms: int, smem: int, per_sm_cap: int) -> int:
    return num_sms * max(1, min(per_sm_cap, _SMEM_PER_SM // (smem + 1024)))


def _max_splits(nc: int, stages: int, half: bool = False) -> int:
    return min(
        _TOPK,
        (stages * (_HSTAGE_BYTES if half else _PAGE_BYTES)) // (_WARPGROUP * nc * 2),
    )


@functools.lru_cache(maxsize=4096)
def plan_sparse_decode(
    *,
    total_q: int,
    num_q_heads: int,
    num_kv_heads: int,
    num_sms: int,
    seqlen_q: int,
    max_pages: int,
) -> HopperDecodePlan:
    """Select the decode variant: exact port of the Cake source planner ``plan_decode_sm90_v5`` (decode round 4).

    The head fold follows the GQA group (8 heads per CTA up to group 8, else 16); the split count and ring depth
    minimize the modelled per-CTA chain plus wave cost.  Three class rules then replace the model's 128-key pick by a
    64-key (HALF) ring form: (A) the q-len-1 one-split class with 8-head CTAs and more items than resident CTAs runs
    two 64-key stages in the VSEQ form (four CTAs per SM, one wave); (B) the 16-item eight-split class with 8-head
    CTAs runs four 64-key stages; (C) the q-len-1 one-split class with 16-head CTAs beyond the resident CTAs runs two
    64-key stages in the register-lean VSEQ form.  Any other one-split class with more items than resident CTAs runs
    the persistent grid.  The page loads use the default L2 policy when the tokens of one sequence re-select the same
    pages inside the launch (``seqlen_q * topk / max_pages`` at or above 1.5), and for the multi-token (MTP) grids
    beyond one wave (re-selection at or above 0.5) or filling a full two-CTAs-per-SM wave (at or above 1.0), whose
    co-scheduled tokens re-select pages that ``evict_first`` has already dropped.  When the delivered program set has
    no route for a HALF / persistent form, the plan falls back to the 128-key variant the model picked.
    """

    group = num_q_heads // num_kv_heads
    nc = 8 if group <= 8 else 16
    chunks = -(-group // nc)
    base = total_q * num_kv_heads * chunks
    c_page = 0.9 if nc == 8 else 1.3
    hbm_us_per_byte = 1.0e6 / 3.0e12
    best = None
    for stages in (2, 3):
        smem = _decode_smem_bytes(nc, stages)
        resident = _resident_ctas(num_sms, smem, 2)
        for splits in (1, 2, 4, 8, 16):
            if splits > _max_splits(nc, stages):
                continue
            pages = _TOPK // splits
            grid = base * splits
            active = min(grid, resident)
            waves = -(-grid // resident)
            page_stream = active * _PAGE_BYTES * hbm_us_per_byte
            first_land = 1.0 + page_stream
            later = max(c_page, page_stream)
            epi = (
                0.2
                if splits == 1
                else 0.8 + 0.25 + 0.056 * splits * (nc // 8) * (1.0 if nc == 8 else 1.6)
            )
            chain = 0.9 + first_land + (pages - 1) * later + c_page + epi
            launch = 0.004 * max(0, active - 128)
            cost = 0.7 + launch + chain + (waves - 1) * (pages * later + 1.5)
            if nc == 8 and grid <= num_sms and pages == 1:
                cost -= 0.3
            if stages == 3:
                cost += 0.1
            if best is None or cost < best[0]:
                best = (cost, splits, stages, resident)
    if best is None:
        raise ValueError("no admissible decode variant")
    _, splits, stages, resident = best
    resident_plain = resident
    grid_plain = base * splits
    half = vseq = False
    lean: tuple[str, ...] = ()
    minb = None
    if _HALF_RULE and nc == 8 and seqlen_q == 1 and splits == 1 and base > resident:
        # rule A: the q-len-1 one-split class beyond the resident 128-key CTAs (GQA 16x4 b128)
        half, stages, vseq = True, 2, True
        resident = _resident_ctas(num_sms, _decode_smem_bytes(nc, 2, half=True), 4)
    elif _HALF_RULE and nc == 8 and splits == 8 and base * splits <= resident:
        # rule B: the 16-item eight-split class (b16 q1; MTP g16x2 q8 b1 / g16x4 q4 b1)
        half, stages = True, 4
        resident = _resident_ctas(num_sms, _decode_smem_bytes(nc, 4, half=True), 4)
    if (
        _HALF_RULE_NC16
        and nc == 16
        and seqlen_q == 1
        and splits == 1
        and base > resident
    ):
        # rule C: the NC 16 one-split class beyond the resident 128-key CTAs (GQA 64x4 b128): register-lean HALF VSEQ
        half, stages, vseq, lean, minb = True, 2, True, _NC16_LEAN, _NC16_MINB
        resident = _resident_ctas(
            num_sms, _decode_smem_bytes(nc, 2, half=True), minb or 4
        )
    reuse = (seqlen_q * _TOPK) / max_pages if max_pages else 0.0
    hint = _HINT_DEFAULT if reuse >= _HINT_REUSE_THRESHOLD else _HINT_STREAM
    if (
        seqlen_q > 1
        and hint != _HINT_DEFAULT
        and (
            (reuse >= _HINT_REUSE_MULTIWAVE and base > resident_plain)
            or (reuse >= _HINT_REUSE_FULLWAVE and grid_plain > num_sms)
        )
    ):
        hint = _HINT_DEFAULT
    persist = _PERSIST_RULE and splits == 1 and base > resident and not half
    model = HopperDecodePlan(
        nc=nc,
        chunks=chunks,
        splits=splits,
        stages=best[2],
        pre=1,
        hint=hint,
        ctas=base * splits,
    )
    plan = model
    if half:
        plan = replace(model, stages=stages, half=True, vseq=vseq, lean=lean, minb=minb)
    elif persist:
        slots = _resident_ctas(num_sms, _decode_smem_bytes_persistent(nc, stages), 2)
        plan = replace(model, persist=True, ctas=min(base, slots))
    if plan is not model and not _route_available(plan.route):
        plan = model
    return plan


_decode_scratch: dict[tuple[int, int, int, int], tuple[torch.Tensor, ...]] = {}
_decode_scratch_lock = threading.Lock()


def _scratch(
    device: torch.device, *, items: int, splits: int, nc: int
) -> tuple[torch.Tensor, ...]:
    """Split partials, merge statistics and the monotonic per-item counters of the decode program.

    Retained for the process per ``(device, items, splits, nc)``: the counters
    advance across launches (the merge protocol never resets them) and a
    captured graph keeps pointers into these buffers.
    """

    key = (_device_index(device), int(items), int(splits), int(nc))
    with _decode_scratch_lock:
        buffers = _decode_scratch.get(key)
        if buffers is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "msa_sparse_decode_attention must be invoked eagerly with this batch, head and "
                    "page-table geometry before CUDA graph capture (its split scratch is allocated once)"
                )
            buffers = (
                torch.empty(
                    (items * splits * _WARPGROUP * (nc // 2),),
                    dtype=torch.uint32,
                    device=device,
                ),
                torch.empty(
                    (items * splits * 2 * nc,), dtype=torch.float32, device=device
                ),
                torch.zeros((items,), dtype=torch.uint32, device=device),
                torch.zeros((items,), dtype=torch.uint32, device=device),
            )
            _decode_scratch[key] = buffers
    return buffers


@functools.lru_cache(maxsize=4096)
def _decode_template(
    *,
    total_q: int,
    num_q_heads: int,
    num_kv_heads: int,
    seqlen_q: int,
    max_pages: int,
    device_index: int,
) -> tuple[_Program, tuple[int, int, int], dict[str, Any]]:
    """The launch of one decode geometry resolved once: program, grid and the per-plan arguments (kernel scalars,
    the trace carrier, the retained split scratch).  Pure in its key; the per-call tensors join at launch time.
    A first use under CUDA-graph capture raises exactly as the eager-first rules of the buffers did (nothing cached)."""

    plan = plan_sparse_decode(
        total_q=total_q,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        num_sms=_num_sms(device_index),
        seqlen_q=seqlen_q,
        max_pages=max_pages,
    )
    device = torch.device("cuda", device_index)
    items = total_q * num_kv_heads * plan.chunks
    constants: dict[str, Any] = dict(
        total_q=total_q,
        seqlen_q=seqlen_q,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        max_pages=max_pages,
        num_chunks=plan.chunks,
        zero_u32=0,
        trace=_trace_carrier(device),
    )
    if plan.persist:
        # One CTA per residency slot loops over the items: the item count is a kernel scalar, no split scratch.
        constants["num_items"] = items
    else:
        part_o, part_ml, counters, done = _scratch(
            device, items=items, splits=plan.splits, nc=plan.nc
        )
        constants.update(part_o=part_o, part_ml=part_ml, counters=counters, done=done)
    return (
        _program(plan.route),
        plan.grid(total_q=total_q, num_kv_heads=num_kv_heads),
        constants,
    )


def hopper_msa_sparse_decode_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_indices: torch.Tensor,
    *,
    page_table: torch.Tensor,
    seqused_k: torch.Tensor,
    seqlen_q: int,
    softmax_scale: Optional[float],
    out: torch.Tensor,
) -> torch.Tensor:
    """Sparse decode attention on compute capability 9.0.  Writes and returns ``out``.

    ``k`` and ``v`` are the ``(num_pages, num_kv_heads, 128, 128)`` fp8 e4m3
    halves of the interleaved cache (the layout contract is checked by the
    caller); the program reads them through their own strides.  Query ``i``
    of a sequence sits at position ``seqused_k - seqlen_q + i`` (right-aligned
    causal); the program takes no query-offset input, so the public surface
    rejects ``q_offset`` and ``causal=False`` instead of ignoring them.
    """

    total_q, num_q_heads, head_dim = (int(x) for x in q.shape)
    if head_dim != _HEAD_DIM:
        raise ValueError(f"head_dim must be {_HEAD_DIM}, got {head_dim}")
    if q.dtype != torch.bfloat16 or not q.is_contiguous():
        raise ValueError("SM90 sparse decode needs contiguous bf16 q")
    if k.dtype != torch.float8_e4m3fn or v.dtype != torch.float8_e4m3fn:
        raise ValueError("SM90 sparse decode needs an fp8 e4m3 KV cache")
    if (
        k.ndim != 4
        or k.shape[2] != _BLOCK_SIZE
        or k.shape[3] != _HEAD_DIM
        or k.shape != v.shape
    ):
        raise ValueError(
            f"paged k/v must be (num_pages, num_kv_heads, {_BLOCK_SIZE}, {_HEAD_DIM})"
        )
    if k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("k and v must be dense along head_dim")
    num_kv_heads = int(k.shape[1])
    if seqlen_q <= 0 or total_q % seqlen_q:
        raise ValueError(
            f"q rows ({total_q}) must be batch_size * seqlen_q ({seqlen_q})"
        )
    batch = total_q // seqlen_q
    if q2k_indices.dtype != torch.int32 or not q2k_indices.is_contiguous():
        raise ValueError("q2k_indices must be contiguous int32")
    if tuple(q2k_indices.shape) != (num_kv_heads, total_q, _TOPK):
        raise NotImplementedError(
            f"SM90 sparse decode serves q2k_indices of shape (num_kv_heads, total_q, {_TOPK}); "
            f"got {tuple(q2k_indices.shape)}"
        )
    if (
        page_table.dtype != torch.int32
        or not page_table.is_contiguous()
        or page_table.ndim != 2
    ):
        raise ValueError("page_table must be contiguous int32 (batch_size, max_pages)")
    if int(page_table.shape[0]) != batch:
        raise ValueError(
            f"page_table has {page_table.shape[0]} rows for batch_size {batch}"
        )
    if (
        seqused_k.dtype != torch.int32
        or not seqused_k.is_contiguous()
        or int(seqused_k.numel()) != batch
    ):
        raise ValueError(f"seqused_k must be contiguous int32 with {batch} entries")
    if out.shape != q.shape or out.dtype != torch.bfloat16 or not out.is_contiguous():
        raise ValueError("out must be a contiguous bf16 tensor shaped like q")
    max_pages = int(page_table.shape[1])
    program, grid, constants = _decode_template(
        total_q=total_q,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        seqlen_q=int(seqlen_q),
        max_pages=max_pages,
        device_index=_device_index(q.device),
    )
    scale = (
        1.0 / math.sqrt(_HEAD_DIM) if softmax_scale is None else float(softmax_scale)
    )
    program.launch(
        grid,
        constants,
        Q32=q.view(torch.uint32),
        K=k.view(torch.uint8),
        V=v.view(torch.uint8),
        O=out,
        q2k_indices=q2k_indices,
        page_table=page_table,
        seqused_k=seqused_k,
        softmax_scale_log2=scale * _LOG2E,
    )
    return out


# ---------------------------------------------------------------------------
# Proxy score, decode regime
# ---------------------------------------------------------------------------


def proxy_decode_route_available(
    *, q_dtype: torch.dtype, num_q_heads: int, num_kv_heads: int, max_seqlen_q: int
) -> bool:
    """Whether a Cake proxy-score program serves these host-known coordinates.

    The decode-regime programs exist for fp8 e4m3 and bf16 q / index cache,
    one index (KV) head, ``Hq`` in {1, 2, 4} and ``max_seqlen_q`` in
    {1, 2, 3, 4}; ``_sm90_dispatch`` keeps its other decode schedules for the
    remaining admitted coordinates.
    """

    return (
        q_dtype in (torch.float8_e4m3fn, torch.bfloat16)
        and int(num_kv_heads) == 1
        and int(num_q_heads) in _PROXY_HEADS
        and int(max_seqlen_q) in _PROXY_QLENS
    )


def proxy_route(*, q_dtype: torch.dtype, num_q_heads: int, max_seqlen_q: int) -> str:
    dtype = "fp8" if q_dtype == torch.float8_e4m3fn else "bf16"
    return f"proxy_decode:{dtype}:hq{int(num_q_heads)}:sq{int(max_seqlen_q)}:main"


def plan_proxy_score(*, batch: int, max_k_tiles: int, num_pages: int) -> bool:
    """CTA order: walk the batch for a fixed tile unless the (batch x tiles) rectangle is ragged."""

    return bool(num_pages * 20 >= batch * max_k_tiles * 17)


_L2_BYTES = 50 << 20  # compute capability 9.0 (H100 / H200) L2


def proxy_evict_first(
    *, batch: int, max_k_tiles: int, num_pages: int, fp8: bool
) -> bool:
    """L2 eviction policy of the index-K page loads: ``evict_first`` while the footprint is short.

    The footprint upper bound is ``min(batch * max_k_tiles, num_pages)`` pages (no device sync);
    below five L2 capacities the streamed pages are inserted ahead of the resident lines, beyond
    it the default policy avoids the ~3 % steady-state cost of the hint.
    """

    page_bytes = _BLOCK_SIZE * _HEAD_DIM * (1 if fp8 else 2)
    footprint = min(int(batch) * int(max_k_tiles), int(num_pages)) * page_bytes
    return footprint < 5 * _L2_BYTES


@functools.lru_cache(maxsize=4096)
def _proxy_decode_template(
    *,
    q_dtype: torch.dtype,
    num_q_heads: int,
    max_seqlen_q: int,
    batch: int,
    max_k_tiles: int,
    total_q: int,
    num_pages: int,
    pt_stride: int,
    has_qoff: bool,
    device_index: int,
) -> tuple[_Program, tuple[int, int, int], dict[str, Any]]:
    """The launch of one decode-regime proxy geometry resolved once: program, grid (CTA order) and kernel scalars."""

    batch_fast = plan_proxy_score(
        batch=batch, max_k_tiles=max_k_tiles, num_pages=num_pages
    )
    evict_first = proxy_evict_first(
        batch=batch,
        max_k_tiles=max_k_tiles,
        num_pages=num_pages,
        fp8=q_dtype == torch.float8_e4m3fn,
    )
    constants: dict[str, Any] = dict(
        has_qoff=1 if has_qoff else 0,
        pt_stride=pt_stride,
        max_k_tiles=max_k_tiles,
        total_q=total_q,
        batch_fast=1 if batch_fast else 0,
        evict_first=1 if evict_first else 0,
        trace=_trace_carrier(torch.device("cuda", device_index)),
    )
    program = _program(
        proxy_route(q_dtype=q_dtype, num_q_heads=num_q_heads, max_seqlen_q=max_seqlen_q)
    )
    grid = (batch, max_k_tiles, 1) if batch_fast else (max_k_tiles, batch, 1)
    return program, grid, constants


def hopper_msa_proxy_score_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    page_table: torch.Tensor,
    seqused_k: torch.Tensor,
    per_head: torch.Tensor,
    max_seqlen_q: int,
    batch_size: int,
    q_offset: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """MSA proxy score, decode regime, on compute capability 9.0.  Writes ``per_head``."""

    total_q, num_q_heads, head_dim = (int(x) for x in q.shape)
    if head_dim != _HEAD_DIM:
        raise ValueError(f"head_dim must be {_HEAD_DIM}, got {head_dim}")
    if (
        k.ndim != 4
        or k.shape[1] != 1
        or k.shape[2] != _BLOCK_SIZE
        or k.shape[3] != _HEAD_DIM
    ):
        raise ValueError(
            f"the MSA index cache must be (num_pages, 1, {_BLOCK_SIZE}, {_HEAD_DIM})"
        )
    if q.dtype != k.dtype or not q.is_contiguous() or not k.is_contiguous():
        raise ValueError(
            "q and the index cache must be contiguous tensors of one dtype"
        )
    if not proxy_decode_route_available(
        q_dtype=q.dtype,
        num_q_heads=num_q_heads,
        num_kv_heads=int(k.shape[1]),
        max_seqlen_q=max_seqlen_q,
    ):
        raise NotImplementedError(
            f"no Cake SM90 proxy-score program for dtype {q.dtype}, Hq {num_q_heads}, max_seqlen_q {max_seqlen_q}"
        )
    sq = int(max_seqlen_q)
    batch = int(batch_size)
    if total_q != batch * sq:
        raise ValueError(
            f"decode proxy needs total_q == batch_size * max_seqlen_q ({batch} * {sq}), got {total_q}"
        )
    if (
        page_table.dtype != torch.int32
        or page_table.ndim != 2
        or page_table.stride(1) != 1
    ):
        raise ValueError(
            "page_table must be int32 (batch_size, max_pages) with unit column stride"
        )
    if int(page_table.shape[0]) != batch:
        raise ValueError(
            f"page_table has {page_table.shape[0]} rows for batch_size {batch}"
        )
    if (
        seqused_k.dtype != torch.int32
        or not seqused_k.is_contiguous()
        or int(seqused_k.numel()) != batch
    ):
        raise ValueError(f"seqused_k must be contiguous int32 with {batch} entries")
    num_heads, max_k_tiles, out_q = (int(x) for x in per_head.shape)
    if (
        (num_heads, out_q) != (num_q_heads, total_q)
        or per_head.dtype != torch.float32
        or not per_head.is_contiguous()
    ):
        raise ValueError(
            f"per_head must be contiguous float32 ({num_q_heads}, max_k_tiles, {total_q})"
        )
    if int(page_table.shape[1]) < max_k_tiles:
        raise ValueError("page_table is narrower than max_k_tiles")
    if q_offset is not None and (
        q_offset.dtype != torch.int32
        or not q_offset.is_contiguous()
        or int(q_offset.numel()) != batch
    ):
        raise ValueError(f"q_offset must be contiguous int32 with {batch} entries")
    num_pages = int(k.shape[0])
    if q.dtype == torch.float8_e4m3fn:
        q2 = q.view(torch.uint8).view(total_q * num_q_heads, _HEAD_DIM)
        k2 = k.view(torch.uint8).view(num_pages * _BLOCK_SIZE, _HEAD_DIM)
    else:
        q2 = q.view(total_q * num_q_heads, _HEAD_DIM)
        k2 = k.view(num_pages * _BLOCK_SIZE, _HEAD_DIM)
    program, grid, constants = _proxy_decode_template(
        q_dtype=q.dtype,
        num_q_heads=num_q_heads,
        max_seqlen_q=sq,
        batch=batch,
        max_k_tiles=max_k_tiles,
        total_q=total_q,
        num_pages=num_pages,
        pt_stride=int(page_table.stride(0)),
        has_qoff=q_offset is not None,
        device_index=_device_index(q.device),
    )
    program.launch(
        grid,
        constants,
        Q=q2,
        K=k2,
        out=per_head,
        page_table=page_table,
        seqused_k=seqused_k,
        q_offset=q_offset if q_offset is not None else seqused_k,
    )
    return per_head


# ---------------------------------------------------------------------------
# Proxy score, prefill regime (chunked prefill scoring)
# ---------------------------------------------------------------------------

# Prefill-proxy program geometry and planner constants, mirrored from the Cake source module
# (``minimax_sparse_attention_proxy_score_prefill_sm90``): both production variants are 256-row CTA tiles with four
# consumer warpgroups (one CTA per SM), four 16 KiB K ring stages, TAIL4 scalar tail.
_PXP_BM = 128  # query rows of the 128-row forms (the planner's reference tile)
_PXP_BM256 = 256  # query rows per CTA of the WG4 production forms
_PXP_STAGES = 4
_PXP_VARIANT_FP32 = (
    "wg4st"  # exact-product f32 accumulation (default: the original MSA numerics)
)
_PXP_VARIANT_F16 = (
    "f16db"  # f16 accumulation (FI #6140 numerics), two accumulators per warpgroup
)
_PXP_VARIANT_F16_SHORT = "f16t4F"  # f16, FI #6140's shape: 128-row CTAs, two consumer warpgroups, two CTAs per SM, FI's fold
_PXP_VARIANT_F16_W3 = "f16w3L"  # f16, 192-row three-warpgroup form, one CTA per SM, longest-first dispatch
_PXP_WG_ROWS = 64
_PXP_F16_SHORT_MAX_K_TILES = (
    512  # <= 64K-token K cache: the 128 / 192-row f16 forms; beyond: the 256-row f16db
)
_PXP_F16_SHORT_MIN_TILES_PER_CTA = (
    4  # FI's pick below this many key tiles per CTA -> one CTA per SM instead
)
_PXP_F16_W3_MAX_MTILES = 12  # chunk <= 1536 tokens in 128-row tiles
_PXP_F16_W3_MAX_SPLIT = 8  # largest admissible one-CTA-per-SM split of the 192-row form
_PXP_F16_W3_MIN_TILES_PER_CTA = 8  # key tiles per CTA that split must leave
# CTA shape of each production variant (``variant_bm`` / ``variant_slots_per_sm`` / the ``LPT`` knob of the module).
_PXP_FORMS = {
    "wg4st": dict(bm=256, slots=1, lpt=False, f16=False),
    "f16db": dict(bm=256, slots=1, lpt=False, f16=True),
    "f16t4F": dict(bm=128, slots=2, lpt=True, f16=True),
    "f16w3L": dict(bm=192, slots=1, lpt=True, f16=True),
}
# Causal-aware key-split planner (``select_nsplit_causal`` / ``estimate_span_us``): CTA cost = C_CTA + tiles x C_TILE
# + dead x C_DEAD, CTAs list-scheduled longest-first onto the residency slots.  Constants are H100 trace fits.
_PXP_C_TILE2 = (
    0.78  # us per 128 x 128 x 128 page tile per CTA with two CTAs resident per SM
)
_PXP_C_TILE1 = 0.53  # us per tile for a CTA alone on its SM
_PXP_C_TILE2_SLOW = 0.93  # us per tile of the slowest CTAs of a full single wave
_PXP_C_CTA = 3.2  # us per CTA outside the tile loop (start -> first tile landed, drain)
_PXP_C_DEAD_TAIL4 = (
    0.02  # us per -inf tile of the TAIL4 scalar tail (the production variants)
)
_PXP_SLOTS_PER_SM_REF = 2  # residency of the 128-row reference form
_PXP_C_TILE_M256 = (
    0.81  # us per 256-row x 128-key page tile of a one-CTA-per-SM variant
)
_PXP_C_TILE_M256_SLOW = 0.834  # per-tile p90 of the same trace
_PXP_DRAIN_M256 = 0.25  # list-schedule drain of one-CTA-per-SM 256-row CTAs as a fraction of the shortest CTA
_PXP_LPT_MAX_JOBS = 3000
# Per-tile constants of the f16-accumulation forms (``PLAN_C_TILE_F16`` ... of the module): aliases of the f32
# constants at the mirrored revision ("= f32 until the round-4 trace lands"); separate names so a recalibration
# changes one line here and in the module.
_PXP_C_TILE_F16 = _PXP_C_TILE_M256
_PXP_C_TILE_F16_SLOW = _PXP_C_TILE_M256_SLOW
_PXP_C_TILE2_F16 = 0.78
_PXP_C_CTA_F16 = _PXP_C_CTA
_PXP_DRAIN_F16 = _PXP_DRAIN_M256
_PXP_PLAN_CANDIDATES = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    14,
    16,
    20,
    24,
    28,
    32,
    40,
    48,
    64,
    96,
    128,
)


def _pxp_cta_classes(
    *, n_mtiles: int, chunk: int, ctx: int, max_k_tiles: int, nsplit: int, bm: int
):
    """Yield ``(n_comp, n_dead, multiplicity)`` over the (row tile, split) CTAs of one (head, sequence) with
    ``chunk`` query tokens after ``ctx - chunk`` prefix tokens (the kernel's own t_lim / round-robin arithmetic)."""
    pfx = max(int(ctx) - int(chunk), 0)
    nb = min(-(-int(ctx) // _BLOCK_SIZE), int(max_k_tiles))
    q2, r2 = divmod(int(max_k_tiles), nsplit)
    for mt in range(int(n_mtiles)):
        m0 = max(0, min(mt * bm, int(chunk) - bm))
        qmax = min(m0 + bm - 1, int(chunk) - 1) + pfx
        t_lim = min(qmax // _BLOCK_SIZE + 1, nb)
        q, r = divmod(t_lim, nsplit)
        bounds = sorted({0, min(r, nsplit), min(r2, nsplit), nsplit})
        for lo, hi in zip(bounds, bounds[1:], strict=False):
            if hi > lo:
                nc = q + 1 if lo < r else q
                tot = q2 + 1 if lo < r2 else q2
                yield nc, max(tot - nc, 0), hi - lo


def _pxp_estimate_span_us(
    *,
    n_mtiles: int,
    hq: int,
    batch: int,
    chunk: int,
    ctx: int,
    max_k_tiles: int,
    nsplit: int,
    c_dead: float,
    sm_count: int,
    slots_per_sm: int,
    bm: int,
    f16: bool,
) -> float:
    """Cost model of one launch (us): exact port of the Cake planner's ``estimate_span_us`` (``f16`` selects the
    per-tile constants of the f16-accumulation forms)."""
    nsplit = max(1, min(int(nsplit), int(max_k_tiles)))
    n_ctas = int(n_mtiles) * int(hq) * int(batch) * nsplit
    single = n_ctas <= sm_count
    c_t2 = _PXP_C_TILE2_F16 if f16 else _PXP_C_TILE2
    c_tile = _PXP_C_TILE1 if single else c_t2 * slots_per_sm / _PXP_SLOTS_PER_SM_REF
    one_cta = (
        bm != _PXP_BM
    )  # 192 / 256-row one-ring forms: one CTA per SM, no sibling sharing
    c_m, c_m_slow = (
        (_PXP_C_TILE_F16, _PXP_C_TILE_F16_SLOW)
        if f16
        else (_PXP_C_TILE_M256, _PXP_C_TILE_M256_SLOW)
    )
    if one_cta:
        c_tile = c_m * bm / _PXP_BM256
    slots = sm_count if single else sm_count * slots_per_sm
    mult = int(hq) * int(batch)
    classes = list(
        _pxp_cta_classes(
            n_mtiles=n_mtiles,
            chunk=chunk,
            ctx=ctx,
            max_k_tiles=max_k_tiles,
            nsplit=nsplit,
            bm=bm,
        )
    )
    c_cta = _PXP_C_CTA_F16 if f16 else _PXP_C_CTA
    jobs = [(c_cta + nc * c_tile + nd * c_dead, m * mult) for nc, nd, m in classes]
    jobs.sort(reverse=True)
    c_max = jobs[0][0]
    if n_ctas <= slots:
        if single and not one_cta:
            return c_max
        nc_max = max(nc for nc, _nd, _m in classes)
        if one_cta:
            return c_max + nc_max * (c_m_slow - c_m) * bm / _PXP_BM256
        return c_max + nc_max * (_PXP_C_TILE2_SLOW - _PXP_C_TILE2)
    total = sum(c * m for c, m in jobs)
    drain = ((_PXP_DRAIN_F16 if f16 else _PXP_DRAIN_M256) if one_cta else 0.5) * jobs[
        -1
    ][0]
    if n_ctas <= _PXP_LPT_MAX_JOBS:
        h = [0.0] * slots
        for c, m in jobs:
            for _ in range(m):
                heapq.heapreplace(h, h[0] + c)
        return max(h) + drain
    return total / slots + drain


@functools.lru_cache(maxsize=4096)
def select_proxy_prefill_nsplit(
    *,
    n_mtiles: int,
    hq: int,
    batch: int,
    max_k_tiles: int,
    chunk: int,
    ctx: int,
    sm_count: int,
    bm: int,
    slots_per_sm: int,
    f16: bool,
) -> int:
    """Key-split count of a prefill-proxy launch: the Cake planner's ``select_nsplit_causal`` -- the ``nsplit``
    minimising the modelled span over the candidate counts plus the slot-filling counts; the one-CTA-per-SM forms
    (192 / 256-row tiles) take the plain minimum, the 128-row form the smallest count within 1 % of the best."""
    base = int(n_mtiles) * int(hq) * int(batch)
    slots = sm_count * slots_per_sm
    cands = set(_PXP_PLAN_CANDIDATES)
    for k in range(1, 9):
        for fill in ((k * slots) // base, (k * sm_count) // base):
            cands.update((fill - 1, fill, fill + 1))
    ests: dict[int, float] = {}
    for cand in sorted(cands):
        nk = min(cand, int(max_k_tiles))
        if nk < 1 or nk in ests or base * nk > 64 * slots:
            continue
        ests[nk] = _pxp_estimate_span_us(
            n_mtiles=n_mtiles,
            hq=hq,
            batch=batch,
            chunk=chunk,
            ctx=ctx,
            max_k_tiles=max_k_tiles,
            nsplit=nk,
            c_dead=_PXP_C_DEAD_TAIL4,
            sm_count=sm_count,
            slots_per_sm=slots_per_sm,
            bm=bm,
            f16=f16,
        )
    best = min(ests.values())
    if bm != _PXP_BM:
        return min(ests, key=ests.get)
    return min(nk for nk, e in ests.items() if e <= best * 1.01)


@dataclass(frozen=True)
class HopperProxyPrefillPlan:
    """Physical prefill-proxy variant (accumulation precision and CTA shape) and the launch geometry of one call.

    ``lpt``: the variant dispatches its CTAs in global longest-first order, which needs the launch grid
    ``(hq * nsplit, n_mtiles, batch)`` instead of ``(hq * n_mtiles, nsplit, batch)``.
    """

    variant: str
    stages: int
    n_mtiles: int
    nsplit: int
    lpt: bool = False

    @property
    def route(self) -> str:
        return f"proxy_prefill:fp8:st{self.stages}:sb:{self.variant}:main"

    def grid(self, *, hq: int, batch: int) -> tuple[int, int, int]:
        if self.lpt:
            return (hq * self.nsplit, self.n_mtiles, batch)
        return (hq * self.n_mtiles, self.nsplit, batch)


@functools.lru_cache(maxsize=256)
def proxy_prefill_route_available(
    *, q_dtype: torch.dtype, num_kv_heads: int, use_fp32_acc: bool
) -> bool:
    """Whether Cake prefill-proxy programs serve these host-known coordinates (fp8 q with a one-head index cache)
    for the requested accumulation precision: every CTA shape the plan can pick for it must be delivered."""

    return (
        q_dtype == torch.float8_e4m3fn
        and int(num_kv_heads) == 1
        and all(
            _route_available(r)
            for r in proxy_prefill_routes(use_fp32_acc=bool(use_fp32_acc))
        )
    )


def proxy_prefill_routes(*, use_fp32_acc: bool) -> tuple[str, ...]:
    """Every route the host plan can select for one accumulation precision (the f16 path has three CTA shapes)."""
    variants = (
        (_PXP_VARIANT_FP32,)
        if use_fp32_acc
        else (_PXP_VARIANT_F16_SHORT, _PXP_VARIANT_F16_W3, _PXP_VARIANT_F16)
    )
    return tuple(f"proxy_prefill:fp8:st{_PXP_STAGES}:sb:{v}:main" for v in variants)


def select_proxy_prefill_nsplit_fi(
    *, n_mtiles: int, hq: int, batch: int, max_k_tiles: int, sm_count: int
) -> int:
    """FI #6140's SM90 split rule (``select_nsplit`` of the module): minimise waves x (tiles per CTA + 1) with a
    x64 penalty when the grid does not fill the machine; negative candidates are ``-cand * SMs // base CTAs``."""
    base_ctas = int(n_mtiles) * int(hq) * int(batch)
    best_cost, best_nk = 0x7FFFFFFF, 1
    for cand in (
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        14,
        16,
        20,
        24,
        32,
        40,
        48,
        64,
        96,
        128,
        -1,
        -2,
        -3,
        -4,
        -5,
        -6,
        -8,
        -10,
        -12,
        -16,
        -20,
        -24,
        -32,
        -40,
        -48,
    ):
        nk = cand if cand > 0 else (-cand * int(sm_count)) // base_ctas
        nk = max(1, min(nk, int(max_k_tiles)))
        n_ctas = base_ctas * nk
        cost = ((n_ctas + int(sm_count) - 1) // int(sm_count)) * (
            (int(max_k_tiles) + nk - 1) // nk + 1
        )
        if n_ctas <= int(sm_count):
            cost *= 64
        if cost < best_cost:
            best_cost, best_nk = cost, nk
    return best_nk


def f16_short_small_work_nsplit(
    nsplit: int, *, n_mtiles: int, hq: int, batch: int, max_k_tiles: int, sm_count: int
) -> int:
    """FI's split on FI's CTA shape, except in the prologue-bound regime (``f16_short_small_work_nsplit`` of the
    module): when the pick leaves fewer than ``_PXP_F16_SHORT_MIN_TILES_PER_CTA`` key tiles per CTA, take the largest
    split that still puts one CTA on (nearly) every SM.  Only ever lowers the split."""
    base = int(n_mtiles) * int(hq) * int(batch)
    if -(-int(max_k_tiles) // int(nsplit)) >= _PXP_F16_SHORT_MIN_TILES_PER_CTA:
        return int(nsplit)
    one = max(1, int(sm_count) // max(1, base))
    return int(one) if one < int(nsplit) else int(nsplit)


def f16_short_form(
    *, hq: int, batch: int, max_k_tiles: int, max_seqlen_q: int, sm_count: int
) -> tuple[str, str]:
    """``(variant, planner)`` of the f16 path for K caches of at most ``_PXP_F16_SHORT_MAX_K_TILES`` pages
    (``f16_short_form`` of the module): the 192-row three-warpgroup LPT form with the causal planner for short
    chunks (at most ``_PXP_F16_W3_MAX_MTILES`` 128-row tiles) whose one-CTA-per-SM split is at most
    ``_PXP_F16_W3_MAX_SPLIT`` and leaves at least ``_PXP_F16_W3_MIN_TILES_PER_CTA`` key tiles per CTA; otherwise FI's
    128-row shape with FI's split rule."""
    if -(-int(max_seqlen_q) // _PXP_WG_ROWS // 2) <= _PXP_F16_W3_MAX_MTILES:
        base = (
            -(-int(max_seqlen_q) // _PXP_FORMS[_PXP_VARIANT_F16_W3]["bm"])
            * int(hq)
            * int(batch)
        )
        ns_w3 = max(1, int(sm_count) // max(1, base))
        if (
            ns_w3 <= _PXP_F16_W3_MAX_SPLIT
            and -(-int(max_k_tiles) // ns_w3) >= _PXP_F16_W3_MIN_TILES_PER_CTA
        ):
            return _PXP_VARIANT_F16_W3, "causal"
    return _PXP_VARIANT_F16_SHORT, "fi"


@functools.lru_cache(maxsize=4096)
def plan_proxy_score_prefill(
    *,
    hq: int,
    batch: int,
    max_k_tiles: int,
    max_seqlen_q: int,
    use_fp32_acc: bool,
    num_sms: int,
) -> HopperProxyPrefillPlan:
    """Mirror of the Cake planner ``plan_proxy_score_prefill_sm90``: f32 accumulation -> the 256-row ``wg4st`` form
    with the causal planner; f16 accumulation -> ``f16_short_form`` (FI's 128-row shape with FI's split rule and the
    small-work lowering, or the 192-row form with the causal planner) up to ``_PXP_F16_SHORT_MAX_K_TILES`` pages,
    the 256-row ``f16db`` form with the causal planner beyond.  Four ring stages; ``ceil(max_seqlen_q / rows)`` row
    tiles; the causal planner models every sequence as ``max_seqlen_q`` tokens after ``max_k_tiles * 128 -
    max_seqlen_q`` prefix tokens."""

    if use_fp32_acc:
        variant, planner = _PXP_VARIANT_FP32, "causal"
    elif int(max_k_tiles) <= _PXP_F16_SHORT_MAX_K_TILES:
        variant, planner = f16_short_form(
            hq=int(hq),
            batch=int(batch),
            max_k_tiles=int(max_k_tiles),
            max_seqlen_q=int(max_seqlen_q),
            sm_count=int(num_sms),
        )
    else:
        variant, planner = _PXP_VARIANT_F16, "causal"
    form = _PXP_FORMS[variant]
    n_mtiles = -(-int(max_seqlen_q) // form["bm"])
    if planner == "causal":
        nsplit = select_proxy_prefill_nsplit(
            n_mtiles=n_mtiles,
            hq=int(hq),
            batch=int(batch),
            max_k_tiles=int(max_k_tiles),
            chunk=int(max_seqlen_q),
            ctx=int(max_k_tiles) * _BLOCK_SIZE,
            sm_count=int(num_sms),
            bm=form["bm"],
            slots_per_sm=form["slots"],
            f16=form["f16"],
        )
    else:
        nsplit = select_proxy_prefill_nsplit_fi(
            n_mtiles=n_mtiles,
            hq=int(hq),
            batch=int(batch),
            max_k_tiles=int(max_k_tiles),
            sm_count=int(num_sms),
        )
        if variant == _PXP_VARIANT_F16_SHORT:
            nsplit = f16_short_small_work_nsplit(
                nsplit,
                n_mtiles=n_mtiles,
                hq=int(hq),
                batch=int(batch),
                max_k_tiles=int(max_k_tiles),
                sm_count=int(num_sms),
            )
    nsplit = int(max(1, min(int(nsplit), max(1, int(max_k_tiles)))))
    return HopperProxyPrefillPlan(
        variant=variant,
        stages=_PXP_STAGES,
        n_mtiles=n_mtiles,
        nsplit=nsplit,
        lpt=form["lpt"],
    )


@functools.lru_cache(maxsize=4096)
def _proxy_prefill_template(
    *,
    hq: int,
    batch: int,
    max_k_tiles: int,
    max_seqlen_q: int,
    use_fp32_acc: bool,
    total_q: int,
    pt_stride: int,
    has_qoff: bool,
    device_index: int,
) -> tuple[_Program, tuple[int, int, int], dict[str, Any]]:
    """The launch of one prefill-regime proxy geometry resolved once: program, grid and kernel scalars."""

    plan = plan_proxy_score_prefill(
        hq=hq,
        batch=batch,
        max_k_tiles=max_k_tiles,
        max_seqlen_q=max_seqlen_q,
        use_fp32_acc=use_fp32_acc,
        num_sms=_num_sms(device_index),
    )
    constants: dict[str, Any] = dict(
        has_qoff=1 if has_qoff else 0,
        pt_stride=pt_stride,
        max_k_tiles=max_k_tiles,
        total_q=total_q,
        num_heads=hq,
        nsplit=plan.nsplit,
        trace=_trace_carrier(torch.device("cuda", device_index)),
    )
    return _program(plan.route), plan.grid(hq=hq, batch=batch), constants


def hopper_msa_proxy_score_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    *,
    page_table: torch.Tensor,
    seqused_k: torch.Tensor,
    per_head: torch.Tensor,
    max_seqlen_q: int,
    batch_size: int,
    q_offset: Optional[torch.Tensor] = None,
    use_fp32_acc: bool = True,
) -> torch.Tensor:
    """MSA proxy score, prefill regime, on compute capability 9.0.  Writes ``per_head``.

    ``use_fp32_acc`` selects the accumulation precision of the fp8 products: exact-product f32 (default, the
    original MSA numerics) or f16 (the SM90 CuTe DSL kernel's numerics).  Query ``i`` of sequence ``b`` sits at
    position ``q_offset[b] + i``; without ``q_offset`` the program derives ``seqused_k[b] - chunk_len`` in the
    kernel (no host allocation, graph-safe).
    """

    total_q, num_q_heads, head_dim = (int(x) for x in q.shape)
    if head_dim != _HEAD_DIM:
        raise ValueError(f"head_dim must be {_HEAD_DIM}, got {head_dim}")
    if k.ndim != 4 or k.shape[2] != _BLOCK_SIZE or k.shape[3] != _HEAD_DIM:
        raise ValueError(
            f"the MSA index cache must be (num_pages, num_kv_heads, {_BLOCK_SIZE}, {_HEAD_DIM})"
        )
    if q.dtype != torch.float8_e4m3fn or k.dtype != torch.float8_e4m3fn:
        raise NotImplementedError(
            "SM90 proxy-score prefill requires fp8 e4m3 q with an fp8 e4m3 index cache"
        )
    if not q.is_contiguous() or not k.is_contiguous():
        raise ValueError("q and the index cache must be contiguous")
    if not proxy_prefill_route_available(
        q_dtype=q.dtype, num_kv_heads=int(k.shape[1]), use_fp32_acc=use_fp32_acc
    ):
        raise NotImplementedError(
            f"no Cake SM90 proxy-score prefill program for Hkv {int(k.shape[1])}, use_fp32_acc={bool(use_fp32_acc)}"
        )
    batch = int(batch_size)
    if (
        cu_seqlens_q.dtype != torch.int32
        or not cu_seqlens_q.is_contiguous()
        or int(cu_seqlens_q.numel()) != batch + 1
    ):
        raise ValueError(
            f"cu_seqlens_q must be contiguous int32 with {batch + 1} entries"
        )
    if (
        page_table.dtype != torch.int32
        or page_table.ndim != 2
        or page_table.stride(1) != 1
    ):
        raise ValueError(
            "page_table must be int32 (batch_size, max_pages) with unit column stride"
        )
    if int(page_table.shape[0]) != batch:
        raise ValueError(
            f"page_table has {page_table.shape[0]} rows for batch_size {batch}"
        )
    if (
        seqused_k.dtype != torch.int32
        or not seqused_k.is_contiguous()
        or int(seqused_k.numel()) != batch
    ):
        raise ValueError(f"seqused_k must be contiguous int32 with {batch} entries")
    num_heads, max_k_tiles, out_q = (int(x) for x in per_head.shape)
    if (
        (num_heads, out_q) != (num_q_heads, total_q)
        or per_head.dtype != torch.float32
        or not per_head.is_contiguous()
    ):
        raise ValueError(
            f"per_head must be contiguous float32 ({num_q_heads}, max_k_tiles, {total_q})"
        )
    if int(page_table.shape[1]) < max_k_tiles:
        raise ValueError("page_table is narrower than max_k_tiles")
    if q_offset is not None and (
        q_offset.dtype != torch.int32
        or not q_offset.is_contiguous()
        or int(q_offset.numel()) != batch
    ):
        raise ValueError(f"q_offset must be contiguous int32 with {batch} entries")
    if int(max_seqlen_q) <= 0:
        raise ValueError(f"max_seqlen_q must be positive, got {max_seqlen_q}")
    num_pages = int(k.shape[0])
    program, grid, constants = _proxy_prefill_template(
        hq=num_q_heads,
        batch=batch,
        max_k_tiles=max_k_tiles,
        max_seqlen_q=int(max_seqlen_q),
        use_fp32_acc=bool(use_fp32_acc),
        total_q=total_q,
        pt_stride=int(page_table.stride(0)),
        has_qoff=q_offset is not None,
        device_index=_device_index(q.device),
    )
    program.launch(
        grid,
        constants,
        Q=q.view(torch.uint8),
        K=k.view(torch.uint8).view(
            num_pages * int(k.shape[1]) * _BLOCK_SIZE, _HEAD_DIM
        ),
        out=per_head,
        cu_seqlens_q=cu_seqlens_q,
        page_table=page_table,
        seqused_k=seqused_k,
        q_offset=q_offset if q_offset is not None else seqused_k,
    )
    return per_head


# ---------------------------------------------------------------------------
# Top-k select
# ---------------------------------------------------------------------------

# Top-k program geometry and planner constants, mirrored from the Cake source module
# (``minimax_sparse_attention_topk_select_sm90``).  The planner picks (columns per CTA, tile groups per column) from
# a latency / issue / bandwidth model; constants are hardware properties (H100 SXM) or instruction counts of the
# traced kernel, none is a workload threshold.
_TK_CHUNK = 16  # tiles per chunk of one thread
_TK_MAX_INDEX_BITS = 14
_TK_MAX_TILES = 1 << _TK_MAX_INDEX_BITS
_TK_CLK_HZ = 1.7e9
_TK_LANES_PER_CYCLE = 128
_TK_INSTR_CHUNK = 330
_TK_INSTR_MERGE = 130
_TK_INSTR_EPI = 350
_TK_LAT_DRAM_US = 0.7
_TK_T_CHUNK_US = 0.30
_TK_T_SHFL_US = 0.08
_TK_T_SMEM_US = 0.15
_TK_T_EPI_US = 0.20
_TK_BW_DRAM = 2.3e12
_TK_BW_L2 = 5.5e12
_TK_MAX_THREADS_PER_SM = 2048
_TK_C_CHOICES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
# Stage-2 structure window of the module (measured, not derived): the un-prefetched blocked 32-tile stream.
_TK_T32_GEOMETRIES = frozenset({(16, 16)})
_TK_T32_THREADS = 256
_TK_T32_MIN_ROWS = 17
_TK_T32_MAX_ROWS = 32
# Single-register-tile stream window of the module (``ONE_GEOMETRIES`` / ``one_default``, top-k round 4): the geometries
# whose thread holds exactly one register tile stream it without the prefetch buffer / loop.  Measured, not derived.
_TK_ONE_GEOMETRIES = frozenset({(8, 4), (2, 32), (8, 32), (16, 16), (2, 128)})
_TK_INSTR_TILE32 = 620
_TK_T_TILE32_US = 0.56
_TK_W_CHOICES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


def topk_index_bits(tiles: int) -> int:
    """Low key bits that carry the tile index (at least one)."""
    return max(1, int(tiles - 1).bit_length())


@dataclass(frozen=True)
class HopperTopkPlan:
    """Physical top-k variant: columns per CTA, tile groups per column, index bits and the mask form."""

    c: int
    w: int
    l: int
    masked: bool
    nvp: bool
    # Stream structure of the module's stage 2: the un-prefetched blocked 32-tile form (one 32-tile per thread) that
    # the planner selects for the (16, 16) geometry when a full wave of its 8-warp CTAs holds 17..32 tile rows per
    # thread.  Not part of the route key (the geometry implies it); it sets the ``num_chunks`` kernel scalar.
    t32: bool = False

    @property
    def lat(self) -> bool:
        return self.masked and self.c >= 2

    def chunks(self, tiles: int) -> int:
        """Register tiles per thread (module ``chunks_for``): blocked 32-tiles ``ceil(ceil(tiles / W) / 32)`` for the
        T32 stream, interleaved 16-chunks ``ceil(tiles / (16 W))`` otherwise."""
        if self.t32:
            rows = -(-int(tiles) // self.w)
            return -(-rows // 32)
        return -(-int(tiles) // (_TK_CHUNK * self.w))

    def one(self, tiles: int) -> bool:
        """Module ``one_default``: the single-register-tile stream (no prefetch buffer / loop, the sorted tile is the
        running list) for the geometries in ``_TK_ONE_GEOMETRIES`` when the thread holds exactly one register tile and
        the host FILTER knob is off (always, in production).  Not part of the route key and not a launch argument:
        the exported program is built with the same rule (the public ``topk_module`` resolves it), so this mirror
        states the structure the binary behind the route carries."""
        return (self.c, self.w) in _TK_ONE_GEOMETRIES and self.chunks(tiles) == 1

    @property
    def ilv(self) -> bool:
        """Interleaved tile walk: a measured-off A/B knob of the module (round 4); False for every planner shape."""
        return False

    @property
    def route(self) -> str:
        key = f"topk_select:c{self.c}:w{self.w}:l{self.l}:{'masked' if self.masked else 'plain'}"
        if self.nvp:
            key += ":nvp"
        if self.lat:
            key += ":lat"
        if self.t32:
            key += ":t32"
        return key + ":main"


@functools.lru_cache(maxsize=4096)
def plan_topk_select(
    *, num_heads: int, tiles: int, total_q: int, num_sms: int, masked: bool, nvp: bool
) -> HopperTopkPlan:
    """Mirror of the Cake planner ``plan_topk_select_sm90`` (stage 1 geometry, stage 2 structure).

    Stage 1: ``cost = max(chain x waves, issue, dram, l2) + 0.004 max(0, ctas - 128)`` over the admissible (C, W);
    ties go to the larger C below C 32; among the full-line configurations (C >= 32) first to the grid that keeps
    more SMs busy (``min(ctas, SMs)``), then the larger C, then MORE threads; below C 32 fewer threads.
    Stage 2: the geometries in ``_TK_T32_GEOMETRIES`` with ``_TK_T32_THREADS`` threads, a full wave of CTAs
    (``ctas >= SMs``) and ``_TK_T32_MIN_ROWS <= ceil(tiles / W) <= _TK_T32_MAX_ROWS`` tile rows per thread take the
    un-prefetched blocked 32-tile stream and override stage 1 (lowest T32 cost, then larger C)."""

    if tiles < 1 or tiles > _TK_MAX_TILES:
        raise ValueError(f"tiles must be 1..{_TK_MAX_TILES}, got {tiles}")
    L = topk_index_bits(tiles)
    best = None
    best32 = None
    for Cc in _TK_C_CHOICES:
        for Ww in _TK_W_CHOICES:
            threads = Cc * Ww
            if threads < 32 or threads > 512:
                continue
            chunks = -(-tiles // (_TK_CHUNK * Ww))
            if chunks < 1:
                continue
            CW = min(Cc, 32)
            WPW = 32 // CW
            if Ww % WPW != 0 and Ww < WPW:
                continue
            nshfl = (min(Ww, WPW)).bit_length() - 1
            nsmem = max(0, (Ww // WPW).bit_length() - 1) if Ww >= WPW else 0
            ctas = -(-total_q // Cc) * num_heads
            resident = num_sms * max(1, _TK_MAX_THREADS_PER_SM // threads)
            waves = -(-ctas // resident)
            chain = (
                _TK_LAT_DRAM_US
                + chunks * _TK_T_CHUNK_US
                + nshfl * _TK_T_SHFL_US
                + nsmem * _TK_T_SMEM_US
                + _TK_T_EPI_US
            )
            lane_instr = (
                ctas
                * threads
                * (chunks * _TK_INSTR_CHUNK + (nshfl + nsmem) * _TK_INSTR_MERGE)
                + ctas * Cc * _TK_INSTR_EPI
            )
            issue = lane_instr / (num_sms * _TK_LANES_PER_CYCLE * _TK_CLK_HZ) * 1e6
            data = float(num_heads) * tiles * total_q * 4
            active = min(ctas, resident)
            bw_sm = _TK_BW_DRAM * min(1.0, min(ctas, num_sms) / num_sms)
            bw_little = active * threads * _TK_CHUNK * 4 * 2 / (_TK_LAT_DRAM_US * 1e-6)
            dram = data / min(bw_sm, bw_little) * 1e6
            l2 = (
                data
                * max(1.0, 8.0 / Cc)
                / (_TK_BW_L2 * min(1.0, min(ctas, num_sms) / num_sms))
                * 1e6
            )
            cost = max(chain * waves, issue, dram, l2) + 0.004 * max(0, ctas - 128)
            cand = (
                cost,
                (-Cc if Cc < 32 else -32, -min(ctas, num_sms) if Cc >= 32 else 0, -Cc),
                -threads if Cc >= 32 else threads,
                Cc,
                Ww,
            )
            if best is None or cand < best:
                best = cand
            if (
                (Cc, Ww) in _TK_T32_GEOMETRIES
                and threads == _TK_T32_THREADS
                and ctas >= num_sms
                and _TK_T32_MIN_ROWS <= -(-tiles // Ww) <= _TK_T32_MAX_ROWS
            ):
                rows = -(-tiles // Ww)
                rows32 = -(-rows // 32)
                chain32 = (
                    _TK_LAT_DRAM_US
                    + rows32 * _TK_T_TILE32_US
                    + nshfl * _TK_T_SHFL_US
                    + nsmem * _TK_T_SMEM_US
                    + _TK_T_EPI_US
                )
                lane32 = (
                    ctas
                    * threads
                    * (rows32 * _TK_INSTR_TILE32 + (nshfl + nsmem) * _TK_INSTR_MERGE)
                    + ctas * Cc * _TK_INSTR_EPI
                )
                issue32 = lane32 / (num_sms * _TK_LANES_PER_CYCLE * _TK_CLK_HZ) * 1e6
                cost32 = max(chain32 * waves, issue32, dram, l2) + 0.004 * max(
                    0, ctas - 128
                )
                cand32 = (cost32, -Cc, Cc, Ww)
                if best32 is None or cand32 < best32:
                    best32 = cand32
    if best is None:
        raise ValueError("no admissible top-k variant")
    if best32 is not None:
        _cost, _negc, Cc, Ww = best32
        t32 = True
    else:
        _cost, _tie_c, _tie_t, Cc, Ww = best
        t32 = False
    return HopperTopkPlan(
        c=Cc, w=Ww, l=L, masked=bool(masked) or bool(nvp), nvp=bool(nvp), t32=t32
    )


@functools.lru_cache(maxsize=4096)
def _topk_resolution(
    *, num_heads: int, tiles: int, total_q: int, num_sms: int, masked: bool, nvp: bool
) -> tuple[HopperTopkPlan, Optional[str]]:
    """``(plan, program name or None)`` of one top-k geometry: the planner's coordinate and whether a Cake program
    was delivered for its route -- resolved once per geometry for the route check and the launcher alike."""

    plan = plan_topk_select(
        num_heads=num_heads,
        tiles=tiles,
        total_q=total_q,
        num_sms=num_sms,
        masked=masked,
        nvp=nvp,
    )
    return plan, ROUTES.get(plan.route)


@functools.lru_cache(maxsize=4096)
def _topk_template(
    *,
    num_heads: int,
    tiles: int,
    total_q: int,
    num_sms: int,
    masked: bool,
    nvp: bool,
    device_index: int,
) -> tuple[_Program, tuple[int, int, int], dict[str, Any]]:
    """The launch of one top-k geometry resolved once: program, grid and kernel scalars (the never-read int32 filler
    of the form without per-token valid pages included)."""

    plan, name = _topk_resolution(
        num_heads=num_heads,
        tiles=tiles,
        total_q=total_q,
        num_sms=num_sms,
        masked=masked,
        nvp=nvp,
    )
    if name is None:
        raise NotImplementedError(
            f"no Cake SM90 top-k program for the planner coordinate {plan.route!r} (Hq {num_heads}, tiles {tiles}, total_q {total_q})"
        )
    constants: dict[str, Any] = dict(
        tiles=tiles,
        total_q=total_q,
        num_heads=num_heads,
        num_chunks=plan.chunks(tiles),
    )
    if not nvp:
        constants["nvp"] = _int32_dummy(torch.device("cuda", device_index))
    return _program(plan.route), (-(-total_q // plan.c), num_heads, 1), constants


def topk_route_available(
    *, num_heads: int, tiles: int, total_q: int, num_sms: int, masked: bool, nvp: bool
) -> bool:
    """Whether a Cake top-k program serves these host-known coordinates: the planner's (C, W, index bits, mask form)
    for this geometry is one of the delivered programs (``_sm90_dispatch`` keeps the CuTe DSL top-k otherwise)."""

    if tiles < 1 or tiles > _TK_MAX_TILES or total_q < 1 or num_heads < 1:
        return False
    return (
        _topk_resolution(
            num_heads=int(num_heads),
            tiles=int(tiles),
            total_q=int(total_q),
            num_sms=int(num_sms),
            masked=bool(masked),
            nvp=bool(nvp),
        )[1]
        is not None
    )


def hopper_msa_topk_select(
    max_score: torch.Tensor,
    topk: int,
    output: torch.Tensor,
    *,
    num_valid_pages: Optional[torch.Tensor] = None,
    force_begin_blocks: int = 0,
    force_end_blocks: int = 0,
) -> torch.Tensor:
    """MSA top-k block selection on compute capability 9.0.  Writes the ascending, ``-1``-padded indices into
    ``output`` ``(total_q, num_qo_heads, 16)`` int32 and returns it.

    ``num_valid_pages`` is the per-token int32 tensor of the surface (``None`` = no clamping beyond the ``-inf``
    tiles the scores already carry); forced sink / window blocks are applied at the key.  No allocation, no host
    synchronisation (graph-capturable).
    """

    if topk != _TOPK:
        raise NotImplementedError(
            f"SM90 msa_topk_select supports topk={_TOPK} only, got {topk}"
        )
    if (
        max_score.ndim != 3
        or max_score.dtype != torch.float32
        or not max_score.is_contiguous()
    ):
        raise ValueError(
            "max_score must be a contiguous fp32 (num_qo_heads, max_k_tiles, total_q) tensor"
        )
    hq, tiles, total_q = (int(x) for x in max_score.shape)
    if (
        tuple(output.shape) != (total_q, hq, _TOPK)
        or output.dtype != torch.int32
        or not output.is_contiguous()
    ):
        raise ValueError(
            f"output must be a contiguous int32 ({total_q}, {hq}, {_TOPK}) tensor, got {tuple(output.shape)}"
        )
    fb, fe = int(force_begin_blocks), int(force_end_blocks)
    if fb < 0 or fe < 0 or fb + fe > _TOPK:
        raise ValueError(
            f"force_begin_blocks + force_end_blocks must be 0..{_TOPK}, got {fb} + {fe}"
        )
    nvp = None
    if num_valid_pages is not None:
        if (
            num_valid_pages.dtype != torch.int32
            or not num_valid_pages.is_contiguous()
            or int(num_valid_pages.numel()) != total_q
        ):
            raise ValueError(
                "num_valid_pages must be a contiguous int32 (total_q,) tensor"
            )
        nvp = num_valid_pages
    device_index = _device_index(max_score.device)
    program, grid, constants = _topk_template(
        num_heads=hq,
        tiles=tiles,
        total_q=total_q,
        num_sms=_num_sms(device_index),
        masked=fb > 0 or fe > 0,
        nvp=nvp is not None,
        device_index=device_index,
    )
    if nvp is not None:
        program.launch(
            grid,
            constants,
            S=max_score.view(torch.uint32),
            nvp=nvp,
            out=output,
            fb=fb,
            fe=fe,
        )
    else:
        program.launch(
            grid, constants, S=max_score.view(torch.uint32), out=output, fb=fb, fe=fe
        )
    return output


# ---------------------------------------------------------------------------
# Sparse prefill (v1 / v2 kernels by pages per sequence)
# ---------------------------------------------------------------------------

# Mirror of the Cake host dispatch ``plan_prefill_sm90`` / ``dispatch_prefill_impl``: pages per sequence up to which
# the v2 kernel (FA3 orientation, 128 rows = 128 / G tokens per union walk) is dispatched, per GQA group; the v1
# kernel (one warpgroup per CTA, 32 rows = 32 / G tokens per union walk) beyond.  The delivered programs are the
# production variants of both kernels (v1: NR 32, two ring stages, split K widening; v2: two consumer warpgroups,
# four stages, two operand buffers, Q in registers, pingpong) for the groups the export delivered.
_PF_V2_PAGES_MAX = {4: 128, 8: 96, 16: 64}
_PF_MAX_BLOCKS = 4096  # union byte map: blocks per sequence the kernels can union (page_table width bound)
_PF_V1_NR = 32
_PF_V2_NCONS = 2


@dataclass(frozen=True)
class HopperPrefillPlan:
    """Physical sparse-prefill variant: the kernel and the tokens per union walk of one launch."""

    impl: str
    group: int

    @property
    def rows(self) -> int:
        return _PF_V1_NR if self.impl == "v1" else 64 * _PF_V2_NCONS

    @property
    def tokens_per_group(self) -> int:
        return self.rows // self.group

    @property
    def route(self) -> str:
        if self.impl == "v1":
            return f"prefill_v1:g{self.group}:nr{_PF_V1_NR}:st2:none:ksplit:main"
        return f"prefill_v2:g{self.group}:nc{_PF_V2_NCONS}:st4:fb2:none:qrs:pp:pvwait:role:amma:main"


@functools.lru_cache(maxsize=4096)
def plan_sparse_prefill(
    *, num_q_heads: int, num_kv_heads: int, max_pages: int
) -> HopperPrefillPlan:
    """The kernel of one launch: v2 up to ``_PF_V2_PAGES_MAX[G]`` pages per sequence, v1 otherwise."""

    group = int(num_q_heads) // int(num_kv_heads)
    if group * int(num_kv_heads) != int(num_q_heads) or group not in (1, 2, 4, 8, 16):
        raise NotImplementedError(
            f"SM90 sparse prefill serves GQA groups 1, 2, 4, 8 and 16, got {num_q_heads} / {num_kv_heads}"
        )
    limit = _PF_V2_PAGES_MAX.get(group, 0)
    impl = "v2" if limit and int(max_pages) <= limit else "v1"
    return HopperPrefillPlan(impl=impl, group=group)


@functools.lru_cache(maxsize=4096)
def prefill_route_available(
    *, num_q_heads: int, num_kv_heads: int, max_pages: int
) -> bool:
    """Whether a Cake sparse-prefill program serves these host-known coordinates: a power-of-two GQA group whose
    v1 / v2 programs were delivered and at most 4096 pages per sequence (``_sm90_dispatch`` keeps the CuTe DSL
    kernels otherwise)."""

    group = int(num_q_heads) // max(1, int(num_kv_heads))
    if group * int(num_kv_heads) != int(num_q_heads) or group not in (1, 2, 4, 8, 16):
        return False
    if int(max_pages) < 1 or int(max_pages) > _PF_MAX_BLOCKS:
        return False
    return _route_available(
        plan_sparse_prefill(
            num_q_heads=num_q_heads, num_kv_heads=num_kv_heads, max_pages=max_pages
        ).route
    )


@functools.lru_cache(maxsize=4096)
def _prefill_template(
    *,
    num_q_heads: int,
    num_kv_heads: int,
    max_pages: int,
    total_q: int,
    batch: int,
    use_q_offset: bool,
    device_index: int,
) -> tuple[_Program, tuple[int, int, int], dict[str, Any]]:
    """The launch of one sparse-prefill geometry resolved once: program, grid and kernel scalars."""

    plan = plan_sparse_prefill(
        num_q_heads=num_q_heads, num_kv_heads=num_kv_heads, max_pages=max_pages
    )
    if not _route_available(plan.route):
        raise NotImplementedError(
            f"no Cake SM90 sparse-prefill program for GQA group {plan.group} ({plan.route!r})"
        )
    # >= sum_b ceil(qlen_b / tokens per group); the extra CTAs find no group and idle.
    groups = max(1, -(-total_q // plan.tokens_per_group) + batch - 1)
    constants: dict[str, Any] = dict(
        total_q=total_q,
        batch=batch,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        max_pages=max_pages,
        use_q_offset=1 if use_q_offset else 0,
        zero_u32=0,
        trace=_trace_carrier(torch.device("cuda", device_index)),
    )
    return _program(plan.route), (groups, num_kv_heads, 1), constants


def hopper_msa_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    *,
    page_table: torch.Tensor,
    seqused_k: torch.Tensor,
    out: torch.Tensor,
    q_offset: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Sparse prefill attention on compute capability 9.0.  Writes and returns ``out``.

    ``k`` and ``v`` are the ``(num_pages, num_kv_heads, 128, 128)`` fp8 e4m3 halves of the interleaved dense
    ``(num_pages, num_kv_heads, 128, 256)`` cache (checked here and by the caller's ``_as_packed_kv``).  Query ``i``
    of sequence ``b`` attends the keys of its selected blocks at positions ``<= q_offset[b] + i`` (``q_offset``
    defaults to ``seqused_k[b] - qlen_b``, derived in the kernel); ``softmax_scale`` is a kernel parameter.  No
    scratch, no allocation: graph-capturable as is.
    """

    total_q, num_q_heads, head_dim = (int(x) for x in q.shape)
    if head_dim != _HEAD_DIM:
        raise ValueError(f"head_dim must be {_HEAD_DIM}, got {head_dim}")
    if q.dtype != torch.bfloat16 or not q.is_contiguous():
        raise ValueError("SM90 sparse prefill needs contiguous bf16 q")
    if k.dtype != torch.float8_e4m3fn or v.dtype != torch.float8_e4m3fn:
        raise ValueError("SM90 sparse prefill needs an fp8 e4m3 KV cache")
    if (
        k.ndim != 4
        or k.shape[2] != _BLOCK_SIZE
        or k.shape[3] != _HEAD_DIM
        or k.shape != v.shape
        or k.stride() != v.stride()
    ):
        raise ValueError(
            f"paged k/v must be matching (num_pages, num_kv_heads, {_BLOCK_SIZE}, {_HEAD_DIM}) views"
        )
    num_kv_heads = int(k.shape[1])
    if (
        k.stride(-1) != 1
        or k.stride(-2) != 2 * _HEAD_DIM
        or k.stride(1) != _BLOCK_SIZE * 2 * _HEAD_DIM
        or k.stride(0) != num_kv_heads * _BLOCK_SIZE * 2 * _HEAD_DIM
        or v.data_ptr() - k.data_ptr() != _HEAD_DIM * k.element_size()
    ):
        raise NotImplementedError(
            "SM90 sparse prefill needs K and V as the halves of one dense interleaved "
            f"(num_pages, num_kv_heads, {_BLOCK_SIZE}, {2 * _HEAD_DIM}) cache"
        )
    if q2k_indices.dtype != torch.int32 or not q2k_indices.is_contiguous():
        raise ValueError("q2k_indices must be contiguous int32")
    if tuple(q2k_indices.shape) != (num_kv_heads, total_q, _TOPK):
        raise NotImplementedError(
            f"SM90 sparse prefill serves q2k_indices of shape (num_kv_heads, total_q, {_TOPK}); "
            f"got {tuple(q2k_indices.shape)}"
        )
    if (
        cu_seqlens_q.dtype != torch.int32
        or not cu_seqlens_q.is_contiguous()
        or cu_seqlens_q.ndim != 1
        or int(cu_seqlens_q.numel()) < 2
    ):
        raise ValueError("cu_seqlens_q must be contiguous int32 (batch_size + 1,)")
    batch = int(cu_seqlens_q.numel()) - 1
    if (
        page_table.dtype != torch.int32
        or not page_table.is_contiguous()
        or page_table.ndim != 2
        or int(page_table.shape[0]) != batch
    ):
        raise ValueError(
            f"page_table must be contiguous int32 (batch_size={batch}, max_pages)"
        )
    if (
        seqused_k.dtype != torch.int32
        or not seqused_k.is_contiguous()
        or int(seqused_k.numel()) != batch
    ):
        raise ValueError(f"seqused_k must be contiguous int32 with {batch} entries")
    if q_offset is not None and (
        q_offset.dtype != torch.int32
        or not q_offset.is_contiguous()
        or int(q_offset.numel()) != batch
    ):
        raise ValueError(f"q_offset must be contiguous int32 with {batch} entries")
    if out.shape != q.shape or out.dtype != torch.bfloat16 or not out.is_contiguous():
        raise ValueError("out must be a contiguous bf16 tensor shaped like q")
    max_pages = int(page_table.shape[1])
    if max_pages > _PF_MAX_BLOCKS:
        raise NotImplementedError(
            f"SM90 sparse prefill unions at most {_PF_MAX_BLOCKS} blocks per sequence, got {max_pages}"
        )
    program, grid, constants = _prefill_template(
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        max_pages=max_pages,
        total_q=total_q,
        batch=batch,
        use_q_offset=q_offset is not None,
        device_index=_device_index(q.device),
    )
    scale = (
        1.0 / math.sqrt(_HEAD_DIM) if softmax_scale is None else float(softmax_scale)
    )
    program.launch(
        grid,
        constants,
        Q32=q.view(torch.uint32),
        K=k.view(torch.uint8),
        V=v.view(torch.uint8),
        O=out,
        q2k_indices=q2k_indices,
        cu_seqlens_q=cu_seqlens_q,
        page_table=page_table,
        seqused_k=seqused_k,
        q_offset=q_offset if q_offset is not None else seqused_k,
        softmax_scale_log2=scale * _LOG2E,
    )
    return out


__all__ = [
    "PROGRAM_KINDS",
    "HopperDecodePlan",
    "HopperPrefillPlan",
    "HopperProxyPrefillPlan",
    "HopperTopkPlan",
    "hopper_msa_proxy_score_decode",
    "hopper_msa_proxy_score_prefill",
    "hopper_msa_sparse_attention",
    "hopper_msa_sparse_decode_attention",
    "hopper_msa_topk_select",
    "is_hopper_msa_device",
    "plan_proxy_score",
    "plan_proxy_score_prefill",
    "plan_sparse_decode",
    "plan_sparse_prefill",
    "plan_topk_select",
    "preload_programs",
    "prefill_route_available",
    "proxy_decode_route_available",
    "proxy_evict_first",
    "proxy_prefill_routes",
    "proxy_prefill_route_available",
    "proxy_route",
    "f16_short_form",
    "f16_short_small_work_nsplit",
    "select_proxy_prefill_nsplit",
    "select_proxy_prefill_nsplit_fi",
    "topk_index_bits",
    "topk_route_available",
]
