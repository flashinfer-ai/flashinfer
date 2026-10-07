# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SM120 KDA prefill: the stable internal entry point, and the variant choice.

This is a facade with no compilation side effects.  Importing it -- on a CPU,
on an SM100 box, or with the optional CuTe DSL stack missing -- loads no device
code and builds nothing.  The device modules are imported only once a call has
been validated and a variant chosen, which is what makes
``import flashinfer`` free for a caller that never runs KDA prefill on SM120.

Three names leave this package::

    can_implement_kda_prefill_sm120   fail-closed predicate, no side effects
    run_kda_prefill_sm120             the call
    clear_kda_prefill_sm120_caches    drop everything both variants hold

``flashinfer/kda_prefill.py`` reaches them through
:mod:`flashinfer.kda_kernels`, never by importing this package directly, and
nothing outside FlashInfer is expected to import it at all: the public entry
point is :func:`flashinfer.recurrent_kda`.  There is no ``sm120`` in any public
name -- the architecture appears in module paths, in the guard, and in the
compile cache key, and nowhere a caller has to type.

Two variants implement the same contract:

``decomp``
    a chunk-parallel prepare and a serial recurrence, issued through one
    compiled host entry.

``fused``
    one kernel that does both.

They agree numerically -- on the 67-shape suite they match to the last bit on
20 of 24 tail cells -- so the choice between them is a performance one, and
:func:`choose_variant` makes it from a measured table rather than a formula.
"""

from __future__ import annotations

import functools
import threading
import weakref
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, NamedTuple, Optional

import torch

from .runtime import (
    DK,
    DV,
    LOWER_BOUND_RANGE,
    SM120_CAPABILITY,
    KDAPrefillValidationError,
    SM120PrefillResources,
    UnsupportedArchitectureError,
    canonical_offsets,
    current_stream_ptr,
    resource_cache_token,
    sm120a_available,
    tensor_identity,
    tensor_layout_identity,
    validate_inputs,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .runtime import CanonicalInputs

__all__ = [
    "can_implement_kda_prefill_sm120",
    "clear_kda_prefill_sm120_caches",
    "run_kda_prefill_sm120",
]

#: The variants :func:`run_kda_prefill_sm120` can dispatch to.  ``"auto"``
#: resolves to one of the other two before anything is launched.
VARIANTS = ("decomp", "fused", "auto")

DEFAULT_VARIANT = "auto"


# ---------------------------------------------------------------------------
# Variant choice.
#
# Deliberately a table rather than a formula.  The obvious formula -- scale the
# CTA term by SM count -- is an untested assumption: it presumes the crossover
# is set purely by how many waves the grid takes, when the measured data also
# turns on H and T, which have nothing to do with SM count.
#
# **Keyed on SM count, not on the device name.**  A name is not a stable unique
# selector and can collide across devices.  SM count is also the better key on
# the merits -- the CTA term is a statement about a grid against the machine's
# width.
# ---------------------------------------------------------------------------


class AutoProfile(NamedTuple):
    """Thresholds for one device.  See :data:`AUTO_PROFILES`."""

    #: Heads at or above which the fused variant wins regardless of T, or
    #: ``None`` where the measured data needs no such term.  ``None`` means
    #: "not used", where a large int would read as a threshold someone
    #: measured.
    heads: Optional[int]
    #: Per-sequence length at or below which it wins regardless of H.
    tokens: Optional[int]
    #: Decomp CTA count (``2 * batch * heads``) at or above which it wins.
    ctas: Optional[int]
    #: Human name for the device, for reporting.  Not used for matching.
    device: str
    #: Where the numbers came from, printed by :func:`describe_variant_policy`.
    source: str


AUTO_PROFILES: dict[int, AutoProfile] = {
    # Each row is fitted independently. The 156-SM and 188-SM sweeps both place
    # the crossover at T <= 32 or CTA >= 144; keeping two rows records that both
    # devices were measured rather than treating one as an inferred fallback.
    156: AutoProfile(
        heads=None,
        tokens=32,
        ctas=144,
        device="156-SM SM120 part",
        source="147-shape fit, 156 SMs",
    ),
    188: AutoProfile(
        heads=None,
        tokens=32,
        ctas=144,
        device="188-SM SM120 part",
        source="147-shape fit, 188 SMs",
    ),
    # The 110-SM fit crosses one measured CTA step earlier.
    110: AutoProfile(
        heads=None,
        tokens=32,
        ctas=128,
        device="110-SM SM120 part",
        source="67-shape sweep + FlashInfer's twelve, 74 shapes, 110 SMs",
    ),
}

#: Used for any CC 12.0 device without a row above.
#:
#: A fallback, not a claim.  These are the only thresholds anyone has measured,
#: so they are better than a guess and worse than a measurement;
#: :func:`describe_variant_policy` says which case a given machine is in, so a
#: number taken on an unprofiled card cannot quietly be read as tuned.
FALLBACK_AUTO_PROFILE = 156


def _device_index(device) -> Optional[int]:
    """``device``'s index, the current device's, or ``None`` without a driver.

    Resolved here rather than inside :func:`_sm_count`, which is keyed on what
    it is handed: caching "whichever device happened to be current the first
    time" under a ``None`` key would keep answering for that one after a
    ``set_device``.
    """
    if isinstance(device, int):
        return device
    if device is not None:
        index = torch.device(device).index
        if index is not None:
            return index
    try:
        return torch.cuda.current_device()
    except Exception:  # noqa: BLE001 -- an absent driver is not an error here
        return None


@functools.lru_cache(maxsize=8)
def _sm_count(index: Optional[int]) -> int:
    """That device's SM count, asked once per device.

    ``get_device_properties`` is a driver query, and a device's SM count cannot
    change under a live process -- so asking per call is measurable overhead on
    a path that runs before every launch.  ``-1`` on a host with no driver,
    which is what keeps the host-only selector tests working.

    The index is the input's device, not device 0.  A host can hold two CC 12.0
    parts with different SM counts, and the rows of the table above disagree
    between them -- reading device 0 would apply the 110-SM thresholds to every
    call on a 188-SM card, and nothing in the output would say so.
    """
    if index is None:
        return -1
    try:
        return torch.cuda.get_device_properties(index).multi_processor_count
    except Exception:  # noqa: BLE001 -- an absent driver is not an error here
        return -1


def auto_profile(
    sm_count: Optional[int] = None, device=None
) -> tuple[int, AutoProfile]:
    """``(profile_key, profile)`` for a device, falling back where unmeasured.

    ``sm_count`` defaults to ``device``'s and ``device`` to the current one; on
    a host without either the fallback is returned, which is what keeps the
    host-only tests from needing a driver.
    """
    if sm_count is None:
        sm_count = _sm_count(_device_index(device))
    if sm_count in AUTO_PROFILES:
        return sm_count, AUTO_PROFILES[sm_count]
    return FALLBACK_AUTO_PROFILE, AUTO_PROFILES[FALLBACK_AUTO_PROFILE]


def describe_variant_policy(sm_count: Optional[int] = None, device=None) -> str:
    """One line naming the thresholds in force and whether this device set them.

    Print it in any report that quotes a time from ``auto``: a number taken
    under fallback thresholds is not a number taken under tuned ones, and
    nothing else in the output distinguishes them.
    """
    name = "<no CUDA device>"
    if sm_count is None:
        index = _device_index(device)
        try:
            name = torch.cuda.get_device_properties(index).name
            sm_count = _sm_count(index)
        except Exception:  # noqa: BLE001
            sm_count = -1
    else:
        name = "<given>"
    key, profile = auto_profile(sm_count)
    tuned = key == sm_count
    device_name = f"{name}, {sm_count} SMs" if sm_count > 0 else name
    terms = []
    if profile.heads is not None:
        terms.append(f"H>={profile.heads}")
    if profile.tokens is not None:
        terms.append(f"T<={profile.tokens}")
    if profile.ctas is not None:
        terms.append(f"CTA>={profile.ctas}")
    provenance = (
        "measured on this device"
        if tuned
        else f"FALLBACK from the {key}-SM {profile.device}"
    )
    return (
        f"variant=auto on {device_name!r}: {' or '.join(terms)} -> fused  "
        f"[{provenance}; {profile.source}]"
    )


def choose_variant(
    batch: int,
    heads: int,
    tokens: int,
    sm_count: Optional[int] = None,
    device=None,
) -> str:
    """Which variant the measured table says is faster for this shape.

    ``tokens`` is the per-sequence length, not the packed total: the table was
    measured on equal-length sequences and the CTA count is what varies with
    ``batch``.  For a ragged batch the caller passes the *longest* sequence --
    the recurrence is serial within a sequence, so the longest one sets the
    critical path, and a batch containing a 130 behaves like a 130 rather than
    like the 27 its lengths average to.

    Returns ``"decomp"`` or ``"fused"``, never ``"auto"``.
    """
    _, profile = auto_profile(sm_count, device)
    # The recurrence issues one CTA per (sequence, DV half), so the decomp grid
    # is twice batch*heads.  The fused kernel issues one per (sequence, head);
    # both grow the same way, and the threshold is expressed in the decomp
    # units the sweep tabulated.
    ctas = 2 * batch * heads
    # A ``None`` threshold is a condition this device's data did not need, not
    # a condition that is always true.
    if profile.heads is not None and heads >= profile.heads:
        return "fused"
    if profile.tokens is not None and tokens <= profile.tokens:
        return "fused"
    if profile.ctas is not None and ctas >= profile.ctas:
        return "fused"
    return "decomp"


# ---------------------------------------------------------------------------
# Lazy variant import.
#
# Nothing above this line touches the CuTe DSL.  The import below is the first
# thing that does, and it happens only after a call has been validated and a
# variant chosen -- so a process that imports FlashInfer and never runs SM120
# KDA prefill never pays for it, and a CPU-only import cannot fail here.
# ---------------------------------------------------------------------------

_MODULES: dict[str, Any] = {}
_MODULES_LOCK = threading.RLock()


def _variant_module(name: str):
    module = _MODULES.get(name)
    if module is not None:
        return module
    with _MODULES_LOCK:
        module = _MODULES.get(name)
        if module is None:
            if name == "decomp":
                from . import decomp  # noqa: PLC0415

                module = decomp
            elif name == "fused":
                from . import fused  # noqa: PLC0415

                module = fused
            else:
                raise ValueError(
                    f"unknown variant {name!r}; expected one of {VARIANTS}"
                )
            _MODULES[name] = module
    return module


# ---------------------------------------------------------------------------
# The backend ABI.
# ---------------------------------------------------------------------------


def can_implement_kda_prefill_sm120(**kwargs) -> bool:
    """Can this backend run this call?  Fail-closed, and free of side effects.

    Allocates nothing, launches nothing, compiles nothing and does not
    synchronize, so the public dispatcher can call it on every request.  The
    structural argument checks belong to ``flashinfer/kda_prefill.py`` and have
    already run by the time this is reached; what is decided here is the one
    question that needs this package: whether the device is CC 12.0 *and* the
    installed CuTe DSL and CUDA toolkit can natively build ``sm_120a``.

    Both variants share that gate.  A device where only a family-conditional
    target is available is refused outright rather than allowed to run one
    variant at a target the other cannot use.
    """
    q = kwargs.get("q")
    if not isinstance(q, torch.Tensor) or not q.is_cuda:
        return False
    return sm120a_available(q.device)


def run_kda_prefill_sm120(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: Optional[float] = None,
    lower_bound: float,
    initial_state: Optional[torch.Tensor] = None,
    final_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    variant: str = DEFAULT_VARIANT,
    safe_gate: bool = True,
    resources: Optional[SM120PrefillResources] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run SM120 KDA prefill and return ``(output, final_state)``.

    This is the *backend* ABI, and it deliberately is not the public one.
    ``initial_state`` is read-only and ``final_state`` is written; passing the
    same tensor as both is the exact alias the kernels' schedules prove safe,
    and passing ``final_state=None`` means "do not store a state at all", which
    the kernels implement by skipping the store rather than by writing a buffer
    nobody asked for.

    The public contract -- where a supplied ``initial_state`` is updated in
    place whether or not the caller asked for a final state -- is one level up,
    in ``flashinfer/kda_prefill.py``.  Keeping the split here is what lets a
    cross-implementation A/B compare like with like: the comparison path has
    exactly this ABI, so a harness can hand both paths the same two tensors and
    a difference in the result is a difference in the kernels.

    ``resources`` is the SM120 half of a caller-owned
    :class:`~flashinfer.kda_prefill.RecurrentKDAPrefillWorkspace`.  Passing one
    is what makes CUDA graph capture possible: it gives the canonical metadata,
    the scratch and the descriptors a lifetime that outlives this module's
    caches, which replay needs because it never re-enters Python.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")

    out = output if output is not None else torch.empty_like(v)
    resolved_scale = float(scale) if scale is not None else DK**-0.5

    tensors = (
        q,
        k,
        v,
        g,
        beta,
        out,
        A_log,
        dt_bias,
        initial_state,
        final_state,
        cu_seqlens,
    )
    scalars = (resolved_scale, float(lower_bound), variant, bool(safe_gate))

    resolved = _resolved_call(q.device, tensors, scalars, resources)
    if resolved is not None:
        # The whole host path already ran for these exact tensors: replay the
        # plan it produced rather than walking eleven tensors again to find it.
        # The executor is stored already bound, so this is one call rather than
        # a dict lookup and an attribute fetch -- which is measurable when the
        # kernel it precedes is nine microseconds long.
        # `bind` is not repeated here: the memo key already carries the
        # variant, the stream and every tensor's identity, so the only state
        # that can have changed since it was bound is capture -- which happens
        # after a launch, not during one, and is what makes the workspace spent.
        if resources is not None and resources.captured:
            raise RuntimeError(
                "this RecurrentKDAPrefillWorkspace has already participated in "
                "a CUDA graph capture and cannot be reused; create another one"
            )
        execute, plan = resolved
        execute(plan, initial_state, final_state)
        return out, final_state

    info = validate_inputs(
        q,
        k,
        v,
        g,
        beta,
        out,
        A_log,
        dt_bias,
        scale=resolved_scale,
        lower_bound=float(lower_bound),
        initial_state=initial_state,
        final_state=final_state,
        cu_seqlens=cu_seqlens,
    )
    offsets = canonical_offsets(
        cu_seqlens,
        batch=info.batch,
        tokens=info.tokens_per_sequence,
        total_tokens=info.total_tokens,
        device=q.device,
    )
    chosen = _resolve_variant(
        variant, info, offsets, safe_gate, resources, device=q.device
    )

    if resources is not None:
        # Pin the workspace to this variant, stream and shape before anything
        # writes to it.  The state machine has been here since the backend
        # landed and nothing called it, so every constraint it encodes was
        # unenforced: a workspace already spent on a capture could be driven
        # again from eager Python, and one workspace could be shared across two
        # streams or two variants with the second silently overwriting scratch
        # the first had not finished reading.
        resources.bind(
            variant=chosen,
            stream_ptr=current_stream_ptr(q.device),
            signature=(
                tuple(tensor_layout_identity(t) for t in tensors),
                (resolved_scale, float(lower_bound), bool(safe_gate)),
            ),
        )

    module = _variant_module(chosen)
    plan = module.run(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        out=out,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=resolved_scale,
        lower_bound=float(lower_bound),
        initial_state=initial_state,
        final_state=final_state,
        cu_seqlens=cu_seqlens,
        info=info,
        offsets=offsets,
        resources=resources,
        safe_gate=safe_gate,
    )
    _remember_call(q.device, tensors, scalars, resources, (module.execute, plan))
    return out, final_state


# ---------------------------------------------------------------------------
# The facade's own fast path.
#
# Validation, offset canonicalization and the variant choice are all pure
# functions of the tensors' addresses, shapes, dtypes and versions plus the
# scalars -- and all three are expensive enough to matter. Measured on this
# backend before this memo existed: +0.11 ms of host time on *every* call,
# which is invisible next to a 1 ms kernel and 12x the whole call at
# B=1 T=16 H=4. The variants' own plan caches did not help, because this work
# happens before they are reached.
#
# The three costs, in order of size:
#
# * ``validate_inputs`` walks eleven tensors and the full alias matrix;
# * ``canonical_offsets`` goes through a per-device cache whose *hit* path
#   issues ``wait_event`` and ``record_stream`` -- driver calls, not dict
#   lookups;
# * ``choose_variant`` asked the driver for the device's SM count once per
#   call, which is a property that cannot change under a live process.
#
# The stream is in the key even though none of the three results depend on it.
# That is deliberate: the offsets cache's ``wait_event`` is what orders a
# consumer on a new stream against the buffer's creation, and memoizing past it
# would skip that edge the first time a second stream appeared.
# ---------------------------------------------------------------------------

#: One entry per distinct buffer set and workspace; a serving loop needs one.
#: Each value carries weak references to the key tensors so an allocator-reused
#: address cannot turn a different tensor object into a stale plan hit.
#: Its values carry the variants' plan objects, so it retains what they retain;
#: see ``CALL_PLAN_MAX_ENTRIES`` for the measurements behind the number.
RESOLVED_CALL_MAX_ENTRIES = 16

#: Serializes the two mutating paths through ``_RESOLVED``.  Reentrant on
#: purpose: dropping an entry releases its weak referents, and their ``_purge``
#: callbacks run on the releasing thread and take this lock again.
#:
#: The ``_RESOLVED_LAST`` fast path deliberately stays outside it.  That global
#: holds an immutable tuple, so reading it is one atomic load, and it is the
#: path a warm serving loop takes before a nine-microsecond kernel.
_RESOLVED_LOCK = threading.RLock()
_RESOLVED: "OrderedDict[tuple, tuple]" = OrderedDict()

#: The previous call, compared by object identity and the complete tensor
#: identity before the key is built.  Object identity alone is insufficient:
#: ``tensor.data = other`` keeps the Python object and can replace its storage
#: and layout without a readable version bump under ``inference_mode``.
_RESOLVED_LAST: Optional[tuple] = None


def _resolved_key(device, tensors, scalars, resources) -> tuple:
    # The stream of the *inputs'* device.  Asked with no argument the handle is
    # the current device's, which is not the one the launch uses: a plan bakes
    # ``torch.cuda.current_stream(q.device)`` into its argument tuple, so a
    # process holding tensors on cuda:1 while cuda:0 is current would key two
    # different cuda:1 streams onto one entry and reuse a plan bound to the
    # first of them.
    return (
        tuple(tensor_identity(t) for t in tensors),
        scalars,
        resource_cache_token(resources),
        current_stream_ptr(device),
    )


def _resolved_call(device, tensors, scalars, resources):
    """The memoized ``(execute, plan)`` for this call, or ``None`` on a miss.

    Written with explicit loops and early returns rather than ``all(...)`` over
    generator expressions.  That reads worse and costs less: this runs before
    every launch, and at the smallest supported shape the kernel it precedes is
    nine microseconds long, so two generator frames per call are visible in a
    paired benchmark.
    """
    last = _RESOLVED_LAST
    if last is not None:
        (
            last_refs,
            last_identities,
            last_scalars,
            last_resources,
            last_stream,
            value,
        ) = last
        if (
            last_scalars == scalars
            and last_resources is resource_cache_token(resources)
            and last_stream == current_stream_ptr(device)
        ):
            for index, tensor in enumerate(tensors):
                # Weak, for the same reason the LRU below is: this entry
                # outlives the call, and eleven strong references to q, k, v, g
                # and out would keep one whole activation set off the caching
                # allocator until the next call replaced it.
                ref = last_refs[index]
                if ref is None:
                    if tensor is not None:
                        break
                elif ref() is not tensor:
                    break
                if tensor_identity(tensor) != last_identities[index]:
                    break
            else:
                return value

    key = _resolved_key(device, tensors, scalars, resources)
    with _RESOLVED_LOCK:
        entry = _RESOLVED.get(key)
        if entry is None:
            return None

        _token, refs, value = entry
        for ref, tensor in zip(refs, tensors, strict=True):
            if ref is None:
                if tensor is not None:
                    break
            elif ref() is not tensor:
                break
        else:
            _RESOLVED.move_to_end(key)
            return value

        # The identity key matched only because an address was recycled.
        # Remove the stale entry now; its weakref callback may not have run.
        _RESOLVED.pop(key, None)
    return None


def _remember_call(device, tensors, scalars, resources, value) -> None:
    global _RESOLVED_LAST
    _RESOLVED_LAST = (
        tuple(None if t is None else weakref.ref(t) for t in tensors),
        tuple(tensor_identity(t) for t in tensors),
        scalars,
        resource_cache_token(resources),
        current_stream_ptr(device),
        value,
    )
    key = _resolved_key(device, tensors, scalars, resources)
    token = object()

    def _purge(_ref, _key=key, _token=token):
        with _RESOLVED_LOCK:
            entry = _RESOLVED.get(_key)
            if entry is not None and entry[0] is _token:
                _RESOLVED.pop(_key, None)

    refs = tuple(
        None if tensor is None else weakref.ref(tensor, _purge) for tensor in tensors
    )
    with _RESOLVED_LOCK:
        _RESOLVED[key] = (token, refs, value)
        while len(_RESOLVED) > RESOLVED_CALL_MAX_ENTRIES:
            _RESOLVED.popitem(last=False)


def _resolve_variant(variant, info, offsets, safe_gate, resources, device=None) -> str:
    """Turn ``"auto"`` into a concrete variant, once, and remember it.

    A workspace that has already chosen keeps its choice: the selection is part
    of the captured signature, and re-deciding it during replay is not possible
    anyway -- replay does not run this code.  Re-deciding it between warmup and
    capture would be worse, because it would silently record a different kernel
    than the one the warmup proved.
    """
    if resources is not None and resources.variant is not None:
        if variant != "auto" and variant != resources.variant:
            raise KDAPrefillValidationError(
                "RecurrentKDAPrefillWorkspace is already bound to variant "
                f"{resources.variant!r}, so it cannot run requested variant "
                f"{variant!r}; create a separate workspace"
            )
        chosen = resources.variant
    elif variant != "auto":
        chosen = variant
    elif not safe_gate:
        # Only one variant implements the unbounded gate, so the shape rule has
        # nothing left to choose between.  Stated here rather than left to
        # produce a confusing refusal from the decomp branch.
        chosen = "fused"
    else:
        chosen = choose_variant(
            offsets.sequences or info.batch,
            info.heads,
            offsets.longest_sequence if offsets.lengths else info.tokens_per_sequence,
            device=device,
        )

    if not safe_gate and chosen == "decomp":
        raise KDAPrefillValidationError(
            "the decomp variant does not support safe_gate=False; pass "
            "variant='fused' or leave it on auto"
        )
    return chosen


def clear_kda_prefill_sm120_caches() -> None:
    """Drop every cache this backend holds, in both variants and the runtime.

    Only clears a variant that has actually been imported: asking for the
    others would compile nothing but would import device modules a process may
    have deliberately never loaded.

    Deliberately does not drop graph pins.  A live CUDA graph reads its
    captured resources at their captured addresses, and freeing those is not
    something a cache-clear should be able to do by accident.
    """
    from .runtime import clear_shared_caches  # noqa: PLC0415

    global _RESOLVED_LAST
    with _RESOLVED_LOCK:
        _RESOLVED.clear()
    _RESOLVED_LAST = None
    _sm_count.cache_clear()
    with _MODULES_LOCK:
        modules = tuple(_MODULES.values())
    for module in modules:
        module.clear_caches()
    clear_shared_caches()
