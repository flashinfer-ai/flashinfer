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

They agree numerically within the tolerances the test suite asserts, so the
choice between them is a performance one, and :func:`choose_variant` makes it
from a measured table rather than a formula.
"""

from __future__ import annotations

import functools
import threading
import weakref
from collections import OrderedDict
from typing import Any, NamedTuple, Optional

import torch

from .runtime import (
    DK,
    DV,
    KDAPrefillValidationError,
    SM120PrefillResources,
    canonical_offsets,
    current_stream_ptr,
    resource_cache_token,
    sm120a_available,
    tensor_identity,
    tensor_layout_identity,
    validate_inputs,
)

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
# A table keyed on SM count rather than a formula: the crossover between the
# two variants depends on H and T as well as on how many waves the grid takes,
# and a device name is not a stable unique selector.  Each row was fitted on
# one device from timings of both pinned variants; any other CC 12.0 device
# uses ``FALLBACK_AUTO_PROFILE`` and :func:`describe_variant_policy` says so.
#
# A threshold is a fit, not a property of the kernel, and it goes stale when
# either kernel changes.  To re-fit a row, time all three benchmark backends
# over the shapes of interest and move the boundary to where the pinned
# variants cross:
#
#   python benchmarks/flashinfer_benchmark.py --routine recurrent_kda_prefill \
#       --backends flashinfer flashinfer-decomp flashinfer-fused \
#       --batch_size B --s_qo T --num_q_heads H --refcheck
# ---------------------------------------------------------------------------


class AutoProfile(NamedTuple):
    """Thresholds for one device.  See :data:`AUTO_PROFILES`."""

    #: Heads at or above which the fused variant wins regardless of T, or
    #: ``None`` when the fit needed no such term.
    heads: Optional[int]
    #: Per-sequence length at or below which it wins regardless of H.
    tokens: Optional[int]
    #: Decomp CTA count (``2 * batch * heads``) at or above which it wins.
    ctas: Optional[int]
    #: Human name for the device, for reporting.  Not used for matching.
    device: str
    #: Where the numbers came from, printed by :func:`describe_variant_policy`.
    source: str
    #: Lower CTA threshold when all sequences have the same length. Ragged
    #: batches retain ``ctas``: short sequences do not fill the long tail.
    uniform_ctas: Optional[int] = None
    #: Optional length cap on the lower uniform CTA threshold, not on ``ctas``.
    uniform_max_tokens: Optional[int] = None


AUTO_PROFILES: dict[int, AutoProfile] = {
    # Short sequences always take the fused kernel; above that the decomposed
    # kernel keeps narrow grids and the fused one takes wide grids. Re-fitting
    # the optimized kernels on 156/188 SMs admits uniform grids at 96 CTAs;
    # use a 128-CTA line for ragged batches, where lowering it loses
    # on long-tail shapes such as H=12, lengths=(4096, 1024, 65, 17).
    # On 156 SMs the new line is limited to T <= 8192: the 16K/32K H=48
    # validation points cross back to decomp. The 188-SM points stay fused.
    156: AutoProfile(
        heads=None,
        tokens=130,
        ctas=128,
        device="156-SM SM120 part",
        source="fitted from decomp/fused timings on a 156-SM CC 12.0 device",
        uniform_ctas=96,
        uniform_max_tokens=8192,
    ),
    188: AutoProfile(
        heads=None,
        tokens=130,
        ctas=128,
        device="188-SM SM120 part",
        source="fitted from decomp/fused timings on a 188-SM CC 12.0 device",
        uniform_ctas=96,
    ),
    110: AutoProfile(
        heads=None,
        tokens=130,
        ctas=96,
        device="110-SM SM120 part",
        source="fitted from decomp/fused timings on a 110-SM CC 12.0 device",
    ),
}

#: Used for any CC 12.0 device without a row above.  A fallback, not a claim:
#: :func:`describe_variant_policy` reports which case a machine is in so a
#: number taken on an unprofiled card cannot be read as tuned.
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
    if profile.uniform_ctas is not None:
        uniform_term = f"equal lengths and CTA>={profile.uniform_ctas}"
        if profile.uniform_max_tokens is not None:
            uniform_term += f" and T<={profile.uniform_max_tokens}"
        terms.append(f"({uniform_term})")
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
    *,
    uniform: bool = True,
) -> str:
    """Which variant the measured table says is faster for this shape.

    ``tokens`` is the per-sequence length, not the packed total: the table was
    measured on equal-length sequences and the CTA count is what varies with
    ``batch``.  For a ragged batch the caller passes the *longest* sequence --
    the recurrence is serial within a sequence, so the longest one sets the
    critical path, and a batch containing a 130 behaves like a 130 rather than
    like the 27 its lengths average to.

    ``uniform`` must be false for unequal lengths (including empty sequences)
    so the lower, equal-length CTA threshold cannot select a slower long tail.

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
    if (
        uniform
        and profile.uniform_ctas is not None
        and ctas >= profile.uniform_ctas
        and (profile.uniform_max_tokens is None or tokens <= profile.uniform_max_tokens)
    ):
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
    final_state_is_private: bool = False,
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
    in ``flashinfer/kda_prefill.py``.  Keeping that adaptation out of the
    backend ABI lets tests and benchmarks drive the kernels directly.

    ``final_state_is_private`` says the caller allocated ``final_state`` for
    this call alone -- it is written, never read, and nothing outside the call
    held it when the call began.  That licenses the call memo to verify the slot
    by address and layout instead of by object identity, which is the difference
    between a warm call and a full plan rebuild when the buffer is a fresh
    ``torch.empty`` each time.  Leave it false for a caller-supplied buffer:
    there the object check is what stops a recycled address from turning someone
    else's tensor into a plan hit.

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
    private = _PRIVATE_FINAL_STATE if final_state_is_private else _NO_PRIVATE_SLOTS

    resolved = _resolved_call(q.device, tensors, scalars, resources)
    if resolved is not None:
        # The whole host path already ran for these exact tensors: replay the
        # plan it produced rather than walking eleven tensors again to find it.
        # The executor is stored already bound, so this is one call rather than
        # a dict lookup and an attribute fetch.
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
        # writes to it.  ``bind`` refuses a workspace already spent on a
        # capture, and one shared across streams or variants, where a second
        # caller could overwrite scratch the first has not finished reading.
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
    _remember_call(
        q.device, tensors, scalars, resources, (module.execute, plan), private
    )
    return out, final_state


# ---------------------------------------------------------------------------
# The facade's own fast path.
#
# Validation, offset canonicalization and the variant choice are all pure
# functions of the tensors' addresses, shapes, dtypes and versions plus the
# scalars -- and together they dominate the host time of a small call.  The
# variants' own plan caches cannot help, because this work happens before they
# are reached.
#
# The three costs, in order of size:
#
# * ``validate_inputs`` walks eleven tensors and the full alias matrix;
# * ``canonical_offsets`` goes through a per-device cache whose *hit* path
#   issues ``wait_event`` and ``record_stream`` -- driver calls, not dict
#   lookups;
# * ``choose_variant`` resolves the device and its SM count.
#
# The stream is in the key even though none of the three results depend on it.
# That is deliberate: the offsets cache's ``wait_event`` is what orders a
# consumer on a new stream against the buffer's creation, and memoizing past it
# would skip that edge the first time a second stream appeared.
# ---------------------------------------------------------------------------

#: Stored in place of a weak reference for a slot the immediate caller
#: allocated for this call alone -- an output buffer that exists only to be
#: written, whose lifetime ends when the caller drops it.
#:
#: Such a slot must be verified by address and layout, not by object identity.
#: The weak reference exists to catch an address the allocator recycled into a
#: *different* tensor; a buffer allocated for this call cannot be that, because
#: whatever now lives at the address is, by construction, this call's own
#: output. Checking object identity there is not conservative, it is simply
#: wrong for the case, and it costs a full plan rebuild on every call: a fresh
#: ``torch.empty`` is a new Python object even when the caching allocator hands
#: back the same block, so ``ref() is not tensor`` fires on a hit, and the
#: object's death then purges the LRU entry as well.
_BY_ADDRESS = object()

#: Position of ``final_state`` in the memo's ``tensors`` tuple. Named rather
#: than written inline because the tuple is built in one place and read in
#: another, and an index that drifts would silently license the wrong slot.
_FINAL_STATE_SLOT = 9

#: Both answers, precomputed.  Indices rather than a parallel mask, so a caller
#: passing a shorter tensor tuple needs no matching mask, and built once
#: because this runs before every launch.
_NO_PRIVATE_SLOTS: tuple = ()
_PRIVATE_FINAL_STATE: tuple = (_FINAL_STATE_SLOT,)

# ``out`` is deliberately not on this list even though a call that allocates it
# owns it just as exclusively.  Marking it would buy nothing: a live plan
# retains the buffer its descriptors address, at the C level, so the previous
# output is still allocated when the next call asks for one and the allocator
# hands back a *different* block every time.  An address-keyed slot whose
# address always changes is a relaxed check that never turns a miss into a
# hit, which is the worst of both.  Callers who want a warm memo should pass
# ``output``.

#: One entry per distinct buffer set and workspace; a serving loop needs one.
#: Each value carries weak references to the key tensors so an allocator-reused
#: address cannot turn a different tensor object into a stale plan hit.
#: Its values carry the variants' plan objects, so it retains what they retain.
RESOLVED_CALL_MAX_ENTRIES = 16

#: Serializes the two mutating paths through ``_RESOLVED``.  Reentrant on
#: purpose: dropping an entry releases its weak referents, and their ``_purge``
#: callbacks run on the releasing thread and take this lock again.
#:
#: The ``_RESOLVED_LAST`` fast path deliberately stays outside it.  That global
#: holds an immutable tuple, so reading it is one atomic load, and it is the
#: path a warm serving loop takes on every call.
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
    every launch, and at the smallest supported shapes the extra generator
    frames are a visible fraction of the call.
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
                elif ref is not _BY_ADDRESS and ref() is not tensor:
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
            elif ref is not _BY_ADDRESS and ref() is not tensor:
                break
        else:
            _RESOLVED.move_to_end(key)
            return value

        # The identity key matched only because an address was recycled.
        # Remove the stale entry now; its weakref callback may not have run.
        _RESOLVED.pop(key, None)
    return None


def _remember_call(device, tensors, scalars, resources, value, private=()) -> None:
    def _hold(index, tensor, callback=None):
        if tensor is None:
            return None
        if index in private:
            # No callback either: this buffer dies at the end of every call, and
            # a purge on its death would drop the entry the next call wants.
            return _BY_ADDRESS
        if callback is None:
            return weakref.ref(tensor)
        return weakref.ref(tensor, callback)

    global _RESOLVED_LAST
    _RESOLVED_LAST = (
        tuple(_hold(i, t) for i, t in enumerate(tensors)),
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

    refs = tuple(_hold(i, tensor, _purge) for i, tensor in enumerate(tensors))
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
        longest = (
            offsets.longest_sequence if offsets.lengths else info.tokens_per_sequence
        )
        chosen = choose_variant(
            offsets.sequences or info.batch,
            info.heads,
            longest,
            device=device,
            # Canonical offsets already hold validated host metadata; no new
            # device synchronization is needed, including during graph capture.
            uniform=offsets.total_tokens == offsets.sequences * longest,
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
