# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Specialized fused-GDN-decode backends (see README.md).

This module owns the specialized backends of
:func:`flashinfer.gdn_fused_decode_step`: a JSON registry
(``gdn_fused_decode_registry.json``) maps complete workload signatures to
kernel implementation modules (``kernel/gdn_fused_decode_<impl>.py``), and
the thin hook in :mod:`.gdn_fused_decode` delegates here.

There is no environment gate anywhere in this package.  The registry and
the probe answer whether a call is *supported*; whether a supported call
should be made at all is the calling framework's decision, taken on the
framework's own configuration surface.  (A kill switch belongs to an
integration that *replaces* an existing FlashInfer implementation, where
unsetting it has something to fall back to.  This fused step is a new
operation: there is no in-FlashInfer alternative to fall back to, so a
variable here would only be a second, unmeasured policy surface.)

Selection is internal and has exactly one entry point,
:func:`try_run_gdn_fused_decode_specialized`: a registered signature is
tried against the impl families in preference order (``cute_dsl``, then
``cuda``), and anything not served returns ``None`` so the composable
reference path takes the call.  The op exposes no backend argument -- these
families are an implementation detail of one operation, not a user-facing
choice -- so there is no "run this kernel or raise" path to maintain.  A
kernel failure warns once and latches that impl off for the rest of the
process; the stock path is never broken.

That latch is invisible to the caller by design, which is exactly what makes
it dangerous to a *measurement*: a benchmark or accuracy run keeps passing
while a different kernel produces its numbers.  So dispatch also attests --
each impl logs one line the first time it serves, and
:func:`gdn_fused_decode_stats` reports ``served_impls`` alongside
``failed_impls``.  A harness that pins a specific impl should assert on those
rather than on the call merely returning.

Routing-probe cost: serving asks :func:`gdn_fused_decode_supported_geometry`
once per layer per decode step, outside the CUDA graph, and the answer is
the same every time.  Registry rows are therefore indexed by
``(cc, signature)`` instead of scanned, device compute capability is looked
up once per device, and the geometry -> answer mapping is memoized.  The
memo is invalidated whenever anything it depends on changes: the registry
object (tests and harnesses substitute it) or the failure latch.

CUDA-graph contract: impl modules compile lazily, per variant, on the first
eager (non-capturing) dispatch of that exact call -- vLLM's profile run
precedes its capture phase and warms exactly the serving variants.  During
capture an impl dispatches only when ``ready_for_graph_capture`` confirms
this exact call is already compiled and warmed; a capture-unready impl falls
through to the next one, then to the (capture-safe) composable path.
"""

import functools
import importlib
import json
from importlib import resources
from typing import List, Optional, Tuple

import torch

from ...jit.core import logger as jit_logger

_REGISTRY_RESOURCE = "gdn_fused_decode_registry.json"
# The signature fields, in the fixed order the registry index keys on.
_SIGNATURE_FIELDS = (
    "b",
    "hidden",
    "n_ba",
    "qkv_dim",
    "h_q",
    "hv",
    "d",
    "conv_width",
    "conv_state_len",
    "conv_layout",
)
_REGISTRY_ROW_FIELDS = ("impl", "cc") + _SIGNATURE_FIELDS
_CONV_LAYOUTS = ("SD", "DS")

# INTERNAL impl families -> impl modules serving them, in registry-matching
# order within one family.  Dispatch tries the families in
# _AUTO_BACKEND_ORDER.  These names are not part of the public API (the op
# takes no backend argument); they exist so a registry row's ``impl`` can be
# grouped with the other kernels that implement the same strategy.  New
# kernels extend these tuples (or add a family) plus registry rows -- the
# dispatch logic does not change.
BACKEND_IMPLS = {
    "cute_dsl": ("cutedsl_sm120_pdl",),
    "cuda": ("cuda_sm120_persistent",),
}
_AUTO_BACKEND_ORDER = ("cute_dsl", "cuda")

# Impls latched off after a kernel failure: the specialized kernel stops
# dispatching for the rest of the process so the stock path is never broken
# twice.
_failed_impls: set = set()

# Impls that have actually served at least one call in this process.
#
# The latch above is deliberately silent to the caller -- that is what keeps
# the stock path unbreakable -- but it means a benchmark or an accuracy run
# can measure a DIFFERENT kernel than the one it believes it is measuring and
# still pass every gate.  This set, the one-time log line each impl emits the
# first time it serves, and ``served_impls`` in
# :func:`gdn_fused_decode_stats` exist so a measurement run can ATTEST which
# kernel produced its numbers instead of inferring it from the registry.
# Cost on the dispatch path is one set membership test.
_served_impls: set = set()

# (cc, *signature) -> bool answers of the routing probe, and the registry
# object they were derived from.  Serving repeats one question per layer per
# decode step; see the module docstring.
_probe_memo: dict = {}
_registry_index_cache: tuple = ()  # (rows_object, index) or ()


def _latch_impl_off(impl: str) -> None:
    """Stop dispatching ``impl`` for the rest of the process.

    Also drops the probe memo: a latched impl can turn a ``True`` answer
    into ``False``, and a consumer that keeps asking must see that.
    """
    _failed_impls.add(impl)
    _probe_memo.clear()


def _attest_served(impl: str) -> None:
    """Record -- and announce once -- that ``impl`` served a call.

    A measurement run must be able to name the kernel behind its numbers.
    The registry cannot answer that: the preferred impl can be latched off
    mid-run and the next one takes over without the op ever raising.  So the
    first dispatch of each impl logs one line naming it, at INFO (this is
    normal operation, not a problem), and the answer is also readable from
    :func:`gdn_fused_decode_stats`.
    """
    if impl in _served_impls:
        return
    _served_impls.add(impl)
    jit_logger.info_once(
        "Fused GDN decode step is being served by specialized impl '%s'.", impl
    )


@functools.cache
def load_gdn_fused_decode_registry() -> tuple:
    """Load the packaged fused-GDN-decode registry rows (empty on failure)."""
    try:
        payload = json.loads(
            resources.files(__package__).joinpath(_REGISTRY_RESOURCE).read_text()
        )
        if payload["op"] != "gdn_fused_decode_step" or payload["schema_version"] != 1:
            raise ValueError(
                f"unsupported registry header: op={payload['op']!r} "
                f"schema_version={payload['schema_version']!r}"
            )
        rows = tuple(payload["workloads"])
        for row in rows:
            missing = [field for field in _REGISTRY_ROW_FIELDS if field not in row]
            if missing:
                raise ValueError(f"registry row missing fields {missing}: {row}")
            if row["conv_layout"] not in _CONV_LAYOUTS:
                raise ValueError(f"unexpected conv_layout in registry row: {row}")
        return rows
    except (
        FileNotFoundError,
        ModuleNotFoundError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        jit_logger.warning_once(
            "Unable to load the specialized fused GDN decode registry "
            "(%s: %s); the specialized backends have no workloads.",
            type(exc).__name__,
            exc,
        )
        return ()


def _registry_index() -> dict:
    """``(cc, *signature) -> rows`` index of the current registry.

    Rebuilt whenever :func:`load_gdn_fused_decode_registry` starts returning
    a different rows object -- which is how tests and benchmark harnesses
    restrict or widen the registry in process.  The rows object is held in
    the cache so its identity cannot be recycled, and the probe memo (whose
    answers came from the previous index) is dropped with it.
    """
    global _registry_index_cache
    rows = load_gdn_fused_decode_registry()
    if _registry_index_cache and _registry_index_cache[0] is rows:
        return _registry_index_cache[1]
    index: dict = {}
    for row in rows:
        key = (row["cc"],) + tuple(row[field] for field in _SIGNATURE_FIELDS)
        index.setdefault(key, []).append(row)
    frozen = {key: tuple(value) for key, value in index.items()}
    _registry_index_cache = (rows, frozen)
    _probe_memo.clear()
    return frozen


def _rows_for_impl(impl: str) -> tuple:
    """Registry rows served by ``impl`` -- the impl's own dispatch surface."""
    return tuple(row for row in load_gdn_fused_decode_registry() if row["impl"] == impl)


# The geometry fields, in the order the kernel modules take them.
_GEOMETRY_FIELDS = (
    "hidden",
    "n_ba",
    "qkv_dim",
    "h_q",
    "hv",
    "d",
    "conv_width",
    "conv_state_len",
)


def registry_geometries() -> List[tuple]:
    """Distinct layer geometries in the registry, in first-seen order.

    The layer geometry is a compile-time parameter of both impls, so this is
    the set that would have to be covered to precompile this op ahead of time
    (one CUDA module per geometry); batch size, the query scale and the
    conv-state layout are handled inside a module or per compiled CuTe-DSL
    variant.  Both impls JIT-compile on first eager dispatch instead (see the
    AOT note in README.md), so nothing in the build path calls this today --
    it is what the tiling test enumerates, and what a JIT-disabled deployment
    would iterate over to restore an AOT entry.
    """
    geometries: dict = {}
    for row in load_gdn_fused_decode_registry():
        geometries.setdefault(
            tuple(int(row[field]) for field in _GEOMETRY_FIELDS), None
        )
    return list(geometries)


@functools.cache
def _load_impl(impl: str):
    """Import the ``kernel/gdn_fused_decode_<impl>`` module, or None."""
    try:
        return importlib.import_module(f".kernel.gdn_fused_decode_{impl}", __package__)
    except (ImportError, RuntimeError) as exc:
        jit_logger.warning_once(
            "Specialized fused GDN decode impl '%s' is unavailable (%s).",
            impl,
            type(exc).__name__,
        )
        return None


def conv_state_layout(conv_state: torch.Tensor) -> Optional[str]:
    """Layout tag of a logical ``[P, qkv_dim, state_len]`` conv-state view.

    ``"SD"`` for a transposed SD pool (stride-1 channels, the vLLM default
    ``(state_len, dim)`` physical rows), ``"DS"`` for a DS-dense pool
    (stride-1 time steps), ``None`` for anything else.  Derived from the
    view's own sizes, so the classifier needs no fixed geometry.
    """
    if conv_state.ndim != 3:
        return None
    qkv_dim, state_len = conv_state.shape[1], conv_state.shape[2]
    if conv_state.stride(1) == 1 and conv_state.stride(2) == qkv_dim:
        return "SD"
    if conv_state.stride(1) == state_len and conv_state.stride(2) == 1:
        return "DS"
    return None


@functools.lru_cache(maxsize=32)
def _device_cc_for_index(index: int) -> Optional[int]:
    """Compute capability of CUDA device ``index`` as ``major * 10 + minor``.

    Cached on a resolved ordinal: a device's capability cannot change, so
    this is the only part of the lookup that is safe to memoize.
    """
    try:
        major, minor = torch.cuda.get_device_capability(index)
    except (RuntimeError, ValueError, AssertionError, TypeError):
        return None
    return major * 10 + minor


def _device_cc(device) -> Optional[int]:
    """Compute capability of ``device`` as ``major * 10 + minor``, or None.

    ``device`` may be ``None``, an ordinal, a string or a
    :class:`torch.device`; ``None`` and an index-less ``"cuda"`` both mean
    "whatever device is current *now*".  That is why the ordinal is resolved
    BEFORE the cache is consulted: memoizing under the literal argument
    would pin the first answer to the key ``None``, and a later
    ``torch.cuda.set_device()`` onto a device of a different capability
    would keep reading the old one -- routing a call into a kernel built for
    another architecture, or declining one it could serve.  On the
    per-layer-per-step probe path this costs one ``current_device()`` call
    (tens of nanoseconds) on top of the memo lookup, only when the caller
    did not name a device.
    """
    try:
        if device is None:
            index: int = torch.cuda.current_device()
        elif isinstance(device, int):
            index = device
        else:
            resolved = torch.device(device)
            if resolved.type != "cuda":
                return None
            index = (
                resolved.index
                if resolved.index is not None
                else torch.cuda.current_device()
            )
    except (RuntimeError, ValueError, AssertionError, TypeError):
        return None
    return _device_cc_for_index(index)


def match_gdn_fused_decode_signature(signature: dict, device) -> List[dict]:
    """Registry rows serving one signature dict, in registry order.

    A pure query: no kernel loading and no GPU work.  ``signature`` carries
    the non-``impl``/``cc`` registry fields; the device supplies ``cc``.  A
    complete signature is answered from the ``(cc, signature)`` index; a
    partial one (a subset of the fields, which callers use to ask coarser
    questions) still scans, since it is not on any hot path.
    """
    cc = _device_cc(device)
    if cc is None:
        return []
    if len(signature) == len(_SIGNATURE_FIELDS):
        key = (cc,) + tuple(signature[field] for field in _SIGNATURE_FIELDS)
        return list(_registry_index().get(key, ()))
    return [
        row
        for row in load_gdn_fused_decode_registry()
        if row["cc"] == cc
        and all(row[field] == signature[field] for field in signature)
    ]


def signature_from_geometry(
    batch_size: int,
    hidden_size: int,
    n_ba: int,
    qkv_dim: int,
    num_qk_heads: int,
    num_v_heads: int,
    head_dim: int,
    conv_width: int,
    conv_state_len: int,
    conv_layout: str,
) -> Optional[dict]:
    """Signature dict for the probe's scalar geometry, or None if malformed.

    The layout parameter is named ``conv_layout`` -- the registry field name
    -- rather than ``conv_state_layout``, which is the module-level
    classifier :func:`conv_state_layout`: a parameter of that name shadows
    the function inside this body, so a later edit reaching for the
    classifier here would silently call a string.
    """
    if conv_layout not in _CONV_LAYOUTS:
        return None
    return {
        "b": int(batch_size),
        "hidden": int(hidden_size),
        "n_ba": int(n_ba),
        "qkv_dim": int(qkv_dim),
        "h_q": int(num_qk_heads),
        "hv": int(num_v_heads),
        "d": int(head_dim),
        "conv_width": int(conv_width),
        "conv_state_len": int(conv_state_len),
        "conv_layout": str(conv_layout),
    }


def signature_from_tensors(
    hidden_states: torch.Tensor,
    w_ba: torch.Tensor,
    mixed_qkv: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    conv_state: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_state: torch.Tensor,
    state_indices: torch.Tensor,
    use_qk_l2norm: bool,
    out: Optional[torch.Tensor] = None,
) -> Optional[dict]:
    """Signature dict of a full call, or None when the call violates the op
    contract the specialized kernels assume.

    Beyond the registry's exact sizes this validates the fixed (non-per-row)
    parts of the contract: bf16 activations/weights/conv pool, fp32
    ``A_log``/``ssm_state``, int32 indices, dense inner layouts with the
    documented stride freedoms (row-strided ``mixed_qkv`` views, padded
    conv/ssm pool page strides, SD or DS conv pools), qk-L2-norm enabled,
    and a dense bf16 ``out`` destination when provided.
    """
    if not use_qk_l2norm:
        return None
    if not hidden_states.is_cuda:
        return None
    if not all(
        tensor.is_cuda and tensor.device == hidden_states.device
        for tensor in (
            w_ba,
            mixed_qkv,
            conv_weight,
            conv_bias,
            conv_state,
            A_log,
            dt_bias,
            ssm_state,
            state_indices,
        )
    ):
        return None
    if (
        hidden_states.ndim != 2
        or w_ba.ndim != 2
        or mixed_qkv.ndim != 2
        or conv_weight.ndim != 2
        or conv_bias.ndim != 1
        or conv_state.ndim != 3
        or A_log.ndim != 1
        or dt_bias.ndim != 1
        or ssm_state.ndim != 4
        or state_indices.ndim != 1
    ):
        return None

    B, hidden = hidden_states.shape
    n_ba = w_ba.shape[1]
    qkv_dim = mixed_qkv.shape[1]
    hv = A_log.shape[0]
    d = ssm_state.shape[-1]
    conv_width = conv_weight.shape[1]
    conv_state_len = conv_state.shape[2]
    if d <= 0 or hv * d >= qkv_dim or (qkv_dim - hv * d) % (2 * d) != 0:
        return None
    h_q = (qkv_dim - hv * d) // (2 * d)

    if (
        w_ba.shape != (hidden, n_ba)
        or mixed_qkv.shape != (B, qkv_dim)
        or conv_weight.shape != (qkv_dim, conv_width)
        or conv_bias.shape != (qkv_dim,)
        or conv_state.shape[1:] != (qkv_dim, conv_state_len)
        or dt_bias.shape != (hv,)
        or ssm_state.shape[1:] != (hv, d, d)
        or state_indices.shape != (B,)
    ):
        return None

    if (
        hidden_states.dtype != torch.bfloat16
        or w_ba.dtype != torch.bfloat16
        or mixed_qkv.dtype != torch.bfloat16
        or conv_weight.dtype != torch.bfloat16
        or conv_bias.dtype != torch.bfloat16
        or conv_state.dtype != torch.bfloat16
        or A_log.dtype != torch.float32
        or dt_bias.dtype != torch.bfloat16
        or ssm_state.dtype != torch.float32
        or state_indices.dtype != torch.int32
    ):
        return None

    # Dense inner layout; fp32 state pool rows may be stride-padded and
    # mixed_qkv rows may be strided (a view into a wider fused projection).
    if (
        hidden_states.stride(1) != 1
        or hidden_states.stride(0) != hidden
        or mixed_qkv.stride(1) != 1
        or mixed_qkv.stride(0) < qkv_dim
        or not w_ba.is_contiguous()
        or not conv_weight.is_contiguous()
        or not conv_bias.is_contiguous()
        or not A_log.is_contiguous()
        or not dt_bias.is_contiguous()
        or not state_indices.is_contiguous()
    ):
        return None
    # conv_state must be one of the two recognized pool layouts (SD arrives
    # as the transposed view of the physical pool); the page stride may be
    # padded -- the kernels consume the strides directly.
    conv_layout = conv_state_layout(conv_state)
    if conv_layout is None or conv_state.stride(0) < qkv_dim * conv_state_len:
        return None
    if (
        ssm_state.stride(3) != 1
        or ssm_state.stride(2) != d
        or ssm_state.stride(1) != d * d
        or ssm_state.stride(0) < hv * d * d
    ):
        return None

    # Optional destination: the specialized kernels write it directly, so it
    # must be a dense [B, 1, HV, D] bf16 buffer on the same device.
    if out is not None and (
        out.dtype != torch.bfloat16
        or out.shape != (B, 1, hv, d)
        or out.device != hidden_states.device
        or not out.is_contiguous()
    ):
        return None

    return {
        "b": int(B),
        "hidden": int(hidden),
        "n_ba": int(n_ba),
        "qkv_dim": int(qkv_dim),
        "h_q": int(h_q),
        "hv": int(hv),
        "d": int(d),
        "conv_width": int(conv_width),
        "conv_state_len": int(conv_state_len),
        "conv_layout": conv_layout,
    }


def gdn_fused_decode_probe(signature: dict, device) -> bool:
    """Routing-probe backend: True when an importable impl serves the
    signature on this device (registry hit + loadable impl module + no
    failure latch).  Host-side only, no memo -- callers on the serving path
    use :func:`gdn_fused_decode_supported_geometry`."""
    for row in match_gdn_fused_decode_signature(signature, device):
        if row["impl"] in _failed_impls:
            continue
        if _load_impl(row["impl"]) is not None:
            return True
    return False


def gdn_fused_decode_supported_geometry(
    batch_size: int,
    hidden_size: int,
    n_ba: int,
    qkv_dim: int,
    num_qk_heads: int,
    num_v_heads: int,
    head_dim: int,
    conv_width: int,
    conv_state_len: int,
    conv_layout: str,
    device=None,
) -> bool:
    """Memoized routing probe over a scalar geometry (the serving path).

    Backs :func:`flashinfer.gdn_fused_decode_step_supported`.  A
    framework asks this once per layer per decode step with the same
    arguments, outside the CUDA graph, so the repeat cost must be a dict
    lookup and nothing else: the first call for a geometry resolves the
    device capability, indexes the registry and imports the winning impl
    module; later calls read the memo.  The memo is dropped whenever the
    registry object changes (see :func:`_registry_index`) or an impl is
    latched off by a kernel failure (see :func:`_latch_impl_off`), so it can
    never report a surface that has gone away.
    """
    cc = _device_cc(device)
    if cc is None:
        return False
    key = (
        cc,
        batch_size,
        hidden_size,
        n_ba,
        qkv_dim,
        num_qk_heads,
        num_v_heads,
        head_dim,
        conv_width,
        conv_state_len,
        conv_layout,
    )
    # _registry_index() is what invalidates the memo when the registry is
    # substituted, so it must run before the memo is consulted.  Both are
    # dict lookups once warm.
    _registry_index()
    answer = _probe_memo.get(key)
    if answer is None:
        signature = signature_from_geometry(
            batch_size,
            hidden_size,
            n_ba,
            qkv_dim,
            num_qk_heads,
            num_v_heads,
            head_dim,
            conv_width,
            conv_state_len,
            conv_layout,
        )
        answer = signature is not None and gdn_fused_decode_probe(signature, device)
        _probe_memo[key] = answer
    return answer


def _impl_for_backend(backend: str, rows: List[dict]):
    """(impl_module, row) of the internal family ``backend`` among matched
    rows, else (None, None)."""
    impl_names = BACKEND_IMPLS.get(backend, ())
    for row in rows:
        if row["impl"] in impl_names:
            return _load_impl(row["impl"]), row
    return None, None


def try_run_gdn_fused_decode_specialized(
    hidden_states: torch.Tensor,
    w_ba: torch.Tensor,
    mixed_qkv: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    conv_state: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    ssm_state: torch.Tensor,
    state_indices: torch.Tensor,
    use_qk_l2norm: bool,
    out: Optional[torch.Tensor] = None,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """The only specialized entry point; None means "run the composable path".

    The internal impl families are tried in preference order (``cute_dsl``,
    then ``cuda``).  A kernel failure warns once, latches that impl off for
    the rest of the process, and moves on -- eventually letting the
    (capture-safe) composable path serve the call.  Nothing here can make
    the op raise: an unregistered signature, an unloadable impl, a
    capture-unready variant and a kernel exception all return ``None``.

    CUDA-graph contract: never compiles during capture -- a backend whose
    exact variant is not yet compiled and warmed is skipped, falling
    through to the next backend and finally to the composable path (which
    then gets baked for that shape).  Outside capture the first dispatch of
    a variant compiles it (vLLM's eager profile run precedes its capture
    phase).
    """
    signature = signature_from_tensors(
        hidden_states,
        w_ba,
        mixed_qkv,
        conv_weight,
        conv_bias,
        conv_state,
        A_log,
        dt_bias,
        ssm_state,
        state_indices,
        use_qk_l2norm,
        out,
    )
    if signature is None:
        return None
    rows = match_gdn_fused_decode_signature(signature, hidden_states.device)
    if not rows:
        return None
    # Everything below runs with the tensors' device current.  The impls take
    # the launch stream from the *ambient* device (that is what
    # ``torch.cuda.current_stream()`` and tvm_ffi's stream hook both read), so
    # a call made on a non-ambient device -- e.g. a TP rank driving cuda:1
    # from a thread whose current device is cuda:0 -- would otherwise pick up
    # another device's stream.  ``is_current_stream_capturing()`` reads the
    # ambient device too, so it is inside the block as well.  Cost when the
    # device is already current is one C-level device exchange, which is why
    # this wraps the whole dispatch rather than each impl call.
    with torch.cuda.device(hidden_states.device):
        capturing = torch.cuda.is_current_stream_capturing()
        skipped_unready = False
        for backend in _AUTO_BACKEND_ORDER:
            impl, row = _impl_for_backend(backend, rows)
            if row is None or impl is None or row["impl"] in _failed_impls:
                continue
            if capturing and not impl.ready_for_graph_capture(
                signature, hidden_states, conv_state, scale
            ):
                skipped_unready = True
                continue
            try:
                result = impl.execute(
                    hidden_states,
                    w_ba,
                    mixed_qkv,
                    conv_weight,
                    conv_bias,
                    conv_state,
                    A_log,
                    dt_bias,
                    scale,
                    ssm_state,
                    state_indices,
                    out=out,
                )
            except Exception as exc:
                # A kernel failure on the auto path must never break the
                # composable path; latch this impl off for the process.
                #
                # Say so loudly enough that a measurement run cannot mistake
                # it for a cosmetic warning: from here on a DIFFERENT kernel
                # serves these calls, every gate downstream stays green, and
                # any number produced after this line describes that other
                # kernel.
                _latch_impl_off(row["impl"])
                jit_logger.warning_once(
                    "Specialized fused GDN decode impl '%s' failed (%s); not "
                    "dispatching it again in this process. A different "
                    "implementation now serves these calls -- any measurement "
                    "taken from here on describes that one, not '%s'.",
                    row["impl"],
                    type(exc).__name__,
                    row["impl"],
                )
            else:
                _attest_served(row["impl"])
                return result
        if capturing and skipped_unready:
            jit_logger.warning_once(
                "CUDA-graph capture hit a registered gdn_fused_decode_step "
                "signature before any specialized backend was compiled and "
                "warmed; capturing the composable path instead. Run one eager "
                "call per (layer geometry, batch size, scale, conv-state "
                "layout) before capture to enable the specialized kernels."
            )
    return None


def gdn_fused_decode_stats() -> dict:
    """JSON-serializable snapshot of the specialized fused-GDN-decode state.

    Intended for benchmark/integration harnesses.  Top level:

    * ``registry_entries``: total registered rows (a signature counts once
      per capable impl).
    * ``probe_memo_entries``: distinct geometries the routing probe has
      answered since the last invalidation.
    * ``failed_impls``: impls latched off by an auto-path kernel failure.
    * ``served_impls``: impls that actually served at least one call.  This
      is the attestation a measurement run should record: ``failed_impls``
      non-empty, or ``served_impls`` naming an impl other than the expected
      one, means the numbers describe a different kernel than the registry's
      preferred choice.
    * ``impls``: per-impl compile/launch introspection --
      ``compiled_variants``/``variant_keys`` (kernels compiled so far),
      ``distinct_kernels_for_registry`` (how many compiled kernels cover
      this impl's rows; None when the impl cannot load), and
      ``launch_count`` (a CUDA-graph capture counts once; replays do not
      re-run host code).
    """
    rows = load_gdn_fused_decode_registry()
    stats: dict = {
        "registry_entries": len(rows),
        "probe_memo_entries": len(_probe_memo),
        "failed_impls": sorted(_failed_impls),
        "served_impls": sorted(_served_impls),
        "impls": {},
    }
    for impl_name in sorted({row["impl"] for row in rows}):
        impl_rows = _rows_for_impl(impl_name)
        entry: dict = {
            "registry_entries": len(impl_rows),
            "compiled_variants": 0,
            "variant_keys": [],
            "distinct_kernels_for_registry": None,
            "launch_count": 0,
        }
        impl = _load_impl(impl_name)
        if impl is not None:
            keys = impl.compiled_variant_keys()
            entry["compiled_variants"] = len(keys)
            entry["variant_keys"] = keys
            entry["distinct_kernels_for_registry"] = len(impl.variant_plan(impl_rows))
            entry["launch_count"] = impl.launch_count()
        stats["impls"][impl_name] = entry
    return stats


__all__ = [
    "BACKEND_IMPLS",
    "conv_state_layout",
    "gdn_fused_decode_probe",
    "gdn_fused_decode_stats",
    "gdn_fused_decode_supported_geometry",
    "load_gdn_fused_decode_registry",
    "match_gdn_fused_decode_signature",
    "registry_geometries",
    "signature_from_geometry",
    "signature_from_tensors",
    "try_run_gdn_fused_decode_specialized",
]
