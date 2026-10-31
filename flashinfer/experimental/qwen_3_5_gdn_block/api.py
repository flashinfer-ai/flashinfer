# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The backend registry, the plans and the functional entries of the Qwen 3.5 GDN block.

A backend is a ``(support_fn, build_fn)`` pair registered under a name (:func:`register_backend`); ``"frost"`` is
registered here. A :class:`Qwen35GdnPlan` binds one backend to one step shape -- the :class:`Qwen35GdnPlanSpec`:
backend, form, requests, tokens per request, geometry, device and the dtypes of ``A_log`` / ``dt_bias`` /
``norm_weight`` -- and owns the backend's runner and a default FP8 output. ``prepare`` builds a plan outside
CUDA-graph capture; ``plan.run`` checks one call's tensors against the spec (:func:`bind_inputs`), lets the runner
specialize for them when eager, and launches. The functional entries keep an LRU of plans keyed on the spec; a plan a
CUDA graph was captured over outlives its eviction (the graph baked its buffers) until :func:`clear_plan_cache`.

Nothing here imports a kernel: a backend's ``build_fn`` does, on first use.
"""

from __future__ import annotations

import collections
import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Protocol, Tuple

import torch

from .backends.frost import admission as _frost_admission
from .support import Qwen35GdnDecline, Qwen35GdnGeometry, require_served

__all__ = [
    "DEFAULT_PARAM_DTYPES",
    "PLAN_CACHE_SIZE",
    "Qwen35GdnBackend",
    "Qwen35GdnInputs",
    "Qwen35GdnPlan",
    "Qwen35GdnPlanSpec",
    "Qwen35GdnRunner",
    "bind_inputs",
    "cached_plan",
    "clear_plan_cache",
    "decode",
    "get_backend",
    "prepare",
    "register_backend",
    "registered_backends",
    "unregister_backend",
    "verify",
]

ParamDtypes = Tuple[torch.dtype, torch.dtype, torch.dtype]
#: the dtypes of (A_log, dt_bias, norm_weight) a plan is built for when none are given (vLLM's Qwen3.5 layer: an FP32
#: A_log, the rest in the model dtype)
DEFAULT_PARAM_DTYPES: ParamDtypes = (torch.float32, torch.bfloat16, torch.bfloat16)
_PARAM_DTYPE_CHOICES = (torch.float32, torch.bfloat16)
_ALIGN_BYTES = 16


# --------------------------------------------------------------------------- the backend interface


@dataclass(frozen=True)
class Qwen35GdnPlanSpec:
    """Everything a plan is built for; the plan-cache key. ``tokens_per_request`` is 1 for ``decode``."""

    backend: str
    form: str
    requests: int
    tokens_per_request: int
    geometry: Qwen35GdnGeometry
    device: torch.device
    param_dtypes: ParamDtypes = DEFAULT_PARAM_DTYPES

    @property
    def rows(self) -> int:
        """Token rows of one call: ``requests * tokens_per_request``."""
        return self.requests * self.tokens_per_request


@dataclass(frozen=True)
class Qwen35GdnInputs:
    """One call's tensors after :func:`bind_inputs` checked them against the plan's spec: what a runner reads.

    ``out`` is always a tensor (the caller's or the plan's); ``out_bf16`` is ``None`` unless the caller asked for the
    BF16 normed value. ``cu_seqlens`` / ``num_accepted_tokens`` are set on ``verify`` only.
    """

    qkv: torch.Tensor
    z: torch.Tensor
    hidden: torch.Tensor
    ba_weight: torch.Tensor
    A_log: torch.Tensor
    dt_bias: torch.Tensor
    ssm_state: torch.Tensor
    state_indices: torch.Tensor
    norm_weight: torch.Tensor
    out_scale: torch.Tensor
    conv_state: torch.Tensor
    conv_weight: torch.Tensor
    out: torch.Tensor
    out_bf16: Optional[torch.Tensor] = None
    cu_seqlens: Optional[torch.Tensor] = None
    num_accepted_tokens: Optional[torch.Tensor] = None


class Qwen35GdnRunner(Protocol):
    """What a backend's ``build_fn`` returns for one plan.

    ``prepare(inputs)`` runs before every eager ``run`` and never under CUDA-graph capture: compile or specialize for
    this binding (strides, pointers' alignment); cheap when already done. It raises ``NotImplementedError`` for a
    binding the backend does not serve. ``run(inputs)`` launches on the current stream, writes ``inputs.out`` (and
    ``inputs.out_bf16`` when set) and updates ``conv_state`` / ``ssm_state`` in place; it must be capturable once
    ``prepare`` saw an equivalent binding, and raise ``RuntimeError`` under capture otherwise.
    """

    def prepare(self, inputs: Qwen35GdnInputs) -> None: ...

    def run(self, inputs: Qwen35GdnInputs) -> None: ...


SupportFn = Callable[
    [torch.device, Qwen35GdnGeometry, str, Optional[int], Optional[int]],
    Optional[Qwen35GdnDecline],
]
BuildFn = Callable[[Qwen35GdnPlanSpec], Qwen35GdnRunner]


@dataclass(frozen=True)
class Qwen35GdnBackend:
    """A registered backend.

    ``support_fn(device, geometry, form, requests, tokens_per_request)`` returns ``None`` when served, else a
    :class:`~flashinfer.experimental.qwen_3_5_gdn_block.support.Qwen35GdnDecline`; ``requests`` /
    ``tokens_per_request`` are ``None`` when the caller asks without a step shape. It must be cheap, must not import
    kernels, and should not raise. ``build_fn(spec)`` returns the plan's :class:`Qwen35GdnRunner`; it runs outside
    CUDA-graph capture, after ``support_fn`` admitted the spec, and may compile and allocate.
    """

    name: str
    support_fn: SupportFn
    build_fn: BuildFn


_BACKENDS: Dict[str, Qwen35GdnBackend] = {}
_BACKENDS_LOCK = threading.Lock()


def register_backend(
    name: str, support_fn: SupportFn, build_fn: BuildFn, *, replace: bool = False
) -> None:
    """Register a backend under ``name`` (see :class:`Qwen35GdnBackend` for the two callables' contract).

    Re-registering a name needs ``replace=True``, which also drops that backend's cached plans."""
    if not isinstance(name, str) or not name:
        raise ValueError(f"backend name must be a non-empty str, got {name!r}")
    if not callable(support_fn) or not callable(build_fn):
        raise ValueError("support_fn and build_fn must be callable")
    with _BACKENDS_LOCK:
        if name in _BACKENDS and not replace:
            raise ValueError(
                f"a Qwen 3.5 GDN block backend named {name!r} is already registered; pass replace=True"
            )
        _BACKENDS[name] = Qwen35GdnBackend(name, support_fn, build_fn)
    _drop_backend_plans(name)


def unregister_backend(name: str) -> None:
    """Remove a backend and drop its cached plans (``KeyError`` if it is not registered)."""
    with _BACKENDS_LOCK:
        del _BACKENDS[name]
    _drop_backend_plans(name)


def registered_backends() -> Tuple[str, ...]:
    with _BACKENDS_LOCK:
        return tuple(_BACKENDS)


def get_backend(name: str) -> Optional[Qwen35GdnBackend]:
    with _BACKENDS_LOCK:
        return _BACKENDS.get(name)


# --------------------------------------------------------------------------- argument checks (host-side, no device read)


def _capturing() -> bool:
    return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())


def _device_of(device) -> torch.device:
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    return dev


def _fail(message: str):
    raise ValueError(f"qwen_3_5_gdn_block: {message}")


def _tensor(
    x,
    name: str,
    shape: Tuple[Optional[int], ...],
    dtypes: Tuple[torch.dtype, ...],
    device: torch.device,
    *,
    contiguous: bool = True,
) -> torch.Tensor:
    """``x`` as a tensor of ``shape`` (``None`` = any extent) with a dtype in ``dtypes`` on ``device``, 16-byte aligned."""
    if not isinstance(x, torch.Tensor):
        _fail(f"{name} must be a torch.Tensor, got {type(x).__name__}")
    want = "[" + ", ".join("*" if s is None else str(s) for s in shape) + "]"
    if x.dim() != len(shape) or any(
        s is not None and int(d) != s for d, s in zip(x.shape, shape, strict=True)
    ):
        _fail(f"{name} must have shape {want}, got {list(x.shape)}")
    if x.dtype not in dtypes:
        _fail(f"{name} must be {' or '.join(str(d) for d in dtypes)}, got {x.dtype}")
    if x.device != device:
        _fail(f"{name} must live on {device} (the plan's device), got {x.device}")
    if contiguous and not x.is_contiguous():
        _fail(f"{name} must be contiguous, got strides {list(x.stride())}")
    if x.numel() > 1 and x.data_ptr() % _ALIGN_BYTES:
        _fail(
            f"{name} must be {_ALIGN_BYTES}-byte aligned, got data_ptr {x.data_ptr():#x}"
        )
    return x


def bind_inputs(
    spec: Qwen35GdnPlanSpec,
    *,
    qkv,
    z,
    hidden,
    ba_weight,
    A_log,
    dt_bias,
    ssm_state,
    state_indices,
    norm_weight,
    out_scale,
    conv_state,
    conv_weight,
    out,
    cu_seqlens=None,
    num_accepted_tokens=None,
    out_bf16=None,
) -> Qwen35GdnInputs:
    """Check one call's tensors against ``spec`` (typed ``ValueError`` naming the operand); see the README for the
    layouts. Device-side values (slot ids, offsets, accepted counts) are not read."""
    g, dev = spec.geometry, spec.device
    rows, hv, dv, dk = spec.rows, g.num_v_heads, g.head_v_dim, g.head_k_dim
    bf16, f32 = (torch.bfloat16,), (torch.float32,)
    qkv = _tensor(qkv, "qkv", (rows, g.conv_dim), bf16, dev, contiguous=False)
    if qkv.stride(1) != 1 or (qkv.stride(0) * 2) % _ALIGN_BYTES:
        _fail(
            f"qkv's channel axis must be contiguous and its row stride a {_ALIGN_BYTES}-byte multiple, got strides "
            f"{list(qkv.stride())}"
        )
    z = _tensor(z, "z", (rows, hv, dv), bf16, dev, contiguous=False)
    if z.stride(2) != 1 or z.stride(1) != dv or (z.stride(0) * 2) % _ALIGN_BYTES:
        _fail(
            f"z must be [rows, {hv}, {dv}] with contiguous heads and a {_ALIGN_BYTES}-byte multiple row stride, got "
            f"strides {list(z.stride())}"
        )
    hidden = _tensor(hidden, "hidden", (rows, g.d_model), bf16, dev)
    ba_weight = _tensor(ba_weight, "ba_weight", (2 * hv, g.d_model), bf16, dev)
    a_dt, dt_dt, w_dt = spec.param_dtypes
    A_log = _tensor(A_log, "A_log", (hv,), (a_dt,), dev)
    dt_bias = _tensor(dt_bias, "dt_bias", (hv,), (dt_dt,), dev)
    norm_weight = _tensor(norm_weight, "norm_weight", (dv,), (w_dt,), dev)
    if not isinstance(out_scale, torch.Tensor) or out_scale.numel() != 1:
        _fail(
            "out_scale must be a one-element float32 tensor (the static FP8 output scale)"
        )
    out_scale = _tensor(out_scale, "out_scale", tuple(out_scale.shape), f32, dev)
    conv_weight = _tensor(
        conv_weight, "conv_weight", (g.conv_dim, g.conv_kernel), bf16, dev
    )
    state_len = g.conv_state_len(spec.tokens_per_request)
    conv_state = _tensor(
        conv_state,
        "conv_state",
        (None, g.conv_dim, state_len),
        bf16,
        dev,
        contiguous=False,
    )
    ssm_state = _tensor(
        ssm_state, "ssm_state", (None, hv, dv, dk), f32, dev, contiguous=False
    )
    if (
        tuple(ssm_state.stride())[1:] != (dv * dk, dk, 1)
        or (ssm_state.stride(0) * 4) % _ALIGN_BYTES
    ):
        _fail(
            f"ssm_state must be [num_slots, {hv}, {dv}, {dk}] with each slot contiguous and a {_ALIGN_BYTES}-byte "
            f"multiple slot stride, got strides {list(ssm_state.stride())}"
        )
    i32 = (torch.int32,)
    if spec.form == "decode":
        state_indices = _tensor(state_indices, "state_indices", (rows,), i32, dev)
        if cu_seqlens is not None or num_accepted_tokens is not None:
            _fail("cu_seqlens / num_accepted_tokens belong to the verify form")
    else:
        state_indices = _tensor(
            state_indices,
            "state_indices",
            (spec.requests, spec.tokens_per_request),
            i32,
            dev,
            contiguous=False,
        )
        if state_indices.stride(1) != 1:
            _fail(
                f"state_indices' token axis must be contiguous, got strides {list(state_indices.stride())}"
            )
        cu_seqlens = _tensor(cu_seqlens, "cu_seqlens", (spec.requests + 1,), i32, dev)
        num_accepted_tokens = _tensor(
            num_accepted_tokens, "num_accepted_tokens", (spec.requests,), i32, dev
        )
    out = _tensor(out, "out", (rows, hv, dv), (torch.float8_e4m3fn,), dev)
    if out_bf16 is not None:
        out_bf16 = _tensor(out_bf16, "out_bf16", (rows, hv, dv), bf16, dev)
    return Qwen35GdnInputs(
        qkv=qkv,
        z=z,
        hidden=hidden,
        ba_weight=ba_weight,
        A_log=A_log,
        dt_bias=dt_bias,
        ssm_state=ssm_state,
        state_indices=state_indices,
        norm_weight=norm_weight,
        out_scale=out_scale,
        conv_state=conv_state,
        conv_weight=conv_weight,
        out=out,
        out_bf16=out_bf16,
        cu_seqlens=cu_seqlens,
        num_accepted_tokens=num_accepted_tokens,
    )


# --------------------------------------------------------------------------- the plan


class Qwen35GdnPlan:
    """One backend bound to one step shape (:attr:`spec`). Built by :func:`prepare` (or the functional entries' LRU).

    Runs of one plan must be ordered on one stream: the plan's default output and the runner's buffers (FROST: the
    per-request conv-reader tickets) are reused by every run.
    """

    def __init__(self, spec: Qwen35GdnPlanSpec, runner: Qwen35GdnRunner):
        self.spec = spec
        self._runner = runner
        g = spec.geometry
        self._out = torch.empty(
            spec.rows,
            g.num_v_heads,
            g.head_v_dim,
            dtype=torch.float8_e4m3fn,
            device=spec.device,
        )
        #: set once a CUDA graph was captured over this plan; the LRU then retains it past eviction
        self._captured = False

    def __repr__(self) -> str:
        s = self.spec
        return (
            f"Qwen35GdnPlan(backend={s.backend!r}, form={s.form!r}, requests={s.requests}, "
            f"tokens_per_request={s.tokens_per_request}, device={s.device})"
        )

    def run(
        self,
        *,
        qkv: torch.Tensor,
        z: torch.Tensor,
        hidden: torch.Tensor,
        ba_weight: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_state: torch.Tensor,
        state_indices: torch.Tensor,
        norm_weight: torch.Tensor,
        out_scale: torch.Tensor,
        conv_state: torch.Tensor,
        conv_weight: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        num_accepted_tokens: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        out_bf16: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run one step on the current stream; returns the FP8 output (``out``, or the plan's own buffer, overwritten
        by the next run). The plan's device must be the current CUDA device."""
        dev = self.spec.device
        if dev.type == "cuda" and torch.cuda.current_device() != dev.index:
            _fail(
                f"the plan runs on {dev}, but the current CUDA device is cuda:{torch.cuda.current_device()}; call "
                f"under torch.cuda.device({dev.index})"
            )
        inputs = bind_inputs(
            self.spec,
            qkv=qkv,
            z=z,
            hidden=hidden,
            ba_weight=ba_weight,
            A_log=A_log,
            dt_bias=dt_bias,
            ssm_state=ssm_state,
            state_indices=state_indices,
            norm_weight=norm_weight,
            out_scale=out_scale,
            conv_state=conv_state,
            conv_weight=conv_weight,
            cu_seqlens=cu_seqlens,
            num_accepted_tokens=num_accepted_tokens,
            out=self._out if out is None else out,
            out_bf16=out_bf16,
        )
        if _capturing():
            self._captured = True
        else:
            self._runner.prepare(inputs)
        self._runner.run(inputs)
        return inputs.out


def _check_not_capturing(what: str) -> None:
    if _capturing():
        raise RuntimeError(
            f"qwen_3_5_gdn_block: {what} builds a plan (compiles, allocates) and cannot run under CUDA-graph "
            "capture; prepare outside the capture, run once eagerly, then capture plan.run"
        )


def _param_dtypes(param_dtypes) -> ParamDtypes:
    dtypes = tuple(param_dtypes)
    if len(dtypes) != 3 or any(d not in _PARAM_DTYPE_CHOICES for d in dtypes):
        _fail(
            "param_dtypes must be the (A_log, dt_bias, norm_weight) dtypes, each torch.float32 or torch.bfloat16; "
            f"got {param_dtypes!r}"
        )
    return dtypes  # type: ignore[return-value]


def prepare(
    form: str,
    *,
    geometry: Qwen35GdnGeometry,
    device,
    requests: int,
    tokens_per_request: int = 1,
    backend: str = "frost",
    param_dtypes: ParamDtypes = DEFAULT_PARAM_DTYPES,
) -> Qwen35GdnPlan:
    """Build a plan outside CUDA-graph capture (not cached). ``ValueError`` for a malformed argument,
    ``NotImplementedError`` naming what the backend does not serve, ``RuntimeError`` under capture."""
    _check_not_capturing("prepare")
    dev = _device_of(device)
    dtypes = _param_dtypes(param_dtypes)
    require_served(
        dev,
        geometry,
        form,
        backend,
        requests=requests,
        tokens_per_request=tokens_per_request,
    )
    spec = Qwen35GdnPlanSpec(
        backend, form, int(requests), int(tokens_per_request), geometry, dev, dtypes
    )
    entry = get_backend(backend)
    assert entry is not None  # require_served declined an unregistered name
    return Qwen35GdnPlan(spec, entry.build_fn(spec))


# --------------------------------------------------------------------------- the functional entries' plan cache

_PLANS: "collections.OrderedDict[Qwen35GdnPlanSpec, Qwen35GdnPlan]" = (
    collections.OrderedDict()
)
_RETAINED: List[Qwen35GdnPlan] = []
_PLANS_LOCK = threading.Lock()
PLAN_CACHE_SIZE = 32


def clear_plan_cache() -> None:
    """Drop every cached plan, including the ones CUDA graphs were captured over. Call it only once no such graph can
    replay (after ``graph.reset()``, at teardown); refused under capture."""
    if _capturing():
        raise RuntimeError(
            "qwen_3_5_gdn_block: clear_plan_cache() frees buffers captured graphs replay into; refused under capture"
        )
    with _PLANS_LOCK:
        _PLANS.clear()
        _RETAINED.clear()


def _drop_backend_plans(name: str) -> None:
    with _PLANS_LOCK:
        for spec in [s for s in _PLANS if s.backend == name]:
            plan = _PLANS.pop(spec)
            if plan._captured:
                _RETAINED.append(plan)


def cached_plan(
    form: str,
    *,
    geometry: Qwen35GdnGeometry,
    device,
    requests: int,
    tokens_per_request: int = 1,
    backend: str = "frost",
    param_dtypes: ParamDtypes = DEFAULT_PARAM_DTYPES,
) -> Qwen35GdnPlan:
    """The functional entries' plan: an LRU of :data:`PLAN_CACHE_SIZE` plans keyed on the spec. Under CUDA-graph capture
    only a cached plan is served (and then retained past its eviction); a cold one raises ``RuntimeError``."""
    dev = _device_of(device)
    capturing = _capturing()
    try:
        spec = Qwen35GdnPlanSpec(
            backend,
            form,
            requests,
            tokens_per_request,
            geometry,
            dev,
            tuple(param_dtypes),  # type: ignore[arg-type]
        )
        with _PLANS_LOCK:
            plan = _PLANS.get(spec)
            if plan is not None:
                _PLANS.move_to_end(spec)
                if capturing:
                    plan._captured = True
                return plan
    except TypeError:  # an unhashable argument: prepare names it
        pass
    if capturing:
        raise RuntimeError(
            "qwen_3_5_gdn_block: this step shape has no cached plan and CUDA-graph capture cannot build one; run the "
            "same call once eagerly before capturing"
        )
    plan = prepare(
        form,
        geometry=geometry,
        device=dev,
        requests=requests,
        tokens_per_request=tokens_per_request,
        backend=backend,
        param_dtypes=param_dtypes,
    )
    with _PLANS_LOCK:
        cached = _PLANS.get(plan.spec)
        if cached is not None:
            return cached
        _PLANS[plan.spec] = plan
        while len(_PLANS) > PLAN_CACHE_SIZE:
            _, evicted = _PLANS.popitem(last=False)
            if evicted._captured:
                _RETAINED.append(evicted)
    return plan


def _resolve_plan(
    plan: Optional[Qwen35GdnPlan],
    form: str,
    geometry,
    backend: Optional[str],
    device,
    requests: int,
    tokens_per_request: int,
    param_dtypes,
) -> Qwen35GdnPlan:
    if plan is None:
        return cached_plan(
            form,
            geometry=geometry,
            device=device,
            requests=requests,
            tokens_per_request=tokens_per_request,
            backend="frost" if backend is None else backend,
            param_dtypes=param_dtypes,
        )
    if not isinstance(plan, Qwen35GdnPlan):
        _fail(f"plan must be a Qwen35GdnPlan, got {type(plan).__name__}")
    s = plan.spec
    if s.form != form or s.geometry != geometry or (backend not in (None, s.backend)):
        _fail(
            f"the plan was built for backend={s.backend!r}, form={s.form!r}, geometry={s.geometry}; the call asks "
            f"for backend={backend!r}, form={form!r}, geometry={geometry}"
        )
    return plan


def _dtype_of(x, name: str) -> torch.dtype:
    if not isinstance(x, torch.Tensor):
        _fail(f"{name} must be a torch.Tensor, got {type(x).__name__}")
    return x.dtype


def decode(
    qkv,
    z,
    hidden,
    ba_weight,
    A_log,
    dt_bias,
    ssm_state,
    state_indices,
    norm_weight,
    out_scale,
    conv_state,
    conv_weight,
    *,
    geometry,
    backend=None,
    out=None,
    out_bf16=None,
    plan=None,
):
    """See :func:`flashinfer.qwen_3_5_gdn_block.qwen_3_5_gdn_block_decode`."""
    if not isinstance(qkv, torch.Tensor) or qkv.dim() != 2:
        _fail("qkv must be a 2-D [rows, conv_dim] tensor")
    dtypes = (
        _dtype_of(A_log, "A_log"),
        _dtype_of(dt_bias, "dt_bias"),
        _dtype_of(norm_weight, "norm_weight"),
    )
    plan = _resolve_plan(
        plan, "decode", geometry, backend, qkv.device, int(qkv.shape[0]), 1, dtypes
    )
    return plan.run(
        qkv=qkv,
        z=z,
        hidden=hidden,
        ba_weight=ba_weight,
        A_log=A_log,
        dt_bias=dt_bias,
        ssm_state=ssm_state,
        state_indices=state_indices,
        norm_weight=norm_weight,
        out_scale=out_scale,
        conv_state=conv_state,
        conv_weight=conv_weight,
        out=out,
        out_bf16=out_bf16,
    )


def verify(
    qkv,
    z,
    hidden,
    ba_weight,
    A_log,
    dt_bias,
    ssm_state,
    state_indices,
    cu_seqlens,
    num_accepted_tokens,
    norm_weight,
    out_scale,
    conv_state,
    conv_weight,
    *,
    geometry,
    backend=None,
    out=None,
    out_bf16=None,
    plan=None,
):
    """See :func:`flashinfer.qwen_3_5_gdn_block.qwen_3_5_gdn_block_verify`."""
    if not isinstance(state_indices, torch.Tensor) or state_indices.dim() != 2:
        _fail("state_indices must be a 2-D [requests, tokens_per_request] int32 tensor")
    requests, tokens = (int(d) for d in state_indices.shape)
    dtypes = (
        _dtype_of(A_log, "A_log"),
        _dtype_of(dt_bias, "dt_bias"),
        _dtype_of(norm_weight, "norm_weight"),
    )
    plan = _resolve_plan(
        plan,
        "verify",
        geometry,
        backend,
        state_indices.device,
        requests,
        tokens,
        dtypes,
    )
    return plan.run(
        qkv=qkv,
        z=z,
        hidden=hidden,
        ba_weight=ba_weight,
        A_log=A_log,
        dt_bias=dt_bias,
        ssm_state=ssm_state,
        state_indices=state_indices,
        cu_seqlens=cu_seqlens,
        num_accepted_tokens=num_accepted_tokens,
        norm_weight=norm_weight,
        out_scale=out_scale,
        conv_state=conv_state,
        conv_weight=conv_weight,
        out=out,
        out_bf16=out_bf16,
    )


# --------------------------------------------------------------------------- the built-in backend


def _build_frost(spec: Qwen35GdnPlanSpec) -> Qwen35GdnRunner:
    from .backends.frost.runner import build

    return build(spec)


register_backend("frost", _frost_admission.support, _build_frost)
