"""CUDA (tvm_ffi) fast path for the GLM mHC PRE (hyper-connections) op at the
GLM-5.3 geometry: HC=4, H=4096, 24 projection rows, sinkhorn_repeat=20 and all
epsilons 1e-6. It computes the FULL op (projection GEMM included) in one fused
launch sequence and writes ``post_mix``/``comb``/``layer_input`` directly.

Scope: ``residual`` of shape ``(T, 4, 4096)`` bfloat16 with ``1 <= T <= 16384``
(the validated range) on SM103; everything else stays on the composable
big-fuse path. Kill switch: FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1.

Numerics contract: the fused projection runs on tensor cores with
reduced-precision operands (the fp32 ``fn`` weights are converted to
bf16/tf32 depending on the token-count arm) and fp32 accumulation, and some
epilogues accumulate in bf16. The backend is validated against an fp32
reference at the audited tolerance (atol 4e-3, rtol 1e-2, matched ratio
>= 0.995, max-error cap 0.05) on GLM-5.3 workload statistics; fp32 weight
structure below bf16/tf32 resolution is not preserved. Callers that need
strict fp32 projection semantics should select ``backend="big-fuse"``.
Automatic selection inherits this reduced-precision contract. Fixed-shape
repeatability does not imply batch invariance: changing the token count can
change the kernel, operand precision, split-K count and accumulation
order, so a token's result can change across batches.

CUDA graphs: the route is taken under capture only after the module was built
eagerly (single fatbin, no per-shape JIT); dispatch is host-int based with no
syncs, and the kernels use programmatic dependent launch on the current
stream, which is capturable.

Generated-kernel provenance: agent-generated kernel (search-campaign winner),
source-vetted and re-timed per workload row on GB300 against both the
FlashInfer big-fuse composition and a TileLang study pin.
"""

import functools
import os as _os

import torch

from .jit.mhc_pre_glm import gen_mhc_pre_glm_module

_HC = 4
_H = 4096
_NF = _HC * (_HC + 2)  # 24 projection rows
_EPS = 1e-6
_SINK = 20
_MAX_TOKENS = 16384  # validated workload range upper bound

KILL_SWITCH_ENV = "FLASHINFER_SPECIALIZED_KERNEL_DISABLE"
DISPATCH_STATS = {"specialized": 0, "fallback": 0}
LAST_FALLBACK_REASONS: list = []

_BUILT = {"ready": False}

# A failed module load is remembered so "auto" does not re-run the build
# (ninja compile/link, or the immediate raise when JIT is disabled) on every
# later eligible call. An explicit backend="specialized" call always retries,
# so callers can recover after the toolchain or artifact is fixed; a retry
# that succeeds clears the marker for "auto" too.
_LOAD_FAILED = {"type": None}


@functools.cache
def get_mhc_pre_glm_module():
    mod = gen_mhc_pre_glm_module().build_and_load()
    _BUILT["ready"] = True
    _LOAD_FAILED["type"] = None
    return mod


def supported_with_reasons(
    residual,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    mhc_pre_eps,
    mhc_sinkhorn_eps,
    mhc_post_mult_value,
    sinkhorn_repeat,
    _ignore_residual_layout=False,
):
    """Guard order: GPU-free (kill switch, scalar-parameter pins, dtype,
    geometry allowlist) -> layout guards -> device guards. Returns [] when
    the specialized kernel can serve the call, else fallback reasons. No
    device syncs.

    ``_ignore_residual_layout=True`` skips the residual contiguity and
    alignment checks (both re-established by a ``.contiguous()`` copy):
    the dispatcher uses it to decide whether a non-contiguous residual is
    worth normalizing at all."""
    if _os.environ.get(KILL_SWITCH_ENV, "0") == "1":
        return ["kill_switch:" + KILL_SWITCH_ENV + "=1"]
    r = []
    # The kernel bakes in the GLM-5.3 scalar parameters.
    for name, ok in (
        ("rms_eps_pinned", rms_eps == _EPS),
        ("mhc_pre_eps_pinned", mhc_pre_eps == _EPS),
        ("mhc_sinkhorn_eps_pinned", mhc_sinkhorn_eps == _EPS),
        ("post_mult_pinned", mhc_post_mult_value == 1.0),
        ("sinkhorn_repeat_pinned", sinkhorn_repeat == _SINK),
    ):
        if not ok:
            r.append(name)
    if (
        residual.dim() != 3
        or residual.shape[1] != _HC
        or residual.shape[2] != _H
        or residual.dtype != torch.bfloat16
        or not (_ignore_residual_layout or residual.is_contiguous())
    ):
        r.append("residual_geometry")
    elif not (1 <= residual.shape[0] <= _MAX_TOKENS):
        r.append("tokens_out_of_validated_range")
    if (
        fn.dim() != 2
        or fn.shape[0] != _NF
        or fn.shape[1] != _HC * _H
        or fn.dtype != torch.float32
        or not fn.is_contiguous()
    ):
        r.append("fn_geometry")
    if (
        hc_scale.dim() != 1
        or hc_scale.shape[0] != 3
        or hc_scale.dtype != torch.float32
        or not hc_scale.is_contiguous()
    ):
        r.append("hc_scale_geometry")
    if (
        hc_base.dim() != 1
        or hc_base.shape[0] != _NF
        or hc_base.dtype != torch.float32
        or not hc_base.is_contiguous()
    ):
        r.append("hc_base_geometry")
    if r:
        return r
    if residual.device.type != "cuda" or any(
        t.device != residual.device for t in (fn, hc_scale, hc_base)
    ):
        return ["devices_not_same_cuda"]
    # the kernel's vectorized/TMA paths need 16-byte-aligned bases;
    # contiguity alone does not establish that (e.g. backing[1:].view(...))
    for _name, _t in (("residual", residual), ("fn", fn)):
        if _t.data_ptr() % 16 and not (_ignore_residual_layout and _name == "residual"):
            r.append(_name + "_not_16B_aligned")
    if r:
        return r
    if torch.cuda.get_device_capability(residual.device) != (10, 3):
        return ["not_sm103"]
    # The module load (and the launch) happen on residual's device, but the
    # capture query reads the CURRENT device's stream: check both, so a call
    # with residual on a non-current device cannot slip a module load into
    # that device's active capture (and a capture on the current device is
    # still refused even though the launch targets another GPU).
    if not _BUILT["ready"]:
        capturing = torch.cuda.is_current_stream_capturing()
        if (
            not capturing
            and residual.device.index is not None
            and (residual.device.index != torch.cuda.current_device())
        ):
            with torch.cuda.device(residual.device):
                capturing = torch.cuda.is_current_stream_capturing()
        if capturing:
            return ["module_not_built_before_graph_capture"]
    return []


def rejects_before_flatten(
    total_tokens,
    residual,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    mhc_pre_eps,
    mhc_sinkhorn_eps,
    mhc_post_mult_value,
    sinkhorn_repeat,
):
    """Layout-independent eligibility pre-screen for an "auto" call whose
    residual flatten would materialize a copy (ndim != 3 and non-contiguous).
    Runs the dispatcher's layout-ignoring checks against a zero-storage
    stand-in with the flattened shape, so an ineligible call (wrong scalar
    pins, geometry, dtype, device, ...) reaches the big-fuse fallback without
    paying for a flatten copy the fallback does not need. Returns True (and
    records the fallback) when the call is rejected regardless of layout."""
    probe = torch.empty_strided(
        (total_tokens,) + tuple(residual.shape[-2:]),
        (0, 0, 0),
        dtype=residual.dtype,
        device=residual.device,
    )
    reasons = supported_with_reasons(
        probe,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        mhc_pre_eps,
        mhc_sinkhorn_eps,
        mhc_post_mult_value,
        sinkhorn_repeat,
        _ignore_residual_layout=True,
    )
    if reasons:
        LAST_FALLBACK_REASONS[:] = reasons
        DISPATCH_STATS["fallback"] += 1
        return True
    return False


def maybe_dispatch_specialized(
    backend,
    residual,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    mhc_pre_eps,
    mhc_sinkhorn_eps,
    mhc_post_mult_value,
    sinkhorn_repeat,
):
    """Route for backend in ("auto", "specialized"). Returns the output tuple
    when served; None when "auto" should continue on the big-fuse composition.
    An explicitly requested "specialized" backend is never silently swapped:
    if it cannot serve the call it raises.

    A non-contiguous residual is normalized here, and only after the cheap
    layout-independent checks pass: a call the device or scalar pins exclude
    must not pay for a (potentially hundreds of MiB) contiguous copy the
    big-fuse fallback does not need."""
    if not residual.is_contiguous():
        reasons = supported_with_reasons(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            mhc_pre_eps,
            mhc_sinkhorn_eps,
            mhc_post_mult_value,
            sinkhorn_repeat,
            _ignore_residual_layout=True,
        )
        if reasons:
            LAST_FALLBACK_REASONS[:] = reasons
            if backend == "specialized":
                raise ValueError(
                    "backend='specialized' cannot serve this call and is never "
                    f"silently swapped; fallback_reasons={reasons}"
                )
            DISPATCH_STATS["fallback"] += 1
            return None
        residual = residual.contiguous()
    reasons = supported_with_reasons(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        mhc_pre_eps,
        mhc_sinkhorn_eps,
        mhc_post_mult_value,
        sinkhorn_repeat,
    )
    LAST_FALLBACK_REASONS[:] = reasons
    if not reasons:
        if backend != "specialized" and _LOAD_FAILED["type"] is not None:
            # The load already failed once; do not re-run the build on every
            # eligible "auto" call. backend="specialized" retries below.
            LAST_FALLBACK_REASONS[:] = [
                "specialized_module_unavailable:" + _LOAD_FAILED["type"]
            ]
            DISPATCH_STATS["fallback"] += 1
            return None
        try:
            get_mhc_pre_glm_module()
        except Exception as e:  # pragma: no cover - environment-dependent
            # A missing prebuilt JIT artifact (or a build failure) must not
            # take down "auto": record the reason and keep the big-fuse path.
            _LOAD_FAILED["type"] = type(e).__name__
            if backend == "specialized":
                raise
            LAST_FALLBACK_REASONS[:] = [
                "specialized_module_unavailable:" + type(e).__name__
            ]
            DISPATCH_STATS["fallback"] += 1
            return None
        DISPATCH_STATS["specialized"] += 1
        return _run_specialized(residual, fn, hc_scale, hc_base)
    if backend == "specialized":
        raise ValueError(
            "backend='specialized' cannot serve this call and is never "
            f"silently swapped; fallback_reasons={reasons}"
        )
    DISPATCH_STATS["fallback"] += 1
    return None


def _run_specialized(residual, fn, hc_scale, hc_base):
    tokens = residual.shape[0]
    post_mix = torch.empty(
        (tokens, _HC, 1), dtype=torch.float32, device=residual.device
    )
    comb = torch.empty((tokens, _HC, _HC), dtype=torch.float32, device=residual.device)
    layer_input = torch.empty(
        (tokens, _H), dtype=torch.bfloat16, device=residual.device
    )
    mod = get_mhc_pre_glm_module()
    mod.run(residual, fn, hc_scale, hc_base, post_mix, comb, layer_input)
    return post_mix, comb, layer_input
