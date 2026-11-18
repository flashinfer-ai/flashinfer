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


@functools.cache
def get_mhc_pre_glm_module():
    mod = gen_mhc_pre_glm_module().build_and_load()
    _BUILT["ready"] = True
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
):
    """Guard order: GPU-free (kill switch, scalar-parameter pins, dtype,
    geometry allowlist) -> layout guards -> device guards. Returns [] when
    the specialized kernel can serve the call, else fallback reasons. No
    device syncs."""
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
        or not residual.is_contiguous()
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
        if _t.data_ptr() % 16:
            r.append(_name + "_not_16B_aligned")
    if r:
        return r
    if torch.cuda.get_device_capability(residual.device) != (10, 3):
        return ["not_sm103"]
    if torch.cuda.is_current_stream_capturing() and not _BUILT["ready"]:
        return ["module_not_built_before_graph_capture"]
    return []


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
    if it cannot serve the call it raises."""
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
        try:
            get_mhc_pre_glm_module()
        except Exception as e:  # pragma: no cover - environment-dependent
            # A missing prebuilt JIT artifact (or a build failure) must not
            # take down "auto": record the reason and keep the big-fuse path.
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
