"""CUDA (tvm_ffi) fast path for ragged MLA prefill at the DeepSeek-R1 head
geometry: fp8 e4m3 q/k/v streams (12 q heads, qk head dim 192 = 128 nope +
64 rope, v head dim 128), bf16 out, softmax scale 192**-0.5, ragged
(unpaged) per-request segments.

Scope: a strict shape allowlist of measured serving classes (non-causal
context chunks AND the causal new-tokens class) on SM103 at the
``return_lse=True`` call form; everything else stays on the stock
``trtllm_ragged_attention_deepseek`` dispatch. Kill switch:
``FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1``.

CUDA graphs: the specialized route BAILS to the stock path whenever the
current stream is capturing (``torch.cuda.is_current_stream_capturing()``),
on every allowlisted shape. Two of the kernel's shape-specialized launch
routes fail graph capture with ``cudaErrorInvalidValue`` (measured on
GB300); rather than allowlisting capture per route, capture is uniformly
served by the stock path, which keeps graphed and eager results consistent
with stock semantics. Eager dispatch is host-int based with no device
syncs, the stream is re-queried per call, and split-KV scratch comes from
the caller's ``workspace_buffer`` (no allocation on the dispatch path).

LSE: the kernel EMITS the log-sum-exp output (fp32 ``[TQ, 12]``, base-2
with the bmm1 scale in the exponent — the FI ``return_lse=True``
convention), and the specialized route serves exactly the measured serve
call form: ``return_lse=True`` calls on the allowlist (now including the
causal class). ``return_lse=False`` calls keep the stock path
(``lse_not_requested`` fallback reason) — routing is per measured row.
Rows with runtime ``max_kv_len <= 256`` take an extra exact-LSE epilogue
kernel (LSE accuracy is discontinuous at kv=256; see the PR disclosure).

Generated-kernel provenance: agent-generated kernel (search-campaign
winner), source-vetted (runtime shape-guarded dispatch over general
split-KV fallback paths; ragged cu_q/cu_kv honored on every path; fp32
stats, fp16 split partials, residual-boosted fp8 P where the dynamic range
needs it) and re-timed per workload row on GB300 against the stock
trtllm-gen path of this entry point.
"""

import functools
import os as _os

import torch

from .jit.mla_ragged_prefill_dsr1 import gen_mla_ragged_prefill_dsr1_module

_SCALE = 0.07216878364870322  # 192 ** -0.5 (the geometry this path serves)
KILL_SWITCH_ENV = "FLASHINFER_SPECIALIZED_KERNEL_DISABLE"
DISPATCH_STATS = {"specialized": 0, "fallback": 0}
LAST_FALLBACK_REASONS: list = []

# Measured serving census classes (exact tuples): (batch_size, total_q_tokens,
# total_kv_tokens, max_q_len, max_kv_len, is_causal). At the serve call form
# (return_lse=True) the kernel was measured faster than the stock path on
# every class INCLUDING the causal q16128/kv16128 one (which the LSE-free
# predecessor lost); off-list shapes always keep the stock path, so routing
# is per measured row, not on average.
ALLOWLIST = frozenset(
    [
        (1, 128, 66304, 128, 66304, 0),
        (1, 128, 58112, 128, 58112, 0),
        (1, 128, 20480, 128, 20480, 0),
        (1, 128, 12416, 128, 12416, 0),
        (2, 418, 66304, 256, 65920, 0),
        (1, 256, 48384, 256, 48384, 0),
        (1, 256, 30464, 256, 30464, 0),
        (1, 256, 15872, 256, 15872, 0),
        (1, 256, 66304, 256, 66304, 0),
        (1, 384, 51584, 384, 51584, 0),
        (1, 512, 66304, 512, 66304, 0),
        (1, 2101, 66304, 2101, 66304, 0),
        (1, 16128, 66304, 16128, 66304, 0),
        (1, 16128, 16128, 16128, 16128, 1),
    ]
)


@functools.cache
def get_mla_ragged_prefill_dsr1_module():
    return gen_mla_ragged_prefill_dsr1_module().build_and_load()


# A failed module build/load (e.g. a cache-only deployment without the
# compiled artifact, FLASHINFER_DISABLE_JIT=1) is cached as a fallback
# reason so "auto" keeps the stock path without retrying the build on
# every call.  [reason_string, exception] once populated.
_MODULE_LOAD_FAILURE: list = []


def _module_load_reasons():
    """Load (and cache) the specialized module; return [] on success or the
    cached fallback-reason list when the module cannot be built/loaded."""
    if _MODULE_LOAD_FAILURE:
        return [_MODULE_LOAD_FAILURE[0]]
    try:
        get_mla_ragged_prefill_dsr1_module()
    except Exception as e:
        _MODULE_LOAD_FAILURE.append(
            "specialized_module_unavailable:" + type(e).__name__
        )
        _MODULE_LOAD_FAILURE.append(e)
        return [_MODULE_LOAD_FAILURE[0]]
    return []


def supported_with_reasons(
    query,
    key,
    value,
    workspace_buffer,
    seq_lens,
    max_q_len,
    max_kv_len,
    cum_seq_lens_q,
    cum_seq_lens_kv,
    batch_size,
    out,
    extras,
):
    """Guard order: GPU-free checks (kill switch, capture bail, API-surface
    flags, dtype, geometry, shape allowlist) -> layout guards -> device
    guards -> workspace capacity. Returns [] when the specialized kernel can
    serve the call, else the fallback reasons. No device syncs."""
    if _os.environ.get(KILL_SWITCH_ENV, "0") == "1":
        return ["kill_switch:" + KILL_SWITCH_ENV + "=1"]
    _capturing = False
    if query.is_cuda and hasattr(torch.cuda, "is_current_stream_capturing"):
        with torch.cuda.device(query.device):
            _capturing = torch.cuda.is_current_stream_capturing()
    if _capturing:
        # Uniform capture bail (all routes): capture is served by the stock
        # path. See the module docstring.
        return ["cuda_graph_capture_bail"]
    r = []
    for name, ok in (
        ("lse_not_requested", bool(extras.get("return_lse"))),
        ("window_left_not_full", extras.get("window_left") == -1),
        ("attention_sinks", extras.get("attention_sinks") is None),
        (
            "skip_softmax_threshold",
            extras.get("skip_softmax_threshold_scale_factor") is None,
        ),
        (
            "sage_attention",
            all(t is None for t in extras.get("sage_attn_sfs", ()))
            and not any(extras.get("num_elts_per_sage_attn_blk", ())),
        ),
        ("fp16_softmax_variant", not extras.get("use_fp16_softmax")),
        ("spcompress_variant", not extras.get("uses_spcompress")),
        (
            "cpu_len_mirrors",
            extras.get("q_seq_lens_cpu") is None
            and extras.get("kv_seq_lens_cpu") is None,
        ),
        (
            "bmm1_scale_not_float",
            not isinstance(extras.get("bmm1_scale"), torch.Tensor),
        ),
        (
            "bmm2_scale_not_float",
            not isinstance(extras.get("bmm2_scale"), torch.Tensor),
        ),
    ):
        if not ok:
            r.append(name)
    if r:
        return r
    if abs(float(extras["bmm1_scale"]) - _SCALE) > 1e-7:
        r.append("bmm1_scale_mismatch")
    if abs(float(extras["bmm2_scale"]) - 1.0) > 1e-7:
        r.append("bmm2_scale_mismatch")
    if (
        extras.get("o_sf_scale") is not None
        and abs(float(extras["o_sf_scale"]) - 1.0) > 1e-7
    ):
        r.append("o_sf_scale_mismatch")
    if (
        query.dim() != 3
        or query.shape[1] != 12
        or query.shape[2] != 192
        or query.dtype != torch.float8_e4m3fn
        or not query.is_contiguous()
    ):
        r.append("q_not_fp8e4m3_12x192")
    if (
        key.dim() != 3
        or key.shape[1] != 12
        or key.shape[2] != 192
        or key.dtype != torch.float8_e4m3fn
        or not key.is_contiguous()
    ):
        r.append("k_not_fp8e4m3_12x192")
    if (
        value.dim() != 3
        or value.shape[0:2] != key.shape[0:2]
        or value.shape[2] != 128
        or value.dtype != torch.float8_e4m3fn
        or not value.is_contiguous()
    ):
        r.append("v_not_fp8e4m3_12x128")
    if r:
        return r
    # TMA globalAddress and the vectorized output/workspace accesses need
    # 16B pointers; contiguity does not exclude a storage-offset view.
    if any(t.data_ptr() % 16 for t in (query, key, value, workspace_buffer)):
        r.append("qkv_or_workspace_not_16B_aligned")
    if out is not None and out.data_ptr() % 16:
        r.append("out_not_16B_aligned")
    lse_buf = extras.get("lse")
    if lse_buf is not None and (
        lse_buf.dtype != torch.float32
        or tuple(lse_buf.shape) != (query.shape[0], 12)
        or not lse_buf.is_contiguous()
        or lse_buf.device != query.device
    ):
        r.append("lse_not_fp32_TQx12_contiguous")
    sig = (
        int(batch_size),
        int(query.shape[0]),
        int(key.shape[0]),
        int(max_q_len),
        int(max_kv_len),
        int(bool(extras.get("is_causal"))),
    )
    if sig not in ALLOWLIST:
        return ["shape_not_in_measured_allowlist"]
    if (
        seq_lens.dim() != 1
        or seq_lens.shape[0] != batch_size
        or seq_lens.dtype != torch.int32
        or not seq_lens.is_cuda
        or not seq_lens.is_contiguous()
    ):
        r.append("seq_lens_not_int32_cuda_B")
    for name, cu in (
        ("cum_seq_lens_q", cum_seq_lens_q),
        ("cum_seq_lens_kv", cum_seq_lens_kv),
    ):
        if (
            cu.dim() != 1
            or cu.shape[0] != batch_size + 1
            or cu.dtype != torch.int32
            or not cu.is_cuda
            or not cu.is_contiguous()
        ):
            r.append(name + "_not_int32_cuda_Bp1")
    if out is not None and (
        out.dtype != torch.bfloat16
        or tuple(out.shape) != (query.shape[0], 12, 128)
        or not out.is_contiguous()
        or out.device != query.device
    ):
        r.append("out_not_bf16_q_shaped_contiguous")
    if (
        workspace_buffer.dim() != 1
        or workspace_buffer.dtype != torch.uint8
        or not workspace_buffer.is_contiguous()
        or workspace_buffer.device != query.device
    ):
        r.append("workspace_not_1d_uint8")
    if r:
        return r
    if torch.cuda.get_device_capability(query.device) != (10, 3):
        return ["not_sm103"]
    # Workspace capacity, from the kernel's own host-side size model
    # (builds the module on the first on-allowlist eager call: the
    # warm-once requirement).  A failed build/load here is a fallback
    # reason, not an error: "auto" keeps the stock path, which serves the
    # same call (an explicit backend="specialized" call still raises, with
    # this reason in the message).
    load_reasons = _module_load_reasons()
    if load_reasons:
        return load_reasons
    mod = get_mla_ragged_prefill_dsr1_module()
    # The size model depends on the SM count, so scope the query to the
    # query tensor's device: a different current device must not plan this
    # call's splits (Run re-derives the same plan under its device guard).
    with torch.cuda.device(query.device):
        need = int(
            mod.mla_ragged_prefill_dsr1_workspace_bytes(
                int(batch_size),
                int(query.shape[0]),
                int(key.shape[0]),
                int(max_q_len),
                int(max_kv_len),
                int(bool(extras.get("is_causal"))),
            )
        )
    if workspace_buffer.shape[0] < need:
        return [f"workspace_too_small_need_{need}_bytes"]
    return []


def maybe_dispatch_specialized(
    backend,
    query,
    key,
    value,
    workspace_buffer,
    seq_lens,
    max_q_len,
    max_kv_len,
    cum_seq_lens_q,
    cum_seq_lens_kv,
    batch_size,
    out,
    extras,
):
    """Route for backend in ("auto", "specialized"). Returns the output
    tensor when served; None when "auto" should continue on the stock
    dispatch. An explicitly requested "specialized" backend is never
    silently swapped: if it cannot serve the call it raises."""
    reasons = supported_with_reasons(
        query,
        key,
        value,
        workspace_buffer,
        seq_lens,
        max_q_len,
        max_kv_len,
        cum_seq_lens_q,
        cum_seq_lens_kv,
        batch_size,
        out,
        extras,
    )
    LAST_FALLBACK_REASONS[:] = reasons
    if not reasons:
        try:
            get_mla_ragged_prefill_dsr1_module()
        except Exception as e:  # pragma: no cover - environment-dependent
            if backend == "specialized":
                raise
            LAST_FALLBACK_REASONS[:] = [
                "specialized_module_unavailable:" + type(e).__name__
            ]
            DISPATCH_STATS["fallback"] += 1
            return None
        DISPATCH_STATS["specialized"] += 1
        return _run_specialized(
            query,
            key,
            value,
            workspace_buffer,
            seq_lens,
            max_q_len,
            max_kv_len,
            cum_seq_lens_q,
            cum_seq_lens_kv,
            out,
            extras.get("lse"),
            bool(extras.get("is_causal")),
        )
    if backend == "specialized":
        err = ValueError(
            "backend='specialized' cannot serve this call and is never "
            f"silently swapped; fallback_reasons={reasons}"
        )
        if _MODULE_LOAD_FAILURE and _MODULE_LOAD_FAILURE[0] in reasons:
            # Keep the original build/load failure in the chain.
            raise err from _MODULE_LOAD_FAILURE[1]
        raise err
    DISPATCH_STATS["fallback"] += 1
    return None


def _run_specialized(
    query,
    key,
    value,
    workspace_buffer,
    seq_lens,
    max_q_len,
    max_kv_len,
    cum_seq_lens_q,
    cum_seq_lens_kv,
    out,
    lse,
    is_causal,
):
    if out is None:
        out = torch.empty(
            query.shape[0], 12, 128, device=query.device, dtype=torch.bfloat16
        )
    if lse is None:
        lse = torch.empty(query.shape[0], 12, device=query.device, dtype=torch.float32)
    # Per-request query lengths from the ragged structure (int32 diff; no
    # host sync). seq_lens carry the KV lengths (non-causal) or the query
    # lengths (causal new-tokens form, where kv == q per the serve ABI).
    q_lens = cum_seq_lens_q[1:] - cum_seq_lens_q[:-1]
    mod = get_mla_ragged_prefill_dsr1_module()
    mod.mla_ragged_prefill_dsr1_run(
        query,
        key,
        value,
        q_lens.contiguous(),
        seq_lens,
        cum_seq_lens_q,
        cum_seq_lens_kv,
        int(max_q_len),
        int(max_kv_len),
        int(is_causal),
        workspace_buffer,
        out,
        lse,
    )
    return out, lse
