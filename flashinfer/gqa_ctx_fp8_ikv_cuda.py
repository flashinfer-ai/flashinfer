"""CUDA (tvm_ffi) fast path for fp8-QKV GQA context (chunked prefill) over a
token-interleaved paged KV cache, 16:1 head group, head_dim 128, page 128.

Scope: single-page requests (max_kv_len <= 128) at this exact geometry and
softmax scale on SM103; everything else stays on the stock trtllm-gen path.
Kill switch: FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1. CUDA graphs: the route is
taken under capture only after the module was built eagerly (single fatbin,
no per-shape JIT); dispatch is host-int based with no syncs.

Generated-kernel provenance: agent-generated kernel (search-campaign winner),
source-vetted (runtime-guarded dispatch with a general page-indirect
fallback family) and re-timed per workload row on GB300 against the
trtllm-gen incumbent.
"""

import functools

import torch

from .jit.gqa_ctx_fp8_ikv import gen_gqa_ctx_fp8_ikv_module

_SCALE = 0.08838834764831845  # 128 ** -0.5
_BUILT = {"ready": False}


@functools.cache
def get_gqa_ctx_fp8_ikv_module():
    mod = gen_gqa_ctx_fp8_ikv_module().build_and_load()
    _BUILT["ready"] = True
    return mod


# ---------------------------------------------------------------------------
# "specialized" backend adapter: auto routing on a strict allowlist, explicit
# backends never swapped, fallback reasons + dispatch stats.
# ---------------------------------------------------------------------------
import os as _os

_SCALE = 0.08838834764831845  # 128 ** -0.5 (the geometry this path serves)
KILL_SWITCH_ENV = "FLASHINFER_SPECIALIZED_KERNEL_DISABLE"
DISPATCH_STATS = {"specialized": 0, "fallback": 0}
LAST_FALLBACK_REASONS: list = []


def supported_with_reasons(
    query,
    kv_cache,
    block_tables,
    seq_lens,
    max_q_len,
    max_kv_len,
    bmm1_scale,
    bmm2_scale,
    out,
    extras,
):
    """Guard order: GPU-free (kill switch, API-surface flags, dtype,
    geometry allowlist) -> layout guards -> device guards. Returns [] when
    the specialized kernel can serve the call, else fallback reasons. No
    device syncs."""
    if _os.environ.get(KILL_SWITCH_ENV, "0") == "1":
        return ["kill_switch:" + KILL_SWITCH_ENV + "=1"]
    r = []
    for name, ok in (
        ("out_provided", out is not None),
        ("out_dtype_default", extras.get("out_dtype") is None),
        ("no_lse", extras.get("lse") is None and not extras.get("return_lse")),
        ("no_sinks", extras.get("sinks") is None),
        ("no_kv_cache_sf", extras.get("kv_cache_sf") is None),
        ("no_o_sf", extras.get("o_sf_scale") is None),
        ("no_skip_softmax", extras.get("skip_softmax") is None),
        ("shared_paged_kv_idx", bool(extras.get("uses_shared_paged_kv_idx"))),
        ("no_multi_ctas_buf", extras.get("multi_ctas") is None),
        ("causal", bool(extras.get("causal"))),
        ("hnd_layout", extras.get("kv_layout") == "HND"),
        (
            "dense_window",
            extras.get("window_left") == -1 and extras.get("window_right", -1) == -1,
        ),
        ("no_fp16_softmax", not extras.get("use_fp16_softmax")),
        ("no_spcompress", not extras.get("uses_spcompress")),
        (
            "no_variable_window",
            extras.get("vw_starts") is None and extras.get("vw_ends") is None,
        ),
    ):
        if not ok:
            r.append(name)
    if r:
        return r
    if not isinstance(kv_cache, tuple) or len(kv_cache) != 2:
        return ["kv_cache_not_tuple"]
    k, v = kv_cache
    if (
        query.dtype != torch.float8_e4m3fn
        or k.dtype != torch.float8_e4m3fn
        or v.dtype != torch.float8_e4m3fn
    ):
        r.append("dtype_not_fp8e4m3")
    if out.dtype != torch.bfloat16 or not out.is_contiguous():
        r.append("out_not_bf16_contiguous")
    if (
        query.dim() != 3
        or query.shape[1] != 16
        or query.shape[2] != 128
        or not query.is_contiguous()
    ):
        r.append("query_geometry")
    if (
        isinstance(bmm1_scale, torch.Tensor)
        or isinstance(bmm2_scale, torch.Tensor)
        or abs(bmm1_scale - _SCALE) > 1e-7
        or bmm2_scale != 1.0
    ):
        r.append("scale_mismatch")
    if max_kv_len > 128 or max_q_len > 128:
        r.append("multi_page_regime")  # stock path wins there (measured)
    if block_tables.dim() != 2 or block_tables.dtype != torch.int32:
        r.append("block_tables_geometry")
    if seq_lens.dtype != torch.int32:
        r.append("seq_lens_dtype")
    if r:
        return r
    if (
        k.dim() != 4
        or k.shape[1] != 1
        or k.shape[2] != 128
        or k.shape[3] != 128
        or k.stride(3) != 1
        or k.stride(2) != 256
        or v.stride(2) != 256
        or v.data_ptr() != k.data_ptr() + 128
    ):
        return ["kv_layout_not_token_interleaved"]
    if torch.cuda.get_device_capability(query.device) != (10, 3):
        return ["not_sm103"]
    if torch.cuda.is_current_stream_capturing() and not _BUILT["ready"]:
        return ["module_not_built_before_graph_capture"]
    return []


def maybe_dispatch_specialized(
    backend,
    query,
    kv_cache,
    block_tables,
    seq_lens,
    max_q_len,
    max_kv_len,
    bmm1_scale,
    bmm2_scale,
    batch_size,
    cum_seq_lens_q,
    cum_seq_lens_kv,
    out,
    extras,
):
    """Route for backend in ("auto", "specialized"). Returns the output when
    served; None when "auto" should continue on the stock path. An
    explicitly requested "specialized" backend is never silently swapped: if
    it cannot serve the call it raises (no composable fallback exists at
    this entry)."""
    reasons = supported_with_reasons(
        query,
        kv_cache,
        block_tables,
        seq_lens,
        max_q_len,
        max_kv_len,
        bmm1_scale,
        bmm2_scale,
        out,
        extras,
    )
    LAST_FALLBACK_REASONS[:] = reasons
    if not reasons:
        DISPATCH_STATS["specialized"] += 1
        return _run_specialized(
            query,
            kv_cache,
            block_tables,
            seq_lens,
            max_q_len,
            max_kv_len,
            cum_seq_lens_q,
            cum_seq_lens_kv,
            out,
        )
    if backend == "specialized":
        raise ValueError(
            "backend='specialized' cannot serve this call and is never "
            f"silently swapped; fallback_reasons={reasons}"
        )
    DISPATCH_STATS["fallback"] += 1
    return None


def _run_specialized(
    query,
    kv_cache,
    block_tables,
    seq_lens,
    max_q_len,
    max_kv_len,
    cum_seq_lens_q,
    cum_seq_lens_kv,
    out,
):
    k, _v = kv_cache
    mod = get_gqa_ctx_fp8_ikv_module()
    mod.gqa_ctx_fp8_ikv_run(
        query,
        k,
        block_tables,
        cum_seq_lens_q,
        cum_seq_lens_kv,
        seq_lens,
        int(max_q_len),
        int(max_kv_len),
        out,
    )
    return out
