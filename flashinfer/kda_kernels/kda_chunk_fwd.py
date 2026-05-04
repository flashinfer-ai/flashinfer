"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Kimi Delta Attention (KDA) — Chunk Prefill Forward Pass
========================================================

Host orchestration for the KDA chunk prefill chain on Blackwell (SM100+):

    1) `kda_fuse_k123` — fused K1 (gate cumsum + Q/K scaling) +
                          K2 (intra-chunk A_qk, A_kk MMAs) +
                          K3 (per-tile causal mask & store)
                          persistent kernel.
    2) `kda_akk_inv`   — BF16 in-place lower-triangular inverse on A_kk
                          (Neumann series), with beta epilogue absorbed.
    3) `kda_fuse_k4`   — Persistent K4: inter-chunk recurrence, V update,
                          state propagation, output store.

The exported API `chunk_kda_fwd` is a thin host wrapper: it allocates
intermediates once per shape, builds CuTe tensor wrappers, JIT-compiles each
kernel on first call, then on every subsequent call hits a fast cache and
just dispatches the three kernels. Supports equal-length and variable-length
(`cu_seqlens`) inputs, both `softplus` and `safe_gate` activations, and
optional `dt_bias`.
"""

import math
import torch
import torch.nn.functional as F
import triton
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda

from .kda_fuse_k123 import make_host_function as _fused_make_host
from .kda_akk_inv import akk_inv_host as _akk_inv_host
from .kda_fuse_k4 import (
    make_host_fn as _k4p_make_host,
    NUM_TENSORMAPS as _K4P_NTM,
    BYTES_PER_TENSORMAP as _K4P_BTM,
)


# ===========================================================================
# Vendored utilities (originally from flash-linear-attention)
# ===========================================================================
RCP_LN2 = 1.4426950408889634  # 1.0 / log(2.0); used implicitly inside the kernels.


def prepare_lens(cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Per-sequence lengths from a cumulative-seqlen tensor."""
    return torch.diff(cu_seqlens)


def prepare_chunk_indices(
    cu_seqlens: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """For varlen, compute (seq_id, intra_seq_chunk_id) for every chunk in the
    batch — the table the persistent kernel uses to recover (seq_idx,
    chunk_within_seq) from a flat chunk index.

    Returns int tensor of shape ``[total_chunks, 2]`` matching ``cu_seqlens``'
    dtype/device.
    """
    indices = torch.cat(
        [torch.arange(n) for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()]
    )
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


def prepare_chunk_offsets(
    cu_seqlens: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Cumulative chunk-count prefix-sum across sequences."""
    return F.pad(
        triton.cdiv(prepare_lens(cu_seqlens), chunk_size), (1, 0), value=0
    ).cumsum(-1)


# ===========================================================================
# Caches
# ===========================================================================
# Per-shape compiled kernels.
_fused_k123_cache = {}
_akk_inv_cache = {}
_k4p_cache = {}
_k4p_tm_ws = {}

# id(cu_seqlens) -> bool, marks whether every seqlen is BT-aligned (so the
# 4 mask sites in K123 can be compile-time elided via VARLEN_PURE=1).
_varlen_pure_cache = {}
# id(cu_seqlens) -> int seqlen (single-seq only).
_varlen_single_seqlen_cache = {}

# id(input_tensor) -> cute_wrapper. Repeat-call inputs (typical in
# inference / training loops) reuse the same cute wrapper.
_input_wrap_cache = {}

# Cached eqlen dummy cu/ci cute wrappers (B+1=2, NT+1=2 sentinel tensors).
_eqlen_dummy_cache = {}

# Cached id(dt_bias)-keyed wrapper.
_dt_bias_cache = {}

# Empty 1x1 fp32 bias used when dt_bias is None.
_empty_bias_cache = {}

# K4 varlen cu_seqlens + chunk_offsets cute wrappers.
_k4_varlen_cu_co_cache = {}
_v_ct_cache = {}
_varlen_k4_input_cache = {}

# All allocated intermediate buffers + cute wrappers, keyed by shape.
_buf_cache = {}

# Side-stream cache (per device) — used to overlap initial_state copy with K123.
_side_streams = {}

# Sentinel-padded g buffer cache: keyed by (B, T_padded, H, K, dtype, dev, real_T).
_g_sentinel_cache = {}
# Padded-input scratch cache (q/k/v/g/beta) for eqlen non-aligned partial chunks.
_padded_input_cache = {}


# ===========================================================================
# Helper functions
# ===========================================================================
def _ct(t, etype):
    """Build a CuTe tensor view over an existing PyTorch tensor."""
    r = from_dlpack(t, assumed_align=16)
    r.element_type = etype
    return r


def _ct_cached(t, etype):
    """`_ct(t, etype)` with id(t)-based cache. Saves ~5-10 us/call of
    `from_dlpack` overhead when the same tensor object is reused."""
    key = (id(t), etype)
    w = _input_wrap_cache.get(key)
    if w is None:
        w = _ct(t, etype)
        _input_wrap_cache[key] = w
    return w


def _cute_int_type(dtype):
    """Map PyTorch integer dtype to CUTLASS element type."""
    if dtype == torch.int32:
        return cutlass.Int32
    elif dtype == torch.int64:
        return cutlass.Int64
    raise ValueError(f"Unsupported integer dtype: {dtype}")


def _get_eqlen_dummies(device, idx_dtype=torch.int64):
    """Cached (cu_ct, ci_ct) cute wrappers for eqlen (no varlen tables)."""
    key = (device.index if device.index is not None else 0, idx_dtype)
    if key not in _eqlen_dummy_cache:
        cu_t = torch.empty(2, dtype=idx_dtype, device=device)
        ci_t = torch.empty(1, 2, dtype=idx_dtype, device=device)
        cu_etype = cutlass.Int64 if idx_dtype == torch.int64 else cutlass.Int32
        _eqlen_dummy_cache[key] = (_ct(cu_t, cu_etype), _ct(ci_t, cu_etype))
    return _eqlen_dummy_cache[key]


def _get_dt_bias_ct(dt_bias, H, K):
    """`dt_bias.float().view(H, K)` cute wrapper, keyed by id(dt_bias)."""
    key = (id(dt_bias), H, K)
    entry = _dt_bias_cache.get(key)
    if entry is None:
        bias_t = dt_bias.float().contiguous().view(H, K)
        entry = (bias_t, _ct(bias_t, cutlass.Float32))
        _dt_bias_cache[key] = entry
    return entry[1]


def _get_empty_bias_ct(device):
    """Reusable 1x1 fp32 dummy when dt_bias is None."""
    idx = device.index if device.index is not None else 0
    if idx not in _empty_bias_cache:
        t = torch.empty(1, 1, dtype=torch.float32, device=device)
        _empty_bias_cache[idx] = _ct(t, cutlass.Float32)
    return _empty_bias_cache[idx]


def _get_k4_varlen_cu_co(cu_seqlens, chunk_offsets):
    """Cached (cu_ct, co_ct) for varlen K4 launch. int32-cast on first call."""
    key = (id(cu_seqlens), id(chunk_offsets))
    entry = _k4_varlen_cu_co_cache.get(key)
    if entry is None:
        cu_int32 = cu_seqlens if cu_seqlens.dtype == torch.int32 else cu_seqlens.to(torch.int32)
        cu_int32 = cu_int32.contiguous()
        co_int32 = chunk_offsets if chunk_offsets.dtype == torch.int32 else chunk_offsets.to(torch.int32)
        co_int32 = co_int32.contiguous()
        cu_ct = from_dlpack(cu_int32, assumed_align=4).mark_layout_dynamic()
        cu_ct.element_type = cutlass.Int32
        co_ct = from_dlpack(co_int32, assumed_align=4).mark_layout_dynamic()
        co_ct.element_type = cutlass.Int32
        # Hold refs so the underlying storage stays alive.
        entry = (cu_int32, co_int32, cu_ct, co_ct)
        _k4_varlen_cu_co_cache[key] = entry
    return entry[2], entry[3]


def _get_varlen_k4_inputs(cu_seqlens, BT):
    """Compute & cache (cu_int32, chunk_offsets_int32) on-GPU. No host sync."""
    key = id(cu_seqlens)
    entry = _varlen_k4_input_cache.get(key)
    if entry is None:
        cu_int32 = cu_seqlens if cu_seqlens.dtype == torch.int32 else cu_seqlens.to(torch.int32)
        cu_int32 = cu_int32.contiguous()
        seq_lens = cu_int32[1:] - cu_int32[:-1]
        chunk_counts = (seq_lens + (BT - 1)) // BT
        zero = torch.zeros(1, dtype=torch.int32, device=cu_int32.device)
        co_int32 = torch.cat([zero, torch.cumsum(chunk_counts, dim=0).to(torch.int32)])
        co_int32 = co_int32.contiguous()
        _varlen_k4_input_cache[key] = (cu_int32, co_int32)
    return _varlen_k4_input_cache[key]


def _get_side_stream(dev):
    idx = dev.index or 0
    if idx not in _side_streams:
        _side_streams[idx] = torch.cuda.Stream(device=dev)
    return _side_streams[idx]


def _get_g_sentinel_buffer(B, T_padded, H, K, dtype_g, device, real_T):
    """Sentinel-padded g scratch (g[real_T:] = -1e3) for the varlen single-seq
    pure path. Persisted by shape so subsequent calls only memcpy the prefix."""
    key = (B, T_padded, H, K, dtype_g,
           device.index if device.index is not None else 0, real_T)
    e = _g_sentinel_cache.get(key)
    if e is None:
        e = torch.zeros(B, T_padded, H, K, dtype=dtype_g, device=device)
        if real_T < T_padded:
            e[:, real_T:] = -1000.0
        _g_sentinel_cache[key] = e
    return e


def _get_padded_input_buffers(B, T_padded, H, K, dtype_qkv, dtype_g, dtype_beta, device, real_T):
    """Pre-allocated padded q/k/v/g/beta buffers for the eqlen non-aligned
    partial-chunk path. q/k/v/beta zero-padded; g uses -1e3 sentinel in the
    tail so the gate activation saturates to zero past the seq end."""
    key = (B, T_padded, H, K, dtype_qkv, dtype_g, dtype_beta,
           device.index if device.index is not None else 0, real_T)
    e = _padded_input_cache.get(key)
    if e is None:
        q_pad = torch.zeros(B, T_padded, H, K, dtype=dtype_qkv, device=device)
        k_pad = torch.zeros(B, T_padded, H, K, dtype=dtype_qkv, device=device)
        v_pad = torch.zeros(B, T_padded, H, K, dtype=dtype_qkv, device=device)
        beta_pad = torch.zeros(B, T_padded, H, dtype=dtype_beta, device=device)
        g_pad = torch.zeros(B, T_padded, H, K, dtype=dtype_g, device=device)
        if real_T < T_padded:
            g_pad[:, real_T:] = -1000.0
        e = (q_pad, k_pad, v_pad, g_pad, beta_pad)
        _padded_input_cache[key] = e
    return e


def _get_buffers(dev, dtype_k, B, T, H, K_dim, V_dim, NT, N_seqs, BT):
    """Allocate all per-shape intermediates + cute wrappers. Cached by shape."""
    key = (dev.index or 0, B, T, H, K_dim, V_dim, NT, N_seqs)
    if key not in _buf_cache:
        bf16 = cutlass.BFloat16
        fp32 = cutlass.Float32
        k_scaled = torch.empty(B, T, H, K_dim, device=dev, dtype=dtype_k)
        kg = torch.empty(B, T, H, K_dim, device=dev, dtype=dtype_k)
        q_scaled = torch.empty(B, T, H, K_dim, device=dev, dtype=dtype_k)
        gk_last_exp = torch.empty(B, NT, H, K_dim, device=dev, dtype=torch.float32)
        A_qk = torch.zeros(B, T, H, BT, device=dev, dtype=dtype_k)
        A_kk = torch.zeros(B, T, H, BT, device=dev, dtype=dtype_k)
        O_flat = torch.empty(B, T, H, V_dim, device=dev, dtype=dtype_k)
        # K4 reads initial state from S_out (caller copies it in) and writes
        # final state back into the same buffer — no separate s_4d.
        S_out = torch.empty(N_seqs, H, K_dim, V_dim, device=dev, dtype=torch.float32)
        cu_eqlen = torch.arange(0, (B + 1) * T, T, dtype=torch.int32, device=dev)
        co_eqlen = torch.arange(0, (B + 1) * (T // BT), T // BT, dtype=torch.int32, device=dev)

        T_total = B * T
        A_kk_flat = A_kk.reshape(T_total, H, BT)
        A_qk_flat = A_qk.reshape(T_total, H, BT)
        KS_flat = k_scaled.reshape(T_total, H, K_dim)
        QS_flat = q_scaled.reshape(T_total, H, K_dim)
        KG_flat = kg.reshape(T_total, H, K_dim)
        O_token = O_flat.reshape(T_total, H, V_dim)
        gk_flat = gk_last_exp.reshape(-1, H, K_dim)

        def _wrap(t, etype):
            r = from_dlpack(t, assumed_align=16).mark_layout_dynamic()
            r.element_type = etype
            return r

        a_ct = _wrap(A_kk_flat, bf16)
        aqc_ct = _wrap(A_qk_flat, bf16)
        ks_ct = _wrap(KS_flat, bf16)
        qs_ct = _wrap(QS_flat, bf16)
        kg_ct = _wrap(KG_flat, bf16)
        o_ct = _wrap(O_token, bf16)
        gk_ct = _wrap(gk_flat, fp32)
        cu_eqlen_ct = from_dlpack(cu_eqlen, assumed_align=4).mark_layout_dynamic()
        cu_eqlen_ct.element_type = cutlass.Int32
        co_eqlen_ct = from_dlpack(co_eqlen, assumed_align=4).mark_layout_dynamic()
        co_eqlen_ct.element_type = cutlass.Int32

        # akk_inv reinterprets the bf16 A_kk storage as fp32 (2 bf16 -> 1 fp32).
        # Compiled signature requires the type_layout to be stable across calls.
        akk_in_view = from_dlpack(A_kk, assumed_align=16); akk_in_view.element_type = fp32
        akk_out_view = from_dlpack(A_kk, assumed_align=16); akk_out_view.element_type = fp32

        # K4 uses S_out for both read (initial state) and write (final state).
        s_ct = from_dlpack(S_out, assumed_align=16); s_ct.element_type = fp32

        main_stream_cached = torch.cuda.current_stream(dev)
        side_stream_cached = _get_side_stream(dev)

        cute_wrappers = dict(
            a_ct=a_ct, aqc_ct=aqc_ct, ks_ct=ks_ct, qs_ct=qs_ct,
            kg_ct=kg_ct, o_ct=o_ct, gk_ct=gk_ct,
            cu_eqlen_ct=cu_eqlen_ct, co_eqlen_ct=co_eqlen_ct,
            akk_in_view=akk_in_view, akk_out_view=akk_out_view,
            s_ct=s_ct,
            main_stream=main_stream_cached, side_stream=side_stream_cached,
            _k123_fns={}, _akk_inv_fn=None, _k4_fn=None,
        )

        _buf_cache[key] = (k_scaled, kg, q_scaled, gk_last_exp,
                           A_qk, A_kk, O_flat, S_out, cu_eqlen, co_eqlen, cute_wrappers)
    return _buf_cache[key]


# ===========================================================================
# Inner launchers
# ===========================================================================
def _launch_k4_persistent(cute_wrappers, v, S_in, S_out,
                          cu_seqlens, chunk_offsets,
                          cu_eqlen_passed=False, num_sm=148, H=None, V_dim=None):
    """Persistent K4 (inter-chunk recurrence + output store)."""
    fast_key = (id(v),
                0 if cu_eqlen_passed else id(cu_seqlens),
                id(S_in), id(S_out))
    fast_cache = cute_wrappers.get('_k4_fast_cache')
    if fast_cache is None:
        fast_cache = {}
        cute_wrappers['_k4_fast_cache'] = fast_cache
    fast_entry = fast_cache.get(fast_key)
    if fast_entry is not None:
        k4_fn, args = fast_entry
        k4_fn(*args)
        return

    bf16 = cutlass.BFloat16
    N_seqs = cu_seqlens.shape[0] - 1
    BH = N_seqs * H
    nsm = min(BH, num_sm)
    dev = v.device
    dev_idx = dev.index or 0

    s_ct = cute_wrappers['s_ct']
    a_ct = cute_wrappers['a_ct']
    b_ct = cute_wrappers['ks_ct']  # raw k_scaled (beta absorbed inside akk_inv)
    q_ct = cute_wrappers['qs_ct']
    aqc_ct = cute_wrappers['aqc_ct']
    kg_ct = cute_wrappers['kg_ct']
    o_ct = cute_wrappers['o_ct']
    gk_ct = cute_wrappers['gk_ct']

    v_key = id(v)
    v_entry = _v_ct_cache.get(v_key)
    if v_entry is None:
        v_view = v.reshape(-1, H, V_dim) if v.dim() == 4 else v
        v_ct = from_dlpack(v_view, assumed_align=16).mark_layout_dynamic()
        v_ct.element_type = bf16
        _v_ct_cache[v_key] = (v_view, v_ct)
    else:
        v_ct = v_entry[1]

    if cu_eqlen_passed:
        cu_ct = cute_wrappers['cu_eqlen_ct']
        co_ct = cute_wrappers['co_eqlen_ct']
    else:
        cu_ct, co_ct = _get_k4_varlen_cu_co(cu_seqlens, chunk_offsets)

    tm_key = (dev_idx, nsm)
    if tm_key not in _k4p_tm_ws:
        tm_ws_t = torch.zeros(nsm * _K4P_NTM * _K4P_BTM, dtype=torch.uint8, device=dev)
        tm_ct = from_dlpack(tm_ws_t, assumed_align=16); tm_ct.element_type = cutlass.Uint8
        _k4p_tm_ws[tm_key] = (tm_ws_t, tm_ct)
    else:
        tm_ws_t, tm_ct = _k4p_tm_ws[tm_key]

    k4_fn = cute_wrappers.get('_k4_fn')
    if k4_fn is None:
        cache_key = (dev_idx, nsm, N_seqs, H)
        k4_fn = _k4p_cache.get(cache_key)
        if k4_fn is None:
            host_fn = _k4p_make_host(num_sm=nsm)
            k4_fn = cute.compile(
                host_fn, a_ct, b_ct, v_ct, q_ct, aqc_ct, kg_ct, o_ct, gk_ct,
                s_ct, cu_ct, co_ct, tm_ct)
            _k4p_cache[cache_key] = k4_fn
        cute_wrappers['_k4_fn'] = k4_fn

    args = (a_ct, b_ct, v_ct, q_ct, aqc_ct, kg_ct, o_ct, gk_ct,
            s_ct, cu_ct, co_ct, tm_ct)
    fast_cache[fast_key] = (k4_fn, args)
    k4_fn(*args)


def _launch_fused_k123_inv(q, k, g, A_log, beta, scale,
                           k_scaled, kg, q_scaled, gk_last_exp, A_qk, A_kk_inv,
                           cu_seqlens, chunk_indices, is_varlen, NT,
                           dt_bias=None, safe_gate=False, lower_bound=None,
                           akk_in_view=None, akk_out_view=None,
                           cute_wrappers=None):
    """Fused K1+K2+K3 persistent kernel chained with BF16 akk_inv (in-place).
    A_kk on output is the lower-triangular inverse `(I+L)^-1` with the
    diagonal scaled by beta (epilogue absorbed inside akk_inv)."""

    if cute_wrappers is not None:
        fast_key = (id(q), id(k), id(g), id(beta), id(A_log),
                    id(dt_bias) if dt_bias is not None else 0,
                    id(cu_seqlens) if cu_seqlens is not None else 0,
                    id(chunk_indices) if chunk_indices is not None else 0,
                    bool(safe_gate),
                    float(lower_bound) if lower_bound is not None else 0.0)
        fast_cache = cute_wrappers.setdefault('_k123_fast_cache', {})
        fast_entry = fast_cache.get(fast_key)
        if fast_entry is not None:
            k123_fn, k123_args, akk_fn, akk_args = fast_entry
            k123_fn(*k123_args)
            akk_fn(*akk_args)
            return

    B, T, H, K = q.shape
    BT = 64
    dev = q.device.index or 0
    T_padded = T if is_varlen else None
    has_bias = dt_bias is not None

    varlen_pure = False
    if is_varlen and cu_seqlens is not None:
        _vp_key = id(cu_seqlens)
        if _vp_key not in _varlen_pure_cache:
            cu_cpu = cu_seqlens.cpu().tolist()
            seq_lens = [cu_cpu[i+1] - cu_cpu[i] for i in range(len(cu_cpu) - 1)]
            _varlen_pure_cache[_vp_key] = all((sl % BT) == 0 for sl in seq_lens)
        varlen_pure = _varlen_pure_cache[_vp_key]
    cache_key = (B, NT, H, is_varlen, T_padded, dev, has_bias, safe_gate, varlen_pure)

    q_ct = _ct_cached(q, cutlass.BFloat16)
    k_ct = _ct_cached(k, cutlass.BFloat16)
    g_ct = _ct_cached(g, cutlass.BFloat16)
    alog_ct = _ct_cached(A_log if A_log.dtype == torch.float32 else A_log.float(),
                         cutlass.Float32)
    beta_ct = _ct_cached(beta, cutlass.BFloat16)

    ks_ct = _ct_cached(k_scaled, cutlass.BFloat16)
    kg_ct = _ct_cached(kg, cutlass.BFloat16)
    qs_ct = _ct_cached(q_scaled, cutlass.BFloat16)
    gk_ct = _ct_cached(gk_last_exp, cutlass.Float32)
    aqk_ct = _ct_cached(A_qk, cutlass.BFloat16)
    akk_ct = _ct_cached(A_kk_inv, cutlass.BFloat16)

    if is_varlen:
        cu_ct = _ct_cached(cu_seqlens, _cute_int_type(cu_seqlens.dtype))
        ci_ct = _ct_cached(chunk_indices, _cute_int_type(chunk_indices.dtype))
    else:
        cu_ct, ci_ct = _get_eqlen_dummies(q.device, torch.int64)

    if dt_bias is not None:
        bias_ct = _get_dt_bias_ct(dt_bias, H, K)
    else:
        bias_ct = _get_empty_bias_ct(q.device)
    lb_val = float(lower_bound) if lower_bound is not None else 0.0

    ct_args = (q_ct, k_ct, g_ct, alog_ct, beta_ct, scale,
               ks_ct, kg_ct, qs_ct, gk_ct, aqk_ct, akk_ct,
               cu_ct, ci_ct, bias_ct, lb_val)

    if cache_key not in _fused_k123_cache:
        host_fn = _fused_make_host(B, NT, H,
                                   is_varlen=is_varlen, T_padded=T_padded,
                                   has_bias=has_bias, use_safe_gate=safe_gate,
                                   varlen_pure=varlen_pure)
        _fused_k123_cache[cache_key] = cute.compile(host_fn, *ct_args)
    k123_fn = _fused_k123_cache[cache_key]
    k123_fn(*ct_args)

    # ===== Chained BF16 akk_inv (in-place: A_kk_inv = (I+L)^-1) =====
    if akk_in_view is None:
        akk_in_view = from_dlpack(A_kk_inv, assumed_align=16)
        akk_in_view.element_type = cutlass.Float32
        akk_out_view = from_dlpack(A_kk_inv, assumed_align=16)
        akk_out_view.element_type = cutlass.Float32

    if is_varlen:
        akk_cu_ct = cu_ct
        akk_ci_ct = ci_ct
        is_varlen_int = 1
        T_val = T
    else:
        akk_cu_ct, akk_ci_ct = cu_ct, ci_ct
        is_varlen_int = 0
        T_val = NT * BT

    akk_cache_key = (B, NT, H, is_varlen, dev, T_val)
    if akk_cache_key not in _akk_inv_cache:
        _akk_inv_cache[akk_cache_key] = cute.compile(
            _akk_inv_host, akk_in_view, akk_out_view, beta_ct, B, NT, H,
            akk_cu_ct, akk_ci_ct, is_varlen_int, T_val)
    akk_fn = _akk_inv_cache[akk_cache_key]
    akk_args = (akk_in_view, akk_out_view, beta_ct, akk_cu_ct, akk_ci_ct)
    akk_fn(*akk_args)

    if cute_wrappers is not None:
        cute_wrappers['_k123_fast_cache'][fast_key] = (k123_fn, ct_args, akk_fn, akk_args)


# ===========================================================================
# Public API
# ===========================================================================
def chunk_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = 64,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
):
    """Kimi Delta Attention chunk prefill forward pass.

    Parameters
    ----------
    q, k : ``torch.Tensor``
        Queries / keys, shape ``[B, T, H, K]``, dtype ``torch.bfloat16``.
        Caller is expected to pre-l2-normalize them (as in the FLA reference).
    v : ``torch.Tensor``
        Values, shape ``[B, T, H, K]`` (V dim equals K), dtype bfloat16.
    g : ``torch.Tensor``
        Per-token gate features, shape ``[B, T, H, K]``, dtype bfloat16. Will
        be activated inside the K1 kernel as either ``-exp(A_log) * softplus(g + dt_bias)``
        (when ``safe_gate=False``) or ``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))``
        (when ``safe_gate=True``).
    beta : ``torch.Tensor``
        Per-token beta scalars, shape ``[B, T, H]``, dtype bfloat16.
    scale : float
        Q-scale used inside the kernel (typical: ``K ** -0.5``).
    initial_state : ``torch.Tensor`` or None
        Optional initial recurrent state, shape ``[N_seqs, H, K, V]`` fp32.
        When None, treated as zero.
    output_final_state : bool
        Whether to return the final state.
    cu_seqlens : ``torch.Tensor`` or None
        Variable-length cumulative sequence lengths, shape ``[N_seqs+1]`` int.
        When None, equal-length batch is assumed.
    chunk_indices : ``torch.Tensor`` or None
        Optional pre-built (seq_id, chunk_within_seq) table for varlen — pass
        the result of :func:`prepare_chunk_indices` to avoid recomputing.
    chunk_size : int
        Chunk size; only ``64`` is currently supported.
    safe_gate : bool
        When True, use sigmoid-based safe gate; otherwise softplus.
    lower_bound : float or None
        Lower bound used in safe_gate mode. Defaults to ``-5.0`` when
        ``safe_gate=True`` and ``lower_bound is None``.
    A_log : ``torch.Tensor``
        Per-head log-decay-rate, shape ``[H]``, dtype fp32.
    dt_bias : ``torch.Tensor`` or None
        Optional per-head-per-feature bias, shape ``[H*K]`` (or ``[H, K]``),
        dtype fp32. Added inside the K1 gate activation.

    Returns
    -------
    o : ``torch.Tensor``
        Output, same shape and dtype as ``v``.
    final_state : ``torch.Tensor`` or None
        Final recurrent state (only when ``output_final_state=True``).
    """
    if chunk_size != 64:
        raise NotImplementedError(f"chunk_size != 64 is not supported (got {chunk_size}).")
    if A_log is None:
        raise ValueError("A_log is required for KDA chunk forward.")

    if safe_gate and lower_bound is None:
        lower_bound = -5.0

    is_varlen = cu_seqlens is not None
    B, T, H, K = q.shape
    V_dim = v.shape[-1]
    device = q.device
    BT = 64

    # Phase 1: eqlen with T % 64 != 0 — host-pad to a CHUNKS_PER_BLOCK*BT=256
    # multiple so the persistent scheduler's `cgs_per_head = NT // 4` divides
    # cleanly and every chunk has 64 valid rows of (zero-padded / sentinel-
    # padded) data.
    real_T = T
    needs_eqlen_pad = (not is_varlen) and (T % BT != 0)
    if needs_eqlen_pad:
        if B != 1:
            raise NotImplementedError(
                f"eqlen with B>1 and T % {BT} != 0 not supported "
                f"(got B={B}, T={T}).")
        CPB_BT = 4 * BT
        T_padded = ((T + CPB_BT - 1) // CPB_BT) * CPB_BT
        q_pad, k_pad, v_pad, g_pad, beta_pad = _get_padded_input_buffers(
            B, T_padded, H, K, q.dtype, g.dtype, beta.dtype, q.device, real_T)
        q_pad[:, :real_T].copy_(q)
        k_pad[:, :real_T].copy_(k)
        v_pad[:, :real_T].copy_(v)
        beta_pad[:, :real_T].copy_(beta)
        g_pad[:, :real_T].copy_(g)
        q, k, v, g, beta = q_pad, k_pad, v_pad, g_pad, beta_pad
        T = T_padded

    # Phase 2.1: varlen with a SINGLE non-aligned sequence — sentinel-pad g
    # and force VARLEN_PURE=1 so all 4 mask sites compile-elide.
    needs_varlen_single_pad = False
    if is_varlen and cu_seqlens is not None and cu_seqlens.shape[0] == 2 and B == 1:
        _vl_key = id(cu_seqlens)
        if _vl_key not in _varlen_pure_cache:
            cu_cpu = cu_seqlens.cpu().tolist()
            sl = cu_cpu[1] - cu_cpu[0]
            _varlen_pure_cache[_vl_key] = (sl % BT == 0)
            _varlen_single_seqlen_cache[_vl_key] = sl
        if not _varlen_pure_cache[_vl_key]:
            real_T = _varlen_single_seqlen_cache[_vl_key]
            needs_varlen_single_pad = True
    if needs_varlen_single_pad:
        cur_T = q.shape[1]
        g_pad = _get_g_sentinel_buffer(B, cur_T, H, K, g.dtype, g.device, real_T)
        g_pad[:, :real_T].copy_(g[:, :real_T])
        g = g_pad
        _varlen_pure_cache[id(cu_seqlens)] = True

    if is_varlen:
        if chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        NT = len(chunk_indices)
        N_seqs = len(cu_seqlens) - 1
    else:
        NT = T // BT
        N_seqs = B

    (k_scaled, kg, q_scaled, gk_last_exp,
     A_qk, A_kk, O_flat, S_out, cu_eqlen, co_eqlen, cute_wrappers) = (
        _get_buffers(device, k.dtype, B, T, H, K, V_dim, NT, N_seqs, BT)
    )

    # State copy on side stream — overlaps with K123.
    if initial_state is None:
        S_in = torch.zeros(N_seqs, H, K, V_dim, dtype=torch.float32, device=device)
    else:
        S_in = initial_state
    main_stream = cute_wrappers['main_stream']
    side_stream = cute_wrappers['side_stream']
    needs_copy = S_in.data_ptr() != S_out.data_ptr()
    if needs_copy:
        side_stream.wait_stream(main_stream)
        with torch.cuda.stream(side_stream):
            S_out.copy_(S_in)

    _launch_fused_k123_inv(q, k, g, A_log, beta, scale,
                           k_scaled, kg, q_scaled, gk_last_exp, A_qk, A_kk,
                           cu_seqlens, chunk_indices, is_varlen, NT,
                           dt_bias=dt_bias, safe_gate=safe_gate, lower_bound=lower_bound,
                           akk_in_view=cute_wrappers['akk_in_view'],
                           akk_out_view=cute_wrappers['akk_out_view'],
                           cute_wrappers=cute_wrappers)

    if is_varlen:
        cu_for_k4, chunk_offsets_for_k4 = _get_varlen_k4_inputs(cu_seqlens, BT)
    else:
        cu_for_k4 = cu_eqlen
        chunk_offsets_for_k4 = co_eqlen

    if needs_copy:
        main_stream.wait_stream(side_stream)

    _launch_k4_persistent(
        cute_wrappers, v, S_in, S_out,
        cu_for_k4, chunk_offsets_for_k4,
        cu_eqlen_passed=(not is_varlen),
        H=H, V_dim=V_dim)

    o = O_flat
    if needs_eqlen_pad:
        o = o[:, :real_T]
    final_state = S_out if output_final_state else None

    return o, final_state
