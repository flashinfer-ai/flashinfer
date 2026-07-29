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
"""

import functools
from typing import Optional, Tuple, Union

import torch

from ..api_logging import flashinfer_api
from ..autotuner import AutoTuner, TunableRunner, TuningConfig
from ..trace.templates.msa import (
    msa_proxy_score_fp4_trace,
    msa_proxy_score_trace,
)
from ..utils import get_device_sm_count, is_sm12x_supported
from ._common import _BLK_KV

# NVFP4 scale granularity: one e4m3 block scale per 16 e2m1 elements.
_SF_VEC_SIZE = 16


def _resolve_proxy_dims(cu_seqlens_q, cu_k, max_seqlen_q, max_k_tiles, output):
    """Resolve grid dims ``(max_seqlen_q, max_k_tiles)`` as Python ints; deriving
    them reads ``cu_seqlens`` on host, so under CUDA-graph capture we raise."""
    if max_k_tiles is None and output is not None:
        max_k_tiles = int(output.shape[1])
    if max_seqlen_q is not None and max_k_tiles is not None:
        return max_seqlen_q, max_k_tiles
    if torch.cuda.is_current_stream_capturing():
        raise ValueError(
            "msa_proxy_score[_fp4] needs max_seqlen_q and max_k_tiles during CUDA "
            "graph capture (deriving them from cu_seqlens would sync the stream). "
            "Pass both (computed once at plan time) or pre-allocate `output`."
        )
    if max_seqlen_q is None:
        cu_q_cpu = cu_seqlens_q.cpu()
        max_seqlen_q = int((cu_q_cpu[1:] - cu_q_cpu[:-1]).max().item())
    if max_k_tiles is None:
        cu_k_cpu = cu_k.cpu()
        seqlens_k = cu_k_cpu[1:] - cu_k_cpu[:-1]
        max_k_tiles = int((seqlens_k.max().item() + _BLK_KV - 1) // _BLK_KV)
    return max_seqlen_q, max_k_tiles


def _proxy_split_k(
    base_ctas: int,
    max_k_tiles: int,
    device,
    *,
    wave_target: float,
    gate_factor: float,
    ceil_to_target: bool,
) -> int:
    """KV-block split factor that fills the SMs when the base proxy grid is small
    (long context, low batch). Splits need no reduction: the per-(head, kv_block,
    query) output means each split writes disjoint columns. Returns 1 once the
    base grid already covers ``gate_factor`` SM-waves."""
    if base_ctas <= 0 or max_k_tiles <= 1:
        return 1
    num_sms = get_device_sm_count(device)
    if base_ctas >= gate_factor * num_sms:
        return 1
    target = wave_target * num_sms
    splits = (
        -(-int(target) // base_ctas) if ceil_to_target else round(target / base_ctas)
    )
    return max(1, min(splits, max_k_tiles))


# bf16 split-K heuristic, tuned empirically: target ~1.5 waves and round (the
# Q reload per split makes over-splitting regress).
_proxy_split_k_bf16 = functools.partial(
    _proxy_split_k, wave_target=1.5, gate_factor=1.0, ceil_to_target=False
)


@functools.lru_cache(maxsize=None)
def _split_k_makespan_argmin(
    base_ctas: int, max_k_tiles: int, resident_ctas: int
) -> int:
    """Split factor minimizing a makespan model: full resident-CTA waves times
    per-split work, with ~one KV block of fixed per-CTA overhead (launch + Q
    load). Ties break to the smaller split (fewer Q reloads). A model rather
    than a wave target because the optimum shifts between filling residency
    exactly (moderate batch), max split (batch 1), and no split (large batch,
    short kv) -- no single target covers all three regimes."""
    best_ns, best_cost = 1, None
    for ns in range(1, min(max_k_tiles, 4096) + 1):
        waves = -(-(base_ctas * ns) // resident_ctas)
        cost = waves * (-(-max_k_tiles // ns) + 1)
        if best_cost is None or cost < best_cost:
            best_ns, best_cost = ns, cost
    return best_ns


def _proxy_split_k_fp4(base_ctas: int, max_k_tiles: int, device) -> int:
    """fp4 split-K: 4-bit K reads make splits cheap, so pick by the makespan
    model at the kernel's 2-CTA/SM residency (see _MIN_BLOCKS_PER_MP)."""
    if base_ctas <= 0 or max_k_tiles <= 1:
        return 1
    return _split_k_makespan_argmin(
        base_ctas, max_k_tiles, 2 * get_device_sm_count(device)
    )


class _ProxySplitKRunner(TunableRunner):
    """Autotunes the proxy kv-block split factor: ``tactic`` *is* the split
    factor (a runtime arg, so one compiled kernel covers every candidate)."""

    def __init__(
        self,
        call_fn,
        *,
        max_seqlen_q,
        max_k_tiles,
        base_ctas,
        device,
        heuristic,
        causal,
        paged,
        reduce_heads,
    ):
        # call_fn(inputs, num_splits) -> output; lets one runner serve both the
        # bf16 and (longer-signature) fp4 proxy invocations.
        self._call = call_fn
        self._msq = int(max_seqlen_q)
        self._mkt = int(max_k_tiles)
        self._base = int(base_ctas)
        self._device = device
        self._heuristic = heuristic  # callable(base_ctas, max_k_tiles, device) -> int
        self._causal = bool(causal)
        self._paged = bool(paged)
        self._reduce = bool(reduce_heads)

    def __hash__(self):
        return hash(type(self))

    def get_valid_tactics(self, inputs, profile):
        # Powers of two up to ~4 SM-waves, capped by max_k_tiles; always include
        # the closed-form heuristic so a tuned op is never worse than eager.
        num_sms = get_device_sm_count(self._device)
        cap = min(self._mkt, max(1, -(-4 * num_sms // max(self._base, 1))))
        cands = {s for s in (1, 2, 4, 8, 16, 32) if s <= cap}
        cands.add(self._heuristic(self._base, self._mkt, self._device))
        return sorted(cands)

    def get_cache_key_extras(self, inputs):
        # max_seqlen_q / max_k_tiles are varlen-derived (not in any tensor shape),
        # so they must key the cache; causal/paged/reduce flip the work pattern.
        return (self._causal, self._paged, self._reduce, self._msq, self._mkt)

    def forward(self, inputs, tactic: int = -1, do_preparation: bool = False, **kwargs):
        # tactic == -1 falls back to the closed-form heuristic, so eager
        # (un-tuned) behavior is unchanged.
        ns = (
            tactic
            if tactic >= 0
            else self._heuristic(self._base, self._mkt, self._device)
        )
        return self._call(inputs, int(ns))


# (op, shape) keys that already went through the autotuner in this process. Until
# a shape is tuned the proxy uses its closed-form heuristic and skips choose_one,
# whose per-call lock/bookkeeping (~tens of us) would dominate the tiny low-batch
# decode.
_PROXY_TUNED: set = set()


def _proxy_tuned_key(
    op_name, *, causal, paged, reduce_heads, max_seqlen_q, max_k_tiles, base_ctas
):
    # base_ctas folds in batch/heads/max_seqlen_q, so this matches the autotuner's
    # own cache granularity (shape + get_cache_key_extras) closely enough to gate on.
    return (
        op_name,
        bool(causal),
        bool(paged),
        bool(reduce_heads),
        int(max_seqlen_q),
        int(max_k_tiles),
        int(base_ctas),
    )


def _run_proxy_autotuned(
    op_name,
    call_fn,
    tensors,
    *,
    max_seqlen_q,
    max_k_tiles,
    base_ctas,
    device,
    heuristic,
    causal,
    paged,
    reduce_heads,
):
    """Run the proxy with an autotuned split factor; un-tuned shapes use the
    closed-form heuristic directly and skip the autotuner entirely."""
    tuner = AutoTuner.get()
    tuning = tuner.is_tuning_mode
    tuned_key = _proxy_tuned_key(
        op_name,
        causal=causal,
        paged=paged,
        reduce_heads=reduce_heads,
        max_seqlen_q=max_seqlen_q,
        max_k_tiles=max_k_tiles,
        base_ctas=base_ctas,
    )
    if tuning:
        _PROXY_TUNED.add(tuned_key)
    elif tuned_key not in _PROXY_TUNED:
        call_fn(tensors, heuristic(base_ctas, max_k_tiles, device))
        return

    runner = _ProxySplitKRunner(
        call_fn,
        max_seqlen_q=max_seqlen_q,
        max_k_tiles=max_k_tiles,
        base_ctas=base_ctas,
        device=device,
        heuristic=heuristic,
        causal=causal,
        paged=paged,
        reduce_heads=reduce_heads,
    )
    # No DynamicTensorSpec: the autotuner would synthesize a shape-correct but
    # value-invalid varlen cu_seqlens; with static inputs it profiles the
    # candidates on the caller's real (valid) tensors.
    runner_sel, tactic = tuner.choose_one(op_name, [runner], TuningConfig(), tensors)
    runner_sel(inputs=tensors, tactic=tactic)


@functools.cache
def _proxy_dummies(device_index: int):
    # Signature fillers for paths that never read them; cached so repeat decode
    # calls do not launch fill kernels.
    dev = torch.device("cuda", device_index)
    return (
        torch.zeros((1, 1), dtype=torch.int32, device=dev),
        torch.zeros(1, dtype=torch.int32, device=dev),
    )


@flashinfer_api(trace=msa_proxy_score_trace)
def msa_proxy_score(
    q: torch.Tensor,
    k: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    *,
    page_table: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    causal: bool = True,
    max_seqlen_q: Optional[int] = None,
    max_k_tiles: Optional[int] = None,
    output: Optional[torch.Tensor] = None,
    reduce_heads: bool = False,
    q_offset=None,
) -> torch.Tensor:
    """MSA dense proxy pass for SM120/SM121: per-KV-block max attention
    logits.

    Computes ``max_score[h, t, q]``, the maximum of the unscaled,
    causally-masked ``Q K^T`` logits over the 128 tokens of KV block ``t``,
    for every query token and query head. The output feeds directly into
    :func:`msa_topk_select`. KV blocks beyond a sequence's valid range, or
    entirely above the causal limit, yield ``-inf``.

    There is no softmax and no V: this is MSA pipeline stage 1.

    Single-token decode uses a dim-parallel scalar schedule that streams
    index-K straight to registers. Short multi-token decode (MTP) uses a
    head-fused packed schedule that scores all ``group_size`` heads of a
    kv_head from one shared index-K read. fp8 K and prefill use the general
    schedule; see the dispatch below for the exact regime bounds.

    Parameters
    ----------
    q : torch.Tensor
        ``(total_q, num_qo_heads, 128)``, bf16 or fp16 (the cheap proxy Q).
    k : torch.Tensor
        Flat ``(total_k, num_kv_heads, 128)`` with ``cu_seqlens_k``, or
        paged ``(num_pages, num_kv_heads, 128, 128)`` with ``page_table`` +
        ``seqused_k``. May be fp8 E4M3 (upconverted in-kernel).
        ``num_qo_heads`` must be a multiple of ``num_kv_heads``.
    cu_seqlens_q : torch.Tensor
        ``(batch_size + 1,)`` int32 cumulative query lengths.
    cu_seqlens_k : torch.Tensor, optional
        ``(batch_size + 1,)`` int32 cumulative KV lengths. Required for flat
        KV layout and unused for paged KV layout.
    page_table : torch.Tensor, optional
        Page-table mapping for paged KV layout. Required together with
        ``seqused_k`` when ``k`` is paged.
    seqused_k : torch.Tensor, optional
        Per-sequence valid KV-token counts. Required together with
        ``page_table`` for paged KV layout.
    causal : bool
        Right-aligned causal masking, applied *before* the block max.
    max_seqlen_q : int, optional
        Maximum query length. Required by some CUDA Graph / compilation paths.
    max_k_tiles : int, optional
        Number of KV-block columns in the output; defaults to the maximum
        ``ceil(seqlen_k / 128)`` over the batch.
    output : torch.Tensor, optional
        Pre-allocated float32 output. Shape is
        ``(num_qo_heads, max_k_tiles, total_q)`` normally, or
        ``(1, max_k_tiles, total_q)`` when ``reduce_heads=True``.
    reduce_heads : bool
        If ``True``, max-reduce the per-head ``max_score`` over the query-head
        axis and return a single ``(1, max_k_tiles, total_q)`` score. Use this
        when the query heads are an indexer's proxy heads that collapse to one
        shared block selection per query (MiniMax-M3 indexer semantics).
        Defaults to ``False``: per-head ``max_score``, where each head selects
        its own blocks.
        The reduction is currently a post-kernel ``amax`` over the per-head
        buffer (the kernel is one CTA per head, so a cross-head epilogue would
        need cross-CTA float atomics); folding it into the kernel is a possible
        future optimization (saves materializing the per-head buffer).
    q_offset : int or torch.Tensor, optional
        Optional query-position offset used by the causal alignment logic.

    Returns
    -------
    torch.Tensor
        Float32 ``max_score`` ready for :func:`msa_topk_select`:
        ``(num_qo_heads, max_k_tiles, total_q)``, or
        ``(1, max_k_tiles, total_q)`` when ``reduce_heads=True``.
    """
    import cutlass
    import cutlass.cute as cute

    from .cute_dsl.proxy_score_sm12x import (
        MsaProxyScoreDecodePackedSm12x,
        MsaProxyScoreDecodeStreamSm12x,
        MsaProxyScoreSm12x,
    )
    from ._common import (
        _compile_cache,
        _cutlass_dtype,
        _fake,
        _q_offset_tensor,
    )

    if not is_sm12x_supported(q.device):
        raise RuntimeError("msa_proxy_score requires SM120 or SM121 and CUDA >= 12.8")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"q must be bf16 or fp16, got {q.dtype}")
    total_q, num_qo_heads, head_dim = q.shape
    if head_dim != 128:
        raise ValueError(f"head_dim must be 128, got {head_dim}")
    num_kv_heads = k.shape[1]
    if num_qo_heads % num_kv_heads != 0:
        raise ValueError("num_qo_heads must be a multiple of num_kv_heads")
    kv_fp8 = k.dtype == torch.float8_e4m3fn
    if not kv_fp8 and k.dtype != q.dtype:
        raise ValueError("k dtype must match q (or be float8_e4m3fn)")
    dev = q.device

    paged = page_table is not None
    if paged:
        if seqused_k is None:
            raise ValueError("paged proxy requires seqused_k")
        if k.ndim != 4 or k.shape[2] != _BLK_KV or k.shape[3] != head_dim:
            raise ValueError(
                f"paged k must be (num_pages, num_kv_heads, {_BLK_KV}, {head_dim})"
            )
        batch_size = seqused_k.numel()
        cu_k = torch.zeros(batch_size + 1, dtype=torch.int32, device=dev)
        cu_k[1:] = seqused_k.to(dev).cumsum(0)
        pt_dev = page_table.contiguous()
    else:
        if cu_seqlens_k is None:
            raise ValueError("flat proxy requires cu_seqlens_k")
        if k.ndim != 3:
            raise ValueError("flat k must be (total_k, num_kv_heads, head_dim)")
        cu_k = cu_seqlens_k.to(dev)
        pt_dev = _proxy_dummies(dev.index)[0]
        batch_size = cu_k.numel() - 1

    cu_q_dev = cu_seqlens_q.to(dev)
    max_seqlen_q, max_k_tiles = _resolve_proxy_dims(
        cu_seqlens_q, cu_k, max_seqlen_q, max_k_tiles, output
    )

    per_head_shape = (num_qo_heads, max_k_tiles, total_q)
    final_shape = (1, max_k_tiles, total_q) if reduce_heads else per_head_shape
    if output is not None:
        if output.shape != final_shape:
            raise ValueError(f"output must be {final_shape}")
        if output.dtype != torch.float32:
            raise ValueError("output must be float32")

    # The kernel always writes the per-head buffer; reduce_heads then collapses
    # it. When not reducing and an output was given, that buffer *is* the output.
    if reduce_heads or output is None:
        per_head = torch.empty(per_head_shape, dtype=torch.float32, device=dev)
    else:
        per_head = output

    group_size = num_qo_heads // num_kv_heads
    # Single-token decode uses the dim-parallel stream schedule (see
    # MsaProxyScoreDecodeStreamSm12x). total_q == batch_size guarantees exactly
    # one q token per request, which its token == batch indexing relies on.
    use_stream = (
        not kv_fp8 and max_seqlen_q == 1 and total_q == batch_size and group_size <= 8
    )
    # Right-aligned decode on the stream path computes the causal limit
    # in-kernel, so no offset tensor (and its build kernels) is needed.
    qoff_default = q_offset is None
    if use_stream and qoff_default:
        qoff_dev = _proxy_dummies(dev.index)[1]
    else:
        qoff_dev = _q_offset_tensor(q_offset, cu_q_dev, cu_k, dev)

    # Head-fused decode path: pack group_size heads x pack_q_len q-tokens into one
    # 64-row MMA tile so index-K is read once per (batch, kv_head). Outside the regime
    # (prefill, fp8 K, group_size not dividing 64) use the general schedule.
    _PACK_ROWS = 64  # bf16 MMA q-tile rows (== m_block_size)
    use_packed = (
        not use_stream
        and not kv_fp8
        and group_size >= 2
        and _PACK_ROWS % group_size == 0
        and max_seqlen_q <= _PACK_ROWS // group_size
    )
    pack_q_len = _PACK_ROWS // group_size if use_packed else 0

    # group_size keys the packed/stream schedules: it is constexpr-baked into
    # the gather/epilogue, so each factorization is its own kernel.
    key = (
        "proxy",
        str(q.dtype),
        causal,
        paged,
        kv_fp8,
        use_stream,
        use_packed,
        group_size if use_stream else pack_q_len,
        qoff_default if use_stream else None,
    )
    compiled = _compile_cache.get(key)
    if compiled is None:
        cdt = _cutlass_dtype(q.dtype)
        kdt = _cutlass_dtype(k.dtype)
        i32 = _cutlass_dtype(torch.int32)
        f32 = _cutlass_dtype(torch.float32)
        s_tq, s_hq, s_tk, s_hkv, s_b1, s_b0, s_pb, s_pm, s_mt = (
            cute.sym_int() for _ in range(9)
        )
        k_shape = (s_tk, s_hkv, _BLK_KV, head_dim) if paged else (s_tk, s_hkv, head_dim)
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        kernel_obj: Union["MsaProxyScoreDecodeStreamSm12x", "MsaProxyScoreSm12x"]
        if use_stream:
            kernel_obj = MsaProxyScoreDecodeStreamSm12x(
                head_dim=head_dim,
                group_size=group_size,
                is_causal=causal,
                paged=paged,
                qoff_default=qoff_default,
            )
        elif use_packed:
            kernel_obj = MsaProxyScoreDecodePackedSm12x(
                head_dim=head_dim,
                m_block_size=64,
                n_block_size=_BLK_KV,
                num_threads=128,
                is_causal=causal,
                paged=paged,
                kv_fp8=kv_fp8,
                qhead_per_kv=group_size,
                pack_q_len=pack_q_len,
            )
        else:
            kernel_obj = MsaProxyScoreSm12x(
                head_dim=head_dim,
                m_block_size=64,
                n_block_size=_BLK_KV,
                num_threads=128,
                is_causal=causal,
                paged=paged,
                kv_fp8=kv_fp8,
            )
        compiled = cute.compile(
            kernel_obj,
            _fake(cdt, (s_tq, s_hq, head_dim)),
            _fake(kdt, k_shape),
            _fake(i32, (s_pb, s_pm), align=4),
            _fake(f32, (s_hq, s_mt, s_tq), align=4),
            _fake(i32, (s_b1,), align=4),
            _fake(i32, (s_b1,), align=4),
            _fake(i32, (s_b0,), align=4),
            cutlass.Int32(1),
            cutlass.Int32(1),
            cutlass.Int32(1),
            cutlass.Int32(1),
            cutlass.Int32(1),
            stream_fake,
            options="--enable-tvm-ffi",
        )
        _compile_cache[key] = compiled

    # Base grid CTAs (feeds split-K, see _proxy_split_k): the packed path is one
    # CTA per (batch, kv_head); the general path one per (q-tile, batch, head) with
    # 64-row q-tiles. num_splits is a runtime arg, not part of the cache key.
    if use_packed:
        base_ctas = batch_size * num_kv_heads
    else:
        base_ctas = ((max_seqlen_q + 63) // 64) * batch_size * num_qo_heads

    def _call(tensors, ns):
        q_, k_, pt_, ph_, cq_, ck_, qoff_ = tensors
        compiled(
            q_,
            k_,
            pt_,
            ph_,
            cq_,
            ck_,
            qoff_,
            int(max_seqlen_q),
            int(batch_size),
            int(num_qo_heads),
            int(max_k_tiles),
            ns,
        )
        return ph_

    if use_stream:
        # One CTA per (KV block, token, kv_head): the grid is already maximally
        # split, so there is no split factor to tune.
        _call([q, k, pt_dev, per_head, cu_q_dev, cu_k, qoff_dev], 1)
    else:
        _run_proxy_autotuned(
            "msa_proxy_score",
            _call,
            [q, k, pt_dev, per_head, cu_q_dev, cu_k, qoff_dev],
            max_seqlen_q=max_seqlen_q,
            max_k_tiles=max_k_tiles,
            base_ctas=base_ctas,
            device=dev,
            heuristic=_proxy_split_k_bf16,
            causal=causal,
            paged=paged,
            reduce_heads=reduce_heads,
        )

    if not reduce_heads:
        return per_head
    if output is None:
        output = torch.empty(final_shape, dtype=torch.float32, device=dev)
    torch.amax(per_head, dim=0, keepdim=True, out=output)
    return output


def _quantize_qk_to_nvfp4(
    x: torch.Tensor,
    global_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Test/bench/trace glue (deployment quantizes upstream): quantize a
    ``(total, num_heads, 128)`` bf16/fp16 proxy Q or K to packed NVFP4 in the
    128x4 tiled scale layout; returns ``(x_fp4, block scales, 1/global_scale)``."""
    from flashinfer import nvfp4_quantize

    if x.ndim != 3 or x.shape[2] != 128:
        raise ValueError(f"x must be (total, num_heads, 128), got {tuple(x.shape)}")
    total, num_heads, head_dim = x.shape
    x2d = x.reshape(-1, head_dim)
    if global_scale is None:
        global_scale = (448.0 * 6.0) / x2d.float().abs().max().clamp_min(1e-12)
    gsf = torch.as_tensor([float(global_scale)], dtype=torch.float32, device=x.device)
    # Default sfLayout is the cuBLAS 128x4 tiled layout; SF row = token*num_heads
    # + head (the natural (total, num_heads, d).reshape(-1, d) row order).
    xq, sf = nvfp4_quantize(x2d, gsf, sf_vec_size=_SF_VEC_SIZE)
    x_fp4 = xq.view(torch.uint8).reshape(total, num_heads, head_dim // 2)
    x_scale = sf.view(torch.uint8).reshape(-1)  # flat, 128x4 tiled
    return x_fp4, x_scale, 1.0 / float(global_scale)


@flashinfer_api(trace=msa_proxy_score_fp4_trace)
def msa_proxy_score_fp4(
    q_fp4: torch.Tensor,
    k_fp4: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    q_global_scale: float,
    k_global_scale: float,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    *,
    page_table: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    causal: bool = True,
    max_seqlen_q: Optional[int] = None,
    max_k_tiles: Optional[int] = None,
    output: Optional[torch.Tensor] = None,
    reduce_heads: bool = False,
    q_offset=None,
) -> torch.Tensor:
    """NVFP4 MSA dense proxy pass for SM120/SM121 (the FP4 counterpart of
    :func:`msa_proxy_score`).

    Same contract and output as :func:`msa_proxy_score` (per-KV-block max of
    the unscaled, causally-masked ``Q K^T`` logits), but Q/K arrive pre-quantized
    as packed NVFP4 (``e2m1`` + per-16 ``e4m3`` block scales + per-tensor global
    scales), so the index K is read from HBM at ~4 bits/elem. The full-KV index
    read is the dominant decode-step cost, so this is the bandwidth-saving
    path; :func:`msa_proxy_score` stays as the bf16 precision reference.

    Numerics equal a torch dequant of the same packed inputs (not the bf16
    reference, which differs by fp4 rounding). The two global scales are folded
    into the logits as ``q_global_scale * k_global_scale`` before the block-max.
    Short q (decode, including multi-token verify) uses a head-fused packed
    schedule on the fp4 tensor cores (``MmaMXF4NVF4Op``) that scores all
    ``group_size`` heads of a kv_head from one shared index-K read; longer q
    uses the general tensor-core schedule. Both flat and paged K are supported;
    see the dispatch below for the exact regime bounds.

    Parameters
    ----------
    q_fp4 : torch.Tensor
        ``(total_q, num_qo_heads, 64)`` uint8, packed e2m1 (2 nibbles/byte).
    k_fp4 : torch.Tensor
        Flat ``(total_k, num_kv_heads, 64)`` with ``cu_seqlens_k``, or paged
        ``(num_pages, num_kv_heads, 128, 64)`` with ``page_table`` +
        ``seqused_k``. ``num_qo_heads`` must be a multiple of ``num_kv_heads``.
    q_scale, k_scale : torch.Tensor
        Flat uint8 e4m3 block scales in the cuBLAS 128x4 tiled layout, indexed by
        logical row ``token*num_heads + head`` (paged K: ``(page*num_kv_heads +
        kv_head)*128 + token_in_page``). Produced by :func:`nvfp4_quantize`
        (128x4 layout, ``sf_vec_size=16``); shared with the attention + decode kernels.
    q_global_scale, k_global_scale : float
        Per-tensor inverse global scales (``1 / global_scale``).
    cu_seqlens_q, cu_seqlens_k : torch.Tensor
        As in :func:`msa_proxy_score`.
    page_table : torch.Tensor, optional
        As in :func:`msa_proxy_score`.
    seqused_k : torch.Tensor, optional
        As in :func:`msa_proxy_score`.
    causal : bool
        As in :func:`msa_proxy_score`.
    max_seqlen_q : int, optional
        As in :func:`msa_proxy_score`.
    max_k_tiles : int, optional
        As in :func:`msa_proxy_score`.
    output : torch.Tensor, optional
        As in :func:`msa_proxy_score`.
    reduce_heads : bool
        As in :func:`msa_proxy_score`.
    q_offset : int or torch.Tensor, optional
        As in :func:`msa_proxy_score`.

    Returns
    -------
    torch.Tensor
        Float32 ``max_score`` for :func:`msa_topk_select`:
        ``(num_qo_heads, max_k_tiles, total_q)``, or
        ``(1, max_k_tiles, total_q)`` when ``reduce_heads=True``.
    """
    import cutlass
    import cutlass.cute as cute

    from .cute_dsl.proxy_score_fp4_sm12x import (
        MsaProxyScoreFp4MmaDecodePackedSm12x,
        MsaProxyScoreFp4MmaSm12x,
    )
    from ._common import (
        _compile_cache,
        _cutlass_dtype,
        _fake,
        _q_offset_tensor,
    )

    if not is_sm12x_supported(q_fp4.device):
        raise RuntimeError(
            "msa_proxy_score_fp4 requires SM120 or SM121 and CUDA >= 12.8"
        )
    if q_fp4.dtype != torch.uint8 or k_fp4.dtype != torch.uint8:
        raise ValueError("q_fp4/k_fp4 must be packed uint8 (e2m1x2)")
    if q_scale.dtype != torch.uint8 or k_scale.dtype != torch.uint8:
        raise ValueError("q_scale/k_scale must be uint8 (e4m3 bytes)")
    hd2 = _BLK_KV // 2  # 64 packed bytes / row
    q_scale = q_scale.reshape(-1)
    k_scale = k_scale.reshape(-1)
    total_q, num_qo_heads, q_last = q_fp4.shape
    if q_last != hd2:
        raise ValueError(f"q_fp4 last dim must be {hd2} (head_dim/2), got {q_last}")
    num_kv_heads = k_fp4.shape[1]
    if num_qo_heads % num_kv_heads != 0:
        raise ValueError("num_qo_heads must be a multiple of num_kv_heads")
    dev = q_fp4.device

    paged = page_table is not None
    if paged:
        if seqused_k is None:
            raise ValueError("paged proxy requires seqused_k")
        if k_fp4.ndim != 4 or k_fp4.shape[2] != _BLK_KV or k_fp4.shape[3] != hd2:
            raise ValueError(
                f"paged k_fp4 must be (num_pages, num_kv_heads, {_BLK_KV}, {hd2})"
            )
        batch_size = seqused_k.numel()
        cu_k = torch.zeros(batch_size + 1, dtype=torch.int32, device=dev)
        cu_k[1:] = seqused_k.to(dev).cumsum(0)
        pt_dev = page_table.contiguous()
    else:
        if cu_seqlens_k is None:
            raise ValueError("flat proxy requires cu_seqlens_k")
        if k_fp4.ndim != 3:
            raise ValueError("flat k_fp4 must be (total_k, num_kv_heads, 64)")
        cu_k = cu_seqlens_k.to(dev)
        pt_dev = torch.zeros((1, 1), dtype=torch.int32, device=dev)
        batch_size = cu_k.numel() - 1

    cu_q_dev = cu_seqlens_q.to(dev)
    qoff_dev = _q_offset_tensor(q_offset, cu_q_dev, cu_k, dev)
    max_seqlen_q, max_k_tiles = _resolve_proxy_dims(
        cu_seqlens_q, cu_k, max_seqlen_q, max_k_tiles, output
    )

    per_head_shape = (num_qo_heads, max_k_tiles, total_q)
    final_shape = (1, max_k_tiles, total_q) if reduce_heads else per_head_shape
    if output is not None:
        if output.shape != final_shape:
            raise ValueError(f"output must be {final_shape}")
        if output.dtype != torch.float32:
            raise ValueError("output must be float32")
    if reduce_heads or output is None:
        per_head = torch.empty(per_head_shape, dtype=torch.float32, device=dev)
    else:
        per_head = output

    # Head-fused decode path, as in msa_proxy_score, but on the 128-row fp4-MMA
    # tile and with no fp8-K exclusion.
    _PACK_ROWS = MsaProxyScoreFp4MmaSm12x._M
    group_size = num_qo_heads // num_kv_heads
    use_packed = (
        group_size >= 2
        and _PACK_ROWS % group_size == 0
        and max_seqlen_q <= _PACK_ROWS // group_size
    )
    pack_q_len = _PACK_ROWS // group_size if use_packed else 0

    # Base grid CTAs (feeds split-K) and cache key, as in msa_proxy_score.
    if use_packed:
        base_ctas = batch_size * num_kv_heads
    else:
        base_ctas = ((max_seqlen_q + 127) // 128) * batch_size * num_qo_heads

    key = ("proxy_fp4", causal, paged, use_packed, pack_q_len)
    compiled = _compile_cache.get(key)
    if compiled is None:
        u8 = _cutlass_dtype(torch.uint8)
        i32 = _cutlass_dtype(torch.int32)
        f32 = _cutlass_dtype(torch.float32)
        (
            s_tq,
            s_hq,
            s_tk,
            s_hkv,
            s_b1,
            s_b0,
            s_pb,
            s_pm,
            s_mt,
            s_qsf,
            s_ksf,
        ) = (cute.sym_int() for _ in range(11))
        if paged:
            k_shape: tuple = (s_tk, s_hkv, _BLK_KV, hd2)
        else:
            k_shape = (s_tk, s_hkv, hd2)
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        kernel_obj: "MsaProxyScoreFp4MmaSm12x"
        if use_packed:
            kernel_obj = MsaProxyScoreFp4MmaDecodePackedSm12x(
                head_dim=_BLK_KV,
                is_causal=causal,
                paged=paged,
                qhead_per_kv=group_size,
                pack_q_len=pack_q_len,
            )
        else:
            kernel_obj = MsaProxyScoreFp4MmaSm12x(
                head_dim=_BLK_KV,
                is_causal=causal,
                paged=paged,
            )
        compiled = cute.compile(
            kernel_obj,
            _fake(u8, (s_tq, s_hq, hd2), align=4),
            _fake(u8, k_shape, align=4),
            _fake(u8, (s_qsf,), align=4),
            _fake(u8, (s_ksf,), align=4),
            _fake(i32, (s_pb, s_pm), align=4),
            _fake(f32, (s_hq, s_mt, s_tq), align=4),
            _fake(i32, (s_b1,), align=4),
            _fake(i32, (s_b1,), align=4),
            _fake(i32, (s_b0,), align=4),
            cutlass.Float32(1.0),
            cutlass.Float32(1.0),
            cutlass.Int32(1),
            cutlass.Int32(1),
            cutlass.Int32(1),
            cutlass.Int32(1),
            cutlass.Int32(1),  # num_splits
            stream_fake,
            options="--enable-tvm-ffi",
        )
        _compile_cache[key] = compiled

    def _call(tensors, ns):
        qf, kf, qs, ks, pt_, ph_, cq_, ck_, qoff_ = tensors
        compiled(
            qf,
            kf,
            qs,
            ks,
            pt_,
            ph_,
            cq_,
            ck_,
            qoff_,
            float(q_global_scale),
            float(k_global_scale),
            int(max_seqlen_q),
            int(batch_size),
            int(num_qo_heads),
            int(max_k_tiles),
            ns,
        )
        return ph_

    _run_proxy_autotuned(
        "msa_proxy_score_fp4",
        _call,
        [q_fp4, k_fp4, q_scale, k_scale, pt_dev, per_head, cu_q_dev, cu_k, qoff_dev],
        max_seqlen_q=max_seqlen_q,
        max_k_tiles=max_k_tiles,
        base_ctas=base_ctas,
        device=dev,
        heuristic=_proxy_split_k_fp4,
        causal=causal,
        paged=paged,
        reduce_heads=reduce_heads,
    )

    if not reduce_heads:
        return per_head
    if output is None:
        output = torch.empty(final_shape, dtype=torch.float32, device=dev)
    torch.amax(per_head, dim=0, keepdim=True, out=output)
    return output
