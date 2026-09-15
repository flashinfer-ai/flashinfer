"""SM90 (Hopper) dispatch for the MSA ops.

Routes to the Hopper CuTe DSL backends in ``cute_dsl/*_sm90.py``. Selection is
on capability, never on measured shape thresholds: a schedule is chosen only
where it is the one that can run the shape correctly.
"""

from typing import Optional

import torch

_BLK_KV = 128


def _prefix_lens(cu_seqlens_q: torch.Tensor, seqused_k: torch.Tensor) -> torch.Tensor:
    """Tokens already resident before this chunk, per sequence.

    The proxy kernels align the causal mask on it, which is what ``q_offset``
    means for a chunked prefill: seqused_k counts the whole sequence, so the
    query rows in this call sit at the end of it.
    """
    qlen = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).to(torch.int32)
    return (seqused_k.to(torch.int32) - qlen).contiguous()


def proxy_score_sm90(
    q: torch.Tensor,
    k: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    page_table: torch.Tensor,
    seqused_k: torch.Tensor,
    per_head: torch.Tensor,
    *,
    max_seqlen_q: int,
    batch_size: int,
    kv_fp8: bool,
    q_offset: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """MSA proxy score on Hopper. Writes ``per_head`` (Hq, max_k_tiles, total_q)."""
    total_q = q.shape[0]
    sq = total_q // batch_size if batch_size else 0
    decode = max_seqlen_q <= 4 and batch_size * max_seqlen_q == total_q and sq == max_seqlen_q

    cu = cu_seqlens_q.to(torch.int32).contiguous()
    pt = page_table.to(torch.int32).contiguous()
    sk = seqused_k.to(torch.int32).contiguous()

    if decode:
        if kv_fp8:
            from .cute_dsl.proxy_score_decode_sm90 import run as _decode
        else:
            from .cute_dsl.proxy_score_decode_bf16_sm90 import run as _decode
        _decode(q, k, cu, pt, sk, per_head)
        return per_head

    if not kv_fp8:
        raise NotImplementedError(
            "SM90 proxy-score prefill requires an fp8 e4m3 index cache; "
            "bf16 is supported for decode only"
        )
    from .cute_dsl.proxy_score_prefill_sm90 import run as _prefill

    pfx = q_offset.to(torch.int32).contiguous() if q_offset is not None else _prefix_lens(cu, sk)
    _prefill(q, k, cu, pt, sk, pfx, per_head)
    return per_head


def topk_select_sm90(
    max_score: torch.Tensor,
    topk: int,
    output: torch.Tensor,
    *,
    num_valid_pages=None,
    force_begin_blocks: int = 0,
    force_end_blocks: int = 0,
) -> torch.Tensor:
    """MSA top-k block selection on Hopper. Writes sorted indices into ``output``.

    The kernel reads validity from the ``-inf`` tiles ``msa_proxy_score`` already
    writes, so it needs no sequence-length arguments.
    """
    if topk != 16:
        raise NotImplementedError(f"SM90 msa_topk_select supports topk=16 only, got {topk}")
    if force_begin_blocks or force_end_blocks:
        raise NotImplementedError("SM90 msa_topk_select does not implement forced blocks")
    if num_valid_pages is not None and not (
        isinstance(num_valid_pages, int) and num_valid_pages == max_score.shape[1]
    ):
        raise NotImplementedError(
            "SM90 msa_topk_select clamps via the -inf tiles in max_score; "
            "an explicit num_valid_pages is not implemented"
        )
    from .cute_dsl.topk_select_sm90 import run as _topk

    # The kernel emits (Hq, total_q, topk); this op returns (total_q, Hq, topk).
    # Permuting the caller's buffer gives the kernel's view for free whenever that
    # view is contiguous -- always so for the MQA indexer (Hq == 1), which is the
    # MiniMax-M3 path. Staging through a temporary instead costs an allocation and
    # a copy launch, ~12us against a ~7us kernel, so only pay it when forced.
    hq, _, total_q = max_score.shape
    view = output.permute(1, 0, 2)
    if view.is_contiguous():
        _topk(max_score, None, None, view)
        return output
    tmp = torch.empty((hq, total_q, topk), dtype=torch.int32, device=max_score.device)
    _topk(max_score, None, None, tmp)
    output.copy_(tmp.permute(1, 0, 2))
    return output


def _as_packed_kv(k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Recover the (num_pages, Hkv, page, 2*D) cache the SM90 decode kernel reads.

    vLLM stores MSA K and V interleaved in one allocation and hands out halves;
    when that is what we were given the packed view is free. Materializing it
    otherwise would copy the whole cache every call, so refuse instead.
    """
    if k.dtype != v.dtype or k.shape != v.shape or k.stride() != v.stride():
        raise NotImplementedError("SM90 sparse decode needs matching k/v layouts")
    d = k.shape[-1]
    same_buf = k.untyped_storage().data_ptr() == v.untyped_storage().data_ptr()
    adjacent = v.data_ptr() - k.data_ptr() == d * k.element_size()
    if not (same_buf and adjacent and k.stride()[-1] == 1):
        raise NotImplementedError(
            "SM90 sparse decode requires K and V interleaved in one cache "
            "(v must be the second half of k's last dim); separate allocations "
            "would need a full-cache copy per call"
        )
    base = k.as_strided(k.shape[:-1] + (2 * d,), k.stride()[:-1] + (1,), 0)
    return base


def sparse_decode_sm90(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_indices: torch.Tensor,
    page_table: torch.Tensor,
    seqused_k: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """MSA sparse decode attention on Hopper. Writes ``out``."""
    from .cute_dsl.sparse_decode_sm90 import run as _decode

    kv = _as_packed_kv(k, v)
    _decode(
        q,
        kv,
        q2k_indices.to(torch.int32).contiguous(),
        page_table.to(torch.int32).contiguous(),
        seqused_k.to(torch.int32).contiguous(),
        out,
    )
    return out


def sparse_prefill_sm90(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    page_table: torch.Tensor,
    seqused_k: torch.Tensor,
    out: torch.Tensor,
    *,
    q_offset: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """MSA sparse prefill attention on Hopper. Writes ``out``."""
    from .cute_dsl.sparse_prefill_sm90 import run as _prefill

    kv = _as_packed_kv(k, v)
    cu = cu_seqlens_q.to(torch.int32).contiguous()
    sk = seqused_k.to(torch.int32).contiguous()
    pfx = q_offset.to(torch.int32).contiguous() if q_offset is not None else _prefix_lens(cu, sk)
    _prefill(
        q,
        kv,
        q2k_indices.to(torch.int32).contiguous(),
        cu,
        page_table.to(torch.int32).contiguous(),
        sk,
        pfx,
        out,
    )
    return out
