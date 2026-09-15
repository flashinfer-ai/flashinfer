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
