"""Backend-independent, small-shape PyTorch reference for paged MLA.

The reference models a paged logical KV sequence from dense page-table
metadata.  It deliberately uses only PyTorch tensor operations and never calls
FlashInfer wrapper or functional attention kernels.

LSE values are returned as ``[total_queries, num_heads]``.  ``basee`` is the
natural logarithm of the softmax denominator and ``base2`` is that same value
divided by ``log(2)``.  ``none`` returns ``None`` for LSE.
"""

from dataclasses import dataclass
from typing import Literal, Optional

import torch


_LSEMode = Literal["none", "base2", "basee"]
_KVLayout = Literal["combined", "adjacent-split", "independent-split"]
_OutputScale = Literal["none", "per-tensor"]
_ScaleMode = Literal["default", "kv-per-tensor", "bmm-scalar", "bmm-tensor"]


@dataclass(frozen=True)
class MLAReferenceContract:
    """Five-mode run contract consumed by :func:`mla_paged_attention_reference`.

    ``skip_softmax`` records the wrapper capability mode.  The runtime
    threshold optimization preserves mathematical softmax output, so the
    independent reference deliberately computes the unoptimized result.
    """

    lse_mode: _LSEMode = "none"
    kv_layout: _KVLayout = "independent-split"
    output_dtype: Optional[torch.dtype] = None
    output_scale: _OutputScale = "none"
    scale_mode: _ScaleMode = "default"
    skip_softmax: bool = False

    def __post_init__(self) -> None:
        if self.lse_mode not in ("none", "base2", "basee"):
            raise ValueError(f"unsupported LSE mode {self.lse_mode!r}")
        if self.kv_layout not in (
            "combined",
            "adjacent-split",
            "independent-split",
        ):
            raise ValueError(f"unsupported KV layout {self.kv_layout!r}")
        if self.output_scale not in ("none", "per-tensor"):
            raise ValueError(f"unsupported output scale {self.output_scale!r}")
        if self.scale_mode not in (
            "default",
            "kv-per-tensor",
            "bmm-scalar",
            "bmm-tensor",
        ):
            raise ValueError(f"unsupported scale mode {self.scale_mode!r}")


def _scalar_scale(name: str, value: object) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(
                f"{name} must be a scalar tensor, got {value.numel()} values"
            )
        return float(value.item())
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    raise ValueError(f"{name} must be a scalar float or tensor")


def _adjacent_split_views(ckv_cache: torch.Tensor, kpe_cache: torch.Tensor) -> bool:
    return (
        ckv_cache.device == kpe_cache.device
        and ckv_cache.dtype == kpe_cache.dtype
        and ckv_cache.shape[:-1] == kpe_cache.shape[:-1]
        and ckv_cache.stride()[:-1] == kpe_cache.stride()[:-1]
        and ckv_cache.stride(-1) == 1
        and kpe_cache.stride(-1) == 1
        and ckv_cache.untyped_storage().data_ptr()
        == kpe_cache.untyped_storage().data_ptr()
        and kpe_cache.storage_offset()
        == ckv_cache.storage_offset() + ckv_cache.shape[-1]
    )


def _resolve_kv_cache(
    *,
    ckv_cache: Optional[torch.Tensor],
    kpe_cache: Optional[torch.Tensor],
    kv_cache: Optional[torch.Tensor],
    contract: MLAReferenceContract,
    ckv_width: int,
    kpe_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if contract.kv_layout == "combined":
        if kv_cache is None:
            raise ValueError("combined KV layout requires kv_cache")
        if kv_cache.ndim != 3:
            raise ValueError("kv_cache must have shape [pages, page_size, dimensions]")
        if kv_cache.shape[-1] != ckv_width + kpe_width:
            raise ValueError("combined KV width does not match split cache widths")
        return kv_cache[..., :ckv_width], kv_cache[..., ckv_width:]

    if ckv_cache is None or kpe_cache is None:
        raise ValueError("split KV layouts require ckv_cache and kpe_cache")
    if ckv_cache.ndim != 3 or kpe_cache.ndim != 3:
        raise ValueError("split KV cache tensors must have rank 3")
    if ckv_cache.shape[:-1] != kpe_cache.shape[:-1]:
        raise ValueError("split KV cache tensors must share page dimensions")
    if contract.kv_layout == "adjacent-split" and not _adjacent_split_views(
        ckv_cache, kpe_cache
    ):
        raise ValueError(
            "adjacent-split KV layout requires adjacent last-dimension views"
        )
    return ckv_cache, kpe_cache


def _validate_metadata(
    qo_indptr: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    page_size: int,
) -> None:
    if qo_indptr.ndim != 1 or block_tables.ndim != 2 or seq_lens.ndim != 1:
        raise ValueError("MLA reference metadata ranks must be 1, 2, and 1")
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    batch_size = qo_indptr.numel() - 1
    if (
        batch_size < 0
        or block_tables.shape[0] != batch_size
        or seq_lens.numel() != batch_size
    ):
        raise ValueError("MLA reference metadata batch dimensions disagree")
    qo_values = qo_indptr.to(dtype=torch.int64, device="cpu")
    if (
        qo_values.numel() == 0
        or qo_values[0].item() != 0
        or bool(torch.any(qo_values[1:] < qo_values[:-1]))
    ):
        raise ValueError("qo_indptr must begin at zero and be nondecreasing")


def _apply_scales(
    ckv: torch.Tensor,
    kpe: torch.Tensor,
    *,
    contract: MLAReferenceContract,
    ckv_scale: object,
    kpe_scale: object,
    bmm1_scale: object,
    bmm2_scale: object,
) -> tuple[torch.Tensor, torch.Tensor, Optional[float], float]:
    if contract.scale_mode == "kv-per-tensor":
        if ckv_scale is None or kpe_scale is None:
            raise ValueError(
                "kv-per-tensor scale mode requires ckv_scale and kpe_scale"
            )
        ckv = ckv.float() * _scalar_scale("ckv_scale", ckv_scale)
        kpe = kpe.float() * _scalar_scale("kpe_scale", kpe_scale)
        return ckv, kpe, None, 1.0
    if ckv_scale is not None or kpe_scale is not None:
        raise ValueError("CKV/KPE scales require kv-per-tensor scale mode")
    if contract.scale_mode in ("bmm-scalar", "bmm-tensor"):
        if bmm1_scale is None or bmm2_scale is None:
            raise ValueError("BMM scale modes require bmm1_scale and bmm2_scale")
        if contract.scale_mode == "bmm-scalar" and (
            isinstance(bmm1_scale, torch.Tensor) or isinstance(bmm2_scale, torch.Tensor)
        ):
            raise ValueError("bmm-scalar scale mode requires Python scalar scales")
        if contract.scale_mode == "bmm-tensor" and not (
            isinstance(bmm1_scale, torch.Tensor)
            and isinstance(bmm2_scale, torch.Tensor)
        ):
            raise ValueError("bmm-tensor scale mode requires scalar tensor scales")
        return (
            ckv.float(),
            kpe.float(),
            _scalar_scale("bmm1_scale", bmm1_scale),
            _scalar_scale("bmm2_scale", bmm2_scale),
        )
    if bmm1_scale is not None or bmm2_scale is not None:
        raise ValueError("BMM scales require a BMM scale mode")
    return ckv.float(), kpe.float(), None, 1.0


def mla_paged_attention_reference(
    *,
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    qo_indptr: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    page_size: int,
    contract: MLAReferenceContract,
    ckv_cache: Optional[torch.Tensor] = None,
    kpe_cache: Optional[torch.Tensor] = None,
    kv_cache: Optional[torch.Tensor] = None,
    sm_scale: float = 1.0,
    ckv_scale: object = None,
    kpe_scale: object = None,
    bmm1_scale: object = None,
    bmm2_scale: object = None,
    o_scale: object = None,
    causal: bool = False,
    sinks: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Compute paged MLA output without using FlashInfer backend kernels.

    Query tensors have shape ``[total_queries, heads, split_dim]``; cache
    tensors have shape ``[pages, page_size, split_dim]``.  ``block_tables`` and
    ``seq_lens`` select each logical KV sequence.  The optional FP8 scales use
    explicit real-value semantics: ``real_kv = quantized_kv * kv_scale`` and
    ``quantized_out = real_out / o_scale``.
    """
    _validate_metadata(qo_indptr, block_tables, seq_lens, page_size)
    if q_nope.ndim != 3 or q_pe.ndim != 3 or q_nope.shape[:2] != q_pe.shape[:2]:
        raise ValueError(
            "query splits must have shape [total_queries, heads, dimensions]"
        )
    ckv_cache, kpe_cache = _resolve_kv_cache(
        ckv_cache=ckv_cache,
        kpe_cache=kpe_cache,
        kv_cache=kv_cache,
        contract=contract,
        ckv_width=q_nope.shape[-1],
        kpe_width=q_pe.shape[-1],
    )
    ckv_real, kpe_real, bmm1_override, bmm2 = _apply_scales(
        ckv_cache,
        kpe_cache,
        contract=contract,
        ckv_scale=ckv_scale,
        kpe_scale=kpe_scale,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
    )
    if q_nope.shape[-1] != ckv_real.shape[-1] or q_pe.shape[-1] != kpe_real.shape[-1]:
        raise ValueError("query and KV split widths must match")

    total_queries, num_heads = q_nope.shape[:2]
    if not isinstance(causal, bool):
        raise ValueError("causal must be a boolean")
    if sinks is not None:
        if sinks.ndim != 1 or sinks.numel() != num_heads:
            raise ValueError("sinks must have shape [num_heads]")
        sinks = sinks.to(device=q_nope.device, dtype=torch.float32)
    output = torch.empty(
        (total_queries, num_heads, ckv_real.shape[-1]),
        dtype=torch.float32,
        device=q_nope.device,
    )
    lse = (
        torch.empty(
            (total_queries, num_heads), dtype=torch.float32, device=q_nope.device
        )
        if contract.lse_mode != "none"
        else None
    )
    qo_values = qo_indptr.to(dtype=torch.int64, device="cpu")
    kv_values = seq_lens.to(dtype=torch.int64, device="cpu")

    for batch_idx in range(qo_values.numel() - 1):
        q_start, q_end = int(qo_values[batch_idx]), int(qo_values[batch_idx + 1])
        kv_len = int(kv_values[batch_idx])
        if q_end > total_queries or kv_len < 0:
            raise ValueError("metadata points outside the supplied query/KV tensors")
        pages_needed = (kv_len + page_size - 1) // page_size
        page_ids = block_tables[batch_idx, :pages_needed].to(dtype=torch.long)
        if pages_needed and (
            page_ids.numel() != pages_needed or int(page_ids.max()) >= ckv_real.shape[0]
        ):
            raise ValueError("block table points outside the KV cache")
        logical_ckv = ckv_real[page_ids].reshape(-1, ckv_real.shape[-1])[:kv_len]
        logical_kpe = kpe_real[page_ids].reshape(-1, kpe_real.shape[-1])[:kv_len]
        q_ckv = q_nope[q_start:q_end].float()
        q_kpe = q_pe[q_start:q_end].float()
        logit_scale = float(sm_scale) if bmm1_override is None else bmm1_override
        logits = (
            torch.einsum("qhd,kd->qhk", q_ckv, logical_ckv)
            + torch.einsum("qhd,kd->qhk", q_kpe, logical_kpe)
        ) * logit_scale
        if causal:
            q_len = q_end - q_start
            query_positions = torch.arange(q_len, device=logits.device) + kv_len - q_len
            key_positions = torch.arange(kv_len, device=logits.device)
            causal_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
            logits = logits.masked_fill(~causal_mask.unsqueeze(1), float("-inf"))
        logits_lse = torch.logsumexp(logits, dim=-1)
        if sinks is None:
            denominator_lse = logits_lse
        else:
            denominator_lse = torch.logaddexp(
                logits_lse, sinks.view(1, num_heads).expand(q_end - q_start, -1)
            )
        if lse is not None:
            batch_lse = denominator_lse
            if contract.lse_mode == "base2":
                batch_lse = batch_lse / torch.log(
                    torch.tensor(2.0, device=logits.device)
                )
            lse[q_start:q_end] = batch_lse
        probabilities = torch.exp(logits - denominator_lse.unsqueeze(-1))
        output[q_start:q_end] = (
            torch.einsum("qhk,kd->qhd", probabilities, logical_ckv) * bmm2
        )

    output_dtype = (
        q_nope.dtype if contract.output_dtype is None else contract.output_dtype
    )
    if contract.output_scale == "per-tensor":
        if o_scale is None:
            raise ValueError("per-tensor output scale requires o_scale")
        output = output / _scalar_scale("o_scale", o_scale)
    elif o_scale is not None:
        raise ValueError("o_scale requires per-tensor output scale mode")
    return output.to(output_dtype), lse
