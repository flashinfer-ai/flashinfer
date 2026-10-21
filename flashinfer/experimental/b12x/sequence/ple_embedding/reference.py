"""Exact PyTorch oracles for hashed PLE embedding lookup."""

from __future__ import annotations

import torch

from b12x.sequence.ple_hash.reference import ple_hash_packed_reference


_BF16_MODE = "bf16"
_FP8_MODE = "fp8_e4m3_per_tensor"
_NVFP4_MODE = "nvfp4_group16"
_E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def _selected_rows(
    tensor: torch.Tensor,
    live_ids: torch.Tensor,
    *,
    shard_start: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather local rows while mapping non-local IDs to a masked safe row."""
    if tensor.shape[0] == 0:
        raise ValueError("weight storage must contain at least one row")
    shard_end = shard_start + int(tensor.shape[0])
    local = (live_ids >= shard_start) & (live_ids < shard_end)
    safe_rows = (live_ids - shard_start).clamp(0, int(tensor.shape[0]) - 1)
    selected = torch.index_select(tensor, 0, safe_rows.reshape(-1))
    return selected, local


def _dequantize_selected_nvfp4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    scale_2: torch.Tensor,
    *,
    head_dim: int,
) -> torch.Tensor:
    """Dequantize already-selected ModelOpt NVFP4 rows."""
    low = packed & 0xF
    high = (packed >> 4) & 0xF
    codes = torch.stack((low, high), dim=-1).reshape(*packed.shape[:-1], head_dim)
    e2m1 = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=packed.device)
    values = e2m1[codes.long()].reshape(*packed.shape[:-1], head_dim // 16, 16)
    return (
        values
        * scales.float().unsqueeze(-1)
        * scale_2.float().reshape((1,) * (values.ndim - 1) + (1,))
    ).reshape(*packed.shape[:-1], head_dim)


def lookup(
    weight: torch.Tensor,
    weight_scale: torch.Tensor | None,
    ids: torch.Tensor,
    *,
    quant_mode: str = _FP8_MODE,
    weight_scale_2: torch.Tensor | None = None,
    shard_start: int,
    embedding_dim: int,
    num_tokens: int | None = None,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Gather selected local rows, then dequantize and flatten n-gram heads."""
    mode = str(quant_mode)
    if mode not in (_BF16_MODE, _FP8_MODE, _NVFP4_MODE):
        raise ValueError(f"unsupported quant_mode {mode!r}")
    if weight.ndim != 2:
        raise TypeError("weight must be a rank-2 tensor")
    if ids.ndim != 2 or ids.dtype != torch.int64:
        raise TypeError("ids must be a rank-2 torch.int64 tensor")
    if weight.device != ids.device:
        raise ValueError("weight and ids must use the same device")
    if output_dtype != torch.bfloat16:
        raise TypeError(f"output_dtype must be torch.bfloat16, got {output_dtype}")
    max_tokens, head_count = ids.shape
    embedding_dim = int(embedding_dim)
    if embedding_dim <= 0 or embedding_dim % head_count:
        raise ValueError(
            f"embedding_dim must be positive and divisible by head_count={head_count}, "
            f"got {embedding_dim}"
        )
    head_dim = embedding_dim // head_count

    if mode == _BF16_MODE:
        if weight.dtype != torch.bfloat16 or int(weight.shape[1]) != head_dim:
            raise TypeError(
                "bf16 weight must have dtype torch.bfloat16 and logical head width"
            )
        if weight_scale is not None or weight_scale_2 is not None:
            raise ValueError("bf16 lookup does not accept weight scales")
    elif mode == _FP8_MODE:
        if weight.dtype != torch.float8_e4m3fn or int(weight.shape[1]) != head_dim:
            raise TypeError(
                "FP8 weight must have dtype torch.float8_e4m3fn and logical head width"
            )
        if weight_scale is None:
            raise TypeError("FP8 weight_scale is required")
        if tuple(weight_scale.shape) != (1,) or weight_scale.dtype != torch.bfloat16:
            raise TypeError(
                "FP8 weight_scale must have shape (1,) and dtype torch.bfloat16"
            )
        if weight_scale_2 is not None:
            raise ValueError("FP8 lookup does not accept weight_scale_2")
    else:
        if head_dim % 16:
            raise ValueError(
                f"NVFP4 logical head width must be divisible by 16, got {head_dim}"
            )
        if weight.dtype != torch.uint8 or int(weight.shape[1]) != head_dim // 2:
            raise TypeError(
                "NVFP4 weight must have dtype torch.uint8 and packed logical head width"
            )
        if weight_scale is None:
            raise TypeError("NVFP4 weight_scale is required")
        expected_scale_shape = (int(weight.shape[0]), head_dim // 16)
        if (
            tuple(weight_scale.shape) != expected_scale_shape
            or weight_scale.dtype != torch.float8_e4m3fn
        ):
            raise TypeError(
                "NVFP4 weight_scale must have shape "
                f"{expected_scale_shape} and dtype torch.float8_e4m3fn"
            )
        if weight_scale_2 is None:
            raise TypeError("NVFP4 weight_scale_2 is required")
        if tuple(weight_scale_2.shape) != (1,) or weight_scale_2.dtype != torch.float32:
            raise TypeError(
                "NVFP4 weight_scale_2 must have shape (1,) and dtype torch.float32"
            )

    for name, tensor in (
        ("weight_scale", weight_scale),
        ("weight_scale_2", weight_scale_2),
    ):
        if tensor is not None and tensor.device != ids.device:
            raise ValueError(f"weight, {name}, and ids must use the same device")

    live_tokens = max_tokens if num_tokens is None else int(num_tokens)
    if live_tokens < 0 or live_tokens > max_tokens:
        raise ValueError(f"num_tokens must be in [0, {max_tokens}], got {live_tokens}")
    result = torch.zeros(
        (max_tokens, embedding_dim), dtype=output_dtype, device=ids.device
    )
    if live_tokens == 0:
        return result
    live_ids = ids[:live_tokens]
    shard_start = int(shard_start)
    selected, local = _selected_rows(weight, live_ids, shard_start=shard_start)
    if mode == _BF16_MODE:
        dequantized = selected.reshape(live_tokens, head_count, head_dim)
    elif mode == _FP8_MODE:
        assert weight_scale is not None
        dequantized = selected.reshape(live_tokens, head_count, head_dim).float()
        dequantized = dequantized * weight_scale.float().reshape(1, 1, 1)
    else:
        assert weight_scale is not None and weight_scale_2 is not None
        selected_scales, _ = _selected_rows(
            weight_scale, live_ids, shard_start=shard_start
        )
        dequantized = _dequantize_selected_nvfp4(
            selected.reshape(live_tokens, head_count, head_dim // 2),
            selected_scales.reshape(live_tokens, head_count, head_dim // 16),
            weight_scale_2,
            head_dim=head_dim,
        )
    dequantized = dequantized.to(output_dtype)
    dequantized.masked_fill_(~local[..., None], 0)
    result[:live_tokens].copy_(dequantized.flatten(-2))
    return result


def fused(
    weight: torch.Tensor,
    weight_scale: torch.Tensor | None,
    token_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    committed_history: torch.Tensor,
    *,
    quant_mode: str = _FP8_MODE,
    weight_scale_2: torch.Tensor | None = None,
    num_seqs: int,
    num_tokens: int,
    eos_token_id: int,
    multipliers: torch.Tensor,
    prime_sizes: torch.Tensor,
    table_offsets: torch.Tensor,
    heads_per_order: int,
    shard_start: int,
    embedding_dim: int,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Reference the public fixed-capacity hash, gather, and dequant contract."""
    max_tokens = int(token_ids.numel())
    live_tokens = int(num_tokens)
    live_seqs = int(num_seqs)
    if live_tokens < 0 or live_tokens > max_tokens:
        raise ValueError(f"num_tokens must be in [0, {max_tokens}], got {live_tokens}")
    if live_seqs < 0 or live_seqs > int(committed_history.shape[0]):
        raise ValueError(
            f"num_seqs must be in [0, {committed_history.shape[0]}], got {live_seqs}"
        )
    if live_seqs == 0:
        if live_tokens != 0:
            raise ValueError("num_tokens must be zero when num_seqs is zero")
        ids = torch.empty(
            (max_tokens, int(prime_sizes.numel())),
            dtype=torch.int64,
            device=token_ids.device,
        )
        return lookup(
            weight,
            weight_scale,
            ids,
            quant_mode=quant_mode,
            weight_scale_2=weight_scale_2,
            shard_start=shard_start,
            embedding_dim=embedding_dim,
            num_tokens=0,
            output_dtype=output_dtype,
        )
    starts = query_start_loc[: live_seqs + 1]
    live_ids = ple_hash_packed_reference(
        token_ids[:live_tokens],
        starts,
        committed_history[:live_seqs],
        eos_token_id=int(eos_token_id),
        multipliers=multipliers,
        prime_sizes=prime_sizes,
        table_offsets=table_offsets,
        heads_per_order=int(heads_per_order),
    )
    ids = torch.full(
        (max_tokens, int(prime_sizes.numel())),
        -1,
        dtype=torch.int64,
        device=token_ids.device,
    )
    ids[:live_tokens].copy_(live_ids)
    return lookup(
        weight,
        weight_scale,
        ids,
        quant_mode=quant_mode,
        weight_scale_2=weight_scale_2,
        shard_start=shard_start,
        embedding_dim=embedding_dim,
        num_tokens=live_tokens,
        output_dtype=output_dtype,
    )


__all__ = ["lookup", "fused"]
