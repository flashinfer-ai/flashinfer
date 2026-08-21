"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
"""

from dataclasses import dataclass
from typing import Literal, Optional, Union, overload

import torch


MLAInputAvailability = Literal["packed", "split", "redundant"]
MLAChosenRepresentation = Literal["packed", "split"]
MLAStructuralInputKind = Literal[
    "packed", "adjacent-split", "independent-split", "dual"
]


def _raise_planned_run_mismatch(name: str, planned: object, actual: object) -> None:
    raise ValueError(
        f"MLA planned run argument {name} mismatch: planned {planned!r}, got "
        f"{actual!r}; re-plan with the needed arguments."
    )


@dataclass(frozen=True)
class MLAPlanMetadata:
    """Canonical CSR and/or dense metadata for a Batch MLA plan."""

    qo_indptr: Optional[torch.Tensor] = None
    kv_indptr: Optional[torch.Tensor] = None
    kv_indices: Optional[torch.Tensor] = None
    kv_len_arr: Optional[torch.Tensor] = None
    cum_seq_lens_q: Optional[torch.Tensor] = None
    block_tables: Optional[torch.Tensor] = None
    seq_lens: Optional[torch.Tensor] = None
    max_q_len: Optional[int] = None

    @classmethod
    def csr(
        cls,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        kv_len_arr: torch.Tensor,
    ) -> "MLAPlanMetadata":
        return cls(
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            kv_len_arr=kv_len_arr,
        )

    @classmethod
    def dense(
        cls,
        cum_seq_lens_q: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_q_len: Optional[int] = None,
    ) -> "MLAPlanMetadata":
        return cls(
            cum_seq_lens_q=cum_seq_lens_q,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_q_len=max_q_len,
        )

    @classmethod
    def dual(
        cls,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        kv_len_arr: torch.Tensor,
        *,
        cum_seq_lens_q: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_q_len: Optional[int] = None,
    ) -> "MLAPlanMetadata":
        return cls(
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            kv_len_arr=kv_len_arr,
            cum_seq_lens_q=cum_seq_lens_q,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_q_len=max_q_len,
        )


@dataclass(frozen=True)
class MLAInputContract:
    """Run options that must remain compatible with a planned wrapper."""

    lse_mode: str
    output_dtype: torch.dtype
    output_scale: str
    scale_mode: str
    query_layout: Literal["packed", "split"] = "packed"
    kv_cache_layout: Literal["packed", "split"] = "packed"
    head_dim_ckv: Optional[int] = None
    head_dim_kpe: Optional[int] = None

    def validate_run_options(
        self,
        *,
        out: Optional[torch.Tensor],
        lse: Optional[torch.Tensor],
        return_lse: bool,
        return_lse_base_on_e: bool,
        o_scale: Optional[float],
        ckv_scale: Optional[float],
        ckv_scale_arr: Optional[torch.Tensor],
        kpe_scale: Optional[float],
    ) -> None:
        """Validate run-time options against this planned contract."""

        actual_lse_mode = "none"
        if return_lse or lse is not None or return_lse_base_on_e:
            actual_lse_mode = "basee" if return_lse_base_on_e else "base2"
        if actual_lse_mode != self.lse_mode:
            _raise_planned_run_mismatch("LSE mode", self.lse_mode, actual_lse_mode)

        actual_output_dtype = self.output_dtype if out is None else out.dtype
        if actual_output_dtype != self.output_dtype:
            _raise_planned_run_mismatch(
                "output dtype", self.output_dtype, actual_output_dtype
            )
        actual_output_scale = "per-tensor" if o_scale is not None else "none"
        if actual_output_scale != self.output_scale:
            _raise_planned_run_mismatch(
                "o_scale", self.output_scale, actual_output_scale
            )

        has_ckv_scale = ckv_scale is not None or ckv_scale_arr is not None
        has_kpe_scale = kpe_scale is not None
        if has_ckv_scale and has_kpe_scale:
            actual_scale_mode = "kv-per-tensor"
        elif has_ckv_scale or has_kpe_scale:
            actual_scale_mode = "incomplete-kv-per-tensor"
        else:
            actual_scale_mode = "default"
        if actual_scale_mode != self.scale_mode:
            _raise_planned_run_mismatch(
                "scale mode", self.scale_mode, actual_scale_mode
            )


def _as_tensor_leaf(value: object, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} structural MLA input leaves must be torch.Tensor.")
    return value


def _as_split_structural_pair(
    value: object, *, name: str
) -> tuple[torch.Tensor, torch.Tensor]:
    if type(value) is not tuple:
        raise TypeError(
            f"{name} structural MLA input must be a tensor or exact 2-tuple."
        )
    if len(value) != 2:
        raise ValueError(f"{name} structural MLA input tuples must have length 2.")
    left, right = value
    if type(left) is tuple or type(right) is tuple:
        raise ValueError(f"{name} structural MLA input has malformed nesting.")
    return _as_tensor_leaf(left, name=name), _as_tensor_leaf(right, name=name)


def _parse_structural_mla_input(
    value: object, *, name: str
) -> tuple[Optional[torch.Tensor], Optional[tuple[torch.Tensor, torch.Tensor]]]:
    if isinstance(value, torch.Tensor):
        return value, None
    if type(value) is not tuple:
        raise TypeError(
            f"{name} structural MLA input must be a tensor or exact 2-tuple."
        )
    if len(value) != 2:
        raise ValueError(f"{name} structural MLA input tuples must have length 2.")
    left, right = value
    left_is_tuple = type(left) is tuple
    right_is_tuple = type(right) is tuple
    if left_is_tuple and right_is_tuple:
        raise ValueError(f"{name} structural MLA input has malformed nesting.")
    if left_is_tuple:
        return _as_tensor_leaf(right, name=name), _as_split_structural_pair(
            left, name=name
        )
    if right_is_tuple:
        return _as_tensor_leaf(left, name=name), _as_split_structural_pair(
            right, name=name
        )
    return None, (_as_tensor_leaf(left, name=name), _as_tensor_leaf(right, name=name))


def _adjacent_last_dim_view(
    left: torch.Tensor, right: torch.Tensor
) -> Optional[torch.Tensor]:
    """Return a single in-bounds contiguous view for adjacent split tensors."""
    if not _are_adjacent_last_dim_views(left, right):
        return None
    shape = left.shape[:-1] + (left.shape[-1] + right.shape[-1],)
    return left.as_strided(shape, left.stride(), left.storage_offset())


def _are_adjacent_last_dim_views(left: torch.Tensor, right: torch.Tensor) -> bool:
    if (
        left.ndim == 0
        or right.ndim != left.ndim
        or left.shape[:-1] != right.shape[:-1]
        or left.dtype != right.dtype
        or left.device != right.device
        or left.stride() != right.stride()
        or left.shape[-1] == 0
        or right.shape[-1] == 0
    ):
        return False
    storage = left.untyped_storage()
    if storage._cdata != right.untyped_storage()._cdata:
        return False
    stride = left.stride()
    storage_offset = left.storage_offset()
    if right.storage_offset() != storage_offset + left.shape[-1] * stride[-1]:
        return False
    shape = left.shape[:-1] + (left.shape[-1] + right.shape[-1],)
    storage_numel, remainder = divmod(storage.nbytes(), left.element_size())
    if remainder:
        return False
    last_offset = storage_offset + sum(
        (size - 1) * step for size, step in zip(shape, stride, strict=True)
    )
    if last_offset >= storage_numel:
        return False
    expected_stride = 1
    for size, step in zip(reversed(shape), reversed(stride), strict=True):
        if size != 1:
            if step != expected_stride:
                return False
            expected_stride *= size
    return True


def _validate_packed_structural_input(
    packed: torch.Tensor, *, widths: tuple[int, int], name: str
) -> tuple[torch.dtype, tuple[int, ...]]:
    if packed.ndim != 3:
        raise ValueError(f"packed {name} must have rank 3.")
    if packed.shape[-1] != sum(widths):
        raise ValueError(
            f"packed {name} last dimension does not match planned split widths."
        )
    return packed.dtype, tuple(packed.shape)


def _validate_split_structural_input(
    split: tuple[torch.Tensor, torch.Tensor],
    *,
    widths: tuple[int, int],
    name: str,
) -> tuple[torch.dtype, tuple[int, ...]]:
    left, right = split
    if left.ndim != 3 or right.ndim != 3:
        raise ValueError(f"split {name} tensors must have rank 3.")
    if left.shape[:-1] != right.shape[:-1]:
        raise ValueError(f"split {name} tensor shapes must match before the last axis.")
    if left.dtype != right.dtype:
        raise ValueError(f"split {name} tensor dtypes must match.")
    if left.device != right.device:
        raise ValueError(f"split {name} tensor devices must match.")
    if (left.shape[-1], right.shape[-1]) != widths:
        raise ValueError(f"split {name} last dimensions do not match planned widths.")
    return left.dtype, tuple(left.shape[:-1]) + (sum(widths),)


def _structural_mla_input_facts(
    value: object,
    *,
    widths: tuple[int, int],
    name: str,
) -> tuple[MLAStructuralInputKind, torch.dtype, tuple[int, ...]]:
    """Validate the primary packed member of a trusted redundant value."""
    packed, split = _parse_structural_mla_input(value, name=name)
    if packed is not None:
        dtype, shape = _validate_packed_structural_input(
            packed, widths=widths, name=name
        )
        return ("dual" if split is not None else "packed"), dtype, shape
    assert split is not None
    dtype, shape = _validate_split_structural_input(split, widths=widths, name=name)
    left, right = split
    kind: MLAStructuralInputKind = (
        "adjacent-split"
        if _are_adjacent_last_dim_views(left, right)
        else "independent-split"
    )
    return kind, dtype, shape


def _split_packed_last_dim(
    packed: torch.Tensor,
    widths: tuple[int, int],
    *,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    left_width, right_width = widths
    if left_width <= 0 or right_width < 0:
        raise ValueError("left split width must be positive and right non-negative.")
    _validate_packed_structural_input(packed, widths=widths, name=name)
    return packed[..., :left_width], packed[..., left_width:]


def _structural_mla_input_kind(value: object) -> MLAStructuralInputKind:
    if isinstance(value, torch.Tensor):
        return "packed"
    packed, split = _parse_structural_mla_input(value, name="MLA input")
    if packed is not None:
        return "dual"
    assert split is not None
    return (
        "adjacent-split"
        if _adjacent_last_dim_view(*split) is not None
        else "independent-split"
    )


@overload
def _resolve_structural_mla_input(
    value: object,
    *,
    desired: Literal["packed"],
    widths: Optional[tuple[int, int]],
    name: str,
) -> torch.Tensor: ...


@overload
def _resolve_structural_mla_input(
    value: object,
    *,
    desired: Literal["split"],
    widths: Optional[tuple[int, int]],
    name: str,
) -> tuple[torch.Tensor, torch.Tensor]: ...


def _resolve_structural_mla_input(
    value: object,
    *,
    desired: MLAChosenRepresentation,
    widths: Optional[tuple[int, int]],
    name: str,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    if desired not in ("packed", "split"):
        raise ValueError(f"unsupported {name} representation {desired!r}.")
    packed, split = _parse_structural_mla_input(value, name=name)
    if widths is None:
        if desired == "packed" and packed is not None:
            return packed
        if desired == "split" and split is not None:
            return split
        raise ValueError(f"planned split widths are required for {name} conversion.")
    if desired == "split":
        if split is not None:
            _validate_split_structural_input(split, widths=widths, name=name)
            return split
        assert packed is not None
        return _split_packed_last_dim(packed, widths, name=name)
    if packed is not None:
        _validate_packed_structural_input(packed, widths=widths, name=name)
        return packed
    assert split is not None
    _validate_split_structural_input(split, widths=widths, name=name)
    left, right = split
    if widths[1] == 0:
        return left
    adjacent = _adjacent_last_dim_view(left, right)
    if adjacent is not None:
        return adjacent
    raise ValueError(
        f"{name} cannot provide the planned packed representation zero-copy; "
        "re-plan for split input."
    )
