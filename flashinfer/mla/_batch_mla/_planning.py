"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional

import torch

from ._contracts import MLAPlanMetadata, MLAStructuralInputKind


@dataclass(frozen=True)
class _CSRPlanMetadata:
    qo_indptr: torch.Tensor
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    kv_len_arr: torch.Tensor


@dataclass(frozen=True)
class _DensePlanMetadata:
    cum_seq_lens_q: torch.Tensor
    block_tables: torch.Tensor
    seq_lens: torch.Tensor
    max_q_len: int


def _validate_metadata_tensor(
    name: str,
    tensor: object,
    *,
    rank: int,
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{name} metadata must be a torch.Tensor.")
    if tensor.ndim != rank:
        raise ValueError(
            f"{name} metadata must be rank-{rank}, got shape {tuple(tensor.shape)}."
        )
    if tensor.dtype != torch.int32:
        raise ValueError(
            f"{name} metadata must have dtype torch.int32, got {tensor.dtype}."
        )
    if tensor.device.type != "cpu" and tensor.device != device:
        raise ValueError(
            f"{name} metadata must be on CPU or wrapper device {device}, "
            f"got {tensor.device}."
        )
    if not tensor.is_contiguous():
        raise ValueError(f"{name} metadata must be contiguous.")
    return tensor


def _indptr_values(name: str, indptr: torch.Tensor) -> torch.Tensor:
    if indptr.numel() == 0:
        raise ValueError(f"{name} metadata must contain at least one element.")
    values = indptr.to(device="cpu", dtype=torch.int64)
    if int(values[0].item()) != 0:
        raise ValueError(f"{name} metadata must start at zero.")
    if bool(torch.any(values[1:] < values[:-1]).item()):
        raise ValueError(f"{name} metadata must be nondecreasing.")
    return values


def _max_q_len(cum_seq_lens_q: torch.Tensor) -> int:
    values = _indptr_values("cum_seq_lens_q", cum_seq_lens_q)
    if values.numel() <= 1:
        return 0
    return int((values[1:] - values[:-1]).max().item())


def _validate_csr_metadata(
    *,
    qo_indptr: object,
    kv_indptr: object,
    kv_indices: object,
    kv_len_arr: object,
    page_size: int,
    device: torch.device,
    strict: bool = True,
) -> _CSRPlanMetadata:
    if not isinstance(page_size, int) or isinstance(page_size, bool) or page_size <= 0:
        raise ValueError(f"page_size must be a positive int, got {page_size!r}.")
    validated_qo_indptr = _validate_metadata_tensor(
        "qo_indptr", qo_indptr, rank=1, device=device
    )
    validated_kv_indptr = _validate_metadata_tensor(
        "kv_indptr", kv_indptr, rank=1, device=device
    )
    validated_kv_indices = _validate_metadata_tensor(
        "kv_indices", kv_indices, rank=1, device=device
    )
    validated_kv_len_arr = _validate_metadata_tensor(
        "kv_len_arr", kv_len_arr, rank=1, device=device
    )
    qo_values = _indptr_values("qo_indptr", validated_qo_indptr)
    kv_values = _indptr_values("kv_indptr", validated_kv_indptr)
    batch_size = qo_values.numel() - 1
    has_canonical_batch_shape = (
        kv_values.numel() == batch_size + 1
        and validated_kv_len_arr.numel() == batch_size
    )
    if strict and not has_canonical_batch_shape:
        raise ValueError(
            "CSR metadata batch dimensions must agree: "
            f"qo_indptr describes {batch_size}, kv_indptr describes "
            f"{kv_values.numel() - 1}, and kv_len_arr has "
            f"{validated_kv_len_arr.numel()} entries."
        )
    kv_end = int(kv_values[-1].item())
    if (strict or has_canonical_batch_shape) and kv_end > validated_kv_indices.numel():
        raise ValueError(
            f"kv_indices metadata has {validated_kv_indices.numel()} entries but "
            f"kv_indptr[-1] is {kv_end}."
        )
    kv_lens_values = validated_kv_len_arr.to(device="cpu", dtype=torch.int64)
    if bool(torch.any(kv_lens_values < 0).item()):
        raise ValueError("kv_len_arr metadata must be nonnegative.")
    if strict or has_canonical_batch_shape:
        expected_pages = torch.div(
            kv_lens_values + page_size - 1,
            page_size,
            rounding_mode="floor",
        )
        actual_pages = kv_values[1:] - kv_values[:-1]
        if not torch.equal(expected_pages, actual_pages):
            raise ValueError(
                "CSR metadata page counts in kv_indptr must equal ceil(kv_len_arr / "
                "page_size)."
            )
    return _CSRPlanMetadata(
        qo_indptr=validated_qo_indptr,
        kv_indptr=validated_kv_indptr,
        kv_indices=validated_kv_indices,
        kv_len_arr=validated_kv_len_arr,
    )


def _validate_dense_metadata(
    *,
    cum_seq_lens_q: object,
    block_tables: object,
    seq_lens: object,
    max_q_len: object,
    page_size: int,
    device: torch.device,
    table_width_alignment: Optional[int] = None,
) -> _DensePlanMetadata:
    if not isinstance(page_size, int) or isinstance(page_size, bool) or page_size <= 0:
        raise ValueError(f"page_size must be a positive int, got {page_size!r}.")
    validated_cum_seq_lens_q = _validate_metadata_tensor(
        "cum_seq_lens_q", cum_seq_lens_q, rank=1, device=device
    )
    validated_block_tables = _validate_metadata_tensor(
        "block_tables", block_tables, rank=2, device=device
    )
    validated_seq_lens = _validate_metadata_tensor(
        "seq_lens", seq_lens, rank=1, device=device
    )
    _indptr_values("cum_seq_lens_q", validated_cum_seq_lens_q)
    batch_size = validated_cum_seq_lens_q.numel() - 1
    if (
        validated_block_tables.shape[0] != batch_size
        or validated_seq_lens.numel() != batch_size
    ):
        raise ValueError(
            "dense metadata batch dimensions must agree: "
            f"cum_seq_lens_q describes {batch_size}, block_tables has "
            f"{validated_block_tables.shape[0]}, and seq_lens has "
            f"{validated_seq_lens.numel()}."
        )
    seq_lens_values = validated_seq_lens.to(device="cpu", dtype=torch.int64)
    if bool(torch.any(seq_lens_values < 0).item()):
        raise ValueError("seq_lens metadata must be nonnegative.")
    live_pages = torch.div(
        seq_lens_values + page_size - 1,
        page_size,
        rounding_mode="floor",
    )
    if (
        live_pages.numel()
        and int(live_pages.max().item()) > validated_block_tables.shape[1]
    ):
        raise ValueError(
            "block_tables metadata width is smaller than the live page count "
            "implied by seq_lens and page_size."
        )
    if table_width_alignment is not None:
        _check_table_width_alignment(table_width_alignment)
        if (
            validated_block_tables.shape[1] == 0
            or validated_block_tables.shape[1] % table_width_alignment != 0
        ):
            raise ValueError(
                "block_tables metadata width must be a positive multiple of "
                f"{table_width_alignment}, got {validated_block_tables.shape[1]}."
            )

    actual_max_q_len = _max_q_len(validated_cum_seq_lens_q)
    if max_q_len is None:
        resolved_max_q_len = actual_max_q_len
    elif not isinstance(max_q_len, int) or isinstance(max_q_len, bool):
        raise ValueError("max_q_len metadata must be an int or None.")
    elif max_q_len < actual_max_q_len:
        raise ValueError(
            f"max_q_len metadata must be at least {actual_max_q_len}, got {max_q_len}."
        )
    else:
        resolved_max_q_len = max_q_len

    return _DensePlanMetadata(
        cum_seq_lens_q=validated_cum_seq_lens_q,
        block_tables=validated_block_tables,
        seq_lens=validated_seq_lens,
        max_q_len=resolved_max_q_len,
    )


def _derive_csr_from_dense(
    dense: _DensePlanMetadata,
    *,
    page_size: int,
) -> _CSRPlanMetadata:
    live_pages = torch.div(
        dense.seq_lens + page_size - 1,
        page_size,
        rounding_mode="floor",
    ).to(dtype=torch.int32)
    kv_indptr = torch.cat(
        (
            torch.zeros((1,), dtype=torch.int32, device=dense.seq_lens.device),
            torch.cumsum(live_pages, dim=0, dtype=torch.int32),
        )
    )
    live_page_counts = live_pages.to(device="cpu").tolist()
    rows = [
        dense.block_tables[row, : int(live_page_counts[row])]
        for row in range(dense.block_tables.shape[0])
    ]
    kv_indices = (
        torch.cat(rows)
        if rows
        else dense.block_tables.new_empty((0,), dtype=torch.int32)
    )
    return _CSRPlanMetadata(
        qo_indptr=dense.cum_seq_lens_q,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        kv_len_arr=dense.seq_lens,
    )


def _derive_dense_from_csr(
    csr: _CSRPlanMetadata,
    *,
    table_width_alignment: Optional[int],
) -> _DensePlanMetadata:
    kv_indptr_host = csr.kv_indptr.to(device="cpu", dtype=torch.int64)
    page_counts = kv_indptr_host[1:] - kv_indptr_host[:-1]
    max_pages = int(page_counts.max().item()) if page_counts.numel() else 0
    if table_width_alignment is None:
        table_width = max(1, max_pages)
    else:
        _check_table_width_alignment(table_width_alignment)
        table_width = max(
            table_width_alignment,
            ((max_pages + table_width_alignment - 1) // table_width_alignment)
            * table_width_alignment,
        )
    block_tables = torch.zeros(
        (page_counts.numel(), table_width),
        dtype=torch.int32,
        device=csr.kv_indices.device,
    )
    for row in range(page_counts.numel()):
        start = int(kv_indptr_host[row].item())
        end = int(kv_indptr_host[row + 1].item())
        if end > start:
            block_tables[row, : end - start].copy_(csr.kv_indices[start:end])
    return _DensePlanMetadata(
        cum_seq_lens_q=csr.qo_indptr,
        block_tables=block_tables,
        seq_lens=csr.kv_len_arr,
        max_q_len=_max_q_len(csr.qo_indptr),
    )


def _check_table_width_alignment(table_width_alignment: int) -> None:
    if (
        not isinstance(table_width_alignment, int)
        or isinstance(table_width_alignment, bool)
        or table_width_alignment <= 0
    ):
        raise ValueError("dense table-width alignment must be positive.")


class _MLAPlanMetadataResolver:
    """Lazily validate and normalize metadata for one wrapper plan request."""

    def __init__(
        self,
        *,
        metadata: MLAPlanMetadata,
        page_size: int,
        device: torch.device,
        strict_csr: bool = True,
    ) -> None:
        self._raw_csr = (
            metadata.qo_indptr,
            metadata.kv_indptr,
            metadata.kv_indices,
            metadata.kv_len_arr,
        )
        self._raw_dense = (
            metadata.cum_seq_lens_q,
            metadata.block_tables,
            metadata.seq_lens,
            metadata.max_q_len,
        )
        self.page_size = page_size
        self.device = device
        self.strict_csr = strict_csr
        self._forms_checked = False
        self._has_csr = False
        self._has_dense = False
        self._validated_csr: Optional[_CSRPlanMetadata] = None
        self._validated_dense: Optional[_DensePlanMetadata] = None
        self._derived_csr: Optional[_CSRPlanMetadata] = None
        self._derived_dense_by_alignment: dict[int, _DensePlanMetadata] = {}
        self._derived_native_dense: Optional[_DensePlanMetadata] = None
        self._device_dense_by_alignment: dict[int, _DensePlanMetadata] = {}
        self._device_native_dense: Optional[_DensePlanMetadata] = None
        self._equivalence_checked = False

    def _check_forms(self) -> None:
        if self._forms_checked:
            return
        if (
            not isinstance(self.page_size, int)
            or isinstance(self.page_size, bool)
            or self.page_size <= 0
        ):
            raise ValueError(
                f"page_size must be a positive int, got {self.page_size!r}."
            )
        csr_present = tuple(value is not None for value in self._raw_csr)
        dense_present = tuple(value is not None for value in self._raw_dense[:3])
        max_q_len_present = self._raw_dense[3] is not None
        if any(csr_present) and not all(csr_present):
            names: tuple[str, ...] = (
                "qo_indptr",
                "kv_indptr",
                "kv_indices",
                "kv_len_arr",
            )
            missing = [
                name
                for name, present in zip(names, csr_present, strict=True)
                if not present
            ]
            raise ValueError(
                "CSR metadata form is partial; missing required fields: "
                + ", ".join(missing)
                + "."
            )
        if any(dense_present) and not all(dense_present):
            names = ("cum_seq_lens_q", "block_tables", "seq_lens")
            missing = [
                name
                for name, present in zip(names, dense_present, strict=True)
                if not present
            ]
            raise ValueError(
                "dense metadata form is partial; missing required fields: "
                + ", ".join(missing)
                + "."
            )
        if max_q_len_present and not all(dense_present):
            raise ValueError(
                "max_q_len metadata requires the complete dense metadata form, "
                "including cum_seq_lens_q, block_tables, and seq_lens."
            )
        self._has_csr = all(csr_present)
        self._has_dense = all(dense_present)
        if not self._has_csr and not self._has_dense:
            raise ValueError("A complete CSR or dense metadata form is required.")
        self._forms_checked = True

    def _validate_csr(self) -> _CSRPlanMetadata:
        if self._validated_csr is None:
            qo_indptr, kv_indptr, kv_indices, kv_len_arr = self._raw_csr
            self._validated_csr = _validate_csr_metadata(
                qo_indptr=qo_indptr,
                kv_indptr=kv_indptr,
                kv_indices=kv_indices,
                kv_len_arr=kv_len_arr,
                page_size=self.page_size,
                device=self.device,
                strict=self.strict_csr,
            )
        return self._validated_csr

    def _validate_dense(
        self, table_width_alignment: Optional[int]
    ) -> _DensePlanMetadata:
        if self._validated_dense is None:
            cum_seq_lens_q, block_tables, seq_lens, max_q_len = self._raw_dense
            self._validated_dense = _validate_dense_metadata(
                cum_seq_lens_q=cum_seq_lens_q,
                block_tables=block_tables,
                seq_lens=seq_lens,
                max_q_len=max_q_len,
                page_size=self.page_size,
                device=self.device,
                table_width_alignment=table_width_alignment,
            )
        elif table_width_alignment is not None:
            _check_table_width_alignment(table_width_alignment)
            if (
                self._validated_dense.block_tables.shape[1] == 0
                or self._validated_dense.block_tables.shape[1] % table_width_alignment
                != 0
            ):
                raise ValueError(
                    "block_tables metadata width must be a positive multiple of "
                    f"{table_width_alignment}, got "
                    f"{self._validated_dense.block_tables.shape[1]}."
                )
        return self._validated_dense

    @staticmethod
    def _equal_tensor_values(left: torch.Tensor, right: torch.Tensor) -> bool:
        return torch.equal(left.to(device="cpu"), right.to(device="cpu"))

    def _ensure_equivalent(
        self,
        csr: _CSRPlanMetadata,
        dense: _DensePlanMetadata,
    ) -> None:
        if self._equivalence_checked:
            return
        dense_as_csr = self._derive_csr(dense)
        csr_live_end = int(csr.kv_indptr.to(device="cpu")[-1].item())
        equivalent = (
            self._equal_tensor_values(csr.qo_indptr, dense_as_csr.qo_indptr)
            and self._equal_tensor_values(csr.kv_indptr, dense_as_csr.kv_indptr)
            and self._equal_tensor_values(
                csr.kv_indices[:csr_live_end], dense_as_csr.kv_indices
            )
            and self._equal_tensor_values(csr.kv_len_arr, dense_as_csr.kv_len_arr)
        )
        if not equivalent:
            raise ValueError(
                "CSR and dense metadata forms must be logically equivalent."
            )
        self._equivalence_checked = True

    def _ensure_dual_forms_equivalent(
        self,
        *,
        csr: Optional[_CSRPlanMetadata] = None,
        dense: Optional[_DensePlanMetadata] = None,
    ) -> None:
        if not self._has_csr or not self._has_dense:
            return
        self._ensure_equivalent(
            csr or self._validate_csr(), dense or self._validate_dense(None)
        )

    def _derive_csr(self, dense: _DensePlanMetadata) -> _CSRPlanMetadata:
        if self._derived_csr is None:
            self._derived_csr = _derive_csr_from_dense(dense, page_size=self.page_size)
        return self._derived_csr

    def resolve_csr(self) -> _CSRPlanMetadata:
        self._check_forms()
        if self._has_csr:
            csr = self._validate_csr()
            self._ensure_dual_forms_equivalent(csr=csr)
            return csr
        return self._derive_csr(self._validate_dense(None))

    def resolve_dense(self, *, table_width_alignment: int) -> _DensePlanMetadata:
        self._check_forms()
        _check_table_width_alignment(table_width_alignment)
        if self._has_dense:
            dense = self._validate_dense(table_width_alignment)
            self._ensure_dual_forms_equivalent(dense=dense)
            return dense
        if table_width_alignment not in self._derived_dense_by_alignment:
            self._derived_dense_by_alignment[table_width_alignment] = (
                _derive_dense_from_csr(
                    self._validate_csr(),
                    table_width_alignment=table_width_alignment,
                )
            )
        return self._derived_dense_by_alignment[table_width_alignment]

    def _dense_on_device(self, dense: _DensePlanMetadata) -> _DensePlanMetadata:
        if all(
            tensor.device == self.device
            for tensor in (dense.cum_seq_lens_q, dense.block_tables, dense.seq_lens)
        ):
            return dense
        return _DensePlanMetadata(
            cum_seq_lens_q=dense.cum_seq_lens_q.to(
                device=self.device, non_blocking=True
            ),
            block_tables=dense.block_tables.to(device=self.device, non_blocking=True),
            seq_lens=dense.seq_lens.to(device=self.device, non_blocking=True),
            max_q_len=dense.max_q_len,
        )

    def resolve_device_dense(
        self,
        *,
        table_width_alignment: int,
    ) -> _DensePlanMetadata:
        if table_width_alignment not in self._device_dense_by_alignment:
            self._device_dense_by_alignment[table_width_alignment] = (
                self._dense_on_device(
                    self.resolve_dense(table_width_alignment=table_width_alignment)
                )
            )
        return self._device_dense_by_alignment[table_width_alignment]

    def resolve_native_dense(self) -> _DensePlanMetadata:
        self._check_forms()
        if self._has_dense:
            dense = self._validate_dense(None)
            self._ensure_dual_forms_equivalent(dense=dense)
            return dense
        if self._derived_native_dense is None:
            self._derived_native_dense = _derive_dense_from_csr(
                self._validate_csr(),
                table_width_alignment=None,
            )
        return self._derived_native_dense

    def resolve_native_device_dense(self) -> _DensePlanMetadata:
        if self._device_native_dense is None:
            self._device_native_dense = self._dense_on_device(
                self.resolve_native_dense()
            )
        return self._device_native_dense


@dataclass(frozen=True, kw_only=True, eq=False)
class _MLAPlanArguments:
    metadata: MLAPlanMetadata
    num_heads: int
    head_dim_ckv: int
    head_dim_kpe: int
    page_size: int
    causal: bool
    sm_scale: float
    q_data_type: torch.dtype
    kv_data_type: torch.dtype
    query_kind: Optional[MLAStructuralInputKind] = None
    kv_kind: Optional[MLAStructuralInputKind] = None
    query_layout: Literal["packed", "split"] = "packed"
    kv_cache_layout: Literal["packed", "split"] = "packed"
    lse_mode: Literal["none", "base2", "basee"] = "none"
    kv_layout: Literal["combined", "adjacent-split", "independent-split"] = (
        "independent-split"
    )
    output_dtype: torch.dtype = torch.float16
    output_scale: Literal["none", "per-tensor"] = "none"
    scale_mode: Literal["default", "kv-per-tensor"] = "default"
    skip_softmax: bool = False
    use_profiler: bool = False
    legacy_flat_csr: bool = False
    _float_workspace_buffer: torch.Tensor = field(repr=False, compare=False)
    _use_cuda_graph: bool = field(repr=False, compare=False)
    _qo_indptr_buf: Optional[torch.Tensor] = field(repr=False, compare=False)
    _kv_indptr_buf: Optional[torch.Tensor] = field(repr=False, compare=False)
    _kv_indices_buf: Optional[torch.Tensor] = field(repr=False, compare=False)
    _kv_len_arr_buf: Optional[torch.Tensor] = field(repr=False, compare=False)
    _graph_plan_int_workspace_buffer: Optional[torch.Tensor] = field(
        default=None, repr=False, compare=False
    )
    _metadata_resolver: _MLAPlanMetadataResolver = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, MLAPlanMetadata):
            raise TypeError("metadata must be an MLAPlanMetadata instance.")
        if self.head_dim_ckv <= 0:
            raise ValueError(f"head_dim_ckv must be > 0, got {self.head_dim_ckv}.")
        if self.head_dim_kpe < 0:
            raise ValueError(f"head_dim_kpe must be >= 0, got {self.head_dim_kpe}.")
        if self.query_layout not in ("packed", "split"):
            raise ValueError(f"unsupported query_layout {self.query_layout!r}.")
        if self.kv_cache_layout not in ("packed", "split"):
            raise ValueError(f"unsupported kv_cache_layout {self.kv_cache_layout!r}.")
        if self.lse_mode not in ("none", "base2", "basee"):
            raise ValueError(f"unsupported lse_mode {self.lse_mode!r}.")
        if self.output_scale not in ("none", "per-tensor"):
            raise ValueError(f"unsupported output_scale {self.output_scale!r}.")
        if self.scale_mode not in ("default", "kv-per-tensor"):
            raise ValueError(f"unsupported scale_mode {self.scale_mode!r}.")
        if not isinstance(self.output_dtype, torch.dtype):
            raise TypeError("output_dtype must be a torch.dtype.")
        if not isinstance(self.skip_softmax, bool):
            raise TypeError("skip_softmax must be a bool.")
        object.__setattr__(
            self,
            "_metadata_resolver",
            _MLAPlanMetadataResolver(
                metadata=self.metadata,
                page_size=self.page_size,
                device=self._float_workspace_buffer.device,
                strict_csr=not self.legacy_flat_csr,
            ),
        )

    def csr(self) -> _CSRPlanMetadata:
        return self._metadata_resolver.resolve_csr()

    def dense(self, *, table_width_alignment: int) -> _DensePlanMetadata:
        return self._metadata_resolver.resolve_dense(
            table_width_alignment=table_width_alignment
        )

    def native_dense(self) -> _DensePlanMetadata:
        return self._metadata_resolver.resolve_native_dense()
