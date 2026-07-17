"""Private orchestration-owned planning for batch MLA wrapper backends.

This module owns metadata normalization and typed plan-request transport.
Backend selection and module loading deliberately remain in the wrapper.
"""

from dataclasses import InitVar, dataclass, field
from typing import Optional

import torch


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
    if tensor.device != device:
        raise ValueError(
            f"{name} metadata must be on wrapper device {device}, got {tensor.device}."
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
    if torch.any(values[1:] < values[:-1]).item():
        raise ValueError(f"{name} metadata must be nondecreasing.")
    return values


def _max_q_len(cum_seq_lens_q: torch.Tensor) -> int:
    values = _indptr_values("cum_seq_lens_q", cum_seq_lens_q)
    if values.numel() <= 1:
        return 0
    return int((values[1:] - values[:-1]).max().item())


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
    cum_seq_lens_q = _validate_metadata_tensor(
        "cum_seq_lens_q", cum_seq_lens_q, rank=1, device=device
    )
    block_tables = _validate_metadata_tensor(
        "block_tables", block_tables, rank=2, device=device
    )
    seq_lens = _validate_metadata_tensor("seq_lens", seq_lens, rank=1, device=device)
    _indptr_values("cum_seq_lens_q", cum_seq_lens_q)
    batch_size = cum_seq_lens_q.numel() - 1
    if block_tables.shape[0] != batch_size or seq_lens.numel() != batch_size:
        raise ValueError(
            "dense metadata batch dimensions must agree: "
            f"cum_seq_lens_q describes {batch_size}, block_tables has "
            f"{block_tables.shape[0]}, and seq_lens has {seq_lens.numel()}."
        )

    seq_lens_host = seq_lens.to(device="cpu", dtype=torch.int64)
    if torch.any(seq_lens_host < 0).item():
        raise ValueError("seq_lens metadata must be nonnegative.")
    live_pages = torch.div(
        seq_lens_host + page_size - 1, page_size, rounding_mode="floor"
    )
    if live_pages.numel() and int(live_pages.max().item()) > block_tables.shape[1]:
        raise ValueError(
            "block_tables metadata width is smaller than the live page count "
            "implied by seq_lens and page_size."
        )
    if table_width_alignment is not None:
        if table_width_alignment <= 0:
            raise ValueError("dense table-width alignment must be positive.")
        if (
            block_tables.shape[1] == 0
            or block_tables.shape[1] % table_width_alignment != 0
        ):
            raise ValueError(
                "block_tables metadata width must be a positive multiple of "
                f"{table_width_alignment}, got {block_tables.shape[1]}."
            )

    actual_max_q_len = _max_q_len(cum_seq_lens_q)
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
        cum_seq_lens_q=cum_seq_lens_q,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_q_len=resolved_max_q_len,
    )


def _validate_csr_metadata(
    *,
    qo_indptr: object,
    kv_indptr: object,
    kv_indices: object,
    kv_len_arr: object,
    page_size: int,
    device: torch.device,
) -> _CSRPlanMetadata:
    """Validate and return complete CSR metadata."""
    if not isinstance(page_size, int) or isinstance(page_size, bool) or page_size <= 0:
        raise ValueError(f"page_size must be a positive int, got {page_size!r}.")
    qo_indptr = _validate_metadata_tensor("qo_indptr", qo_indptr, rank=1, device=device)
    kv_indptr = _validate_metadata_tensor("kv_indptr", kv_indptr, rank=1, device=device)
    kv_indices = _validate_metadata_tensor(
        "kv_indices", kv_indices, rank=1, device=device
    )
    kv_len_arr = _validate_metadata_tensor(
        "kv_len_arr", kv_len_arr, rank=1, device=device
    )
    qo_values = _indptr_values("qo_indptr", qo_indptr)
    kv_values = _indptr_values("kv_indptr", kv_indptr)
    batch_size = qo_values.numel() - 1
    if kv_values.numel() != batch_size + 1 or kv_len_arr.numel() != batch_size:
        raise ValueError(
            "CSR metadata batch dimensions must agree across qo_indptr, "
            "kv_indptr, and kv_len_arr."
        )
    kv_end = int(kv_values[-1].item())
    if kv_end > kv_indices.numel():
        raise ValueError(
            f"kv_indices metadata has {kv_indices.numel()} entries but "
            f"kv_indptr[-1] is {kv_end}."
        )
    kv_lens_host = kv_len_arr.to(device="cpu", dtype=torch.int64)
    if torch.any(kv_lens_host < 0).item():
        raise ValueError("kv_len_arr metadata must be nonnegative.")
    expected_pages = torch.div(
        kv_lens_host + page_size - 1, page_size, rounding_mode="floor"
    )
    actual_pages = kv_values[1:] - kv_values[:-1]
    if not torch.equal(expected_pages, actual_pages):
        raise ValueError(
            "CSR metadata page counts in kv_indptr must equal ceil(kv_len_arr / "
            "page_size)."
        )
    return _CSRPlanMetadata(
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        kv_len_arr=kv_len_arr,
    )


def _derive_csr_from_dense(
    dense: _DensePlanMetadata,
    *,
    page_size: int,
) -> _CSRPlanMetadata:
    """Derive CSR metadata from validated canonical dense metadata."""
    live_pages = torch.div(
        dense.seq_lens + page_size - 1,
        page_size,
        rounding_mode="floor",
    ).to(dtype=torch.int32)
    kv_indptr = torch.cat(
        (
            torch.zeros(
                (1,), dtype=torch.int32, device=dense.cum_seq_lens_q.device
            ),
            torch.cumsum(live_pages, dim=0, dtype=torch.int32),
        )
    )
    rows = [
        dense.block_tables[row, : int(live_pages[row].item())]
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
    """Derive canonical dense metadata, optionally padding its table width."""
    kv_indptr_host = csr.kv_indptr.to(device="cpu", dtype=torch.int64)
    page_counts = kv_indptr_host[1:] - kv_indptr_host[:-1]
    max_pages = int(page_counts.max().item()) if page_counts.numel() else 0
    if table_width_alignment is None:
        table_width = max(1, max_pages)
    else:
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


class _MLAPlanMetadataResolver:
    """Lazily validate and normalize metadata for one wrapper plan request.

    The resolver intentionally owns no backend policy.  A backend adapter asks
    for its native representation, and the resolver validates or derives only
    what that request needs.  All caches are instance-local because caller-owned
    tensors may be mutated between separate ``plan()`` calls.
    """

    def __init__(
        self,
        *,
        qo_indptr: object = None,
        kv_indptr: object = None,
        kv_indices: object = None,
        kv_len_arr: object = None,
        cum_seq_lens_q: object = None,
        block_tables: object = None,
        seq_lens: object = None,
        max_q_len: object = None,
        kv_len: object = None,
        page_table: object = None,
        page_size: int,
        device: torch.device,
    ) -> None:
        self._raw_csr = (qo_indptr, kv_indptr, kv_indices, kv_len_arr)
        self._raw_dense = (cum_seq_lens_q, block_tables, seq_lens, max_q_len)
        self._raw_legacy = (kv_len, page_table)
        self.page_size = page_size
        self.device = device

        self._forms_checked = False
        self._has_csr = False
        self._has_dense = False

        self._validated_csr: Optional[_CSRPlanMetadata] = None
        self._validated_dense: Optional[_DensePlanMetadata] = None
        self._derived_csr: Optional[_CSRPlanMetadata] = None
        self._derived_dense_by_alignment: dict[int, _DensePlanMetadata] = {}
        self._derived_native_dense: Optional[_DensePlanMetadata] = None
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
        legacy_present = tuple(value is not None for value in self._raw_legacy)

        if any(csr_present) and not all(csr_present):
            missing = [
                name
                for name, present in zip(
                    ("qo_indptr", "kv_indptr", "kv_indices", "kv_len_arr"),
                    csr_present,
                    strict=True,
                )
                if not present
            ]
            raise ValueError(
                "CSR metadata form is partial; missing required fields: "
                + ", ".join(missing)
                + "."
            )
        if any(dense_present) and not all(dense_present):
            missing = [
                name
                for name, present in zip(
                    ("cum_seq_lens_q", "block_tables", "seq_lens"),
                    dense_present,
                    strict=True,
                )
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
        if any(legacy_present) and not all(legacy_present):
            raise ValueError("kv_len and page_table must be provided together.")
        if all(dense_present) and all(legacy_present):
            raise ValueError(
                "kv_len/page_table duplicate canonical dense metadata and are only "
                "a compatibility exception with the legacy CSR form."
            )

        self._has_csr = all(csr_present)
        self._has_dense = all(dense_present)
        if not self._has_csr and not self._has_dense:
            raise ValueError("A complete CSR or dense metadata form is required.")
        self._forms_checked = True

    @staticmethod
    def _check_table_width_alignment(table_width_alignment: int) -> None:
        if (
            not isinstance(table_width_alignment, int)
            or isinstance(table_width_alignment, bool)
            or table_width_alignment <= 0
        ):
            raise ValueError("dense table-width alignment must be positive.")

    def _validate_csr(self) -> _CSRPlanMetadata:
        if self._validated_csr is not None:
            return self._validated_csr
        assert self._has_csr
        qo_indptr, kv_indptr, kv_indices, kv_len_arr = self._raw_csr
        self._validated_csr = _validate_csr_metadata(
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            kv_len_arr=kv_len_arr,
            page_size=self.page_size,
            device=self.device,
        )
        return self._validated_csr

    @staticmethod
    def _check_dense_alignment(
        dense: _DensePlanMetadata, table_width_alignment: int
    ) -> None:
        if (
            dense.block_tables.shape[1] == 0
            or dense.block_tables.shape[1] % table_width_alignment != 0
        ):
            raise ValueError(
                "block_tables metadata width must be a positive multiple of "
                f"{table_width_alignment}, got {dense.block_tables.shape[1]}."
            )

    def _validate_dense(
        self, table_width_alignment: Optional[int]
    ) -> _DensePlanMetadata:
        assert self._has_dense
        if table_width_alignment is not None:
            self._check_table_width_alignment(table_width_alignment)

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
            self._check_dense_alignment(
                self._validated_dense, table_width_alignment
            )
        return self._validated_dense

    def _derive_csr(self, dense: _DensePlanMetadata) -> _CSRPlanMetadata:
        if self._derived_csr is None:
            self._derived_csr = _derive_csr_from_dense(
                dense,
                page_size=self.page_size,
            )
        return self._derived_csr

    def _ensure_equivalent(
        self,
        csr: _CSRPlanMetadata,
        dense: _DensePlanMetadata,
    ) -> None:
        if self._equivalence_checked:
            return

        dense_as_csr = self._derive_csr(dense)
        csr_live_end = int(csr.kv_indptr[-1].item())
        equivalent = (
            torch.equal(csr.qo_indptr, dense_as_csr.qo_indptr)
            and torch.equal(csr.kv_indptr, dense_as_csr.kv_indptr)
            and torch.equal(csr.kv_indices[:csr_live_end], dense_as_csr.kv_indices)
            and torch.equal(csr.kv_len_arr, dense_as_csr.kv_len_arr)
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
        if csr is None:
            csr = self._validate_csr()
        if dense is None:
            dense = self._validate_dense(None)
        self._ensure_equivalent(csr, dense)

    def resolve_csr(self) -> _CSRPlanMetadata:
        """Return validated native CSR metadata, or derive it from dense."""
        self._check_forms()
        if self._has_csr:
            csr = self._validate_csr()
            self._ensure_dual_forms_equivalent(csr=csr)
            return csr

        dense = self._validate_dense(None)
        return self._derive_csr(dense)

    @property
    def resolved_csr(self) -> Optional[_CSRPlanMetadata]:
        """Return the CSR form already validated or derived by this resolver."""
        return self._validated_csr or self._derived_csr

    @property
    def resolved_dense(self) -> Optional[_DensePlanMetadata]:
        """Return the validated supplied dense form, when present."""
        return self._validated_dense

    def resolve_dense(
        self,
        *,
        table_width_alignment: int,
    ) -> _DensePlanMetadata:
        """Return validated or derived dense metadata for one native policy."""
        self._check_forms()
        self._check_table_width_alignment(table_width_alignment)

        if self._has_dense:
            dense = self._validate_dense(table_width_alignment)
            self._ensure_dual_forms_equivalent(dense=dense)
            return dense

        if table_width_alignment not in self._derived_dense_by_alignment:
            csr = self._validate_csr()
            self._derived_dense_by_alignment[table_width_alignment] = (
                _derive_dense_from_csr(
                    csr,
                    table_width_alignment=table_width_alignment,
                )
            )
        return self._derived_dense_by_alignment[table_width_alignment]

    def resolve_native_dense(self) -> _DensePlanMetadata:
        """Return validated or derived dense metadata without width padding."""
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


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class _MLAPlanArguments:
    """Normalized public arguments and lazy metadata for one plan request."""

    qo_indptr: Optional[torch.Tensor] = None
    kv_indptr: Optional[torch.Tensor] = None
    kv_indices: Optional[torch.Tensor] = None
    kv_len_arr: Optional[torch.Tensor] = None
    num_heads: int
    head_dim_ckv: int
    head_dim_kpe: int
    page_size: int
    causal: bool
    sm_scale: float
    q_data_type: torch.dtype
    kv_data_type: torch.dtype
    use_profiler: bool = False
    cum_seq_lens_q: Optional[torch.Tensor] = None
    block_tables: Optional[torch.Tensor] = None
    seq_lens: Optional[torch.Tensor] = None
    max_q_len: Optional[int] = None
    qk_nope_head_dim: Optional[int] = None
    kv_len: Optional[torch.Tensor] = None
    page_table: Optional[torch.Tensor] = None
    enable_pdl: Optional[bool] = None
    is_var_seq: Optional[bool] = None
    cute_dsl_impl: str = "auto"
    use_sinks: bool = False
    device: InitVar[torch.device]
    _metadata_resolver: _MLAPlanMetadataResolver = field(init=False, repr=False)

    def __post_init__(self, device: torch.device) -> None:
        object.__setattr__(
            self,
            "_metadata_resolver",
            _MLAPlanMetadataResolver(
                qo_indptr=self.qo_indptr,
                kv_indptr=self.kv_indptr,
                kv_indices=self.kv_indices,
                kv_len_arr=self.kv_len_arr,
                cum_seq_lens_q=self.cum_seq_lens_q,
                block_tables=self.block_tables,
                seq_lens=self.seq_lens,
                max_q_len=self.max_q_len,
                kv_len=self.kv_len,
                page_table=self.page_table,
                page_size=self.page_size,
                device=device,
            ),
        )

    @property
    def has_canonical_dense_metadata(self) -> bool:
        """Whether any canonical dense-form argument was supplied."""
        return any(
            value is not None
            for value in (
                self.cum_seq_lens_q,
                self.block_tables,
                self.seq_lens,
                self.max_q_len,
            )
        )

    @property
    def has_legacy_dense_metadata(self) -> bool:
        """Whether either CUTLASS compatibility argument was supplied."""
        return self.kv_len is not None or self.page_table is not None

    @property
    def csr(self) -> _CSRPlanMetadata:
        return self._metadata_resolver.resolve_csr()

    def dense(self, *, table_width_alignment: int) -> _DensePlanMetadata:
        return self._metadata_resolver.resolve_dense(
            table_width_alignment=table_width_alignment
        )

    @property
    def native_dense(self) -> _DensePlanMetadata:
        return self._metadata_resolver.resolve_native_dense()

    @property
    def resolved_csr(self) -> Optional[_CSRPlanMetadata]:
        return self._metadata_resolver.resolved_csr

    @property
    def resolved_dense(self) -> Optional[_DensePlanMetadata]:
        return self._metadata_resolver.resolved_dense


@dataclass(frozen=True)
class _MLAWrapperPlanResult:
    backend_impl: object
    csr: Optional[_CSRPlanMetadata]
    dense: Optional[_DensePlanMetadata]
