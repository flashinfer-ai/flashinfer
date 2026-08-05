"""Shared input, planning, and functional contracts for Batch MLA."""

from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Union

import torch

from flashinfer.autotuner import TunableRunner


def _adjacent_last_dim_view(
    left: torch.Tensor, right: torch.Tensor
) -> Optional[torch.Tensor]:
    """Return one in-bounds contiguous view when split tensors are adjacent."""
    if (
        left.ndim == 0
        or left.ndim != right.ndim
        or left.shape[:-1] != right.shape[:-1]
        or left.dtype != right.dtype
        or left.device != right.device
        or left.stride() != right.stride()
        or left.shape[-1] == 0
        or right.shape[-1] == 0
    ):
        return None
    storage = left.untyped_storage()
    if storage.data_ptr() != right.untyped_storage().data_ptr():
        return None
    stride = left.stride()
    storage_offset = left.storage_offset()
    if right.storage_offset() != storage_offset + left.shape[-1] * stride[-1]:
        return None
    shape = left.shape[:-1] + (left.shape[-1] + right.shape[-1],)
    storage_numel, remainder = divmod(storage.nbytes(), left.element_size())
    if remainder:
        return None
    if any(size == 0 for size in shape):
        if storage_offset > storage_numel:
            return None
    else:
        last_offset = storage_offset + sum(
            (size - 1) * step for size, step in zip(shape, stride, strict=True)
        )
        if last_offset >= storage_numel:
            return None
    combined = left.as_strided(shape, stride, storage_offset)
    return combined if combined.is_contiguous() else None


def _concat_adjacent_views_or_cat(
    left: torch.Tensor, right: torch.Tensor
) -> torch.Tensor:
    """Join adjacent split views without a copy, otherwise preserve cat fallback."""
    combined = _adjacent_last_dim_view(left, right)
    return torch.cat((left, right), dim=-1) if combined is None else combined


@dataclass(frozen=True)
class MLAInputContract:
    """Tensor-free plan facts that a run-time MLA input must satisfy."""

    query_split_widths: tuple[int, int]
    kv_split_widths: tuple[int, int]
    q_data_type: torch.dtype
    kv_data_type: torch.dtype
    kv_layout: str
    lse_mode: str
    output_dtype: torch.dtype
    output_scale: str
    scale_mode: str
    skip_softmax: bool

    def validate(self, query: "MLAQuery", kv: "MLAKVCache") -> None:
        """Reject value objects that are incompatible with this plan."""
        q_nope, q_pe = query.split_views(self.query_split_widths)
        ckv_cache, kpe_cache = kv.split_views(self.kv_split_widths)
        if kv.layout != self.kv_layout:
            raise ValueError(
                "MLA planned input contract mismatch: KV layout "
                f"planned {self.kv_layout!r}, got {kv.layout!r}; re-plan with "
                "the needed arguments."
            )
        for name, tensor in (("q_nope", q_nope), ("q_pe", q_pe)):
            if tensor.dtype != self.q_data_type:
                raise ValueError(
                    "MLA planned input contract mismatch: "
                    f"{name} dtype planned {self.q_data_type!r}, got "
                    f"{tensor.dtype!r}; re-plan with the needed arguments."
                )
        for name, tensor, dtype in (
            ("ckv_cache", ckv_cache, self.kv_data_type),
            ("kpe_cache", kpe_cache, self.kv_data_type),
        ):
            if tensor.dtype != dtype:
                raise ValueError(
                    "MLA planned input contract mismatch: "
                    f"{name} dtype planned {dtype!r}, got {tensor.dtype!r}; "
                    "re-plan with the needed arguments."
                )
        for name, tensor, width in (
            ("q_nope", q_nope, self.query_split_widths[0]),
            ("q_pe", q_pe, self.query_split_widths[1]),
            ("ckv_cache", ckv_cache, self.kv_split_widths[0]),
            ("kpe_cache", kpe_cache, self.kv_split_widths[1]),
        ):
            if tensor.shape[-1] != width:
                raise ValueError(
                    "MLA planned input contract mismatch: "
                    f"{name} last dimension planned {width}, got "
                    f"{tensor.shape[-1]}; re-plan with the needed arguments."
                )


@dataclass(frozen=True)
class MLAQuery:
    """One packed or split MLA query representation."""

    q: Optional[torch.Tensor] = None
    q_nope: Optional[torch.Tensor] = None
    q_pe: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        has_packed = self.q is not None
        has_split = self.q_nope is not None or self.q_pe is not None
        if has_packed == has_split or (
            has_split and (self.q_nope is None or self.q_pe is None)
        ):
            raise ValueError(
                "MLAQuery requires exactly one packed or complete split representation."
            )
        if any(
            value is not None and not isinstance(value, torch.Tensor)
            for value in (self.q, self.q_nope, self.q_pe)
        ):
            raise TypeError(
                "MLAQuery representation values must be torch.Tensor instances."
            )

    @classmethod
    def packed(cls, q: torch.Tensor) -> "MLAQuery":
        return cls(q=q)

    @classmethod
    def split(cls, q_nope: torch.Tensor, q_pe: torch.Tensor) -> "MLAQuery":
        return cls(q_nope=q_nope, q_pe=q_pe)

    def split_views(
        self, widths: Optional[tuple[int, int]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return split query tensors, deriving zero-copy views when packed."""
        if self.q is None:
            if self.q_nope is None or self.q_pe is None:
                raise ValueError("provide MLAQuery.packed or MLAQuery.split inputs.")
            return self.q_nope, self.q_pe
        if widths is None:
            raise ValueError("MLAQuery.packed requires plan() before run().")
        q_nope_width, q_pe_width = widths
        if self.q.ndim == 0:
            raise ValueError("packed query must have at least one dimension.")
        if q_nope_width <= 0 or q_pe_width <= 0:
            raise ValueError("split widths must be positive.")
        if self.q.shape[-1] != q_nope_width + q_pe_width:
            raise ValueError(
                "packed query last dimension does not match planned split widths."
            )
        return self.q[..., :q_nope_width], self.q[..., q_nope_width:]

    def packed_or_adjacent(self) -> Optional[torch.Tensor]:
        """Return the packed query or a zero-copy view of adjacent splits."""
        if self.q is not None:
            return self.q
        if not isinstance(self.q_nope, torch.Tensor) or not isinstance(
            self.q_pe, torch.Tensor
        ):
            return None
        return _adjacent_last_dim_view(self.q_nope, self.q_pe)

    def packed_or_cat(self) -> torch.Tensor:
        """Return a packed query, copying only non-adjacent split inputs."""
        packed = self.packed_or_adjacent()
        if packed is not None:
            return packed
        if not isinstance(self.q_nope, torch.Tensor) or not isinstance(
            self.q_pe, torch.Tensor
        ):
            raise ValueError("provide MLAQuery.packed or MLAQuery.split inputs.")
        return torch.cat((self.q_nope, self.q_pe), dim=-1)

    def require_packed(self) -> torch.Tensor:
        """Return the packed query, copying non-adjacent split inputs if needed."""
        return self.packed_or_cat()


@dataclass(frozen=True)
class MLAKVCache:
    """One packed or split paged MLA KV-cache representation."""

    kv_cache: Optional[torch.Tensor] = None
    ckv_cache: Optional[torch.Tensor] = None
    kpe_cache: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        has_packed = self.kv_cache is not None
        has_split = self.ckv_cache is not None or self.kpe_cache is not None
        if has_packed == has_split or (
            has_split and (self.ckv_cache is None or self.kpe_cache is None)
        ):
            raise ValueError(
                "MLAKVCache requires exactly one packed or complete split representation."
            )
        if any(
            value is not None and not isinstance(value, torch.Tensor)
            for value in (self.kv_cache, self.ckv_cache, self.kpe_cache)
        ):
            raise TypeError(
                "MLAKVCache representation values must be torch.Tensor instances."
            )

    @classmethod
    def packed(cls, kv_cache: torch.Tensor) -> "MLAKVCache":
        return cls(kv_cache=kv_cache)

    @classmethod
    def split(cls, ckv_cache: torch.Tensor, kpe_cache: torch.Tensor) -> "MLAKVCache":
        return cls(ckv_cache=ckv_cache, kpe_cache=kpe_cache)

    def split_views(
        self, widths: Optional[tuple[int, int]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return split KV tensors, deriving zero-copy views when packed."""
        if self.kv_cache is None:
            if self.ckv_cache is None or self.kpe_cache is None:
                raise ValueError(
                    "provide MLAKVCache.packed or MLAKVCache.split inputs."
                )
            return self.ckv_cache, self.kpe_cache
        if widths is None:
            raise ValueError("MLAKVCache.packed requires planned split widths.")
        ckv_width, kpe_width = widths
        if self.kv_cache.ndim == 0:
            raise ValueError("packed KV-cache must have at least one dimension.")
        if ckv_width <= 0 or kpe_width <= 0:
            raise ValueError("split widths must be positive.")
        if self.kv_cache.shape[-1] != ckv_width + kpe_width:
            raise ValueError(
                "packed KV-cache last dimension does not match planned split widths."
            )
        return self.kv_cache[..., :ckv_width], self.kv_cache[..., ckv_width:]

    def packed_or_adjacent(self) -> Optional[torch.Tensor]:
        """Return the packed cache or a zero-copy packed view of adjacent splits."""
        if self.kv_cache is not None:
            return self.kv_cache
        if not isinstance(self.ckv_cache, torch.Tensor) or not isinstance(
            self.kpe_cache, torch.Tensor
        ):
            return None
        return _adjacent_last_dim_view(self.ckv_cache, self.kpe_cache)

    def require_packed_view(self) -> torch.Tensor:
        """Return a zero-copy packed cache view or reject independent splits."""
        packed = self.packed_or_adjacent()
        if packed is None:
            raise ValueError(
                "MLA KV cache must be packed or use adjacent split tensor views."
            )
        return packed

    @property
    def layout(self) -> str:
        if self.kv_cache is not None:
            return "combined"
        return (
            "adjacent-split"
            if self.packed_or_adjacent() is not None
            else "independent-split"
        )


def _split_mla_value_objects(
    query: MLAQuery,
    kv: MLAKVCache,
    contract: Optional[MLAInputContract],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return backend-native split views from raw wrapper value objects."""
    query_widths = None if contract is None else contract.query_split_widths
    if (
        query_widths is None
        and query.q is not None
        and kv.ckv_cache is not None
        and kv.kpe_cache is not None
    ):
        query_widths = (kv.ckv_cache.shape[-1], kv.kpe_cache.shape[-1])
    q_nope, q_pe = query.split_views(query_widths)
    kv_widths = (
        (q_nope.shape[-1], q_pe.shape[-1])
        if contract is None
        else contract.kv_split_widths
    )
    ckv_cache, kpe_cache = kv.split_views(kv_widths)
    return q_nope, q_pe, ckv_cache, kpe_cache


@dataclass(frozen=True)
class MLAPlanMetadata:
    """Canonical CSR and/or dense metadata for one Batch MLA plan.

    Use :meth:`csr` for metadata already represented as CSR. It is native to
    FA2 and FA3. Use :meth:`dense` for page-table metadata; it is native to
    CUTLASS, TRTLLM-GEN, CuTe DSL, and XQA. :meth:`dual` is useful when both
    forms already exist and asserts that they describe the same requests.

    The object only retains the supplied references. ``plan()`` lazily derives
    the selected backend's other form only when that backend needs it.
    """

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
        """Create CSR metadata, the native representation for FA2 and FA3."""
        return cls(qo_indptr, kv_indptr, kv_indices, kv_len_arr)

    @classmethod
    def dense(
        cls,
        cum_seq_lens_q: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_q_len: Optional[int] = None,
    ) -> "MLAPlanMetadata":
        """Create dense page-table metadata, native to dense MLA backends."""
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
        """Create equivalent CSR and dense metadata without deriving either form."""
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
class _FunctionalMLARequest:
    query: torch.Tensor
    kv_cache: torch.Tensor
    workspace_buffer: torch.Tensor
    qk_nope_head_dim: int
    kv_lora_rank: int
    qk_rope_head_dim: int
    block_tables: torch.Tensor
    seq_lens: Optional[torch.Tensor]
    max_seq_len: int
    sparse_mla_top_k: int
    out: Optional[torch.Tensor]
    bmm1_scale: Union[float, torch.Tensor]
    bmm2_scale: Union[float, torch.Tensor]
    sinks: Optional[List[torch.Tensor]]
    skip_softmax_threshold_scale_factor: Optional[float]
    enable_pdl: Optional[bool]
    is_var_seq: bool
    uses_shared_paged_kv_idx: bool
    lse: Optional[torch.Tensor]
    return_lse: bool
    cute_dsl_impl: str
    kv_scale_format: str
    cum_seq_lens_q: Optional[torch.Tensor]
    max_q_len: Optional[int]
    multi_ctas_kv_counter_buffer: Optional[torch.Tensor]
    sparse_mla_top_k_lens: Optional[torch.Tensor]
    enable_dcp: bool
    cp_world: int
    cp_rank: int
    causal_seqlens_kv_global: Optional[torch.Tensor]


class _FunctionalMLARunner(TunableRunner):
    def __init__(self, request: _FunctionalMLARequest) -> None:
        self.request = request

    @property
    @abstractmethod
    def inputs(self) -> list[torch.Tensor]:
        raise NotImplementedError

    def prepare_for_dispatch(self) -> None:
        """Prepare caller-specific state before explicit or tuned dispatch."""


class _FunctionalBackendUnsupportedError(RuntimeError):
    """The backend cannot implement a valid request; auto may try another."""
