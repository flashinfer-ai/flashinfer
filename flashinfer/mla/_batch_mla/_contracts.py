"""Shared input, planning, and functional contracts for Batch MLA."""

from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Literal, Optional, Union, overload

import torch

from flashinfer.autotuner import TunableRunner


MLAInputAvailability = Literal["packed", "split", "redundant"]
MLAChosenRepresentation = Literal["packed", "split"]
MLAStructuralInputKind = Literal[
    "packed", "adjacent-split", "independent-split", "dual"
]


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
    return (
        _as_tensor_leaf(left, name=name),
        _as_tensor_leaf(right, name=name),
    )


def _parse_structural_mla_input(
    value: object, *, name: str
) -> tuple[
    Optional[torch.Tensor],
    Optional[tuple[torch.Tensor, torch.Tensor]],
]:
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
        return (
            _as_tensor_leaf(right, name=name),
            _as_split_structural_pair(left, name=name),
        )
    if right_is_tuple:
        return (
            _as_tensor_leaf(left, name=name),
            _as_split_structural_pair(right, name=name),
        )
    return None, (
        _as_tensor_leaf(left, name=name),
        _as_tensor_leaf(right, name=name),
    )


def _structural_mla_input_facts(
    value: object,
    *,
    widths: tuple[int, int],
    name: str,
) -> tuple[MLAStructuralInputKind, torch.dtype, tuple[int, ...]]:
    """Validate and describe an input while trusting an unselected dual member."""
    packed, split = _parse_structural_mla_input(value, name=name)
    packed_width = sum(widths)
    if packed is not None:
        kind: MLAStructuralInputKind = "dual" if split is not None else "packed"
        tensor = packed
        if tensor.ndim != 3:
            raise ValueError(f"packed {name} must have rank 3.")
        if tensor.shape[-1] != packed_width:
            raise ValueError(
                f"packed {name} last dimension does not match explicit split widths."
            )
        return kind, tensor.dtype, tuple(tensor.shape)

    assert split is not None
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
        raise ValueError(f"split {name} last dimensions do not match explicit widths.")
    kind = (
        "adjacent-split"
        if _are_adjacent_last_dim_views(left, right)
        else "independent-split"
    )
    canonical_shape = tuple(left.shape[:-1]) + (packed_width,)
    return kind, left.dtype, canonical_shape


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
    if type(value) is tuple:
        if len(value) != 2:
            raise ValueError(f"{name} structural MLA input tuples must have length 2.")
        left, right = value
        left_is_tuple = type(left) is tuple
        right_is_tuple = type(right) is tuple
        if left_is_tuple != right_is_tuple:
            if desired == "packed":
                return _as_tensor_leaf(right if left_is_tuple else left, name=name)
            return _as_split_structural_pair(
                left if left_is_tuple else right, name=name
            )
    packed, split = _parse_structural_mla_input(value, name=name)
    if desired == "split":
        if split is not None:
            return split
        return _split_mla_tensor_references(
            packed=packed,
            left=None,
            right=None,
            representation="packed",
            widths=widths,
            name=name,
        )
    if packed is not None:
        return packed
    assert split is not None
    left, right = split
    return _packed_mla_tensor_reference(
        packed=None,
        left=left,
        right=right,
        representation="split",
        widths=widths,
        name=name,
    )


def _validate_mla_reference_presence(
    *, packed: object, split_1: object, split_2: object, name: str
) -> None:
    """Validate MLA representation availability using identity checks only."""
    has_packed = packed is not None
    has_split_1 = split_1 is not None
    has_split_2 = split_2 is not None
    if has_split_1 != has_split_2 or not (has_packed or has_split_1):
        raise ValueError(
            f"{name} requires packed, complete split, or trusted redundant tensors."
        )


def _classify_mla_references(
    *,
    packed: object,
    split_1: object,
    split_2: object,
    name: str,
) -> MLAInputAvailability:
    """Classify complete MLA representations using presence checks only."""
    _validate_mla_reference_presence(
        packed=packed,
        split_1=split_1,
        split_2=split_2,
        name=name,
    )
    has_packed = packed is not None
    has_split_1 = split_1 is not None
    if has_packed and has_split_1:
        return "redundant"
    return "packed" if has_packed else "split"


def _choose_mla_representation(
    availability: MLAInputAvailability, preferred: MLAChosenRepresentation
) -> MLAChosenRepresentation:
    """Resolve trusted redundant references to one backend-native form."""
    return preferred if availability == "redundant" else availability


def _choose_mla_references(
    *,
    packed: Optional[torch.Tensor],
    split_1: Optional[torch.Tensor],
    split_2: Optional[torch.Tensor],
    availability: MLAInputAvailability,
    preferred: MLAChosenRepresentation,
) -> tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    MLAChosenRepresentation,
]:
    """Keep only the complete form selected for a backend."""
    representation = _choose_mla_representation(availability, preferred)
    if representation == "packed":
        assert packed is not None
        return packed, None, None, representation
    assert split_1 is not None and split_2 is not None
    return None, split_1, split_2, representation


def _adjacent_last_dim_view(
    left: torch.Tensor, right: torch.Tensor
) -> Optional[torch.Tensor]:
    """Return one in-bounds contiguous view when split tensors are adjacent."""
    if not _are_adjacent_last_dim_views(left, right):
        return None
    shape = left.shape[:-1] + (left.shape[-1] + right.shape[-1],)
    return left.as_strided(shape, left.stride(), left.storage_offset())


def _are_adjacent_last_dim_views(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Return whether split tensors form one in-bounds contiguous tensor."""
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
        return False
    storage = left.untyped_storage()
    if storage.data_ptr() != right.untyped_storage().data_ptr():
        return False
    stride = left.stride()
    storage_offset = left.storage_offset()
    if right.storage_offset() != storage_offset + left.shape[-1] * stride[-1]:
        return False
    shape = left.shape[:-1] + (left.shape[-1] + right.shape[-1],)
    storage_numel, remainder = divmod(storage.nbytes(), left.element_size())
    if remainder:
        return False
    if any(size == 0 for size in shape):
        if storage_offset > storage_numel:
            return False
        return True
    else:
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


def _split_mla_tensor_references(
    *,
    packed: Optional[torch.Tensor],
    left: Optional[torch.Tensor],
    right: Optional[torch.Tensor],
    representation: MLAChosenRepresentation,
    widths: Optional[tuple[int, int]],
    name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a backend-native split pair from validated raw references."""
    if representation == "split":
        assert left is not None and right is not None
        return left, right
    assert packed is not None
    if widths is None:
        raise ValueError(f"packed {name} requires planned split widths.")
    left_width, right_width = widths
    if packed.ndim == 0:
        raise ValueError(f"packed {name} must have at least one dimension.")
    if left_width <= 0 or right_width <= 0:
        raise ValueError("split widths must be positive.")
    if packed.shape[-1] != left_width + right_width:
        raise ValueError(
            f"packed {name} last dimension does not match planned split widths."
        )
    return packed[..., :left_width], packed[..., left_width:]


def _packed_mla_tensor_reference(
    *,
    packed: Optional[torch.Tensor],
    left: Optional[torch.Tensor],
    right: Optional[torch.Tensor],
    representation: MLAChosenRepresentation,
    widths: Optional[tuple[int, int]],
    name: str,
) -> torch.Tensor:
    """Return a backend-native packed tensor without representation copies."""
    if representation == "packed":
        assert packed is not None
        return packed
    assert left is not None and right is not None
    adjacent = _adjacent_last_dim_view(left, right)
    if adjacent is not None:
        return adjacent
    raise ValueError(
        f"{name} cannot provide the planned packed representation zero-copy; "
        "re-plan for split input."
    )


@dataclass(frozen=True)
class MLAInputContract:
    """Run options that must remain compatible with an MLA plan."""

    lse_mode: str
    output_dtype: torch.dtype
    output_scale: str
    scale_mode: str
    skip_softmax: bool
    query_layout: MLAChosenRepresentation = "packed"
    kv_cache_layout: MLAChosenRepresentation = "packed"
    head_dim_ckv: Optional[int] = None
    head_dim_kpe: Optional[int] = None


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
    query: Optional[torch.Tensor]
    q_nope: Optional[torch.Tensor]
    q_pe: Optional[torch.Tensor]
    kv_cache: Optional[torch.Tensor]
    ckv_cache: Optional[torch.Tensor]
    kpe_cache: Optional[torch.Tensor]
    query_availability: MLAInputAvailability
    kv_availability: MLAInputAvailability
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
    native_query_representation: MLAChosenRepresentation
    native_kv_representation: MLAChosenRepresentation

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
