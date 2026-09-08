"""Generated-FA state, validation, and staging for MLA CUDA-graph plan updates."""

from dataclasses import dataclass
from typing import Literal, Sequence

import torch

from .._contracts import MLAPlanMetadata


@dataclass(frozen=True)
class _MLACudaGraphTensorIdentity:
    object_id: int
    data_ptr: int
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


@dataclass(frozen=True)
class _MLACudaGraphFrozenContract:
    backend_type: type[object]
    module: object
    device: torch.device
    float_workspace: _MLACudaGraphTensorIdentity
    int_workspace: _MLACudaGraphTensorIdentity
    qo_indptr: _MLACudaGraphTensorIdentity
    kv_indptr: _MLACudaGraphTensorIdentity
    kv_indices: _MLACudaGraphTensorIdentity
    kv_len_arr: _MLACudaGraphTensorIdentity
    batch_size: int
    total_qo_rows: int
    num_heads: int
    head_dim_ckv: int
    page_size: int
    causal: bool
    sm_scale: float
    q_data_type: torch.dtype
    kv_data_type: torch.dtype
    use_profiler: bool
    plan_info: tuple[int, ...]
    staged_int_workspace_bytes: int


@dataclass(frozen=True)
class _ResolvedMLACudaGraphPlanUpdate:
    qo_indptr: torch.Tensor
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    kv_len_arr: torch.Tensor
    live_kv_indices: int


@dataclass
class _MLACudaGraphPlanCandidateState:
    int_workspace: torch.Tensor
    qo_indptr: torch.Tensor
    kv_indptr: torch.Tensor
    kv_len_arr: torch.Tensor


@dataclass
class _MLACudaGraphPlanUpdateSlot:
    planner_workspace: torch.Tensor
    qo_indptr: torch.Tensor
    kv_indptr: torch.Tensor
    kv_len_arr: torch.Tensor
    completion_event: torch.cuda.Event
    status: Literal["idle", "in-use", "pending", "poisoned"]


@dataclass
class _MLACudaGraphPlanUpdateState:
    frozen: _MLACudaGraphFrozenContract
    candidate: _MLACudaGraphPlanCandidateState
    slots: tuple[_MLACudaGraphPlanUpdateSlot, _MLACudaGraphPlanUpdateSlot]


_TENSOR_IDENTITY_FIELDS = (
    "float_workspace",
    "int_workspace",
    "qo_indptr",
    "kv_indptr",
    "kv_indices",
    "kv_len_arr",
)
_CPU_DEVICE = torch.device("cpu")


def _tensor_identity(tensor: torch.Tensor) -> _MLACudaGraphTensorIdentity:
    return _MLACudaGraphTensorIdentity(
        object_id=id(tensor),
        data_ptr=tensor.data_ptr(),
        shape=tuple(tensor.shape),
        dtype=tensor.dtype,
        device=tensor.device,
    )


def _make_mla_cuda_graph_frozen_contract(
    *,
    backend_type: type[object],
    module: object,
    device: torch.device,
    float_workspace: torch.Tensor,
    int_workspace: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_len_arr: torch.Tensor,
    batch_size: int,
    total_qo_rows: int,
    num_heads: int,
    head_dim_ckv: int,
    page_size: int,
    causal: bool,
    sm_scale: float,
    q_data_type: torch.dtype,
    kv_data_type: torch.dtype,
    use_profiler: bool,
    plan_info: Sequence[int],
    staged_int_workspace_bytes: int,
) -> _MLACudaGraphFrozenContract:
    tensors = (
        float_workspace,
        int_workspace,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_len_arr,
    )
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError("CUDA graph frozen buffers must be torch.Tensor instances.")
    captured_device = float_workspace.device
    requested_device = torch.device(device)
    if requested_device.type != captured_device.type or (
        requested_device.index is not None and requested_device != captured_device
    ):
        raise ValueError(
            "CUDA graph frozen device must match the float workspace device."
        )
    for tensor in tensors:
        if tensor.device != captured_device:
            raise ValueError("CUDA graph frozen buffers must share one device.")
        if not tensor.is_contiguous():
            raise ValueError("CUDA graph frozen buffers must be contiguous.")
    if not isinstance(staged_int_workspace_bytes, int) or isinstance(
        staged_int_workspace_bytes, bool
    ):
        raise TypeError("staged_int_workspace_bytes must be an int.")
    int_workspace_capacity_bytes = int_workspace.numel() * int_workspace.element_size()
    if not 0 <= staged_int_workspace_bytes <= int_workspace_capacity_bytes:
        raise ValueError(
            "staged_int_workspace_bytes exceeds the integer workspace capacity."
        )
    return _MLACudaGraphFrozenContract(
        backend_type=backend_type,
        module=module,
        device=captured_device,
        float_workspace=_tensor_identity(float_workspace),
        int_workspace=_tensor_identity(int_workspace),
        qo_indptr=_tensor_identity(qo_indptr),
        kv_indptr=_tensor_identity(kv_indptr),
        kv_indices=_tensor_identity(kv_indices),
        kv_len_arr=_tensor_identity(kv_len_arr),
        batch_size=batch_size,
        total_qo_rows=total_qo_rows,
        num_heads=num_heads,
        head_dim_ckv=head_dim_ckv,
        page_size=page_size,
        causal=causal,
        sm_scale=sm_scale,
        q_data_type=q_data_type,
        kv_data_type=kv_data_type,
        use_profiler=use_profiler,
        plan_info=tuple(plan_info),
        staged_int_workspace_bytes=staged_int_workspace_bytes,
    )


def _make_mla_cuda_graph_plan_update_state(
    *,
    frozen: _MLACudaGraphFrozenContract,
    reusable_device_workspace: torch.Tensor | None = None,
    reusable_planner_workspace: torch.Tensor | None = None,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_len_arr: torch.Tensor,
) -> _MLACudaGraphPlanUpdateState:
    """Allocate, seed, and warm the bounded state owned by one graph plan."""

    staged_int_workspace_bytes = frozen.staged_int_workspace_bytes
    with torch.cuda.device(frozen.device):
        candidate_int_workspace: torch.Tensor | None = None
        if reusable_device_workspace is not None:
            if reusable_device_workspace.device != frozen.device:
                raise ValueError(
                    "reusable device workspace must be on the plan device."
                )
            if not reusable_device_workspace.is_contiguous():
                raise ValueError("reusable device workspace must be contiguous.")
            reusable_device_bytes = reusable_device_workspace.view(torch.uint8)
            if reusable_device_bytes.numel() >= 2 * staged_int_workspace_bytes:
                # The captured kernels address only the frozen staged prefix.
                # Keep the next schedule in the unused tail until the native
                # commit publishes it back to that prefix.
                candidate_int_workspace = reusable_device_bytes[
                    staged_int_workspace_bytes : 2 * staged_int_workspace_bytes
                ]
        if candidate_int_workspace is None:
            candidate_int_workspace = torch.empty(
                (staged_int_workspace_bytes,),
                dtype=torch.uint8,
                device=frozen.device,
            )
        candidate = _MLACudaGraphPlanCandidateState(
            int_workspace=candidate_int_workspace,
            qo_indptr=torch.empty_like(qo_indptr),
            kv_indptr=torch.empty_like(kv_indptr),
            kv_len_arr=torch.empty_like(kv_len_arr),
        )

        def _new_planner_workspace() -> torch.Tensor:
            return torch.empty(
                (staged_int_workspace_bytes,),
                dtype=torch.uint8,
                device="cpu",
                pin_memory=True,
            )

        second_planner_workspace: torch.Tensor | None = None
        if reusable_planner_workspace is None:
            first_planner_workspace = _new_planner_workspace()
        else:
            if reusable_planner_workspace.device.type != "cpu":
                raise ValueError("reusable planner workspace must be on CPU.")
            if not reusable_planner_workspace.is_contiguous():
                raise ValueError("reusable planner workspace must be contiguous.")
            if not reusable_planner_workspace.is_pinned():
                raise ValueError("reusable planner workspace must be pinned.")
            reusable_capacity_bytes = (
                reusable_planner_workspace.numel()
                * reusable_planner_workspace.element_size()
            )
            if reusable_capacity_bytes < staged_int_workspace_bytes:
                raise ValueError(
                    "reusable planner workspace is smaller than the staged prefix."
                )
            reusable_planner_bytes = reusable_planner_workspace.view(torch.uint8)
            first_planner_workspace = reusable_planner_bytes[
                :staged_int_workspace_bytes
            ]
            if reusable_capacity_bytes >= 2 * staged_int_workspace_bytes:
                second_planner_workspace = reusable_planner_bytes[
                    staged_int_workspace_bytes : 2 * staged_int_workspace_bytes
                ]

        def _new_slot(
            planner_workspace: torch.Tensor | None = None,
        ) -> _MLACudaGraphPlanUpdateSlot:
            return _MLACudaGraphPlanUpdateSlot(
                planner_workspace=(
                    _new_planner_workspace()
                    if planner_workspace is None
                    else planner_workspace
                ),
                qo_indptr=torch.empty(
                    tuple(qo_indptr.shape),
                    dtype=qo_indptr.dtype,
                    device="cpu",
                    pin_memory=True,
                ),
                kv_indptr=torch.empty(
                    tuple(kv_indptr.shape),
                    dtype=kv_indptr.dtype,
                    device="cpu",
                    pin_memory=True,
                ),
                kv_len_arr=torch.empty(
                    tuple(kv_len_arr.shape),
                    dtype=kv_len_arr.dtype,
                    device="cpu",
                    pin_memory=True,
                ),
                completion_event=torch.cuda.Event(),
                status="idle",
            )

        # The generated planner's legacy pinned workspace is backend-owned and
        # idle after the initial plan. Its first two staged-prefix regions can
        # back both bounded in-flight slots. A smaller caller-owned workspace
        # still reuses slot zero and allocates only the second planner region.
        slots = (
            _new_slot(first_planner_workspace),
            _new_slot(second_planner_workspace),
        )
        for slot in slots:
            slot.completion_event.record()
        for slot in slots:
            slot.completion_event.synchronize()

    return _MLACudaGraphPlanUpdateState(
        frozen=frozen,
        candidate=candidate,
        slots=slots,
    )


def _acquire_mla_cuda_graph_plan_update_slot(
    state: _MLACudaGraphPlanUpdateState,
) -> _MLACudaGraphPlanUpdateSlot:
    """Acquire the first completed slot without allocating or waiting."""

    for slot in state.slots:
        if slot.status in ("in-use", "poisoned"):
            continue
        if slot.completion_event.query():
            slot.status = "in-use"
            return slot
    if all(slot.status == "poisoned" for slot in state.slots):
        raise RuntimeError(
            "CUDA graph MLA plan-update staging slots are poisoned; call plan() again."
        )
    raise RuntimeError("CUDA graph MLA plan-update staging slots are busy.")


def _record_mla_cuda_graph_plan_update_slot(
    slot: _MLACudaGraphPlanUpdateSlot,
) -> None:
    try:
        slot.completion_event.record()
    except Exception:
        slot.status = "poisoned"
        raise
    slot.status = "pending"


def _stage_mla_cuda_graph_plan_update_copies(
    slot: _MLACudaGraphPlanUpdateSlot,
    candidate_int_workspace: torch.Tensor,
    candidate_qo_indptr: torch.Tensor,
    candidate_kv_indptr: torch.Tensor,
    candidate_kv_len_arr: torch.Tensor,
) -> None:
    """Queue the fixed candidate schedule and controls without allocating."""

    if slot.status != "in-use":
        raise RuntimeError("CUDA graph MLA staging slot must be in-use.")
    submitted_copy = False
    try:
        candidate_int_workspace.copy_(slot.planner_workspace, non_blocking=True)
        submitted_copy = True
        candidate_qo_indptr.copy_(slot.qo_indptr, non_blocking=True)
        candidate_kv_indptr.copy_(slot.kv_indptr, non_blocking=True)
        candidate_kv_len_arr.copy_(slot.kv_len_arr, non_blocking=True)
    except Exception:
        if submitted_copy:
            _record_mla_cuda_graph_plan_update_slot(slot)
        else:
            slot.status = "idle"
        raise
    _record_mla_cuda_graph_plan_update_slot(slot)


def _raise_changed_invariant(name: str) -> None:
    raise RuntimeError(f"CUDA graph MLA plan invariant changed: {name}.")


def _check_current_tensor_identity(
    name: str,
    expected: _MLACudaGraphTensorIdentity,
    current: torch.Tensor,
) -> None:
    if expected.object_id != id(current):
        _raise_changed_invariant(f"{name}_object")
    if expected.data_ptr != current.data_ptr():
        _raise_changed_invariant(f"{name}_pointer")
    if expected.shape != current.shape:
        _raise_changed_invariant(f"{name}_capacity")
    if expected.dtype != current.dtype:
        _raise_changed_invariant(f"{name}_dtype")
    if expected.device != current.device:
        _raise_changed_invariant(f"{name}_device")


def _check_current_mla_cuda_graph_frozen_contract(
    frozen: _MLACudaGraphFrozenContract,
    *,
    backend_type: type[object],
    module: object,
    device: torch.device,
    float_workspace: torch.Tensor,
    int_workspace: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_len_arr: torch.Tensor,
    head_dim_ckv: int,
    page_size: int,
    causal: bool,
    sm_scale: float,
    q_data_type: torch.dtype,
    kv_data_type: torch.dtype,
    use_profiler: bool,
    plan_info: Sequence[int],
    staged_int_workspace_bytes: int,
) -> None:
    """Check the live backend directly without rebuilding its frozen record."""

    if frozen.backend_type is not backend_type:
        _raise_changed_invariant("backend_type")
    if frozen.module is not module:
        _raise_changed_invariant("module")
    if frozen.device != device:
        _raise_changed_invariant("device")
    _check_current_tensor_identity(
        "float_workspace", frozen.float_workspace, float_workspace
    )
    _check_current_tensor_identity("int_workspace", frozen.int_workspace, int_workspace)
    _check_current_tensor_identity("qo_indptr", frozen.qo_indptr, qo_indptr)
    _check_current_tensor_identity("kv_indptr", frozen.kv_indptr, kv_indptr)
    _check_current_tensor_identity("kv_indices", frozen.kv_indices, kv_indices)
    _check_current_tensor_identity("kv_len_arr", frozen.kv_len_arr, kv_len_arr)

    # Only backend state that can affect planning or publication is mirrored
    # here. Other plan/run options are immutable behind the wrapper API and do
    # not need a second, unenforced snapshot in the update state.
    if frozen.head_dim_ckv != head_dim_ckv:
        _raise_changed_invariant("head_dim_ckv")
    if frozen.page_size != page_size:
        _raise_changed_invariant("page_size")
    if frozen.causal != causal:
        _raise_changed_invariant("causal")
    if frozen.sm_scale != sm_scale:
        _raise_changed_invariant("sm_scale")
    if frozen.q_data_type != q_data_type:
        _raise_changed_invariant("q_data_type")
    if frozen.kv_data_type != kv_data_type:
        _raise_changed_invariant("kv_data_type")
    if frozen.use_profiler != use_profiler:
        _raise_changed_invariant("use_profiler")
    if len(frozen.plan_info) != len(plan_info):
        _raise_changed_invariant("plan_info")
    for index, expected in enumerate(frozen.plan_info):
        if expected != plan_info[index]:
            _raise_changed_invariant("plan_info")
    if frozen.staged_int_workspace_bytes != staged_int_workspace_bytes:
        _raise_changed_invariant("staged_int_workspace_bytes")


def _validate_update_tensor(
    name: str,
    value: object,
    *,
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor.")
    if value.ndim != 1:
        raise ValueError(f"{name} must be rank 1, got shape {tuple(value.shape)}.")
    if value.dtype != torch.int32:
        raise ValueError(f"{name} must have dtype torch.int32, got {value.dtype}.")
    if value.device != device:
        location = "CPU" if device.type == "cpu" else f"wrapper device {device}"
        raise ValueError(f"{name} must be on {location}, got {value.device}.")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous.")
    return value


def _host_int_values(tensor: torch.Tensor) -> list[int]:
    # Placement is checked before this helper. ``tolist`` performs no device
    # transfer for the required CPU controls.
    return tensor.tolist()


def _is_nondecreasing(values: list[int]) -> bool:
    return all(left <= right for left, right in zip(values, values[1:], strict=False))


def _absolute_interval(
    tensor: torch.Tensor,
    *,
    elements: int,
) -> tuple[torch.device, int, int]:
    start = tensor.data_ptr()
    return tensor.device, start, start + elements * tensor.element_size()


def _identity_interval(
    identity: _MLACudaGraphTensorIdentity,
) -> tuple[torch.device, int, int]:
    elements = 1
    for extent in identity.shape:
        elements *= extent
    start = identity.data_ptr
    return identity.device, start, start + elements * identity.dtype.itemsize


def _intervals_overlap(
    left: tuple[torch.device, int, int],
    right: tuple[torch.device, int, int],
) -> bool:
    left_device, left_start, left_end = left
    right_device, right_start, right_end = right
    return (
        left_device == right_device
        and left_start < left_end
        and right_start < right_end
        and max(left_start, right_start) < min(left_end, right_end)
    )


def _check_source_overlaps(
    *,
    name: str,
    source: torch.Tensor,
    elements: int,
    frozen: _MLACudaGraphFrozenContract,
) -> None:
    source_interval = _absolute_interval(source, elements=elements)
    for target_name in _TENSOR_IDENTITY_FIELDS:
        target = getattr(frozen, target_name)
        if not _intervals_overlap(source_interval, _identity_interval(target)):
            continue
        raise ValueError(f"CUDA graph {name} source overlaps reserved {target_name}.")


def _resolve_mla_cuda_graph_plan_update(
    *,
    metadata: MLAPlanMetadata,
    frozen: _MLACudaGraphFrozenContract,
) -> _ResolvedMLACudaGraphPlanUpdate:
    """Resolve complete CSR metadata without any device-to-host conversion."""

    if not isinstance(metadata, MLAPlanMetadata):
        raise TypeError("metadata must be an MLAPlanMetadata instance.")
    qo_indptr_value = metadata.qo_indptr
    kv_indptr_value = metadata.kv_indptr
    kv_indices_value = metadata.kv_indices
    kv_len_arr_value = metadata.kv_len_arr
    cum_seq_lens_q = metadata.cum_seq_lens_q
    block_tables = metadata.block_tables
    seq_lens = metadata.seq_lens
    max_q_len = metadata.max_q_len
    if (
        qo_indptr_value is None
        or kv_indptr_value is None
        or kv_indices_value is None
        or kv_len_arr_value is None
        or cum_seq_lens_q is not None
        or block_tables is not None
        or seq_lens is not None
        or max_q_len is not None
    ):
        raise ValueError("CUDA graph plan updates require complete CSR metadata only.")

    qo_indptr = _validate_update_tensor(
        "qo_indptr", qo_indptr_value, device=_CPU_DEVICE
    )
    kv_indptr = _validate_update_tensor(
        "kv_indptr", kv_indptr_value, device=_CPU_DEVICE
    )
    kv_indices = _validate_update_tensor(
        "kv_indices", kv_indices_value, device=frozen.device
    )
    kv_len_arr = _validate_update_tensor(
        "kv_len_arr", kv_len_arr_value, device=_CPU_DEVICE
    )

    expected_indptr = frozen.batch_size + 1
    if (
        qo_indptr.numel() != expected_indptr
        or kv_indptr.numel() != expected_indptr
        or kv_len_arr.numel() != frozen.batch_size
    ):
        raise ValueError(
            "CSR metadata batch dimensions must match the frozen batch size."
        )

    qo_values = _host_int_values(qo_indptr)
    kv_values = _host_int_values(kv_indptr)
    kv_lens = _host_int_values(kv_len_arr)
    if qo_values[0] != 0:
        raise ValueError("qo_indptr must start at zero.")
    if not _is_nondecreasing(qo_values):
        raise ValueError("qo_indptr must be nondecreasing.")
    if qo_values[-1] != frozen.total_qo_rows:
        raise ValueError(
            "qo_indptr terminal value must match the frozen total query rows."
        )
    if kv_values[0] != 0:
        raise ValueError("kv_indptr must start at zero.")
    if not _is_nondecreasing(kv_values):
        raise ValueError("kv_indptr must be nondecreasing.")

    live_kv_indices = kv_values[-1]
    if live_kv_indices > kv_indices.numel():
        raise ValueError("kv_indices source has insufficient capacity.")
    frozen_kv_capacity = frozen.kv_indices.shape[0]
    if live_kv_indices > frozen_kv_capacity:
        raise ValueError("updated kv_indptr exceeds the reserved kv_indices capacity.")
    if any(length < 0 for length in kv_lens):
        raise ValueError("kv_len_arr must be nonnegative.")
    for request_index in range(frozen.batch_size):
        length = kv_lens[request_index]
        expected_pages = (length + frozen.page_size - 1) // frozen.page_size
        actual_pages = kv_values[request_index + 1] - kv_values[request_index]
        if actual_pages != expected_pages:
            raise ValueError(
                "kv_indptr page counts must equal ceil(kv_len_arr / page_size)."
            )

    _check_source_overlaps(
        name="kv_indices",
        source=kv_indices,
        elements=kv_indices.numel(),
        frozen=frozen,
    )

    return _ResolvedMLACudaGraphPlanUpdate(
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        kv_len_arr=kv_len_arr,
        live_kv_indices=live_kv_indices,
    )
