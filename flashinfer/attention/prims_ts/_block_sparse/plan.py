# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Atomic wrapper plan state and CUDA-stream lifetime management."""

from collections.abc import Callable
from dataclasses import dataclass
import functools
import _thread
from typing import Concatenate, Literal, ParamSpec, Protocol, TypeVar, cast

import torch

from .common import _validate_sparse_block_size


_P = ParamSpec("_P")
_R = TypeVar("_R")
_BlockSparseCompileKey = tuple[
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    str,
    str,
    str,
    Literal["dense", "causal"],
    bool,
    bool,
    bool,
    bool,
]


class _PlanLockOwner(Protocol):
    """Structural type required by the plan-serialization decorator."""

    _plan_lock: _thread.LockType


_S = TypeVar("_S", bound=_PlanLockOwner)


def _serialize_plan(
    method: Callable[Concatenate[_S, _P], _R],
) -> Callable[Concatenate[_S, _P], _R]:
    """Serialize plan calls without making run wait for a replan."""

    @functools.wraps(method)
    def serialized(self: _S, /, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        with self._plan_lock:
            return method(self, *args, **kwargs)

    return cast(Callable[Concatenate[_S, _P], _R], serialized)


@dataclass(frozen=True)
class _BlockSparseExecutionGeometry:
    """Kernel tiles derived from, but distinct from, semantic BSR blocks.

    One semantic Q block expands into Q64 or Q128 execution tiles. Selected
    semantic KV blocks expand into KV64 fragments, which are then paired into
    fixed KV128 execution routes.
    """

    q_tile_size: int
    kv_tile_size: int


@dataclass(frozen=True)
class _BlockSparseLaunchSpec:
    """Resolved kernel configuration and compile traits for one sparse plan."""

    config: object
    policy: tuple[tuple[str, object], ...]
    compile_key: _BlockSparseCompileKey


def _validate_state_integer(
    value: object,
    name: str,
    *,
    allow_zero: bool = False,
) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a Python integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")


def _validate_state_tensor(
    tensor: object,
    name: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device != device:
        raise ValueError(f"{name} must be on planned device {device}")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")


def _resolve_execution_geometry(
    q_block_size: int,
    kv_block_size: int,
    *,
    q_tile_size: int | None = None,
) -> _BlockSparseExecutionGeometry:
    """Resolve Q64/Q128 and KV128 geometry for the raw-BSR kernel."""

    q_block_size = _validate_sparse_block_size(q_block_size, "q_block_size")
    kv_block_size = _validate_sparse_block_size(kv_block_size, "kv_block_size")
    canonical_q_tile_size = 128 if q_block_size % 128 == 0 else 64
    if q_tile_size is None:
        resolved_q_tile_size = canonical_q_tile_size
    else:
        if isinstance(q_tile_size, bool) or not isinstance(q_tile_size, int):
            raise TypeError("q_tile_size must be a Python integer")
        if q_tile_size not in (64, 128):
            raise ValueError("q_tile_size must be 64 or 128")
        resolved_q_tile_size = q_tile_size
        if resolved_q_tile_size != canonical_q_tile_size:
            raise ValueError(
                "q_tile_size must equal the canonical q_tile_size selected from "
                "q_block_size"
            )

    kv_tile_size = 128
    return _BlockSparseExecutionGeometry(
        q_tile_size=resolved_q_tile_size,
        kv_tile_size=kv_tile_size,
    )


@dataclass(frozen=True)
class _BlockSparsePlanState:
    """One complete revision published by a block-sparse wrapper.

    Every revision executes the raw-BSR kernel, including a pattern whose
    rows select every KV block. The state retains caller BSR tensors because
    run consumes them directly; no route-remapping payload exists in the
    public lifecycle.

    Geometry, dtypes, metadata tensor identities, compiled launch resources,
    and the readiness event are published together. ``run()`` therefore sees
    either the complete old revision or the complete new one, never a mix.

    BSR tensors and an effective caller mask are borrowed and must remain
    immutable until their queued work completes. The dummy mask and event are
    state-owned. Config and policy are treated as immutable after publication;
    the cached compiled adapter is shared read-only. ``record_stream()``
    extends allocator lifetime only—it cannot prevent the caller from
    modifying borrowed metadata in place.
    """

    # API geometry and dtype contract.
    revision: int
    device: torch.device
    device_index: int
    batch_size: int
    seq_len_q: int
    seq_len_kv: int
    num_heads: int
    head_dim: int
    q_block_size: int
    kv_block_size: int
    geometry: _BlockSparseExecutionGeometry
    q_dtype: torch.dtype
    kv_dtype: torch.dtype
    output_dtype: torch.dtype
    mask_type: Literal["dense", "causal"]

    # Caller-owned canonical BSR read directly by every run.
    block_indptr: torch.Tensor
    block_indices: torch.Tensor
    kv_valid_bits: torch.Tensor | None

    # ``runtime_kv_valid_bits`` is either the caller mask or a plan-owned
    # shape-correct dummy required by the compiled ABI.
    runtime_kv_valid_bits: torch.Tensor
    max_row_nnz: int
    config: object
    policy: tuple[tuple[str, object], ...]
    compiled: Callable[..., object]

    # All plan-stream work happens-before run after waiting on this event.
    ready_event: torch.cuda.Event
    ready_stream_handle: int

    def __post_init__(self) -> None:
        _validate_state_integer(self.revision, "revision", allow_zero=True)
        if not isinstance(self.device, torch.device):
            raise TypeError("device must be a torch.device")
        if self.device.type != "cuda":
            raise ValueError("device must identify a CUDA device")
        _validate_state_integer(self.device_index, "device_index", allow_zero=True)
        if self.device.index != self.device_index:
            raise ValueError("device_index must match device.index")
        for name in (
            "batch_size",
            "seq_len_q",
            "seq_len_kv",
            "num_heads",
            "head_dim",
            "q_block_size",
            "kv_block_size",
        ):
            _validate_state_integer(getattr(self, name), name)
        if not isinstance(self.geometry, _BlockSparseExecutionGeometry):
            raise TypeError("geometry must be a _BlockSparseExecutionGeometry")
        if self.geometry != _resolve_execution_geometry(
            self.q_block_size,
            self.kv_block_size,
            q_tile_size=self.geometry.q_tile_size,
        ):
            raise ValueError("geometry must match the semantic block sizes")
        for name in ("q_dtype", "kv_dtype", "output_dtype"):
            if not isinstance(getattr(self, name), torch.dtype):
                raise TypeError(f"{name} must be a torch.dtype")
        if self.mask_type not in ("dense", "causal"):
            raise ValueError("mask_type must be 'dense' or 'causal'")
        _validate_state_tensor(
            self.block_indptr,
            "block_indptr",
            device=self.device,
            dtype=torch.int32,
        )
        expected_indptr_shape = (
            self.batch_size,
            self.num_heads,
            (self.seq_len_q + self.q_block_size - 1) // self.q_block_size + 1,
        )
        if tuple(self.block_indptr.shape) != expected_indptr_shape:
            raise ValueError(f"block_indptr must have shape {expected_indptr_shape}")
        _validate_state_tensor(
            self.block_indices,
            "block_indices",
            device=self.device,
            dtype=torch.int32,
        )
        if self.block_indices.ndim != 1:
            raise ValueError("block_indices must be rank 1")
        expected_valid_shape = (
            self.batch_size,
            (self.seq_len_kv + 31) // 32,
        )
        if self.kv_valid_bits is not None:
            _validate_state_tensor(
                self.kv_valid_bits,
                "kv_valid_bits",
                device=self.device,
                dtype=torch.uint32,
            )
            if tuple(self.kv_valid_bits.shape) != expected_valid_shape:
                raise ValueError(
                    f"kv_valid_bits must have shape {expected_valid_shape}"
                )
        _validate_state_tensor(
            self.runtime_kv_valid_bits,
            "runtime_kv_valid_bits",
            device=self.device,
            dtype=torch.uint32,
        )
        if tuple(self.runtime_kv_valid_bits.shape) != expected_valid_shape:
            raise ValueError(
                f"runtime_kv_valid_bits must have shape {expected_valid_shape}"
            )
        _validate_state_integer(self.max_row_nnz, "max_row_nnz", allow_zero=True)
        if not isinstance(self.policy, tuple) or any(
            not isinstance(entry, tuple)
            or len(entry) != 2
            or not isinstance(entry[0], str)
            for entry in self.policy
        ):
            raise TypeError("policy must be a tuple of (str, value) pairs")
        if not isinstance(self.ready_event, torch.cuda.Event):
            raise TypeError("ready_event must be a torch.cuda.Event")
        _validate_state_integer(
            self.ready_stream_handle,
            "ready_stream_handle",
            allow_zero=True,
        )
        if self.config is None or self.compiled is None:
            raise ValueError("raw-BSR plan requires a config and compiled launch")
        if not callable(self.compiled):
            raise TypeError("compiled must be callable")

        execution_path = dict(self.policy).get("execution_path")
        if execution_path != "raw_bsr_decode":
            raise ValueError("block-sparse plan must use raw_bsr_decode")
        use_token_mask = dict(self.policy).get("use_kv_valid_bits")
        if not isinstance(use_token_mask, bool):
            raise TypeError("raw sparse policy requires bool use_kv_valid_bits")
        if use_token_mask:
            if (
                self.kv_valid_bits is None
                or self.runtime_kv_valid_bits is not self.kv_valid_bits
            ):
                raise ValueError(
                    "masked raw sparse plan must retain the caller token mask"
                )
        elif self.kv_valid_bits is not None:
            raise ValueError(
                "unmasked raw specialization must not retain the caller token mask"
            )


def _allocate_dummy_kv_valid_bits(
    *,
    batch_size: int,
    seq_len_kv: int,
    device: torch.device,
) -> torch.Tensor:
    """Allocate the shape-correct placeholder required by the raw kernel ABI."""

    return torch.zeros(
        (batch_size, (seq_len_kv + 31) // 32),
        dtype=torch.uint32,
        device=device,
    )


def _record_block_sparse_plan_ready_event(
    stream: torch.cuda.Stream,
) -> torch.cuda.Event:
    """Record the event that closes every plan-owned GPU operation."""

    # External events remain legal wait dependencies inside CUDA Graph capture.
    event = torch.cuda.Event(external=True)
    event.record(stream)
    return event


def _record_tensor_once(
    tensor: torch.Tensor,
    stream: torch.cuda.Stream,
    recorded: set[int],
) -> None:
    """Extend allocator lifetime once per tensor data pointer on the run stream."""

    pointer = tensor.data_ptr()
    if pointer in recorded:
        return
    tensor.record_stream(stream)
    recorded.add(pointer)


def _wait_and_record_block_sparse_plan(
    state: _BlockSparsePlanState,
    stream: torch.cuda.Stream,
) -> None:
    """Acquire one state on ``stream`` and retain all launch storage."""

    if stream.device != state.device:
        raise ValueError("run stream must share the planned CUDA device")
    if stream.cuda_stream != state.ready_stream_handle:
        stream.wait_event(state.ready_event)

    recorded: set[int] = set()

    # Raw BSR is a true run-time input. Extending all three launch tensors is
    # required for eager cross-stream use and captured revisions.
    _record_tensor_once(state.block_indptr, stream, recorded)
    _record_tensor_once(state.block_indices, stream, recorded)
    _record_tensor_once(state.runtime_kv_valid_bits, stream, recorded)


__all__ = [
    "_BlockSparseCompileKey",
    "_BlockSparsePlanState",
    "_BlockSparseExecutionGeometry",
    "_BlockSparseLaunchSpec",
    "_allocate_dummy_kv_valid_bits",
    "_record_block_sparse_plan_ready_event",
    "_resolve_execution_geometry",
    "_serialize_plan",
    "_wait_and_record_block_sparse_plan",
]
