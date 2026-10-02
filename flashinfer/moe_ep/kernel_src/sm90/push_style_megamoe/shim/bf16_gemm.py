"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Private grouped BF16 GEMM adapter for the SM90 push MegaMoE backend.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Union

import torch

from ......jit import env as jit_env
from ......jit.core import JitSpec, gen_jit_spec, sm90a_nvcc_flags
from ......jit.cpp_ext import is_cuda_version_at_least
from .bf16_tactics import (
    Bf16GemmFamilyTactic,
    Bf16GemmTactic,
    CORE_BF16_GEMM_TACTICS,
    DEFAULT_BF16_GEMM_TACTIC,
    SUPPORTED_BF16_GEMM_FAMILY_TACTICS,
    SUPPORTED_BF16_GEMM_TACTICS,
    bf16_gemm_cuda_flags,
    estimate_sm90_push_bf16_expected_m,
    normalize_bf16_gemm_tactic,
    select_sm90_push_bf16_gemm_tactic,
)

__all__ = [
    "Sm90PushBf16GroupedGemm",
    "Bf16GemmFamilyTactic",
    "Bf16GemmTactic",
    "CORE_BF16_GEMM_TACTICS",
    "DEFAULT_BF16_GEMM_TACTIC",
    "SUPPORTED_BF16_GEMM_FAMILY_TACTICS",
    "SUPPORTED_BF16_GEMM_TACTICS",
    "normalize_bf16_gemm_tactic",
    "select_sm90_push_bf16_gemm_tactic",
    "estimate_sm90_push_bf16_expected_m",
    "create_sm90_push_bf16_gemm_runner",
    "gen_sm90_push_bf16_gemm_module",
    "sm90_push_bf16_gemm_uri",
]

_SOURCE_DIR = Path(__file__).resolve().parents[1] / "src" / "bf16_gemm"
_SOURCE_TREE_ROOT = Path(__file__).resolve().parents[6]
_SOURCE_NAMES = (
    "bf16_grouped_gemm.cuh",
    "bf16_grouped_gemm_binding.cu",
)
_DEPENDENCY_NAMES = (
    "csrc/tvm_ffi_utils.h",
    "include/flashinfer/allocator.h",
    "include/flashinfer/cutlass_utils.cuh",
    "include/flashinfer/exception.h",
    "include/flashinfer/gemm/group_gemm_sm90.cuh",
    "include/flashinfer/layout.cuh",
    "include/flashinfer/utils.cuh",
)


def _canonical_source(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def _source_tree_root() -> Path:
    if jit_env.FLASHINFER_CSRC_DIR.is_dir():
        return jit_env.FLASHINFER_CSRC_DIR.parent
    return _SOURCE_TREE_ROOT


@dataclass(frozen=True)
class _SourceSnapshot:
    sources: tuple[tuple[str, bytes], ...]
    dependencies: tuple[tuple[str, bytes], ...]
    generator: bytes
    tactic_generator: bytes


@dataclass(frozen=True)
class _PreparedSchedule:
    offsets_address: int
    offsets_shape: tuple[int, ...]
    row_capacity: int
    device: torch.device
    stream: int


class _ScheduleState(Enum):
    INVALID = auto()
    PREPARING = auto()
    READY = auto()
    CONSUMED = auto()


@dataclass(frozen=True)
class _ScheduleLease:
    epoch: int
    prepared: _PreparedSchedule


def _guard_stream_handoff(
    *,
    current_stream: torch.cuda.Stream,
    stream: int,
    active_stream: int | None,
    completion_event: torch.cuda.Event | None,
    capturing: bool,
    error_message: str,
) -> None:
    if completion_event is None or active_stream == stream:
        return
    if capturing:
        # Capture callers complete eager warmup before entering the graph;
        # querying or waiting on that uncaptured event is not capture-safe.
        return
    if not completion_event.query():
        raise RuntimeError(error_message)


@dataclass
class _Bf16ScheduleWorkspace:
    tensor: torch.Tensor
    signature: tuple[object, ...]
    epoch: int = 0
    state: _ScheduleState = _ScheduleState.INVALID
    lease: _ScheduleLease | None = None
    active_stream: int | None = None
    completion_event: torch.cuda.Event | None = None
    capture_stream: int | None = None
    graph_owned: bool = False

    def _invalidate(self) -> None:
        self.state = _ScheduleState.INVALID
        self.lease = None

    def begin_prepare(self, prepared: _PreparedSchedule) -> _ScheduleLease:
        if self.state is _ScheduleState.PREPARING:
            raise RuntimeError("BF16 schedule preparation is already in progress")
        if self.epoch == (1 << 63) - 1:
            self._invalidate()
            raise RuntimeError("BF16 schedule epoch is exhausted")
        self.epoch += 1
        lease = _ScheduleLease(self.epoch, prepared)
        self.state = _ScheduleState.PREPARING
        self.lease = lease
        return lease

    def commit_prepare(self, lease: _ScheduleLease) -> None:
        if self.state is not _ScheduleState.PREPARING or self.lease != lease:
            self._invalidate()
            raise RuntimeError("BF16 schedule preparation lease is stale")
        self.state = _ScheduleState.READY

    def abort_prepare(self, lease: _ScheduleLease) -> None:
        if self.state is not _ScheduleState.PREPARING or self.lease != lease:
            self._invalidate()
            raise RuntimeError("BF16 schedule preparation lease is stale")
        self._invalidate()

    def consume_prepared(self, prepared: _PreparedSchedule) -> _ScheduleLease:
        if self.state is not _ScheduleState.READY or self.lease is None:
            raise RuntimeError("prepared BF16 schedule is unavailable")
        if self.lease.prepared != prepared:
            self._invalidate()
            raise RuntimeError(
                "prepared BF16 schedule does not match offsets, row capacity, "
                "device, or stream"
            )
        lease = self.lease
        self.state = _ScheduleState.CONSUMED
        return lease

    def check_stream_access(
        self,
        current_stream: torch.cuda.Stream,
        stream: int,
        *,
        capturing: bool,
        prepare_schedule: bool,
    ) -> None:
        if self.graph_owned:
            raise RuntimeError(
                "BF16 schedule workspace is reserved for CUDA graph replay"
            )
        if self.capture_stream is not None:
            if not capturing or stream != self.capture_stream or prepare_schedule:
                raise RuntimeError("BF16 schedule capture is incomplete")
        _guard_stream_handoff(
            current_stream=current_stream,
            stream=stream,
            active_stream=self.active_stream,
            completion_event=self.completion_event,
            capturing=capturing,
            error_message=(
                "BF16 schedule workspace cannot overlap calls on different CUDA streams"
            ),
        )

    def record_stream_access(
        self,
        current_stream: torch.cuda.Stream,
        stream: int,
        *,
        capturing: bool,
        prepare_schedule: bool,
    ) -> None:
        self.active_stream = stream
        if capturing:
            if prepare_schedule:
                self.capture_stream = stream
            else:
                self.capture_stream = None
                self.graph_owned = True
            return
        if self.completion_event is None:
            self.completion_event = torch.cuda.Event()
        self.completion_event.record(current_stream)


def _capture_source_snapshot() -> _SourceSnapshot:
    root = _source_tree_root()
    return _SourceSnapshot(
        sources=tuple(
            (name, _canonical_source(_SOURCE_DIR / name)) for name in _SOURCE_NAMES
        ),
        dependencies=tuple(
            (name, _canonical_source(root / name)) for name in _DEPENDENCY_NAMES
        ),
        generator=_canonical_source(Path(__file__).resolve()),
        tactic_generator=_canonical_source(
            Path(__file__).resolve().with_name("bf16_tactics.py")
        ),
    )


def _cuda_flags(tactic: Bf16GemmTactic) -> tuple[str, ...]:
    return (
        tuple(sm90a_nvcc_flags)
        + ("-DCUTLASS_ENABLE_GDC_FOR_SM90=1",)
        + bf16_gemm_cuda_flags(tactic)
    )


def _source_digest(
    snapshot: _SourceSnapshot | None = None,
    tactic: Bf16GemmTactic = DEFAULT_BF16_GEMM_TACTIC,
) -> str:
    tactic = normalize_bf16_gemm_tactic(tactic)
    if snapshot is None:
        snapshot = _capture_source_snapshot()
    digest = hashlib.sha256()
    for name, content in (
        *snapshot.sources,
        *snapshot.dependencies,
        (Path(__file__).name, snapshot.generator),
        ("bf16_tactics.py", snapshot.tactic_generator),
    ):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    digest.update(
        json.dumps(tactic.as_dict(), sort_keys=True, separators=(",", ":")).encode()
    )
    digest.update(json.dumps(_cuda_flags(tactic), separators=(",", ":")).encode())
    return digest.hexdigest()[:20]


def sm90_push_bf16_gemm_uri(
    tactic: Bf16GemmTactic | str = DEFAULT_BF16_GEMM_TACTIC,
) -> str:
    """Return the content-addressed module name for the BF16 GEMM sources."""
    tactic = normalize_bf16_gemm_tactic(tactic)
    return f"sm90_push_bf16_grouped_gemm_{tactic.tag}_{_source_digest(tactic=tactic)}"


def _snapshot_matches(path: Path, content: bytes) -> bool:
    try:
        return path.read_bytes() == content
    except (FileNotFoundError, IsADirectoryError, PermissionError):
        return False


def _write_snapshot_atomic(path: Path, content: bytes) -> None:
    if _snapshot_matches(path, content):
        return
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        temporary = Path(name)
        with os.fdopen(fd, "wb") as output:
            output.write(content)
        os.replace(temporary, path)
        temporary = None
    except PermissionError:
        if not _snapshot_matches(path, content):
            raise
    finally:
        if temporary is not None:
            with contextlib.suppress(FileNotFoundError):
                temporary.unlink()


def _materialize_source_snapshot(
    uri: str, snapshot: _SourceSnapshot
) -> tuple[Path, Path]:
    root = jit_env.FLASHINFER_GEN_SRC_DIR / uri
    source_dir = root / "bf16_gemm"
    for name, content in snapshot.sources:
        _write_snapshot_atomic(source_dir / name, content)
    for name, content in snapshot.dependencies:
        if name == "csrc/tvm_ffi_utils.h":
            _write_snapshot_atomic(root / "tvm_ffi_utils.h", content)
        else:
            _write_snapshot_atomic(root / name, content)
    return source_dir, root


def _make_jit_spec(
    snapshot: _SourceSnapshot | None = None,
    source_digest: str | None = None,
    tactic: Bf16GemmTactic = DEFAULT_BF16_GEMM_TACTIC,
) -> JitSpec:
    if not is_cuda_version_at_least("12.0"):
        raise RuntimeError("SM90 push BF16 grouped GEMM requires CUDA 12.0 or newer")
    if snapshot is None:
        snapshot = _capture_source_snapshot()
    tactic = normalize_bf16_gemm_tactic(tactic)
    actual_digest = _source_digest(snapshot, tactic)
    if source_digest is None:
        source_digest = actual_digest
    elif source_digest != actual_digest:
        raise ValueError("source_digest does not match source_snapshot")
    uri = f"sm90_push_bf16_grouped_gemm_{tactic.tag}_{source_digest}"
    source_dir, snapshot_root = _materialize_source_snapshot(uri, snapshot)
    return gen_jit_spec(
        uri,
        [source_dir / "bf16_grouped_gemm_binding.cu"],
        extra_cuda_cflags=list(_cuda_flags(tactic)),
        extra_include_paths=[snapshot_root, snapshot_root / "include", source_dir],
    )


def gen_sm90_push_bf16_gemm_module(
    tactic: Bf16GemmTactic | str = DEFAULT_BF16_GEMM_TACTIC,
) -> JitSpec:
    """Snapshot the BF16 grouped GEMM sources and return their SM90 JIT spec."""
    return _make_jit_spec(tactic=normalize_bf16_gemm_tactic(tactic))


@functools.cache
def _load_sm90_push_bf16_gemm_module_cached(
    tactic: Bf16GemmTactic, source_digest: str, snapshot: _SourceSnapshot
):
    return _make_jit_spec(snapshot, source_digest, tactic).build_and_load()


def _load_sm90_push_bf16_gemm_module(tactic: Bf16GemmTactic):
    snapshot = _capture_source_snapshot()
    digest = _source_digest(snapshot, tactic)
    return _load_sm90_push_bf16_gemm_module_cached(tactic, digest, snapshot)


class Sm90PushBf16GroupedGemm:
    """Stateful BF16 grouped GEMM runner with preallocated device workspace.

    A captured instance is reserved for replay by that graph. Eager execution
    after capture requires a separate runner and workspace.
    """

    def __init__(
        self,
        *,
        max_rows: int,
        num_experts: int,
        n: int,
        k: int,
        device: Union[torch.device, str],
        shared_schedule_workspace: Optional[_Bf16ScheduleWorkspace] = None,
        tactic: Bf16GemmTactic | str = "production",
        expected_m: float | None = None,
        sm_count: int | None = None,
    ) -> None:
        self.max_rows = int(max_rows)
        self.num_experts = int(num_experts)
        self.n = int(n)
        self.k = int(k)
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("SM90 push BF16 grouped GEMM requires a CUDA device")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if sm_count is None:
            sm_count = int(
                torch.cuda.get_device_properties(self.device).multi_processor_count
            )
        self.sm_count = int(sm_count)
        if self.sm_count <= 0:
            raise ValueError(f"sm_count must be positive, got {self.sm_count}")
        if expected_m is None:
            expected_m = self.max_rows / self.num_experts
        self.expected_m = max(float(expected_m), 0.0)
        tactic_name = tactic.lower() if isinstance(tactic, str) else None
        if tactic_name == "production":
            self.tactic = DEFAULT_BF16_GEMM_TACTIC
            self.selector_kind = "production_default"
            self.selection_reason = "production BF16 GEMM tactic"
        elif tactic_name == "auto":
            self.tactic, self.selection_reason = select_sm90_push_bf16_gemm_tactic(
                expected_m=self.expected_m,
                n=self.n,
                k=self.k,
                sm_count=self.sm_count,
            )
            self.selector_kind = "shape_load_sm_count_v2"
        else:
            self.tactic = normalize_bf16_gemm_tactic(tactic)
            self.selector_kind = "forced"
            self.selection_reason = "forced internal BF16 GEMM tactic"
        self.module_uri = sm90_push_bf16_gemm_uri(self.tactic)
        self._module = _load_sm90_push_bf16_gemm_module(self.tactic)
        self.ffi_runner = self._module.init()
        self.workspace_size = int(
            self.ffi_runner.get_workspace_size(
                self.max_rows,
                self.num_experts,
                self.n,
                self.k,
                self.device.index,
            )
        )
        self.schedule_workspace_size = int(
            self.ffi_runner.get_schedule_workspace_size()
        )
        self.workspace = torch.empty(
            (max(self.workspace_size, 1),), dtype=torch.uint8, device=self.device
        )
        schedule_signature = (self.max_rows, self.num_experts, self.device)
        if shared_schedule_workspace is None:
            schedule_tensor = torch.empty(
                (max(self.schedule_workspace_size, 1),),
                dtype=torch.uint8,
                device=self.device,
            )
            self.schedule_workspace = _Bf16ScheduleWorkspace(
                tensor=schedule_tensor,
                signature=schedule_signature,
            )
        else:
            if shared_schedule_workspace.signature != schedule_signature:
                raise ValueError(
                    "shared BF16 schedule workspace has an incompatible shape envelope"
                )
            if shared_schedule_workspace.tensor.numel() < self.schedule_workspace_size:
                raise ValueError("shared BF16 schedule workspace is too small")
            self.schedule_workspace = shared_schedule_workspace
        self.ffi_runner.configure_workspace(
            self.workspace, self.schedule_workspace.tensor
        )
        self._active_stream: int | None = None
        self._completion_event: torch.cuda.Event | None = None
        self._graph_owned = False

    @staticmethod
    def _is_capturing() -> bool:
        is_capturing = getattr(torch.cuda, "is_current_stream_capturing", None)
        return False if is_capturing is None else bool(is_capturing())

    def run(
        self,
        output: torch.Tensor,
        activation: torch.Tensor,
        weights: torch.Tensor,
        offsets: torch.Tensor,
        *,
        prepare_schedule: bool = True,
    ) -> torch.Tensor:
        """Execute one BF16 GEMM using a newly prepared or shared device schedule."""
        current_stream = torch.cuda.current_stream(self.device)
        stream = int(current_stream.cuda_stream)
        capturing = self._is_capturing()
        if self._graph_owned:
            raise RuntimeError(
                "SM90 push BF16 grouped GEMM runner is reserved for CUDA graph replay"
            )
        _guard_stream_handoff(
            current_stream=current_stream,
            stream=stream,
            active_stream=self._active_stream,
            completion_event=self._completion_event,
            capturing=capturing,
            error_message=(
                "SM90 push BF16 grouped GEMM runner cannot overlap calls on "
                "different CUDA streams"
            ),
        )
        self.schedule_workspace.check_stream_access(
            current_stream,
            stream,
            capturing=capturing,
            prepare_schedule=prepare_schedule,
        )
        prepared = _PreparedSchedule(
            offsets_address=int(offsets.data_ptr()),
            offsets_shape=tuple(offsets.shape),
            row_capacity=int(activation.shape[0]),
            device=activation.device,
            stream=stream,
        )
        if prepare_schedule:
            lease = self.schedule_workspace.begin_prepare(prepared)
        else:
            lease = self.schedule_workspace.consume_prepared(prepared)
        try:
            if prepare_schedule:
                self.ffi_runner.grouped_run(
                    output, activation, weights, offsets, lease.epoch
                )
            else:
                self.ffi_runner.grouped_run_prepared(
                    output, activation, weights, offsets, lease.epoch
                )
            self.schedule_workspace.record_stream_access(
                current_stream,
                stream,
                capturing=capturing,
                prepare_schedule=prepare_schedule,
            )
            if capturing:
                self._graph_owned = True
            else:
                if self._completion_event is None:
                    self._completion_event = torch.cuda.Event()
                self._completion_event.record(current_stream)
            self._active_stream = stream
            if prepare_schedule:
                self.schedule_workspace.commit_prepare(lease)
        except BaseException:
            if prepare_schedule:
                if (
                    self.schedule_workspace.state is _ScheduleState.PREPARING
                    and self.schedule_workspace.lease == lease
                ):
                    self.schedule_workspace.abort_prepare(lease)
                else:
                    self.schedule_workspace._invalidate()
            raise
        return output

    def kernel_resource_usage(self) -> dict[str, dict[str, object]]:
        """Return runtime resource attributes for each compiled M-tile family."""
        result: dict[str, dict[str, object]] = {}
        for family in self.tactic.families:
            resources = tuple(
                map(int, self.ffi_runner.kernel_resource_usage(family.block_m))
            )
            if len(resources) < 4:
                raise RuntimeError("BF16 GEMM resource metadata is incomplete")
            blocks, registers, local_memory, dynamic_smem = resources[:4]
            result[family.tag] = {
                "block_m": family.block_m,
                "block_n": family.block_n,
                "block_k": family.block_k,
                "stages": family.stages,
                "cluster_m": family.cluster_m,
                "kernel_schedule": family.schedule,
                "blocks_per_sm": blocks,
                "registers_per_thread": registers,
                "local_memory_bytes_per_thread": local_memory,
                "dynamic_smem_bytes": dynamic_smem,
            }
        return result

    def tactic_provenance(self) -> dict[str, object]:
        """Describe the internal row-family selector and compiled resources."""
        return {
            "implementation": "cutlass_prepared",
            "selector": self.selector_kind,
            "selected_tactic": self.tactic.as_dict(),
            "selection_reason": self.selection_reason,
            "module_uri": self.module_uri,
            "expected_m": self.expected_m,
            "sm_count": self.sm_count,
            "family_launches": len(self.tactic.families),
            "zero_m_descriptors_supported": True,
            "families": self.kernel_resource_usage(),
        }


def create_sm90_push_bf16_gemm_runner(
    *,
    max_rows: int,
    num_experts: int,
    n: int,
    k: int,
    device: Union[torch.device, str],
    shared_schedule_workspace: Optional[_Bf16ScheduleWorkspace] = None,
    tactic: Bf16GemmTactic | str = "production",
    expected_m: float | None = None,
    sm_count: int | None = None,
) -> Sm90PushBf16GroupedGemm:
    """Create a BF16 grouped GEMM runner bound to one shape envelope and device."""
    return Sm90PushBf16GroupedGemm(
        max_rows=max_rows,
        num_experts=num_experts,
        n=n,
        k=k,
        device=device,
        shared_schedule_workspace=shared_schedule_workspace,
        tactic=tactic,
        expected_m=expected_m,
        sm_count=sm_count,
    )
