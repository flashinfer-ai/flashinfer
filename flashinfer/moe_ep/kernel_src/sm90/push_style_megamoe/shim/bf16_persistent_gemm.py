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

Private persistent-offset BF16 GEMM adapter for SM90 push MegaMoE.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import torch

from ......jit import env as jit_env
from ......jit.core import JitSpec, gen_jit_spec, sm90a_nvcc_flags
from ......jit.cpp_ext import is_cuda_version_at_least
from .bf16_gemm import _guard_stream_handoff
from .bf16_tactics import (
    Bf16GemmTactic,
    DEFAULT_BF16_GEMM_TACTIC,
    bf16_gemm_cuda_flags,
    normalize_bf16_gemm_tactic,
    select_sm90_push_bf16_gemm_tactic,
)

__all__ = [
    "Sm90PushBf16PersistentGroupedGemm",
    "create_sm90_push_bf16_persistent_gemm_runner",
    "gen_sm90_push_bf16_persistent_gemm_module",
    "sm90_push_bf16_persistent_gemm_uri",
]

_SOURCE_DIR = Path(__file__).resolve().parents[1] / "src" / "bf16_persistent_gemm"
_SOURCE_TREE_ROOT = Path(__file__).resolve().parents[6]
_SOURCE_NAMES = (
    "bf16_persistent_gemm.cuh",
    "bf16_persistent_gemm_binding.cu",
)
_DEPENDENCY_NAMES = (
    "csrc/tvm_ffi_utils.h",
    "csrc/nv_internal/tensorrt_llm/deep_gemm/mma_utils.cuh",
    "csrc/nv_internal/tensorrt_llm/deep_gemm/tma_utils.cuh",
    "csrc/nv_internal/tensorrt_llm/deep_gemm/utils.cuh",
    "include/flashinfer/attention/hopper.cuh",
    "include/flashinfer/cp_async.cuh",
    "include/flashinfer/layout.cuh",
    "include/flashinfer/mma.cuh",
    "include/flashinfer/permuted_smem.cuh",
)
_ARCHIVED_ENGINE_ENV = "SM90_PUSH_BF16_ENABLE_ARCHIVED"
_ARCHIVED_ENGINE_QUALIFICATION = "bf16_single_gpu_20260817"
_ARCHIVED_REASON = (
    "SM90 push persistent BF16 GEMM is archived: "
    f"{_ARCHIVED_ENGINE_QUALIFICATION} measured 0.606x of CUTLASS; set "
    f"{_ARCHIVED_ENGINE_ENV}=1 to construct it"
)


def _check_archived_gate() -> None:
    if os.environ.get(_ARCHIVED_ENGINE_ENV) == "1":
        return
    raise RuntimeError(_ARCHIVED_REASON)


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


def _normalize_shape(n: int, k: int) -> tuple[int, int]:
    n, k = int(n), int(k)
    if n <= 0 or k <= 0:
        raise ValueError(f"N and K must be positive, got N={n}, K={k}")
    if n % 64 != 0 or k % 64 != 0:
        raise ValueError(f"N and K must be multiples of 64, got N={n}, K={k}")
    return n, k


def _validate_tactic_shape(tactic: Bf16GemmTactic, n: int, k: int) -> tuple[int, int]:
    n, k = _normalize_shape(n, k)
    for family in tactic.families:
        if n % family.block_n != 0:
            raise ValueError(
                f"N={n} must be divisible by {family.tag} BlockN={family.block_n}"
            )
        if k % family.block_k != 0:
            raise ValueError(
                f"K={k} must be divisible by {family.tag} BlockK={family.block_k}"
            )
    return n, k


def _cuda_flags(tactic: Bf16GemmTactic, n: int, k: int) -> tuple[str, ...]:
    tactic = normalize_bf16_gemm_tactic(tactic)
    n, k = _validate_tactic_shape(tactic, n, k)
    return (
        tuple(sm90a_nvcc_flags)
        + ("-DCUTLASS_ENABLE_GDC_FOR_SM90=1",)
        + bf16_gemm_cuda_flags(tactic)
        + (
            f"-DSM90_PUSH_BF16_SHAPE_N={n}",
            f"-DSM90_PUSH_BF16_SHAPE_K={k}",
        )
    )


def _source_digest(
    snapshot: _SourceSnapshot | None = None,
    tactic: Bf16GemmTactic = DEFAULT_BF16_GEMM_TACTIC,
    n: int = 128,
    k: int = 128,
) -> str:
    tactic = normalize_bf16_gemm_tactic(tactic)
    n, k = _validate_tactic_shape(tactic, n, k)
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
    digest.update(json.dumps(_cuda_flags(tactic, n, k), separators=(",", ":")).encode())
    return digest.hexdigest()[:20]


def sm90_push_bf16_persistent_gemm_uri(
    tactic: Bf16GemmTactic | str = DEFAULT_BF16_GEMM_TACTIC,
    *,
    n: int,
    k: int,
) -> str:
    """Return the content-addressed module name for one persistent GEMM shape."""
    tactic = normalize_bf16_gemm_tactic(tactic)
    n, k = _validate_tactic_shape(tactic, n, k)
    digest = _source_digest(tactic=tactic, n=n, k=k)
    return f"sm90_push_bf16_persistent_gemm_n{n}_k{k}_{tactic.tag}_{digest}"


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
    source_dir = root / "bf16_persistent_gemm"
    for name, content in snapshot.sources:
        _write_snapshot_atomic(source_dir / name, content)
    for name, content in snapshot.dependencies:
        if name == "csrc/tvm_ffi_utils.h":
            _write_snapshot_atomic(root / "tvm_ffi_utils.h", content)
        elif name.startswith("csrc/"):
            _write_snapshot_atomic(root / name.removeprefix("csrc/"), content)
        else:
            _write_snapshot_atomic(root / name, content)
    return source_dir, root


def _make_jit_spec(
    snapshot: _SourceSnapshot | None = None,
    source_digest: str | None = None,
    tactic: Bf16GemmTactic = DEFAULT_BF16_GEMM_TACTIC,
    n: int = 128,
    k: int = 128,
) -> JitSpec:
    if not is_cuda_version_at_least("12.0"):
        raise RuntimeError("SM90 push persistent BF16 GEMM requires CUDA 12.0 or newer")
    if snapshot is None:
        snapshot = _capture_source_snapshot()
    tactic = normalize_bf16_gemm_tactic(tactic)
    n, k = _validate_tactic_shape(tactic, n, k)
    actual_digest = _source_digest(snapshot, tactic, n, k)
    if source_digest is None:
        source_digest = actual_digest
    elif source_digest != actual_digest:
        raise ValueError("source_digest does not match source_snapshot")
    uri = f"sm90_push_bf16_persistent_gemm_n{n}_k{k}_{tactic.tag}_{source_digest}"
    source_dir, snapshot_root = _materialize_source_snapshot(uri, snapshot)
    return gen_jit_spec(
        uri,
        [source_dir / "bf16_persistent_gemm_binding.cu"],
        extra_cuda_cflags=list(_cuda_flags(tactic, n, k)),
        extra_include_paths=[snapshot_root, snapshot_root / "include", source_dir],
    )


def gen_sm90_push_bf16_persistent_gemm_module(
    tactic: Bf16GemmTactic | str = "production",
    *,
    n: int,
    k: int,
) -> JitSpec:
    """Snapshot one persistent BF16 GEMM shape and return its SM90 JIT spec."""
    if isinstance(tactic, str) and tactic.lower() == "production":
        tactic = DEFAULT_BF16_GEMM_TACTIC
    return _make_jit_spec(tactic=normalize_bf16_gemm_tactic(tactic), n=n, k=k)


@functools.cache
def _load_sm90_push_bf16_persistent_gemm_module_cached(
    tactic: Bf16GemmTactic,
    n: int,
    k: int,
    source_digest: str,
    snapshot: _SourceSnapshot,
):
    return _make_jit_spec(snapshot, source_digest, tactic, n, k).build_and_load()


def _load_sm90_push_bf16_persistent_gemm_module(tactic: Bf16GemmTactic, n: int, k: int):
    snapshot = _capture_source_snapshot()
    digest = _source_digest(snapshot, tactic, n, k)
    return _load_sm90_push_bf16_persistent_gemm_module_cached(
        tactic, n, k, digest, snapshot
    )


class Sm90PushBf16PersistentGroupedGemm:
    """Shape-specialized BF16 GEMM that schedules directly from expert offsets."""

    def __init__(
        self,
        *,
        max_rows: int,
        num_experts: int,
        n: int,
        k: int,
        device: Union[torch.device, str],
        tactic: Bf16GemmTactic | str = "production",
        expected_m: float | None = None,
        sm_count: int | None = None,
        trusted_offsets: bool = False,
    ) -> None:
        _check_archived_gate()
        self.max_rows = int(max_rows)
        self.num_experts = int(num_experts)
        self.n, self.k = _normalize_shape(n, k)
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("SM90 push persistent BF16 GEMM requires a CUDA device")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if self.max_rows <= 0:
            raise ValueError(f"max_rows must be positive, got {self.max_rows}")
        if self.num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {self.num_experts}")
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
        _validate_tactic_shape(self.tactic, self.n, self.k)
        self.trusted_offsets = bool(trusted_offsets)
        self.module_uri = sm90_push_bf16_persistent_gemm_uri(
            self.tactic, n=self.n, k=self.k
        )
        self._module = _load_sm90_push_bf16_persistent_gemm_module(
            self.tactic, self.n, self.k
        )
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
        self.workspace = torch.empty(
            (max(self.workspace_size, 1),), dtype=torch.uint8, device=self.device
        )
        self.ffi_runner.configure_workspace(self.workspace)
        self._active_stream: int | None = None
        self._completion_event: torch.cuda.Event | None = None
        self._graph_owned = False
        self._map_identity: tuple[int, int, int] | None = None

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
        """Write the active ``offsets[-1]`` prefix and leave capacity tail rows unchanged."""
        del prepare_schedule
        current_stream = torch.cuda.current_stream(self.device)
        stream = int(current_stream.cuda_stream)
        capturing = self._is_capturing()
        map_identity = (activation.data_ptr(), weights.data_ptr(), activation.shape[0])
        if self._graph_owned:
            raise RuntimeError(
                "SM90 push persistent BF16 GEMM runner is reserved for CUDA graph replay"
            )
        if capturing and self._map_identity != map_identity:
            raise RuntimeError(
                "SM90 push persistent BF16 GEMM requires an eager warmup with the "
                "same activation and weight storage before CUDA graph capture"
            )
        _guard_stream_handoff(
            current_stream=current_stream,
            stream=stream,
            active_stream=self._active_stream,
            completion_event=self._completion_event,
            capturing=capturing,
            error_message=(
                "SM90 push persistent BF16 GEMM runner cannot overlap calls on "
                "different CUDA streams"
            ),
        )
        self.ffi_runner.grouped_run(
            output,
            activation,
            weights,
            offsets,
            self.trusted_offsets,
        )
        if capturing:
            self._graph_owned = True
        else:
            self._map_identity = map_identity
            if self._completion_event is None:
                self._completion_event = torch.cuda.Event()
            self._completion_event.record(current_stream)
        self._active_stream = stream
        return output

    def kernel_resource_usage(self) -> dict[str, dict[str, object]]:
        """Return runtime resource attributes for each compiled M-tile family."""
        result: dict[str, dict[str, object]] = {}
        for family in self.tactic.families:
            resources = tuple(
                map(int, self.ffi_runner.kernel_resource_usage(family.block_m))
            )
            if len(resources) != 6:
                raise RuntimeError(
                    "persistent BF16 GEMM resource metadata is incomplete"
                )
            (
                blocks,
                registers,
                local_memory,
                dynamic_smem,
                cluster_m,
                threads,
            ) = resources
            if cluster_m != family.cluster_m:
                raise RuntimeError(
                    "persistent BF16 GEMM resource metadata disagrees with the tactic"
                )
            result[family.tag] = {
                "block_m": family.block_m,
                "block_n": family.block_n,
                "block_k": family.block_k,
                "stages": family.stages,
                "cluster_m": cluster_m,
                "tactic_schedule": family.schedule,
                "wait_policy": (
                    "blocking" if family.schedule == "pingpong" else "poll_nanosleep"
                ),
                "threads_per_block": threads,
                "blocks_per_sm": blocks,
                "registers_per_thread": registers,
                "local_memory_bytes_per_thread": local_memory,
                "dynamic_smem_bytes": dynamic_smem,
            }
        return result

    def tactic_provenance(self) -> dict[str, object]:
        """Describe the persistent-offset implementation and compiled tactic."""
        return {
            "implementation": "persistent_offsets",
            "archived_engine": True,
            "qualification": _ARCHIVED_ENGINE_QUALIFICATION,
            "selector": self.selector_kind,
            "selected_tactic": self.tactic.as_dict(),
            "selection_reason": self.selection_reason,
            "module_uri": self.module_uri,
            "expected_m": self.expected_m,
            "sm_count": self.sm_count,
            "shape_n": self.n,
            "shape_k": self.k,
            "family_launches": len(self.tactic.families),
            "trusted_offsets": self.trusted_offsets,
            "active_rows_source": "offsets[-1]",
            "capacity_tail_written": False,
            "comparison_scope": "full_implementation_matched_tactic",
            "families": self.kernel_resource_usage(),
        }


def create_sm90_push_bf16_persistent_gemm_runner(
    *,
    max_rows: int,
    num_experts: int,
    n: int,
    k: int,
    device: Union[torch.device, str],
    tactic: Bf16GemmTactic | str = "production",
    expected_m: float | None = None,
    sm_count: int | None = None,
    trusted_offsets: bool = False,
) -> Sm90PushBf16PersistentGroupedGemm:
    """Create a persistent BF16 GEMM runner for one shape envelope and device."""
    return Sm90PushBf16PersistentGroupedGemm(
        max_rows=max_rows,
        num_experts=num_experts,
        n=n,
        k=k,
        device=device,
        tactic=tactic,
        expected_m=expected_m,
        sm_count=sm_count,
        trusted_offsets=trusted_offsets,
    )
