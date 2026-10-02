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

Independent fused FC1 tactic for the SM90 push BF16 backend.
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

__all__ = [
    "Sm90PushBf16FusedFc1",
    "create_sm90_push_bf16_fused_fc1_runner",
    "gen_sm90_push_bf16_fused_fc1_module",
    "sm90_push_bf16_fused_fc1_uri",
]

_SOURCE_DIR = Path(__file__).resolve().parents[1] / "src" / "bf16_fc1_fused"
_SOURCE_TREE_ROOT = Path(__file__).resolve().parents[6]
_SOURCE_NAMES = (
    "bf16_fc1_fused.cuh",
    "bf16_fc1_fused_binding.cu",
)
_DEPENDENCY_NAMES = (
    "csrc/tvm_ffi_utils.h",
    "include/flashinfer/layout.cuh",
)
_ARCHIVED_ENGINE_ENV = "SM90_PUSH_BF16_ENABLE_ARCHIVED"
_ARCHIVED_ENGINE_QUALIFICATION = "bf16_single_gpu_20260817"
_ARCHIVED_REASON = (
    "SM90 push fused WMMA BF16 FC1 is archived: "
    f"{_ARCHIVED_ENGINE_QUALIFICATION} measured 0.127x of CUTLASS; set "
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
    )


def _cuda_flags() -> tuple[str, ...]:
    return tuple(sm90a_nvcc_flags)


def _source_digest(snapshot: _SourceSnapshot | None = None) -> str:
    if snapshot is None:
        snapshot = _capture_source_snapshot()
    digest = hashlib.sha256()
    for name, content in (
        *snapshot.sources,
        *snapshot.dependencies,
        (Path(__file__).name, snapshot.generator),
    ):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    digest.update(json.dumps(_cuda_flags(), separators=(",", ":")).encode())
    return digest.hexdigest()[:20]


def sm90_push_bf16_fused_fc1_uri() -> str:
    """Return the content-addressed module name for the fused FC1 tactic."""
    return f"sm90_push_bf16_fc1_fused_{_source_digest()}"


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
    source_dir = root / "bf16_fc1_fused"
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
) -> JitSpec:
    if not is_cuda_version_at_least("12.0"):
        raise RuntimeError("SM90 push BF16 fused FC1 requires CUDA 12.0 or newer")
    if snapshot is None:
        snapshot = _capture_source_snapshot()
    actual_digest = _source_digest(snapshot)
    if source_digest is None:
        source_digest = actual_digest
    elif source_digest != actual_digest:
        raise ValueError("source_digest does not match source_snapshot")
    uri = f"sm90_push_bf16_fc1_fused_{source_digest}"
    source_dir, snapshot_root = _materialize_source_snapshot(uri, snapshot)
    return gen_jit_spec(
        uri,
        [source_dir / "bf16_fc1_fused_binding.cu"],
        extra_cuda_cflags=list(_cuda_flags()),
        extra_include_paths=[snapshot_root, snapshot_root / "include", source_dir],
    )


def gen_sm90_push_bf16_fused_fc1_module() -> JitSpec:
    """Snapshot the fused FC1 sources and return their SM90 JIT spec."""
    return _make_jit_spec()


@functools.cache
def _load_sm90_push_bf16_fused_fc1_module_cached(
    source_digest: str, snapshot: _SourceSnapshot
):
    return _make_jit_spec(snapshot, source_digest).build_and_load()


def _load_sm90_push_bf16_fused_fc1_module():
    snapshot = _capture_source_snapshot()
    digest = _source_digest(snapshot)
    return _load_sm90_push_bf16_fused_fc1_module_cached(digest, snapshot)


class Sm90PushBf16FusedFc1:
    """BF16 grouped FC1 with FP32 gated activation before BF16 rounding."""

    def __init__(
        self,
        *,
        max_rows: int,
        num_experts: int,
        intermediate_size: int,
        k: int,
        device: Union[torch.device, str],
    ) -> None:
        _check_archived_gate()
        self.max_rows = int(max_rows)
        self.num_experts = int(num_experts)
        self.intermediate_size = int(intermediate_size)
        self.k = int(k)
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("SM90 push BF16 fused FC1 requires a CUDA device")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self._module = _load_sm90_push_bf16_fused_fc1_module()
        self.ffi_runner = self._module.init()
        self.workspace_size = int(
            self.ffi_runner.get_workspace_size(
                self.max_rows,
                self.num_experts,
                self.intermediate_size,
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

    @staticmethod
    def _is_capturing() -> bool:
        is_capturing = getattr(torch.cuda, "is_current_stream_capturing", None)
        return False if is_capturing is None else bool(is_capturing())

    def _before_launch(self) -> tuple[torch.cuda.Stream, int, bool]:
        current_stream = torch.cuda.current_stream(self.device)
        stream = int(current_stream.cuda_stream)
        capturing = self._is_capturing()
        if self._graph_owned:
            raise RuntimeError(
                "SM90 push BF16 fused FC1 runner is reserved for CUDA graph replay"
            )
        _guard_stream_handoff(
            current_stream=current_stream,
            stream=stream,
            active_stream=self._active_stream,
            completion_event=self._completion_event,
            capturing=capturing,
            error_message=(
                "SM90 push BF16 fused FC1 runner cannot overlap calls on "
                "different CUDA streams"
            ),
        )
        return current_stream, stream, capturing

    def _after_launch(
        self, current_stream: torch.cuda.Stream, stream: int, capturing: bool
    ) -> None:
        if capturing:
            self._graph_owned = True
        else:
            if self._completion_event is None:
                self._completion_event = torch.cuda.Event()
            self._completion_event.record(current_stream)
        self._active_stream = stream

    def run(
        self,
        output: torch.Tensor,
        activation: torch.Tensor,
        weights: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        """Run grouped FC1 and round only the gated product to BF16."""
        current_stream, stream, capturing = self._before_launch()
        self.ffi_runner.run(output, activation, weights, offsets)
        self._after_launch(current_stream, stream, capturing)
        return output

    def run_unfused_epilogue(
        self,
        output: torch.Tensor,
        projected: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the BF16 baseline epilogue to a materialized paired projection."""
        current_stream, stream, capturing = self._before_launch()
        self.ffi_runner.run_unfused_epilogue(output, projected, offsets)
        self._after_launch(current_stream, stream, capturing)
        return output

    def kernel_resource_usage(self) -> dict[str, dict[str, int]]:
        """Return runtime resource attributes for the fused and oracle kernels."""
        result: dict[str, dict[str, int]] = {}
        for name, fused in (("unfused_epilogue", 0), ("wmma_fused_fc1", 1)):
            blocks, registers, local_memory, shared_memory = map(
                int, self.ffi_runner.kernel_resource_usage(fused)
            )
            result[name] = {
                "blocks_per_sm": blocks,
                "registers_per_thread": registers,
                "local_memory_bytes_per_thread": local_memory,
                "static_shared_memory_bytes": shared_memory,
            }
        return result

    def tactic_provenance(self) -> dict[str, object]:
        """Describe the independently selected FC1 fusion tactic."""
        return {
            "tactic": "wmma_16x16_fused_fc1",
            "default_enabled": False,
            "archived_engine": True,
            "qualification": _ARCHIVED_ENGINE_QUALIFICATION,
            "accumulator_dtype": "float32",
            "output_dtype": "bfloat16",
            "rounding_boundary": "after_fp32_silu_mul",
            "rounding_mode": "round_to_nearest_even",
            "unfused_oracle": "grouped_bf16_gemm_then_bf16_gated_epilogue",
            "resources": self.kernel_resource_usage(),
        }


def create_sm90_push_bf16_fused_fc1_runner(
    *,
    max_rows: int,
    num_experts: int,
    intermediate_size: int,
    k: int,
    device: Union[torch.device, str],
) -> Sm90PushBf16FusedFc1:
    """Create one independently selectable fused FC1 tactic."""
    return Sm90PushBf16FusedFc1(
        max_rows=max_rows,
        num_experts=num_experts,
        intermediate_size=intermediate_size,
        k=k,
        device=device,
    )
