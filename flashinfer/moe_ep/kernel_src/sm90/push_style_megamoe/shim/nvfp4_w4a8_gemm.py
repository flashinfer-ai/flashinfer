import functools
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, replace
from enum import Enum, auto
from pathlib import Path
from typing import Any, Optional

import torch

from ......jit import env as jit_env
from ......jit.core import JitSpec, gen_jit_spec, sm90a_nvcc_flags
from ......jit.cpp_ext import is_cuda_version_at_least


KERNEL_VERSION = 3
SUPPORTED_GROUP_SIZES = (32, 64, 128)
SUPPORTED_RESIDUAL_SCHEMES = ("generic", "pow2")
SUPPORTED_BLOCK_M = (64, 128)
SUPPORTED_BLOCK_N = (64, 128)
# Layout 3 is the legacy cell layout, retained ONLY as the bitwise oracle for
# layout-4 tests (requires allow_legacy_layout=True); not AOT-registered.
SUPPORTED_PAYLOAD_LAYOUTS = (3, 4)
# Optimized kernels can retain a small compiler-managed ABI frame without
# spilling their accumulator state. Larger frames are rejected before launch.
_MAX_LOCAL_MEMORY_BYTES_PER_THREAD = 64
_KNOB_SPECS = (
    ("decode_vector", "dv", "W4A8_DECODE_VECTOR", 1),
    ("generic_decode_lut", "dl", "W4A8_GENERIC_DECODE_LUT", 1),
    ("overlap", "ov", "W4A8_OVERLAP", 1),
)
STATIC_VARIANT_COUNT = (
    len(SUPPORTED_GROUP_SIZES)
    * len(SUPPORTED_RESIDUAL_SCHEMES)
    * len(SUPPORTED_BLOCK_M)
    * len(SUPPORTED_BLOCK_N)
    * 2
)
_INSTANTIATION_UNIT_NAMES = (
    "kernel_inst_m64_n64.cu",
    "kernel_inst_m64_n128.cu",
    "kernel_inst_m128_n64.cu",
    "kernel_inst_m128_n128.cu",
)
_TRANSLATION_UNIT_NAMES = ("binding.cu", *_INSTANTIATION_UNIT_NAMES)
_SOURCE_NAMES = (
    "decode.cuh",
    "scheduler.cuh",
    "kernel.cuh",
    "kernel_launchers.cuh",
    "kernel_instantiation.cuh",
    *_TRANSLATION_UNIT_NAMES,
)
_DEEP_GEMM_DEPENDENCIES = (
    "mma_utils.cuh",
    "utils.cuh",
    "nvrtc_cutlass.cuh",
)


@dataclass(frozen=True)
class _OptimizationKnobs:
    decode_vector: int
    generic_decode_lut: int
    overlap: int
    payload_layout: int
    prefer_n64_main: int
    m64_stages: int
    m128_stages: int
    tma_cache_capacity: int


def _optimization_knobs(
    decode_vector: bool | None = None,
    generic_decode_lut: bool | None = None,
    overlap: bool | None = None,
    payload_layout: int | None = None,
    prefer_n64_main: bool | None = None,
    m64_stages: int | None = None,
    m128_stages: int | None = None,
    tma_cache_capacity: int | None = None,
) -> _OptimizationKnobs:
    explicit = {
        "decode_vector": decode_vector,
        "generic_decode_lut": generic_decode_lut,
        "overlap": overlap,
    }
    values = {
        name: default if explicit[name] is None else int(bool(explicit[name]))
        for name, _tag, _macro, default in _KNOB_SPECS
    }
    knobs = _OptimizationKnobs(
        decode_vector=values["decode_vector"],
        generic_decode_lut=values["generic_decode_lut"],
        overlap=values["overlap"],
        payload_layout=4 if payload_layout is None else int(payload_layout),
        prefer_n64_main=int(bool(prefer_n64_main)),
        m64_stages=3 if m64_stages is None else int(m64_stages),
        m128_stages=4 if m128_stages is None else int(m128_stages),
        tma_cache_capacity=(
            128 if tma_cache_capacity is None else int(tma_cache_capacity)
        ),
    )
    if knobs.generic_decode_lut and not knobs.decode_vector:
        raise ValueError("generic_decode_lut requires decode_vector=True")
    if knobs.m64_stages not in (2, 3):
        raise ValueError("m64_stages must be 2 or 3")
    if knobs.m128_stages not in (3, 4):
        raise ValueError("m128_stages must be 3 or 4")
    if knobs.payload_layout not in SUPPORTED_PAYLOAD_LAYOUTS:
        raise ValueError(
            f"payload_layout must be one of {SUPPORTED_PAYLOAD_LAYOUTS}, "
            f"got {knobs.payload_layout}"
        )
    if not 1 <= knobs.tma_cache_capacity <= 128:
        raise ValueError("tma_cache_capacity must be in [1, 128]")
    return knobs


def _knob_tag(knobs: _OptimizationKnobs) -> str:
    boolean_tag = "_".join(
        f"{tag}{getattr(knobs, name)}" for name, tag, _macro, _default in _KNOB_SPECS
    )
    return f"{boolean_tag}_pv{knobs.payload_layout}"


def _knob_arguments(knobs: _OptimizationKnobs) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        name: bool(getattr(knobs, name)) for name, _tag, _macro, _default in _KNOB_SPECS
    }
    arguments["payload_layout"] = knobs.payload_layout
    return arguments


def _compilation_knobs(knobs: _OptimizationKnobs) -> _OptimizationKnobs:
    return replace(
        knobs,
        prefer_n64_main=0,
        m64_stages=3,
        m128_stages=4,
        tma_cache_capacity=128,
    )


@dataclass(frozen=True)
class _SourceSnapshot:
    sources: tuple[tuple[str, bytes], ...]
    deep_gemm_dependencies: tuple[tuple[str, bytes], ...]
    tvm_ffi_utils: bytes
    layout_cuh: bytes
    generator: bytes


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
    counter_bank: int


@dataclass
class _W4A8ScheduleWorkspace:
    tensor: torch.Tensor
    signature: tuple[object, ...]
    epoch: int = 0
    state: _ScheduleState = _ScheduleState.INVALID
    lease: _ScheduleLease | None = None
    consumed_banks: set[int] | None = None

    def _invalidate(self) -> None:
        self.state = _ScheduleState.INVALID
        self.lease = None
        self.consumed_banks = None

    def begin_prepare(
        self, prepared: _PreparedSchedule, counter_bank: int
    ) -> _ScheduleLease:
        if self.state is _ScheduleState.PREPARING:
            raise RuntimeError("W4A8 schedule preparation is already in progress")
        self.epoch += 1
        lease = _ScheduleLease(self.epoch, prepared, counter_bank)
        self.state = _ScheduleState.PREPARING
        self.lease = lease
        self.consumed_banks = None
        return lease

    def commit_prepare(self, lease: _ScheduleLease) -> None:
        if self.state is not _ScheduleState.PREPARING or self.lease != lease:
            self._invalidate()
            raise RuntimeError("W4A8 schedule preparation lease is stale")
        self.state = _ScheduleState.READY
        self.consumed_banks = {lease.counter_bank}

    def abort_prepare(self, lease: _ScheduleLease) -> None:
        if self.state is not _ScheduleState.PREPARING or self.lease != lease:
            self._invalidate()
            raise RuntimeError("W4A8 schedule preparation lease is stale")
        self._invalidate()

    def consume_prepared(self, prepared: _PreparedSchedule, counter_bank: int) -> None:
        if self.state is not _ScheduleState.READY or self.lease is None:
            raise RuntimeError("prepared W4A8 schedule is unavailable")
        if self.lease.prepared != prepared:
            raise RuntimeError(
                "prepared W4A8 schedule does not match offsets, row capacity, device, or stream"
            )
        if self.consumed_banks is None or counter_bank in self.consumed_banks:
            raise RuntimeError("prepared W4A8 schedule counter bank is unavailable")
        self.consumed_banks.add(counter_bank)
        if len(self.consumed_banks) == 2:
            self.state = _ScheduleState.CONSUMED


def _source_directory() -> Path:
    return Path(__file__).resolve().parents[1] / "src" / "nvfp4_w4a8_gemm"


def _csrc_directory() -> Path:
    if jit_env.FLASHINFER_CSRC_DIR.is_dir():
        return jit_env.FLASHINFER_CSRC_DIR
    return Path(__file__).resolve().parents[6] / "csrc"


def _canonical_source(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def _source_paths() -> tuple[Path, ...]:
    directory = _source_directory()
    return tuple(directory / name for name in _SOURCE_NAMES)


def _capture_source_snapshot() -> _SourceSnapshot:
    directory = _source_directory()
    source_root = _csrc_directory()
    deep_gemm = source_root / "nv_internal" / "tensorrt_llm" / "deep_gemm"
    return _SourceSnapshot(
        sources=tuple(
            (name, _canonical_source(directory / name)) for name in _SOURCE_NAMES
        ),
        deep_gemm_dependencies=tuple(
            (name, _canonical_source(deep_gemm / name))
            for name in _DEEP_GEMM_DEPENDENCIES
        ),
        tvm_ffi_utils=_canonical_source(source_root / "tvm_ffi_utils.h"),
        layout_cuh=_canonical_source(
            source_root.parent / "include" / "flashinfer" / "layout.cuh"
        ),
        generator=_canonical_source(Path(__file__).resolve()),
    )


def _write_snapshot_file(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == content:
        return
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
            temporary.write(content)
            temporary_path = Path(temporary.name)
        try:
            os.replace(temporary_path, path)
        except OSError:
            try:
                destination_matches = path.read_bytes() == content
            except OSError:
                destination_matches = False
            if not destination_matches:
                raise
        else:
            temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _materialize_source_snapshot(
    uri: str, snapshot: _SourceSnapshot
) -> tuple[Path, Path]:
    source_root = jit_env.FLASHINFER_GEN_SRC_DIR / uri
    directory = source_root / "nvfp4_w4a8_gemm"
    for name, content in snapshot.sources:
        _write_snapshot_file(directory / name, content)
    deep_gemm = source_root / "nv_internal" / "tensorrt_llm" / "deep_gemm"
    for name, content in snapshot.deep_gemm_dependencies:
        _write_snapshot_file(deep_gemm / name, content)
    _write_snapshot_file(source_root / "tvm_ffi_utils.h", snapshot.tvm_ffi_utils)
    _write_snapshot_file(source_root / "flashinfer" / "layout.cuh", snapshot.layout_cuh)
    return directory, source_root


def _cuda_flags(knobs: _OptimizationKnobs | None = None) -> tuple[str, ...]:
    if knobs is None:
        knobs = _optimization_knobs()
    return (
        *sm90a_nvcc_flags,
        "--ftz=false",
        "--prec-div=true",
        "--prec-sqrt=true",
        f"-DSM90_W4A8_GEMM_VERSION={KERNEL_VERSION}",
        *(
            f"-D{macro}={getattr(knobs, name)}"
            for name, _tag, macro, _default in _KNOB_SPECS
        ),
        f"-DW4A8_PAYLOAD_V4={int(knobs.payload_layout == 4)}",
    )


def _source_digest(
    snapshot: _SourceSnapshot | None = None,
    *,
    knobs: _OptimizationKnobs | None = None,
) -> str:
    if snapshot is None:
        snapshot = _capture_source_snapshot()
    if knobs is None:
        knobs = _optimization_knobs()
    digest = hashlib.sha256(
        (
            f"v{KERNEL_VERSION}:m64+m128:n64+n128:g32+g64+g128:generic+pow2:"
            f"{_knob_tag(knobs)}"
        ).encode()
    )
    for name, content in (
        *((f"nvfp4_w4a8_gemm/{name}", content) for name, content in snapshot.sources),
        *(
            (f"nv_internal/tensorrt_llm/deep_gemm/{name}", content)
            for name, content in snapshot.deep_gemm_dependencies
        ),
        ("tvm_ffi_utils.h", snapshot.tvm_ffi_utils),
        ("flashinfer/layout.cuh", snapshot.layout_cuh),
        (Path(__file__).name, snapshot.generator),
    ):
        digest.update(name.encode())
        digest.update(content)
    digest.update(json.dumps(_cuda_flags(knobs), separators=(",", ":")).encode())
    return digest.hexdigest()[:16]


def _uri(knobs: _OptimizationKnobs, source_digest: str | None = None) -> str:
    if source_digest is None:
        source_digest = _source_digest(knobs=knobs)
    return (
        f"sm90_push_nvfp4_w4a8_gemm_v{KERNEL_VERSION}_"
        f"{_knob_tag(knobs)}_{source_digest}"
    )


def get_sm90_push_nvfp4_w4a8_gemm_uri(
    *,
    decode_vector: bool | None = None,
    generic_decode_lut: bool | None = None,
    overlap: bool | None = None,
    payload_layout: int | None = None,
    prefer_n64_main: bool | None = None,
    m64_stages: int | None = None,
    m128_stages: int | None = None,
) -> str:
    knobs = _optimization_knobs(
        decode_vector=decode_vector,
        generic_decode_lut=generic_decode_lut,
        overlap=overlap,
        payload_layout=payload_layout,
        prefer_n64_main=prefer_n64_main,
        m64_stages=m64_stages,
        m128_stages=m128_stages,
    )
    return _uri(_compilation_knobs(knobs))


def _make_jit_spec(
    knobs: _OptimizationKnobs,
    source_digest: str | None = None,
    source_snapshot: _SourceSnapshot | None = None,
) -> JitSpec:
    if not is_cuda_version_at_least("12.0"):
        raise RuntimeError("SM90 push NVFP4 W4A8 GEMM requires CUDA 12.0 or newer")
    if source_snapshot is None:
        source_snapshot = _capture_source_snapshot()
    snapshot_digest = _source_digest(source_snapshot, knobs=knobs)
    if source_digest is None:
        source_digest = snapshot_digest
    elif source_digest != snapshot_digest:
        raise ValueError("source_digest does not match source_snapshot")
    uri = _uri(knobs, source_digest)
    directory, source_root = _materialize_source_snapshot(uri, source_snapshot)
    return gen_jit_spec(
        uri,
        [directory / name for name in _TRANSLATION_UNIT_NAMES],
        extra_cuda_cflags=list(_cuda_flags(knobs)),
        extra_include_paths=[source_root, directory],
    )


def gen_sm90_push_nvfp4_w4a8_gemm_module(
    *,
    decode_vector: bool | None = None,
    generic_decode_lut: bool | None = None,
    overlap: bool | None = None,
    payload_layout: int | None = None,
    prefer_n64_main: bool | None = None,
    m64_stages: int | None = None,
    m128_stages: int | None = None,
) -> JitSpec:
    knobs = _optimization_knobs(
        decode_vector=decode_vector,
        generic_decode_lut=generic_decode_lut,
        overlap=overlap,
        payload_layout=payload_layout,
        prefer_n64_main=prefer_n64_main,
        m64_stages=m64_stages,
        m128_stages=m128_stages,
    )
    return _make_jit_spec(_compilation_knobs(knobs))


@functools.cache
def _load_sm90_push_nvfp4_w4a8_gemm_module_cached(
    knobs: _OptimizationKnobs,
    source_digest: str,
    source_snapshot: _SourceSnapshot,
):
    return _make_jit_spec(knobs, source_digest, source_snapshot).build_and_load()


def load_sm90_push_nvfp4_w4a8_gemm_module(
    *,
    decode_vector: bool | None = None,
    generic_decode_lut: bool | None = None,
    overlap: bool | None = None,
    payload_layout: int | None = None,
    prefer_n64_main: bool | None = None,
    m64_stages: int | None = None,
    m128_stages: int | None = None,
):
    knobs = _optimization_knobs(
        decode_vector=decode_vector,
        generic_decode_lut=generic_decode_lut,
        overlap=overlap,
        payload_layout=payload_layout,
        prefer_n64_main=prefer_n64_main,
        m64_stages=m64_stages,
        m128_stages=m128_stages,
    )
    knobs = _compilation_knobs(knobs)
    source_snapshot = _capture_source_snapshot()
    source_digest = _source_digest(source_snapshot, knobs=knobs)
    return _load_sm90_push_nvfp4_w4a8_gemm_module_cached(
        knobs, source_digest, source_snapshot
    )


def _validate_static_configuration(
    group_size: int, residual_scheme: str
) -> tuple[int, str]:
    group_size = int(group_size)
    residual_scheme = str(residual_scheme).lower()
    if group_size not in SUPPORTED_GROUP_SIZES:
        raise ValueError(
            f"group_size must be one of {SUPPORTED_GROUP_SIZES}, got {group_size}"
        )
    if residual_scheme not in SUPPORTED_RESIDUAL_SCHEMES:
        raise ValueError(
            "residual_scheme must be one of "
            f"{SUPPORTED_RESIDUAL_SCHEMES}, got {residual_scheme!r}"
        )
    return group_size, residual_scheme


class Sm90W4A8GroupedGemm:
    """Grouped SM90 push NVFP4 W4A8 runner bound to one layout-matched view."""

    def __init__(
        self,
        max_m: int,
        weight_view,
        *,
        total_experts: Optional[int] = None,
        decode_vector: bool | None = None,
        generic_decode_lut: bool | None = None,
        overlap: bool | None = None,
        payload_layout: int | None = None,
        allow_legacy_layout: bool = False,
        prefer_n64_main: bool | None = None,
        m64_stages: int | None = None,
        m128_stages: int | None = None,
        tma_cache_capacity: int | None = None,
        shared_schedule_workspace: Optional[_W4A8ScheduleWorkspace] = None,
        counter_bank: int = 0,
    ) -> None:
        from .nvfp4_repack import (
            NVFP4SM90WeightViewV3,
            NVFP4SM90WeightViewV4,
        )

        if not isinstance(weight_view, (NVFP4SM90WeightViewV3, NVFP4SM90WeightViewV4)):
            raise TypeError("weight_view must be an NVFP4 SM90 W4A8 view")
        weight_view.verify_checksums()
        self.max_m = int(max_m)
        if self.max_m < 0:
            raise ValueError("max_m must be non-negative")
        self.weight_view = weight_view
        logical_shape = weight_view.manifest.logical_shape
        self.bucket_experts = int(logical_shape[0])
        self.logical_n = int(logical_shape[1])
        self.logical_k = int(logical_shape[2])
        padded_shape = weight_view.manifest.padded_shape
        padded_experts = int(padded_shape[0])
        self.padded_n = int(padded_shape[1])
        self.padded_k = int(padded_shape[2])
        if padded_experts != self.bucket_experts:
            raise ValueError("logical and padded expert counts must match")
        if self.padded_n % 64 != 0 or self.padded_k % 128 != 0:
            raise ValueError("v3 W4A8 storage requires N%64 == 0 and K%128 == 0")
        self.group_size, self.residual_scheme = _validate_static_configuration(
            weight_view.manifest.group_size,
            weight_view.manifest.residual_scheme,
        )
        mapping = tuple(weight_view.manifest.expert_mapping)
        inferred_experts = max(mapping, default=-1) + 1
        self.total_experts = (
            inferred_experts if total_experts is None else int(total_experts)
        )
        if self.total_experts <= 0 or any(
            expert < 0 or expert >= self.total_experts for expert in mapping
        ):
            raise ValueError("expert_mapping is outside total_experts")
        self.counter_bank = int(counter_bank)
        if self.counter_bank not in (0, 1):
            raise ValueError("counter_bank must be 0 or 1")
        if shared_schedule_workspace is None and self.counter_bank != 0:
            raise ValueError("counter_bank 1 requires a shared schedule workspace")
        self.optimization_knobs = _optimization_knobs(
            decode_vector=decode_vector,
            generic_decode_lut=generic_decode_lut,
            overlap=overlap,
            payload_layout=payload_layout,
            prefer_n64_main=prefer_n64_main,
            m64_stages=m64_stages,
            m128_stages=m128_stages,
            tma_cache_capacity=tma_cache_capacity,
        )
        if self.optimization_knobs.payload_layout == 3 and not allow_legacy_layout:
            raise ValueError(
                "payload_layout=3 is a legacy oracle and requires "
                "allow_legacy_layout=True"
            )
        if (
            weight_view.manifest.layout_version
            != self.optimization_knobs.payload_layout
        ):
            raise ValueError(
                "W4A8 weight layout does not match the compiled payload layout"
            )
        device = weight_view.packed_e2m1.device
        if device.type != "cuda":
            raise ValueError("SM90 W4A8 weights must be on CUDA")
        self.expert_mapping = torch.tensor(mapping, dtype=torch.int32, device=device)
        self.schedule_signature = (
            self.max_m,
            self.bucket_experts,
            self.total_experts,
            mapping,
            device,
        )
        if shared_schedule_workspace is not None:
            if not isinstance(shared_schedule_workspace, _W4A8ScheduleWorkspace):
                raise TypeError(
                    "shared_schedule_workspace must be _W4A8ScheduleWorkspace"
                )
            if shared_schedule_workspace.signature != self.schedule_signature:
                raise ValueError(
                    "shared W4A8 schedule requires identical row capacity, experts, "
                    "mapping, and device"
                )
            workspace = shared_schedule_workspace.tensor
            if workspace.device != device:
                raise ValueError(
                    "shared schedule workspace must be on the weight device"
                )
            if workspace.dtype != torch.uint8:
                raise ValueError("shared schedule workspace must use uint8 storage")
            if not workspace.is_contiguous():
                raise ValueError("shared schedule workspace must be contiguous")
        self._compiled_module = load_sm90_push_nvfp4_w4a8_gemm_module(
            **_knob_arguments(self.optimization_knobs)
        )
        if (
            self.optimization_knobs.m64_stages == 3
            and self.optimization_knobs.m128_stages == 4
            and not self.optimization_knobs.prefer_n64_main
            and self.optimization_knobs.tma_cache_capacity == 128
        ):
            self.ffi_runner = self._compiled_module.init()
        else:
            self.ffi_runner = self._compiled_module.init_with_tactics(
                self.optimization_knobs.m64_stages,
                self.optimization_knobs.m128_stages,
                bool(self.optimization_knobs.prefer_n64_main),
                self.optimization_knobs.tma_cache_capacity,
            )
        self.workspace_size = int(
            self.ffi_runner.get_workspace_size(
                self.max_m,
                self.logical_n,
                self.padded_n,
                self.padded_k,
                self.bucket_experts,
                self.total_experts,
                self.group_size,
                self.residual_scheme,
            )
        )
        if shared_schedule_workspace is None:
            workspace = torch.empty(
                (max(self.workspace_size, 1),), dtype=torch.uint8, device=device
            )
            self.schedule_workspace = _W4A8ScheduleWorkspace(
                tensor=workspace,
                signature=self.schedule_signature,
            )
            self.workspace = workspace
            self.ffi_runner.configure_workspace(workspace)
        else:
            if workspace.numel() < self.workspace_size:
                raise ValueError("shared schedule workspace is too small")
            self.schedule_workspace = shared_schedule_workspace
            self.workspace = workspace
            self.ffi_runner.configure_workspace_bank(workspace, self.counter_bank)
        self.resource_status = self._resource_status()
        self.required_resource_variants = self._required_resource_variants()
        for name, entry in self.resource_status.items():
            entry["required"] = name in self.required_resource_variants
        self.usable = all(
            self.resource_status[name]["usable"]
            for name in self.required_resource_variants
        )

    def _allocate_output(
        self, activation: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        return torch.empty(
            (activation.shape[0], self.logical_n),
            dtype=dtype,
            device=activation.device,
        )

    def kernel_resource_usage(self) -> dict[str, dict[str, dict[str, int]]]:
        result: dict[str, dict[str, dict[str, int]]] = {}
        for output_kind, debug_fp32 in (("bf16", False), ("fp32_debug", True)):
            variants: dict[str, dict[str, int]] = {}
            for block_m in SUPPORTED_BLOCK_M:
                for block_n in SUPPORTED_BLOCK_N:
                    blocks, registers, local_memory = map(
                        int,
                        self.ffi_runner.kernel_resource_usage(
                            block_m, block_n, debug_fp32
                        ),
                    )
                    variants[f"m{block_m}_n{block_n}"] = {
                        "blocks_per_sm": blocks,
                        "registers_per_thread": registers,
                        "local_memory_bytes_per_thread": local_memory,
                    }
            result[output_kind] = variants
        return result

    def _resource_status(self) -> dict[str, dict[str, int | str | bool | None]]:
        status: dict[str, dict[str, int | str | bool | None]] = {}
        for block_m in SUPPORTED_BLOCK_M:
            for block_n in SUPPORTED_BLOCK_N:
                (
                    registers,
                    local_bytes,
                    stack_bytes,
                    producer_registers,
                    consumer_register_cap,
                    register_limit,
                    blocks_per_sm,
                ) = map(
                    int,
                    self.ffi_runner.kernel_resources(block_m, block_n, False),
                )
                reason = None
                if local_bytes > _MAX_LOCAL_MEMORY_BYTES_PER_THREAD:
                    reason = (
                        f"local memory is {local_bytes} bytes per thread, exceeding "
                        f"the {_MAX_LOCAL_MEMORY_BYTES_PER_THREAD}-byte limit"
                    )
                # numRegs is the whole-kernel maximum across specialized warp
                # roles, while consumer_registers is a per-warp setmaxnreg target.
                status[f"m{block_m}_n{block_n}"] = {
                    "usable": reason is None,
                    "reason": reason,
                    "num_regs": registers,
                    "local_size_bytes": local_bytes,
                    "stack_bytes": None if stack_bytes < 0 else stack_bytes,
                    "producer_registers": producer_registers,
                    "consumer_register_cap": consumer_register_cap,
                    "consumer_registers": register_limit,
                    "blocks_per_sm": blocks_per_sm,
                }
        return status

    def _required_resource_variants(self) -> tuple[str, ...]:
        block_ms = (64, 128)
        block_ns: tuple[int, ...]
        if self.optimization_knobs.prefer_n64_main:
            block_ns = (64,)
        else:
            block_ns = tuple(
                block_n
                for block_n, enabled in (
                    (128, self.padded_n >= 128),
                    (64, self.padded_n % 128 != 0),
                )
                if enabled
            )
        return tuple(
            f"m{block_m}_n{block_n}" for block_m in block_ms for block_n in block_ns
        )

    def _require_usable_resources(self) -> None:
        if self.usable:
            return
        reasons = ", ".join(
            f"{name}: {self.resource_status[name]['reason']}"
            for name in self.required_resource_variants
            if not self.resource_status[name]["usable"]
        )
        raise RuntimeError(f"SM90 W4A8 kernel resource guard rejected {reasons}")

    def _prepared_schedule(
        self, activation: torch.Tensor, offsets: torch.Tensor
    ) -> _PreparedSchedule:
        stream = torch.cuda.current_stream(activation.device)
        return _PreparedSchedule(
            offsets_address=int(offsets.data_ptr()),
            offsets_shape=tuple(offsets.shape),
            row_capacity=int(activation.shape[0]),
            device=activation.device,
            stream=int(stream.cuda_stream),
        )

    def run(
        self,
        activation: torch.Tensor,
        activation_scales: torch.Tensor,
        offsets: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
        trusted_offsets: bool = False,
        prepare_schedule: bool = True,
    ) -> torch.Tensor:
        self._require_usable_resources()
        if out is None:
            out = self._allocate_output(activation, torch.bfloat16)
        if prepare_schedule:
            if self.counter_bank != 0:
                raise ValueError("schedule preparation requires counter bank 0")
        else:
            if self.counter_bank != 1:
                raise ValueError("prepared schedule execution requires counter bank 1")
            if not trusted_offsets:
                raise ValueError("prepared schedule execution requires trusted offsets")
        view = self.weight_view
        prepared = self._prepared_schedule(activation, offsets)
        run = (
            self.ffi_runner.grouped_run
            if prepare_schedule
            else self.ffi_runner.grouped_run_prepared
        )
        lease = None
        if prepare_schedule:
            lease = self.schedule_workspace.begin_prepare(prepared, self.counter_bank)
        else:
            self.schedule_workspace.consume_prepared(prepared, self.counter_bank)
        try:
            run(
                out,
                activation,
                activation_scales,
                view.packed_e2m1,
                view.promotion_residual,
                view.promotion_group_scale,
                view.global_alpha,
                self.expert_mapping,
                offsets,
                bool(trusted_offsets),
            )
        except BaseException:
            if lease is not None:
                self.schedule_workspace.abort_prepare(lease)
            raise
        if lease is not None:
            self.schedule_workspace.commit_prepare(lease)
        return out

    def debug_decode(self, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Decode one operand image for internal correctness tests."""
        if out is None:
            experts, padded_n, padded_k = self.weight_view.manifest.padded_shape
            out = torch.empty(
                (experts, padded_k // 32, padded_n // 64, 64, 32),
                dtype=torch.uint8,
                device=self.weight_view.packed_e2m1.device,
            )
        self.ffi_runner.debug_decode(
            out,
            self.weight_view.packed_e2m1,
            self.weight_view.promotion_residual,
        )
        return out

    def run_debug_fp32(
        self,
        activation: torch.Tensor,
        activation_scales: torch.Tensor,
        offsets: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
        trusted_offsets: bool = False,
    ) -> torch.Tensor:
        """Run the FP32-output path for internal correctness tests."""
        self._require_usable_resources()
        if out is None:
            out = self._allocate_output(activation, torch.float32)
        if self.counter_bank != 0:
            raise ValueError("FP32 debug execution requires counter bank 0")
        view = self.weight_view
        prepared = self._prepared_schedule(activation, offsets)
        run = self.ffi_runner.debug_run_fp32
        lease = self.schedule_workspace.begin_prepare(prepared, self.counter_bank)
        try:
            run(
                out,
                activation,
                activation_scales,
                view.packed_e2m1,
                view.promotion_residual,
                view.promotion_group_scale,
                view.global_alpha,
                self.expert_mapping,
                offsets,
                bool(trusted_offsets),
            )
        except BaseException:
            self.schedule_workspace.abort_prepare(lease)
            raise
        self.schedule_workspace.commit_prepare(lease)
        return out


def create_sm90_push_nvfp4_w4a8_gemm(
    max_m: int,
    weight_view,
    *,
    total_experts: Optional[int] = None,
    decode_vector: bool | None = None,
    generic_decode_lut: bool | None = None,
    overlap: bool | None = None,
    payload_layout: int | None = None,
    allow_legacy_layout: bool = False,
    prefer_n64_main: bool | None = None,
    m64_stages: int | None = None,
    m128_stages: int | None = None,
    tma_cache_capacity: int | None = None,
    shared_schedule_workspace: Optional[_W4A8ScheduleWorkspace] = None,
    counter_bank: int = 0,
) -> Sm90W4A8GroupedGemm:
    return Sm90W4A8GroupedGemm(
        max_m,
        weight_view,
        total_experts=total_experts,
        decode_vector=decode_vector,
        generic_decode_lut=generic_decode_lut,
        overlap=overlap,
        payload_layout=payload_layout,
        allow_legacy_layout=allow_legacy_layout,
        prefer_n64_main=prefer_n64_main,
        m64_stages=m64_stages,
        m128_stages=m128_stages,
        tma_cache_capacity=tma_cache_capacity,
        shared_schedule_workspace=shared_schedule_workspace,
        counter_bank=counter_bank,
    )
