"""Runtime owner for the fixed 11-stage SM100 BF16 rank-major MoE-EP DAG."""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from filelock import FileLock


_WORLD_SIZE = 8
_TOKENS_PER_RANK = 128
_RANK_MAJOR_TOKENS = _WORLD_SIZE * _TOKENS_PER_RANK
_HIDDEN_SIZE = 7168
_INTERMEDIATE_SIZE = 2048
_NUM_EXPERTS = 256
_LOCAL_EXPERTS = _NUM_EXPERTS // _WORLD_SIZE
_TOP_K = 8
_FIXED_EXPERT_ROWS = 32768
_MAX_Y_GROUPS = 512
_DESCRIPTOR_BYTES = 128
_SOURCE_NAME = "flashinfer_blackwell_moe_ep_layer_sm100.cu"
_STAGE_NAMES = (
    "input_barrier",
    "dispatch",
    "route_reset",
    "route_count",
    "route_finalize",
    "route_scatter",
    "gemm1_swiglu",
    "gemm2",
    "local_unpermute",
    "partial_barrier",
    "combine",
)
_STAGE_BINDINGS = {
    "input_barrier": (
        "expert_ids",
        "topk_ids",
        "pg_world",
        "pg_rank",
        "pg_flags",
    ),
    "dispatch": (
        "recv_hidden",
        "recv_local_ids",
        "recv_weights",
        "pg_world",
        "pg_rank",
        "pg_flags",
        "hidden_states",
        "hidden_states_peers",
        "topk_ids",
        "topk_ids_peers",
        "topk_weights",
        "topk_weights_peers",
    ),
    "route_reset": ("expert_scatter_offsets", "zero_sentinel"),
    "route_count": (
        "recv_local_ids",
        "expert_scatter_offsets",
        "token_to_permuted",
    ),
    "route_finalize": (
        "expert_scatter_offsets",
        "cta_to_expert",
        "cta_to_mn_limit",
        "expert_padded_row_offsets",
        "num_non_exiting_ctas",
        "total_padded_rows",
        "route_map",
    ),
    "route_scatter": (
        "recv_local_ids",
        "expert_padded_row_offsets",
        "route_map",
        "token_to_permuted",
    ),
    "gemm1_swiglu": (
        "weights",
        "recv_hidden",
        "compact_intermediate",
        "route_map",
        "num_non_exiting_ctas",
        "cta_idx_y_to_batch_idx",
        "cta_idx_y_to_mn_limit",
        "K",
    ),
    "gemm2": (
        "A",
        "B",
        "C",
        "num_non_exiting_ctas",
        "cta_idx_y_to_batch_idx",
        "cta_idx_y_to_mn_limit",
        "K",
    ),
    "local_unpermute": (
        "expert_output",
        "topk_weights",
        "token_to_permuted",
        "final_output",
        "hidden_size",
    ),
    "partial_barrier": ("pg_world", "pg_rank", "pg_flags"),
    "combine": (
        "output",
        "pg_world",
        "pg_rank",
        "pg_flags",
        "local_partials",
        "local_partials_peers",
    ),
}
_STAGE_LAUNCH_CONTRACTS = {
    "input_barrier": (
        (1, 1, 1),
        (32, 1, 1),
        (1, 1, 1),
        0,
        (),
        False,
        False,
        False,
    ),
    "dispatch": (
        (1024, 1, 1),
        (256, 1, 1),
        (1, 1, 1),
        15488,
        (),
        False,
        False,
        False,
    ),
    "route_reset": (
        (1, 1, 1),
        (256, 1, 1),
        (1, 1, 1),
        0,
        (),
        False,
        False,
        False,
    ),
    "route_count": (
        (32, 1, 1),
        (256, 1, 1),
        (1, 1, 1),
        0,
        (),
        False,
        False,
        False,
    ),
    "route_finalize": (
        (1, 1, 1),
        (32, 1, 1),
        (1, 1, 1),
        0,
        (),
        False,
        False,
        False,
    ),
    "route_scatter": (
        (32, 1, 1),
        (256, 1, 1),
        (1, 1, 1),
        0,
        (),
        False,
        True,
        False,
    ),
    "gemm1_swiglu": (
        (32, 512, 1),
        (384, 1, 1),
        (2, 1, 1),
        223232,
        (("K", 7168),),
        True,
        True,
        True,
    ),
    "gemm2": (
        (56, 512, 1),
        (256, 1, 1),
        (2, 1, 1),
        223360,
        (("K", 2048),),
        True,
        True,
        True,
    ),
    "local_unpermute": (
        (1024, 1, 1),
        (128, 1, 1),
        (1, 1, 1),
        128,
        (("hidden_size", 7168),),
        True,
        False,
        True,
    ),
    "partial_barrier": (
        (1, 1, 1),
        (32, 1, 1),
        (1, 1, 1),
        0,
        (),
        False,
        False,
        False,
    ),
    "combine": (
        (128, 1, 1),
        (256, 1, 1),
        (1, 1, 1),
        29696,
        (),
        False,
        False,
        False,
    ),
}


def _check_cuda(result: tuple[Any, ...], operation: str) -> tuple[Any, ...]:
    if not result:
        raise RuntimeError(f"{operation} returned no CUDA status")
    if int(result[0]) != 0:
        raise RuntimeError(f"{operation} failed with CUDA status {result[0]}")
    return result[1:]


def _source_dir() -> Path:
    return Path(__file__).resolve().parent / "src"


def _load_manifest() -> tuple[dict[str, Any], Path]:
    source_dir = _source_dir()
    source_path = source_dir / _SOURCE_NAME
    manifest_path = source_dir / "manifest.json"
    if not source_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(
            "Blackwell BF16 rank-major kernel package is incomplete: expected "
            f"{source_path} and {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text())
    expected_constraints = {
        "activation_dtype": "bfloat16",
        "weight_dtype": "bfloat16",
        "output_dtype": "bfloat16",
        "world_size": _WORLD_SIZE,
        "tokens_per_rank": _TOKENS_PER_RANK,
        "hidden_dim": _HIDDEN_SIZE,
        "intermediate_dim": _INTERMEDIATE_SIZE,
        "num_experts": _NUM_EXPERTS,
        "local_experts": _LOCAL_EXPERTS,
        "top_k": _TOP_K,
        "layout": "rank_major",
    }
    if manifest.get("schema_version") != 1:
        raise RuntimeError("Blackwell BF16 rank-major manifest schema must be 1")
    if manifest.get("arch") != "sm_100a":
        raise RuntimeError("Blackwell BF16 rank-major manifest arch must be sm_100a")
    if manifest.get("compile_flags") != ["--use_fast_math"]:
        raise RuntimeError(
            "Blackwell BF16 rank-major compile flags must be ['--use_fast_math']"
        )
    if manifest.get("constraints") != expected_constraints:
        raise RuntimeError("Blackwell BF16 rank-major manifest constraints drifted")
    stages = manifest.get("stages")
    if (
        not isinstance(stages, list)
        or tuple(stage.get("name") for stage in stages) != _STAGE_NAMES
    ):
        raise RuntimeError("Blackwell BF16 rank-major stage order drifted")
    symbols = manifest.get("kernel_symbols")
    if symbols != [stage.get("symbol") for stage in stages]:
        raise RuntimeError("Blackwell BF16 rank-major kernel symbol order drifted")
    source_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if manifest.get("source_sha256") != source_digest:
        raise RuntimeError(
            "Blackwell BF16 rank-major source checksum differs from manifest"
        )
    for stage in stages:
        if not isinstance(stage.get("symbol"), str):
            raise RuntimeError("Blackwell BF16 rank-major stage has no symbol")
        stage_name = stage["name"]
        bindings = stage.get("bindings")
        if (
            not isinstance(bindings, list)
            or tuple(bindings) != _STAGE_BINDINGS[stage_name]
        ):
            raise RuntimeError(
                f"Blackwell BF16 rank-major stage {stage_name} binding order drifted"
            )
        if type(stage.get("use_pdl")) is not bool:
            raise RuntimeError(
                f"Blackwell BF16 rank-major stage {stage_name} use_pdl must be bool"
            )
        for field in ("pdl_sync", "pdl_launch"):
            if type(stage.get(field)) is not bool:
                raise RuntimeError(
                    f"Blackwell BF16 rank-major stage {stage_name} {field} must be bool"
                )
        for field in ("grid", "block", "cluster"):
            value = stage.get(field)
            if (
                not isinstance(value, list)
                or len(value) != 3
                or not all(isinstance(item, int) and item > 0 for item in value)
            ):
                raise RuntimeError(
                    f"Blackwell BF16 rank-major stage {stage.get('name')} has invalid {field}"
                )
        if (
            not isinstance(stage.get("dynamic_smem_bytes"), int)
            or stage["dynamic_smem_bytes"] < 0
        ):
            raise RuntimeError(
                f"Blackwell BF16 rank-major stage {stage.get('name')} has invalid shared memory"
            )
        scalar_bindings = stage.get("scalar_bindings")
        if not isinstance(scalar_bindings, list) or not all(
            isinstance(binding, list)
            and len(binding) == 2
            and isinstance(binding[0], str)
            and isinstance(binding[1], int)
            for binding in scalar_bindings
        ):
            raise RuntimeError(
                f"Blackwell BF16 rank-major stage {stage_name} has invalid scalar bindings"
            )
        launch_contract = (
            tuple(stage["grid"]),
            tuple(stage["block"]),
            tuple(stage["cluster"]),
            stage["dynamic_smem_bytes"],
            tuple(tuple(binding) for binding in scalar_bindings),
            stage["pdl_sync"],
            stage["pdl_launch"],
            stage["use_pdl"],
        )
        if launch_contract != _STAGE_LAUNCH_CONTRACTS[stage_name]:
            raise RuntimeError(
                f"Blackwell BF16 rank-major stage {stage_name} launch contract drifted"
            )
    return manifest, source_path


def _nvcc() -> Path:
    candidate = shutil.which("nvcc")
    if candidate is None:
        cuda_root = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        if cuda_root:
            path = Path(cuda_root) / "bin" / "nvcc"
            if path.is_file():
                candidate = str(path)
    if candidate is None:
        raise RuntimeError(
            "nvcc is required to build the Blackwell BF16 rank-major MoE-EP backend"
        )
    return Path(candidate).resolve()


def _compile_cubin(source_path: Path, manifest: dict[str, Any]) -> Path:
    from flashinfer.jit import env as jit_env

    nvcc = _nvcc()
    digest = hashlib.sha256()
    digest.update(source_path.read_bytes())
    digest.update(json.dumps(manifest, sort_keys=True).encode())
    digest.update(str(nvcc).encode())
    key = digest.hexdigest()[:20]
    build_dir = jit_env.FLASHINFER_JIT_DIR / f"blackwell_bf16_rank_major_{key}"
    build_dir.mkdir(parents=True, exist_ok=True)
    cubin_path = build_dir / "blackwell_bf16_rank_major_sm100.cubin"
    with FileLock(build_dir / "build.lock", thread_local=False):
        if not cubin_path.is_file():
            temporary = build_dir / f"kernel.{os.getpid()}.tmp.cubin"
            command = [
                str(nvcc),
                "-cubin",
                "-arch=sm_100a",
                "--std=c++17",
                "-O3",
                *manifest["compile_flags"],
                str(source_path),
                "-o",
                str(temporary),
            ]
            process = subprocess.run(command, text=True, capture_output=True)
            if process.returncode != 0:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Blackwell BF16 rank-major nvcc build failed:\n{process.stderr}"
                )
            os.replace(temporary, cubin_path)
    return cubin_path


def _group_name(process_group: Any) -> str:
    name = getattr(process_group, "group_name", None)
    if not isinstance(name, str) or not name:
        raise RuntimeError(
            "Blackwell BF16 rank-major session requires a named torch process group"
        )
    return name


@dataclass
class _SymmetricTensor:
    local: torch.Tensor
    peers: torch.Tensor
    handle: Any


@dataclass
class _WeightDescriptorSet:
    storage: torch.Tensor
    descriptors: dict[str, int]
    w13: torch.Tensor
    w2: torch.Tensor


def _alloc_symmetric(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    *,
    device: torch.device,
    group_name: str,
) -> _SymmetricTensor:
    from flashinfer.comm.torch_symmetric_memory import _alloc_symm_buffer_bytes

    elements = 1
    for extent in shape:
        elements *= extent
    element_size = torch.empty((), dtype=dtype).element_size()
    pointers, flat, handle = _alloc_symm_buffer_bytes(
        elements * element_size,
        _WORLD_SIZE,
        dtype,
        device,
        group_name,
    )
    local = flat.view(shape)
    peers = torch.tensor(pointers, dtype=torch.int64, device=device)
    return _SymmetricTensor(local=local, peers=peers, handle=handle)


def _encode_tensor_map(
    tensor: torch.Tensor,
    *,
    global_dim: tuple[int, ...],
    global_stride_bytes: tuple[int, ...],
    box_dim: tuple[int, ...],
) -> bytes:
    import cuda.bindings.driver as drv

    if tensor.dtype is not torch.bfloat16:
        raise TypeError(f"tensor map source must be bfloat16, got {tensor.dtype}")
    if tensor.data_ptr() % 16:
        raise ValueError("tensor map source must be 16-byte aligned")
    rank = len(global_dim)
    if len(global_stride_bytes) != rank - 1 or len(box_dim) != rank:
        raise ValueError("tensor map geometry rank mismatch")
    err, tensor_map = drv.cuTensorMapEncodeTiled(
        drv.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        rank,
        tensor.data_ptr(),
        [drv.cuuint64_t(value) for value in global_dim],
        [drv.cuuint64_t(value) for value in global_stride_bytes],
        [drv.cuuint32_t(value) for value in box_dim],
        [drv.cuuint32_t(1)] * rank,
        drv.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        drv.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
        drv.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,
        drv.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    if int(err) != 0:
        raise RuntimeError(f"cuTensorMapEncodeTiled failed with CUDA status {err}")
    return bytes(ctypes.string_at(tensor_map.getPtr(), _DESCRIPTOR_BYTES))


class _KernelLibrary:
    def __init__(
        self,
        cubin_path: Path,
        stages: list[dict[str, Any]],
        device_index: int,
    ) -> None:
        import cuda.bindings.driver as drv

        self._drv = drv
        (self._library,) = _check_cuda(
            drv.cuLibraryLoadFromFile(
                bytes(str(cubin_path), encoding="utf-8"), [], [], 0, [], [], 0
            ),
            "cuLibraryLoadFromFile",
        )
        self._kernels: dict[str, Any] = {}
        try:
            for stage in stages:
                (kernel,) = _check_cuda(
                    drv.cuLibraryGetKernel(
                        self._library, bytes(stage["symbol"], encoding="utf-8")
                    ),
                    f"cuLibraryGetKernel({stage['symbol']})",
                )
                shared_mem = int(stage["dynamic_smem_bytes"])
                if shared_mem > 48 * 1024:
                    _check_cuda(
                        drv.cuKernelSetAttribute(
                            drv.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                            shared_mem,
                            kernel,
                            drv.CUdevice(device_index),
                        ),
                        f"cuKernelSetAttribute({stage['symbol']})",
                    )
                self._kernels[stage["name"]] = kernel
        except BaseException:
            drv.cuLibraryUnload(self._library)
            raise

    def launch(
        self,
        stage: dict[str, Any],
        values: tuple[Any, ...],
        types: tuple[Any, ...],
        stream: int,
    ) -> None:
        drv = self._drv
        attributes = []
        if tuple(stage["cluster"]) != (1, 1, 1):
            value = drv.CUlaunchAttributeValue()
            value.clusterDim.x, value.clusterDim.y, value.clusterDim.z = stage[
                "cluster"
            ]
            attribute = drv.CUlaunchAttribute()
            attribute.id = drv.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
            attribute.value = value
            attributes.append(attribute)
        if stage["use_pdl"]:
            value = drv.CUlaunchAttributeValue()
            value.programmaticStreamSerializationAllowed = 1
            attribute = drv.CUlaunchAttribute()
            attribute.id = drv.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
            attribute.value = value
            attributes.append(attribute)

        config = drv.CUlaunchConfig()
        config.gridDimX, config.gridDimY, config.gridDimZ = stage["grid"]
        config.blockDimX, config.blockDimY, config.blockDimZ = stage["block"]
        config.sharedMemBytes = stage["dynamic_smem_bytes"]
        config.hStream = drv.CUstream(stream)
        config.attrs = attributes
        config.numAttrs = len(attributes)
        _check_cuda(
            drv.cuLaunchKernelEx(
                config,
                self._kernels[stage["name"]],
                (values, types),
                0,
            ),
            f"cuLaunchKernelEx({stage['symbol']})",
        )

    def close(self) -> None:
        if getattr(self, "_library", None) is not None:
            _check_cuda(self._drv.cuLibraryUnload(self._library), "cuLibraryUnload")
            self._library = None


class BlackwellBf16RankMajorSession:
    """Own exact workspace, descriptors, and launches for one EP rank."""

    def __init__(
        self,
        process_group: Any,
        rank: int,
        world_size: int,
        max_tokens_per_rank: int,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
    ) -> None:
        expected = {
            "world_size": (world_size, _WORLD_SIZE),
            "max_tokens_per_rank": (max_tokens_per_rank, _TOKENS_PER_RANK),
            "hidden_size": (hidden_size, _HIDDEN_SIZE),
            "intermediate_size": (intermediate_size, _INTERMEDIATE_SIZE),
            "num_experts": (num_experts, _NUM_EXPERTS),
            "top_k": (top_k, _TOP_K),
        }
        for name, (actual, wanted) in expected.items():
            if actual != wanted:
                raise ValueError(
                    f"Blackwell BF16 rank-major requires {name}={wanted}, got {actual}"
                )
        if not torch.cuda.is_available():
            raise RuntimeError("Blackwell BF16 rank-major session requires CUDA")
        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            raise RuntimeError(
                "Blackwell BF16 rank-major session requires initialized torch.distributed"
            )
        actual_world = torch.distributed.get_world_size(process_group)
        actual_rank = torch.distributed.get_rank(process_group)
        if (actual_world, actual_rank) != (_WORLD_SIZE, rank):
            raise RuntimeError(
                "Blackwell BF16 rank-major process-group topology mismatch: "
                f"expected world/rank={_WORLD_SIZE}/{rank}, got {actual_world}/{actual_rank}"
            )

        self._closed = False
        self._inputs_staged = False
        self._process_group = process_group
        self._rank = rank
        self._device_index = torch.cuda.current_device()
        self._device = torch.device("cuda", self._device_index)
        if torch.cuda.get_device_capability(self._device) != (10, 0):
            raise RuntimeError(
                "Blackwell BF16 rank-major session requires SM100, got compute capability "
                f"{torch.cuda.get_device_capability(self._device)}"
            )
        name = _group_name(process_group)
        self._hidden_states = _alloc_symmetric(
            (_TOKENS_PER_RANK, _HIDDEN_SIZE),
            torch.bfloat16,
            device=self._device,
            group_name=name,
        )
        self._topk_ids = _alloc_symmetric(
            (_TOKENS_PER_RANK, _TOP_K),
            torch.int32,
            device=self._device,
            group_name=name,
        )
        self._topk_weights = _alloc_symmetric(
            (_TOKENS_PER_RANK, _TOP_K),
            torch.float32,
            device=self._device,
            group_name=name,
        )
        self._local_partials = _alloc_symmetric(
            (_RANK_MAJOR_TOKENS, _HIDDEN_SIZE),
            torch.bfloat16,
            device=self._device,
            group_name=name,
        )
        self._flags = _alloc_symmetric(
            (2,), torch.uint32, device=self._device, group_name=name
        )
        self._flags.local.zero_()

        self._expert_ids_i64 = torch.empty(
            (_TOKENS_PER_RANK, _TOP_K), dtype=torch.int64, device=self._device
        )
        self._recv_hidden = torch.empty(
            (_RANK_MAJOR_TOKENS, _HIDDEN_SIZE),
            dtype=torch.bfloat16,
            device=self._device,
        )
        self._recv_local_ids = torch.empty(
            (_RANK_MAJOR_TOKENS, _TOP_K), dtype=torch.int32, device=self._device
        )
        self._recv_weights = torch.empty(
            (_RANK_MAJOR_TOKENS, _TOP_K), dtype=torch.float32, device=self._device
        )
        self._expert_scatter_offsets = torch.empty(
            (_LOCAL_EXPERTS,), dtype=torch.int32, device=self._device
        )
        self._expert_padded_row_offsets = torch.empty_like(self._expert_scatter_offsets)
        self._cta_to_expert = torch.empty(
            (_MAX_Y_GROUPS,), dtype=torch.int32, device=self._device
        )
        self._cta_to_mn_limit = torch.empty_like(self._cta_to_expert)
        self._num_non_exiting_ctas = torch.empty(
            (1,), dtype=torch.int32, device=self._device
        )
        self._total_padded_rows = torch.empty_like(self._num_non_exiting_ctas)
        self._route_map = torch.empty(
            (_FIXED_EXPERT_ROWS,), dtype=torch.int32, device=self._device
        )
        self._token_to_permuted = torch.empty(
            (_RANK_MAJOR_TOKENS, _TOP_K), dtype=torch.int32, device=self._device
        )
        self._compact_intermediate = torch.empty(
            (_FIXED_EXPERT_ROWS, _INTERMEDIATE_SIZE),
            dtype=torch.bfloat16,
            device=self._device,
        )
        self._gemm2_output = torch.empty(
            (_FIXED_EXPERT_ROWS + 1, _HIDDEN_SIZE),
            dtype=torch.bfloat16,
            device=self._device,
        )

        self._manifest, source_path = _load_manifest()
        cubin_path = _compile_cubin(source_path, self._manifest)
        torch.distributed.barrier(group=process_group)
        self._library = _KernelLibrary(
            cubin_path, self._manifest["stages"], self._device_index
        )
        self._weight_descriptor_sets: dict[tuple[int, int], _WeightDescriptorSet] = {}
        self._active_weight_key: tuple[int, int] | None = None
        self._descriptors: dict[str, int] = {}

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Blackwell BF16 rank-major session is closed")

    def stage_inputs(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> None:
        self._require_open()
        expected = (
            (
                "hidden_states",
                hidden_states,
                (_TOKENS_PER_RANK, _HIDDEN_SIZE),
                torch.bfloat16,
            ),
            ("topk_ids", topk_ids, (_TOKENS_PER_RANK, _TOP_K), torch.int64),
            ("topk_weights", topk_weights, (_TOKENS_PER_RANK, _TOP_K), torch.float32),
        )
        for name, tensor, shape, dtype in expected:
            if tuple(tensor.shape) != shape or tensor.dtype is not dtype:
                raise ValueError(
                    f"{name} must have shape {shape} and dtype {dtype}, got "
                    f"shape={tuple(tensor.shape)} dtype={tensor.dtype}"
                )
            if tensor.device != self._device or not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous on {self._device}")
        self._hidden_states.local.copy_(hidden_states)
        self._expert_ids_i64.copy_(topk_ids)
        self._topk_weights.local.copy_(topk_weights)
        self._inputs_staged = True

    def bind_weights(
        self, w13_block_major: torch.Tensor, w2_block_major: torch.Tensor
    ) -> None:
        self._require_open()
        expected = (
            (
                "w13_block_major",
                w13_block_major,
                (_LOCAL_EXPERTS, _HIDDEN_SIZE // 64, 2 * _INTERMEDIATE_SIZE, 64),
            ),
            (
                "w2_block_major",
                w2_block_major,
                (_LOCAL_EXPERTS, _INTERMEDIATE_SIZE // 64, _HIDDEN_SIZE, 64),
            ),
        )
        for name, tensor, shape in expected:
            if tuple(tensor.shape) != shape or tensor.dtype is not torch.bfloat16:
                raise ValueError(
                    f"{name} must have shape {shape} and dtype torch.bfloat16"
                )
            if tensor.device != self._device or not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous on {self._device}")
        key = (w13_block_major.data_ptr(), w2_block_major.data_ptr())
        cached = self._weight_descriptor_sets.get(key)
        if cached is not None:
            self._active_weight_key = key
            self._descriptors = cached.descriptors
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Blackwell BF16 rank-major weight descriptors cannot be built "
                "during CUDA graph capture; warm up every MoE layer first"
            )

        xlarge = 1 << 35
        dim_max = 1 << 31
        specs = (
            (
                "fc1_weights",
                w13_block_major,
                (64, 2 * _INTERMEDIATE_SIZE, _HIDDEN_SIZE // 64, _LOCAL_EXPERTS),
                (
                    128,
                    2 * _INTERMEDIATE_SIZE * 64 * 2,
                    (_HIDDEN_SIZE // 64) * 2 * _INTERMEDIATE_SIZE * 64 * 2,
                ),
                (64, 64, 2, 1),
            ),
            (
                "fc1_recv_hidden",
                self._recv_hidden,
                (_HIDDEN_SIZE, _RANK_MAJOR_TOKENS),
                (_HIDDEN_SIZE * 2,),
                (64, 1),
            ),
            (
                "fc1_output",
                self._compact_intermediate,
                (_INTERMEDIATE_SIZE, 64, dim_max, dim_max),
                (
                    _INTERMEDIATE_SIZE * 2,
                    (xlarge - _INTERMEDIATE_SIZE) * 2,
                    _INTERMEDIATE_SIZE * 2,
                ),
                (64, 64, 1, 1),
            ),
            (
                "fc2_weights",
                w2_block_major,
                (64, _HIDDEN_SIZE, _INTERMEDIATE_SIZE // 64, _LOCAL_EXPERTS),
                (
                    128,
                    _HIDDEN_SIZE * 64 * 2,
                    (_INTERMEDIATE_SIZE // 64) * _HIDDEN_SIZE * 64 * 2,
                ),
                (64, 128, 2, 1),
            ),
            (
                "fc2_input",
                self._compact_intermediate,
                (_INTERMEDIATE_SIZE, 64, dim_max, dim_max),
                (
                    _INTERMEDIATE_SIZE * 2,
                    (xlarge - _INTERMEDIATE_SIZE) * 2,
                    _INTERMEDIATE_SIZE * 2,
                ),
                (64, 32, 1, 1),
            ),
            (
                "fc2_output",
                self._gemm2_output,
                (_HIDDEN_SIZE, 64, dim_max, dim_max),
                (_HIDDEN_SIZE * 2, (xlarge - _HIDDEN_SIZE) * 2, _HIDDEN_SIZE * 2),
                (64, 64, 1, 1),
            ),
        )
        packed = bytearray()
        for _name, tensor, dims, strides, box in specs:
            packed += _encode_tensor_map(
                tensor,
                global_dim=dims,
                global_stride_bytes=strides,
                box_dim=box,
            )
        descriptor_storage = torch.tensor(
            list(packed), dtype=torch.uint8, device=self._device
        )
        base = descriptor_storage.data_ptr()
        if base % 64:
            raise RuntimeError("tensor-map upload must be at least 64-byte aligned")
        descriptors = {
            name: base + index * _DESCRIPTOR_BYTES
            for index, (name, *_rest) in enumerate(specs)
        }
        self._weight_descriptor_sets[key] = _WeightDescriptorSet(
            storage=descriptor_storage,
            descriptors=descriptors,
            w13=w13_block_major,
            w2=w2_block_major,
        )
        self._active_weight_key = key
        self._descriptors = descriptors

    @staticmethod
    def _ptr(value: torch.Tensor) -> int:
        return value.data_ptr()

    @staticmethod
    def _pointer_types(count: int) -> tuple[Any, ...]:
        return (ctypes.c_void_p,) * count

    def _stage_arguments(
        self, name: str, output: torch.Tensor
    ) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        p = self._ptr
        world_rank = (_WORLD_SIZE, self._rank)
        if name == "input_barrier":
            return (
                (
                    p(self._expert_ids_i64),
                    p(self._topk_ids.local),
                    *world_rank,
                    p(self._flags.peers),
                ),
                (
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_int32,
                    ctypes.c_int32,
                    ctypes.c_void_p,
                ),
            )
        if name == "dispatch":
            values: tuple[Any, ...] = (
                p(self._recv_hidden),
                p(self._recv_local_ids),
                p(self._recv_weights),
                *world_rank,
                p(self._flags.peers),
                p(self._hidden_states.local),
                p(self._hidden_states.peers),
                p(self._topk_ids.local),
                p(self._topk_ids.peers),
                p(self._topk_weights.local),
                p(self._topk_weights.peers),
            )
            return values, (
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int32,
                ctypes.c_int32,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
            )
        if name == "route_reset":
            sentinel = self._gemm2_output[_FIXED_EXPERT_ROWS]
            values = (p(self._expert_scatter_offsets), p(sentinel))
            return values, self._pointer_types(2)
        if name == "route_count":
            values = (
                p(self._recv_local_ids),
                p(self._expert_scatter_offsets),
                p(self._token_to_permuted),
            )
            return values, self._pointer_types(3)
        if name == "route_finalize":
            values = (
                p(self._expert_scatter_offsets),
                p(self._cta_to_expert),
                p(self._cta_to_mn_limit),
                p(self._expert_padded_row_offsets),
                p(self._num_non_exiting_ctas),
                p(self._total_padded_rows),
                p(self._route_map),
            )
            return values, self._pointer_types(7)
        if name == "route_scatter":
            values = (
                p(self._recv_local_ids),
                p(self._expert_padded_row_offsets),
                p(self._route_map),
                p(self._token_to_permuted),
            )
            return values, self._pointer_types(4)
        if name == "gemm1_swiglu":
            values = (
                self._descriptors["fc1_weights"],
                self._descriptors["fc1_recv_hidden"],
                self._descriptors["fc1_output"],
                p(self._route_map),
                p(self._num_non_exiting_ctas),
                p(self._cta_to_expert),
                p(self._cta_to_mn_limit),
                _HIDDEN_SIZE,
            )
            return values, (*self._pointer_types(7), ctypes.c_int32)
        if name == "gemm2":
            values = (
                self._descriptors["fc2_weights"],
                self._descriptors["fc2_input"],
                self._descriptors["fc2_output"],
                p(self._num_non_exiting_ctas),
                p(self._cta_to_expert),
                p(self._cta_to_mn_limit),
                _INTERMEDIATE_SIZE,
            )
            return values, (*self._pointer_types(6), ctypes.c_int32)
        if name == "local_unpermute":
            values = (
                p(self._gemm2_output),
                p(self._recv_weights),
                p(self._token_to_permuted),
                p(self._local_partials.local),
                _HIDDEN_SIZE,
            )
            return values, (*self._pointer_types(4), ctypes.c_int32)
        if name == "partial_barrier":
            return (
                (*world_rank, p(self._flags.peers)),
                (ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p),
            )
        if name == "combine":
            values = (
                p(output),
                *world_rank,
                p(self._flags.peers),
                p(self._local_partials.local),
                p(self._local_partials.peers),
            )
            return values, (
                ctypes.c_void_p,
                ctypes.c_int32,
                ctypes.c_int32,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
            )
        raise RuntimeError(f"unknown Blackwell BF16 rank-major stage {name!r}")

    def run(self, output: torch.Tensor) -> torch.Tensor:
        self._require_open()
        if self._active_weight_key is None:
            raise RuntimeError("Blackwell BF16 rank-major expert weights are not bound")
        if not self._inputs_staged:
            raise RuntimeError("Blackwell BF16 rank-major inputs are not staged")
        if (
            tuple(output.shape) != (_TOKENS_PER_RANK, _HIDDEN_SIZE)
            or output.dtype is not torch.bfloat16
            or output.device != self._device
            or not output.is_contiguous()
        ):
            raise ValueError(
                "output must be contiguous bfloat16 [128, 7168] on the session device"
            )
        stream = torch.cuda.current_stream(self._device).cuda_stream
        for stage in self._manifest["stages"]:
            values, types = self._stage_arguments(stage["name"], output)
            if len(values) != len(stage["bindings"]):
                raise RuntimeError(
                    f"stage {stage['name']} argument count differs from the manifest"
                )
            self._library.launch(stage, values, types, stream)
        self._inputs_staged = False
        return output

    def destroy(self) -> None:
        if self._closed:
            return
        # Raw driver launches do not register their tensor/module lifetimes
        # with PyTorch.  Wait for the last submitted DAG before unloading its
        # cubin; the owning layer may release the pooled tensor workspace as
        # soon as this method returns.
        torch.cuda.synchronize(self._device)
        self._closed = True
        self._library.close()
        self._weight_descriptor_sets.clear()
        self._active_weight_key = None
        self._descriptors.clear()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.destroy()


__all__ = ["BlackwellBf16RankMajorSession"]
