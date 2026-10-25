"""Runtime owner for the capacity-bounded SM100 BF16 rank-major MoE-EP DAG."""

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
_BARRIER_PHASES = 2
_BARRIER_BANKS = 2
_BARRIER_SLOTS_PER_RANK = _BARRIER_PHASES + (
    _BARRIER_PHASES * _BARRIER_BANKS * _WORLD_SIZE
)
_TOKENS_PER_RANK = 128
_RANK_MAJOR_TOKENS = _WORLD_SIZE * _TOKENS_PER_RANK
_HIDDEN_SIZE = 7168
_INTERMEDIATE_SIZE = 2048
_NUM_EXPERTS = 256
_LOCAL_EXPERTS = _NUM_EXPERTS // _WORLD_SIZE
_TOP_K = 8
_FIXED_EXPERT_ROWS = 32768
_MAX_Y_GROUPS = 512
_ROUTE_IN_DISPATCH_MAX_T = 128
# Up to this many active rows the forward uses the LL (flag-in-data) path.
_LL_MAX_T = 32
_LL_CAPACITY_T = 32
_LL_BANKS = 2
_LL_HEADER_WORDS = 16
_LL_ROW_WORDS = 3584
_LL_SLOT_WORDS = _LL_HEADER_WORDS + _LL_ROW_WORDS
_DESCRIPTOR_BYTES = 128
_FC1_WEIGHT_BOX_DIM = (64, 128, 2, 1)
_SOURCE_NAME = "flashinfer_blackwell_moe_ep_layer_sm100.cu"
_STAGE_NAMES = (
    "stage_inputs",
    "input_arrive",
    "dispatch",
    "route_fused",
    "gemm1_swiglu",
    "gemm2",
    "local_unpermute",
    "partial_arrive",
    "combine",
    "ll_push",
    "ll_recv",
    "ll_unpermute",
    "ll_combine",
)
_STAGE_BINDINGS = {
    "stage_inputs": (
        "hidden_in",
        "ids_in",
        "weights_in",
        "hidden_states",
        "topk_ids",
        "topk_weights",
        "ids_is_int64",
        "hidden_copy",
        "active_tokens_per_rank",
    ),
    "input_arrive": (
        "topk_ids",
        "active_tokens_per_rank",
        "pg_world",
        "pg_rank",
        "pg_flags",
        "routed_peers",
        "route_codes_peers",
    ),
    "dispatch": (
        "recv_hidden",
        "recv_local_ids",
        "recv_weights",
        "active_tokens_per_rank",
        "pg_world",
        "pg_rank",
        "pg_flags",
        "hidden_states",
        "hidden_states_peers",
        "topk_ids",
        "topk_ids_peers",
        "topk_weights",
        "topk_weights_peers",
        "routed",
        "expert_scatter_offsets",
        "token_to_permuted",
        "cta_to_expert",
        "cta_to_mn_limit",
        "expert_padded_row_offsets",
        "num_non_exiting_ctas",
        "total_padded_rows",
        "route_map",
        "zero_sentinel",
        "route_codes",
    ),
    "route_fused": (
        "recv_local_ids",
        "expert_scatter_offsets",
        "token_to_permuted",
        "cta_to_expert",
        "cta_to_mn_limit",
        "expert_padded_row_offsets",
        "num_non_exiting_ctas",
        "total_padded_rows",
        "route_map",
        "zero_sentinel",
        "active_tokens_per_rank",
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
        "active_tokens_per_rank",
    ),
    "partial_arrive": ("pg_world", "pg_rank", "pg_flags"),
    "combine": (
        "output",
        "pg_world",
        "pg_rank",
        "pg_flags",
        "local_partials",
        "local_partials_peers",
        "topk_ids",
    ),
    "ll_push": (
        "hidden_in",
        "ids_in",
        "weights_in",
        "topk_ids",
        "ids_is_int64",
        "active_tokens_per_rank",
        "pg_world",
        "pg_rank",
        "ll_dispatch_peers",
        "ll_control",
        "ll_reuse_peers",
        "ll_combine_peers",
    ),
    "ll_recv": (
        "recv_hidden",
        "recv_local_ids",
        "recv_weights",
        "active_tokens_per_rank",
        "pg_world",
        "pg_rank",
        "ll_dispatch",
        "ll_control",
        "expert_scatter_offsets",
        "token_to_permuted",
        "cta_to_expert",
        "cta_to_mn_limit",
        "expert_padded_row_offsets",
        "num_non_exiting_ctas",
        "total_padded_rows",
        "route_map",
        "zero_sentinel",
        "row_splits",
    ),
    "ll_unpermute": (
        "expert_output",
        "topk_weights",
        "token_to_permuted",
        "ll_combine_peers",
        "ll_control",
        "hidden_size",
        "active_tokens_per_rank",
        "pg_rank",
    ),
    "ll_combine": (
        "output",
        "pg_rank",
        "ll_combine",
        "ll_control",
        "topk_ids",
        "ll_reuse",
    ),
}
_STAGE_LAUNCH_CONTRACTS = {
    "stage_inputs": (
        (128, 1, 1),
        (256, 1, 1),
        (1, 1, 1),
        0,
        (("active_tokens_per_rank", 128),),
        False,
        True,
        False,
    ),
    "input_arrive": (
        (1, 1, 1),
        (32, 1, 1),
        (1, 1, 1),
        0,
        (("active_tokens_per_rank", 128),),
        True,
        True,
        True,
    ),
    "dispatch": (
        (1025, 1, 1),
        (256, 1, 1),
        (1, 1, 1),
        15488,
        (("active_tokens_per_rank", 128),),
        True,
        True,
        True,
    ),
    "route_fused": (
        (1, 1, 1),
        (1024, 1, 1),
        (1, 1, 1),
        0,
        (("active_tokens_per_rank", 128),),
        True,
        True,
        True,
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
        (1024, 7, 1),
        (128, 1, 1),
        (1, 1, 1),
        128,
        (("hidden_size", 7168), ("active_tokens_per_rank", 128)),
        True,
        True,
        True,
    ),
    "partial_arrive": ((1, 1, 1), (32, 1, 1), (1, 1, 1), 0, (), True, True, True),
    "combine": ((128, 7, 1), (128, 1, 1), (1, 1, 1), 17408, (), True, False, True),
    "ll_push": (
        (32, 8, 8),
        (288, 1, 1),
        (1, 1, 1),
        0,
        (("active_tokens_per_rank", 32),),
        False,
        True,
        False,
    ),
    "ll_recv": (
        (257, 1, 1),
        (256, 1, 1),
        (1, 1, 1),
        0,
        (("active_tokens_per_rank", 32), ("row_splits", 8)),
        True,
        True,
        True,
    ),
    "ll_unpermute": (
        (256, 7, 1),
        (128, 1, 1),
        (1, 1, 1),
        0,
        (("hidden_size", 7168), ("active_tokens_per_rank", 32)),
        True,
        True,
        True,
    ),
    "ll_combine": (
        (32, 7, 1),
        (160, 1, 1),
        (1, 1, 1),
        0,
        (),
        True,
        False,
        True,
    ),
}


def _check_cuda(result: tuple[Any, ...], operation: str) -> tuple[Any, ...]:
    if not result:
        raise RuntimeError(f"{operation} returned no CUDA status")
    if int(result[0]) != 0:
        raise RuntimeError(f"{operation} failed with CUDA status {result[0]}")
    return result[1:]


def _ll_row_splits(active_tokens_per_rank: int) -> int:
    """Row splits of the LL push/receive, ``max(1, 8 // T)`` for T active rows.

    The push and the receive launch ``8 * T * splits`` row CTAs: 64 for T in
    {1, 2, 4, 8}, 48 for T = 3, and ``8 * T`` for every other T.
    """
    return max(1, 8 // active_tokens_per_rank)


def _active_stage_grids(
    active_tokens_per_rank: int,
) -> dict[str, tuple[int, int, int]]:
    """Return launch grids for one active prefix within the fixed capacity."""
    if not 1 <= active_tokens_per_rank <= _TOKENS_PER_RANK:
        raise ValueError(
            f"active tokens per rank must be in [1, 128], got {active_tokens_per_rank}"
        )
    # One local expert group needs at most 64 routed rows.  With 32 local
    # experts, spreading the first row across every expert costs 32 groups;
    # each further group needs another 64 routes, so N + 31 bounds the active
    # y-groups, which stays below the y=512 capacity for every active prefix.
    gemm_y = min(_MAX_Y_GROUPS, active_tokens_per_rank + _LOCAL_EXPERTS - 1)
    grids = {
        "stage_inputs": (active_tokens_per_rank, 1, 1),
        # one extra dispatch CTA computes the route from pushed route codes (no route kernel)
        "dispatch": (
            _WORLD_SIZE * active_tokens_per_rank
            + (1 if active_tokens_per_rank <= _ROUTE_IN_DISPATCH_MAX_T else 0),
            1,
            1,
        ),
        "gemm1_swiglu": (32, gemm_y, 1),
        "gemm2": (56, gemm_y, 1),
        "local_unpermute": (_WORLD_SIZE * active_tokens_per_rank, 7, 1),
        "combine": (active_tokens_per_rank, 7, 1),
    }
    if active_tokens_per_rank <= _LL_MAX_T:
        grids.update(
            {
                "ll_push": (
                    active_tokens_per_rank,
                    _WORLD_SIZE,
                    _ll_row_splits(active_tokens_per_rank),
                ),
                "ll_recv": (
                    _WORLD_SIZE
                    * active_tokens_per_rank
                    * _ll_row_splits(active_tokens_per_rank)
                    + 1,
                    1,
                    1,
                ),
                "ll_unpermute": (_WORLD_SIZE * active_tokens_per_rank, 7, 1),
                "ll_combine": (active_tokens_per_rank, 7, 1),
            }
        )
    return grids


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

    def capture_id(self, stream: int) -> int | None:
        """Id of the CUDA graph capture active on ``stream`` (unique in the process), or None."""
        drv = self._drv
        status, capture_id, *_ = _check_cuda(
            drv.cuStreamGetCaptureInfo(drv.CUstream(stream)), "cuStreamGetCaptureInfo"
        )
        if status == drv.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_NONE:
            return None
        return int(capture_id)

    def launch(
        self,
        stage: dict[str, Any],
        values: tuple[Any, ...],
        types: tuple[Any, ...],
        stream: int,
        *,
        grid: tuple[int, int, int] | None = None,
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

        launch_grid = tuple(stage["grid"]) if grid is None else grid
        manifest_grid = tuple(stage["grid"])
        if (
            len(launch_grid) != 3
            or any(type(value) is not int or value <= 0 for value in launch_grid)
            or any(
                value > maximum
                for value, maximum in zip(launch_grid, manifest_grid, strict=True)
            )
        ):
            raise RuntimeError(
                f"stage {stage['name']} grid {launch_grid} exceeds its positive "
                f"manifest bound {manifest_grid}"
            )
        cluster = tuple(stage["cluster"])
        if any(
            value % cluster_dim
            for value, cluster_dim in zip(launch_grid, cluster, strict=True)
        ):
            raise RuntimeError(
                f"stage {stage['name']} grid {launch_grid} is not divisible by "
                f"cluster {cluster}"
            )

        config = drv.CUlaunchConfig()
        config.gridDimX, config.gridDimY, config.gridDimZ = launch_grid
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
    """Own exact workspace, descriptors, and launches for one EP rank.

    Every rank in the process group must stage the same active row count for a
    launch.  The count may vary between launches from one through the fixed
    128-row capacity.

    Up to 32 active rows per rank, ``stage_inputs`` already sends the rows to
    the ranks that own their experts, so every rank must call ``stage_inputs``
    and ``run`` symmetrically.  Before the first ``bind_weights``,
    ``stage_inputs`` keeps a copy of the inputs instead and ``run`` sends them,
    so every rank must also call ``bind_weights`` at the same point relative to
    ``stage_inputs`` and ``run`` (the layer API binds in ``compute``).
    Otherwise a repeated ``stage_inputs`` completes the pending forward on some
    ranks and replaces it on others, and the ranks pair different forwards:
    ``run`` hangs or returns rows computed from other inputs.

    Peers wait for a staged forward whose rows were sent, or whose kernels
    ``run`` began to launch when a launch failed.  The session completes such a
    forward into internal scratch output when ``stage_inputs`` is called again,
    and when ``run`` rejects its arguments (which also sends deferred rows), if
    expert weights are bound and the forward was staged in the active CUDA graph
    capture (or eagerly, outside any capture); a later ``run`` resumes it at the
    failed launch.  Other staged inputs, including deferred rows not sent yet,
    are replaced, also inside a capture.

    A CUDA graph that captures ``stage_inputs`` must also capture the matching
    ``run``, for every active row count: inputs staged in a capture that is no
    longer active never ran and are dropped, so ``run`` reports that no inputs
    are staged.  Inside a capture, ``stage_inputs`` and ``run`` raise for an
    eagerly staged forward that peers wait for, and ``run`` also for other
    eagerly staged inputs of up to 32 rows.  Kernels that ``run`` recorded into
    a capture before one of its launches failed never run: calls inside that
    capture raise, and afterwards an eagerly staged forward restarts at its
    first kernel.  Do not replay such a graph.
    """

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
        self._active_tokens_per_rank: int | None = None
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
            (_BARRIER_SLOTS_PER_RANK,),
            torch.uint32,
            device=self._device,
            group_name=name,
        )
        self._flags.local.zero_()
        self._routed = _alloc_symmetric(
            (_WORLD_SIZE * 4,),
            torch.int32,
            device=self._device,
            group_name=name,
        )
        self._routed.local.zero_()
        # Route codes [source][token][8] (one byte per route), stored by each source's input_arrive.
        self._route_codes = _alloc_symmetric(
            (_WORLD_SIZE * _TOKENS_PER_RANK * 2,),
            torch.int32,
            device=self._device,
            group_name=name,
        )
        self._route_codes.local.zero_()
        # LL (flag-in-data) buffers of the small-T path: [bank][source][token][header + row] dispatch words and
        # [bank][owner][token][row] combine words; zero words never match an epoch (epochs start at 1).
        self._ll_dispatch = _alloc_symmetric(
            (_LL_BANKS * _WORLD_SIZE * _LL_CAPACITY_T * _LL_SLOT_WORDS,),
            torch.int64,
            device=self._device,
            group_name=name,
        )
        self._ll_dispatch.local.zero_()
        # Buffer-reuse epochs released by every peer at the start of each LL forward.
        self._ll_reuse = _alloc_symmetric(
            (_WORLD_SIZE,),
            torch.int32,
            device=self._device,
            group_name=name,
        )
        self._ll_reuse.local.zero_()
        self._ll_combine = _alloc_symmetric(
            (_LL_BANKS * _WORLD_SIZE * _LL_CAPACITY_T * _LL_ROW_WORDS,),
            torch.int64,
            device=self._device,
            group_name=name,
        )
        self._ll_combine.local.zero_()
        # [0]: epoch of the last LL forward (read by ll_push), [1]: epoch of the current LL forward.
        self._ll_control = torch.zeros((2,), dtype=torch.int32, device=self._device)
        # Output of a staged forward that the session completes itself (see _complete_staged_forward).
        self._scratch_output = torch.empty(
            (_TOKENS_PER_RANK, _HIDDEN_SIZE), dtype=torch.bfloat16, device=self._device
        )
        # Kernels of the staged forward's run() sequence that are already launched,
        # and the CUDA graph capture they were launched in (None: eagerly).
        self._launched_stages = 0
        self._launch_capture_id: int | None = None
        # Copy of the inputs of an LL stage staged before the first bind_weights; run() sends them.
        self._ll_deferred_hidden = torch.empty(
            (_LL_CAPACITY_T, _HIDDEN_SIZE), dtype=torch.bfloat16, device=self._device
        )
        self._ll_deferred_ids = torch.empty(
            (_LL_CAPACITY_T, _TOP_K), dtype=torch.int64, device=self._device
        )
        self._ll_deferred_weights = torch.empty(
            (_LL_CAPACITY_T, _TOP_K), dtype=torch.float32, device=self._device
        )
        # The staged LL inputs are that copy; run() sends it first.
        self._ll_push_deferred = False
        # CUDA graph capture that recorded the staged inputs (None: staged eagerly).
        self._staged_capture_id: int | None = None

        self._recv_hidden = torch.zeros(
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
        self._stage_by_name = {
            stage["name"]: stage for stage in self._manifest["stages"]
        }
        cubin_path = _compile_cubin(source_path, self._manifest)
        torch.distributed.barrier(group=process_group)
        self._library = _KernelLibrary(
            cubin_path, self._manifest["stages"], self._device_index
        )
        self._weight_descriptor_sets: dict[tuple[int, int], _WeightDescriptorSet] = {}
        self._active_weight_key: tuple[int, int] | None = None
        self._descriptors: dict[str, int] = {}

    def _ll_set_epoch_for_test(self, last_epoch: int) -> None:
        """Test hook: make ``last_epoch`` the epoch of the last LL forward.

        Call it on every rank of a quiescent session (no forward in flight);
        it synchronizes the device and the process group before and after the
        update.  The next LL forward uses ``last_epoch + 1`` (``0`` maps to
        ``2``).  Every peer's buffer-reuse word is set to ``last_epoch``, the
        state a completed forward with that epoch leaves behind.
        """
        if not 0 <= last_epoch < 2**32:
            raise ValueError(f"LL epoch must be in [0, 2**32), got {last_epoch}")
        value = last_epoch - 2**32 if last_epoch >= 2**31 else last_epoch
        torch.cuda.synchronize(self._device)
        torch.distributed.barrier(group=self._process_group)
        self._ll_control.fill_(value)
        self._ll_reuse.local.fill_(value)
        torch.cuda.synchronize(self._device)
        torch.distributed.barrier(group=self._process_group)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Blackwell BF16 rank-major session is closed")

    def _stage_needs_completion(self) -> bool:
        """Whether peers wait for the rest of the staged forward.

        Its LL rows were sent, or ``run`` launched part of its kernels.
        """
        if not self._inputs_staged or self._active_tokens_per_rank is None:
            return False
        if self._launched_stages:
            return True
        return self._active_tokens_per_rank <= _LL_MAX_T and not self._ll_push_deferred

    def _current_capture_id(self) -> int | None:
        stream = torch.cuda.current_stream(self._device).cuda_stream
        return self._library.capture_id(stream)

    def _forget_work_of_ended_captures(self, capture_id: int | None) -> None:
        """Forget work recorded in a CUDA graph capture that is no longer active.

        Capture only records kernels.  Inputs staged in such a capture never
        ran: drop them.  Kernels that run() recorded into such a capture never
        ran either: an eagerly staged forward restarts at its first kernel.
        """
        if (
            self._inputs_staged
            and self._staged_capture_id is not None
            and self._staged_capture_id != capture_id
        ):
            self._clear_stage()
        elif (
            self._launched_stages
            and self._launch_capture_id is not None
            and self._launch_capture_id != capture_id
        ):
            self._launched_stages = 0
            self._launch_capture_id = None

    def _launch_failed_in_capture(self, capture_id: int | None) -> bool:
        """Whether a launch of the staged forward failed in the active capture."""
        return (
            self._launched_stages > 0
            and capture_id is not None
            and self._launch_capture_id == capture_id
        )

    def _clear_stage(self) -> None:
        self._inputs_staged = False
        self._active_tokens_per_rank = None
        self._staged_capture_id = None
        self._ll_push_deferred = False
        self._launched_stages = 0
        self._launch_capture_id = None

    def _stage_blocker(self, capture_id: int | None) -> str | None:
        """Why the staged forward cannot be completed now, or None."""
        if self._active_weight_key is None:
            return "expert weights are not bound"
        if self._launch_failed_in_capture(capture_id):
            return "one of its launches failed in this CUDA graph capture"
        if self._staged_capture_id != capture_id:
            return "it was staged outside the active CUDA graph capture"
        return None

    def _complete_staged_forward(self, capture_id: int | None) -> None:
        """Launch the rest of the staged forward into the scratch output.

        The kernels not launched yet (a deferred push first) run exactly as in
        ``run``: they advance the epochs and release the buffers, and only the
        output rows are discarded.  Every peer completes the same forward,
        through ``run`` or here.
        """
        self._launch_staged(
            self._scratch_output[: self._active_tokens_per_rank], capture_id
        )

    def stage_inputs(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> None:
        """Stage one active prefix; all EP ranks must use the same row count."""
        self._require_open()
        active_tokens_per_rank = hidden_states.shape[0]
        if not 1 <= active_tokens_per_rank <= _TOKENS_PER_RANK:
            raise ValueError(
                "active tokens per rank must be in [1, 128], got "
                f"{active_tokens_per_rank}"
            )
        expected = (
            (
                "hidden_states",
                hidden_states,
                (active_tokens_per_rank, _HIDDEN_SIZE),
                torch.bfloat16,
            ),
            ("topk_ids", topk_ids, (active_tokens_per_rank, _TOP_K), None),
            (
                "topk_weights",
                topk_weights,
                (active_tokens_per_rank, _TOP_K),
                torch.float32,
            ),
        )
        for name, tensor, shape, dtype in expected:
            valid_dtype = (
                tensor.dtype in (torch.int32, torch.int64)
                if dtype is None
                else tensor.dtype is dtype
            )
            if tuple(tensor.shape) != shape or not valid_dtype:
                dtype_requirement = (
                    "torch.int32 or torch.int64" if dtype is None else str(dtype)
                )
                raise ValueError(
                    f"{name} must have shape {shape} and dtype {dtype_requirement}, got "
                    f"shape={tuple(tensor.shape)} dtype={tensor.dtype}"
                )
            if tensor.device != self._device or not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous on {self._device}")
        capture_id = self._current_capture_id()
        self._forget_work_of_ended_captures(capture_id)
        if self._stage_needs_completion():
            # Complete the staged forward before staging again, so no epoch is
            # sent twice.  Sent rows imply bound weights (see below).
            blocker = self._stage_blocker(capture_id)
            if blocker is not None:
                raise RuntimeError(
                    "Blackwell BF16 rank-major cannot complete the staged forward "
                    f"that run() did not finish: {blocker}"
                )
            self._complete_staged_forward(capture_id)
        # Nothing waits for other staged inputs: replace them.  If a launch
        # below fails, no inputs are staged.
        self._clear_stage()
        if active_tokens_per_rank <= _LL_MAX_T and self._active_weight_key is None:
            # Sent rows can only be received by a forward with bound weights,
            # and the caller may bind them after this call or fail before it
            # does.  Keep a copy of the inputs; run() sends it after its checks.
            self._ll_deferred_hidden[:active_tokens_per_rank].copy_(hidden_states)
            self._ll_deferred_ids[:active_tokens_per_rank].copy_(topk_ids)
            self._ll_deferred_weights[:active_tokens_per_rank].copy_(topk_weights)
            self._ll_push_deferred = True
            self._active_tokens_per_rank = active_tokens_per_rank
            self._inputs_staged = True
            self._staged_capture_id = capture_id
            return
        # One staging kernel: hidden row copy, ids narrowed to int32, weights (the
        # dependent input barrier waits on it through PDL).
        hidden_copy = 1
        if hidden_states.data_ptr() % 16:
            self._hidden_states.local[:active_tokens_per_rank].copy_(hidden_states)
            hidden_copy = 0
        p = self._ptr
        if active_tokens_per_rank <= _LL_MAX_T:
            self._launch_ll_push(
                hidden_states if hidden_copy else self._hidden_states.local,
                topk_ids,
                topk_weights,
                active_tokens_per_rank,
            )
            self._active_tokens_per_rank = active_tokens_per_rank
            self._inputs_staged = True
            self._staged_capture_id = capture_id
            return
        self._library.launch(
            self._stage_by_name["stage_inputs"],
            (
                p(hidden_states),
                p(topk_ids),
                p(topk_weights),
                p(self._hidden_states.local),
                p(self._topk_ids.local),
                p(self._topk_weights.local),
                1 if topk_ids.dtype is torch.int64 else 0,
                hidden_copy,
                active_tokens_per_rank,
            ),
            (*self._pointer_types(6), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32),
            torch.cuda.current_stream(self._device).cuda_stream,
            grid=(active_tokens_per_rank, 1, 1),
        )
        self._active_tokens_per_rank = active_tokens_per_rank
        self._inputs_staged = True
        self._staged_capture_id = capture_id

    def _launch_ll_push(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        active_tokens_per_rank: int,
    ) -> None:
        """Send one LL forward's ids, weights and routed rows (ll_push).

        One push kernel writes every token's ids/weights (and routed rows)
        into the owners' LL dispatch buffers; nothing waits on a peer here.
        """
        p = self._ptr
        self._library.launch(
            self._stage_by_name["ll_push"],
            (
                p(hidden_states),
                p(topk_ids),
                p(topk_weights),
                p(self._topk_ids.local),
                1 if topk_ids.dtype is torch.int64 else 0,
                active_tokens_per_rank,
                _WORLD_SIZE,
                self._rank,
                p(self._ll_dispatch.peers),
                p(self._ll_control),
                p(self._ll_reuse.peers),
                p(self._ll_combine.peers),
            ),
            (
                *self._pointer_types(4),
                *(ctypes.c_int32,) * 4,
                *self._pointer_types(4),
            ),
            torch.cuda.current_stream(self._device).cuda_stream,
            grid=(
                active_tokens_per_rank,
                _WORLD_SIZE,
                _ll_row_splits(active_tokens_per_rank),
            ),
        )

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
                _FC1_WEIGHT_BOX_DIM,
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
        if self._active_tokens_per_rank is None:
            raise RuntimeError(
                "Blackwell BF16 rank-major active token count is not set"
            )
        active_tokens_per_rank = self._active_tokens_per_rank
        world_rank = (_WORLD_SIZE, self._rank)
        if name in ("input_barrier", "input_arrive"):
            return (
                (
                    p(self._topk_ids.local),
                    active_tokens_per_rank,
                    *world_rank,
                    p(self._flags.peers),
                    p(self._routed.peers),
                    p(self._route_codes.peers),
                ),
                (
                    ctypes.c_void_p,
                    ctypes.c_int32,
                    ctypes.c_int32,
                    ctypes.c_int32,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                ),
            )
        if name == "dispatch":
            values: tuple[Any, ...] = (
                p(self._recv_hidden),
                p(self._recv_local_ids),
                p(self._recv_weights),
                active_tokens_per_rank,
                *world_rank,
                p(self._flags.peers),
                p(self._hidden_states.local),
                p(self._hidden_states.peers),
                p(self._topk_ids.local),
                p(self._topk_ids.peers),
                p(self._topk_weights.local),
                p(self._topk_weights.peers),
                p(self._routed.local),
                p(self._expert_scatter_offsets),
                p(self._token_to_permuted),
                p(self._cta_to_expert),
                p(self._cta_to_mn_limit),
                p(self._expert_padded_row_offsets),
                p(self._num_non_exiting_ctas),
                p(self._total_padded_rows),
                p(self._route_map),
                p(self._gemm2_output[_FIXED_EXPERT_ROWS]),
                p(self._route_codes.local),
            )
            return values, (
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int32,
                ctypes.c_int32,
                ctypes.c_int32,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                *self._pointer_types(10),
            )
        if name == "ll_recv":
            values = (
                p(self._recv_hidden),
                p(self._recv_local_ids),
                p(self._recv_weights),
                active_tokens_per_rank,
                *world_rank,
                p(self._ll_dispatch.local),
                p(self._ll_control),
                p(self._expert_scatter_offsets),
                p(self._token_to_permuted),
                p(self._cta_to_expert),
                p(self._cta_to_mn_limit),
                p(self._expert_padded_row_offsets),
                p(self._num_non_exiting_ctas),
                p(self._total_padded_rows),
                p(self._route_map),
                p(self._gemm2_output[_FIXED_EXPERT_ROWS]),
                _ll_row_splits(active_tokens_per_rank),
            )
            return values, (
                *self._pointer_types(3),
                ctypes.c_int32,
                ctypes.c_int32,
                ctypes.c_int32,
                *self._pointer_types(11),
                ctypes.c_int32,
            )
        if name == "ll_unpermute":
            values = (
                p(self._gemm2_output),
                p(self._recv_weights),
                p(self._token_to_permuted),
                p(self._ll_combine.peers),
                p(self._ll_control),
                _HIDDEN_SIZE,
                active_tokens_per_rank,
                self._rank,
            )
            return values, (
                *self._pointer_types(5),
                ctypes.c_int32,
                ctypes.c_int32,
                ctypes.c_int32,
            )
        if name == "ll_combine":
            values = (
                p(output),
                self._rank,
                p(self._ll_combine.local),
                p(self._ll_control),
                p(self._topk_ids.local),
                p(self._ll_reuse.local),
            )
            return values, (
                ctypes.c_void_p,
                ctypes.c_int32,
                *self._pointer_types(4),
            )
        if name == "route_fused":
            sentinel = self._gemm2_output[_FIXED_EXPERT_ROWS]
            values = (
                p(self._recv_local_ids),
                p(self._expert_scatter_offsets),
                p(self._token_to_permuted),
                p(self._cta_to_expert),
                p(self._cta_to_mn_limit),
                p(self._expert_padded_row_offsets),
                p(self._num_non_exiting_ctas),
                p(self._total_padded_rows),
                p(self._route_map),
                p(sentinel),
                active_tokens_per_rank,
            )
            return values, (*self._pointer_types(10), ctypes.c_int32)
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
                active_tokens_per_rank,
            )
            return values, (*self._pointer_types(4), ctypes.c_int32, ctypes.c_int32)
        if name in ("partial_barrier", "partial_arrive"):
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
                p(self._topk_ids.local),
            )
            return values, (
                ctypes.c_void_p,
                ctypes.c_int32,
                ctypes.c_int32,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
            )
        raise RuntimeError(f"unknown Blackwell BF16 rank-major stage {name!r}")

    def run(self, output: torch.Tensor) -> torch.Tensor:
        self._require_open()
        capture_id = self._current_capture_id()
        self._forget_work_of_ended_captures(capture_id)
        try:
            self._check_run(output, capture_id)
        except Exception:
            # Nothing of this call was launched.  Peers may wait for the staged
            # forward, so complete it (a deferred push is sent first) before
            # reporting the error.
            must_complete = self._stage_needs_completion() or self._ll_push_deferred
            if must_complete and self._stage_blocker(capture_id) is None:
                self._complete_staged_forward(capture_id)
            raise
        return self._launch_staged(output, capture_id)

    def _check_run(self, output: torch.Tensor, capture_id: int | None) -> None:
        if self._active_weight_key is None:
            raise RuntimeError("Blackwell BF16 rank-major expert weights are not bound")
        if not self._inputs_staged:
            raise RuntimeError("Blackwell BF16 rank-major inputs are not staged")
        if self._active_tokens_per_rank is None:
            raise RuntimeError(
                "Blackwell BF16 rank-major active token count is not set"
            )
        if self._launch_failed_in_capture(capture_id):
            raise RuntimeError(
                "Blackwell BF16 rank-major cannot resume the staged forward: one of "
                "its launches failed in this CUDA graph capture"
            )
        active_tokens_per_rank = self._active_tokens_per_rank
        must_match = active_tokens_per_rank <= _LL_MAX_T or self._launched_stages > 0
        if must_match and self._staged_capture_id != capture_id:
            raise RuntimeError(
                "Blackwell BF16 rank-major inputs were staged outside the active "
                "CUDA graph capture"
            )
        if (
            tuple(output.shape) != (active_tokens_per_rank, _HIDDEN_SIZE)
            or output.dtype is not torch.bfloat16
            or output.device != self._device
            or not output.is_contiguous()
        ):
            raise ValueError(
                "output must be contiguous bfloat16 "
                f"[{active_tokens_per_rank}, 7168] on the session device"
            )

    def _launch_staged(
        self, output: torch.Tensor, capture_id: int | None
    ) -> torch.Tensor:
        """Launch the rest of the staged forward (every check already passed)."""
        active_tokens_per_rank = self._active_tokens_per_rank
        if active_tokens_per_rank is None:
            raise RuntimeError(
                "Blackwell BF16 rank-major active token count is not set"
            )
        active_grids = _active_stage_grids(active_tokens_per_rank)
        stream = torch.cuda.current_stream(self._device).cuda_stream
        order: tuple[str, ...]
        if active_tokens_per_rank <= _LL_MAX_T:
            order = ("ll_recv", "gemm1_swiglu", "gemm2", "ll_unpermute", "ll_combine")
            if self._ll_push_deferred:
                order = ("ll_push", *order)
        else:
            order = (
                "input_arrive",
                "dispatch",
                "route_fused",
                "gemm1_swiglu",
                "gemm2",
                "local_unpermute",
                "partial_arrive",
                "combine",
            )
        if not self._launched_stages:
            self._launch_capture_id = capture_id
        # Each launch is recorded at once.  After a failed launch the forward
        # resumes at that kernel, so none of its kernels is launched twice.
        for index in range(self._launched_stages, len(order)):
            stage = self._stage_by_name[order[index]]
            if stage["name"] == "ll_push":
                self._launch_ll_push(
                    self._ll_deferred_hidden,
                    self._ll_deferred_ids,
                    self._ll_deferred_weights,
                    active_tokens_per_rank,
                )
            elif not (
                stage["name"] == "route_fused"
                and active_tokens_per_rank <= _ROUTE_IN_DISPATCH_MAX_T
            ):
                values, types = self._stage_arguments(stage["name"], output)
                if len(values) != len(stage["bindings"]):
                    raise RuntimeError(
                        f"stage {stage['name']} argument count differs from the manifest"
                    )
                self._library.launch(
                    stage,
                    values,
                    types,
                    stream,
                    grid=active_grids.get(stage["name"]),
                )
            self._launched_stages = index + 1
        self._clear_stage()
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
