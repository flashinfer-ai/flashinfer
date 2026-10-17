"""W4A16 routed MoE execution."""

from __future__ import annotations

from b12x._lib.quant.block_codec import BLOCK_CODECS
from b12x.moe._shared.trellis_codebooks import LUT_E4M3
from b12x.moe._shared.kernels.w4a16.route_pack import (
    pack_topk_routes_by_expert as _pack_topk_routes_by_expert,
)
import cuda.bindings.driver as cuda
from b12x._lib.utils.device import current_cuda_stream
import cutlass.cute as cute
import cutlass
from b12x._lib.utils.pointers import make_ptr
from b12x.moe._shared.kernels.w4a16.host import max_packed_route_slots
from b12x.moe._shared.kernels.activations import normalize_moe_activation
from b12x.moe._shared.kernels.w4a16.host import packed_gemm_scratch_elements
from b12x.moe._shared.kernels.w4a16.host import plan_w4a16_buffers
from b12x.moe._shared.kernels.w4a16.host import prefill_fused_sum_eligible
from b12x.moe._shared.kernels.w4a16.host import prefill_fused_sum_enabled
from b12x.moe._shared.kernels.w4a16.host import select_route_block_size_m
import torch
from b12x.moe._shared.kernels.w4a16.host import validate_activation
from b12x.moe._shared.kernels.w4a16.kernel.activation_launch import (
    _ACTIVATION_CACHE,
    _validate_activation_amax,
)
from b12x.moe._shared.kernels.w4a16.kernel.buffers import _get_c_tmp
from b12x.moe._shared.kernels.w4a16.kernel.config import (
    _ALLOWED_ROUTED_SIZES,
    _DEFAULT_MAX_SHARED_MEM,
    _FP32_TOPK_WEIGHTS,
    _MAX_DIRECT_TOPK_ROUTE_M,
    _MODEL_OPT_W13_LAYOUTS,
    _TC_DECODE_MAX_M,
    _TRELLIS256_BITS,
    _TRELLIS256_CODEBOOKS,
    _TRELLIS256_W13_LAYOUTS,
    _W4A16_SMALL_M_DIRECT_MAX_M,
    _WEIGHT_LAYOUTS,
    _covering_count,
    _normalize_activation_swiglu_params,
    _normalize_element_dtype,
    _normalize_scale_format,
    _w4a16_tc_decode_preferred,
)
from b12x.moe._shared.kernels.w4a16.kernel.direct_launch import (
    _FC2_DIRECT_CACHE,
    _SMALL_M_DIRECT_CACHE,
    _small_m_direct_host_barrier_reset_enabled,
    _small_m_direct_supported,
    _w4a16_small_m_direct_launch_flat,
)
from b12x.moe._shared.kernels.w4a16.kernel.fused_launch import (
    _FUSED_CACHE,
    _w4a16_fused_moe_launch_flat,
    compile_w4a16_fused_moe,
)
from b12x.moe._shared.kernels.w4a16.kernel.gemm_launch import _CACHE
from b12x.moe._shared.kernels.w4a16.kernel.sum_launch import (
    _SUM_CACHE,
    _w4a16_topk_sum_launch_flat,
)
from b12x.moe._shared.kernels.w4a16.kernel.types import (
    W4A16FusedMoeCompileResult,
    W4A16SmallMDirectCompileResult,
    W4A16TopKSumCompileResult,
    _W4A16GemmLaunch,
)


def _w4a16_weight_flat_elements(
    *,
    num_experts: int,
    size_n: int,
    size_k: int,
    weight_layout: str,
) -> int:
    """Flat element count of a packed W4A16 weight tensor (uint8 for modelopt,
    int32 otherwise), matching the compile-time fake construction."""

    if weight_layout == "modelopt":
        return int(num_experts) * int(size_n) * (int(size_k) // 2)
    return int(num_experts) * (int(size_k) // 16) * (int(size_n) // 16 * 32)


def clear_w4a16_kernel_cache() -> None:
    _CACHE.clear()
    _FUSED_CACHE.clear()
    _ACTIVATION_CACHE.clear()
    _SUM_CACHE.clear()
    _SMALL_M_DIRECT_CACHE.clear()
    _FC2_DIRECT_CACHE.clear()


def _validate_topk_ids(
    topk_ids: torch.Tensor,
    *,
    require_cuda: bool,
    require_contiguous: bool = True,
) -> None:
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("topk_ids must be torch.int32 or torch.int64")
    if require_cuda and not topk_ids.is_cuda:
        raise ValueError("topk_ids must be a CUDA tensor")
    if require_contiguous and not topk_ids.is_contiguous():
        raise ValueError("topk_ids must be contiguous")


def _validate_expert_map(
    expert_map: torch.Tensor | None,
    *,
    device: torch.device | None = None,
    exact_num_experts: int | None = None,
) -> None:
    if expert_map is None:
        return
    if expert_map.dtype != torch.int32:
        raise TypeError("expert_map must be torch.int32")
    if device is None:
        if not expert_map.is_cuda:
            raise ValueError("expert_map must be a CUDA tensor")
    elif expert_map.device != device:
        raise ValueError("expert_map must be on the same device as a_input")
    if exact_num_experts is None:
        if expert_map.ndim != 1 or not expert_map.is_contiguous():
            raise ValueError("expert_map must be a contiguous rank-1 tensor")
        return
    if not expert_map.is_contiguous():
        raise ValueError("expert_map must be contiguous")
    if expert_map.ndim != 1 or int(expert_map.numel()) != int(exact_num_experts):
        raise ValueError(
            f"expert_map must have shape {(int(exact_num_experts),)}, got {tuple(expert_map.shape)}"
        )


def pack_topk_routes_by_expert(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    *,
    expert_map: torch.Tensor | None = None,
    packed_route_indices: torch.Tensor | None = None,
    block_expert_ids: torch.Tensor | None = None,
    packed_route_count: torch.Tensor | None = None,
    expert_offsets: torch.Tensor | None = None,
    expert_counts: torch.Tensor | None = None,
    launches=None,
    stream: cuda.CUstream | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group top-k routes by expert and pad each group to the GEMM M-block size."""
    _validate_topk_ids(topk_ids, require_cuda=True)
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")
    _validate_expert_map(expert_map, exact_num_experts=int(num_experts))
    del stream
    return _pack_topk_routes_by_expert(
        topk_ids,
        int(block_size),
        int(num_experts),
        expert_map=expert_map,
        packed_route_indices=packed_route_indices,
        block_expert_ids=block_expert_ids,
        packed_route_count=packed_route_count,
        expert_offsets=expert_offsets,
        expert_counts=expert_counts,
        launches=launches,
    )


def _trellis256_dense_tile_config(size_k: int, size_n: int) -> tuple[int, int]:
    """Return an m-invariant t256 tile so dense row bits cannot drift with m."""
    if int(size_k) % 64 != 0:
        raise ValueError(f"trellis_t256 dense K must be divisible by 64, got {size_k}")
    if int(size_n) % 256 == 0:
        return (64, 256)
    if int(size_n) % 128 == 0:
        # With dense M=64, (tile_k=64,tile_n=128) maps to the measured
        # (cta_threads=128, cta_m/n/k_blocks=4/8/4) register specialization.
        # The superficially symmetric 128x128 tile would need a missing
        # 256-thread 4/8/8 register entry and fail before compilation.
        return (64, 128)
    raise ValueError(
        "trellis_t256 dense N must be divisible by 128 (or 256 for the wide tile), "
        f"got N={size_n}"
    )


def _trellis256_dense_launch_geometry(
    *,
    size_m: int,
    size_k: int,
    size_n: int,
    sms: int,
) -> tuple[int, tuple[int, int]]:
    """Avoid short spill waves in the persistent dense-Trellis schedule.

    A default M64/N256 launch can leave only a handful of CTAs in a second or
    third SM wave. Narrow projections benefit from a smaller M tile, while wide
    projections already expose enough N parallelism and only need N128 for a
    two-wave spill. Past three waves the smaller tile's scheduler overhead costs
    more than the remaining imbalance.
    """
    default = (64, (64, 256 if size_n % 256 == 0 else 128))
    if size_n % 256 != 0 or sms <= 0:
        return default
    tasks = _covering_count(int(size_m), 64) * (int(size_n) // 256)
    waves = _covering_count(tasks, int(sms))
    if waves not in (2, 3):
        return default
    spill = tasks - (waves - 1) * int(sms)
    if spill > max(16, int(sms) // 10):
        return default
    n_tiles_256 = int(size_n) // 256
    if n_tiles_256 <= 32:
        return (48, (64, 128 if waves == 2 else 256))
    if waves == 2:
        return (64, (64, 128))
    if int(size_k) <= 4096:
        return (48, (64, 256))
    return default


def _resolve_exl3_hadamard_128(hadamard_128):
    if hadamard_128 is None:
        try:
            import exllamav3_ext
        except ImportError as exc:
            raise RuntimeError(
                "run_trellis256_dense requires exllamav3_ext.had_r_128 for the "
                "EXL3 input/output rotations"
            ) from exc
        hadamard_128 = exllamav3_ext.had_r_128
    elif not callable(hadamard_128):
        hadamard_128 = getattr(hadamard_128, "had_r_128", None)
    if not callable(hadamard_128):
        raise TypeError("hadamard_128 must be callable or expose had_r_128")
    return hadamard_128


def _trellis_dense_buffer(
    name: str,
    buffer: torch.Tensor | None,
    *,
    shape: tuple[int, int],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Validate caller-owned dense scratch or allocate it before capture."""
    if buffer is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"Trellis dense {name} is not initialized for CUDA graph capture; "
                "provide caller-owned storage"
            )
        return torch.empty(shape, dtype=dtype, device=device)
    if (
        tuple(buffer.shape) != shape
        or buffer.dtype != dtype
        or buffer.device != device
        or not buffer.is_contiguous()
        or int(buffer.data_ptr()) % 16 != 0
    ):
        raise ValueError(
            f"{name} must be contiguous, 16-byte-aligned {dtype} with shape "
            f"{shape} on {device}"
        )
    return buffer


def _run_trellis256_dense_current_device(
    x: torch.Tensor,
    prepared_dense,
    *,
    launch: _W4A16GemmLaunch,
    execution_lut: torch.Tensor | None,
    grid_cap: int,
    output: torch.Tensor | None = None,
    gemm_output: torch.Tensor | None = None,
    input_f16: torch.Tensor | None = None,
    rotated_f16: torch.Tensor | None = None,
    rotated_compute: torch.Tensor | None = None,
    gemm_output_f16: torch.Tensor | None = None,
    output_f16: torch.Tensor | None = None,
    hadamard_128,
    stream: cuda.CUstream | None = None,
) -> torch.Tensor:
    """Run one native EXL3 linear on the already-selected CUDA device.

    This is a true dense entry: E=1 and contiguous row identities are synthesized
    inside the kernel, so no top-k tensors or route-packing kernels are created.
    Outer rotations follow EXL3 order exactly: fp16 ``suh`` multiply before the
    input H128, and fp16 ``svh`` multiply after the output H128.
    """
    if getattr(prepared_dense, "weight_layout", None) != "trellis_t256":
        raise ValueError("run_trellis256_dense requires prepared trellis_t256 weights")
    if int(getattr(prepared_dense, "num_experts", 0)) != 1:
        raise ValueError("run_trellis256_dense requires an honest E=1 prepared weight")
    trellis_codebook = str(getattr(prepared_dense, "trellis_codebook", "")).lower()
    if trellis_codebook not in _TRELLIS256_CODEBOOKS:
        raise NotImplementedError(
            f"run_trellis256_dense has no decoder for codebook {trellis_codebook!r}"
        )
    trellis_bits = int(getattr(prepared_dense, "trellis_bits", 0))
    if trellis_bits not in _TRELLIS256_BITS:
        raise ValueError(
            f"prepared dense weight has invalid trellis_bits={trellis_bits}"
        )
    trellis_pair_kind = getattr(prepared_dense, "trellis_pair_kind", None)
    trellis_rate_axis = getattr(prepared_dense, "trellis_rate_axis", None)
    if (trellis_pair_kind is None) != (trellis_rate_axis is None):
        raise ValueError("prepared pair weight has incomplete pair metadata")
    compute_dtype = getattr(prepared_dense, "params_dtype", None)
    if compute_dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(
            "prepared dense weight must select fp16 or bf16 compute, got "
            f"{compute_dtype}"
        )
    cutlass_dtype = (
        cutlass.Float16 if compute_dtype == torch.float16 else cutlass.BFloat16
    )
    if x.ndim != 2 or int(x.shape[0]) <= 0:
        raise ValueError(f"x must be a non-empty rank-2 tensor, got {tuple(x.shape)}")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"x must be fp16 or bf16, got {x.dtype}")
    if not x.is_cuda or not x.is_contiguous():
        raise ValueError("x must be a contiguous CUDA tensor")
    m, size_k = (int(v) for v in x.shape)
    size_n = int(prepared_dense.out_features)
    if size_k != int(prepared_dense.in_features):
        raise ValueError(
            f"x has K={size_k}, prepared dense weight expects {prepared_dense.in_features}"
        )
    if prepared_dense.trellis.device != x.device:
        raise ValueError("x and prepared dense weight must be on the same CUDA device")
    output = _trellis_dense_buffer(
        "output",
        output,
        shape=(m, size_n),
        dtype=x.dtype,
        device=x.device,
    )
    if not callable(hadamard_128):
        raise TypeError("prepared Trellis transform must be callable")

    gemm_output = _trellis_dense_buffer(
        "gemm_output",
        gemm_output,
        shape=(m, size_n),
        dtype=compute_dtype,
        device=x.device,
    )
    if x.dtype == torch.float16:
        x_f16 = x
    else:
        input_f16 = _trellis_dense_buffer(
            "input_f16",
            input_f16,
            shape=(m, size_k),
            dtype=torch.float16,
            device=x.device,
        )
        input_f16.copy_(x)
        x_f16 = input_f16
    rotated_f16 = _trellis_dense_buffer(
        "rotated_f16",
        rotated_f16,
        shape=(m, size_k),
        dtype=torch.float16,
        device=x.device,
    )
    hadamard_128(x_f16, rotated_f16, prepared_dense.suh, None, 1.0)
    if compute_dtype == torch.float16:
        rotated_compute = rotated_f16
    else:
        rotated_compute = _trellis_dense_buffer(
            "rotated_compute",
            rotated_compute,
            shape=(m, size_k),
            dtype=torch.bfloat16,
            device=x.device,
        )
        rotated_compute.copy_(rotated_f16)

    if int(launch.kernel.moe_block_size) <= 0:
        raise ValueError("prepared Trellis launch has invalid block-row geometry")
    moe_block_size = int(launch.kernel.moe_block_size)
    route_blocks = (m + moe_block_size - 1) // moe_block_size
    if int(launch.kernel.max_m_blocks) < route_blocks:
        raise ValueError("prepared Trellis launch cannot cover the requested rows")
    if (
        launch.kernel.trellis_bits,
        launch.kernel.trellis_codebook,
        launch.kernel.trellis_pair_kind,
        launch.kernel.trellis_rate_axis,
    ) != (
        trellis_bits,
        trellis_codebook,
        trellis_pair_kind,
        trellis_rate_axis,
    ):
        raise ValueError("prepared Trellis launcher metadata differs from the weight")
    tile_n = int(launch.kernel.tile_n)
    n_tiles = size_n // tile_n
    grid_x = min(int(grid_cap), max(route_blocks * n_tiles, 1))
    if trellis_codebook != "mcg" and execution_lut is None:
        raise ValueError("prepared Trellis decoder is missing its execution LUT")
    stream = current_cuda_stream() if stream is None else stream
    dummy_i32 = prepared_dense.workspace[:1]
    launch.kernel.compiled(
        make_ptr(
            cutlass_dtype,
            rotated_compute.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass_dtype,
            rotated_compute.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        prepared_dense.trellis,
        make_ptr(
            cutlass_dtype,
            gemm_output.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        prepared_dense.scale.view(torch.uint8).view(torch.int32).view(-1),
        prepared_dense.global_scale,
        dummy_i32,
        dummy_i32,
        dummy_i32,
        prepared_dense.global_scale,
        launch.c_tmp,
        prepared_dense.workspace,
        # MCG kernels never dereference the LUT ABI slot.
        dummy_i32 if trellis_codebook == "mcg" else execution_lut,
        m,
        grid_x,
        stream,
    )

    if compute_dtype == torch.float16:
        gemm_f16 = gemm_output
    else:
        gemm_output_f16 = _trellis_dense_buffer(
            "gemm_output_f16",
            gemm_output_f16,
            shape=(m, size_n),
            dtype=torch.float16,
            device=x.device,
        )
        gemm_output_f16.copy_(gemm_output)
        gemm_f16 = gemm_output_f16
    if output.dtype == torch.float16:
        hadamard_128(gemm_f16, output, None, prepared_dense.svh, 1.0)
    else:
        output_f16 = _trellis_dense_buffer(
            "output_f16",
            output_f16,
            shape=(m, size_n),
            dtype=torch.float16,
            device=x.device,
        )
        hadamard_128(gemm_f16, output_f16, None, prepared_dense.svh, 1.0)
        output.copy_(output_f16)
    return output


def run_trellis256_dense(
    x: torch.Tensor,
    prepared_dense,
    *,
    launch: _W4A16GemmLaunch,
    execution_lut: torch.Tensor | None,
    grid_cap: int,
    output: torch.Tensor | None = None,
    gemm_output: torch.Tensor | None = None,
    input_f16: torch.Tensor | None = None,
    rotated_f16: torch.Tensor | None = None,
    rotated_compute: torch.Tensor | None = None,
    gemm_output_f16: torch.Tensor | None = None,
    output_f16: torch.Tensor | None = None,
    hadamard_128,
) -> torch.Tensor:
    """Run a session-materialized native or compact P24/P33 EXL3 linear."""
    if not isinstance(x, torch.Tensor) or not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    with torch.cuda.device(x.device):
        return _run_trellis256_dense_current_device(
            x,
            prepared_dense,
            launch=launch,
            execution_lut=execution_lut,
            grid_cap=grid_cap,
            output=output,
            gemm_output=gemm_output,
            input_f16=input_f16,
            rotated_f16=rotated_f16,
            rotated_compute=rotated_compute,
            gemm_output_f16=gemm_output_f16,
            output_f16=output_f16,
            hadamard_128=hadamard_128,
            stream=None,
        )


def _resolve_route_block_size_m(
    *,
    m: int,
    topk: int,
    route_num_experts: int,
    planned_block_size_m: int | None,
    fused_launch: W4A16FusedMoeCompileResult | W4A16SmallMDirectCompileResult | None,
) -> int:
    """Resolve one route geometry shared by scratch, packing, and GEMM."""
    planned = None if planned_block_size_m is None else int(planned_block_size_m)
    if planned is not None and planned not in _ALLOWED_ROUTED_SIZES:
        raise ValueError(f"unsupported planned W4A16 moe_block_size={planned}")
    if fused_launch is None:
        return planned or select_route_block_size_m(m, topk, route_num_experts)

    compiled = int(fused_launch.moe_block_size)
    if planned is not None and compiled != planned:
        raise RuntimeError(
            "preplanned W4A16 route geometry does not match the fused launch: "
            f"planned_block_size_m={planned}, launch_block_size_m={compiled}"
        )
    return compiled


def _w4a16_stream_is_capturing(
    stream: cuda.CUstream,
    *,
    current_stream: cuda.CUstream,
) -> bool:
    """Observe capture on either Torch's current stream or an explicit stream."""
    current_capturing = torch.cuda.is_current_stream_capturing()
    if current_capturing or int(stream) == int(current_stream):
        return current_capturing
    result, status = cuda.cuStreamIsCapturing(stream)
    if result != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(
            f"cuStreamIsCapturing failed for the selected W4A16 stream: {result}"
        )
    return int(status) != 0


def run_w4a16_moe(
    a_input: torch.Tensor,
    prepared,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    activation: str,
    intermediate_cache13: torch.Tensor,
    intermediate_cache2: torch.Tensor,
    output: torch.Tensor,
    prefill_sum_accum: torch.Tensor | None = None,
    fc1_c_tmp: torch.Tensor | None = None,
    fc2_c_tmp: torch.Tensor | None = None,
    packed_route_indices: torch.Tensor | None = None,
    block_expert_ids: torch.Tensor | None = None,
    packed_route_count: torch.Tensor | None = None,
    expert_offsets: torch.Tensor | None = None,
    expert_counts: torch.Tensor | None = None,
    expert_map: torch.Tensor | None = None,
    output_expert_map: torch.Tensor | None = None,
    activation_amax: torch.Tensor | None = None,
    layer_idx: int | None = None,
    apply_router_weight_on_input: bool = False,
    fast_math: bool = True,
    swiglu_limit: float | None = None,
    swiglu_clamp_after_silu: bool = False,
    swiglu_alpha: float | None = None,
    swiglu_beta: float | None = None,
    fused_launch: W4A16FusedMoeCompileResult
    | W4A16SmallMDirectCompileResult
    | None = None,
    topk_sum_launch: W4A16TopKSumCompileResult | None = None,
    route_block_size_m: int | None = None,
    intermediate_rotation_scales: torch.Tensor | None = None,
    a_input_up: torch.Tensor | None = None,
    full_rotation: bool = False,
    suh_gate_table: torch.Tensor | None = None,
    suh_up_table: torch.Tensor | None = None,
    svh_table: torch.Tensor | None = None,
    rotation_a_gate: torch.Tensor | None = None,
    rotation_a_up: torch.Tensor | None = None,
    route_mode: str = "auto",
    route_pack_launches=None,
    stream: cuda.CUstream | None = None,
) -> torch.Tensor:
    activation = normalize_moe_activation(activation)
    is_gated = validate_activation(activation)
    swiglu_limit, swiglu_alpha, swiglu_beta = _normalize_activation_swiglu_params(
        activation,
        swiglu_limit,
        swiglu_alpha,
        swiglu_beta,
    )
    full_rotation = bool(full_rotation)
    route_mode = str(route_mode).lower()
    if route_mode not in {"auto", "direct", "packed"}:
        raise ValueError("W4A16 route_mode must be 'auto', 'direct', or 'packed'")
    rotation_input_dtype = _normalize_element_dtype(a_input.dtype)
    prepared_dtype = getattr(prepared, "params_dtype", a_input.dtype)
    if full_rotation:
        element_dtype = _normalize_element_dtype(prepared_dtype)
        if element_dtype != "fp16":
            raise TypeError("full_rotation requires fp16 prepared weights/scratch")
        if output.dtype != torch.float32:
            raise TypeError(
                f"full_rotation output must be torch.float32, got {output.dtype}"
            )
    else:
        element_dtype = rotation_input_dtype
        if output.dtype != a_input.dtype:
            raise TypeError(
                f"output must have dtype {a_input.dtype}, got {output.dtype}"
            )
        if prepared_dtype != a_input.dtype:
            raise TypeError(
                f"prepared weights were built for {prepared_dtype}, but a_input has dtype {a_input.dtype}"
            )
    weight_layout = getattr(prepared, "weight_layout", "packed")
    if weight_layout not in _WEIGHT_LAYOUTS:
        raise ValueError(f"unsupported W4A16 weight_layout {weight_layout!r}")
    trellis_bits = int(getattr(prepared, "trellis_bits", 3))
    intermediate_hadamard = bool(getattr(prepared, "intermediate_hadamard", False))
    trellis_codebook = str(
        getattr(prepared, "trellis_codebook", None) or LUT_E4M3
    ).lower()
    fc1_trellis_pair_kind = getattr(prepared, "fc1_trellis_pair_kind", None)
    fc2_trellis_pair_kind = getattr(prepared, "fc2_trellis_pair_kind", None)
    fc1_trellis_pair_modes = getattr(prepared, "fc1_trellis_pair_modes", None)
    fc2_trellis_pair_modes = getattr(prepared, "fc2_trellis_pair_modes", None)
    prepared_tile_config = getattr(prepared, "tile_config", None)
    if (fc1_trellis_pair_kind is None) != (fc2_trellis_pair_kind is None):
        raise ValueError("prepared trellis pair weights have incomplete pair metadata")
    if weight_layout == "trellis_t256":
        if trellis_bits not in _TRELLIS256_BITS:
            raise ValueError(
                f"prepared trellis_t256 bitrate must be in {_TRELLIS256_BITS}, "
                f"got {trellis_bits}"
            )
        if trellis_codebook not in _TRELLIS256_CODEBOOKS:
            raise NotImplementedError(
                "trellis_t256 execution has no decoder for codebook "
                f"{trellis_codebook!r}"
            )
        if fc1_trellis_pair_kind is not None:
            fc1_trellis_pair_kind = str(fc1_trellis_pair_kind).upper()
            fc2_trellis_pair_kind = str(fc2_trellis_pair_kind).upper()
            dynamic_kinds = {"PDYNAMIC", "P33_P43"}
            static_kinds = {"P24", "P33", "P43", "P44"}
            dynamic_pairs = (
                fc1_trellis_pair_kind in dynamic_kinds
                or fc2_trellis_pair_kind in dynamic_kinds
            )
            if dynamic_pairs:
                if fc1_trellis_pair_kind != fc2_trellis_pair_kind:
                    raise ValueError(
                        "prepared dynamic fused trellis pair kinds must match"
                    )
            elif (
                fc1_trellis_pair_kind not in static_kinds
                or fc2_trellis_pair_kind not in static_kinds
            ):
                raise ValueError("unsupported prepared static trellis pair kind")
            pair_metadata_dtype = (
                torch.int64 if fc1_trellis_pair_kind == "P33_P43" else torch.int32
            )
            if trellis_bits != 3:
                raise ValueError(
                    "prepared trellis pairs require the trellis_bits=3 base "
                    "specialization"
                )
            if dynamic_pairs:
                for name, modes in (
                    ("fc1", fc1_trellis_pair_modes),
                    ("fc2", fc2_trellis_pair_modes),
                ):
                    if (
                        modes is None
                        or modes.dtype != pair_metadata_dtype
                        or modes.device != a_input.device
                        or tuple(modes.shape) != (int(prepared.num_experts),)
                        or not modes.is_contiguous()
                    ):
                        raise ValueError(
                            f"prepared dynamic {name} trellis pair metadata must be "
                            f"contiguous {pair_metadata_dtype}[num_experts] on the "
                            "input device"
                        )
        if activation_amax is not None:
            raise NotImplementedError(
                "trellis_t256 activation-amax collection is not exposed through "
                "the registered launch ABI"
            )
    if intermediate_hadamard and not full_rotation:
        raise ValueError(
            "prepared intermediate-Hadamard metadata requires full rotation; "
            f"got intermediate_hadamard={intermediate_hadamard}, full_rotation={full_rotation}"
        )
    scale_format = _normalize_scale_format(
        getattr(prepared, "scale_format", None)
        or (
            "e8m0_k32"
            if getattr(prepared, "source_format", "") == "fp4_e8m0_k32"
            else "e4m3_k16"
        )
    )
    w13_layout = getattr(
        prepared,
        "w13_layout",
        "w13" if weight_layout == "modelopt" else "packed",
    )
    if weight_layout == "modelopt":
        if w13_layout not in _MODEL_OPT_W13_LAYOUTS:
            raise ValueError(f"unsupported W4A16 w13_layout {w13_layout!r}")
    elif weight_layout == "trellis_t256":
        if w13_layout not in _TRELLIS256_W13_LAYOUTS:
            raise ValueError(f"unsupported trellis_t256 w13_layout {w13_layout!r}")
    else:
        w13_layout = "packed"
    dual_a_required = bool(
        intermediate_rotation_scales is not None
        and weight_layout == "trellis_t256"
        and w13_layout == "trellis_t256_proj"
    )
    if full_rotation and not dual_a_required:
        raise ValueError(
            "full_rotation requires projection-major trellis intermediate rotation"
        )
    if dual_a_required and a_input_up is None and not full_rotation:
        raise ValueError(
            "exact projection-major trellis_t256 rotation requires a_input_up"
        )
    if a_input_up is not None and (not dual_a_required or full_rotation):
        raise ValueError(
            "a_input_up is only valid for exact projection-major trellis_t256 rotation"
        )
    if a_input_up is not None:
        if (
            a_input_up.shape != a_input.shape
            or a_input_up.dtype != a_input.dtype
            or a_input_up.device != a_input.device
        ):
            raise ValueError(
                "a_input_up must match a_input shape, dtype, and device; got "
                f"{tuple(a_input_up.shape)}/{a_input_up.dtype}/{a_input_up.device} "
                f"vs {tuple(a_input.shape)}/{a_input.dtype}/{a_input.device}"
            )
        if not a_input_up.is_contiguous():
            raise ValueError("a_input_up must be contiguous")
    if topk_weights.dtype != torch.float32:
        raise TypeError("topk_weights must be torch.float32")
    _validate_topk_ids(topk_ids, require_cuda=False, require_contiguous=False)
    if (
        not a_input.is_contiguous()
        or not topk_weights.is_contiguous()
        or not topk_ids.is_contiguous()
    ):
        raise ValueError("a_input, topk_weights, and topk_ids must be contiguous")
    _validate_expert_map(expert_map, device=a_input.device)
    _validate_expert_map(output_expert_map, device=a_input.device)
    if output_expert_map is not None and not full_rotation:
        raise ValueError("output_expert_map is only valid with full_rotation")

    m, hidden_size = a_input.shape
    topk = int(topk_ids.shape[1])
    if tuple(topk_weights.shape) != (m, topk):
        raise ValueError(f"topk_weights must have shape {(m, topk)}")
    if int(prepared.hidden_size) != hidden_size:
        raise ValueError("prepared hidden_size does not match a_input")
    if bool(prepared.is_gated) != is_gated:
        raise ValueError("prepared weights do not match activation")
    if tuple(output.shape) != (m, hidden_size):
        raise ValueError(f"output must have shape {(m, hidden_size)}")
    if expert_map is not None and int(expert_map.numel()) < int(prepared.num_experts):
        raise ValueError("expert_map cannot be shorter than the local expert count")
    if output_expert_map is not None and int(output_expert_map.numel()) < int(
        prepared.num_experts
    ):
        raise ValueError(
            "output_expert_map cannot be shorter than the local expert count"
        )
    if full_rotation:
        if apply_router_weight_on_input:
            raise ValueError(
                "full_rotation applies router weights only in the fp32 top-k sum"
            )
        num_local_experts = int(prepared.num_experts)
        intermediate_size_full = int(prepared.intermediate_size)
        # H-side tables may hold a single broadcast row; the kernels index them
        # with a zero expert stride.
        rotation_width = (6 if intermediate_hadamard else 3) * intermediate_size_full
        for name, table, shapes in (
            (
                "suh_gate_table",
                suh_gate_table,
                ((num_local_experts, hidden_size), (1, hidden_size)),
            ),
            (
                "suh_up_table",
                suh_up_table,
                ((num_local_experts, hidden_size), (1, hidden_size)),
            ),
            (
                "svh_table",
                svh_table,
                ((num_local_experts, hidden_size), (1, hidden_size)),
            ),
            (
                "intermediate_rotation_scales",
                intermediate_rotation_scales,
                ((num_local_experts, rotation_width),),
            ),
        ):
            if table is None:
                raise ValueError(f"full_rotation requires {name}")
            if (
                table.dtype != torch.float16
                or table.device != a_input.device
                or tuple(table.shape) not in shapes
                or not table.is_contiguous()
            ):
                raise ValueError(
                    f"{name} must be contiguous fp16 {shapes} on {a_input.device}; "
                    f"got {tuple(table.shape)}/{table.dtype}/{table.device}/"
                    f"contiguous={table.is_contiguous()}"
                )
        required_a = m * topk * hidden_size
        for name, scratch in (
            ("rotation_a_gate", rotation_a_gate),
            ("rotation_a_up", rotation_a_up),
        ):
            if scratch is None:
                raise ValueError(f"full_rotation requires preallocated {name}")
            if (
                scratch.dtype != torch.float16
                or scratch.device != a_input.device
                or not scratch.is_contiguous()
                or int(scratch.numel()) < required_a
            ):
                raise ValueError(
                    f"{name} must be contiguous fp16 on {a_input.device} with at least "
                    f"{required_a} elements"
                )
        assert rotation_a_gate is not None and rotation_a_up is not None
        if (
            not intermediate_hadamard
            and rotation_a_gate.data_ptr() == rotation_a_up.data_ptr()
        ):
            raise ValueError("full_rotation gate/up A scratches must not alias")
    elif any(
        value is not None
        for value in (
            suh_gate_table,
            suh_up_table,
            svh_table,
            rotation_a_gate,
            rotation_a_up,
        )
    ):
        raise ValueError("full-rotation tables/scratch require full_rotation=True")
    layer_idx_int = _validate_activation_amax(
        activation_amax,
        layer_idx=layer_idx,
        num_experts=int(prepared.num_experts),
        device=a_input.device,
    )
    collect_activation_amax = activation_amax is not None

    route_num_experts = (
        int(expert_map.numel()) if expert_map is not None else int(prepared.num_experts)
    )
    block_size_m = _resolve_route_block_size_m(
        m=m,
        topk=topk,
        route_num_experts=route_num_experts,
        planned_block_size_m=route_block_size_m,
        fused_launch=fused_launch,
    )
    if block_size_m not in _ALLOWED_ROUTED_SIZES:
        raise ValueError(f"unsupported W4A16 moe_block_size={block_size_m}")

    props = torch.cuda.get_device_properties(a_input.device)
    sms = int(props.multi_processor_count)
    current_stream = current_cuda_stream()
    stream = current_stream if stream is None else stream
    if (
        route_mode != "packed"
        and (not collect_activation_amax)
        and _small_m_direct_supported(
            m=m,
            hidden_size=hidden_size,
            intermediate_size=int(prepared.intermediate_size),
            num_experts=int(prepared.num_experts),
            topk=topk,
            activation=activation,
            apply_router_weight_on_input=bool(apply_router_weight_on_input),
            swiglu_limit=swiglu_limit,
            swiglu_clamp_after_silu=swiglu_clamp_after_silu,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            element_dtype=element_dtype,
            weight_layout=weight_layout,
            w13_layout=w13_layout,
            scale_format=scale_format,
            expert_map=expert_map,
        )
    ):
        if topk_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("W4A16 small-M direct path requires int32/int64 topk_ids")
        if not topk_ids.is_cuda:
            raise ValueError("W4A16 small-M direct path requires CUDA topk_ids")
        if not intermediate_cache2.is_contiguous() or not output.is_contiguous():
            raise ValueError(
                "W4A16 small-M direct path requires contiguous intermediate_cache2 and output"
            )
        if intermediate_cache2.dtype != a_input.dtype:
            raise TypeError(f"intermediate_cache2 must be {a_input.dtype}")
        if int(prepared.workspace.numel()) < 2:
            raise ValueError("prepared W4A16 workspace is too small for small-M direct")
        intermediate_size = int(prepared.intermediate_size)
        fc2_n_chunks = ((intermediate_size // 2) + 127) // 128
        inter_u32_per_m = fc2_n_chunks * 128 * topk
        inter_u32 = intermediate_cache2.view(-1).view(torch.uint32)
        if int(inter_u32.numel()) < m * inter_u32_per_m:
            raise ValueError(
                "intermediate_cache2 is smaller than the W4A16 small-M direct scratch "
                f"requirement: have_u32={int(inter_u32.numel())}, "
                f"need_u32={m * inter_u32_per_m}"
            )
        micro_w13_scale = getattr(prepared, "micro_w13_scale", None)
        micro_w2_scale = getattr(prepared, "micro_w2_scale", None)
        micro_w13_global = getattr(prepared, "micro_w13_global_scale", None)
        micro_w2_global = getattr(prepared, "micro_w2_global_scale", None)
        if (
            micro_w13_scale is None
            or micro_w2_scale is None
            or micro_w13_global is None
            or micro_w2_global is None
        ):
            raise RuntimeError(
                "W4A16 small-M direct path requires prepared micro scale metadata"
            )
        barrier_count = prepared.workspace[-2:-1]
        barrier_epoch = prepared.workspace[-1:]
        if _small_m_direct_host_barrier_reset_enabled():
            prepared.workspace[-2:].zero_()
        direct_launch = None
        if fused_launch is not None:
            direct_launch = next(
                (
                    launch
                    for launch in fused_launch.small_m_direct_launches
                    if launch.topk_ids_dtype == topk_ids.dtype and launch.m == m
                ),
                None,
            )
            if direct_launch is None or direct_launch.m != m:
                raise RuntimeError(
                    "native W4A16 direct launch was not prepared for this token count and route dtype"
                )
        launch_args = (
            a_input,
            prepared.w13.view(torch.uint8),
            micro_w13_scale,
            micro_w13_global,
            micro_w2_global,
            inter_u32[: m * inter_u32_per_m],
            prepared.w2.view(torch.uint8),
            micro_w2_scale,
            topk_ids,
            topk_weights,
            output,
            barrier_count,
            barrier_epoch,
            m,
            hidden_size,
            intermediate_size,
            int(prepared.num_experts),
            topk,
            activation,
            bool(fast_math),
            scale_format,
            swiglu_limit is not None,
            float(swiglu_limit or 0.0),
            swiglu_clamp_after_silu,
            float(swiglu_alpha),
            float(swiglu_beta),
            w13_layout,
            int(stream),
        )
        if direct_launch is None:
            torch.ops.b12x.w4a16_small_m_direct_launch(*launch_args)
        else:
            _w4a16_small_m_direct_launch_flat(*launch_args, launcher=direct_launch)
        return output

    if isinstance(fused_launch, W4A16SmallMDirectCompileResult):
        raise RuntimeError("execution does not match its prepared native W4A16 route")

    # TC decode keeps the top-k reduction inside the fused tensor-core kernel.
    # Direct-route emit hooks resolve global expert IDs against compact tiers
    # without materializing remapped IDs or masked router weights.
    # Prepared launches retain this contract in tc_decode_fused_sum.
    preplanned_tc_decode = bool(getattr(fused_launch, "tc_decode_fused_sum", False))
    preplanned_prefill_fused_sum = bool(
        getattr(fused_launch, "prefill_fused_sum_fp32", False)
    )
    prefer_tc_decode = route_mode == "direct" or (
        route_mode == "auto"
        and _w4a16_tc_decode_preferred(
            m=m,
            topk=topk,
            num_experts=int(prepared.num_experts),
            sms=sms,
        )
    )
    use_tc_decode = bool(
        (not collect_activation_amax)
        and route_mode != "packed"
        and (fused_launch is None or preplanned_tc_decode)
        and weight_layout in {"packed", "iq2_xs", "iq2_xxs", "q8_0"}
        and (is_gated or (weight_layout in BLOCK_CODECS and activation == "relu2"))
        and element_dtype == "bf16"
        and topk_ids.dtype in (torch.int32, torch.int64)
        and topk_ids.is_cuda
        and prefer_tc_decode
    )
    if use_tc_decode and topk_ids.dtype != torch.int32:
        # The inline direct-topk route path needs int32 route indices.
        topk_ids = topk_ids.to(torch.int32)

    mapped_direct = expert_map is not None
    direct_m_cap = (
        _W4A16_SMALL_M_DIRECT_MAX_M if mapped_direct else _MAX_DIRECT_TOPK_ROUTE_M
    )
    direct_layout_ok = weight_layout in {"packed", "iq2_xs", "iq2_xxs", "q8_0"} or (
        mapped_direct and full_rotation and weight_layout == "trellis_t256"
    )
    direct_topk_eligible = (
        (not collect_activation_amax)
        and route_mode != "packed"
        and (m <= direct_m_cap or use_tc_decode)
        and direct_layout_ok
    )
    use_direct_topk_routes = bool(
        direct_topk_eligible
        and topk_ids.dtype == torch.int32
        and topk_ids.is_cuda
        and (
            fused_launch is None
            or bool(getattr(fused_launch, "direct_topk_routes", False))
        )
        and (
            fused_launch is None
            or bool(getattr(fused_launch, "use_expert_map", False)) == mapped_direct
        )
    )
    if (
        bool(getattr(fused_launch, "direct_topk_routes", False))
        and not use_direct_topk_routes
    ):
        raise RuntimeError(
            "preplanned W4A16 direct top-k routing requires a matching small-M "
            "CUDA int32 route/map contract"
        )

    # TC-decode requires the inline direct-topk route path (no route-pack).
    use_tc_decode = bool(use_tc_decode and use_direct_topk_routes)

    if route_mode == "direct" and not use_direct_topk_routes:
        raise RuntimeError(
            "planned W4A16 direct routing is unsupported for this launch shape"
        )

    prefill_fused_sum_requested = (
        preplanned_prefill_fused_sum
        if fused_launch is not None
        else prefill_fused_sum_enabled()
    )
    use_prefill_fused_sum = prefill_fused_sum_eligible(
        dtype=element_dtype,
        m=int(fused_launch.size_m) if fused_launch is not None else m,
        full_rotation=full_rotation,
        weight_layout=weight_layout,
        collect_activation_amax=collect_activation_amax,
        enabled=prefill_fused_sum_requested,
    )
    use_fused_topk_sum = bool(use_tc_decode or use_prefill_fused_sum)

    if use_prefill_fused_sum:
        required_accum_elements = (
            int(fused_launch.size_m) if fused_launch is not None else int(m)
        ) * hidden_size
        if (
            prefill_sum_accum is None
            or prefill_sum_accum.dtype != torch.float32
            or prefill_sum_accum.device != a_input.device
            or not prefill_sum_accum.is_contiguous()
            or prefill_sum_accum.numel() < required_accum_elements
        ):
            raise ValueError(
                "W4A16 prefill fused sum requires a contiguous FP32 accumulator "
                f"with at least {required_accum_elements} elements on "
                f"{a_input.device}"
            )

    # Prepared TC decode owns the per-token reduction and omits the separate
    # top-k sum launch. A mismatched routing contract would corrupt output.
    if preplanned_tc_decode and not use_tc_decode:
        raise RuntimeError(
            "preplanned TC-decode W4A16 launch requires small-M packed bf16 "
            f"decode (m <= {_TC_DECODE_MAX_M}, cuda int32/int64 topk_ids)"
        )
    if preplanned_prefill_fused_sum and not use_prefill_fused_sum:
        raise RuntimeError(
            "preplanned W4A16 prefill fused-sum launch requires the enabled "
            "large-M packed or modelopt BF16 route-reduction contract"
        )

    route_slots_for_scratch = int(m) * int(topk) * int(block_size_m)
    required_m_blocks = int(m) * int(topk) if use_direct_topk_routes else 0
    if fused_launch is not None and not use_direct_topk_routes:
        route_slots_capacity = max_packed_route_slots(
            int(fused_launch.size_m) * int(topk),
            int(block_size_m),
            route_num_experts,
        )
        if route_pack_launches is not None:
            # A retained small-M GEMM can use a route packer prepared for a
            # larger capacity. Its kernels initialize every planned route slot;
            # do not shrink those views to the selected GEMM's live-row bucket.
            route_slots_capacity = max(
                route_slots_capacity, route_pack_launches.max_packed_routes
            )
        route_blocks_capacity = (route_slots_capacity + int(block_size_m) - 1) // int(
            block_size_m
        )
        if packed_route_indices is not None:
            if int(packed_route_indices.numel()) < route_slots_capacity:
                raise ValueError(
                    "packed_route_indices is smaller than the selected W4A16 launch capacity"
                )
            packed_route_indices = packed_route_indices[:route_slots_capacity]
        if block_expert_ids is not None:
            if int(block_expert_ids.numel()) < route_blocks_capacity:
                raise ValueError(
                    "block_expert_ids is smaller than the selected W4A16 launch capacity"
                )
            block_expert_ids = block_expert_ids[:route_blocks_capacity]
    if use_direct_topk_routes:
        packed_route_indices = topk_ids.view(-1)
        if block_expert_ids is None:
            block_expert_ids = packed_route_indices
        if packed_route_count is None:
            packed_route_count = packed_route_indices
    else:
        if (
            fused_launch is not None
            and not full_rotation
            and route_pack_launches is None
        ):
            raise RuntimeError(
                "prepared W4A16 packed route execution is missing route-pack programs"
            )
        packed_route_indices, block_expert_ids, packed_route_count = (
            pack_topk_routes_by_expert(
                topk_ids,
                block_size_m,
                route_num_experts,
                expert_map=expert_map,
                packed_route_indices=packed_route_indices,
                block_expert_ids=block_expert_ids,
                packed_route_count=packed_route_count,
                expert_offsets=expert_offsets,
                expert_counts=expert_counts,
                launches=route_pack_launches,
                stream=stream,
            )
        )
        route_slots_for_scratch = int(packed_route_indices.numel())
        required_m_blocks = int(block_expert_ids.numel())

    max_shared_mem = int(
        getattr(props, "shared_memory_per_block_optin", _DEFAULT_MAX_SHARED_MEM)
    )
    buffer_plan = plan_w4a16_buffers(
        prepared,
        m=m,
        topk=topk,
        route_num_experts=route_num_experts,
        sms=sms,
        dtype=(prepared_dtype if full_rotation else a_input.dtype),
        full_rotation=full_rotation,
        block_size_m=block_size_m,
        weight_layout=weight_layout,
        collect_activation_amax=collect_activation_amax,
        prefill_fused_sum=use_prefill_fused_sum,
    )
    intermediate_size = int(prepared.intermediate_size)
    fc1_cols = buffer_plan.fc1_cols

    if intermediate_cache13.numel() < buffer_plan.intermediate_cache13_elements:
        raise ValueError(
            f"intermediate_cache13 has {intermediate_cache13.numel()} elements; "
            f"need at least {buffer_plan.intermediate_cache13_elements}"
        )
    if intermediate_cache2.numel() < buffer_plan.intermediate_cache2_elements:
        raise ValueError(
            f"intermediate_cache2 has {intermediate_cache2.numel()} elements; "
            f"need at least {buffer_plan.intermediate_cache2_elements}"
        )
    cache_dtype = prepared_dtype if full_rotation else a_input.dtype
    if (
        intermediate_cache13.dtype != cache_dtype
        or intermediate_cache2.dtype != cache_dtype
    ):
        raise TypeError(f"intermediate caches must be {cache_dtype}")
    if (
        not intermediate_cache13.is_contiguous()
        or not intermediate_cache2.is_contiguous()
        or not output.is_contiguous()
    ):
        raise ValueError("intermediate caches and output must be contiguous")

    intermediate_cache13_flat = intermediate_cache13.view(-1)
    intermediate_cache2_flat = intermediate_cache2.view(-1)

    if int(prepared.workspace.numel()) < sms * 4 + 2:
        raise ValueError("prepared W4A16 workspace is too small for fused FC1+FC2")
    if fused_launch is None:
        fused = compile_w4a16_fused_moe(
            size_m=m,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=int(prepared.num_experts),
            top_k=topk,
            activation=activation,
            apply_router_weight_on_input=bool(apply_router_weight_on_input),
            zero_fc2_output=(
                expert_map is not None
                and not full_rotation
                and not use_direct_topk_routes
            ),
            moe_block_size=block_size_m,
            max_m_blocks=int(required_m_blocks),
            element_dtype=element_dtype,
            fast_math=bool(fast_math),
            sms=sms,
            max_shared_mem=max_shared_mem,
            swiglu_limit=swiglu_limit,
            swiglu_clamp_after_silu=swiglu_clamp_after_silu,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            weight_layout=weight_layout,
            scale_format=scale_format,
            w13_layout=w13_layout,
            trellis_bits=trellis_bits,
            trellis_codebook=trellis_codebook,
            fc1_trellis_pair_kind=fc1_trellis_pair_kind,
            fc2_trellis_pair_kind=fc2_trellis_pair_kind,
            direct_topk_routes=use_direct_topk_routes,
            use_expert_map=mapped_direct and use_direct_topk_routes,
            tc_decode_fused_sum=use_tc_decode,
            prefill_fused_sum_fp32=use_prefill_fused_sum,
            collect_activation_amax=collect_activation_amax,
            intermediate_rotation=intermediate_rotation_scales is not None,
            full_rotation=full_rotation,
            intermediate_hadamard=intermediate_hadamard,
            rotation_input_dtype=rotation_input_dtype,
            broadcast_suh=full_rotation and suh_gate_table.numel() == hidden_size,
            force_tile_config=prepared_tile_config,
            _require_cached=_w4a16_stream_is_capturing(
                stream,
                current_stream=current_stream,
            ),
        )
    else:
        if int(fused_launch.size_m) < m:
            raise RuntimeError(
                "preplanned W4A16 fused MoE launch capacity is smaller than requested rows: "
                f"requested={m}, planned={int(fused_launch.size_m)}"
            )
        expected_fused = (
            hidden_size,
            intermediate_size,
            int(prepared.num_experts),
            topk,
            activation,
            bool(apply_router_weight_on_input),
            (
                expert_map is not None
                and not full_rotation
                and not use_direct_topk_routes
            ),
            element_dtype,
            bool(fast_math),
            swiglu_limit,
            swiglu_clamp_after_silu,
            float(swiglu_alpha),
            float(swiglu_beta),
            weight_layout,
            scale_format,
            w13_layout,
            trellis_bits,
            trellis_codebook,
            fc1_trellis_pair_kind,
            fc2_trellis_pair_kind,
            bool(use_direct_topk_routes),
            mapped_direct and use_direct_topk_routes,
            bool(use_tc_decode),
            bool(use_prefill_fused_sum),
            bool(collect_activation_amax),
            block_size_m,
            bool(intermediate_rotation_scales is not None),
            dual_a_required,
            full_rotation,
            intermediate_hadamard,
            rotation_input_dtype,
        )
        actual_fused = (
            int(fused_launch.hidden_size),
            int(fused_launch.intermediate_size),
            int(fused_launch.num_experts),
            int(fused_launch.top_k),
            fused_launch.activation,
            bool(fused_launch.apply_router_weight_on_input),
            bool(fused_launch.zero_fc2_output),
            fused_launch.element_dtype,
            bool(fused_launch.fast_math),
            fused_launch.swiglu_limit,
            fused_launch.swiglu_clamp_after_silu,
            float(fused_launch.swiglu_alpha),
            float(fused_launch.swiglu_beta),
            getattr(fused_launch, "weight_layout", "packed"),
            getattr(fused_launch, "scale_format", "e4m3_k16"),
            getattr(
                fused_launch,
                "w13_layout",
                "w13"
                if getattr(fused_launch, "weight_layout", "packed") == "modelopt"
                else "packed",
            ),
            int(getattr(fused_launch, "trellis_bits", 3)),
            str(
                getattr(
                    fused_launch,
                    "trellis_codebook",
                    LUT_E4M3,
                )
            ).lower(),
            getattr(fused_launch, "fc1_trellis_pair_kind", None),
            getattr(fused_launch, "fc2_trellis_pair_kind", None),
            bool(getattr(fused_launch, "direct_topk_routes", False)),
            bool(getattr(fused_launch, "use_expert_map", False)),
            bool(getattr(fused_launch, "tc_decode_fused_sum", False)),
            bool(getattr(fused_launch, "prefill_fused_sum_fp32", False)),
            bool(getattr(fused_launch, "collect_activation_amax", False)),
            int(fused_launch.moe_block_size),
            bool(getattr(fused_launch, "intermediate_rotation", False)),
            bool(getattr(fused_launch, "dual_a", False)),
            bool(getattr(fused_launch, "full_rotation", False)),
            bool(getattr(fused_launch, "intermediate_hadamard", False)),
            getattr(fused_launch, "rotation_input_dtype", fused_launch.element_dtype),
        )
        if actual_fused != expected_fused or int(fused_launch.max_m_blocks) < int(
            required_m_blocks
        ):
            raise RuntimeError(
                "preplanned W4A16 fused MoE launch does not match requested contract: "
                f"requested={expected_fused + (int(required_m_blocks),)}, "
                f"planned={actual_fused + (int(fused_launch.max_m_blocks),)}"
            )
        fused = fused_launch
    capacity_m = int(fused.size_m)
    capacity_routed_rows = capacity_m * topk
    required_cache13_elements = (
        capacity_routed_rows * fc1_cols
        if use_prefill_fused_sum
        else capacity_routed_rows * max(fc1_cols, hidden_size)
    )
    if intermediate_cache13_flat.numel() < required_cache13_elements:
        raise ValueError(
            "intermediate_cache13 is smaller than the selected W4A16 launch capacity: "
            f"capacity_rows={capacity_m}, topk={topk}, "
            f"available_elements={intermediate_cache13_flat.numel()}, "
            f"required_elements={required_cache13_elements}, "
            f"fused_topk_sum={use_fused_topk_sum}, "
            f"prefill_fused_sum={use_prefill_fused_sum}, "
            f"collect_activation_amax={collect_activation_amax}, "
            f"full_rotation={full_rotation}, weight_layout={weight_layout}, "
            f"element_dtype={element_dtype}"
        )
    if intermediate_cache2_flat.numel() < capacity_routed_rows * intermediate_size:
        raise ValueError(
            "intermediate_cache2 is smaller than the selected W4A16 launch capacity: "
            f"capacity_rows={capacity_m}, topk={topk}"
        )
    fc1_store_cols = (
        max(fc1_cols, hidden_size)
        if use_tc_decode and weight_layout in BLOCK_CODECS
        else fc1_cols
    )
    fc1_out = intermediate_cache13_flat[: capacity_routed_rows * fc1_store_cols]
    activated = intermediate_cache2_flat[: capacity_routed_rows * intermediate_size]
    if use_prefill_fused_sum:
        assert prefill_sum_accum is not None
        fc2_out = prefill_sum_accum[: capacity_m * hidden_size]
    elif use_tc_decode:
        # The fused kernel owns per-token reduction and output initialization.
        fc2_out = output.view(-1)
    else:
        fc2_out = intermediate_cache13_flat[: capacity_routed_rows * hidden_size]
    fc1_scratch = _get_c_tmp(
        packed_gemm_scratch_elements(
            size_n=fc1_cols,
            route_slots=int(route_slots_for_scratch),
            moe_block_size=block_size_m,
            sms=sms,
            weight_layout=weight_layout,
        ),
        device=a_input.device,
        scratch=fc1_c_tmp,
    )
    fc2_scratch = _get_c_tmp(
        packed_gemm_scratch_elements(
            size_n=hidden_size,
            route_slots=int(route_slots_for_scratch),
            moe_block_size=block_size_m,
            sms=sms,
            weight_layout=weight_layout,
        ),
        device=a_input.device,
        scratch=fc2_c_tmp,
    )
    if weight_layout == "modelopt":
        w13_arg = prepared.w13.view(torch.uint8).view(-1)
        w2_arg = prepared.w2.view(torch.uint8).view(-1)
    else:
        w13_arg = prepared.w13.view(torch.int32).view(-1)
        w2_arg = prepared.w2.view(torch.int32).view(-1)
    w13_scale_or_pair_modes = (
        fc1_trellis_pair_modes
        if fc1_trellis_pair_kind in {"PDYNAMIC", "P33_P43"}
        else prepared.w13_scale.view(torch.uint8).view(torch.int32).view(-1)
    )
    w2_scale_or_pair_modes = (
        fc2_trellis_pair_modes
        if fc2_trellis_pair_kind in {"PDYNAMIC", "P33_P43"}
        else prepared.w2_scale.view(torch.uint8).view(torch.int32).view(-1)
    )
    assert w13_scale_or_pair_modes is not None
    assert w2_scale_or_pair_modes is not None
    launch_common = (
        a_input,
        w13_arg,
        w2_arg,
        fc1_out,
        activated,
        fc2_out,
        w13_scale_or_pair_modes,
        w2_scale_or_pair_modes,
        prepared.w13_global_scale,
        prepared.w2_global_scale,
        packed_route_indices,
        block_expert_ids,
        packed_route_count,
    )
    launch_tail = (
        topk_weights,
        fc1_scratch,
        fc2_scratch,
        prepared.workspace,
        m,
        capacity_m,
        hidden_size,
        intermediate_size,
        int(prepared.num_experts),
        topk,
        activation,
        bool(apply_router_weight_on_input),
        bool(fused.zero_fc2_output),
        block_size_m,
        int(fused.max_m_blocks),
        element_dtype,
        bool(fast_math),
        sms,
        max_shared_mem,
        swiglu_limit is not None,
        float(swiglu_limit or 0.0),
        swiglu_clamp_after_silu,
        float(swiglu_alpha),
        float(swiglu_beta),
        weight_layout,
        scale_format,
        w13_layout,
        int(fused.fc1_tile_k),
        int(fused.fc1_tile_n),
        int(fused.fc2_tile_k),
        int(fused.fc2_tile_n),
    )
    _intermediate_rotation = intermediate_rotation_scales is not None
    if _intermediate_rotation and (
        collect_activation_amax
        or use_tc_decode
        or (use_direct_topk_routes and not full_rotation)
        or weight_layout != "trellis_t256"
    ):
        raise ValueError(
            "intermediate_rotation_scales requires the trellis_t256 fused path "
            "(no calibration / tc-decode; direct routing requires full_rotation)"
        )
    if _intermediate_rotation:
        need = (
            int(prepared.num_experts)
            * (6 if intermediate_hadamard else 3)
            * intermediate_size
            if full_rotation
            else int(m) * topk * 3 * intermediate_size
        )
        rot_arg = intermediate_rotation_scales.reshape(-1)
        if int(rot_arg.numel()) < need or rot_arg.dtype != torch.float16:
            raise ValueError(
                "intermediate_rotation_scales must be fp16 with >= "
                f"{need} elements ({'experts' if full_rotation else 'routes'}"
                f"*{6 if intermediate_hadamard and full_rotation else 3}*intermediate); got "
                f"numel={int(rot_arg.numel())} dtype={rot_arg.dtype}"
            )
        rot_arg = rot_arg[:need]
    else:
        rot_arg = None
    if collect_activation_amax:
        assert activation_amax is not None
        assert layer_idx_int is not None
        torch.ops.b12x.w4a16_fused_moe_calibrated_launch(
            *launch_common,
            activation_amax,
            int(layer_idx_int),
            *launch_tail,
            int(stream),
        )
    elif (
        fused_launch is not None
        or _intermediate_rotation
        or weight_layout in {"trellis_t256", "iq2_xs", "iq2_xxs", "q8_0"}
        or (mapped_direct and use_direct_topk_routes)
        or use_prefill_fused_sum
    ):
        # Native t256 bypasses the registered torch op so its shape-derived
        # bitrate reaches compilation without widening the stable public op ABI.
        # Other production layouts keep the registered path byte-identical.
        (
            _rc_a,
            _rc_w13,
            _rc_w2,
            _rc_fc1,
            _rc_act,
            _rc_fc2,
            _rc_w13s,
            _rc_w2s,
            _rc_w13g,
            _rc_w2g,
            _rc_pri,
            _rc_bei,
            _rc_prc,
        ) = launch_common
        (
            _lt_tw,
            _lt_fc1s,
            _lt_fc2s,
            _lt_ws,
            _lt_m,
            _lt_capm,
            _lt_h,
            _lt_i,
            _lt_ne,
            _lt_tk,
            _lt_act,
            _lt_arwi,
            _lt_zfo,
            _lt_bsm,
            _lt_mmb,
            _lt_edt,
            _lt_fm,
            _lt_sms,
            _lt_msm,
            _lt_hsl,
            _lt_slv,
            _lt_cas,
            _lt_sa,
            _lt_sb,
            _lt_wl,
            _lt_sf,
            _lt_w13l,
            _lt_fc1tk,
            _lt_fc1tn,
            _lt_fc2tk,
            _lt_fc2tn,
        ) = launch_tail
        launch_a = rotation_a_gate if full_rotation else _rc_a
        launch_a_up = rotation_a_up if full_rotation else a_input_up
        # Intermediate-Hadamard gate/up matrices consume the same transformed input. Reuse
        # the gate buffer instead of materializing an identical second row.
        if full_rotation and intermediate_hadamard:
            launch_a_up = rotation_a_gate
        assert launch_a is not None
        _w4a16_fused_moe_launch_flat(
            a_input=launch_a,
            w13_arg=_rc_w13,
            w2_arg=_rc_w2,
            fc1_out=_rc_fc1,
            activated=_rc_act,
            fc2_out=_rc_fc2,
            w13_scale_i32=_rc_w13s,
            w2_scale_i32=_rc_w2s,
            w13_global_scale=_rc_w13g,
            w2_global_scale=_rc_w2g,
            packed_route_indices=_rc_pri,
            block_expert_ids=_rc_bei,
            packed_route_count=_rc_prc,
            activation_amax=None,
            layer_idx=0,
            topk_weights=_lt_tw,
            fc1_scratch=_lt_fc1s,
            fc2_scratch=_lt_fc2s,
            workspace=_lt_ws,
            m=_lt_m,
            size_m=_lt_capm,
            hidden_size=_lt_h,
            intermediate_size=_lt_i,
            num_experts=_lt_ne,
            topk=_lt_tk,
            activation=_lt_act,
            apply_router_weight_on_input=_lt_arwi,
            zero_fc2_output=_lt_zfo,
            moe_block_size=_lt_bsm,
            max_m_blocks=_lt_mmb,
            element_dtype=_lt_edt,
            fast_math=_lt_fm,
            sms=_lt_sms,
            max_shared_mem=_lt_msm,
            has_swiglu_limit=_lt_hsl,
            swiglu_limit_value=_lt_slv,
            swiglu_clamp_after_silu=_lt_cas,
            swiglu_alpha=_lt_sa,
            swiglu_beta=_lt_sb,
            weight_layout=_lt_wl,
            scale_format=_lt_sf,
            w13_layout=_lt_w13l,
            fc1_tile_k=_lt_fc1tk,
            fc1_tile_n=_lt_fc1tn,
            fc2_tile_k=_lt_fc2tk,
            fc2_tile_n=_lt_fc2tn,
            direct_topk_routes=bool(use_direct_topk_routes),
            tc_decode_fused_sum=bool(use_tc_decode),
            prefill_fused_sum_fp32=bool(use_prefill_fused_sum),
            collect_activation_amax=False,
            stream_int=int(stream),
            expert_map=expert_map if use_direct_topk_routes else None,
            rot_scales=rot_arg,
            intermediate_rotation=_intermediate_rotation,
            a_input_up=launch_a_up,
            trellis_bits=trellis_bits,
            trellis_codebook=trellis_codebook,
            fc1_trellis_pair_kind=fc1_trellis_pair_kind,
            fc2_trellis_pair_kind=fc2_trellis_pair_kind,
            full_rotation=full_rotation,
            intermediate_hadamard=intermediate_hadamard,
            rotation_input=_rc_a,
            suh_gate_table=suh_gate_table,
            suh_up_table=suh_up_table,
            launcher=fused,
        )
    else:
        torch.ops.b12x.w4a16_fused_moe_launch(
            *launch_common,
            *launch_tail,
            bool(use_direct_topk_routes),
            bool(use_tc_decode),
            int(stream),
        )

    if use_prefill_fused_sum:
        assert prefill_sum_accum is not None
        output.copy_(prefill_sum_accum[: m * hidden_size].view(m, hidden_size))
        return output
    if use_tc_decode:
        # FC2 already wrote the top-k-summed result into `output`.
        return output

    sum_expert_map = output_expert_map if output_expert_map is not None else expert_map
    sum_uses_map = sum_expert_map is not None
    if topk_sum_launch is not None:
        expected_sum = (
            topk,
            hidden_size,
            full_rotation,
            intermediate_hadamard,
            int(prepared.num_experts) if full_rotation or sum_uses_map else 0,
            0 if not sum_uses_map else int(sum_expert_map.numel()),
            topk_ids.dtype,
            sum_uses_map,
        )
        actual_sum = (
            int(topk_sum_launch.topk),
            int(topk_sum_launch.hidden_size),
            bool(getattr(topk_sum_launch, "full_rotation", False)),
            bool(getattr(topk_sum_launch, "intermediate_hadamard", False)),
            int(getattr(topk_sum_launch, "num_experts", 0)),
            int(getattr(topk_sum_launch, "route_num_experts", 0)),
            getattr(topk_sum_launch, "route_ids_dtype", torch.int32),
            bool(getattr(topk_sum_launch, "use_expert_map", False)),
        )
        if actual_sum != expected_sum:
            raise RuntimeError(
                "preplanned W4A16 top-k sum launch does not match requested contract: "
                f"requested={expected_sum}, planned={actual_sum}"
            )
    if (
        topk_sum_launch is not None
        or full_rotation
        or sum_uses_map
        or _FP32_TOPK_WEIGHTS
    ):
        if full_rotation:
            assert svh_table is not None
        _w4a16_topk_sum_launch_flat(
            fc2_out,
            output,
            m,
            topk,
            hidden_size,
            element_dtype,
            int(stream),
            full_rotation=full_rotation,
            intermediate_hadamard=intermediate_hadamard,
            num_experts=int(prepared.num_experts),
            topk_weights=(
                topk_weights if (full_rotation or _FP32_TOPK_WEIGHTS) else None
            ),
            route_expert_ids=topk_ids,
            expert_map=sum_expert_map if sum_uses_map else None,
            svh_table=svh_table if full_rotation else None,
            launcher=topk_sum_launch,
        )
    else:
        torch.ops.b12x.w4a16_topk_sum_launch(
            fc2_out,
            output,
            topk_ids,
            int(prepared.num_experts),
            m,
            topk,
            hidden_size,
            element_dtype,
            int(stream),
        )
    return output


def build_w4a16_tier_local_map(
    tier0_global_ids,
    tier1_global_ids,
    *,
    map_slots: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build the int32 [map_slots] global-expert descriptor table.

    Entry g = (tier << 8) | local_expert_id for a mapped global expert id g,
    -1 for unmapped. tierN_global_ids[i] is the global id of that tier's local
    expert i, i.e. the order the tier's weights were packed in.
    """

    map_slots = int(map_slots)
    table = torch.full((map_slots,), -1, dtype=torch.int32)
    seen: set[int] = set()
    for tier, ids in ((0, tier0_global_ids), (1, tier1_global_ids)):
        ids_list = [int(v) for v in ids]
        if len(ids_list) > 256:
            raise ValueError(
                f"tier {tier} has {len(ids_list)} experts; the descriptor "
                "local-id field is 8 bits"
            )
        for local, gid in enumerate(ids_list):
            if gid < 0 or gid >= map_slots:
                raise ValueError(
                    f"tier {tier} local expert {local} has global id {gid} "
                    f"outside [0, {map_slots})"
                )
            if gid in seen:
                raise ValueError(f"global expert id {gid} is mapped twice")
            seen.add(gid)
            table[gid] = (tier << 8) | local
    if device is not None:
        table = table.to(device)
    return table.contiguous()
