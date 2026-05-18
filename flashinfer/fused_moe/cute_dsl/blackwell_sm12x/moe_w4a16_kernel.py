"""CuTeDSL W4A16 NVFP4/BF16 W4A16 MoE kernels."""

from __future__ import annotations

from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cutlass_dsl import Int32, Uint32

from .moe_w4a16_fp4_helpers import (
    atomic_add_global_i32,
    bf16_mma_m16n8k16_f32,
    bf16_mma_rhs_fragments_as_mma_a_m16n8k16_f32,
    bfloat2_broadcast_lane,
    bfloat2_mul,
    bfloat2_to_float2_scaled,
    broadcast_f32_to_half2,
    broadcast_f32_to_bfloat2,
    cp_async4_shared_global,
    cp_async4_shared_global_pred,
    f16_mma_m16n8k16_f32,
    f16_mma_rhs_fragments_as_mma_a_m16n8k16_f32,
    half2_to_float2_scaled,
    get_ptr_as_int64,
    half2_mul,
    ld_global_acquire_i32,
    ld_global_v4_f32,
    ld_shared_i32_relaxed,
    ld_shared_u32,
    ld_shared_v2_u32,
    ld_shared_v4_f32,
    ld_shared_v4_u32,
    ldmatrix_m8n8x4_b16,
    ldmatrix_m8n8x2_b16,
    packed_dequant_e2m1x4_to_bfloat2x2,
    packed_dequant_e2m1x4_to_half2x2,
    packed_dequant_e4m3x4_to_bfloat2x2,
    packed_dequant_e4m3x4_to_half2x2,
    pack_f32x2_to_bfloat2,
    pack_f32x2_to_f16x2,
    red_add_global_release_i32,
    shared_ptr_to_u32,
    st_global_v4_u32,
    st_global_i32,
    st_global_v4_f32,
    st_shared_bf16_from_f32,
    st_shared_f16_from_f32,
    st_shared_i32,
    st_shared_u32,
    st_shared_v4_f32,
    threadfence,
)
from flashinfer.cute_dsl.utils import current_cuda_stream
from .moe_w4a16_route_pack import (
    pack_topk_routes_by_expert as _pack_topk_routes_by_expert,
)
from .moe_w4a16_host import (
    _W4A16_ALLOWED_ROUTED_SIZES,
    packed_gemm_scratch_elements,
    plan_w4a16_buffers,
    select_route_block_size_m,
    validate_activation,
)


def raise_if_kernel_resolution_frozen(*args, **kwargs) -> None:
    del args, kwargs


_ALLOWED_ROUTED_SIZES = _W4A16_ALLOWED_ROUTED_SIZES
_PACK_FACTOR = 8
_STAGES = 4
_DEVICE_MAX_REG_BYTES = 255 * 1024
_DEFAULT_MAX_SHARED_MEM = 101_376

# The W4A16 launch model chooses blocks/SM from static resource usage
# for each specialization.  These register counts were measured from the local
# SM121 JIT output and keep launch occupancy stable across refactors.
_W4A16_REGS_SM121 = {
    (256, 1, 8, 8, True): 118,
    (128, 1, 4, 8, True): 118,
    (128, 1, 8, 4, True): 120,
    (256, 1, 8, 8, False): 158,
    (128, 1, 4, 8, False): 154,
    (128, 1, 8, 4, False): 143,
    (256, 2, 16, 4, False): 212,
    (128, 2, 4, 8, False): 215,
    (128, 2, 8, 4, False): 214,
    (256, 3, 16, 4, False): 249,
    (128, 3, 4, 8, False): 249,
    (128, 3, 8, 4, False): 250,
    (256, 4, 16, 4, False): 255,
    (128, 4, 4, 8, False): 255,
    (128, 4, 8, 4, False): 255,
}
_SMALL_BATCH_TILE_CONFIGS = (
    (128, 128, 256),
    (64, 128, 128),
    (128, 64, 128),
)
_LARGE_BATCH_TILE_CONFIGS = (
    (64, 256, 256),
    (64, 128, 128),
    (128, 64, 128),
)


def _covering_count(total: int, quantum: int) -> int:
    return (total + quantum - 1) // quantum


def _w4a16_num_regs(
    *,
    cta_threads: int,
    cta_m_blocks: int,
    cta_n_blocks: int,
    cta_k_blocks: int,
    uses_m_block_8: bool,
) -> int:
    key = (
        int(cta_threads),
        int(cta_m_blocks),
        int(cta_n_blocks),
        int(cta_k_blocks),
        bool(uses_m_block_8),
    )
    try:
        return _W4A16_REGS_SM121[key]
    except KeyError as exc:
        raise ValueError(
            f"missing W4A16 register count for NVFP4 BF16 specialization {key}"
        ) from exc


def _shared_memory_footprint(
    *,
    cta_m_blocks: int,
    tile_n: int,
    tile_k: int,
) -> int:
    cta_m = int(cta_m_blocks) * 16
    cta_n = int(tile_n)
    cta_k = int(tile_k)
    sh_block_meta_size = cta_m * 16
    sh_a_size = _STAGES * (cta_m * cta_k) * 2
    sh_b_size = _STAGES * (cta_k * cta_n // _PACK_FACTOR) * 4
    sh_red_size = cta_m * (cta_n + 8) * 2
    sh_bias_size = cta_n * 2
    tmp_size = min(sh_b_size, sh_red_size) + sh_bias_size
    tmp_size = max(max(sh_b_size, sh_red_size), tmp_size)
    sh_s_size = _covering_count(cta_k, 16) * cta_n * 2 * _STAGES
    return tmp_size + sh_a_size + sh_s_size + sh_block_meta_size


def _determine_blocks_per_sm(
    *,
    problem_m: int,
    problem_n: int,
    top_k: int,
    cta_threads: int,
    cta_m_blocks: int,
    tile_n: int,
    tile_k: int,
    uses_m_block_8: bool,
    sms: int,
    max_shared_mem: int,
) -> int:
    num_regs = _w4a16_num_regs(
        cta_threads=cta_threads,
        cta_m_blocks=cta_m_blocks,
        cta_n_blocks=tile_n // 16,
        cta_k_blocks=tile_k // 16,
        uses_m_block_8=uses_m_block_8,
    )
    register_bytes = max(num_regs, 1) * int(cta_threads) * 4
    smem_bytes = _shared_memory_footprint(
        cta_m_blocks=cta_m_blocks,
        tile_n=tile_n,
        tile_k=tile_k,
    )
    blocks_per_sm_limit = min(
        _DEVICE_MAX_REG_BYTES // register_bytes,
        int(max_shared_mem) // (smem_bytes + 1536),
    )
    if cta_m_blocks == 1:
        blocks_per_sm_limit = max(min(blocks_per_sm_limit, 4), 1)
    else:
        blocks_per_sm_limit = max(min(blocks_per_sm_limit, 2), 1)

    work_cta_count = (int(problem_n) // int(tile_n)) * int(problem_m) * int(top_k) * 4
    if work_cta_count < int(sms) * blocks_per_sm_limit:
        blocks_per_sm_limit = max(work_cta_count // int(sms), 1)
    return int(blocks_per_sm_limit)


def _candidate_tile_fits(
    *,
    problem_n: int,
    problem_k: int,
    cta_m_blocks: int,
    tile_n: int,
    tile_k: int,
    cta_threads: int,
    max_shared_mem: int,
) -> bool:
    if int(tile_k) == -1 or int(tile_n) == -1 or int(cta_threads) == -1:
        return False
    if int(problem_k) % int(tile_k) != 0 or int(problem_n) % int(tile_n) != 0:
        return False
    if int(tile_n) < 64 or int(tile_k) < 64 or int(cta_threads) < 128:
        return False
    smem_bytes = _shared_memory_footprint(
        cta_m_blocks=cta_m_blocks,
        tile_n=tile_n,
        tile_k=tile_k,
    )
    return smem_bytes <= int(max_shared_mem)


def _select_tile_config(
    *,
    problem_m: int,
    problem_n: int,
    problem_k: int,
    top_k: int,
    moe_block_size: int,
    sms: int,
    max_shared_mem: int,
    required_cta_threads: int | None = None,
) -> tuple[int, int, int, int]:
    cta_m_blocks = _covering_count(moe_block_size, 16)
    uses_m_block_8 = moe_block_size == 8
    configs = (
        _LARGE_BATCH_TILE_CONFIGS if cta_m_blocks > 1 else _SMALL_BATCH_TILE_CONFIGS
    )
    best_blocks_per_sm = 0
    best_tile_config: tuple[int, int, int, int] | None = None
    for tile_k, tile_n, cta_threads in configs:
        if required_cta_threads is not None and int(cta_threads) != int(
            required_cta_threads
        ):
            continue
        if not _candidate_tile_fits(
            problem_n=problem_n,
            problem_k=problem_k,
            cta_m_blocks=cta_m_blocks,
            tile_n=tile_n,
            tile_k=tile_k,
            cta_threads=cta_threads,
            max_shared_mem=int(max_shared_mem) - 512,
        ):
            continue
        blocks_per_sm_limit = _determine_blocks_per_sm(
            problem_m=problem_m,
            problem_n=problem_n,
            top_k=top_k,
            cta_threads=cta_threads,
            cta_m_blocks=cta_m_blocks,
            tile_n=tile_n,
            tile_k=tile_k,
            uses_m_block_8=uses_m_block_8,
            sms=sms,
            max_shared_mem=max_shared_mem,
        )
        if blocks_per_sm_limit > best_blocks_per_sm:
            best_blocks_per_sm = blocks_per_sm_limit
            best_tile_config = (tile_k, tile_n, cta_threads, blocks_per_sm_limit)
    if best_tile_config is None:
        cta_thread_msg = (
            ""
            if required_cta_threads is None
            else f", required_cta_threads={required_cta_threads}"
        )
        raise ValueError(
            "no valid W4A16 tile config for "
            f"M/N/K={problem_m}/{problem_n}/{problem_k}, moe_block_size={moe_block_size}"
            f"{cta_thread_msg}"
        )
    return best_tile_config


@dataclass(frozen=True)
class W4A16GemmCompileResult:
    compiled: object
    tile_n: int
    tile_k: int
    moe_block_size: int
    max_m_blocks: int
    blocks_per_sm: int


@dataclass(frozen=True)
class W4A16ActivationCompileResult:
    compiled: object
    rows: int
    intermediate_size: int
    activation: str


@dataclass(frozen=True)
class W4A16TopKSumCompileResult:
    compiled: object
    m: int
    topk: int
    hidden_size: int


@dataclass(frozen=True)
class W4A16FusedMoeCompileResult:
    compiled: object
    fc1_tile_n: int
    fc1_tile_k: int
    fc2_tile_n: int
    fc2_tile_k: int
    moe_block_size: int
    max_m_blocks: int
    blocks_per_sm: int


@dataclass(frozen=True)
class _W4A16GemmLaunch:
    kernel: W4A16GemmCompileResult
    c_tmp: torch.Tensor


class W4A16GemmKernel:
    def __init__(
        self,
        *,
        size_m: int,
        size_n: int,
        size_k: int,
        num_experts: int,
        top_k: int,
        mul_topk_weights: bool,
        tile_n: int,
        tile_k: int,
        moe_block_size: int,
        max_m_blocks: int,
        element_dtype: str = "bf16",
        epilogue_activation: str | None = None,
    ):
        if element_dtype not in {"bf16", "fp16"}:
            raise ValueError(f"unsupported element_dtype {element_dtype!r}")
        if epilogue_activation not in (None, "relu2"):
            raise ValueError(
                "W4A16 GEMM epilogue activation currently supports only relu2"
            )
        if size_n % tile_n != 0:
            raise ValueError("size_n must be divisible by tile_n")
        if size_k % tile_k != 0:
            raise ValueError("size_k must be divisible by tile_k")
        if tile_n % 16 != 0 or tile_k % 16 != 0:
            raise ValueError("tile_n/tile_k must be multiples of 16")
        if moe_block_size not in _ALLOWED_ROUTED_SIZES:
            raise ValueError(f"unsupported moe_block_size {moe_block_size}")
        if moe_block_size != 8 and moe_block_size % 16 != 0:
            raise ValueError("moe_block_size must be 8 or a multiple of 16")
        cta_threads = tile_n * tile_k // 64
        if cta_threads not in (128, 256):
            raise ValueError("W4A16 GEMM expects 128 or 256 CTA threads")
        self.size_m = int(size_m)
        self.size_n = int(size_n)
        self.size_k = int(size_k)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.mul_topk_weights = bool(mul_topk_weights)
        self.tile_n = int(tile_n)
        self.tile_k = int(tile_k)
        self.cta_n_blocks = int(tile_n // 16)
        self.cta_k_blocks = int(tile_k // 16)
        self.cta_threads = int(cta_threads)
        self.moe_block_size = int(moe_block_size)
        self.element_dtype = element_dtype
        self.is_fp16 = element_dtype == "fp16"
        self.epilogue_relu2 = epilogue_activation == "relu2"
        self.cta_m_blocks = int(_covering_count(moe_block_size, 16))
        self.uses_m_block_8 = moe_block_size == 8
        self.max_m_blocks = int(max_m_blocks)
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            self.sms = int(props.multi_processor_count)
            max_shared_mem = int(
                getattr(props, "shared_memory_per_block_optin", _DEFAULT_MAX_SHARED_MEM)
            )
        else:
            self.sms = 120
            max_shared_mem = _DEFAULT_MAX_SHARED_MEM
        self.blocks_per_sm = _determine_blocks_per_sm(
            problem_m=self.size_m,
            problem_n=self.size_n,
            top_k=self.top_k,
            cta_threads=self.cta_threads,
            cta_m_blocks=self.cta_m_blocks,
            tile_n=self.tile_n,
            tile_k=self.tile_k,
            uses_m_block_8=self.uses_m_block_8,
            sms=self.sms,
            max_shared_mem=max_shared_mem,
        )

        # W4A16 shared-memory geometry, in int4 units unless noted.
        self.a_sh_stride = 16 * self.cta_k_blocks // 8
        self.a_sh_stage = self.a_sh_stride * (16 * self.cta_m_blocks)
        self.a_gl_rd_delta_o = 16 * self.cta_k_blocks // 8
        self.a_sh_wr_delta = self.a_sh_stride * (
            self.cta_threads // self.a_gl_rd_delta_o
        )
        self.a_sh_wr_iters = _covering_count(self.a_sh_stage, self.a_sh_wr_delta)
        self.a_sh_rd_delta_i = self.a_sh_stride * 16

        self.b_sh_stride = ((self.cta_n_blocks * 16) * 16 // _PACK_FACTOR) // 4
        self.b_thread_vecs = 1
        self.b_sh_stride_threads = self.b_sh_stride
        self.b_sh_stage = self.b_sh_stride * self.cta_k_blocks
        self.b_sh_wr_iters = self.b_sh_stage // self.cta_threads

        self.s_sh_stride = 16 * self.cta_n_blocks // 16
        self.s_tb_groups = self.cta_k_blocks
        self.s_sh_stage = self.s_tb_groups * self.s_sh_stride
        self.tb_n_warps = self.cta_n_blocks // 4

        sh_block_route_indices = self.moe_block_size // 4
        sh_rd_block_route_indices = self.moe_block_size // 4
        sh_block_topk_weights = self.moe_block_size // 2
        self.sh_valid_count_off = (
            sh_block_route_indices + sh_rd_block_route_indices + sh_block_topk_weights
        )
        self.sh_route_off = 0
        self.sh_rd_route_off = sh_block_route_indices
        self.sh_topk_off = sh_block_route_indices + sh_rd_block_route_indices

        sh_red_size = (2 * self.cta_n_blocks + 1) * 16 * self.cta_m_blocks
        sh_b_size = _STAGES * self.b_sh_stage
        sh_size_min = min(sh_red_size, sh_b_size)
        sh_size_max = max(sh_red_size, sh_b_size)
        sh_bias_size = self.cta_n_blocks * 16 // 8
        sh_b_red_bias_size = max(sh_size_max, sh_size_min + sh_bias_size)
        self.sh_b_off = self.sh_valid_count_off
        self.sh_red_off = self.sh_valid_count_off
        self.sh_s_off = self.sh_valid_count_off + sh_b_red_bias_size
        self.sh_a_off = self.sh_s_off + _STAGES * self.s_sh_stage
        self.shared_int4 = self.sh_a_off + _STAGES * self.a_sh_stage
        self.shared_words = self.shared_int4 * 4

    @cute.jit
    def _activation_smem_permuted_offset(self, i: Int32) -> Int32:
        row = i // Int32(self.a_gl_rd_delta_o)
        return Int32(self.a_gl_rd_delta_o) * row + (
            (i - row * Int32(self.a_gl_rd_delta_o)) ^ (row & Int32(7))
        )

    @cute.jit
    def _int4_addr(self, smem_base: Int32, int4_off: Int32) -> Int32:
        return smem_base + int4_off * Int32(16)

    @cute.jit
    def _dequant_e2m1x4_to_elem2x2(self, packed: Uint32):
        if cutlass.const_expr(self.is_fp16):
            return packed_dequant_e2m1x4_to_half2x2(packed)
        return packed_dequant_e2m1x4_to_bfloat2x2(packed)

    @cute.jit
    def _dequant_e4m3x4_to_elem2x2(self, packed: Uint32):
        if cutlass.const_expr(self.is_fp16):
            return packed_dequant_e4m3x4_to_half2x2(packed)
        return packed_dequant_e4m3x4_to_bfloat2x2(packed)

    @cute.jit
    def _elem2_mul(self, a: Uint32, b: Uint32) -> Uint32:
        if cutlass.const_expr(self.is_fp16):
            return half2_mul(a, b)
        return bfloat2_mul(a, b)

    @cute.jit
    def _broadcast_f32_to_elem2(self, x: cutlass.Float32) -> Uint32:
        if cutlass.const_expr(self.is_fp16):
            return broadcast_f32_to_half2(x)
        return broadcast_f32_to_bfloat2(x)

    @cute.jit
    def _pack_f32x2_to_elem2(self, x0: cutlass.Float32, x1: cutlass.Float32) -> Uint32:
        if cutlass.const_expr(self.is_fp16):
            return pack_f32x2_to_f16x2(x0, x1)
        return pack_f32x2_to_bfloat2(x0, x1)

    @cute.jit
    def _elem2_to_f32x2(self, packed: Uint32):
        if cutlass.const_expr(self.is_fp16):
            return half2_to_float2_scaled(packed, cutlass.Float32(1.0))
        return bfloat2_to_float2_scaled(packed, cutlass.Float32(1.0))

    @cute.jit
    def _relu2_elem2(self, packed: Uint32) -> Uint32:
        x0, x1 = self._elem2_to_f32x2(packed)
        if x0 < cutlass.Float32(0.0):
            x0 = cutlass.Float32(0.0)
        if x1 < cutlass.Float32(0.0):
            x1 = cutlass.Float32(0.0)
        return self._pack_f32x2_to_elem2(x0 * x0, x1 * x1)

    @cute.jit
    def _st_shared_elem_from_f32(self, addr: Int32, val: cutlass.Float32):
        if cutlass.const_expr(self.is_fp16):
            st_shared_f16_from_f32(addr, val)
        else:
            st_shared_bf16_from_f32(addr, val)

    @cute.jit
    def _mma_m16n8k16_f32(
        self,
        d0: cutlass.Float32,
        d1: cutlass.Float32,
        d2: cutlass.Float32,
        d3: cutlass.Float32,
        a0: Uint32,
        a1: Uint32,
        a2: Uint32,
        a3: Uint32,
        b0: Uint32,
        b1: Uint32,
    ):
        if cutlass.const_expr(self.is_fp16):
            return f16_mma_m16n8k16_f32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1)
        return bf16_mma_m16n8k16_f32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1)

    @cute.jit
    def _mma_rhs_fragments_as_mma_a_m16n8k16_f32(
        self,
        d0: cutlass.Float32,
        d1: cutlass.Float32,
        d2: cutlass.Float32,
        d3: cutlass.Float32,
        b0_0: Uint32,
        b1_0: Uint32,
        b0_1: Uint32,
        b1_1: Uint32,
        a0: Uint32,
        a1: Uint32,
    ):
        if cutlass.const_expr(self.is_fp16):
            return f16_mma_rhs_fragments_as_mma_a_m16n8k16_f32(
                d0, d1, d2, d3, b0_0, b1_0, b0_1, b1_1, a0, a1
            )
        return bf16_mma_rhs_fragments_as_mma_a_m16n8k16_f32(
            d0, d1, d2, d3, b0_0, b1_0, b0_1, b1_1, a0, a1
        )

    @cute.jit
    def __call__(
        self,
        a_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        c_bf16_flat: cute.Tensor,
        scales_i32_flat: cute.Tensor,
        global_scale: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        topk_weights_flat: cute.Tensor,
        c_tmp_f32_flat: cute.Tensor,
        locks_i32_flat: cute.Tensor,
        stream: cuda.CUstream,
    ):
        grid_x = self.sms * self.blocks_per_sm
        grid = (grid_x, 1, 1)
        self.kernel(
            a_bf16_flat,
            b_i32_flat,
            c_bf16_flat,
            scales_i32_flat,
            global_scale,
            packed_route_indices,
            block_expert_ids,
            packed_route_count,
            topk_weights_flat,
            c_tmp_f32_flat,
            locks_i32_flat,
        ).launch(
            grid=grid,
            block=[self.cta_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        a_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        c_bf16_flat: cute.Tensor,
        scales_i32_flat: cute.Tensor,
        global_scale: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        topk_weights_flat: cute.Tensor,
        c_tmp_f32_flat: cute.Tensor,
        locks_i32_flat: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        tid = Int32(tidx)
        cta = Int32(bidx)

        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            words: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint32, self.shared_words],
                1024,
            ]

        storage = smem.allocate(Storage)
        smem_base = shared_ptr_to_u32(storage.words.data_ptr())

        grid_x, _, _ = cute.arch.grid_dim()
        self._run_persistent_gemm(
            a_bf16_flat,
            b_i32_flat,
            c_bf16_flat,
            scales_i32_flat,
            global_scale,
            packed_route_indices,
            block_expert_ids,
            packed_route_count,
            topk_weights_flat,
            c_tmp_f32_flat,
            locks_i32_flat,
            smem_base,
            tid,
            cta,
            Int32(grid_x),
        )

    @cute.jit
    def _run_persistent_gemm(
        self,
        a_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        c_bf16_flat: cute.Tensor,
        scales_i32_flat: cute.Tensor,
        global_scale: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        topk_weights_flat: cute.Tensor,
        c_tmp_f32_flat: cute.Tensor,
        locks_i32_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        cta: Int32,
        grid_x: Int32,
    ):
        n_tiles = Int32(self.size_n // self.tile_n)
        route_blocks = packed_route_count[Int32(0)].to(Int32) // Int32(
            self.moe_block_size
        )
        k_tiles = Int32(self.size_k // self.tile_k)
        global_mn_tiles = route_blocks * n_tiles

        tail_mn_tiles = global_mn_tiles
        full_grid_mn_iters = Int32(0)
        if global_mn_tiles > grid_x:
            tail_mn_tiles = global_mn_tiles - (global_mn_tiles // grid_x) * grid_x
            if tail_mn_tiles * Int32(3) <= grid_x:
                tail_mn_tiles += grid_x
            full_grid_mn_iters = (global_mn_tiles - tail_mn_tiles) // grid_x

        iters = (k_tiles * tail_mn_tiles + grid_x - Int32(1)) // grid_x

        lock_slot = Int32(0)
        if tail_mn_tiles >= grid_x:
            lock_slot = cta
        else:
            lock_slot = (iters * cta) // k_tiles - Int32(1)

        in_tail_region = Int32(0)
        has_work = Int32(1)
        work_mn_tile = cta
        reduce_k_tile = Int32(0)
        route_block_idx = Int32(0)
        output_n_tile = Int32(0)
        if iters == Int32(0) and full_grid_mn_iters == Int32(0):
            has_work = Int32(0)

        while has_work != Int32(0):
            reduce_tile_count = Int32(0)
            reduce_slice_count = Int32(1)
            reduce_slice_idx = Int32(0)

            if in_tail_region == Int32(0) and full_grid_mn_iters > Int32(0):
                route_block_idx = work_mn_tile // n_tiles
                output_n_tile = work_mn_tile - route_block_idx * n_tiles
                reduce_k_tile = Int32(0)
                reduce_tile_count = k_tiles
                full_grid_mn_iters -= Int32(1)
            else:
                if in_tail_region == Int32(0):
                    in_tail_region = Int32(1)
                    tail_mn_base = global_mn_tiles - tail_mn_tiles
                    cta_iter_start = iters * cta
                    work_mn_tile = cta_iter_start // k_tiles
                    reduce_k_tile = cta_iter_start - work_mn_tile * k_tiles
                    global_mn_tile = work_mn_tile + tail_mn_base
                    route_block_idx = global_mn_tile // n_tiles
                    output_n_tile = global_mn_tile - route_block_idx * n_tiles

                if work_mn_tile < tail_mn_tiles and iters > Int32(0):
                    reduce_tile_count = iters * (cta + Int32(1)) - (
                        k_tiles * work_mn_tile + reduce_k_tile
                    )
                    if reduce_tile_count < Int32(0):
                        reduce_tile_count = Int32(0)
                    if reduce_k_tile + reduce_tile_count > k_tiles:
                        reduce_tile_count = k_tiles - reduce_k_tile

                    if reduce_tile_count > Int32(0):
                        first_reduce_boundary = iters * (
                            (k_tiles * work_mn_tile + iters - Int32(1)) // iters
                        )
                        if first_reduce_boundary <= k_tiles * (work_mn_tile + Int32(1)):
                            reduce_boundary_offset = (
                                first_reduce_boundary - k_tiles * work_mn_tile
                            )
                            reduce_slice_count = (
                                k_tiles - reduce_boundary_offset + iters - Int32(1)
                            ) // iters
                            if reduce_boundary_offset > Int32(0):
                                reduce_slice_count += Int32(1)
                            reduce_boundary_delta = iters * cta - first_reduce_boundary
                            if reduce_boundary_delta < Int32(0):
                                reduce_slice_idx = reduce_slice_count - Int32(1)
                            else:
                                if reduce_boundary_offset == Int32(
                                    0
                                ) and reduce_boundary_delta == Int32(0):
                                    reduce_slice_idx = reduce_slice_count - Int32(1)
                                else:
                                    reduce_slice_idx = (
                                        reduce_slice_count
                                        - Int32(1)
                                        - reduce_boundary_delta // iters
                                    )
                                    if reduce_boundary_offset > Int32(0):
                                        reduce_slice_idx -= Int32(1)

                        if tail_mn_tiles >= grid_x:
                            if reduce_slice_count > Int32(
                                1
                            ) and reduce_slice_idx == reduce_slice_count - Int32(1):
                                lock_slot += Int32(1)
                        else:
                            lock_slot += Int32(1)
                    else:
                        has_work = Int32(0)
                else:
                    has_work = Int32(0)

            if (
                has_work != Int32(0)
                and reduce_tile_count > Int32(0)
                and route_block_idx < route_blocks
            ):
                expert_idx = block_expert_ids[route_block_idx].to(Int32)
                if expert_idx >= Int32(0):
                    self._run_tile(
                        a_bf16_flat,
                        b_i32_flat,
                        c_bf16_flat,
                        scales_i32_flat,
                        global_scale,
                        packed_route_indices,
                        topk_weights_flat,
                        c_tmp_f32_flat,
                        locks_i32_flat,
                        smem_base,
                        tid,
                        route_block_idx,
                        expert_idx,
                        output_n_tile,
                        reduce_k_tile,
                        reduce_tile_count,
                        reduce_slice_count,
                        reduce_slice_idx,
                        lock_slot,
                    )

            if has_work != Int32(0):
                if in_tail_region == Int32(0):
                    work_mn_tile += grid_x
                else:
                    reduce_k_tile = Int32(0)
                    work_mn_tile += Int32(1)
                    output_n_tile += Int32(1)
                    if output_n_tile == n_tiles:
                        output_n_tile = Int32(0)
                        route_block_idx += Int32(1)

    @cute.jit
    def _read_moe_block_data(
        self,
        packed_route_indices: cute.Tensor,
        topk_weights_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        route_block_idx: Int32,
        global_scale_f32: cutlass.Float32,
    ) -> Int32:
        route_indices_int4_addr = self._int4_addr(
            smem_base, Int32(self.sh_route_off) + tid
        )
        route_indices_gmem = get_ptr_as_int64(
            packed_route_indices,
            route_block_idx * Int32(self.moe_block_size) + tid * Int32(4),
        )
        cp_async4_shared_global_pred(
            route_indices_int4_addr,
            route_indices_gmem,
            (tid < Int32(self.moe_block_size // 4)).to(Int32),
        )
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        if tid >= Int32(self.cta_threads - 32):
            size_per_thread = _covering_count(self.moe_block_size, 32)
            lane = tid - Int32(self.cta_threads - 32)
            local_count = Int32(0)
            for i in cutlass.range_constexpr(size_per_thread):
                j = lane * Int32(size_per_thread) + Int32(i)
                if j < Int32(self.moe_block_size):
                    idx = ld_shared_i32_relaxed(
                        smem_base + Int32(self.sh_route_off * 16) + j * Int32(4)
                    )
                    if idx < Int32(self.size_m * self.top_k):
                        local_count += Int32(1)
            valid = cute.arch.warp_redux_sync(local_count, "add")
            if lane == Int32(0):
                st_shared_i32(smem_base + Int32(self.sh_valid_count_off * 16), valid)

        if tid < Int32(self.moe_block_size):
            idx = ld_shared_i32_relaxed(
                smem_base + Int32(self.sh_route_off * 16) + tid * Int32(4)
            )
            rd_row = idx // Int32(self.top_k)
            st_shared_i32(
                smem_base + Int32(self.sh_rd_route_off * 16) + tid * Int32(4),
                rd_row,
            )
            if cutlass.const_expr(self.mul_topk_weights):
                safe_idx = idx
                if idx >= Int32(self.size_m * self.top_k):
                    safe_idx = Int32(0)
                topk = (
                    topk_weights_flat[safe_idx].to(cutlass.Float32) * global_scale_f32
                )
                packed_topk = self._broadcast_f32_to_elem2(topk)
                # top-k weights are cached as packed element pairs.
                topk_word_addr = (
                    smem_base + Int32(self.sh_topk_off * 16) + tid * Int32(4)
                )
                st_shared_u32(topk_word_addr, packed_topk)

        cute.arch.sync_threads()
        valid_count = ld_shared_i32_relaxed(
            smem_base + Int32(self.sh_valid_count_off * 16)
        )
        cute.arch.sync_threads()
        return valid_count

    @cute.jit
    def _run_tile(
        self,
        a_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        c_bf16_flat: cute.Tensor,
        scales_i32_flat: cute.Tensor,
        global_scale: cute.Tensor,
        packed_route_indices: cute.Tensor,
        topk_weights_flat: cute.Tensor,
        c_tmp_f32_flat: cute.Tensor,
        locks_i32_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        route_block_idx: Int32,
        expert_idx: Int32,
        output_n_tile: Int32,
        reduce_k_tile: Int32,
        reduce_tile_count: Int32,
        reduce_slice_count: Int32,
        reduce_slice_idx: Int32,
        lock_slot: Int32,
    ):
        if cutlass.const_expr(self.uses_m_block_8):
            self._run_tile_m8(
                a_bf16_flat,
                b_i32_flat,
                c_bf16_flat,
                scales_i32_flat,
                global_scale,
                packed_route_indices,
                topk_weights_flat,
                c_tmp_f32_flat,
                locks_i32_flat,
                smem_base,
                tid,
                route_block_idx,
                expert_idx,
                output_n_tile,
                reduce_k_tile,
                reduce_tile_count,
                reduce_slice_count,
                reduce_slice_idx,
                lock_slot,
            )
        else:
            self._run_tile_large_m(
                a_bf16_flat,
                b_i32_flat,
                c_bf16_flat,
                scales_i32_flat,
                global_scale,
                packed_route_indices,
                topk_weights_flat,
                c_tmp_f32_flat,
                locks_i32_flat,
                smem_base,
                tid,
                route_block_idx,
                expert_idx,
                output_n_tile,
                reduce_k_tile,
                reduce_tile_count,
                reduce_slice_count,
                reduce_slice_idx,
                lock_slot,
            )

    @cute.jit
    def _tile_common_prologue(
        self,
        global_scale: cute.Tensor,
        packed_route_indices: cute.Tensor,
        topk_weights_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        route_block_idx: Int32,
        expert_idx: Int32,
        output_n_tile: Int32,
    ):
        global_scale_f32 = global_scale[expert_idx].to(cutlass.Float32)
        block_valid_rows = self._read_moe_block_data(
            packed_route_indices,
            topk_weights_flat,
            smem_base,
            tid,
            route_block_idx,
            global_scale_f32,
        )
        (
            a_gl_stride,
            b_gl_stride,
            s_gl_stride,
            scales_expert_off,
            b_gl_rd_base,
            a_gl_rd_row,
            a_gl_rd_col0,
            a_sh_wr,
            a_rows_per_iter,
            b_sh_rd,
            s_sh_rd,
        ) = self._tile_stream_offsets(tid, expert_idx, output_n_tile)
        return (
            global_scale_f32,
            block_valid_rows,
            a_gl_stride,
            b_gl_stride,
            s_gl_stride,
            scales_expert_off,
            b_gl_rd_base,
            a_gl_rd_row,
            a_gl_rd_col0,
            a_sh_wr,
            a_rows_per_iter,
            b_sh_rd,
            s_sh_rd,
        )

    @cute.jit
    def _tile_stream_offsets(self, tid: Int32, expert_idx: Int32, output_n_tile: Int32):
        a_gl_stride = Int32(self.size_k // 8)
        b_gl_stride = Int32(16 * self.size_n // (_PACK_FACTOR * 4))
        s_gl_stride = Int32(self.size_n // 16)
        scales_expert_stride = Int32((self.size_n * self.size_k) // (16 * 16))
        b_expert_off = (
            Int32((self.size_n * self.size_k) // (_PACK_FACTOR * 4)) * expert_idx
        )
        scales_expert_off = scales_expert_stride * expert_idx

        a_gl_rd_row = tid // Int32(self.a_gl_rd_delta_o)
        a_gl_rd_col0 = tid - a_gl_rd_row * Int32(self.a_gl_rd_delta_o)
        a_sh_wr = Int32(self.a_sh_stride) * (tid // Int32(self.a_gl_rd_delta_o)) + (
            tid - (tid // Int32(self.a_gl_rd_delta_o)) * Int32(self.a_gl_rd_delta_o)
        )
        a_rows_per_iter = Int32(self.cta_threads // self.a_gl_rd_delta_o)

        if cutlass.const_expr(self.cta_threads <= self.b_sh_stride):
            b_gl_rd_base = tid
        else:
            b_gl_rd_base = b_gl_stride * (tid // Int32(self.b_sh_stride)) + (
                tid % Int32(self.b_sh_stride)
            )
        b_gl_rd_base += b_expert_off + Int32(self.b_sh_stride) * output_n_tile
        b_sh_rd = tid
        b_sh_rd += (b_sh_rd // Int32(self.b_sh_stride)) * Int32(
            self.b_sh_stride * (self.b_sh_wr_iters - 1)
        )

        s_sh_rd = Int32(8) * ((tid // Int32(32)) % Int32(self.tb_n_warps)) + (
            tid & Int32(31)
        ) // Int32(4)
        return (
            a_gl_stride,
            b_gl_stride,
            s_gl_stride,
            scales_expert_off,
            b_gl_rd_base,
            a_gl_rd_row,
            a_gl_rd_col0,
            a_sh_wr,
            a_rows_per_iter,
            b_sh_rd,
            s_sh_rd,
        )

    @cute.jit
    def _a_shared_read_offset(self, tid: Int32, lanes_per_row: cutlass.Constexpr[int]):
        a_sh_rd = Int32(self.a_sh_stride) * (
            (tid & Int32(31)) % Int32(lanes_per_row)
        ) + (tid & Int32(31)) // Int32(lanes_per_row)
        a_sh_rd += (
            Int32(2)
            * ((tid // Int32(32)) // Int32(self.tb_n_warps))
            * Int32(self.b_sh_wr_iters)
        )
        return a_sh_rd

    @cute.jit
    def _run_tile_m8(
        self,
        a_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        c_bf16_flat: cute.Tensor,
        scales_i32_flat: cute.Tensor,
        global_scale: cute.Tensor,
        packed_route_indices: cute.Tensor,
        topk_weights_flat: cute.Tensor,
        c_tmp_f32_flat: cute.Tensor,
        locks_i32_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        route_block_idx: Int32,
        expert_idx: Int32,
        output_n_tile: Int32,
        reduce_k_tile: Int32,
        reduce_tile_count: Int32,
        reduce_slice_count: Int32,
        reduce_slice_idx: Int32,
        lock_slot: Int32,
    ):
        (
            global_scale_f32,
            block_valid_rows,
            a_gl_stride,
            b_gl_stride,
            s_gl_stride,
            scales_expert_off,
            b_gl_rd_base,
            a_gl_rd_row,
            a_gl_rd_col0,
            a_sh_wr,
            a_rows_per_iter,
            b_sh_rd,
            s_sh_rd,
        ) = self._tile_common_prologue(
            global_scale,
            packed_route_indices,
            topk_weights_flat,
            smem_base,
            tid,
            route_block_idx,
            expert_idx,
            output_n_tile,
        )
        a_sh_rd = self._a_shared_read_offset(tid, 8)

        acc = cute.make_rmem_tensor((4, 4), cutlass.Float32)
        acc.fill(0.0)

        k_tiles = reduce_tile_count
        self._prefetch_initial_tiles(
            a_bf16_flat,
            b_i32_flat,
            scales_i32_flat,
            smem_base,
            tid,
            k_tiles,
            reduce_k_tile,
            block_valid_rows,
            a_gl_stride,
            b_gl_stride,
            s_gl_stride,
            scales_expert_off,
            b_gl_rd_base,
            a_gl_rd_row,
            a_gl_rd_col0,
            a_sh_wr,
            a_rows_per_iter,
            output_n_tile,
        )

        b_scale_cur = cute.make_rmem_tensor((2, 4), Uint32)
        b_scale_next = cute.make_rmem_tensor((2, 4), Uint32)
        self._load_b_scale_register_bundle(
            b_scale_cur,
            smem_base,
            tid,
            b_sh_rd,
            s_sh_rd,
            Int32(0),
            Int32(0),
        )
        a_regs_cur = cute.make_rmem_tensor((2,), Uint32)
        a_regs_next = cute.make_rmem_tensor((2,), Uint32)
        self._load_a_register_bundle(
            a_regs_cur,
            smem_base,
            a_sh_rd,
            Int32(0),
            Int32(0),
            True,
        )
        self._run_mma_pipeline(
            a_bf16_flat,
            b_i32_flat,
            scales_i32_flat,
            smem_base,
            tid,
            acc,
            b_scale_cur,
            b_scale_next,
            a_regs_cur,
            a_regs_next,
            b_sh_rd,
            s_sh_rd,
            a_sh_rd,
            k_tiles,
            reduce_k_tile,
            block_valid_rows,
            a_gl_stride,
            b_gl_stride,
            s_gl_stride,
            scales_expert_off,
            b_gl_rd_base,
            a_gl_rd_row,
            a_gl_rd_col0,
            a_sh_wr,
            a_rows_per_iter,
            output_n_tile,
            True,
        )

        self._finish_tile(
            acc,
            c_bf16_flat,
            c_tmp_f32_flat,
            locks_i32_flat,
            smem_base,
            tid,
            output_n_tile,
            block_valid_rows,
            global_scale_f32,
            reduce_slice_count,
            reduce_slice_idx,
            lock_slot,
            True,
        )

    @cute.jit
    def _run_tile_large_m(
        self,
        a_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        c_bf16_flat: cute.Tensor,
        scales_i32_flat: cute.Tensor,
        global_scale: cute.Tensor,
        packed_route_indices: cute.Tensor,
        topk_weights_flat: cute.Tensor,
        c_tmp_f32_flat: cute.Tensor,
        locks_i32_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        route_block_idx: Int32,
        expert_idx: Int32,
        output_n_tile: Int32,
        reduce_k_tile: Int32,
        reduce_tile_count: Int32,
        reduce_slice_count: Int32,
        reduce_slice_idx: Int32,
        lock_slot: Int32,
    ):
        (
            global_scale_f32,
            block_valid_rows,
            a_gl_stride,
            b_gl_stride,
            s_gl_stride,
            scales_expert_off,
            b_gl_rd_base,
            a_gl_rd_row,
            a_gl_rd_col0,
            a_sh_wr,
            a_rows_per_iter,
            b_sh_rd,
            s_sh_rd,
        ) = self._tile_common_prologue(
            global_scale,
            packed_route_indices,
            topk_weights_flat,
            smem_base,
            tid,
            route_block_idx,
            expert_idx,
            output_n_tile,
        )
        a_sh_rd = self._a_shared_read_offset(tid, 16)

        acc = cute.make_rmem_tensor((self.cta_m_blocks, 4, 2, 4), cutlass.Float32)
        acc.fill(0.0)

        k_tiles = reduce_tile_count
        self._prefetch_initial_tiles(
            a_bf16_flat,
            b_i32_flat,
            scales_i32_flat,
            smem_base,
            tid,
            k_tiles,
            reduce_k_tile,
            block_valid_rows,
            a_gl_stride,
            b_gl_stride,
            s_gl_stride,
            scales_expert_off,
            b_gl_rd_base,
            a_gl_rd_row,
            a_gl_rd_col0,
            a_sh_wr,
            a_rows_per_iter,
            output_n_tile,
        )

        b_scale_cur = cute.make_rmem_tensor((2, 4), Uint32)
        b_scale_next = cute.make_rmem_tensor((2, 4), Uint32)
        self._load_b_scale_register_bundle(
            b_scale_cur,
            smem_base,
            tid,
            b_sh_rd,
            s_sh_rd,
            Int32(0),
            Int32(0),
        )
        a_regs = cute.make_rmem_tensor((self.cta_m_blocks, 4), Uint32)
        a_regs_next = cute.make_rmem_tensor((self.cta_m_blocks, 4), Uint32)
        self._load_a_register_bundle(
            a_regs,
            smem_base,
            a_sh_rd,
            Int32(0),
            Int32(0),
            False,
        )
        self._run_mma_pipeline(
            a_bf16_flat,
            b_i32_flat,
            scales_i32_flat,
            smem_base,
            tid,
            acc,
            b_scale_cur,
            b_scale_next,
            a_regs,
            a_regs_next,
            b_sh_rd,
            s_sh_rd,
            a_sh_rd,
            k_tiles,
            reduce_k_tile,
            block_valid_rows,
            a_gl_stride,
            b_gl_stride,
            s_gl_stride,
            scales_expert_off,
            b_gl_rd_base,
            a_gl_rd_row,
            a_gl_rd_col0,
            a_sh_wr,
            a_rows_per_iter,
            output_n_tile,
            False,
        )

        self._finish_tile(
            acc,
            c_bf16_flat,
            c_tmp_f32_flat,
            locks_i32_flat,
            smem_base,
            tid,
            output_n_tile,
            block_valid_rows,
            global_scale_f32,
            reduce_slice_count,
            reduce_slice_idx,
            lock_slot,
            False,
        )

    @cute.jit
    def _run_mma_pipeline(
        self,
        a_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        scales_i32_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        acc: cute.Tensor,
        b_scale_cur: cute.Tensor,
        b_scale_next: cute.Tensor,
        a_regs_cur: cute.Tensor,
        a_regs_next: cute.Tensor,
        b_sh_rd: Int32,
        s_sh_rd: Int32,
        a_sh_rd: Int32,
        k_tiles: Int32,
        reduce_k_tile: Int32,
        block_valid_rows: Int32,
        a_gl_stride: Int32,
        b_gl_stride: Int32,
        s_gl_stride: Int32,
        scales_expert_off: Int32,
        b_gl_rd_base: Int32,
        a_gl_rd_row: Int32,
        a_gl_rd_col0: Int32,
        a_sh_wr: Int32,
        a_rows_per_iter: Int32,
        output_n_tile: Int32,
        uses_m_block_8: cutlass.Constexpr[bool],
    ):
        b_frag = cute.make_rmem_tensor((2, 2), Uint32)
        tile_idx = Int32(0)
        while tile_idx < k_tiles:
            for pipe in cutlass.range_constexpr(_STAGES):
                if tile_idx < k_tiles:
                    for kk in cutlass.range_constexpr(self.b_sh_wr_iters):
                        self._load_next_fragment_bundle(
                            b_scale_next,
                            a_regs_next,
                            smem_base,
                            tid,
                            b_sh_rd,
                            s_sh_rd,
                            a_sh_rd,
                            pipe,
                            kk,
                            tile_idx,
                            k_tiles,
                            uses_m_block_8,
                        )

                        self._prefetch_pipeline_step(
                            a_bf16_flat,
                            b_i32_flat,
                            scales_i32_flat,
                            smem_base,
                            tid,
                            pipe,
                            kk,
                            tile_idx,
                            k_tiles,
                            reduce_k_tile,
                            block_valid_rows,
                            a_gl_stride,
                            b_gl_stride,
                            s_gl_stride,
                            scales_expert_off,
                            b_gl_rd_base,
                            a_gl_rd_row,
                            a_gl_rd_col0,
                            a_sh_wr,
                            a_rows_per_iter,
                            output_n_tile,
                        )

                        for jj in cutlass.range_constexpr(4):
                            q, s = self._select_b_scale_register(jj, b_scale_cur)
                            self._scaled_dequant_b_fragment(b_frag, q, s)
                            if cutlass.const_expr(uses_m_block_8):
                                self._mma_accumulate_m8(
                                    acc,
                                    jj,
                                    a_regs_cur,
                                    b_frag,
                                )
                            else:
                                for mb in cutlass.range_constexpr(self.cta_m_blocks):
                                    self._mma_accumulate_large_m(
                                        acc,
                                        a_regs_cur,
                                        mb,
                                        jj,
                                        b_frag,
                                    )

                        if cutlass.const_expr(uses_m_block_8):
                            self._copy_a_register_bundle(
                                a_regs_cur,
                                a_regs_next,
                                uses_m_block_8,
                            )
                            self._copy_b_scale_register_bundle(
                                b_scale_cur, b_scale_next
                            )
                        else:
                            self._copy_b_scale_register_bundle(
                                b_scale_cur, b_scale_next
                            )
                            self._copy_a_register_bundle(
                                a_regs_cur,
                                a_regs_next,
                                uses_m_block_8,
                            )
                    tile_idx += Int32(1)
            cute.arch.sync_threads()
            if tile_idx < k_tiles:
                self._load_b_scale_register_bundle(
                    b_scale_cur,
                    smem_base,
                    tid,
                    b_sh_rd,
                    s_sh_rd,
                    Int32(0),
                    Int32(0),
                )
                self._load_a_register_bundle(
                    a_regs_cur,
                    smem_base,
                    a_sh_rd,
                    Int32(0),
                    Int32(0),
                    uses_m_block_8,
                )

    @cute.jit
    def _finish_tile(
        self,
        acc: cute.Tensor,
        c_bf16_flat: cute.Tensor,
        c_tmp_f32_flat: cute.Tensor,
        locks_i32_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        output_n_tile: Int32,
        block_valid_rows: Int32,
        global_scale_f32: cutlass.Float32,
        reduce_slice_count: Int32,
        reduce_slice_idx: Int32,
        lock_slot: Int32,
        uses_m_block_8: cutlass.Constexpr[bool],
    ):
        if cutlass.const_expr(uses_m_block_8):
            self._fold_cta_partials_m8(acc, smem_base, tid)
        else:
            self._fold_cta_partials_large_m(acc, smem_base, tid)

        if reduce_slice_count > Int32(1):
            self._wait_for_reduction_turn(
                locks_i32_flat, lock_slot, reduce_slice_idx, tid
            )
            self._combine_splitk_accumulators(
                acc,
                c_tmp_f32_flat,
                block_valid_rows,
                lock_slot,
                reduce_slice_idx,
                reduce_slice_count,
                tid,
                uses_m_block_8,
            )
            self._publish_reduction_turn(
                locks_i32_flat,
                lock_slot,
                reduce_slice_idx == reduce_slice_count - Int32(1),
                tid,
            )

        if reduce_slice_idx == reduce_slice_count - Int32(1):
            if cutlass.const_expr(uses_m_block_8):
                self._store_tile_m8(
                    acc,
                    c_bf16_flat,
                    smem_base,
                    tid,
                    output_n_tile,
                    block_valid_rows,
                    global_scale_f32,
                )
            else:
                self._store_tile_large_m(
                    acc,
                    c_bf16_flat,
                    smem_base,
                    tid,
                    output_n_tile,
                    block_valid_rows,
                    global_scale_f32,
                )

    @cute.jit
    def _wait_for_reduction_turn(
        self,
        locks_i32_flat: cute.Tensor,
        lock_slot: Int32,
        count: Int32,
        tid: Int32,
    ):
        lock_addr = get_ptr_as_int64(locks_i32_flat, lock_slot)
        if tid == Int32(0):
            state = Int32(-1)
            while state != count:
                state = ld_global_acquire_i32(lock_addr)
        cute.arch.sync_threads()

    @cute.jit
    def _publish_reduction_turn(
        self,
        locks_i32_flat: cute.Tensor,
        lock_slot: Int32,
        reset,
        tid: Int32,
    ):
        lock_addr = get_ptr_as_int64(locks_i32_flat, lock_slot)
        cute.arch.sync_threads()
        if tid == Int32(0):
            if reset:
                st_global_i32(lock_addr, Int32(0))
            else:
                red_add_global_release_i32(lock_addr, Int32(1))

    @cute.jit
    def _merge_splitk_vec4(
        self,
        c_tmp_f32_flat: cute.Tensor,
        f32_off: Int32,
        reduce_slice_idx: Int32,
        reduce_slice_count: Int32,
        c0: cutlass.Float32,
        c1: cutlass.Float32,
        c2: cutlass.Float32,
        c3: cutlass.Float32,
    ):
        if reduce_slice_idx != Int32(0):
            r0, r1, r2, r3 = ld_global_v4_f32(get_ptr_as_int64(c_tmp_f32_flat, f32_off))
            c0 = c0 + r0
            c1 = c1 + r1
            c2 = c2 + r2
            c3 = c3 + r3
        if reduce_slice_idx != reduce_slice_count - Int32(1):
            st_global_v4_f32(
                get_ptr_as_int64(c_tmp_f32_flat, f32_off),
                c0,
                c1,
                c2,
                c3,
            )
        return c0, c1, c2, c3

    @cute.jit
    def _combine_splitk_accumulators(
        self,
        acc: cute.Tensor,
        c_tmp_f32_flat: cute.Tensor,
        block_valid_rows: Int32,
        lock_slot: Int32,
        reduce_slice_idx: Int32,
        reduce_slice_count: Int32,
        tid: Int32,
        uses_m_block_8: cutlass.Constexpr[bool],
    ):
        active_threads = Int32(32 * self.tb_n_warps)
        c_size_int4 = Int32((self.cta_m_blocks * 16 * self.cta_n_blocks * 16) // 4)
        c_cur_offset = lock_slot * c_size_int4
        if cutlass.const_expr(uses_m_block_8):
            if tid < active_threads:
                for jj in cutlass.range_constexpr(4):
                    k = jj * 2
                    acc[jj, 0], acc[jj, 1], acc[jj, 2], acc[jj, 3] = (
                        self._merge_splitk_slot(
                            c_tmp_f32_flat,
                            c_cur_offset,
                            active_threads,
                            Int32(k),
                            tid,
                            reduce_slice_idx,
                            reduce_slice_count,
                            acc[jj, 0],
                            acc[jj, 1],
                            acc[jj, 2],
                            acc[jj, 3],
                        )
                    )
        else:
            lane_row = (tid & Int32(31)) // Int32(4)
            if tid < active_threads:
                for k in cutlass.range_constexpr(self.cta_m_blocks * 8):
                    mb = k // 8
                    flat_j = k % 8
                    jj = flat_j // 2
                    half = flat_j % 2
                    row_valid = Int32(mb * 16) + lane_row < block_valid_rows
                    if row_valid:
                        (
                            acc[mb, jj, half, 0],
                            acc[mb, jj, half, 1],
                            acc[mb, jj, half, 2],
                            acc[mb, jj, half, 3],
                        ) = self._merge_splitk_slot(
                            c_tmp_f32_flat,
                            c_cur_offset,
                            active_threads,
                            Int32(k),
                            tid,
                            reduce_slice_idx,
                            reduce_slice_count,
                            acc[mb, jj, half, 0],
                            acc[mb, jj, half, 1],
                            acc[mb, jj, half, 2],
                            acc[mb, jj, half, 3],
                        )

    @cute.jit
    def _merge_splitk_slot(
        self,
        c_tmp_f32_flat: cute.Tensor,
        c_cur_offset: Int32,
        active_threads: Int32,
        slot: Int32,
        tid: Int32,
        reduce_slice_idx: Int32,
        reduce_slice_count: Int32,
        c0: cutlass.Float32,
        c1: cutlass.Float32,
        c2: cutlass.Float32,
        c3: cutlass.Float32,
    ):
        int4_off = c_cur_offset + active_threads * slot + tid
        return self._merge_splitk_vec4(
            c_tmp_f32_flat,
            int4_off * Int32(4),
            reduce_slice_idx,
            reduce_slice_count,
            c0,
            c1,
            c2,
            c3,
        )

    @cute.jit
    def _load_a_registers_large_m(
        self,
        smem_base: Int32,
        a_sh_rd: Int32,
        pipe: Int32,
        kk: Int32,
        m_block: Int32,
    ):
        a_addr = self._int4_addr(
            smem_base,
            Int32(self.sh_a_off)
            + pipe * Int32(self.a_sh_stage)
            + self._activation_smem_permuted_offset(
                Int32(2) * kk + m_block * Int32(self.a_sh_rd_delta_i) + a_sh_rd
            ),
        )
        return ldmatrix_m8n8x4_b16(a_addr)

    @cute.jit
    def _load_a_registers_m8(
        self,
        smem_base: Int32,
        a_sh_rd: Int32,
        pipe: Int32,
        kk: Int32,
    ):
        a_addr = self._int4_addr(
            smem_base,
            Int32(self.sh_a_off)
            + pipe * Int32(self.a_sh_stage)
            + self._activation_smem_permuted_offset(Int32(2) * kk + a_sh_rd),
        )
        return ldmatrix_m8n8x2_b16(a_addr)

    @cute.jit
    def _load_a_registers_large_m_bundle(
        self,
        regs: cute.Tensor,
        smem_base: Int32,
        a_sh_rd: Int32,
        pipe: Int32,
        kk: Int32,
    ):
        for mb in cutlass.range_constexpr(self.cta_m_blocks):
            a0, a1, a2, a3 = self._load_a_registers_large_m(
                smem_base,
                a_sh_rd,
                pipe,
                kk,
                Int32(mb),
            )
            regs[mb, 0] = a0
            regs[mb, 1] = a1
            regs[mb, 2] = a2
            regs[mb, 3] = a3

    @cute.jit
    def _load_a_registers_m8_bundle(
        self,
        regs: cute.Tensor,
        smem_base: Int32,
        a_sh_rd: Int32,
        pipe: Int32,
        kk: Int32,
    ):
        a0, a1 = self._load_a_registers_m8(smem_base, a_sh_rd, pipe, kk)
        regs[0] = a0
        regs[1] = a1

    @cute.jit
    def _clear_a_register_bundle_large_m(self, regs: cute.Tensor):
        for mb in cutlass.range_constexpr(self.cta_m_blocks):
            for reg in cutlass.range_constexpr(4):
                regs[mb, reg] = Uint32(0)

    @cute.jit
    def _clear_a_register_bundle_m8(self, regs: cute.Tensor):
        for reg in cutlass.range_constexpr(2):
            regs[reg] = Uint32(0)

    @cute.jit
    def _copy_a_register_bundle_large_m(self, dst: cute.Tensor, src: cute.Tensor):
        for mb in cutlass.range_constexpr(self.cta_m_blocks):
            for reg in cutlass.range_constexpr(4):
                dst[mb, reg] = src[mb, reg]

    @cute.jit
    def _copy_a_register_bundle_m8(self, dst: cute.Tensor, src: cute.Tensor):
        for reg in cutlass.range_constexpr(2):
            dst[reg] = src[reg]

    @cute.jit
    def _load_a_register_bundle(
        self,
        regs: cute.Tensor,
        smem_base: Int32,
        a_sh_rd: Int32,
        pipe: Int32,
        kk: Int32,
        uses_m_block_8: cutlass.Constexpr[bool],
    ):
        if cutlass.const_expr(uses_m_block_8):
            self._load_a_registers_m8_bundle(regs, smem_base, a_sh_rd, pipe, kk)
        else:
            self._load_a_registers_large_m_bundle(regs, smem_base, a_sh_rd, pipe, kk)

    @cute.jit
    def _clear_a_register_bundle(
        self,
        regs: cute.Tensor,
        uses_m_block_8: cutlass.Constexpr[bool],
    ):
        if cutlass.const_expr(uses_m_block_8):
            self._clear_a_register_bundle_m8(regs)
        else:
            self._clear_a_register_bundle_large_m(regs)

    @cute.jit
    def _copy_a_register_bundle(
        self,
        dst: cute.Tensor,
        src: cute.Tensor,
        uses_m_block_8: cutlass.Constexpr[bool],
    ):
        if cutlass.const_expr(uses_m_block_8):
            self._copy_a_register_bundle_m8(dst, src)
        else:
            self._copy_a_register_bundle_large_m(dst, src)

    @cute.jit
    def _load_b_scale_registers(
        self,
        smem_base: Int32,
        tid: Int32,
        b_sh_rd: Int32,
        s_sh_rd: Int32,
        pipe: Int32,
        kk: Int32,
    ):
        b_addr = self._int4_addr(
            smem_base,
            Int32(self.sh_b_off)
            + pipe * Int32(self.b_sh_stage)
            + Int32(self.b_sh_stride) * kk
            + b_sh_rd,
        )
        q0, q1, q2, q3 = ld_shared_v4_u32(b_addr)

        warp_id = tid // Int32(32)
        warp_row = warp_id // Int32(self.tb_n_warps)
        cur_group_id = Int32(self.b_sh_wr_iters) * warp_row + kk
        s_addr = (
            smem_base
            + Int32(self.sh_s_off * 16)
            + pipe * Int32(self.s_sh_stage * 16)
            + (s_sh_rd + cur_group_id * Int32(2 * self.s_sh_stride)) * Int32(8)
        )
        s_pack0, s_pack1 = ld_shared_v2_u32(s_addr)
        s0, s1 = self._dequant_e4m3x4_to_elem2x2(s_pack0)
        s2, s3 = self._dequant_e4m3x4_to_elem2x2(s_pack1)
        return q0, q1, q2, q3, s0, s1, s2, s3

    @cute.jit
    def _load_b_scale_register_bundle(
        self,
        regs: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        b_sh_rd: Int32,
        s_sh_rd: Int32,
        pipe: Int32,
        kk: Int32,
    ):
        q0, q1, q2, q3, s0, s1, s2, s3 = self._load_b_scale_registers(
            smem_base,
            tid,
            b_sh_rd,
            s_sh_rd,
            pipe,
            kk,
        )
        regs[0, 0] = q0
        regs[0, 1] = q1
        regs[0, 2] = q2
        regs[0, 3] = q3
        regs[1, 0] = s0
        regs[1, 1] = s1
        regs[1, 2] = s2
        regs[1, 3] = s3

    @cute.jit
    def _clear_b_scale_register_bundle(self, regs: cute.Tensor):
        for row in cutlass.range_constexpr(2):
            for col in cutlass.range_constexpr(4):
                regs[row, col] = Uint32(0)

    @cute.jit
    def _copy_b_scale_register_bundle(self, dst: cute.Tensor, src: cute.Tensor):
        for row in cutlass.range_constexpr(2):
            for col in cutlass.range_constexpr(4):
                dst[row, col] = src[row, col]

    @cute.jit
    def _select_b_scale_register(self, jj: cutlass.Constexpr[int], regs: cute.Tensor):
        return regs[0, jj], regs[1, jj]

    @cute.jit
    def _load_next_fragment_bundle(
        self,
        b_scale_next: cute.Tensor,
        a_regs_next: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        b_sh_rd: Int32,
        s_sh_rd: Int32,
        a_sh_rd: Int32,
        pipe: cutlass.Constexpr[int],
        kk: cutlass.Constexpr[int],
        tile_idx: Int32,
        k_tiles: Int32,
        uses_m_block_8: cutlass.Constexpr[bool],
    ):
        self._clear_b_scale_register_bundle(b_scale_next)
        self._clear_a_register_bundle(a_regs_next, uses_m_block_8)

        if cutlass.const_expr(kk + 1 < self.b_sh_wr_iters):
            if tile_idx < k_tiles:
                self._load_b_scale_register_bundle(
                    b_scale_next,
                    smem_base,
                    tid,
                    b_sh_rd,
                    s_sh_rd,
                    Int32(pipe),
                    Int32(kk + 1),
                )
                self._load_a_register_bundle(
                    a_regs_next,
                    smem_base,
                    a_sh_rd,
                    Int32(pipe),
                    Int32(kk + 1),
                    uses_m_block_8,
                )
        else:
            next_tile = tile_idx + Int32(1)
            if next_tile < k_tiles:
                self._load_b_scale_register_bundle(
                    b_scale_next,
                    smem_base,
                    tid,
                    b_sh_rd,
                    s_sh_rd,
                    Int32((pipe + 1) % _STAGES),
                    Int32(0),
                )
                self._load_a_register_bundle(
                    a_regs_next,
                    smem_base,
                    a_sh_rd,
                    Int32((pipe + 1) % _STAGES),
                    Int32(0),
                    uses_m_block_8,
                )

    @cute.jit
    def _scaled_dequant_b_fragment(self, frag: cute.Tensor, q: Uint32, s: Uint32):
        bq1 = q
        bq0 = bq1 << Uint32(8)
        b0_0, b0_1 = self._dequant_e2m1x4_to_elem2x2(bq0)
        b1_0, b1_1 = self._dequant_e2m1x4_to_elem2x2(bq1)
        s_lane0 = bfloat2_broadcast_lane(s, Int32(0))
        s_lane1 = bfloat2_broadcast_lane(s, Int32(1))
        b0_0 = self._elem2_mul(b0_0, s_lane0)
        b0_1 = self._elem2_mul(b0_1, s_lane0)
        b1_0 = self._elem2_mul(b1_0, s_lane1)
        b1_1 = self._elem2_mul(b1_1, s_lane1)
        frag[0, 0] = b0_0
        frag[0, 1] = b0_1
        frag[1, 0] = b1_0
        frag[1, 1] = b1_1

    @cute.jit
    def _mma_accumulate_m8(
        self,
        acc: cute.Tensor,
        jj: cutlass.Constexpr[int],
        a_regs: cute.Tensor,
        b_frag: cute.Tensor,
    ):
        d0, d1, d2, d3 = self._mma_rhs_fragments_as_mma_a_m16n8k16_f32(
            acc[jj, 0],
            acc[jj, 1],
            acc[jj, 2],
            acc[jj, 3],
            b_frag[0, 0],
            b_frag[1, 0],
            b_frag[0, 1],
            b_frag[1, 1],
            a_regs[0],
            a_regs[1],
        )
        acc[jj, 0] = d0
        acc[jj, 1] = d1
        acc[jj, 2] = d2
        acc[jj, 3] = d3

    @cute.jit
    def _mma_accumulate_large_m(
        self,
        acc: cute.Tensor,
        a_regs: cute.Tensor,
        mb: cutlass.Constexpr[int],
        jj: cutlass.Constexpr[int],
        b_frag: cute.Tensor,
    ):
        d0, d1, d2, d3 = self._mma_m16n8k16_f32(
            acc[mb, jj, 0, 0],
            acc[mb, jj, 0, 1],
            acc[mb, jj, 0, 2],
            acc[mb, jj, 0, 3],
            a_regs[mb, 0],
            a_regs[mb, 1],
            a_regs[mb, 2],
            a_regs[mb, 3],
            b_frag[0, 0],
            b_frag[0, 1],
        )
        acc[mb, jj, 0, 0] = d0
        acc[mb, jj, 0, 1] = d1
        acc[mb, jj, 0, 2] = d2
        acc[mb, jj, 0, 3] = d3
        d0, d1, d2, d3 = self._mma_m16n8k16_f32(
            acc[mb, jj, 1, 0],
            acc[mb, jj, 1, 1],
            acc[mb, jj, 1, 2],
            acc[mb, jj, 1, 3],
            a_regs[mb, 0],
            a_regs[mb, 1],
            a_regs[mb, 2],
            a_regs[mb, 3],
            b_frag[1, 0],
            b_frag[1, 1],
        )
        acc[mb, jj, 1, 0] = d0
        acc[mb, jj, 1, 1] = d1
        acc[mb, jj, 1, 2] = d2
        acc[mb, jj, 1, 3] = d3

    @cute.jit
    def _stage_k_tile_async(
        self,
        a_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        scales_i32_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        pipe: Int32,
        tile_idx: Int32,
        block_valid_rows: Int32,
        a_gl_stride: Int32,
        b_gl_stride: Int32,
        s_gl_stride: Int32,
        scales_expert_off: Int32,
        b_gl_rd_base: Int32,
        a_gl_rd_row: Int32,
        a_gl_rd_col0: Int32,
        a_sh_wr: Int32,
        a_rows_per_iter: Int32,
        output_n_tile: Int32,
    ):
        for i in cutlass.range_constexpr(self.a_sh_wr_iters):
            row = a_rows_per_iter * Int32(i) + a_gl_rd_row
            route_index = Int32(0)
            if row < Int32(self.moe_block_size):
                route_index = ld_shared_i32_relaxed(
                    smem_base + Int32(self.sh_rd_route_off * 16) + row * Int32(4)
                )
            a_int4 = (
                route_index * a_gl_stride
                + tile_idx * Int32(self.a_gl_rd_delta_o)
                + a_gl_rd_col0
            )
            a_dst = self._int4_addr(
                smem_base,
                Int32(self.sh_a_off)
                + pipe * Int32(self.a_sh_stage)
                + self._activation_smem_permuted_offset(
                    Int32(i * self.a_sh_wr_delta) + a_sh_wr
                ),
            )
            cp_async4_shared_global_pred(
                a_dst,
                get_ptr_as_int64(a_bf16_flat, a_int4 * Int32(8)),
                (row < block_valid_rows).to(Int32),
            )

        for i in cutlass.range_constexpr(self.b_sh_wr_iters):
            b_src_int4 = (
                b_gl_rd_base
                + tile_idx * Int32(self.cta_k_blocks) * b_gl_stride
                + Int32(i * (self.cta_threads // self.b_sh_stride)) * b_gl_stride
            )
            b_dst = self._int4_addr(
                smem_base,
                Int32(self.sh_b_off)
                + pipe * Int32(self.b_sh_stage)
                + Int32(i * self.cta_threads)
                + tid,
            )
            cp_async4_shared_global(
                b_dst,
                get_ptr_as_int64(b_i32_flat, b_src_int4 * Int32(4)),
            )

        if tid < Int32(self.s_sh_stage):
            s_src_int4 = (
                scales_expert_off
                + s_gl_stride
                * (tile_idx * Int32(self.cta_k_blocks) + tid // Int32(self.s_sh_stride))
                + Int32(self.s_sh_stride) * output_n_tile
                + (tid % Int32(self.s_sh_stride))
            )
            s_dst = self._int4_addr(
                smem_base,
                Int32(self.sh_s_off) + pipe * Int32(self.s_sh_stage) + tid,
            )
            cp_async4_shared_global(
                s_dst,
                get_ptr_as_int64(scales_i32_flat, s_src_int4 * Int32(4)),
            )

        cute.arch.cp_async_commit_group()

    @cute.jit
    def _prefetch_pipeline_step(
        self,
        a_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        scales_i32_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        pipe: cutlass.Constexpr[int],
        kk: cutlass.Constexpr[int],
        tile_idx: Int32,
        k_tiles: Int32,
        reduce_k_tile: Int32,
        block_valid_rows: Int32,
        a_gl_stride: Int32,
        b_gl_stride: Int32,
        s_gl_stride: Int32,
        scales_expert_off: Int32,
        b_gl_rd_base: Int32,
        a_gl_rd_row: Int32,
        a_gl_rd_col0: Int32,
        a_sh_wr: Int32,
        a_rows_per_iter: Int32,
        output_n_tile: Int32,
    ):
        if cutlass.const_expr(kk == self.b_sh_wr_iters - 2):
            self._prefetch_lookahead_tile(
                a_bf16_flat,
                b_i32_flat,
                scales_i32_flat,
                smem_base,
                tid,
                pipe,
                tile_idx,
                k_tiles,
                reduce_k_tile,
                block_valid_rows,
                a_gl_stride,
                b_gl_stride,
                s_gl_stride,
                scales_expert_off,
                b_gl_rd_base,
                a_gl_rd_row,
                a_gl_rd_col0,
                a_sh_wr,
                a_rows_per_iter,
                output_n_tile,
            )

    @cute.jit
    def _prefetch_initial_tiles(
        self,
        a_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        scales_i32_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        k_tiles: Int32,
        reduce_k_tile: Int32,
        block_valid_rows: Int32,
        a_gl_stride: Int32,
        b_gl_stride: Int32,
        s_gl_stride: Int32,
        scales_expert_off: Int32,
        b_gl_rd_base: Int32,
        a_gl_rd_row: Int32,
        a_gl_rd_col0: Int32,
        a_sh_wr: Int32,
        a_rows_per_iter: Int32,
        output_n_tile: Int32,
    ):
        for pipe in cutlass.range_constexpr(_STAGES - 1):
            if Int32(pipe) < k_tiles:
                self._stage_k_tile_async(
                    a_bf16_flat,
                    b_i32_flat,
                    scales_i32_flat,
                    smem_base,
                    tid,
                    Int32(pipe),
                    reduce_k_tile + Int32(pipe),
                    block_valid_rows,
                    a_gl_stride,
                    b_gl_stride,
                    s_gl_stride,
                    scales_expert_off,
                    b_gl_rd_base,
                    a_gl_rd_row,
                    a_gl_rd_col0,
                    a_sh_wr,
                    a_rows_per_iter,
                    output_n_tile,
                )
            else:
                cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(_STAGES - 2)
        cute.arch.sync_threads()

    @cute.jit
    def _prefetch_lookahead_tile(
        self,
        a_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        scales_i32_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        pipe: cutlass.Constexpr[int],
        tile_idx: Int32,
        k_tiles: Int32,
        reduce_k_tile: Int32,
        block_valid_rows: Int32,
        a_gl_stride: Int32,
        b_gl_stride: Int32,
        s_gl_stride: Int32,
        scales_expert_off: Int32,
        b_gl_rd_base: Int32,
        a_gl_rd_row: Int32,
        a_gl_rd_col0: Int32,
        a_sh_wr: Int32,
        a_rows_per_iter: Int32,
        output_n_tile: Int32,
    ):
        fetch_tile = tile_idx + Int32(_STAGES - 1)
        if fetch_tile < k_tiles:
            self._stage_k_tile_async(
                a_bf16_flat,
                b_i32_flat,
                scales_i32_flat,
                smem_base,
                tid,
                Int32((pipe + _STAGES - 1) % _STAGES),
                reduce_k_tile + fetch_tile,
                block_valid_rows,
                a_gl_stride,
                b_gl_stride,
                s_gl_stride,
                scales_expert_off,
                b_gl_rd_base,
                a_gl_rd_row,
                a_gl_rd_col0,
                a_sh_wr,
                a_rows_per_iter,
                output_n_tile,
            )
        else:
            cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(_STAGES - 2)
        cute.arch.sync_threads()

    @cute.jit
    def _reduction_offsets(self, tid: Int32):
        red_idx = tid // Int32(self.b_sh_stride_threads)
        red_sh_stride = Int32(self.b_sh_stride_threads * 4 * 2)
        red_sh_delta = Int32(self.b_sh_stride_threads)
        red_sh_rd = red_sh_stride * (tid // Int32(self.b_sh_stride_threads)) + (
            tid % Int32(self.b_sh_stride_threads)
        )
        return red_idx, red_sh_stride, red_sh_delta, red_sh_rd

    @cute.jit
    def _fold_cta_partials_m8(self, acc: cute.Tensor, smem_base: Int32, tid: Int32):
        red_off = self.cta_threads // self.b_sh_stride_threads // 2
        if cutlass.const_expr(red_off >= 1):
            red_idx, red_sh_stride, red_sh_delta, red_sh_rd = self._reduction_offsets(
                tid
            )
            if cutlass.const_expr(red_off == 2):
                if Int32(2) <= red_idx and red_idx < Int32(4):
                    for jj in cutlass.range_constexpr(4):
                        red_sh_wr = red_sh_delta * Int32(jj * 2) + (
                            red_sh_rd - red_sh_stride * Int32(2)
                        )
                        st_shared_v4_f32(
                            self._int4_addr(
                                smem_base, Int32(self.sh_red_off) + red_sh_wr
                            ),
                            acc[jj, 0],
                            acc[jj, 1],
                            acc[jj, 2],
                            acc[jj, 3],
                        )
                cute.arch.sync_threads()

            if Int32(1) <= red_idx and red_idx < Int32(2):
                for jj in cutlass.range_constexpr(4):
                    red_sh_wr = red_sh_delta * Int32(jj * 2) + (
                        red_sh_rd - red_sh_stride
                    )
                    if cutlass.const_expr(red_off > 1):
                        rd_addr = self._int4_addr(
                            smem_base,
                            Int32(self.sh_red_off)
                            + red_sh_delta * Int32(jj * 2)
                            + red_sh_rd,
                        )
                        wr_addr = self._int4_addr(
                            smem_base, Int32(self.sh_red_off) + red_sh_wr
                        )
                        r0, r1, r2, r3 = ld_shared_v4_f32(rd_addr)
                        w0, w1, w2, w3 = ld_shared_v4_f32(wr_addr)
                        acc[jj, 0] = acc[jj, 0] + r0 + w0
                        acc[jj, 1] = acc[jj, 1] + r1 + w1
                        acc[jj, 2] = acc[jj, 2] + r2 + w2
                        acc[jj, 3] = acc[jj, 3] + r3 + w3
                    st_shared_v4_f32(
                        self._int4_addr(smem_base, Int32(self.sh_red_off) + red_sh_wr),
                        acc[jj, 0],
                        acc[jj, 1],
                        acc[jj, 2],
                        acc[jj, 3],
                    )
            cute.arch.sync_threads()

            if red_idx == Int32(0):
                for jj in cutlass.range_constexpr(4):
                    rd_addr = self._int4_addr(
                        smem_base,
                        Int32(self.sh_red_off)
                        + red_sh_delta * Int32(jj * 2)
                        + red_sh_rd,
                    )
                    r0, r1, r2, r3 = ld_shared_v4_f32(rd_addr)
                    acc[jj, 0] = acc[jj, 0] + r0
                    acc[jj, 1] = acc[jj, 1] + r1
                    acc[jj, 2] = acc[jj, 2] + r2
                    acc[jj, 3] = acc[jj, 3] + r3
            cute.arch.sync_threads()

    @cute.jit
    def _output_store_cursor(self, tid: Int32, output_n_tile: Int32):
        c_gl_stride = Int32(self.size_n // 8)
        c_sh_stride = Int32(2 * self.cta_n_blocks + 1)
        c_gl_wr_delta = c_gl_stride * Int32(self.cta_threads // (2 * self.cta_n_blocks))
        c_sh_rd_delta = c_sh_stride * Int32(self.cta_threads // (2 * self.cta_n_blocks))
        c_gl_wr = (
            c_gl_stride * (tid // Int32(2 * self.cta_n_blocks))
            + (tid % Int32(2 * self.cta_n_blocks))
            + Int32(2 * self.cta_n_blocks) * output_n_tile
        )
        c_sh_rd = c_sh_stride * (tid // Int32(2 * self.cta_n_blocks)) + (
            tid % Int32(2 * self.cta_n_blocks)
        )
        return c_gl_stride, c_sh_stride, c_gl_wr_delta, c_sh_rd_delta, c_gl_wr, c_sh_rd

    @cute.jit
    def _drain_output_smem(
        self,
        c_bf16_flat: cute.Tensor,
        smem_base: Int32,
        c_gl_stride: Int32,
        c_gl_wr: Int32,
        c_gl_wr_delta: Int32,
        c_sh_rd: Int32,
        c_sh_rd_delta: Int32,
        block_valid_rows: Int32,
        store_iters: cutlass.Constexpr[int],
    ):
        for _ in cutlass.range_constexpr(store_iters):
            row = c_gl_wr // c_gl_stride
            if row < block_valid_rows:
                route_index = ld_shared_i32_relaxed(
                    smem_base + Int32(self.sh_route_off * 16) + row * Int32(4)
                )
                true_idx = route_index * c_gl_stride + (c_gl_wr % c_gl_stride)
                q0, q1, q2, q3 = ld_shared_v4_u32(
                    self._int4_addr(smem_base, Int32(self.sh_red_off) + c_sh_rd)
                )
                if cutlass.const_expr(self.mul_topk_weights):
                    scale_bf2 = ld_shared_u32(
                        smem_base + Int32(self.sh_topk_off * 16) + row * Int32(4)
                    )
                    q0 = self._elem2_mul(q0, scale_bf2)
                    q1 = self._elem2_mul(q1, scale_bf2)
                    q2 = self._elem2_mul(q2, scale_bf2)
                    q3 = self._elem2_mul(q3, scale_bf2)
                if cutlass.const_expr(self.epilogue_relu2):
                    q0 = self._relu2_elem2(q0)
                    q1 = self._relu2_elem2(q1)
                    q2 = self._relu2_elem2(q2)
                    q3 = self._relu2_elem2(q3)
                st_global_v4_u32(
                    get_ptr_as_int64(c_bf16_flat, true_idx * Int32(8)),
                    q0,
                    q1,
                    q2,
                    q3,
                )
            c_gl_wr += c_gl_wr_delta
            c_sh_rd += c_sh_rd_delta
        cute.arch.sync_threads()

    @cute.jit
    def _store_tile_m8(
        self,
        acc: cute.Tensor,
        c_bf16_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        output_n_tile: Int32,
        block_valid_rows: Int32,
        global_scale_f32: cutlass.Float32,
    ):
        c_gl_stride, c_sh_stride, c_gl_wr_delta, c_sh_rd_delta, c_gl_wr, c_sh_rd = (
            self._output_store_cursor(tid, output_n_tile)
        )
        c_sh_wr = (
            Int32(8) * c_sh_stride * (((tid & Int32(31)) % Int32(4)) * Int32(2))
            + (tid & Int32(31)) // Int32(4)
            + Int32(64) * (tid // Int32(32))
        )

        if tid // Int32(32) < Int32(self.tb_n_warps):
            write_scale = cutlass.Float32(1.0)
            if cutlass.const_expr(not self.mul_topk_weights):
                write_scale = global_scale_f32
            for jj in cutlass.range_constexpr(4):
                wr = c_sh_wr + Int32(16 * jj)
                self._st_shared_elem_from_f32(
                    smem_base + Int32(self.sh_red_off * 16) + (wr * Int32(2)),
                    acc[jj, 0] * write_scale,
                )
                self._st_shared_elem_from_f32(
                    smem_base
                    + Int32(self.sh_red_off * 16)
                    + ((wr + Int32(8) * c_sh_stride) * Int32(2)),
                    acc[jj, 1] * write_scale,
                )
                self._st_shared_elem_from_f32(
                    smem_base
                    + Int32(self.sh_red_off * 16)
                    + ((wr + Int32(8)) * Int32(2)),
                    acc[jj, 2] * write_scale,
                )
                self._st_shared_elem_from_f32(
                    smem_base
                    + Int32(self.sh_red_off * 16)
                    + ((wr + Int32(8) + Int32(8) * c_sh_stride) * Int32(2)),
                    acc[jj, 3] * write_scale,
                )
        cute.arch.sync_threads()

        store_iters = _covering_count(16, self.cta_threads // (2 * self.cta_n_blocks))
        self._drain_output_smem(
            c_bf16_flat,
            smem_base,
            c_gl_stride,
            c_gl_wr,
            c_gl_wr_delta,
            c_sh_rd,
            c_sh_rd_delta,
            block_valid_rows,
            store_iters,
        )

    @cute.jit
    def _fold_cta_partials_large_m(
        self, acc: cute.Tensor, smem_base: Int32, tid: Int32
    ):
        red_off = self.cta_threads // self.b_sh_stride_threads // 2
        if cutlass.const_expr(red_off >= 1):
            red_idx, red_sh_stride, red_sh_delta, red_sh_rd = self._reduction_offsets(
                tid
            )

            for mb in cutlass.range_constexpr(self.cta_m_blocks):
                if cutlass.const_expr(red_off == 2):
                    if Int32(2) <= red_idx and red_idx < Int32(4):
                        for flat_j in cutlass.range_constexpr(8):
                            jj = flat_j // 2
                            half = flat_j % 2
                            red_sh_wr = red_sh_delta * Int32(flat_j) + (
                                red_sh_rd - red_sh_stride * Int32(2)
                            )
                            st_shared_v4_f32(
                                self._int4_addr(
                                    smem_base, Int32(self.sh_red_off) + red_sh_wr
                                ),
                                acc[mb, jj, half, 0],
                                acc[mb, jj, half, 1],
                                acc[mb, jj, half, 2],
                                acc[mb, jj, half, 3],
                            )
                    cute.arch.sync_threads()

                if Int32(1) <= red_idx and red_idx < Int32(2):
                    for flat_j in cutlass.range_constexpr(8):
                        jj = flat_j // 2
                        half = flat_j % 2
                        red_sh_wr = red_sh_delta * Int32(flat_j) + (
                            red_sh_rd - red_sh_stride
                        )
                        if cutlass.const_expr(red_off > 1):
                            rd_addr = self._int4_addr(
                                smem_base,
                                Int32(self.sh_red_off)
                                + red_sh_delta * Int32(flat_j)
                                + red_sh_rd,
                            )
                            wr_addr = self._int4_addr(
                                smem_base,
                                Int32(self.sh_red_off) + red_sh_wr,
                            )
                            r0, r1, r2, r3 = ld_shared_v4_f32(rd_addr)
                            w0, w1, w2, w3 = ld_shared_v4_f32(wr_addr)
                            acc[mb, jj, half, 0] = acc[mb, jj, half, 0] + r0 + w0
                            acc[mb, jj, half, 1] = acc[mb, jj, half, 1] + r1 + w1
                            acc[mb, jj, half, 2] = acc[mb, jj, half, 2] + r2 + w2
                            acc[mb, jj, half, 3] = acc[mb, jj, half, 3] + r3 + w3
                        st_shared_v4_f32(
                            self._int4_addr(
                                smem_base, Int32(self.sh_red_off) + red_sh_wr
                            ),
                            acc[mb, jj, half, 0],
                            acc[mb, jj, half, 1],
                            acc[mb, jj, half, 2],
                            acc[mb, jj, half, 3],
                        )
                cute.arch.sync_threads()

                if red_idx == Int32(0):
                    for flat_j in cutlass.range_constexpr(8):
                        jj = flat_j // 2
                        half = flat_j % 2
                        rd_addr = self._int4_addr(
                            smem_base,
                            Int32(self.sh_red_off)
                            + red_sh_delta * Int32(flat_j)
                            + red_sh_rd,
                        )
                        r0, r1, r2, r3 = ld_shared_v4_f32(rd_addr)
                        acc[mb, jj, half, 0] = acc[mb, jj, half, 0] + r0
                        acc[mb, jj, half, 1] = acc[mb, jj, half, 1] + r1
                        acc[mb, jj, half, 2] = acc[mb, jj, half, 2] + r2
                        acc[mb, jj, half, 3] = acc[mb, jj, half, 3] + r3
                cute.arch.sync_threads()

    @cute.jit
    def _write_bf16x2_shared(
        self,
        smem_base: Int32,
        half2_idx: Int32,
        c0: cutlass.Float32,
        c1: cutlass.Float32,
        write_scale: cutlass.Float32,
    ):
        packed = self._pack_f32x2_to_elem2(c0 * write_scale, c1 * write_scale)
        st_shared_u32(
            smem_base + Int32(self.sh_red_off * 16) + half2_idx * Int32(4),
            packed,
        )

    @cute.jit
    def _store_tile_large_m(
        self,
        acc: cute.Tensor,
        c_bf16_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        output_n_tile: Int32,
        block_valid_rows: Int32,
        global_scale_f32: cutlass.Float32,
    ):
        c_gl_stride, c_sh_stride, c_gl_wr_delta, c_sh_rd_delta, c_gl_wr, c_sh_rd = (
            self._output_store_cursor(tid, output_n_tile)
        )
        c_sh_wr = (
            Int32(4) * c_sh_stride * ((tid & Int32(31)) // Int32(4))
            + (tid & Int32(31)) % Int32(4)
            + Int32(32) * (tid // Int32(32))
        )

        if tid // Int32(32) < Int32(self.tb_n_warps):
            write_scale = cutlass.Float32(1.0)
            if cutlass.const_expr(not self.mul_topk_weights):
                write_scale = global_scale_f32
            for mb in cutlass.range_constexpr(self.cta_m_blocks):
                for jj in cutlass.range_constexpr(4):
                    wr = c_sh_wr + Int32(8 * jj)
                    self._write_bf16x2_shared(
                        smem_base,
                        wr,
                        acc[mb, jj, 0, 0],
                        acc[mb, jj, 0, 1],
                        write_scale,
                    )
                    self._write_bf16x2_shared(
                        smem_base,
                        wr + (Int32(4) * c_sh_stride) * Int32(8) + Int32(0),
                        acc[mb, jj, 0, 2],
                        acc[mb, jj, 0, 3],
                        write_scale,
                    )
                    self._write_bf16x2_shared(
                        smem_base,
                        wr + Int32(4),
                        acc[mb, jj, 1, 0],
                        acc[mb, jj, 1, 1],
                        write_scale,
                    )
                    self._write_bf16x2_shared(
                        smem_base,
                        wr + (Int32(4) * c_sh_stride) * Int32(8) + Int32(4),
                        acc[mb, jj, 1, 2],
                        acc[mb, jj, 1, 3],
                        write_scale,
                    )
                c_sh_wr += Int32(16 * (4 * (2 * self.cta_n_blocks + 1)))
        cute.arch.sync_threads()

        store_iters = _covering_count(
            16 * self.cta_m_blocks,
            self.cta_threads // (2 * self.cta_n_blocks),
        )
        self._drain_output_smem(
            c_bf16_flat,
            smem_base,
            c_gl_stride,
            c_gl_wr,
            c_gl_wr_delta,
            c_sh_rd,
            c_sh_rd_delta,
            block_valid_rows,
            store_iters,
        )


class W4A16FusedMoeKernel:
    def __init__(
        self,
        *,
        size_m: int,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        activation: str,
        apply_router_weight_on_input: bool,
        zero_fc2_output: bool,
        fc1_tile_n: int,
        fc1_tile_k: int,
        fc2_tile_n: int,
        fc2_tile_k: int,
        moe_block_size: int,
        max_m_blocks: int,
        element_dtype: str = "bf16",
    ):
        is_gated = validate_activation(activation)
        fc1_cols = int(intermediate_size) * (2 if is_gated else 1)
        routed_rows = int(size_m) * int(top_k)
        self.size_m = int(size_m)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.fc1_cols = int(fc1_cols)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.activation = activation
        self.activation_is_gated = is_gated
        self.apply_router_weight_on_input = bool(apply_router_weight_on_input)
        self.zero_fc2_output = bool(zero_fc2_output)
        self.element_dtype = element_dtype
        self.is_fp16 = element_dtype == "fp16"
        self.fc1 = W4A16GemmKernel(
            size_m=size_m,
            size_n=fc1_cols,
            size_k=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            mul_topk_weights=bool(apply_router_weight_on_input),
            tile_n=fc1_tile_n,
            tile_k=fc1_tile_k,
            moe_block_size=moe_block_size,
            max_m_blocks=max_m_blocks,
            element_dtype=element_dtype,
            epilogue_activation=None if is_gated else "relu2",
        )
        self.fc2 = W4A16GemmKernel(
            size_m=routed_rows,
            size_n=hidden_size,
            size_k=intermediate_size,
            num_experts=num_experts,
            top_k=1,
            mul_topk_weights=not bool(apply_router_weight_on_input),
            tile_n=fc2_tile_n,
            tile_k=fc2_tile_k,
            moe_block_size=moe_block_size,
            max_m_blocks=max_m_blocks,
            element_dtype=element_dtype,
        )
        self.cta_threads = max(self.fc1.cta_threads, self.fc2.cta_threads)
        if self.fc1.cta_threads != self.fc2.cta_threads:
            raise ValueError(
                "fused W4A16 kernel expects matching FC1/FC2 thread counts"
            )
        self.sms = self.fc1.sms
        self.blocks_per_sm = min(self.fc1.blocks_per_sm, self.fc2.blocks_per_sm)
        self.shared_words = max(self.fc1.shared_words, self.fc2.shared_words)
        self.barrier_count_off = self.sms * 4
        self.barrier_sense_off = self.sms * 4 + 1

    @cute.jit
    def _cast_elem(self, x: cutlass.Float32):
        if cutlass.const_expr(self.is_fp16):
            return cutlass.Float16(x)
        return cutlass.BFloat16(x)

    @cute.jit
    def __call__(
        self,
        a_bf16_flat: cute.Tensor,
        w13_i32_flat: cute.Tensor,
        w2_i32_flat: cute.Tensor,
        fc1_bf16_flat: cute.Tensor,
        activated_bf16_flat: cute.Tensor,
        fc2_bf16_flat: cute.Tensor,
        w13_scales_i32_flat: cute.Tensor,
        w2_scales_i32_flat: cute.Tensor,
        w13_global_scale: cute.Tensor,
        w2_global_scale: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        topk_weights_flat: cute.Tensor,
        fc1_c_tmp_f32_flat: cute.Tensor,
        fc2_c_tmp_f32_flat: cute.Tensor,
        locks_i32_flat: cute.Tensor,
        stream: cuda.CUstream,
    ):
        grid_x = self.sms * self.blocks_per_sm
        grid = (grid_x, 1, 1)
        self.kernel(
            a_bf16_flat,
            w13_i32_flat,
            w2_i32_flat,
            fc1_bf16_flat,
            activated_bf16_flat,
            fc2_bf16_flat,
            w13_scales_i32_flat,
            w2_scales_i32_flat,
            w13_global_scale,
            w2_global_scale,
            packed_route_indices,
            block_expert_ids,
            packed_route_count,
            topk_weights_flat,
            fc1_c_tmp_f32_flat,
            fc2_c_tmp_f32_flat,
            locks_i32_flat,
        ).launch(
            grid=grid,
            block=[self.cta_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        a_bf16_flat: cute.Tensor,
        w13_i32_flat: cute.Tensor,
        w2_i32_flat: cute.Tensor,
        fc1_bf16_flat: cute.Tensor,
        activated_bf16_flat: cute.Tensor,
        fc2_bf16_flat: cute.Tensor,
        w13_scales_i32_flat: cute.Tensor,
        w2_scales_i32_flat: cute.Tensor,
        w13_global_scale: cute.Tensor,
        w2_global_scale: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        topk_weights_flat: cute.Tensor,
        fc1_c_tmp_f32_flat: cute.Tensor,
        fc2_c_tmp_f32_flat: cute.Tensor,
        locks_i32_flat: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_x_raw, _, _ = cute.arch.grid_dim()
        tid = Int32(tidx)
        cta = Int32(bidx)
        grid_x = Int32(grid_x_raw)

        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            words: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint32, self.shared_words],
                1024,
            ]

        storage = smem.allocate(Storage)
        smem_base = shared_ptr_to_u32(storage.words.data_ptr())

        if cutlass.const_expr(self.activation_is_gated):
            self.fc1._run_persistent_gemm(
                a_bf16_flat,
                w13_i32_flat,
                fc1_bf16_flat,
                w13_scales_i32_flat,
                w13_global_scale,
                packed_route_indices,
                block_expert_ids,
                packed_route_count,
                topk_weights_flat,
                fc1_c_tmp_f32_flat,
                locks_i32_flat,
                smem_base,
                tid,
                cta,
                grid_x,
            )
            self._grid_barrier(locks_i32_flat, tid, grid_x)
            self._run_activation(
                fc1_bf16_flat,
                activated_bf16_flat,
                tid,
                cta,
                grid_x,
            )
        else:
            self.fc1._run_persistent_gemm(
                a_bf16_flat,
                w13_i32_flat,
                activated_bf16_flat,
                w13_scales_i32_flat,
                w13_global_scale,
                packed_route_indices,
                block_expert_ids,
                packed_route_count,
                topk_weights_flat,
                fc1_c_tmp_f32_flat,
                locks_i32_flat,
                smem_base,
                tid,
                cta,
                grid_x,
            )
        self._grid_barrier(locks_i32_flat, tid, grid_x)
        if cutlass.const_expr(self.zero_fc2_output):
            self._zero_fc2_output(fc2_bf16_flat, tid, cta, grid_x)
            self._grid_barrier(locks_i32_flat, tid, grid_x)
        self.fc2._run_persistent_gemm(
            activated_bf16_flat,
            w2_i32_flat,
            fc2_bf16_flat,
            w2_scales_i32_flat,
            w2_global_scale,
            packed_route_indices,
            block_expert_ids,
            packed_route_count,
            topk_weights_flat,
            fc2_c_tmp_f32_flat,
            locks_i32_flat,
            smem_base,
            tid,
            cta,
            grid_x,
        )

    @cute.jit
    def _grid_barrier(
        self,
        locks_i32_flat: cute.Tensor,
        tid: Int32,
        grid_x: Int32,
    ):
        cute.arch.sync_threads()
        if tid == Int32(0):
            count_addr = get_ptr_as_int64(locks_i32_flat, Int32(self.barrier_count_off))
            sense_addr = get_ptr_as_int64(locks_i32_flat, Int32(self.barrier_sense_off))
            old_sense = ld_global_acquire_i32(sense_addr)
            old_count = atomic_add_global_i32(count_addr, Int32(1))
            if old_count == grid_x - Int32(1):
                st_global_i32(count_addr, Int32(0))
                threadfence()
                red_add_global_release_i32(sense_addr, Int32(1))
            else:
                sense = old_sense
                while sense == old_sense:
                    sense = ld_global_acquire_i32(sense_addr)
        cute.arch.sync_threads()

    @cute.jit
    def _zero_fc2_output(
        self,
        fc2_bf16_flat: cute.Tensor,
        tid: Int32,
        cta: Int32,
        grid_x: Int32,
    ):
        idx = cta * Int32(self.cta_threads) + tid
        stride = grid_x * Int32(self.cta_threads)
        total = Int32(self.size_m * self.top_k * self.hidden_size)
        zero = self._cast_elem(cutlass.Float32(0.0))
        while idx < total:
            fc2_bf16_flat[idx] = zero
            idx += stride

    @cute.jit
    def _run_activation(
        self,
        fc1_bf16_flat: cute.Tensor,
        activated_bf16_flat: cute.Tensor,
        tid: Int32,
        cta: Int32,
        grid_x: Int32,
    ):
        idx = cta * Int32(self.cta_threads) + tid
        stride = grid_x * Int32(self.cta_threads)
        total = Int32(self.size_m * self.top_k * self.intermediate_size)
        while idx < total:
            if cutlass.const_expr(self.activation_is_gated):
                row = idx // Int32(self.intermediate_size)
                col = idx - row * Int32(self.intermediate_size)
                base = row * Int32(self.fc1_cols)
                gate = fc1_bf16_flat[base + col].to(cutlass.Float32)
                up = fc1_bf16_flat[base + Int32(self.intermediate_size) + col].to(
                    cutlass.Float32
                )
                silu = gate / (
                    cutlass.Float32(1.0) + cute.math.exp(-gate, fastmath=False)
                )
                activated_bf16_flat[idx] = self._cast_elem(
                    self._cast_elem(silu) * self._cast_elem(up)
                )
            else:
                x = fc1_bf16_flat[idx].to(cutlass.Float32)
                if x < cutlass.Float32(0.0):
                    x = cutlass.Float32(0.0)
                activated_bf16_flat[idx] = self._cast_elem(x * x)
            idx += stride


class W4A16ActivationKernel:
    def __init__(
        self,
        *,
        rows: int,
        intermediate_size: int,
        activation: str,
        element_dtype: str = "bf16",
        fast_math: bool = True,
    ):
        is_gated = validate_activation(activation)
        if element_dtype not in {"bf16", "fp16"}:
            raise ValueError(f"unsupported element_dtype {element_dtype!r}")
        if rows <= 0 or intermediate_size <= 0:
            raise ValueError("rows and intermediate_size must be positive")
        self.rows = int(rows)
        self.intermediate_size = int(intermediate_size)
        self.activation = activation
        self.is_gated = is_gated
        self.element_dtype = element_dtype
        self.is_fp16 = element_dtype == "fp16"
        self.fast_math = bool(fast_math)
        self.cta_threads = 256

    @cute.jit
    def _cast_elem(self, x: cutlass.Float32):
        if cutlass.const_expr(self.is_fp16):
            return cutlass.Float16(x)
        return cutlass.BFloat16(x)

    @cute.jit
    def __call__(
        self,
        fc1_flat: cute.Tensor,
        activated_flat: cute.Tensor,
        stream: cuda.CUstream,
    ):
        total = self.rows * self.intermediate_size
        grid = (_covering_count(total, self.cta_threads), 1, 1)
        self.kernel(fc1_flat, activated_flat).launch(
            grid=grid,
            block=[self.cta_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, fc1_flat: cute.Tensor, activated_flat: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        idx = Int32(bidx) * Int32(self.cta_threads) + Int32(tidx)
        total = Int32(self.rows * self.intermediate_size)
        if idx < total:
            if cutlass.const_expr(self.is_gated):
                row = idx // Int32(self.intermediate_size)
                col = idx - row * Int32(self.intermediate_size)
                base = row * Int32(2 * self.intermediate_size)
                gate = fc1_flat[base + col].to(cutlass.Float32)
                up = fc1_flat[base + Int32(self.intermediate_size) + col].to(
                    cutlass.Float32
                )
                silu = gate / (
                    cutlass.Float32(1.0) + cute.math.exp(-gate, fastmath=False)
                )
                activated_flat[idx] = self._cast_elem(
                    self._cast_elem(silu) * self._cast_elem(up)
                )
            else:
                x = fc1_flat[idx].to(cutlass.Float32)
                if x < cutlass.Float32(0.0):
                    x = cutlass.Float32(0.0)
                activated_flat[idx] = self._cast_elem(x * x)


class W4A16TopKSumKernel:
    def __init__(
        self, *, m: int, topk: int, hidden_size: int, element_dtype: str = "bf16"
    ):
        if element_dtype not in {"bf16", "fp16"}:
            raise ValueError(f"unsupported element_dtype {element_dtype!r}")
        if m <= 0 or topk <= 0 or hidden_size <= 0:
            raise ValueError("m, topk, and hidden_size must be positive")
        self.m = int(m)
        self.topk = int(topk)
        self.hidden_size = int(hidden_size)
        self.element_dtype = element_dtype
        self.is_fp16 = element_dtype == "fp16"
        self.cta_threads = 256

    @cute.jit
    def _cast_elem(self, x: cutlass.Float32):
        if cutlass.const_expr(self.is_fp16):
            return cutlass.Float16(x)
        return cutlass.BFloat16(x)

    @cute.jit
    def __call__(
        self,
        fc2_flat: cute.Tensor,
        output_flat: cute.Tensor,
        stream: cuda.CUstream,
    ):
        total = self.m * self.hidden_size
        grid = (_covering_count(total, self.cta_threads), 1, 1)
        self.kernel(fc2_flat, output_flat).launch(
            grid=grid,
            block=[self.cta_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, fc2_flat: cute.Tensor, output_flat: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        idx = Int32(bidx) * Int32(self.cta_threads) + Int32(tidx)
        total = Int32(self.m * self.hidden_size)
        if idx < total:
            token = idx // Int32(self.hidden_size)
            col = idx - token * Int32(self.hidden_size)
            acc = cutlass.Float32(0.0)
            for route in cutlass.range_constexpr(self.topk):
                row = token * Int32(self.topk) + Int32(route)
                acc += fc2_flat[row * Int32(self.hidden_size) + col].to(cutlass.Float32)
            output_flat[idx] = self._cast_elem(acc)


_CACHE: dict[tuple, W4A16GemmCompileResult] = {}
_FUSED_CACHE: dict[tuple, W4A16FusedMoeCompileResult] = {}
_ACTIVATION_CACHE: dict[tuple, W4A16ActivationCompileResult] = {}
_SUM_CACHE: dict[tuple, W4A16TopKSumCompileResult] = {}


def _normalize_element_dtype(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float16:
        return "fp16"
    raise TypeError(f"unsupported W4A16 activation dtype {dtype}")


def _cutlass_element_dtype(element_dtype: str):
    if element_dtype == "bf16":
        return cutlass.BFloat16
    if element_dtype == "fp16":
        return cutlass.Float16
    raise ValueError(f"unsupported element_dtype {element_dtype!r}")


def compile_w4a16_gemm(
    *,
    size_m: int,
    size_n: int,
    size_k: int,
    num_experts: int,
    top_k: int,
    mul_topk_weights: bool,
    tile_n: int,
    tile_k: int,
    moe_block_size: int,
    max_m_blocks: int,
    element_dtype: str = "bf16",
) -> W4A16GemmCompileResult:
    cutlass_dtype = _cutlass_element_dtype(element_dtype)
    if torch.cuda.is_available():
        device = int(torch.cuda.current_device())
        props = torch.cuda.get_device_properties(device)
        sms = int(props.multi_processor_count)
        max_shared_mem = int(
            getattr(props, "shared_memory_per_block_optin", _DEFAULT_MAX_SHARED_MEM)
        )
    else:
        device = None
        sms = 120
        max_shared_mem = _DEFAULT_MAX_SHARED_MEM
    cache_key = (
        "w4a16_gemm",
        device,
        sms,
        max_shared_mem,
        element_dtype,
        size_m,
        size_n,
        size_k,
        num_experts,
        top_k,
        mul_topk_weights,
        tile_n,
        tile_k,
        moe_block_size,
        max_m_blocks,
    )
    cached = _CACHE.get(cache_key)
    if cached is not None:
        return cached

    a_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype,
        (size_m * size_k,),
        assumed_align=16,
    )
    b_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (num_experts * (size_k // 16) * (size_n // 16 * 32),),
        assumed_align=16,
    )
    c_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype,
        (size_m * top_k * size_n,),
        assumed_align=16,
    )
    scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (num_experts * (size_k // 16) * (size_n // 4),),
        assumed_align=16,
    )
    global_scale_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (num_experts,),
        assumed_align=16,
    )
    packed_routes_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (max_m_blocks * moe_block_size,),
        assumed_align=16,
    )
    block_experts_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (max_m_blocks,),
        assumed_align=16,
    )
    route_count_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (1,),
        assumed_align=4,
    )
    topk_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (size_m * top_k,),
        assumed_align=4,
    )
    c_tmp_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (
            max(
                size_n * max_m_blocks * moe_block_size,
                4 * 256 * moe_block_size * 256,
            ),
        ),
        assumed_align=16,
    )
    locks_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (4 * 256,),
        assumed_align=16,
    )

    kernel = W4A16GemmKernel(
        size_m=size_m,
        size_n=size_n,
        size_k=size_k,
        num_experts=num_experts,
        top_k=top_k,
        mul_topk_weights=mul_topk_weights,
        tile_n=tile_n,
        tile_k=tile_k,
        moe_block_size=moe_block_size,
        max_m_blocks=max_m_blocks,
        element_dtype=element_dtype,
    )
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=kernel, cache_key=cache_key
    )
    compiled = cute.compile(
        kernel,
        a_fake,
        b_fake,
        c_fake,
        scales_fake,
        global_scale_fake,
        packed_routes_fake,
        block_experts_fake,
        route_count_fake,
        topk_fake,
        c_tmp_fake,
        locks_fake,
        current_cuda_stream(),
    )
    result = W4A16GemmCompileResult(
        compiled=compiled,
        tile_n=tile_n,
        tile_k=tile_k,
        moe_block_size=moe_block_size,
        max_m_blocks=max_m_blocks,
        blocks_per_sm=kernel.blocks_per_sm,
    )
    _CACHE[cache_key] = result
    return result


def compile_w4a16_fused_moe(
    *,
    size_m: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    activation: str,
    apply_router_weight_on_input: bool,
    zero_fc2_output: bool,
    moe_block_size: int,
    max_m_blocks: int,
    element_dtype: str = "bf16",
    sms: int,
    max_shared_mem: int,
) -> W4A16FusedMoeCompileResult:
    cutlass_dtype = _cutlass_element_dtype(element_dtype)
    device = int(torch.cuda.current_device()) if torch.cuda.is_available() else None
    is_gated = validate_activation(activation)
    fc1_cols = int(intermediate_size) * (2 if is_gated else 1)
    routed_rows = int(size_m) * int(top_k)
    fc1_tile_k, fc1_tile_n, fc1_cta_threads, _ = _select_tile_config(
        problem_m=size_m,
        problem_n=fc1_cols,
        problem_k=hidden_size,
        top_k=top_k,
        moe_block_size=moe_block_size,
        sms=sms,
        max_shared_mem=max_shared_mem,
    )
    fc2_tile_k, fc2_tile_n, fc2_cta_threads, _ = _select_tile_config(
        problem_m=routed_rows,
        problem_n=hidden_size,
        problem_k=intermediate_size,
        top_k=1,
        moe_block_size=moe_block_size,
        sms=sms,
        max_shared_mem=max_shared_mem,
    )
    if fc1_cta_threads != fc2_cta_threads:
        common_cta_threads = min(fc1_cta_threads, fc2_cta_threads)
        fc1_tile_k, fc1_tile_n, fc1_cta_threads, _ = _select_tile_config(
            problem_m=size_m,
            problem_n=fc1_cols,
            problem_k=hidden_size,
            top_k=top_k,
            moe_block_size=moe_block_size,
            sms=sms,
            max_shared_mem=max_shared_mem,
            required_cta_threads=common_cta_threads,
        )
        fc2_tile_k, fc2_tile_n, fc2_cta_threads, _ = _select_tile_config(
            problem_m=routed_rows,
            problem_n=hidden_size,
            problem_k=intermediate_size,
            top_k=1,
            moe_block_size=moe_block_size,
            sms=sms,
            max_shared_mem=max_shared_mem,
            required_cta_threads=common_cta_threads,
        )
        if fc1_cta_threads != fc2_cta_threads:
            raise ValueError(
                "fused W4A16 FC1/FC2 selected different thread counts: "
                f"{fc1_cta_threads} vs {fc2_cta_threads}"
            )
    cache_key = (
        "w4a16_fused_moe",
        device,
        sms,
        max_shared_mem,
        element_dtype,
        size_m,
        hidden_size,
        intermediate_size,
        num_experts,
        top_k,
        activation,
        bool(apply_router_weight_on_input),
        bool(zero_fc2_output),
        fc1_tile_n,
        fc1_tile_k,
        fc2_tile_n,
        fc2_tile_k,
        moe_block_size,
        max_m_blocks,
    )
    cached = _FUSED_CACHE.get(cache_key)
    if cached is not None:
        return cached

    a_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype,
        (size_m * hidden_size,),
        assumed_align=16,
    )
    w13_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (num_experts * (hidden_size // 16) * (fc1_cols // 16 * 32),),
        assumed_align=16,
    )
    w2_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (num_experts * (intermediate_size // 16) * (hidden_size // 16 * 32),),
        assumed_align=16,
    )
    fc1_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype,
        (routed_rows * fc1_cols,),
        assumed_align=16,
    )
    activated_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype,
        (routed_rows * intermediate_size,),
        assumed_align=16,
    )
    fc2_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype,
        (routed_rows * hidden_size,),
        assumed_align=16,
    )
    w13_scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (num_experts * (hidden_size // 16) * (fc1_cols // 4),),
        assumed_align=16,
    )
    w2_scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (num_experts * (intermediate_size // 16) * (hidden_size // 4),),
        assumed_align=16,
    )
    w13_global_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (num_experts,),
        assumed_align=16,
    )
    w2_global_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (num_experts,),
        assumed_align=16,
    )
    packed_routes_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (max_m_blocks * moe_block_size,),
        assumed_align=16,
    )
    block_experts_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (max_m_blocks,),
        assumed_align=16,
    )
    route_count_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (1,),
        assumed_align=4,
    )
    topk_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (size_m * top_k,),
        assumed_align=4,
    )
    fc1_c_tmp_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (
            max(
                fc1_cols * max_m_blocks * moe_block_size,
                4 * 256 * moe_block_size * 256,
            ),
        ),
        assumed_align=16,
    )
    fc2_c_tmp_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (
            max(
                hidden_size * max_m_blocks * moe_block_size,
                4 * 256 * moe_block_size * 256,
            ),
        ),
        assumed_align=16,
    )
    locks_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (4 * 256 + 2,),
        assumed_align=16,
    )

    kernel = W4A16FusedMoeKernel(
        size_m=size_m,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
        zero_fc2_output=zero_fc2_output,
        fc1_tile_n=fc1_tile_n,
        fc1_tile_k=fc1_tile_k,
        fc2_tile_n=fc2_tile_n,
        fc2_tile_k=fc2_tile_k,
        moe_block_size=moe_block_size,
        max_m_blocks=max_m_blocks,
        element_dtype=element_dtype,
    )
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=kernel, cache_key=cache_key
    )
    compiled = cute.compile(
        kernel,
        a_fake,
        w13_fake,
        w2_fake,
        fc1_fake,
        activated_fake,
        fc2_fake,
        w13_scales_fake,
        w2_scales_fake,
        w13_global_fake,
        w2_global_fake,
        packed_routes_fake,
        block_experts_fake,
        route_count_fake,
        topk_fake,
        fc1_c_tmp_fake,
        fc2_c_tmp_fake,
        locks_fake,
        current_cuda_stream(),
    )
    result = W4A16FusedMoeCompileResult(
        compiled=compiled,
        fc1_tile_n=fc1_tile_n,
        fc1_tile_k=fc1_tile_k,
        fc2_tile_n=fc2_tile_n,
        fc2_tile_k=fc2_tile_k,
        moe_block_size=moe_block_size,
        max_m_blocks=max_m_blocks,
        blocks_per_sm=kernel.blocks_per_sm,
    )
    _FUSED_CACHE[cache_key] = result
    return result


def clear_w4a16_kernel_cache() -> None:
    _CACHE.clear()
    _FUSED_CACHE.clear()
    _ACTIVATION_CACHE.clear()
    _SUM_CACHE.clear()


def compile_w4a16_activation(
    *,
    rows: int,
    intermediate_size: int,
    activation: str,
    element_dtype: str = "bf16",
    fast_math: bool = True,
) -> W4A16ActivationCompileResult:
    cutlass_dtype = _cutlass_element_dtype(element_dtype)
    is_gated = validate_activation(activation)
    cache_key = (
        "w4a16_activation",
        element_dtype,
        rows,
        intermediate_size,
        activation,
        fast_math,
    )
    cached = _ACTIVATION_CACHE.get(cache_key)
    if cached is not None:
        return cached

    w13_shards = 2 if is_gated else 1
    fc1_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype,
        (rows * w13_shards * intermediate_size,),
        assumed_align=16,
    )
    activated_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype,
        (rows * intermediate_size,),
        assumed_align=16,
    )
    kernel = W4A16ActivationKernel(
        rows=rows,
        intermediate_size=intermediate_size,
        activation=activation,
        element_dtype=element_dtype,
        fast_math=fast_math,
    )
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=kernel, cache_key=cache_key
    )
    compiled = cute.compile(
        kernel,
        fc1_fake,
        activated_fake,
        current_cuda_stream(),
    )
    result = W4A16ActivationCompileResult(
        compiled=compiled,
        rows=rows,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    _ACTIVATION_CACHE[cache_key] = result
    return result


def compile_w4a16_topk_sum(
    *,
    m: int,
    topk: int,
    hidden_size: int,
    element_dtype: str = "bf16",
) -> W4A16TopKSumCompileResult:
    cutlass_dtype = _cutlass_element_dtype(element_dtype)
    cache_key = ("w4a16_topk_sum", element_dtype, m, topk, hidden_size)
    cached = _SUM_CACHE.get(cache_key)
    if cached is not None:
        return cached

    fc2_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype,
        (m * topk * hidden_size,),
        assumed_align=16,
    )
    output_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype,
        (m * hidden_size,),
        assumed_align=16,
    )
    kernel = W4A16TopKSumKernel(
        m=m,
        topk=topk,
        hidden_size=hidden_size,
        element_dtype=element_dtype,
    )
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=kernel, cache_key=cache_key
    )
    compiled = cute.compile(
        kernel,
        fc2_fake,
        output_fake,
        current_cuda_stream(),
    )
    result = W4A16TopKSumCompileResult(
        compiled=compiled,
        m=m,
        topk=topk,
        hidden_size=hidden_size,
    )
    _SUM_CACHE[cache_key] = result
    return result


def _get_c_tmp(
    elements: int,
    *,
    device: torch.device,
    scratch: torch.Tensor | None = None,
) -> torch.Tensor:
    if scratch is not None:
        if scratch.dtype != torch.float32:
            raise TypeError("W4A16 c_tmp scratch buffers must be torch.float32")
        if scratch.device != device:
            raise ValueError(f"W4A16 c_tmp scratch buffers must be on {device}")
        if not scratch.is_contiguous():
            raise ValueError("W4A16 c_tmp scratch buffers must be contiguous")
        if int(scratch.numel()) >= int(elements):
            return scratch[: int(elements)]
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "W4A16 GEMM scratch is not initialized for CUDA graph capture; "
            "provide a preallocated fc*_c_tmp workspace with sufficient capacity"
        )
    return torch.empty((elements,), dtype=torch.float32, device=device)


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


def _compile_w4a16_gemm_launch(
    *,
    size_m: int,
    size_n: int,
    size_k: int,
    num_experts: int,
    top_k: int,
    mul_topk_weights: bool,
    moe_block_size: int,
    max_m_blocks: int,
    element_dtype: str,
    packed_route_indices: torch.Tensor,
    sms: int,
    max_shared_mem: int,
    device: torch.device,
    c_tmp: torch.Tensor | None = None,
) -> _W4A16GemmLaunch:
    tile_k, tile_n, _, _ = _select_tile_config(
        problem_m=size_m,
        problem_n=size_n,
        problem_k=size_k,
        top_k=top_k,
        moe_block_size=moe_block_size,
        sms=sms,
        max_shared_mem=max_shared_mem,
    )
    kernel = compile_w4a16_gemm(
        size_m=size_m,
        size_n=size_n,
        size_k=size_k,
        num_experts=num_experts,
        top_k=top_k,
        mul_topk_weights=mul_topk_weights,
        tile_n=tile_n,
        tile_k=tile_k,
        moe_block_size=moe_block_size,
        max_m_blocks=max_m_blocks,
        element_dtype=element_dtype,
    )
    c_tmp = _get_c_tmp(
        packed_gemm_scratch_elements(
            size_n=size_n,
            route_slots=int(packed_route_indices.numel()),
            moe_block_size=moe_block_size,
            sms=sms,
        ),
        device=device,
        scratch=c_tmp,
    )
    return _W4A16GemmLaunch(kernel=kernel, c_tmp=c_tmp)


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group top-k routes by expert and pad each group to the GEMM M-block size."""
    _validate_topk_ids(topk_ids, require_cuda=True)
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")
    _validate_expert_map(expert_map, exact_num_experts=int(num_experts))
    return _pack_topk_routes_by_expert(
        topk_ids.view(-1),
        int(block_size),
        int(num_experts),
        expert_map=expert_map,
        packed_route_indices=packed_route_indices,
        block_expert_ids=block_expert_ids,
        packed_route_count=packed_route_count,
        expert_offsets=expert_offsets,
    )


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
    fc1_c_tmp: torch.Tensor | None = None,
    fc2_c_tmp: torch.Tensor | None = None,
    packed_route_indices: torch.Tensor | None = None,
    block_expert_ids: torch.Tensor | None = None,
    packed_route_count: torch.Tensor | None = None,
    expert_offsets: torch.Tensor | None = None,
    expert_map: torch.Tensor | None = None,
    apply_router_weight_on_input: bool = False,
    fast_math: bool = True,
) -> torch.Tensor:
    is_gated = validate_activation(activation)
    element_dtype = _normalize_element_dtype(a_input.dtype)
    if output.dtype != a_input.dtype:
        raise TypeError(f"output must have dtype {a_input.dtype}, got {output.dtype}")
    prepared_dtype = getattr(prepared, "params_dtype", a_input.dtype)
    if prepared_dtype != a_input.dtype:
        raise TypeError(
            f"prepared weights were built for {prepared_dtype}, but a_input has dtype {a_input.dtype}"
        )
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

    route_num_experts = (
        int(expert_map.numel()) if expert_map is not None else int(prepared.num_experts)
    )
    block_size_m = select_route_block_size_m(m, topk, route_num_experts)
    if block_size_m not in _ALLOWED_ROUTED_SIZES:
        raise ValueError(f"unsupported W4A16 moe_block_size={block_size_m}")

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
        )
    )

    props = torch.cuda.get_device_properties(a_input.device)
    sms = int(props.multi_processor_count)
    max_shared_mem = int(
        getattr(props, "shared_memory_per_block_optin", _DEFAULT_MAX_SHARED_MEM)
    )
    buffer_plan = plan_w4a16_buffers(
        prepared,
        m=m,
        topk=topk,
        route_num_experts=route_num_experts,
        sms=sms,
    )
    routed_rows = buffer_plan.routed_rows
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
    if (
        intermediate_cache13.dtype != a_input.dtype
        or intermediate_cache2.dtype != a_input.dtype
    ):
        raise TypeError(f"intermediate caches must be {a_input.dtype}")
    if (
        not intermediate_cache13.is_contiguous()
        or not intermediate_cache2.is_contiguous()
        or not output.is_contiguous()
    ):
        raise ValueError("intermediate caches and output must be contiguous")

    intermediate_cache13_flat = intermediate_cache13.view(-1)
    intermediate_cache2_flat = intermediate_cache2.view(-1)
    fc1_out = intermediate_cache13_flat[: routed_rows * fc1_cols].view(
        routed_rows,
        fc1_cols,
    )
    activated = intermediate_cache2_flat[: routed_rows * intermediate_size].view(
        routed_rows,
        intermediate_size,
    )
    fc2_out = intermediate_cache13_flat[: routed_rows * hidden_size].view(
        routed_rows,
        hidden_size,
    )

    del fast_math
    if int(prepared.workspace.numel()) < sms * 4 + 2:
        raise ValueError("prepared W4A16 workspace is too small for fused FC1+FC2")
    fused = compile_w4a16_fused_moe(
        size_m=m,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=int(prepared.num_experts),
        top_k=topk,
        activation=activation,
        apply_router_weight_on_input=bool(apply_router_weight_on_input),
        zero_fc2_output=expert_map is not None,
        moe_block_size=block_size_m,
        max_m_blocks=int(block_expert_ids.numel()),
        element_dtype=element_dtype,
        sms=sms,
        max_shared_mem=max_shared_mem,
    )
    fc1_scratch = _get_c_tmp(
        packed_gemm_scratch_elements(
            size_n=fc1_cols,
            route_slots=int(packed_route_indices.numel()),
            moe_block_size=block_size_m,
            sms=sms,
        ),
        device=a_input.device,
        scratch=fc1_c_tmp,
    )
    fc2_scratch = _get_c_tmp(
        packed_gemm_scratch_elements(
            size_n=hidden_size,
            route_slots=int(packed_route_indices.numel()),
            moe_block_size=block_size_m,
            sms=sms,
        ),
        device=a_input.device,
        scratch=fc2_c_tmp,
    )
    fused.compiled(
        a_input.view(-1),
        prepared.w13.view(torch.int32).view(-1),
        prepared.w2.view(torch.int32).view(-1),
        fc1_out.view(-1),
        activated.view(-1),
        fc2_out.view(-1),
        prepared.w13_scale.view(torch.uint8).view(torch.int32).view(-1),
        prepared.w2_scale.view(torch.uint8).view(torch.int32).view(-1),
        prepared.w13_global_scale,
        prepared.w2_global_scale,
        packed_route_indices,
        block_expert_ids,
        packed_route_count,
        topk_weights.view(-1),
        fc1_scratch,
        fc2_scratch,
        prepared.workspace,
        current_cuda_stream(),
    )

    sum_kernel = compile_w4a16_topk_sum(
        m=m,
        topk=topk,
        hidden_size=hidden_size,
        element_dtype=element_dtype,
    )
    sum_kernel.compiled(
        fc2_out.view(-1),
        output.view(-1),
        current_cuda_stream(),
    )
    return output


__all__ = [
    "W4A16ActivationCompileResult",
    "W4A16FusedMoeCompileResult",
    "W4A16GemmCompileResult",
    "W4A16TopKSumCompileResult",
    "W4A16FusedMoeKernel",
    "W4A16ActivationKernel",
    "W4A16GemmKernel",
    "W4A16TopKSumKernel",
    "clear_w4a16_kernel_cache",
    "compile_w4a16_activation",
    "compile_w4a16_fused_moe",
    "compile_w4a16_gemm",
    "compile_w4a16_topk_sum",
    "pack_topk_routes_by_expert",
    "run_w4a16_moe",
]
