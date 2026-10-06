"""BF16 prefill projection with compensated FP32 tensor-core accumulation.

Inputs remain BF16 without activation or weight quantization. Output belongs
to the caller. Tensor descriptors keep live row counts dynamic for graph reuse.
The TMA pipeline follows B12X's MHC BF16 projection implementation.
"""
from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as cutlass_utils
import cutlass.utils.hopper_helpers as sm90_utils_basic
import torch
from cutlass import Float32, Int32, Int64, const_expr
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import LayoutEnum

from b12x._lib.compile_plan import attach_programs
from b12x._lib.program_cache import program_cache
from b12x._lib.compiler import DimKey, KernelCompileSpec, compile as b12x_compile, launch, run_compiled, tensor_key
from b12x._lib.utils import current_cuda_stream

def _to_kernel_tensor(tensor, dtype, *, dynamic_layout=False):
    if hasattr(tensor, "fake_mode"):
        from cutlass.cute.runtime import make_fake_tensor
        if dynamic_layout:
            shape = tuple(cute.sym_int(32) for _ in tensor.shape)
            strides = (cute.sym_int(64), 1)
        else:
            shape, strides = tuple(tensor.shape), tuple(tensor.stride())
        return make_fake_tensor(dtype, shape, strides, assumed_align=16)
    result = from_dlpack(tensor.detach(), assumed_align=16)
    result.element_type = dtype
    if dynamic_layout:
        result = result.mark_layout_dynamic(leading_dim=1)
        result.element_type = dtype
    return result


def _assume_tma_source_aligned(tensor):
    row_stride = tensor.stride[0]
    if not isinstance(row_stride, int):
        row_stride = cute.assume(row_stride, divby=8)
    return cute.make_tensor(tensor.iterator, cute.make_layout(
        tensor.shape, stride=(row_stride, 1)))


@cute.jit
def _warp_gemm(tiled_mma, acc, correction, fragment_a, fragment_b, shared_a, shared_b,
               copy_a, copy_b):
    target_a = copy_a.retile(fragment_a)
    target_b = copy_b.retile(fragment_b)
    cute.copy(copy_a, shared_a[None, None, 0], target_a[None, None, 0])
    cute.copy(copy_b, shared_b[None, None, 0], target_b[None, None, 0])
    segment = cute.make_rmem_tensor(acc.shape, Float32)
    for k in cutlass.range_constexpr(cute.size(shared_a.shape[2])):
        if k < cute.size(shared_a.shape[2]) - 1:
            cute.copy(copy_a, shared_a[None, None, k + 1],
                      target_a[None, None, k + 1])
            cute.copy(copy_b, shared_b[None, None, k + 1],
                      target_b[None, None, k + 1])
        # Limit tensor-core accumulation to 16 products; retain the complete
        # projection sum in FP32 ALU registers. Compensated addition retains
        # small terms across the long reduction without tensor-core carry loss.
        segment.fill(0.0)
        cute.gemm(tiled_mma, segment, fragment_a[None, None, k],
                  fragment_b[None, None, k], segment)
        addend = segment.load() - correction.load()
        total = acc.load() + addend
        correction.store((total - acc.load()) - addend)
        acc.store(total)


class Bf16PrefillKernel:
    """TMA-fed BF16 matrix product with FP32 accumulators and runtime rows."""

    num_threads = 160
    num_compute_warps = 4
    producer_warp = 4
    tile_m = 64
    tile_n = 64
    tile_k = 64
    num_stages = 2
    # Keep the long K pipeline compact. Full unrolling increases register live
    # ranges without changing the ordered compensated accumulation contract.
    unroll_k = False
    buffer_align_bytes = 1024

    def __init__(self, n: int, k: int):
        self.n, self.k = int(n), int(k)
        if self.n <= 0 or self.k <= 0 or self.k % self.tile_k:
            raise ValueError("BF16 prefill needs positive N and K divisible by 64")
        self.k_tiles = self.k // self.tile_k
        self.n_tiles = (self.n + self.tile_n - 1) // self.tile_n

    def _get_tiled_mma(self) -> cute.TiledMma:
        return cute.make_tiled_mma(
            warp.MmaF16BF16Op(cutlass.BFloat16, Float32, (16, 8, 16)),
            (self.num_compute_warps, 1, 1),
            permutation_mnk=(self.num_compute_warps * 16, self.tile_n, 16),
        )

    def _get_smem_layouts(self) -> tuple[cute.ComposedLayout, cute.ComposedLayout]:
        a_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(
                LayoutEnum.ROW_MAJOR,
                cutlass.BFloat16,
                self.tile_k,
            ),
            cutlass.BFloat16,
        )
        b_layout_atom = a_layout_atom
        sA_layout = cute.tile_to_shape(
            a_layout_atom,
            (self.tile_m, self.tile_k, self.num_stages),
            order=(0, 1, 2),
        )
        sB_layout = cute.tile_to_shape(
            b_layout_atom,
            (self.tile_n, self.tile_k, self.num_stages),
            order=(0, 1, 2),
        )
        return sA_layout, sB_layout

    def _get_shared_storage_cls(
        self,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
    ):
        class SharedStorage:
            pass

        SharedStorage.__annotations__ = {
            "mbar_ptr": cute.struct.MemRange[
                cutlass.Int64,
                self.num_stages * 2,
            ],
            "sA": cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(sA_layout)],
                self.buffer_align_bytes,
            ],
            "sB": cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(sB_layout)],
                self.buffer_align_bytes,
            ],
        }
        return cute.struct(SharedStorage)

    @cute.jit
    def __call__(
        self,
        source: cute.Tensor,
        weight: cute.Tensor,
        output: cute.Tensor,
        num_tokens: Int32,
        stream: cuda.CUstream,
    ):
        if const_expr(source.element_type != cutlass.BFloat16):
            raise TypeError("source must be BFloat16")
        if const_expr(weight.element_type != cutlass.BFloat16):
            raise TypeError("weight must be BFloat16")
        if const_expr(output.element_type not in (cutlass.Float32, cutlass.BFloat16)):
            raise TypeError("output must be Float32 or BFloat16")

        sA_layout, sB_layout = self._get_smem_layouts()
        tiled_mma = self._get_tiled_mma()
        SharedStorage = self._get_shared_storage_cls(sA_layout, sB_layout)
        sA_tma_layout = cute.slice_(sA_layout, (None, None, 0))
        sB_tma_layout = cute.slice_(sB_layout, (None, None, 0))
        out_tma = _assume_tma_source_aligned(source)
        fn_tma = _assume_tma_source_aligned(weight)
        tma_atom_A, tma_tensor_A = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            out_tma,
            sA_tma_layout,
            (self.tile_m, self.tile_k),
            num_multicast=1,
        )
        tma_atom_B, tma_tensor_B = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            fn_tma,
            sB_tma_layout,
            (self.tile_n, self.tile_k),
            num_multicast=1,
        )
        grid_m = (num_tokens + Int32(self.tile_m - 1)) // Int32(self.tile_m)
        self.kernel(
            tma_tensor_A,
            tma_tensor_B,
            output,
            tma_atom_A,
            tma_atom_B,
            sA_layout,
            sB_layout,
            tiled_mma,
            SharedStorage,
            num_tokens,
        ).launch(
            grid=(grid_m, self.n_tiles, 1),
            block=[self.num_threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        source: cute.Tensor,
        weight: cute.Tensor,
        output: cute.Tensor,
        tma_atom_A: cute.CopyAtom,
        tma_atom_B: cute.CopyAtom,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        tiled_mma: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
        num_tokens: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        m_tile, n_tile, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_A)
            cpasync.prefetch_descriptor(tma_atom_B)

        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sA = storage.sA.get_tensor(sA_layout.outer, swizzle=sA_layout.inner)
        sB = storage.sB.get_tensor(sB_layout.outer, swizzle=sB_layout.inner)

        tma_copy_bytes = (
            (self.tile_m + self.tile_n) * self.tile_k * cutlass.BFloat16.width // 8
        )
        load_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.num_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.num_compute_warps,
            ),
            tx_count=tma_copy_bytes,
            barrier_storage=storage.mbar_ptr.data_ptr(),
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        cute.arch.sync_threads()

        gA = cute.local_tile(
            source,
            (self.tile_m, self.tile_k),
            (None, None),
        )
        gB = cute.local_tile(
            weight,
            (self.tile_n, self.tile_k),
            (None, None),
        )
        cta_layout = cute.make_layout(1)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_A,
            0,
            cta_layout,
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA, 0, 2),
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_B,
            0,
            cta_layout,
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB, 0, 2),
        )

        if warp_idx < Int32(self.num_compute_warps):
            consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                self.num_stages,
            )
            thr_mma = tiled_mma.get_slice(tidx)
            tCsA = thr_mma.partition_A(sA)
            tCsB = thr_mma.partition_B(sB)
            tCrA = thr_mma.make_fragment_A(tCsA[None, None, None, 0])
            tCrB = thr_mma.make_fragment_B(tCsB[None, None, None, 0])
            acc_shape = thr_mma.partition_shape_C((self.tile_m, self.tile_n))
            acc = cute.make_rmem_tensor(acc_shape, Float32)
            acc.fill(0.0)
            correction = cute.make_rmem_tensor(acc_shape, Float32)
            correction.fill(0.0)
            smem_copy_atom_A = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                cutlass.BFloat16,
            )
            smem_copy_atom_B = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                cutlass.BFloat16,
            )
            smem_thr_copy_A = cute.make_tiled_copy_A(
                smem_copy_atom_A,
                tiled_mma,
            ).get_slice(tidx)
            smem_thr_copy_B = cute.make_tiled_copy_B(
                smem_copy_atom_B,
                tiled_mma,
            ).get_slice(tidx)
            tSsA = smem_thr_copy_A.partition_S(sA)
            tSsB = smem_thr_copy_B.partition_S(sB)

            for _k_tile in cutlass.range(self.k_tiles, unroll_full=self.unroll_k):
                load_pipeline.consumer_wait(consumer_state)
                _warp_gemm(
                    thr_mma,
                    acc,
                    correction,
                    tCrA,
                    tCrB,
                    tSsA[None, None, None, consumer_state.index],
                    tSsB[None, None, None, consumer_state.index],
                    smem_thr_copy_A,
                    smem_thr_copy_B,
                )
                load_pipeline.consumer_release(consumer_state)
                consumer_state.advance()

            coordinates = thr_mma.partition_C(
                cute.make_identity_tensor((self.tile_m, self.tile_n))
            )
            for index in cutlass.range_constexpr(cute.size(acc)):
                coord = coordinates[index]
                token = m_tile * Int32(self.tile_m) + coord[0]
                column = n_tile * Int32(self.tile_n) + coord[1]
                if token < num_tokens and column < Int32(self.n):
                    output[Int64(token), Int64(column)] = acc[index].to(
                        output.element_type)

        elif warp_idx == Int32(self.producer_warp):
            producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                self.num_stages,
            )
            for k_tile in cutlass.range(self.k_tiles, unroll_full=self.unroll_k):
                load_pipeline.producer_acquire(producer_state)
                cute.copy(
                    tma_atom_A,
                    tAgA[(None, m_tile, k_tile)],
                    tAsA[(None, producer_state.index)],
                    tma_bar_ptr=load_pipeline.producer_get_barrier(producer_state),
                )
                cute.copy(
                    tma_atom_B,
                    tBgB[(None, n_tile, k_tile)],
                    tBsB[(None, producer_state.index)],
                    tma_bar_ptr=load_pipeline.producer_get_barrier(producer_state),
                )
                load_pipeline.producer_commit(producer_state)
                producer_state.advance()
            load_pipeline.producer_tail(producer_state)



@cache
def _kernel(n, k):
    return Bf16PrefillKernel(n, k)


def supports_prefill(x, weight, out, bias) -> bool:
    """Static projection geometry and layout qualified for the tensor-core path.

    Live rows only select the scalar/tensor-core dispatch at execution. They
    never specialize a compiled callable. Smaller rows retain the scalar kernel.
    """
    return (
        bias is None
        and tuple(weight.shape) in ((384, 5120), (512, 5120), (1024, 5120))
        and x.dtype == weight.dtype == torch.bfloat16
        and all(t.is_contiguous() and t.data_ptr() % 16 == 0
                for t in (x, weight, out))
    )


def prefill_mm(x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor) -> None:
    """Write a contiguous BF16 matrix product into caller-owned BF16/FP32 output."""
    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]:
        raise ValueError("requires x[M,K] and weight[N,K]")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError("input and weight must be BF16")
    if out.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("output must be BF16 or FP32")
    if tuple(out.shape) != (x.shape[0], weight.shape[0]):
        raise ValueError("output must have shape [M,N]")
    if not x.is_cuda or weight.device != x.device or out.device != x.device:
        raise ValueError("all operands must share a CUDA device")
    if any(not t.is_contiguous() or t.data_ptr() % 16 for t in (x, weight, out)):
        raise ValueError("operands must be contiguous and 16-byte aligned")
    if torch._C._overlaps(out, x) or torch._C._overlaps(out, weight):
        raise ValueError("output must not alias either input")
    if not x.shape[0]:
        return
    args = (
        _to_kernel_tensor(x, cutlass.BFloat16, dynamic_layout=True),
        _to_kernel_tensor(weight, cutlass.BFloat16),
        _to_kernel_tensor(out, cutlass.Float32 if out.dtype == torch.float32
                          else cutlass.BFloat16, dynamic_layout=True),
        Int32(x.shape[0]), current_cuda_stream(),
    )
    key = tuple(
        tensor_key(name, t, dims=(
            (DimKey.dynamic() if name != "weight" else DimKey.exact(t.shape[0])),
            DimKey.exact(t.shape[1]),
        ))
        for name, t in (("source", x), ("weight", weight), ("output", out))
    )
    launch(
        _kernel(weight.shape[0], weight.shape[1]),
        compile_spec=KernelCompileSpec.from_key("gemm.bf16_prefill", 3, key),
        compile_args=args, runtime_args=args,
    )


def _prepared_args(x, weight, out):
    return (
        _to_kernel_tensor(x, cutlass.BFloat16, dynamic_layout=True),
        _to_kernel_tensor(weight, cutlass.BFloat16),
        _to_kernel_tensor(out, cutlass.Float32 if out.dtype == torch.float32
                          else cutlass.BFloat16, dynamic_layout=True),
        Int32(x.shape[0]), current_cuda_stream(),
    )


@program_cache
def compile_prefill(ordinal, max_rows, n, k, output_dtype):
    """Retain the native TMA prefill launcher without allocating CUDA operands."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    with torch.cuda.device(ordinal), FakeTensorMode():
        device = torch.device("cuda", ordinal)
        x = torch.empty((max_rows, k), dtype=torch.bfloat16, device=device)
        weight = torch.empty((n, k), dtype=torch.bfloat16, device=device)
        out = torch.empty((max_rows, n), dtype=getattr(torch, output_dtype), device=device)
        args = _prepared_args(x, weight, out)
        key = tuple(
            tensor_key(name, t, dims=(
                DimKey.dynamic() if name != "weight" else DimKey.exact(t.shape[0]),
                DimKey.exact(t.shape[1]),
            ))
            for name, t in (("source", x), ("weight", weight), ("output", out))
        )
        raw = b12x_compile(
            _kernel(n, k), *args,
            compile_spec=KernelCompileSpec.from_key("gemm.bf16_prefill", 3, key),
        )

    def run(x, weight, out, bias=None):
        if bias is not None:
            raise ValueError("TMA prefill projection does not consume bias")
        with torch.cuda.device(ordinal):
            run_compiled(raw, _prepared_args(x, weight, out))

    return attach_programs(run, raw)
