"""Split-axis merge for the active sparse MLA SM120 kernels."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
import torch
from cutlass import Float32, Int32, Uint32
from cutlass.cute.runtime import from_dlpack

from b12x._lib.intrinsics import (
    get_ptr_as_int64,
    ld_global_v4_u32,
    ld_shared_f32,
    shared_ptr_to_u32,
    st_shared_f32,
    store_v4_bf16x2,
)

from b12x.attention._shared.cute import ops as attention_ops
from b12x.attention._shared.workspace import _SPLIT_MAX_CHUNKS
from b12x._lib.compiler import (
    DimKey,
    KernelCompileSpec,
    launch as b12x_launch,
    run_compiled,
    tensor_compile_fact,
)
from b12x._lib.utils import current_cuda_stream

from .decode_math import _exp2_approx_ftz_f32, _u32_to_f32
from .reference import _MLA_GROUP_SIZE, _MLA_NOPE_DIM


_MLA_SCALE_GROUPS = _MLA_NOPE_DIM // _MLA_GROUP_SIZE
_MLA_WARP_THREADS = 32


def _raise_binding_extras(api_name: str, extras: list[str]) -> None:
    raise ValueError(
        f"{api_name} binding owns runtime tensors, scratch, and kernel options; "
        f"do not also pass {', '.join(extras)}"
    )


def _require_bound_arg(value, *, api_name: str, name: str):
    if value is None:
        raise TypeError(f"{api_name} requires {name} or binding")
    return value


@dataclass(frozen=True, kw_only=True)
class SparseMLASplitDecodeMergeBinding:
    tmp_output: torch.Tensor
    tmp_lse: torch.Tensor
    num_chunks_ptr: torch.Tensor
    output: torch.Tensor
    num_chunks: int | None = None
    attn_sink: torch.Tensor | None = None
    scratch: object | None = None

    def run(self) -> None:
        run_sparse_mla_split_decode_merge(binding=self)


def build_sparse_mla_split_decode_merge_binding(
    *,
    tmp_output: torch.Tensor,
    tmp_lse: torch.Tensor,
    num_chunks_ptr: torch.Tensor,
    output: torch.Tensor,
    num_chunks: int | None = None,
    attn_sink: torch.Tensor | None = None,
    scratch: object | None = None,
) -> SparseMLASplitDecodeMergeBinding:
    return SparseMLASplitDecodeMergeBinding(
        tmp_output=tmp_output,
        tmp_lse=tmp_lse,
        num_chunks_ptr=num_chunks_ptr,
        output=output,
        num_chunks=num_chunks,
        attn_sink=attn_sink,
        scratch=scratch,
    )


def _validate_tensor_storage_bounds(tensor: torch.Tensor, *, name: str) -> None:
    if tensor.numel() == 0:
        return
    min_offset = int(tensor.storage_offset())
    max_offset = int(tensor.storage_offset())
    for size, stride in zip(tensor.shape, tensor.stride(), strict=True):
        extent = (int(size) - 1) * int(stride)
        if extent >= 0:
            max_offset += extent
        else:
            min_offset += extent
    storage_elems = tensor.untyped_storage().nbytes() // tensor.element_size()
    if min_offset < 0 or max_offset >= storage_elems:
        raise ValueError(
            f"{name} view is out of storage bounds: shape={tuple(tensor.shape)} "
            f"stride={tuple(tensor.stride())} storage_offset={int(tensor.storage_offset())} "
            f"storage_elems={storage_elems}"
        )


def _validate_split_control_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    device: torch.device,
) -> None:
    if tensor.shape != (1,):
        raise ValueError(f"{name} must have shape (1,), got {tuple(tensor.shape)}")
    if tensor.dtype != torch.int32:
        raise TypeError(f"{name} must have dtype torch.int32, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _torch_to_cutlass_dtype(dtype: torch.dtype) -> type[cutlass.Numeric]:
    if dtype == torch.bfloat16:
        return cutlass.BFloat16
    if dtype == torch.float16:
        return cutlass.Float16
    if dtype == torch.float32:
        return cutlass.Float32
    if dtype == torch.int32:
        return cutlass.Int32
    if dtype == torch.uint8:
        return cutlass.Uint8
    if dtype == torch.uint32:
        return cutlass.Uint32
    raise TypeError(f"unsupported dtype {dtype}")


def _dynamic_leading_dim(tensor: torch.Tensor) -> int | None:
    # Singleton split views can have both head and split strides equal to one.
    # Keep the innermost unit stride fixed so padded split capacities preserve
    # the same dynamic head-stride ABI.
    return next(
        (idx for idx in range(tensor.ndim - 1, -1, -1) if tensor.stride(idx) == 1),
        None,
    )


def _to_kernel_tensor(
    tensor: torch.Tensor,
    dtype: type[cutlass.Numeric],
    *,
    assumed_align: int = 16,
) -> cute.Tensor:
    is_fake = hasattr(tensor, "fake_mode")
    leading_dim = _dynamic_leading_dim(tensor)
    if is_fake:
        from cutlass.cute.runtime import make_fake_tensor
        dynamic = leading_dim is not None and tensor.ndim >= 2
        cute_tensor = make_fake_tensor(
            dtype,
            tuple(cute.sym_int(32) for _ in tensor.shape) if dynamic else tuple(tensor.shape),
            tuple(1 if index == leading_dim else cute.sym_int(64)
                  for index in range(tensor.ndim)) if dynamic else tuple(tensor.stride()),
            assumed_align=assumed_align,
        )
    else:
        cute_tensor = from_dlpack(tensor, assumed_align=assumed_align)
        cute_tensor.element_type = dtype
    if not is_fake and leading_dim is not None and tensor.ndim >= 2:
        cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)
    return cute_tensor


def _tensor_meta_key(
    tensor: torch.Tensor,
    *,
    dynamic_dims: tuple[int, ...] = (),
) -> tuple[tuple[object, ...], tuple[int, ...], str, tuple[str, int | None]]:
    dynamic_dim_set = set(dynamic_dims)
    return (
        tuple(
            DimKey.dynamic() if idx in dynamic_dim_set else int(dim)
            for idx, dim in enumerate(tensor.shape)
        ),
        tuple(tensor.stride()),
        str(tensor.dtype),
        (tensor.device.type, tensor.device.index),
    )


def _tensor_compile_key(
    name: str,
    tensor: torch.Tensor,
    *,
    dynamic_dims: tuple[int, ...] = (),
    dynamic_strides: tuple[int, ...] = (),
) -> tuple[object, ...]:
    return tensor_compile_fact(
        name,
        tensor,
        dynamic_dims=dynamic_dims,
        dynamic_strides=dynamic_strides,
        layout=("leading_dim", _dynamic_leading_dim(tensor)),
    )


@cute.jit
def _split_output_lane_view(
    tmp_output: cute.Tensor,
    q_idx: Int32,
    head_idx: Int32,
    out_base: Int32,
) -> cute.Tensor:
    return cute.make_tensor(
        attention_ops.elem_pointer(tmp_output, (q_idx, head_idx, Int32(0), out_base)),
        cute.make_layout(
            (tmp_output.shape[2], 4),
            stride=(tmp_output.stride[2], 1),
        ),
    )


@cute.jit
def _split_lse_head_view(
    tmp_lse: cute.Tensor,
    q_idx: Int32,
    head_idx: Int32,
) -> cute.Tensor:
    return cute.make_tensor(
        attention_ops.elem_pointer(tmp_lse, (q_idx, head_idx, Int32(0))),
        cute.make_layout(
            (tmp_lse.shape[2],),
            stride=(tmp_lse.stride[2],),
        ),
    )


# Merge CTA geometry: one CTA per (row, head, 128-wide output group) with
# 128 threads. Thread ``t`` owns output vector ``t & 15`` (eight bf16) and
# split lane ``t >> 4``. A pass covers 64 split slots (eight per lane), so
# every partial vector of a pass is fetched by exactly one thread and all of
# a thread's loads are issued before any is consumed; passes repeat until
# ``num_chunks`` slots are covered, and the lanes are summed through shared
# memory.
_MERGE_THREADS = 128
_MERGE_VECTORS = _MLA_GROUP_SIZE // 8
_MERGE_SLOT_LANES = _MERGE_THREADS // _MERGE_VECTORS
_MERGE_SLOTS_PER_LANE = 8
_MERGE_PASS_SLOTS = _MERGE_SLOT_LANES * _MERGE_SLOTS_PER_LANE
_MERGE_PASS_SHIFT = _MERGE_PASS_SLOTS.bit_length() - 1
_MERGE_LSE_PER_LANE = _SPLIT_MAX_CHUNKS // _MLA_WARP_THREADS
# Shared scratch: one weight per split slot, then 8 lanes x 16 vectors x 8 floats.
_MERGE_SMEM_FLOATS = _SPLIT_MAX_CHUNKS + _MERGE_SLOT_LANES * _MERGE_VECTORS * 8
_NEG_INF = float("-inf")

assert _MERGE_PASS_SLOTS == 1 << _MERGE_PASS_SHIFT
assert _SPLIT_MAX_CHUNKS % _MERGE_PASS_SLOTS == 0
assert _SPLIT_MAX_CHUNKS % _MLA_WARP_THREADS == 0


def _merge_pass_plan(static_num_chunks: int | None) -> tuple[int | None, int]:
    """Return (static pass count or None, slots loaded per lane in a pass)."""
    if static_num_chunks is None:
        return None, _MERGE_SLOTS_PER_LANE
    chunks = max(1, min(int(static_num_chunks), _SPLIT_MAX_CHUNKS))
    passes = (chunks + _MERGE_PASS_SLOTS - 1) // _MERGE_PASS_SLOTS
    if passes > 1:
        return passes, _MERGE_SLOTS_PER_LANE
    return 1, (chunks + _MERGE_SLOT_LANES - 1) // _MERGE_SLOT_LANES


@cute.jit
def _merge_accumulate_pass(
    tmp_output: cute.Tensor,
    acc: cute.Tensor,
    weights_addr: Int32,
    q_idx: Int32,
    head_idx: Int32,
    slot_base: Int32,
    slot_lane: Int32,
    last_slot: Int32,
    dim0: Int32,
    *,
    slots_per_lane: cutlass.Constexpr,
    vector_partials: cutlass.Constexpr,
):
    """Accumulate this thread's slots ``slot_base + slot_lane + 8k`` into ``acc``.

    Load indices are clamped to ``last_slot`` so absent slots re-read a valid
    partial; their weight is zero and the zero-weight guard keeps a poisoned
    partial of an empty split out of the sum.
    """
    if cutlass.const_expr(vector_partials):
        loaded = []
        for k in cutlass.range_constexpr(slots_per_lane):
            slot = slot_base + slot_lane + Int32(k * _MERGE_SLOT_LANES)
            if slot > last_slot:
                slot = last_slot
            src = get_ptr_as_int64(
                tmp_output,
                cute.crd2idx((q_idx, head_idx, slot, dim0), tmp_output.layout),
            )
            loaded.append(ld_global_v4_u32(src))
        for k in cutlass.range_constexpr(slots_per_lane):
            w = ld_shared_f32(
                weights_addr
                + (slot_base + slot_lane + Int32(k * _MERGE_SLOT_LANES)) * Int32(4)
            )
            if w != Float32(0.0):
                x0, x1, x2, x3 = loaded[k]
                i = 0
                for x in (x0, x1, x2, x3):
                    acc[i] = acc[i] + w * _u32_to_f32(x << Uint32(16))
                    acc[i + 1] = acc[i + 1] + w * _u32_to_f32(x & Uint32(0xFFFF0000))
                    i += 2
    else:
        for k in cutlass.range_constexpr(slots_per_lane):
            slot = slot_base + slot_lane + Int32(k * _MERGE_SLOT_LANES)
            if slot > last_slot:
                slot = last_slot
            w = ld_shared_f32(
                weights_addr
                + (slot_base + slot_lane + Int32(k * _MERGE_SLOT_LANES)) * Int32(4)
            )
            if w != Float32(0.0):
                for i in cutlass.range_constexpr(8):
                    acc[i] = acc[i] + w * Float32(
                        tmp_output[q_idx, head_idx, slot, dim0 + Int32(i)]
                    )


@cute.jit
def _merge_split_partials(
    tmp_output: cute.Tensor,
    tmp_lse: cute.Tensor,
    output: cute.Tensor,
    attn_sink: cute.Tensor,
    num_chunks: Int32,
    scratch: cute.Tensor,
    *,
    has_sink: cutlass.Constexpr,
    static_num_chunks: cutlass.Constexpr,
    vector_output: cutlass.Constexpr,
    vector_partials: cutlass.Constexpr,
):
    """Reduce ``num_chunks`` normalized split partials of one (row, head) into
    the CTA's 128-dim output group.

    Phase 1 (warp 0): each lane loads the base-2 split LSEs of slots
    ``lane + 32j``, the warp reduces the max and the exp2 sum, folds the
    optional sink into the normalizer exactly as the previous per-lane online
    merge did, and stores every slot's normalized weight (zero past
    ``num_chunks`` and for -inf LSEs) to shared memory.
    Phase 2: passes of 64 slots accumulate weighted partials in fp32 with the
    loads of a pass issued up front, and the eight split lanes are summed
    through shared memory before the sixteen output vectors are stored.
    """
    static_passes, slots_per_lane = _merge_pass_plan(static_num_chunks)
    tid = Int32(cute.arch.thread_idx()[0])
    lane = cute.arch.lane_idx()
    warp_id = tid >> Int32(5)
    q_idx, head_idx, group_idx = cute.arch.block_idx()
    q_idx = Int32(q_idx)
    head_idx = Int32(head_idx)
    group_idx = Int32(group_idx)

    # Cross-warp exchange goes through explicit shared loads and stores in one
    # allocation: one weight per split slot, then the lane-reduction buffer.
    scratch_addr = shared_ptr_to_u32(scratch.iterator)
    weights_addr = scratch_addr
    red_addr = scratch_addr + Int32(_SPLIT_MAX_CHUNKS * 4)

    if num_chunks > Int32(_SPLIT_MAX_CHUNKS):
        num_chunks = Int32(_SPLIT_MAX_CHUNKS)
    last_slot = num_chunks - Int32(1)
    if last_slot < Int32(0):
        last_slot = Int32(0)

    lane_w = cute.make_rmem_tensor(_MERGE_LSE_PER_LANE, Float32)
    acc = cute.make_rmem_tensor(8, Float32)
    for i in cutlass.range_constexpr(8):
        acc[i] = Float32(0.0)

    if warp_id == Int32(0):
        m = Float32(_NEG_INF)
        for j in cutlass.range_constexpr(_MERGE_LSE_PER_LANE):
            c = lane + Int32(j * _MLA_WARP_THREADS)
            lse_j = Float32(_NEG_INF)
            if c < num_chunks:
                lse_j = Float32(tmp_lse[q_idx, head_idx, c])
            lane_w[j] = lse_j
            m = attention_ops.fmax(m, lse_j)
        for off in (16, 8, 4, 2, 1):
            m = attention_ops.fmax(m, cute.arch.shuffle_sync_bfly(m, offset=off))
        d = Float32(0.0)
        for j in cutlass.range_constexpr(_MERGE_LSE_PER_LANE):
            w_j = Float32(0.0)
            if lane_w[j] > Float32(_NEG_INF):
                w_j = _exp2_approx_ftz_f32(lane_w[j] - m)
            lane_w[j] = w_j
            d = d + w_j
        for off in (16, 8, 4, 2, 1):
            d = d + cute.arch.shuffle_sync_bfly(d, offset=off)
        scale = Float32(0.0)
        if d > Float32(0.0):
            if cutlass.const_expr(has_sink):
                sink_m = Float32(attn_sink[head_idx] * attention_ops.LOG2_E)
                new_m = attention_ops.fmax(m, sink_m)
                prev_scale = _exp2_approx_ftz_f32(m - new_m)
                sink_scale = _exp2_approx_ftz_f32(sink_m - new_m)
                scale = prev_scale * cute.arch.rcp_approx(d * prev_scale + sink_scale)
            else:
                scale = cute.arch.rcp_approx(d)
        for j in cutlass.range_constexpr(_MERGE_LSE_PER_LANE):
            st_shared_f32(
                weights_addr + (lane + Int32(j * _MLA_WARP_THREADS)) * Int32(4),
                lane_w[j] * scale,
            )
    cute.arch.barrier()

    vec = tid & Int32(_MERGE_VECTORS - 1)
    slot_lane = tid >> Int32(4)
    dim0 = group_idx * Int32(_MLA_GROUP_SIZE) + vec * Int32(8)
    if cutlass.const_expr(static_passes is None):
        num_passes = (num_chunks + Int32(_MERGE_PASS_SLOTS - 1)) >> Int32(_MERGE_PASS_SHIFT)
        for p in cutlass.range(num_passes, unroll=1):
            _merge_accumulate_pass(
                tmp_output,
                acc,
                weights_addr,
                q_idx,
                head_idx,
                Int32(p) * Int32(_MERGE_PASS_SLOTS),
                slot_lane,
                last_slot,
                dim0,
                slots_per_lane=slots_per_lane,
                vector_partials=vector_partials,
            )
    else:
        for p in cutlass.range_constexpr(static_passes):
            _merge_accumulate_pass(
                tmp_output,
                acc,
                weights_addr,
                q_idx,
                head_idx,
                Int32(p * _MERGE_PASS_SLOTS),
                slot_lane,
                last_slot,
                dim0,
                slots_per_lane=slots_per_lane,
                vector_partials=vector_partials,
            )

    red_base = red_addr + (slot_lane * Int32(_MERGE_VECTORS) + vec) * Int32(32)
    for i in cutlass.range_constexpr(8):
        st_shared_f32(red_base + Int32(i * 4), acc[i])
    cute.arch.barrier()

    if tid < Int32(_MERGE_VECTORS):
        out = [Float32(0.0) for _ in range(8)]
        for sl in cutlass.range_constexpr(_MERGE_SLOT_LANES):
            base = red_addr + (Int32(sl * _MERGE_VECTORS) + tid) * Int32(32)
            for i in cutlass.range_constexpr(8):
                out[i] = out[i] + ld_shared_f32(base + Int32(i * 4))
        out_dim0 = group_idx * Int32(_MLA_GROUP_SIZE) + tid * Int32(8)
        if cutlass.const_expr(vector_output):
            dst = get_ptr_as_int64(
                output, cute.crd2idx((q_idx, head_idx, out_dim0), output.layout)
            )
            store_v4_bf16x2(dst, out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7])
        else:
            for i in cutlass.range_constexpr(8):
                output[q_idx, head_idx, out_dim0 + Int32(i)] = out[i].to(
                    output.element_type
                )


class SparseMLASplitDecodeMergeKernel:
    """Reduce normalized chunk partials into the final decode output."""

    def __init__(
        self,
        static_num_chunks: int | None = None,
        vector_output: bool = True,
        vector_partials: bool = True,
    ):
        self.static_num_chunks = static_num_chunks
        self.vector_output = bool(vector_output)
        self.vector_partials = bool(vector_partials)

    @cute.jit
    def __call__(
        self,
        tmp_output: cute.Tensor,
        tmp_lse: cute.Tensor,
        num_chunks_ptr: cute.Tensor,
        output: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(
            tmp_output,
            tmp_lse,
            num_chunks_ptr,
            output,
        ).launch(
            grid=(output.shape[0], output.shape[1], _MLA_SCALE_GROUPS),
            block=[_MERGE_THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tmp_output: cute.Tensor,
        tmp_lse: cute.Tensor,
        num_chunks_ptr: cute.Tensor,
        output: cute.Tensor,
    ):
        smem = cutlass_utils.SmemAllocator()
        scratch = smem.allocate_tensor(Float32, cute.make_layout(_MERGE_SMEM_FLOATS), 16)
        if cutlass.const_expr(self.static_num_chunks is None):
            num_chunks = Int32(num_chunks_ptr[Int32(0)])
        else:
            num_chunks = Int32(self.static_num_chunks)
        _merge_split_partials(
            tmp_output,
            tmp_lse,
            output,
            tmp_lse,
            num_chunks,
            scratch,
            has_sink=False,
            static_num_chunks=self.static_num_chunks,
            vector_output=self.vector_output,
            vector_partials=self.vector_partials,
        )


class SparseMLASplitDecodeSinkMergeKernel:
    """Reduce chunk partials and fold a zero-value attention sink into softmax."""

    def __init__(
        self,
        static_num_chunks: int | None = None,
        vector_output: bool = True,
        vector_partials: bool = True,
    ):
        self.static_num_chunks = static_num_chunks
        self.vector_output = bool(vector_output)
        self.vector_partials = bool(vector_partials)

    @cute.jit
    def __call__(
        self,
        tmp_output: cute.Tensor,
        tmp_lse: cute.Tensor,
        num_chunks_ptr: cute.Tensor,
        attn_sink: cute.Tensor,
        output: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(
            tmp_output,
            tmp_lse,
            num_chunks_ptr,
            attn_sink,
            output,
        ).launch(
            grid=(output.shape[0], output.shape[1], _MLA_SCALE_GROUPS),
            block=[_MERGE_THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tmp_output: cute.Tensor,
        tmp_lse: cute.Tensor,
        num_chunks_ptr: cute.Tensor,
        attn_sink: cute.Tensor,
        output: cute.Tensor,
    ):
        smem = cutlass_utils.SmemAllocator()
        scratch = smem.allocate_tensor(Float32, cute.make_layout(_MERGE_SMEM_FLOATS), 16)
        if cutlass.const_expr(self.static_num_chunks is None):
            num_chunks = Int32(num_chunks_ptr[Int32(0)])
        else:
            num_chunks = Int32(self.static_num_chunks)
        _merge_split_partials(
            tmp_output,
            tmp_lse,
            output,
            attn_sink,
            num_chunks,
            scratch,
            has_sink=True,
            static_num_chunks=self.static_num_chunks,
            vector_output=self.vector_output,
            vector_partials=self.vector_partials,
        )


@lru_cache(maxsize=None)
def _build_sparse_mla_split_merge_kernel(
    static_num_chunks: int | None = None,
    vector_output: bool = True,
    vector_partials: bool = True,
) -> SparseMLASplitDecodeMergeKernel:
    return SparseMLASplitDecodeMergeKernel(static_num_chunks, vector_output, vector_partials)


@lru_cache(maxsize=None)
def _build_sparse_mla_split_sink_merge_kernel(
    static_num_chunks: int | None = None,
    vector_output: bool = True,
    vector_partials: bool = True,
) -> SparseMLASplitDecodeSinkMergeKernel:
    return SparseMLASplitDecodeSinkMergeKernel(static_num_chunks, vector_output, vector_partials)


def _merge_vector_tensor(tensor: torch.Tensor) -> bool:
    """True when every 8-element row segment of ``tensor`` is a 16-byte-aligned bf16 vector."""
    return bool(
        tensor.dtype == torch.bfloat16
        and int(tensor.stride(-1)) == 1
        and (hasattr(tensor, "fake_mode") or int(tensor.data_ptr()) % 16 == 0)
        and all((int(st) * 2) % 16 == 0 for st in tensor.stride()[:-1])
    )


def clear_sparse_mla_merge_kernel_cache() -> None:
    _build_sparse_mla_split_merge_kernel.cache_clear()
    _build_sparse_mla_split_sink_merge_kernel.cache_clear()


def _sparse_mla_split_decode_merge_flat_launch(
    tmp_output: torch.Tensor,
    tmp_lse: torch.Tensor,
    num_chunks_ptr: torch.Tensor,
    output: torch.Tensor,
    attn_sink: torch.Tensor,
    contract_tmp_output: torch.Tensor,
    contract_tmp_lse: torch.Tensor,
    contract_output: torch.Tensor,
    static_num_chunks: int,
    has_attn_sink: bool,
    _prepared: object | None = None,
) -> object:
    static_num_chunks_or_none = (
        int(static_num_chunks) if int(static_num_chunks) > 0 else None
    )
    vector_output = _merge_vector_tensor(output)
    vector_partials = _merge_vector_tensor(tmp_output)
    if not has_attn_sink:
        merge_args = (
            _to_kernel_tensor(tmp_output, _torch_to_cutlass_dtype(tmp_output.dtype)),
            _to_kernel_tensor(tmp_lse, cutlass.Float32, assumed_align=4),
            _to_kernel_tensor(num_chunks_ptr, cutlass.Int32, assumed_align=4),
            _to_kernel_tensor(output, _torch_to_cutlass_dtype(output.dtype)),
            current_cuda_stream(),
        )
        if _prepared is not None:
            return run_compiled(_prepared, merge_args)
        merge_kernel = _build_sparse_mla_split_merge_kernel(
            static_num_chunks_or_none, vector_output, vector_partials
        )
        merge_cache_key = (
            _tensor_compile_key(
                "tmp_output",
                contract_tmp_output,
                dynamic_dims=(0, 2),
                dynamic_strides=(2,),
            ),
            _tensor_compile_key(
                "tmp_lse",
                contract_tmp_lse,
                dynamic_dims=(0, 2),
                dynamic_strides=(0, 1),
            ),
            _tensor_meta_key(num_chunks_ptr),
            _tensor_compile_key(
                "output",
                contract_output,
                dynamic_dims=(0,),
            ),
            str(tmp_output.dtype),
            str(output.dtype),
            static_num_chunks_or_none,
            int(vector_output),
            int(vector_partials),
        )
        merge_spec = KernelCompileSpec.from_key(
            "attention.mla.merge",
            7,
            merge_cache_key,
            labels=(
                "tmp_output",
                "tmp_lse",
                "num_chunks_ptr",
                "output",
                "tmp_output_dtype",
                "output_dtype",
                "static_num_chunks",
                "vector_output",
                "vector_partials",
            ),
        )
        return b12x_launch(
            merge_kernel,
            compile_spec=merge_spec,
            compile_args=merge_args,
            runtime_args=merge_args,
        )

    merge_args = (
        _to_kernel_tensor(tmp_output, _torch_to_cutlass_dtype(tmp_output.dtype)),
        _to_kernel_tensor(tmp_lse, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(num_chunks_ptr, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(attn_sink, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(output, _torch_to_cutlass_dtype(output.dtype)),
        current_cuda_stream(),
    )
    if _prepared is not None:
        return run_compiled(_prepared, merge_args)
    merge_kernel = _build_sparse_mla_split_sink_merge_kernel(
        static_num_chunks_or_none, vector_output, vector_partials
    )
    merge_cache_key = (
        _tensor_compile_key(
            "tmp_output",
            contract_tmp_output,
            dynamic_dims=(0, 2),
            dynamic_strides=(2,),
        ),
        _tensor_compile_key(
            "tmp_lse",
            contract_tmp_lse,
            dynamic_dims=(0, 2),
            dynamic_strides=(0, 1),
        ),
        _tensor_meta_key(num_chunks_ptr),
        _tensor_meta_key(attn_sink),
        _tensor_compile_key(
            "output",
            contract_output,
            dynamic_dims=(0,),
        ),
        str(tmp_output.dtype),
        str(output.dtype),
        "attn_sink",
        static_num_chunks_or_none,
        int(vector_output),
        int(vector_partials),
    )
    merge_spec = KernelCompileSpec.from_key(
        "attention.mla.sink_merge",
        7,
        merge_cache_key,
        labels=(
            "tmp_output",
            "tmp_lse",
            "num_chunks_ptr",
            "attn_sink",
            "output",
            "tmp_output_dtype",
            "output_dtype",
            "kind",
            "static_num_chunks",
            "vector_output",
            "vector_partials",
        ),
    )
    return b12x_launch(
        merge_kernel,
        compile_spec=merge_spec,
        compile_args=merge_args,
        runtime_args=merge_args,
    )


@torch.library.custom_op(
    "b12x::sparse_mla_sm120_split_decode_merge",
    mutates_args=("output",),
)
def _sparse_mla_split_decode_merge_op(
    tmp_output: torch.Tensor,
    tmp_lse: torch.Tensor,
    num_chunks_ptr: torch.Tensor,
    output: torch.Tensor,
    attn_sink: torch.Tensor,
    contract_tmp_output: torch.Tensor,
    contract_tmp_lse: torch.Tensor,
    contract_output: torch.Tensor,
    static_num_chunks: int,
    has_attn_sink: bool,
) -> None:
    _sparse_mla_split_decode_merge_flat_launch(
        tmp_output,
        tmp_lse,
        num_chunks_ptr,
        output,
        attn_sink,
        contract_tmp_output,
        contract_tmp_lse,
        contract_output,
        static_num_chunks,
        has_attn_sink,
    )


@_sparse_mla_split_decode_merge_op.register_fake
def _sparse_mla_split_decode_merge_fake(
    tmp_output: torch.Tensor,
    tmp_lse: torch.Tensor,
    num_chunks_ptr: torch.Tensor,
    output: torch.Tensor,
    attn_sink: torch.Tensor,
    contract_tmp_output: torch.Tensor,
    contract_tmp_lse: torch.Tensor,
    contract_output: torch.Tensor,
    static_num_chunks: int,
    has_attn_sink: bool,
) -> None:
    return None


def run_sparse_mla_split_decode_merge(
    *,
    tmp_output: torch.Tensor | None = None,
    tmp_lse: torch.Tensor | None = None,
    num_chunks_ptr: torch.Tensor | None = None,
    num_chunks: int | None = None,
    output: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
    workspace: object | None = None,
    binding: SparseMLASplitDecodeMergeBinding | None = None,
    _prepared: object | None = None,
) -> object:
    if binding is not None:
        extras = [
            name
            for name, value in (
                ("tmp_output", tmp_output),
                ("tmp_lse", tmp_lse),
                ("num_chunks_ptr", num_chunks_ptr),
                ("num_chunks", num_chunks),
                ("output", output),
                ("attn_sink", attn_sink),
                ("workspace", workspace),
            )
            if value is not None
        ]
        if extras:
            _raise_binding_extras("run_sparse_mla_split_decode_merge", extras)
        tmp_output = binding.tmp_output
        tmp_lse = binding.tmp_lse
        num_chunks_ptr = binding.num_chunks_ptr
        num_chunks = binding.num_chunks
        output = binding.output
        attn_sink = binding.attn_sink
        workspace = binding.scratch

    tmp_output = _require_bound_arg(
        tmp_output,
        api_name="run_sparse_mla_split_decode_merge",
        name="tmp_output",
    )
    tmp_lse = _require_bound_arg(
        tmp_lse,
        api_name="run_sparse_mla_split_decode_merge",
        name="tmp_lse",
    )
    num_chunks_ptr = _require_bound_arg(
        num_chunks_ptr,
        api_name="run_sparse_mla_split_decode_merge",
        name="num_chunks_ptr",
    )
    output = _require_bound_arg(
        output,
        api_name="run_sparse_mla_split_decode_merge",
        name="output",
    )

    if tmp_output.device != output.device or tmp_lse.device != output.device:
        raise ValueError("sparse MLA merge tensors must be on the same device")
    if tmp_lse.dtype != torch.float32:
        raise TypeError(f"tmp_lse must have dtype torch.float32, got {tmp_lse.dtype}")
    if tmp_output.dtype != output.dtype:
        raise TypeError(
            f"tmp_output dtype {tmp_output.dtype} must match output dtype {output.dtype}"
        )
    if tmp_output.ndim != 4:
        raise ValueError(
            f"tmp_output must have shape [rows, heads, chunks, dim], got {tuple(tmp_output.shape)}"
        )
    if tmp_lse.ndim != 3:
        raise ValueError(
            f"tmp_lse must have shape [rows, heads, chunks], got {tuple(tmp_lse.shape)}"
        )
    if output.ndim != 3:
        raise ValueError(
            f"output must have shape [rows, heads, dim], got {tuple(output.shape)}"
        )
    if (
        int(tmp_output.shape[0]) < int(output.shape[0])
        or int(tmp_output.shape[1]) < int(output.shape[1])
        or int(tmp_output.shape[3]) < int(output.shape[2])
        or int(tmp_lse.shape[0]) < int(output.shape[0])
        or int(tmp_lse.shape[1]) < int(output.shape[1])
        or int(tmp_lse.shape[2]) < int(tmp_output.shape[2])
    ):
        raise ValueError(
            "sparse MLA merge scratch/output shapes are inconsistent: "
            f"tmp_output={tuple(tmp_output.shape)} tmp_lse={tuple(tmp_lse.shape)} "
            f"output={tuple(output.shape)}"
        )
    _validate_tensor_storage_bounds(tmp_output, name="sparse MLA merge tmp_output")
    _validate_tensor_storage_bounds(tmp_lse, name="sparse MLA merge tmp_lse")
    _validate_tensor_storage_bounds(output, name="sparse MLA merge output")
    _validate_split_control_tensor(
        num_chunks_ptr,
        name="num_chunks_ptr",
        device=output.device,
    )
    if num_chunks is not None:
        num_chunks = int(num_chunks)
        if num_chunks <= 0:
            raise ValueError(f"num_chunks must be positive, got {num_chunks}")
        if num_chunks > min(int(tmp_output.shape[2]), _SPLIT_MAX_CHUNKS):
            raise ValueError(
                "num_chunks exceeds merge scratch capacity: "
                f"{num_chunks} > min({int(tmp_output.shape[2])}, {_SPLIT_MAX_CHUNKS})"
            )
    _cto = getattr(workspace, "_contract_tmp_output", None)
    _ctl = getattr(workspace, "_contract_tmp_lse", None)
    _co = getattr(workspace, "_contract_output", None)

    has_attn_sink = attn_sink is not None
    if has_attn_sink:
        attn_sink = attn_sink.detach()
        if attn_sink.dtype != torch.float32:
            raise ValueError(
                f"attn_sink must have dtype torch.float32, got {attn_sink.dtype}"
            )
        if attn_sink.device != output.device:
            raise ValueError("attn_sink must be on the same CUDA device as output")
        if attn_sink.ndim != 1 or int(attn_sink.shape[0]) != int(output.shape[1]):
            raise ValueError(
                f"attn_sink must have shape ({int(output.shape[1])},), got {tuple(attn_sink.shape)}"
            )
        if not attn_sink.is_contiguous():
            raise ValueError("attn_sink must be contiguous for the fused merge path")

    return _sparse_mla_split_decode_merge_flat_launch(
        tmp_output,
        tmp_lse,
        num_chunks_ptr,
        output,
        attn_sink if attn_sink is not None else tmp_lse,
        _cto if _cto is not None else tmp_output,
        _ctl if _ctl is not None else tmp_lse,
        _co if _co is not None else output,
        int(num_chunks or 0),
        bool(has_attn_sink),
        _prepared,
    )


__all__ = [
    "SparseMLASplitDecodeMergeBinding",
    "build_sparse_mla_split_decode_merge_binding",
    "clear_sparse_mla_merge_kernel_cache",
    "run_sparse_mla_split_decode_merge",
]
