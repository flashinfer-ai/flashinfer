"""CuTe row copy; model width/types are static, all row quantities are runtime."""
from b12x._lib.program_cache import program_cache
from b12x._lib.compile_plan import attach_programs
from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint16, Uint32
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm

from ..._lib.intrinsics import q8_0_pair_to_bf16x2
from ..._lib.compiler import KernelCompileSpec, compile as compile_cute, run_compiled
from ..._lib.runtime_control import raise_if_kernel_resolution_frozen
from ..._lib.utils import current_cuda_stream, make_ptr


@dsl_user_op
def _invalid_index(*, loc=None, ip=None):
    llvm.inline_asm(None, [], "trap;", "", has_side_effects=True,
                    is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
                    loc=loc, ip=ip)


class _Embedding:
    def __init__(self, width, q8=False):
        self.width = width
        self.q8 = q8

    @cute.jit
    def __call__(self, weight: cute.Pointer, ids: cute.Pointer, out: cute.Pointer,
                 count: cute.Pointer, capacity: Int32, table_rows: Int64,
                 row_stride: Int64, use_count: Int32, stream: cuda.CUstream):
        self.kernel(weight, ids, out, count, capacity, table_rows,
                    row_stride, use_count).launch(
            grid=(capacity, 1, 1), block=(128, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, weight: cute.Pointer, ids: cute.Pointer, out: cute.Pointer,
               count: cute.Pointer, capacity: Int32, table_rows: Int64,
               row_stride: Int64, use_count: Int32):
        row, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        live = capacity
        if use_count != 0:
            live = count[0]
        if live < 0 or live > capacity:
            _invalid_index()
        else:
            if row < live:
                index = Int64(ids[row])
                if index < 0 or index >= table_rows:
                    _invalid_index()
                else:
                    src = index * row_stride
                    dst = Int64(row) * Int64(self.width)
                    if cutlass.const_expr(self.q8):
                        words = cute.recast_ptr(weight, dtype=Uint16)
                        pairs = cute.recast_ptr(out, dtype=Uint32)
                        for item in cutlass.range_constexpr((self.width // 2 + 127) // 128):
                            col = tid + item * 128
                            if col < self.width // 2:
                                block = src // 2 + Int64(col // 16) * 17
                                pairs[dst // 2 + Int64(col)] = q8_0_pair_to_bf16x2(
                                    Uint32(words[block + 1 + Int64(col % 16)]), Uint32(words[block]))
                    else:
                        for item in cutlass.range_constexpr((self.width + 127) // 128):
                            col = tid + item * 128
                            if col < self.width:
                                out[dst + Int64(col)] = weight[src + Int64(col)]



@dataclass(frozen=True)
class _EmbeddingProgram:
    raw: object
    types: tuple


@program_cache
def compile_embedding(width, weight_dtype, id_dtype, device):
    key = (width, str(weight_dtype), str(id_dtype), device)
    q8 = weight_dtype == torch.uint8
    entry = _Embedding(width, q8)
    raise_if_kernel_resolution_frozen("cute.compile", target=entry, cache_key=key)
    value_type = BFloat16 if weight_dtype in (torch.bfloat16, torch.uint8) else Float32
    index_type = Int32 if id_dtype == torch.int32 else Int64
    types = (Uint8 if q8 else value_type, index_type, value_type, Int32)
    pointers = tuple(make_ptr(t, 16, cute.AddressSpace.gmem,
                              assumed_align=t.width // 8) for t in types)
    with torch.cuda.device(device):
        compiled = compile_cute(entry, *pointers, Int32(1), Int64(1),
                                Int64(width), Int32(0), current_cuda_stream(),
                                compile_spec=KernelCompileSpec.from_key(
                                    "sequence.embedding", 2, key))
    return attach_programs(_EmbeddingProgram(compiled, types), compiled)


def launch(weight, ids, out, num_rows, *, prepared):
    if ids.numel() == 0:
        return
    compiled, types = prepared
    tensors = (weight, ids, out, ids if num_rows is None else num_rows)
    pointers = tuple(make_ptr(t, tensor.data_ptr(), cute.AddressSpace.gmem,
                              assumed_align=t.width // 8)
                     for t, tensor in zip(types, tensors, strict=True))
    with torch.cuda.device(weight.device):
        run_compiled(compiled, (*pointers, Int32(ids.numel()),
                                Int64(weight.shape[0]), Int64(weight.stride(0)),
                                Int32(num_rows is not None), current_cuda_stream()))
