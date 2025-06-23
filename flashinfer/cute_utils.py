"""
Copyright (c) 2023 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
from functools import partial
import subprocess
import ctypes
from math import prod

import torch
import torch.distributed as dist

import cutlass
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cutlass._mlir.ir as ir
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Pointer, Int32
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.base_dsl.dsl import extract_mlir_values
from cutlass._mlir.dialects import scf
from cutlass._mlir.dialects import llvm
from typing import Sequence

def as_tensor(pointer, shape, torch_type):
    if torch_type.itemsize == 1:
        cytype = ctypes.c_uint8
    elif torch_type.itemsize == 2:
        cytype = ctypes.c_uint16
    elif torch_type.itemsize == 4:
        cytype = ctypes.c_uint32
    elif torch_type.itemsize == 8:
        cytype = ctypes.c_uint64
    else:
        raise ValueError(f'Unsupported torch dtype: {torch_type}')
    cpointer = ctypes.cast(pointer, ctypes.POINTER(cytype))
    arr = (cpointer._type_ * prod(shape)).from_address(
        ctypes.addressof(cpointer.contents))
    return torch.frombuffer(arr, dtype=torch_type).view(*shape)

@dsl_user_op
def multimem_ld_reduce(
    mc_ptr: Pointer,
    *,
    loc=None,
    ip=None,
):
    # ld reduce 8x f16 elts
    mc_ptr_int = mc_ptr.toint(loc=loc, ip=ip).ir_value()
    i32 = ir.IntegerType.get_signless(32)
    return_struct = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32,i32,i32,i32)>"),
        [mc_ptr_int],
        "multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.f16x2 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=True,
        asm_dialect=0,
    )
    return_regs = [
          llvm.extractvalue(i32, return_struct, [i]) for i in range(4)
    ]
    return return_regs[0], return_regs[1], return_regs[2], return_regs[3]


@dsl_user_op
def multimem_st(
    mc_ptr: Pointer,
    x: Int32,
    y: Int32,
    z: Int32,
    w: Int32,
    *,
    loc=None,
    ip=None,
):
    # st 8x f16 elts
    mc_ptr_int = mc_ptr.toint(loc=loc, ip=ip).ir_value()
    i32 = ir.IntegerType.get_signless(32)
    llvm.inline_asm(
        i32,
        [mc_ptr_int, x, y, z, w],
        "multimem.st.relaxed.sys.global.v4.f32 [$1], {$2, $3, $4, $5};",
        "=r,l,r,r,r,r",
        has_side_effects=True,
        asm_dialect=0,
    )

@dsl_user_op
def signal_multimem(
    flag_mc,
    is_relaxed=False,
    *,
    loc=None,
    ip=None,
):
    mode = "relaxed" if is_relaxed else "release"
    flag_ptr_int = flag_mc.toint().ir_value()
    llvm.inline_asm(
        None,
        [flag_ptr_int],
        f"""
        {{
            multimem.red.{mode}.sys.global.add.u32 [$0], 1;
            fence.proxy.alias;
        }}""",
        "l",
        has_side_effects=True,
        asm_dialect=0,
    )

@dsl_user_op
def wait_loop(
    flag,
    num_ranks,
    is_relaxed=False,
    *,
    loc=None,
    ip=None,
):
    mode = "relaxed" if is_relaxed else "acquire"
    flag_ptr_int = flag.toint().ir_value()
    llvm.inline_asm(
        None,
        [flag_ptr_int, num_ranks.ir_value()],
        f"""
        {{
            .reg .u32   %tmp32_<1>;
            .reg .pred  %p<1>;
            wait_signal:
                atom.global.sys.{mode}.cas.b32 %tmp32_0, [$0], $1, 0;
                setp.eq.u32 %p0, %tmp32_0, $1;
                @!%p0 bra wait_signal;
        }}""",
        "l,r",
        has_side_effects=True,
        asm_dialect=0,
    )
