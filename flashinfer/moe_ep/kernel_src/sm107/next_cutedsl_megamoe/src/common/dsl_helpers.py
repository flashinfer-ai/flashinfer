"""Tensor annotations shared by kernel families."""

import cutlass.cute as cute


@cute.jit
def mark_alignment(tensor: cute.Tensor, byte_alignment: int) -> cute.Tensor:
    pointer = tensor.iterator
    return cute.make_tensor(
        cute.make_ptr(pointer.dtype, pointer.toint(), pointer.memspace, assumed_align=byte_alignment), tensor.layout
    )
