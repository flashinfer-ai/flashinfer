import pytest
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_ptr

from flashinfer.cute_dsl.utils import is_cute_dsl_available


@cute.kernel
def copy_torch_fp4_tensor_kernel(a_ptr: cute.Pointer, b_ptr: cute.Pointer):
    a = cute.make_tensor(a_ptr, layout=cute.make_ordered_layout((3, 8), order=(1, 0)))
    b = cute.make_tensor(b_ptr, layout=cute.make_ordered_layout((3, 8), order=(1, 0)))
    a = cute.recast_tensor(a, cutlass.Uint8)
    b = cute.recast_tensor(b, cutlass.Uint8)
    cute.print_tensor(a)
    b.store(a.load())


@cute.jit
def copy_torch_fp4_tensor(a_ptr: cute.Pointer, b_ptr: cute.Pointer):
    copy_torch_fp4_tensor_kernel(a_ptr, b_ptr).launch(grid=(1, 1, 1), block=(1, 1, 1))


def test_fp4_tensor_torch_cute():
    if not is_cute_dsl_available():
        pytest.skip("cute-dsl is not available")

    a = torch.randint(
        0, 128, size=(3, 4), dtype=torch.uint8, device=torch.device("cuda:0")
    )
    b = torch.zeros_like(a)
    a_view = a.view(torch.float4_e2m1fn_x2)
    b_view = b.view(torch.float4_e2m1fn_x2)
    print(f"a_view: \n{a_view}")
    print("")

    a_ptr = make_ptr(
        cutlass.Float4E2M1FN,
        a_view.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    b_ptr = make_ptr(
        cutlass.Float4E2M1FN,
        b_view.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    copy_torch_fp4_tensor(a_ptr, b_ptr)
    torch.testing.assert_close(a, b)
    print("Results verified successfully!")
    print(f"Result: \n{b_view}")
