import triton  # type: ignore[import]
import triton.language as tl  # type: ignore[import]

from flashinfer.triton.kernels.quant import scale_and_clamp


@triton.jit
def silu_and_mul_kernel(
    o_ptr,
    o_stride,
    o_scale_ptr,
    x_ptr,
    x_stride,
    x_scale_ptr,
    d,
    BLOCK_SIZE: tl.constexpr,
    HAS_X_SCALE: tl.constexpr,
    HAS_O_SCALE: tl.constexpr,
) -> None:
    """Sigmoid Linear Unit and Multiplication Kernel

    Args:
        o_ptr:       Pointer to the 2D output tensor.
        o_stride:    Output tensor stride.
        o_scale_ptr: The optional, known scale of the output activations.
        x_ptr:       Pointer to the 2D input tensor.
        x_stride:    Input tensor stride.
        x_scale_ptr: The optional, known scale of the input tensor.
        d:           The number of elements along the second dimension.
        BLOCK_SIZE:  Tunable block size to process in each kernel.

    Operating on a 2D grid, computes the following:

    ```
    out[i, j] = sigmoid(x[i, j]) * x[i, j] * x[i, j + d]
    ```

    If scales are provided, the input and output tensors are scaled.
    """

    i = tl.program_id(axis=0).to(tl.int64)
    j = tl.program_id(axis=1)

    o_row_ptr = o_ptr + o_stride * i
    x_row_ptr = x_ptr + x_stride * i

    offsets = j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < d

    a = tl.load(x_row_ptr + offsets, mask=mask).to(tl.float32)
    b = tl.load(x_row_ptr + offsets + d, mask=mask).to(tl.float32)

    if HAS_X_SCALE:
        x_scale = tl.load(x_scale_ptr)
        a *= x_scale
        b *= x_scale

    result = tl.sigmoid(a) * a * b

    if HAS_O_SCALE:
        o_scale = tl.load(o_scale_ptr)
        result = scale_and_clamp(result, o_scale, o_ptr.dtype.element_ty)

    tl.store(o_row_ptr + offsets, result, mask=mask)
