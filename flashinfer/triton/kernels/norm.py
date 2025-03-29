import triton  # type: ignore[import]
import triton.language as tl  # type: ignore[import]

from flashinfer.triton.kernels.quant import scale_and_clamp


@triton.jit
def rms_norm_kernel(
    n,
    b,
    x_ptr,
    x_stride,
    x_scale_ptr,
    r_ptr,
    r_stride,
    w_ptr,
    o_ptr,
    o_stride,
    o_scale_ptr,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_IN_SCALE: tl.constexpr,
    HAS_OUT_SCALE: tl.constexpr,
    HAS_OUTPUT: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
) -> None:
    i = tl.program_id(axis=0).to(tl.int64)

    # If r_ptr is present, the input to norm is x + r.
    x_row = x_ptr + i * x_stride
    o_row = o_ptr + i * o_stride if HAS_OUTPUT else x_row
    r_row = r_ptr + i * r_stride if HAS_RESIDUAL else None

    x_scale = tl.load(x_scale_ptr) if HAS_IN_SCALE else None
    o_scale = tl.load(o_scale_ptr) if HAS_OUT_SCALE else None

    # Find the root mean square for the given row.
    square_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, n, BLOCK_SIZE):
        offsets = off + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n

        x = tl.load(x_row + offsets, mask=mask, other=0.0).to(tl.float32)
        if HAS_IN_SCALE:
            x *= x_scale

        if HAS_RESIDUAL:
            r = tl.load(r_row + offsets, mask=mask, other=0.0).to(tl.float32)
            x += r
            tl.store(r_row + offsets, x, mask=mask)

        square_sum += x * x

    # Compute the norm.
    rms = tl.rsqrt(tl.sum(square_sum) / n + EPS)

    # x[i] = r[i] + x[i] / rms * weight[i]
    output_dtype = o_row.dtype.element_ty
    for off in range(0, n, BLOCK_SIZE):
        offsets = off + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n

        if HAS_RESIDUAL:
            x = tl.load(r_row + offsets, mask=mask).to(tl.float32)
        else:
            x = tl.load(x_row + offsets, mask=mask).to(tl.float32)
            if HAS_IN_SCALE:
                x *= x_scale

        w = tl.load(w_ptr + offsets, mask=mask).to(tl.float32)

        # Multiply x with RMS on float32, but cast to the narrower type before
        # multiplying with the weights to replicate the HF behaviour precisely.
        result = w * (x * rms)
        if HAS_OUT_SCALE:
            result = scale_and_clamp(result, o_scale, output_dtype)
        tl.store(o_row + offsets, result, mask=mask)
