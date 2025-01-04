import triton  # type: ignore[import]
import triton.language as tl  # type: ignore[import]


@triton.jit
def scale_and_clamp(x, scale, dtype):
    """Scales a value and clamps it to the range of the target dtype.

    This function hard-wires the upper/lower bounds in order to be
    compatible with both `torch.compile` and `triton.jit`.
    """
    if dtype == tl.float8e4nv:
        clamp_min = -448.0
        clamp_max = 448.0
    elif dtype == tl.float8e5:
        clamp_min = -57344.0
        clamp_max = 57344.0
    elif dtype == tl.float16:
        clamp_min = -65504.0
        clamp_max = 65504.0
    elif dtype == tl.bfloat16:
        clamp_min = -3.3895313892515355e38
        clamp_max = 3.3895313892515355e38
    else:
        tl.static_assert(False, f"Unsupported dtype: {dtype}")

    return tl.clamp(x.to(tl.float32) * scale, clamp_min, clamp_max).to(dtype)
