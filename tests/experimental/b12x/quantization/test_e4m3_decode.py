"""Check scalar E4M3 decoding across every encoded value."""

import cutlass.cute as cute
from cutlass import Int32, Uint32
from cutlass.cute.runtime import from_dlpack
import torch

from b12x._lib.compiler import KernelCompileSpec, compile as compile_kernel
from b12x._lib.intrinsics import fp8_e4m3_to_f32, fp8_e4m3_to_f32_and_rcp
from b12x._lib.utils import current_cuda_stream

from ..conftest import require_b12x


class _Decode:
    @cute.jit
    def __call__(self, source, decoded, reciprocal, count: Int32, stream):
        self.kernel(source, decoded, reciprocal, count).launch(
            grid=[cute.ceil_div(count, 128), 1, 1], block=[128, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, source, decoded, reciprocal, count: Int32):
        thread, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        index = Int32(block) * 128 + Int32(thread)
        if index < count:
            value = Uint32(source[index])
            decoded[index] = fp8_e4m3_to_f32(value)
            reciprocal[index] = fp8_e4m3_to_f32_and_rcp(value)


def test_e4m3_decode_and_reciprocal_all_codes():
    device = require_b12x()
    source = torch.arange(256, device=device).to(torch.uint8)
    decoded = torch.empty(256, dtype=torch.float32, device=device)
    reciprocal = torch.empty_like(decoded)
    arrays = tuple(from_dlpack(t, assumed_align=16) for t in (source, decoded, reciprocal))
    stream = current_cuda_stream()
    kernel = compile_kernel(_Decode(), *arrays, Int32(256), stream,
        compile_spec=KernelCompileSpec.from_fields('test.e4m3-scalar-decode', 1,
                                                  ('capacity', 256)))
    kernel(*arrays, Int32(256), stream)
    torch.cuda.synchronize()
    reference = source.view(torch.float8_e4m3fn).float()
    torch.testing.assert_close(decoded, reference, rtol=0, atol=0, equal_nan=True)
    # NaN conversion may canonicalize its sign; finite values retain it.
    finite = reference.isfinite()
    assert torch.equal(torch.signbit(decoded[finite]), torch.signbit(reference[finite]))
    expected_reciprocal = torch.where(reference == 0, 0., reference.reciprocal())
    torch.testing.assert_close(reciprocal, expected_reciprocal,
                               rtol=2e-7, atol=0, equal_nan=True)
