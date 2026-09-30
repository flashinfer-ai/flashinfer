"""quantization.nvfp4: the BF16 -> NVFP4 TMA tile quantizer matches the
pure-torch grouped quantizer (the blockscaled GEMM tests' operand builder)
bitwise on both packed values and MMA-layout scales.

The op was dead at the original port baseline and revived by the CUTLASS
DSL 4.6 migration.
"""

from __future__ import annotations

import torch

from b12x._lib.intrinsics import quantize_grouped_nvfp4_torch
from b12x.quantization import nvfp4
from b12x.preparation import PreparationSession, PreparedCall

from ..conftest import require_b12x


def test_run_matches_torch_grouped_quantizer() -> None:
    require_b12x()
    torch.manual_seed(20260720)

    m, k = 256, 512
    source = (torch.randn((m, k), device="cuda", dtype=torch.bfloat16) / 4).contiguous()
    tensor_amax = source.abs().max().to(torch.float32)
    global_scale = (torch.finfo(torch.float8_e4m3fn).max * 6.0 / tensor_amax).reshape(1)

    row_counts = torch.full((1,), m, dtype=torch.int32, device="cuda")
    ref_packed, ref_scale_view = quantize_grouped_nvfp4_torch(
        source.unsqueeze(0), row_counts, global_scale
    )
    # Invert as_grouped_scale_view's logical permutation to recover the
    # physical contiguous scale-storage contract the TMA kernel writes
    # (mirrors b12x tests/test_bf16_to_fp4_tma.py).
    ref_scales = ref_scale_view.permute(5, 2, 4, 0, 1, 3).contiguous().view(torch.uint8)

    declaration = nvfp4.plan(m, k)
    outputs = nvfp4.allocate_outputs(declaration)
    request = declaration.request(
        name="quantize",
        prepare_call=lambda state: PreparedCall(
            run=lambda: state.run(source, global_scale, outputs),
        ),
    )
    with PreparationSession(device=source.device, autotune=False) as session:
        session.prepare((request,))
        nvfp4.run(plan=declaration, x=source,
                  global_scale=global_scale, outputs=outputs)
        torch.cuda.synchronize()

    actual_packed = outputs.packed_a_storage.view(-1)
    expected_packed = ref_packed.contiguous().view(torch.uint8).view(-1)
    assert actual_packed.numel() == expected_packed.numel()
    assert int(torch.count_nonzero(actual_packed).item()) > 0
    assert torch.equal(actual_packed, expected_packed)

    actual_scales = outputs.scale_flat
    expected_scales = ref_scales.view(-1)
    assert actual_scales.numel() == expected_scales.numel()
    assert torch.equal(actual_scales, expected_scales)
