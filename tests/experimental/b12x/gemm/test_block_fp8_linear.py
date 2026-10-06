"""Prepared block-FP8 linear respects source precision and rewrites scratch."""
from __future__ import annotations

import pytest
import torch

from b12x._lib.intrinsics import quant_dequant_mxfp8_torch
from b12x.gemm import block_fp8_linear as bfl
from b12x.gemm._shared.wo_mxfp8 import empty_dense_gemm_mnl_view
from b12x.preparation import PreparationSession, PreparedCall
from ..conftest import require_b12x


def test_block_fp8_functional_quantization_requires_prepared_plan():
    source = torch.empty((1, 128), dtype=torch.bfloat16)
    with pytest.raises(TypeError):
        bfl.quantize_input(source, plan=object())


@pytest.mark.parametrize("rows,source_dtype,output_dtype", [
    (17, torch.bfloat16, torch.float16),
    (3, torch.float16, torch.bfloat16),
    (3, torch.bfloat16, torch.bfloat16),
])
@torch.inference_mode()
def test_prepared_block_fp8_rewrites_poisoned_scratch_on_graph_replay(rows, source_dtype, output_dtype):
    device = require_b12x()
    torch.manual_seed(7214)
    k, n = 256, 256
    source = (torch.randn((rows, k), device=device) / 4).to(source_dtype)
    weight = (torch.randn((n, k), device=device) / 4).to(torch.float8_e4m3fn)
    weight_scale = torch.full((n // 128, k // 128), .125, device=device)
    packed = bfl.pack_weight(weight, weight_scale)
    output = empty_dense_gemm_mnl_view(rows, n, 1, device=source.device, dtype=output_dtype)
    bias = torch.randn(n, device=device, dtype=output_dtype) / 8
    caps = bfl.Caps(device=source.device, max_tokens=rows, in_features=k,
                    out_features=n, source_dtype=source_dtype, output_dtype=output_dtype)
    declaration = bfl.plan(caps)
    owned = {}

    def prepare(state):
        (spec,) = state.scratch.scratch_specs()
        owned["scratch"] = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        binding = state.bind(scratch=owned["scratch"], source=source,
                             packed_weight=packed, output=output, bias=bias)
        return PreparedCall(run=lambda: state.run_binding(binding), output=output)

    def reference():
        activation = quant_dequant_mxfp8_torch(source).float()
        dequantized_weight = weight.float() * .125
        return (activation @ dequantized_weight.T).to(output_dtype) + bias

    request = declaration.request(name="linear", prepare_call=prepare)
    with PreparationSession(device=source.device, autotune=False, compile_workers=2) as session:
        session.prepare((request,))
        binding = bfl.bind(declaration, scratch=owned["scratch"],
                           source=source, packed_weight=packed, output=output, bias=bias)
        owned["scratch"].fill_(255)
        output.fill_(float("nan"))
        actual = bfl.run(binding=binding)
        torch.testing.assert_close(actual, reference(), rtol=2e-2, atol=2e-2)
        graph = torch.cuda.CUDAGraph()
        try:
            with session.capture(), torch.cuda.graph(graph):
                replay_output = bfl.run(binding=binding)
            source.copy_((torch.randn_like(source.float()) / 4).to(source_dtype))
            owned["scratch"].fill_(255)
            output.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize(source.device)
            assert torch.isfinite(replay_output).all()
            torch.testing.assert_close(replay_output, reference(), rtol=2e-2, atol=2e-2)
        finally:
            graph.reset()
