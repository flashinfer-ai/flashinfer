"""IQ2_XS stage records across supported CTA geometries and live row counts."""

from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.moe import fused_moe as moe
from b12x.moe._shared.kernels.w4a16 import kernel
from b12x.moe._shared.kernels.w4a16.host import make_w4a16_packed_buffers
from benchmarks.experimental.b12x.benchmark_iq2_xs_moe import (
    capture_lifetime,
    check,
    make_inputs,
    prepare_experts,
    reference,
)
from benchmarks.experimental.b12x.iq2_xs_checkpoint import IQ2XSLayer
from b12x.testing.reference.helpers import require_b12x
from tests.experimental.b12x.moe.test_iq2_xs import blocks


@pytest.mark.parametrize("tile_k,tile_n", [(64, 128), (128, 128), (256, 64), (64, 256)])
@pytest.mark.parametrize("codec", ["iq2_xs", "iq2_xxs", "q8_0"])
def test_stage_records_with_alternate_tiles(tile_k, tile_n, codec):
    device = require_b12x()
    layer = IQ2XSLayer(
        moe.BlockQuantWeights(blocks(e=8, n=1280, k=1024, codec=codec), blocks(e=8, n=1024, k=1280, codec=codec), codec=codec),
        1024, 1280, 8, 2, tuple(range(8)), Path("synthetic-iq2-xs"), 0, 1, 0,
    )
    experts, _ = prepare_experts(layer, device, activation="relu2")
    prepared = experts._impl.representation.value
    buffers = make_w4a16_packed_buffers(
        prepared, m=8, topk=2, dtype=torch.bfloat16, device=device, block_size_m=8,
    )
    workspace = (
        buffers.intermediate_cache13, buffers.intermediate_cache2,
        buffers.fc1_c_tmp, buffers.fc2_c_tmp,
    )
    pointers = tuple(t.data_ptr() for t in workspace)
    props = torch.cuda.get_device_properties(device)
    with (
        pytest.raises(ValueError, match="force_tile_config fc1 tile .* does not fit")
        if tile_k > 128 else nullcontext()
    ):
        fused = kernel.compile_w4a16_fused_moe(
            size_m=8, hidden_size=1024, intermediate_size=1280, num_experts=8,
            top_k=2, activation="relu2", apply_router_weight_on_input=False,
            zero_fc2_output=False, moe_block_size=8, max_m_blocks=16,
            element_dtype="bf16", sms=props.multi_processor_count,
            max_shared_mem=props.shared_memory_per_block_optin,
            weight_layout=codec, scale_format=codec, w13_layout="packed",
            direct_topk_routes=True, tc_decode_fused_sum=True,
            force_tile_config=(tile_k, tile_n, tile_k, tile_n),
        )
    if tile_k > 128:
        return

    def forbidden(*args, **kwargs):
        raise AssertionError("IQ2_XS stage-record replay attempted compilation")

    with (
        kernel_resolution_guard("IQ2_XS alternate stage tiles"),
        patch.object(kernel, "compile_w4a16_fused_moe", forbidden),
        patch.object(kernel, "compile_w4a16_topk_sum", forbidden),
    ):
        for rows in (1, 2, 4, 8):
            inputs = make_inputs(layer, rows, device, mapped=False)
            expected = reference(layer, inputs, activation="relu2")

            def run():
                return kernel.run_w4a16_moe(
                    inputs.x, prepared, inputs.probabilities, inputs.ids,
                    activation="relu2", output=inputs.output,
                    intermediate_cache13=buffers.intermediate_cache13,
                    intermediate_cache2=buffers.intermediate_cache2,
                    fc1_c_tmp=buffers.fc1_c_tmp, fc2_c_tmp=buffers.fc2_c_tmp,
                    route_mode="direct", route_block_size_m=8, fused_launch=fused,
                )

            inputs.output.fill_(float("nan"))
            actual = run()
            assert torch.isfinite(actual).all(), f"nonfinite output at rows={rows}"
            check(actual, expected)
            graph = torch.cuda.CUDAGraph()
            try:
                with capture_lifetime(), torch.cuda.graph(graph):
                    run()
                for _ in range(3):
                    inputs.output.fill_(float("nan"))
                    graph.replay()
                    check(inputs.output, expected)
                assert tuple(t.data_ptr() for t in workspace) == pointers
            finally:
                graph.reset()
