# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Public SiTU backend correctness and caller-owned graph/workspace behavior."""

import re

import pytest
import torch

from flashinfer.fused_moe import (
    QuantConfig,
    QuantFormat,
    SiTU,
    TrtllmFp4Config,
    cutlass_fused_moe,
    cake_fused_moe_prepare_workspace,
    cutlass_fused_moe_workspace_size,
    trtllm_fp4_block_scale_routed_moe,
)
from flashinfer.fused_moe.cake_kimi_k3_situ import _N8_W2A_M16_FC2_GRID_N_SM_FACTOR
from flashinfer.jit.cake_kimi_k3_situ import PROGRAMS, ROUTES, _source_path
from flashinfer.tllm_enums import ActivationType, RoutingMethodType
from flashinfer.utils import device_support_pdl


HIDDEN, INTERMEDIATE, EXPERTS, TOP_K = 3584, 384, 896, 16
NVFP4_QUANT = QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4)
ARCH_BY_CAPABILITY = {(10, 0): "sm_100a", (10, 3): "sm_103a"}


def _small_row_selector(arch, num_tokens):
    # Route selectors of the small token counts: the two-CTA-per-SM FC2 route
    # for 16 tokens on SM103, the single-token route for 1 token and the fused
    # quantization + router route for 8 tokens (and 16 tokens on SM100).
    if num_tokens == 16 and arch == "sm_103a":
        return "n8_w2a_m16"
    if num_tokens == 1:
        return "m1"
    if num_tokens in (8, 16):
        return "n8_feature"
    return None


def _declares_fused_quant_route(program_key):
    return any(
        name == "quant_route.s2b_num_tokens"
        for _, name in PROGRAMS[program_key]["arg_plan"]
    )


def _stage_kernel_source(program_key, stage):
    # A prepared launch sequence binds one generated kernel per stage inside a
    # `namespace stage_<name> { ... }` block of its binding translation unit.
    record = PROGRAMS[program_key]
    binding = _source_path(record["sources"][-1]).read_text()
    block = re.search(
        rf"namespace stage_{stage} \{{(.*?)\}}  // namespace stage_{stage}",
        binding,
        re.S,
    )
    assert block is not None, (program_key, stage)
    names = set(
        re.findall(
            r"kernel_(cake_kimi_k3_nvfp4_situ_routed_moe_[0-9a-f]{20})", block[1]
        )
    )
    (name,) = names
    (kernel,) = (
        source for source in record["sources"] if source.endswith(f"/{name}_kernel.cu")
    )
    return _source_path(kernel).read_text()


def _fc2_launch_footprint(arch, selector):
    source = _stage_kernel_source(ROUTES[(arch, selector)], "fc2")
    threads, min_blocks = map(
        int, re.search(r"__launch_bounds__\((\d+),\s*(\d+)\)", source).groups()
    )
    smem_bytes = int(re.search(r"#define SMEM_TOTAL (\d+)", source).group(1))
    return threads, min_blocks, smem_bytes


def _workspace_size(**overrides):
    options = dict(
        max_num_tokens=2048,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_experts_total=EXPERTS,
        top_k=TOP_K,
        x_dtype=torch.bfloat16,
        weight_dtype=torch.uint8,
        activation_type=ActivationType.Situ,
        tp_size=8,
        backend="cake",
    )
    options.update(overrides)
    return cutlass_fused_moe_workspace_size(**options)


@pytest.mark.parametrize(
    "options, message",
    [
        ({"backend": "unknown"}, "unsupported.*backend"),
        ({"activation_type": ActivationType.Swiglu}, "activation_type"),
        ({"top_k": 8}, "top_k=16"),
    ],
)
def test_cake_situ_rejects_unsupported_dispatch(options, message):
    # This public size query must reject unsupported semantics using metadata.
    with pytest.raises(ValueError, match=message):
        _workspace_size(**options)


def test_cake_situ_workspace_size_holds_every_smaller_shape():
    # The per-shape layout is not monotonic in the token count (32 and 64 force
    # tile-N16 and need more scratch than the tile-N8 counts after them), but
    # the public size query must still honor the documented contract that a
    # maximum-size buffer holds every smaller prepared shape.
    sizes = [_workspace_size(max_num_tokens=n) for n in range(1, 16385)]
    running_max = 0
    for size in sizes:
        running_max = max(running_max, size)
        assert size == running_max
    assert _workspace_size(max_num_tokens=40) >= _workspace_size(max_num_tokens=32)
    assert _workspace_size(max_num_tokens=100) >= _workspace_size(max_num_tokens=64)


# SM100 and SM103 shared memory per SM and the per-CTA reservation (CUDA C
# Programming Guide, compute capability 10.x), and the register file per SM.
SMEM_PER_SM_BYTES = 228 * 1024
SMEM_RESERVED_PER_CTA_BYTES = 1024
REGISTERS_PER_SM = 65536


def test_cake_situ_two_cta_fc2_program_fits_two_ctas_per_sm():
    # The 16-token SM103 route sizes its FC2 device-workfeed pool as
    # _N8_W2A_M16_FC2_GRID_N_SM_FACTOR CTAs per SM. That is only correct if the
    # generated FC2 program really fits that many CTAs per SM, so pin the
    # program's launch bounds and shared-memory footprint to the host factor.
    threads, min_blocks, smem_bytes = _fc2_launch_footprint("sm_103a", "n8_w2a_m16")
    assert (threads, min_blocks) == (512, _N8_W2A_M16_FC2_GRID_N_SM_FACTOR)
    assert min_blocks * (smem_bytes + SMEM_RESERVED_PER_CTA_BYTES) <= SMEM_PER_SM_BYTES
    # __launch_bounds__(512, 2) caps ptxas at this many registers per thread.
    assert REGISTERS_PER_SM // (threads * min_blocks) >= 64
    # The single-CTA FC2 program of the other small routes does not fit twice.
    for arch, selector in (("sm_103a", "n8_feature"), ("sm_100a", "n8_feature")):
        threads, min_blocks, smem_bytes = _fc2_launch_footprint(arch, selector)
        assert (threads, min_blocks) == (512, 1)
        assert 2 * (smem_bytes + SMEM_RESERVED_PER_CTA_BYTES) > SMEM_PER_SM_BYTES


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a"])
@pytest.mark.parametrize("num_tokens", [1, 8, 16], ids=["m1", "m8", "m16"])
def test_cake_situ_small_row_routes_declare_fused_quant_route(arch, num_tokens):
    # The fused quantization + router stage is part of the 8- and 16-token
    # programs on both architectures and of no other small-row program.
    program_key = ROUTES[(arch, _small_row_selector(arch, num_tokens))]
    assert _declares_fused_quant_route(program_key) is (num_tokens in (8, 16))


def test_cake_situ_two_cta_fc2_route_is_sixteen_tokens_on_sm103_only():
    # Only 16 tokens on SM103 reach the two-CTA-per-SM FC2 route: SM100 has no
    # such route and 8 tokens on SM103 keep the fused-router route.
    two_cta = ROUTES[("sm_103a", "n8_w2a_m16")]
    assert ("sm_100a", "n8_w2a_m16") not in ROUTES
    assert _small_row_selector("sm_100a", 16) == "n8_feature"
    assert ROUTES[("sm_100a", "n8_feature")] != two_cta
    assert _small_row_selector("sm_103a", 8) == "n8_feature"
    assert ROUTES[("sm_103a", "n8_feature")] != two_cta
    assert two_cta not in (ROUTES[("sm_103a", 8)], ROUTES[("sm_103a", 16)])


@pytest.fixture(scope="module")
def cake_situ_device():
    if not torch.cuda.is_available():
        pytest.skip("Cake SiTU requires a CUDA device")
    device = torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.get_device_capability(device) not in {(10, 0), (10, 3)}:
        pytest.skip("Cake SiTU requires SM100 or SM103")
    return device


@pytest.fixture(scope="module")
def cake_situ_weights(cake_situ_device):
    # Use the existing public preparation, including the SiTU gate/up shuffle.
    # Keep one full expert fixture for all token counts instead of repeatedly
    # quantizing the same multi-GiB weights.
    with pytest.MonkeyPatch.context() as patch:
        for name, value in {
            "FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH": "1",
            "FLASHINFER_NVFP4_4OVER6": "0",
            "FLASHINFER_NVFP4_4OVER6_ERR_MODE": "MAE",
            "FLASHINFER_NVFP4_4OVER6_ERR_USE_FAST_MATH": "0",
            "FLASHINFER_NVFP4_4OVER6_E4M3_USE_256": "0",
        }.items():
            patch.setenv(name, value)
        generator = torch.Generator(device=cake_situ_device).manual_seed(4568)
        w1 = torch.randn(
            EXPERTS,
            2 * INTERMEDIATE,
            HIDDEN,
            device=cake_situ_device,
            dtype=torch.bfloat16,
            generator=generator,
        ).mul_(0.125)
        w2 = torch.randn(
            EXPERTS,
            HIDDEN,
            INTERMEDIATE,
            device=cake_situ_device,
            dtype=torch.bfloat16,
            generator=generator,
        ).mul_(0.125)
        prepared = TrtllmFp4Config.prepare_weights(
            w1,
            w2,
            quant=NVFP4_QUANT,
            num_local_experts=EXPERTS,
            hidden_size=HIDDEN,
            intermediate_size=INTERMEDIATE,
            activation=SiTU(),
        )
        del w1, w2
        yield prepared


@pytest.fixture(scope="module")
def cake_situ_workspace(cake_situ_device):
    # A single maximum-size allocation is reused across all four token sizes.
    return torch.empty(_workspace_size(), dtype=torch.uint8, device=cake_situ_device)


@pytest.mark.parametrize(
    "max_num_tokens, num_tokens",
    [(40, 32), (100, 64)],
    ids=["n40_holds_32", "n100_holds_64"],
)
def test_cake_situ_prepare_smaller_shape_in_maximum_size_buffer(
    max_num_tokens,
    num_tokens,
    cake_situ_device,
):
    # Allocating for a token count strictly between two forced tile-N16 counts
    # must still allow preparing the smaller forced count in the same buffer.
    workspace = torch.empty(
        _workspace_size(max_num_tokens=max_num_tokens),
        dtype=torch.uint8,
        device=cake_situ_device,
    )
    for tokens in (num_tokens, max_num_tokens):
        assert (
            cake_fused_moe_prepare_workspace(
                workspace,
                tokens,
                backend="cake",
                weight_layout="trtllm_shuffled_nvfp4_group16",
            )
            is workspace
        )


def _trtllm_reference(x, ids, route_weights, prepared):
    quantized, scales = TrtllmFp4Config.prepare_activations(
        x,
        quant=NVFP4_QUANT,
    )
    output = torch.empty_like(x)
    result = trtllm_fp4_block_scale_routed_moe(
        topk_ids=(ids, route_weights),
        routing_bias=None,
        hidden_states=quantized,
        hidden_states_scale=scales,
        gemm1_weights=prepared["gemm1_weights"],
        gemm1_weights_scale=prepared["gemm1_weights_scale"],
        gemm1_bias=None,
        gemm1_alpha=prepared["gemm1_alpha"],
        gemm1_beta=prepared["gemm1_beta"],
        gemm1_clamp_limit=None,
        gemm2_weights=prepared["gemm2_weights"],
        gemm2_weights_scale=prepared["gemm2_weights_scale"],
        gemm2_bias=None,
        output1_scale_scalar=prepared["output1_scale_scalar"],
        output1_scale_gate_scalar=prepared["output1_scale_gate_scalar"],
        output2_scale_scalar=prepared["output2_scale_scalar"],
        num_experts=EXPERTS,
        top_k=TOP_K,
        n_group=None,
        topk_group=None,
        intermediate_size=INTERMEDIATE,
        local_expert_offset=0,
        local_num_experts=EXPERTS,
        routed_scaling_factor=None,
        routing_method_type=RoutingMethodType.Renormalize.value,
        activation_type=ActivationType.Situ.value,
        do_finalize=True,
        enable_pdl=device_support_pdl(x.device),
        per_token_scale=None,
        output=output,
        tune_max_num_tokens=16384,
    )
    if isinstance(result, (list, tuple)):
        assert len(result) == 1
        result = result[0]
    assert result.data_ptr() == output.data_ptr()
    return output


@pytest.mark.parametrize(
    "num_tokens",
    [1, 8, 16, 64, 256, 512, 2048],
    ids=["m1", "m8", "m16", "n8", "n16", "n32", "n128"],
)
def test_cake_situ_output_workspace_and_external_graph(
    num_tokens,
    cake_situ_device,
    cake_situ_weights,
    cake_situ_workspace,
):
    device, prepared, workspace = (
        cake_situ_device,
        cake_situ_weights,
        cake_situ_workspace,
    )
    generator = torch.Generator(device=device).manual_seed(9000 + num_tokens)
    x = torch.randn(
        num_tokens,
        HIDDEN,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    slots = torch.arange(TOP_K, dtype=torch.int32, device=device)
    tokens = torch.arange(num_tokens, dtype=torch.int32, device=device)
    ids = ((tokens[:, None] * TOP_K + slots[None, :]) % EXPERTS).contiguous()
    route_weights = (
        torch.randn(
            num_tokens,
            TOP_K,
            device=device,
            generator=generator,
        )
        .softmax(dim=-1)
        .to(torch.bfloat16)
    )
    output = torch.full_like(x, float("nan"))
    output_ptr, workspace_ptr = output.data_ptr(), workspace.data_ptr()
    quant_scales = [
        torch.ones(1, device=device, dtype=torch.float32),
        prepared["gemm1_weights_scale"],
        prepared["output1_scale_gate_scalar"],
        prepared["output1_scale_scalar"],
        prepared["gemm2_weights_scale"],
        prepared["output2_scale_scalar"],
    ]

    def submit(*, explicit_parameters=False):
        activation_kwargs = (
            {
                "situ_beta": prepared["gemm1_alpha"],
                "situ_linear_beta": prepared["gemm1_beta"],
            }
            if explicit_parameters
            else {}
        )
        return cutlass_fused_moe(
            x,
            ids,
            route_weights,
            prepared["gemm1_weights"],
            prepared["gemm2_weights"],
            torch.bfloat16,
            quant_scales,
            activation_type=ActivationType.Situ,
            tp_size=8,
            backend="cake",
            output=output,
            workspace_buffer=workspace,
            **activation_kwargs,
        )

    with pytest.raises(ValueError, match="prepar"):
        submit()
    assert torch.isnan(output).all()
    assert (
        cake_fused_moe_prepare_workspace(
            workspace,
            num_tokens,
            backend="cake",
            weight_layout="trtllm_shuffled_nvfp4_group16",
        )
        is workspace
    )
    # The prepared shape must select the expected route: the two-CTA-per-SM
    # FC2 route for 16 tokens on SM103, the single-token and fused-router
    # routes for the other small counts, and the fused quantization + router
    # stage exactly for 8 and 16 tokens on both architectures.
    arch = ARCH_BY_CAPABILITY[torch.cuda.get_device_capability(device)]
    prepared_shape = workspace._flashinfer_cake_situ_workspace["shapes"][num_tokens]
    selector = _small_row_selector(arch, num_tokens)
    assert prepared_shape["n8_w2a_m16"] is (selector == "n8_w2a_m16")
    if selector is not None:
        assert prepared_shape["program_key"] == ROUTES[(arch, selector)]
    assert prepared_shape["fused_quant_route"] is (num_tokens in (8, 16))
    assert _declares_fused_quant_route(prepared_shape["program_key"]) is (
        num_tokens in (8, 16)
    )
    if prepared_shape["n8_w2a_m16"]:
        sm_count = torch.cuda.get_device_properties(device).multi_processor_count
        assert prepared_shape["fc2_grid_n"] == min(
            prepared_shape["max_tiles"],
            _N8_W2A_M16_FC2_GRID_N_SM_FACTOR * sm_count // (HIDDEN // 128),
        )
        assert (
            prepared_shape["fc2_pool_ctas"]
            == (HIDDEN // 128) * prepared_shape["fc2_grid_n"]
        )

    expected = _trtllm_reference(x, ids, route_weights, prepared)
    # Ensure an all-zero output could not satisfy the FP4 absolute tolerance.
    assert expected.abs().max() > 2.0
    assert submit() is output
    torch.testing.assert_close(output, expected, atol=1.0, rtol=0.1)

    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        submit()
        submit()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured_output = submit()
    assert captured_output is output
    output.fill_(float("nan"))
    graph.replay()
    torch.testing.assert_close(output, expected, atol=1.0, rtol=0.1)

    # Retain every input address while changing values and expert load balance:
    # every token now routes to the same 16 experts (the other 880 receive no
    # token), the maximally skewed pattern. Replay must consume these values,
    # not stale routing or hidden states.
    x.mul_(-0.75)
    ids.copy_(slots[None, :].expand_as(ids))
    route_weights.copy_(route_weights.flip(-1))
    changed_expected = _trtllm_reference(x, ids, route_weights, prepared)
    assert not torch.equal(expected, changed_expected)
    output.fill_(float("nan"))
    graph.replay()
    torch.testing.assert_close(output, changed_expected, atol=1.0, rtol=0.1)

    default_parameters_output = output.clone()
    output.fill_(float("nan"))
    assert submit(explicit_parameters=True) is output
    torch.testing.assert_close(output, default_parameters_output, atol=0.0, rtol=0.0)
    assert output.data_ptr() == output_ptr
    assert workspace.data_ptr() == workspace_ptr
