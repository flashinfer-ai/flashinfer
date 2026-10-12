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

"""Cake SiTU backend: route selection, caller-owned workspace and graph behaviour.

Every test observes the backend through its public entry points and the
prepared-shape facts the host records (stages, bound programs, FC2 pool); none
asserts the text or structure of a generated source.
"""

import pytest
import torch
import tvm_ffi

from flashinfer.fused_moe import (
    QuantConfig,
    QuantFormat,
    SiTU,
    TrtllmFp4Config,
    cake_fused_moe_prepare_workspace,
    cutlass_fused_moe,
    cutlass_fused_moe_workspace_size,
    trtllm_fp4_block_scale_routed_moe,
)
from flashinfer.fused_moe.cake_kimi_k3_situ import (
    _ROUTE_MC_CLUSTER,
    _SELECTORS,
    _cake_situ_stage_bindings,
    _cake_situ_workspace_views,
    _fc2_grid_n,
    _geometry,
    _prepared,
    _route,
    _selector,
    _workspace_layout,
)
from flashinfer.jit.cake_kimi_k3_situ import (
    CARRIED_PROGRAMS,
    KERNELS,
    MODULES,
    cake_situ_program,
    get_cake_situ_module,
)
from flashinfer.tllm_enums import ActivationType, RoutingMethodType
from flashinfer.utils import device_support_pdl


HIDDEN, INTERMEDIATE, EXPERTS, TOP_K = 3584, 384, 896, 16
NVFP4_QUANT = QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4)
ARCHES = ("sm_100a", "sm_103a")
ARCH_BY_CAPABILITY = {(10, 0): "sm_100a", (10, 3): "sm_103a"}
LARGE_TOKENS = (8192, 16384)
# SM count of the B200 and B300 parts the FC2 pools were sized on.
REFERENCE_SM_COUNT = 148

# Token counts with a route of their own; every other count runs the generic
# pipeline of its tile-N bucket (8 / 16 / 32 / 128).
ROUTED = {
    1: "m1",
    8: "n8_feature",
    16: "n8_w2a_m16",
    32: "m64_claim8",
    64: "m64_claim8",
    128: "m64_claim8",
    256: "m64_claim8",
    512: "n32_claim8",
    1024: "n32_claim8",
    2048: "mid_work5fd",
    4096: "mid_work5fd",
    8192: "large_c7",
    16384: "large_c7",
}
GENERIC_STAGES = [
    "route_reset",
    "quant",
    "route_histogram",
    "route_prefix",
    "route_scatter",
    "fc1",
    "fc2",
    "finalize",
]
STAGES_BY_SELECTOR = {
    "m1": ["quant_route", "fc1", "fc2", "finalize"],
    "n8_feature": ["quant_route", "fc1", "fc2", "finalize"],
    "n8_w2a_m16": ["quant_route", "fc1", "fc2", "finalize"],
    "m64_claim8": ["quant", "fused_router", "fc1", "fc2", "finalize"],
    "n32_claim8": ["quant", "fused_router", "fc1", "fc2", "finalize"],
    "mid_work5fd": GENERIC_STAGES,
    "large_c7": GENERIC_STAGES[:5] + ["sfb_shuffle"] + GENERIC_STAGES[5:],
}


def _tile_bucket(num_tokens):
    if num_tokens <= 128:
        return 8
    if num_tokens <= 256:
        return 16
    if num_tokens <= 1024:
        return 32
    return 128


def _expected_selector(num_tokens):
    return ROUTED.get(num_tokens, _tile_bucket(num_tokens))


def _expected_fc2_grid_n(selector, max_tiles, sm_count):
    # The device-workfeed pools: one FC2 CTA per SM for 8 tokens, two per SM
    # for 16 tokens, twelve rows for 32..256 tokens and seven rows for 512 and
    # 1024 tokens; every other route covers the whole tile range.
    # The 16-token pool is sized from twice the SM count (two resident CTAs
    # per SM), not from twice the rounded single-CTA pool.
    pools = {
        "n8_feature": sm_count // (HIDDEN // 128),
        "n8_w2a_m16": 2 * sm_count // (HIDDEN // 128),
        "m64_claim8": 12,
        "n32_claim8": 7,
    }
    return min(max_tiles, pools[selector]) if selector in pools else max_tiles


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
    # tile-N16 and need more scratch than the tile-N8 counts after them, and
    # the pre-shuffled scale-factor images of 8192 tokens exceed the ordinary
    # layout of the counts just above), but the public size query must still
    # honor the documented contract that a maximum-size buffer holds every
    # smaller prepared shape.
    sizes = [_workspace_size(max_num_tokens=n) for n in range(1, 16385)]
    running_max = 0
    for size in sizes:
        running_max = max(running_max, size)
        assert size == running_max
    assert _workspace_size(max_num_tokens=40) >= _workspace_size(max_num_tokens=32)
    assert _workspace_size(max_num_tokens=100) >= _workspace_size(max_num_tokens=64)
    for t in LARGE_TOKENS:
        at_t = _workspace_size(max_num_tokens=t)
        for max_tokens in (t, t + 1, t + 128, t + 255, 16384):
            if max_tokens <= 16384:
                assert _workspace_size(max_num_tokens=max_tokens) >= at_t


def test_cake_situ_workspace_layout_adds_pre_shuffled_scale_factors():
    # The 8192- and 16384-token layouts append the pre-shuffled FC1 scale-factor
    # images (one 4096-byte image per N-tile and 512-element K-step) as their
    # last field; every other token count keeps its previous layout.
    for num_tokens, expected_tiles in ((8192, 1913), (16384, 2937)):
        tile_n, _, max_tiles = _geometry(num_tokens)
        assert (tile_n, max_tiles) == (128, expected_tiles)
        layout, nbytes = _workspace_layout(num_tokens)
        offset, size, dtype, shape = layout["sfb_shuffled"]
        assert size == max_tiles * (HIDDEN // 512) * 4096
        assert dtype == torch.uint8 and shape == (size,)
        assert offset == max(start for start, _, _, _ in layout.values())
        assert nbytes >= offset + size
    for num_tokens in (1, 8, 16, 64, 256, 512, 2048, 4096, 8191, 8193):
        assert "sfb_shuffled" not in _workspace_layout(num_tokens)[0]


@pytest.mark.parametrize("arch", ARCHES)
def test_cake_situ_route_table(arch):
    # GPU-free: every supported token count selects the documented route on
    # both architectures, with the documented stage list, a registered program
    # for every stage, the forced tile-N16 geometry of the 32..256-token rows
    # and the documented FC2 device-workfeed pool.
    for num_tokens in range(1, 16385):
        selector = _selector(arch, num_tokens)
        assert selector == _expected_selector(num_tokens), num_tokens
        stages, kernels = _route(arch, num_tokens)
        assert stages == STAGES_BY_SELECTOR.get(selector, GENERIC_STAGES), num_tokens
        assert list(kernels) == stages
        for stage, key in kernels.items():
            assert key.split(":", 1)[0] == stage
            program = cake_situ_program(arch, key)
            assert program == KERNELS[arch][key]
            assert arch in MODULES[program]["arches"]
        tile_n, total_pairs, max_tiles = _geometry(num_tokens, arch)
        assert total_pairs == num_tokens * TOP_K
        assert tile_n == (
            16 if num_tokens in (32, 64, 128, 256) else _tile_bucket(num_tokens)
        )
        occupied = min(EXPERTS, total_pairs)
        assert max_tiles == occupied + (total_pairs - occupied) // tile_n
        for sm_count in (REFERENCE_SM_COUNT, 132, 160):
            assert _fc2_grid_n(selector, max_tiles, sm_count) == _expected_fc2_grid_n(
                selector, max_tiles, sm_count
            ), (num_tokens, sm_count)
    # Every registered kernel key is reachable by some token count.
    reachable = {
        key
        for num_tokens in range(1, 16385)
        for key in _route(arch, num_tokens)[1].values()
    }
    assert reachable == set(KERNELS[arch])


def test_cake_situ_sixteen_token_pool_is_twice_the_eight_token_pool():
    # The 16-token route runs an FC2 program built for two resident CTAs per SM;
    # its device-workfeed pool therefore holds twice the rows of the 8-token
    # route on the same part (280 against 140 CTAs on 148 SMs).
    eight = _fc2_grid_n("n8_feature", _geometry(8)[2], REFERENCE_SM_COUNT)
    sixteen = _fc2_grid_n("n8_w2a_m16", _geometry(16)[2], REFERENCE_SM_COUNT)
    assert (eight, sixteen) == (5, 10)
    assert _SELECTORS["n8_w2a_m16"][1]["fc2"] != _SELECTORS["n8_feature"][1]["fc2"]
    # The 32..256-token rows share a twelve-row pool, the 512/1024-token rows a
    # seven-row pool, on every part with at least that many tiles.
    for num_tokens in (32, 64, 128, 256):
        assert (
            _fc2_grid_n("m64_claim8", _geometry(num_tokens)[2], REFERENCE_SM_COUNT)
            == 12
        )
    for num_tokens in (512, 1024):
        assert (
            _fc2_grid_n("n32_claim8", _geometry(num_tokens)[2], REFERENCE_SM_COUNT) == 7
        )


def test_cake_situ_program_records_are_consistent():
    # Every program record names its two sources, the architectures it compiles
    # for and a complete launch plan; every program is bound by a kernel key on
    # each architecture it lists, and the carried programs are a subset of the
    # registry that records where each one came from.
    bound = {arch: set(KERNELS[arch].values()) for arch in ARCHES}
    for name, record in MODULES.items():
        assert record["sources"][0].endswith(f"{name}_kernel.cu"), name
        assert record["sources"][1].endswith(f"{name}_binding.cu"), name
        assert record["arches"] == sorted(record["arches"]), name
        assert set(record["arches"]) <= set(ARCHES), name
        assert len(record["closure_sha256"]) == 64, name
        kinds = {kind for kind, _ in record["arg_plan"]}
        assert kinds <= {"tma_buffer", "buffer", "parameter", "grid"}, name
        assert [n for kind, n in record["arg_plan"] if kind == "grid"] == [
            "grid_x",
            "grid_y",
            "grid_z",
        ], name
        for arch in record["arches"]:
            assert name in bound[arch], (name, arch)
        assert (name in CARRIED_PROGRAMS) == ("carried" in record), name
    for arch in ARCHES:
        for name in bound[arch]:
            assert arch in MODULES[name]["arches"], (name, arch)
    assert set(CARRIED_PROGRAMS) <= set(MODULES)


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
    # A single maximum-size allocation is reused across all parametrized token
    # sizes; the 16384-token shape needs the largest layout.
    return torch.empty(
        _workspace_size(max_num_tokens=16384),
        dtype=torch.uint8,
        device=cake_situ_device,
    )


@pytest.mark.parametrize(
    "max_num_tokens, num_tokens",
    [(40, 32), (100, 64), (8200, 8192)],
    ids=["n40_holds_32", "n100_holds_64", "n8200_holds_8192"],
)
def test_cake_situ_prepare_smaller_shape_in_maximum_size_buffer(
    max_num_tokens,
    num_tokens,
    cake_situ_device,
):
    # Allocating for a token count strictly between two forced tile-N16 counts
    # (or just above a pre-shuffled count) must still allow preparing the
    # smaller count in the same buffer.
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


def _arch(device):
    return ARCH_BY_CAPABILITY[torch.cuda.get_device_capability(device)]


def _uniform_routing(num_tokens, device):
    slots = torch.arange(TOP_K, dtype=torch.int32, device=device)
    tokens = torch.arange(num_tokens, dtype=torch.int32, device=device)
    return ((tokens[:, None] * TOP_K + slots[None, :]) % EXPERTS).contiguous()


def _grouped_routing(num_tokens, device):
    # Heavy expert skew for the clustered 512- and 1024-token routes: every
    # token picks one of four disjoint 16-expert groups, so 64 experts each
    # receive num_tokens / 4 rows (several 32-row tiles per expert span) and
    # the other 832 experts receive none.
    slots = torch.arange(TOP_K, dtype=torch.int32, device=device)
    tokens = torch.arange(num_tokens, dtype=torch.int32, device=device)
    return ((tokens[:, None] % 4) * TOP_K + slots[None, :]).contiguous()


def _skewed_routing(num_tokens, device):
    # Sixteen experts receive almost every token (127 full tiles plus one
    # partial tile each). The last 96 tokens spread their 1536 pairs over the
    # other 880 experts, so most of those experts own a tile with one or two
    # valid rows, and the routed tile count stays below the maximum.
    slots = torch.arange(TOP_K, dtype=torch.int32, device=device)
    tokens = torch.arange(num_tokens, dtype=torch.int32, device=device)
    ids = slots[None, :].expand(num_tokens, TOP_K).clone()
    tail = tokens >= num_tokens - 96
    ids[tail] = TOP_K + (tokens[tail, None] * TOP_K + slots[None, :]) % (
        EXPERTS - TOP_K
    )
    return ids.contiguous()


ROUTINGS = {
    "uniform": _uniform_routing,
    "grouped": _grouped_routing,
    "skewed": _skewed_routing,
}


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


def _quant_scales(prepared, device):
    return [
        torch.ones(1, device=device, dtype=torch.float32),
        prepared["gemm1_weights_scale"],
        prepared["output1_scale_gate_scalar"],
        prepared["output1_scale_scalar"],
        prepared["gemm2_weights_scale"],
        prepared["output2_scale_scalar"],
    ]


def _inputs(num_tokens, routing, device, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(
        num_tokens,
        HIDDEN,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    ids = ROUTINGS[routing](num_tokens, device)
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
    return x, ids, route_weights


def _options(x, ids, route_weights, output, weights, workspace):
    """The option dictionary the host's stage bindings consume."""
    return dict(
        input=x,
        workspace_buffer=workspace,
        token_selected_experts=ids,
        token_final_scales=route_weights,
        fc1_expert_weights=weights["gemm1_weights"],
        fc2_expert_weights=weights["gemm2_weights"],
        output=output,
        quant_scales=_quant_scales(weights, x.device),
        situ_beta=None,
        situ_linear_beta=None,
    )


def _check_prepared_shape(prepared, arch, num_tokens, device):
    # The prepared shape binds the documented route: its selector, stage list,
    # the registered program of every stage and the FC2 device-workfeed pool.
    selector = _expected_selector(num_tokens)
    stages, kernels = _route(arch, num_tokens)
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    assert prepared["selector"] == selector
    assert prepared["stages"] == stages
    assert prepared["modules"] == {
        stage: cake_situ_program(arch, key) for stage, key in kernels.items()
    }
    assert len(prepared["launches"]) == len(stages)
    submit, plan, slots = prepared["submit"]
    assert len(plan) == sum(3 + len(flat) for _, flat in prepared["launches"])
    index = 0
    for _entry, flat in prepared["launches"]:
        prepare_entry, submit_entry, argc = plan[index : index + 3]
        # The stage shim's two-phase entry points, resolved once per program.
        assert isinstance(prepare_entry, int) and prepare_entry > 0
        assert isinstance(submit_entry, int) and submit_entry > 0
        assert argc == len(flat)
        index += 3 + len(flat)
    assert all(plan[index] is None for index, _ in slots)
    assert callable(submit)
    assert prepared["fc2_grid_n"] == _expected_fc2_grid_n(
        selector, prepared["max_tiles"], sm_count
    )
    assert prepared["fc2_pool_ctas"] == (HIDDEN // 128) * prepared["fc2_grid_n"]
    if selector == "n32_claim8":
        assert (prepared["fc2_grid_n"], prepared["fc2_pool_ctas"]) == (7, 196)
    if selector == "m64_claim8":
        assert (prepared["fc2_grid_n"], prepared["fc2_pool_ctas"]) == (12, 336)
    if selector == "n8_w2a_m16":
        assert prepared["fc2_grid_n"] == min(
            prepared["max_tiles"], 2 * sm_count // (HIDDEN // 128)
        )


@pytest.mark.parametrize(
    "num_tokens, routing",
    [
        (1, "uniform"),
        (8, "uniform"),
        (16, "uniform"),
        (64, "uniform"),
        (256, "uniform"),
        (512, "uniform"),
        (1024, "grouped"),
        (2048, "uniform"),
        (8192, "uniform"),
        (16384, "uniform"),
        (2, "uniform"),
        (200, "uniform"),
        (300, "uniform"),
        (3000, "uniform"),
    ],
    ids=[
        "m1",
        "n8_feature",
        "n8_w2a_m16",
        "m64_claim8_m64",
        "m64_claim8_m256",
        "n32_claim8_m512",
        "n32_claim8_m1024_grouped",
        "mid_work5fd",
        "large_c7_m8192",
        "large_c7_m16384",
        "tile8_generic",
        "tile16_generic",
        "tile32_generic",
        "tile128_generic",
    ],
)
def test_cake_situ_every_route_eager_and_external_graph(
    num_tokens,
    routing,
    cake_situ_device,
    cake_situ_weights,
    cake_situ_workspace,
):
    # Every route the host can select, on this device's architecture: the
    # eleven routed selectors (one token count each, two for the claim8
    # routes) and the four generic tile buckets. Each case checks the
    # prepared-shape facts, the eager result against the trtllm-gen
    # reference, external CUDA-graph capture and replay with changed inputs,
    # and that the default SiTU parameters equal the explicit ones.
    device, weights, workspace = (
        cake_situ_device,
        cake_situ_weights,
        cake_situ_workspace,
    )
    arch = _arch(device)
    x, ids, route_weights = _inputs(num_tokens, routing, device, 9000 + num_tokens)
    output = torch.full_like(x, float("nan"))
    output_ptr, workspace_ptr = output.data_ptr(), workspace.data_ptr()
    quant_scales = _quant_scales(weights, device)

    def submit(*, explicit_parameters=False):
        activation_kwargs = (
            {
                "situ_beta": weights["gemm1_alpha"],
                "situ_linear_beta": weights["gemm1_beta"],
            }
            if explicit_parameters
            else {}
        )
        return cutlass_fused_moe(
            x,
            ids,
            route_weights,
            weights["gemm1_weights"],
            weights["gemm2_weights"],
            torch.bfloat16,
            quant_scales,
            activation_type=ActivationType.Situ,
            tp_size=8,
            backend="cake",
            output=output,
            workspace_buffer=workspace,
            **activation_kwargs,
        )

    # The module-scoped workspace keeps every prepared shape; the unprepared
    # rejection is asserted on the first case of each token count only.
    state = getattr(workspace, "_flashinfer_cake_situ_workspace", None)
    if state is None or num_tokens not in state["shapes"]:
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
    prepared = _prepared(workspace, num_tokens)
    _check_prepared_shape(prepared, arch, num_tokens, device)
    if prepared["selector"] == "n32_claim8":
        # The 512- and 1024-token routes launch their router as one
        # eight-CTA cluster over the seven-row FC2 pool.
        bindings = _cake_situ_stage_bindings(
            _options(x, ids, route_weights, output, weights, workspace), prepared
        )
        assert bindings["fused_router"]["grid"] == (_ROUTE_MC_CLUSTER, 1, 1)
        assert bindings["fused_router"]["fc2_pool_ctas"] == 196

    expected = _trtllm_reference(x, ids, route_weights, weights)
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
    slots = torch.arange(TOP_K, dtype=torch.int32, device=device)
    ids.copy_(slots[None, :].expand_as(ids))
    route_weights.copy_(route_weights.flip(-1))
    changed_expected = _trtllm_reference(x, ids, route_weights, weights)
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


def _submit_prepared(num_tokens, routing, device, weights, workspace):
    """Prepare and submit one complete call; return its tensors and options."""
    x, ids, route_weights = _inputs(num_tokens, routing, device, 7000 + num_tokens)
    output = torch.full_like(x, float("nan"))
    cake_fused_moe_prepare_workspace(
        workspace,
        num_tokens,
        backend="cake",
        weight_layout="trtllm_shuffled_nvfp4_group16",
    )
    assert (
        cutlass_fused_moe(
            x,
            ids,
            route_weights,
            weights["gemm1_weights"],
            weights["gemm2_weights"],
            torch.bfloat16,
            _quant_scales(weights, device),
            activation_type=ActivationType.Situ,
            tp_size=8,
            backend="cake",
            output=output,
            workspace_buffer=workspace,
        )
        is output
    )
    torch.cuda.synchronize(device)
    return (
        x,
        ids,
        route_weights,
        output,
        _options(x, ids, route_weights, output, weights, workspace),
    )


def _fc1_scales_by_row(views, max_tiles):
    # FC1 stores its output scale factors in the tile layout FC2 loads: per
    # tile and per 64 elements of the intermediate size one 512-byte block
    # whose word (row % 32, row // 32) holds the row's four scale bytes.
    blocks = views["intermediate_scales"].view(max_tiles, INTERMEDIATE // 64, 32, 4, 4)
    return blocks.permute(0, 3, 2, 1, 4).reshape(max_tiles * 128, INTERMEDIATE // 16)


def _per_pair_scratch(views, max_tiles):
    # The routing scatter assigns each (token, expert-slot) pair a row inside
    # its expert's tiles; that position is not deterministic from call to
    # call, so the per-row scratch is compared in pair order through
    # token_to_permuted (the map finalization uses), never by row index.
    permuted = views["token_to_permuted"].long()
    return {
        "intermediate_packed": views["intermediate_packed"][permuted].clone(),
        "intermediate_scales": _fc1_scales_by_row(views, max_tiles)[permuted].clone(),
        "expert_output": views["expert_output"][permuted].clone(),
    }


def _launch(program, arch, stage_values):
    # Launch one program alone with the values the host bound for its stage.
    module = get_cake_situ_module(program, arch)
    entry = getattr(module, MODULES[program]["ffi_entry"])
    args = [
        stage_values["grid"][("grid_x", "grid_y", "grid_z").index(name)]
        if kind == "grid"
        else stage_values[name]
        for kind, name in MODULES[program]["arg_plan"]
    ]
    with tvm_ffi.use_torch_stream():
        entry(*args)


@pytest.mark.parametrize(
    "num_tokens, routing",
    [(8192, "skewed"), (16384, "uniform")],
    ids=["m8192_skewed", "m16384_uniform"],
)
def test_cake_situ_pre_shuffled_route_matches_plain_fc1(
    num_tokens,
    routing,
    cake_situ_device,
    cake_situ_weights,
    cake_situ_workspace,
):
    # The 8192- and 16384-token routes run the scale-factor writer and the FC1
    # program that loads the pre-shuffled images. Their FC1 outputs (per routed
    # pair), the FC2 outputs and the complete call must be bitwise identical to
    # the tile-N128 FC1 program that gathers and shuffles the scale factors
    # itself (the generic tile-128 route's FC1), run here on the same
    # workspace, inputs and stage bindings.
    device, weights, workspace = (
        cake_situ_device,
        cake_situ_weights,
        cake_situ_workspace,
    )
    _, _, _, output, options = _submit_prepared(
        num_tokens, routing, device, weights, workspace
    )
    prepared = _prepared(workspace, num_tokens)
    arch = _arch(device)
    assert prepared["selector"] == "large_c7"
    plain_fc1 = cake_situ_program(arch, "fc1:n128")
    assert plain_fc1 != prepared["modules"]["fc1"]
    plain_names = {name for _, name in MODULES[plain_fc1]["arg_plan"]}
    assert "SFB" in plain_names and "SFBS" not in plain_names

    views = _cake_situ_workspace_views(workspace, num_tokens)
    total_tiles = int(views["total_tiles"].item())
    assert 0 < total_tiles < prepared["max_tiles"]
    pre_shuffled = _per_pair_scratch(views, prepared["max_tiles"])
    pre_shuffled_output = output.clone()
    for name in ("intermediate_packed", "intermediate_scales"):
        views[name].zero_()
    views["expert_output"].fill_(float("nan"))
    output.fill_(float("nan"))

    bindings = _cake_situ_stage_bindings(options, prepared)
    # The route's FC1 binding carries the pre-shuffled images (SFBS); the plain
    # program gathers the routed scale factors itself from the quantized
    # activations' scales (SFB), which the host binds for the writer stage.
    _launch(plain_fc1, arch, {**bindings["fc1"], "SFB": views["x_scales"]})
    for stage in ("fc2", "finalize"):
        _launch(prepared["modules"][stage], arch, bindings[stage])
    torch.cuda.synchronize(device)
    assert int(views["total_tiles"].item()) == total_tiles
    for name, tensor in _per_pair_scratch(views, prepared["max_tiles"]).items():
        assert torch.equal(tensor, pre_shuffled[name]), name
    assert torch.equal(output, pre_shuffled_output)


def _reference_sfb_shuffle(x_scales, route_map, tile_mn_limit, total_tiles):
    # Image (tile, k) is the 128 x 32-byte block of routed scale factors for
    # K-step k, stored word by word (one word = the four 16-element groups of
    # 64 elements of K) at word index j * 128 + (row % 32) * 4 + row // 32.
    # Rows past the tile's valid row count are zero.
    device = x_scales.device
    k_tiles = x_scales.shape[1] // 32
    rows = torch.arange(total_tiles * 128, device=device)
    tile, row = rows // 128, rows % 128
    valid = rows < tile_mn_limit.long()[tile]
    tokens = route_map.long()[rows[valid]]
    image = torch.zeros(
        total_tiles, k_tiles, 8, 32, 4, 4, dtype=torch.uint8, device=device
    )
    image[tile[valid], :, :, row[valid] % 32, row[valid] // 32, :] = x_scales[
        tokens
    ].view(-1, k_tiles, 8, 4)
    return image.view(total_tiles, k_tiles * 4096)


@pytest.mark.parametrize(
    "num_tokens, routing",
    [(8192, "uniform"), (16384, "skewed")],
    ids=["m8192_uniform", "m16384_skewed"],
)
def test_cake_situ_sfb_shuffle_writer_matches_reference(
    num_tokens,
    routing,
    cake_situ_device,
    cake_situ_weights,
    cake_situ_workspace,
):
    # Launch the scale-factor writer alone, on the routing tables and
    # quantized scale factors the complete call left in the workspace, and
    # compare its images with a Python shuffle. A sentinel fill shows that
    # padding tiles past the routed tile count are never written and that
    # padding rows of a partial tile are written as zeros.
    device, weights, workspace = (
        cake_situ_device,
        cake_situ_weights,
        cake_situ_workspace,
    )
    _, _, _, _, options = _submit_prepared(
        num_tokens, routing, device, weights, workspace
    )
    prepared = _prepared(workspace, num_tokens)
    assert prepared["selector"] == "large_c7"
    views = _cake_situ_workspace_views(workspace, num_tokens)
    stage = _cake_situ_stage_bindings(options, prepared)["sfb_shuffle"]
    max_tiles = prepared["max_tiles"]
    assert stage["grid"] == (max_tiles, 1, 1) and stage["grid_n"] == max_tiles
    total_tiles = int(views["total_tiles"].item())
    assert 0 < total_tiles < max_tiles
    rows_in_tile = views["tile_mn_limit"][:total_tiles].long() - 128 * torch.arange(
        total_tiles, device=device
    )
    assert rows_in_tile.min() >= 1 and rows_in_tile.max() <= 128
    assert (rows_in_tile < 128).any()
    if routing == "skewed":
        assert (rows_in_tile <= 2).sum() >= 400

    sentinel = 0xFF
    images = views["sfb_shuffled"].view(max_tiles, -1)
    images.fill_(sentinel)
    _launch(prepared["modules"]["sfb_shuffle"], _arch(device), stage)
    torch.cuda.synchronize(device)
    expected = _reference_sfb_shuffle(
        views["x_scales"], views["route_map"], views["tile_mn_limit"], total_tiles
    )
    assert torch.equal(images[:total_tiles], expected)
    assert (images[total_tiles:] == sentinel).all()
    # FC1 consumes the same bytes through its 4-D TMA view: one 512-byte
    # (2 x 256) block per 64 elements of K for every tile.
    fc1_images = _cake_situ_stage_bindings(options, prepared)["fc1"]["SFBS"]
    assert fc1_images.data_ptr() == images.data_ptr()
    assert tuple(fc1_images.shape) == (max_tiles, HIDDEN // 64, 2, 256)
    assert fc1_images.numel() == images.numel()
