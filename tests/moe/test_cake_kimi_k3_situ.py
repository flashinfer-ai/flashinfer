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

# Architectures whose single-token route is the fused quantization + routing
# FC1 (three kernels) -- both today; an architecture outside this set would run
# the separate quantization + router kernel (four kernels, the "m1" selector).
FUSED_M1_ARCHES = ("sm_100a", "sm_103a")
# Architectures whose 8192- and 16384-token route stores the FC2 result as an
# 8-bit per-tile partial with per-tile FP32 scales (eight kernels) -- both
# today; an architecture outside this set would run the pre-shuffled
# scale-factor route (nine kernels, the "large_c7" selector).
LARGE_Q8I_ARCHES = ("sm_100a", "sm_103a")
# Architectures and token counts whose large route runs the routing histogram
# and scatter kernels with per-CTA shared-memory privatization and block
# reservation (1024 routed pairs per CTA; the "large_q8i_p1024" selector) --
# the 16384-token row of sm_100a today; every other large row keeps the
# production routing kernels (256 pairs per CTA).
LARGE_P1024_ARCHES = ("sm_100a",)
LARGE_P1024_TOKENS = (16384,)
# Architectures whose 4096-token route stores the FC2 result as the same 8-bit
# per-tile partial behind the work5fd FC1; the 2048-token route keeps the BF16
# expert output everywhere.
MID_Q8I_ARCHES = ("sm_103a",)
MID_Q8I_TOKENS = (4096,)
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
    "m1_s2a": ["fc1", "fc2", "finalize"],
    "n8_feature": ["quant_route", "fc1", "fc2", "finalize"],
    "n8_w2a_m16": ["quant_route", "fc1", "fc2", "finalize"],
    "m64_claim8": ["quant", "fused_router", "fc1", "fc2", "finalize"],
    "n32_claim8": ["quant", "fused_router", "fc1", "fc2", "finalize"],
    "mid_work5fd": GENERIC_STAGES,
    "mid_q8i": GENERIC_STAGES,
    "large_c7": GENERIC_STAGES[:5] + ["sfb_shuffle"] + GENERIC_STAGES[5:],
    "large_q8i": GENERIC_STAGES,
    "large_q8i_p1024": GENERIC_STAGES,
}


def _tile_bucket(num_tokens):
    if num_tokens <= 128:
        return 8
    if num_tokens <= 256:
        return 16
    if num_tokens <= 1024:
        return 32
    return 128


def _expected_selector(arch, num_tokens):
    if num_tokens == 1 and arch in FUSED_M1_ARCHES:
        return "m1_s2a"
    if num_tokens in LARGE_P1024_TOKENS and arch in LARGE_P1024_ARCHES:
        return "large_q8i_p1024"
    if num_tokens in LARGE_TOKENS and arch in LARGE_Q8I_ARCHES:
        return "large_q8i"
    if num_tokens in MID_Q8I_TOKENS and arch in MID_Q8I_ARCHES:
        return "mid_q8i"
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
    for t in LARGE_TOKENS:
        at_t = _workspace_size(max_num_tokens=t)
        for max_tokens in (t, t + 1, t + 128, t + 255, 16384):
            if max_tokens <= 16384:
                assert _workspace_size(max_num_tokens=max_tokens) >= at_t


def test_cake_situ_workspace_layout_has_no_pre_shuffled_scale_factors():
    # No delivered architecture's selector of 8192 / 16384 tokens is the
    # pre-shuffled scale-factor route any more, so the architecture-independent
    # layout of those token counts carries no ``sfb_shuffled`` images (the
    # INT8 partial and its per-tile scales are carved out of the BF16
    # expert-output scratch); every other token count keeps its layout.
    for num_tokens, expected_tiles in ((8192, 1913), (16384, 2937)):
        tile_n, _, max_tiles = _geometry(num_tokens)
        assert (tile_n, max_tiles) == (128, expected_tiles)
        layout, nbytes = _workspace_layout(num_tokens)
        assert "sfb_shuffled" not in layout
        assert nbytes >= max(start + size for start, size, _, _ in layout.values())
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
        assert selector == _expected_selector(arch, num_tokens), num_tokens
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
    # (or just above a large token count) must still allow preparing the
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
    selector = _expected_selector(arch, num_tokens)
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
        (4096, "skewed"),
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
        "mid_m4096",
        "large_m8192",
        "large_m16384",
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
    if prepared["selector"] == "large_q8i_p1024":
        # The per-CTA privatized routing kernels handle 1024 routed pairs per
        # CTA: a quarter of the production routing grid of the same row.
        bindings = _cake_situ_stage_bindings(
            _options(x, ids, route_weights, output, weights, workspace), prepared
        )
        for stage in ("route_histogram", "route_scatter"):
            assert bindings[stage]["grid"] == (
                (prepared["total_pairs"] + 1023) // 1024,
                1,
                1,
            )
            assert bindings[stage]["total_pairs"] == prepared["total_pairs"]
    if num_tokens == 1:
        # The single-token route quantizes and routes inside the FC1 prologue on
        # the fused architectures (three launches; the FC1 binding carries the
        # activations, the activation scale and the expert ids) and through the
        # separate quantization + router kernel elsewhere (four launches).
        options = _options(x, ids, route_weights, output, weights, workspace)
        bindings = _cake_situ_stage_bindings(options, prepared)
        fused = arch in FUSED_M1_ARCHES
        assert prepared["selector"] == ("m1_s2a" if fused else "m1")
        assert list(bindings) == STAGES_BY_SELECTOR[prepared["selector"]]
        assert ("x" in bindings["fc1"]) == fused
        if fused:
            assert bindings["fc1"]["x"].data_ptr() == x.data_ptr()
            assert bindings["fc1"]["topk_ids"].data_ptr() == ids.data_ptr()
            assert (
                bindings["fc1"]["qx"].data_ptr()
                == options["quant_scales"][0].data_ptr()
            )
            assert bindings["fc1"]["grid"] == (
                INTERMEDIATE // 64,
                prepared["max_tiles"],
                1,
            )
        else:
            assert bindings["quant_route"]["x"] is x
    if prepared["selector"] in ("large_q8i", "large_q8i_p1024", "mid_q8i"):
        # The FC2 stage stores an 8-bit per-tile partial and its per-tile FP32
        # scales inside the BF16 expert-output scratch, and the finalize reads
        # the same two views: no pre-shuffled scale-factor images are bound.
        options = _options(x, ids, route_weights, output, weights, workspace)
        bindings = _cake_situ_stage_bindings(options, prepared)
        views = _cake_situ_workspace_views(workspace, num_tokens)
        scratch = views["expert_output"]
        rows = prepared["max_tiles"] * 128
        partial, scales = bindings["fc2"]["C"], bindings["fc2"]["partial_scale"]
        assert partial.dtype == torch.uint8 and tuple(partial.shape) == (rows, HIDDEN)
        assert partial.data_ptr() == scratch.data_ptr()
        assert scales.dtype == torch.float32
        assert scales.numel() == prepared["max_tiles"] * (HIDDEN // 128)
        assert scales.data_ptr() == scratch.data_ptr() + rows * HIDDEN
        assert scales.data_ptr() + 4 * scales.numel() <= (
            scratch.data_ptr() + scratch.numel() * scratch.element_size()
        )
        assert bindings["fc2"]["C_tma"].data_ptr() == partial.data_ptr()
        assert bindings["finalize"]["expert_output"].data_ptr() == partial.data_ptr()
        assert bindings["finalize"]["partial_scale"].data_ptr() == scales.data_ptr()
        assert "SFBS" not in bindings["fc1"] and "sfb_shuffle" not in bindings

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
    if prepared["selector"] in ("large_q8i", "large_q8i_p1024", "mid_q8i"):
        # The 8-bit FC2 partial of this route carries one FP32 scale per
        # 128x128 tile, and the atomic-counter routing assigns rows to tiles
        # in a call-dependent order, so the tile membership (hence each
        # tile's scale and rounding) differs between two calls: the outputs
        # are not bitwise reproducible, but the explicit parameters must
        # still agree within the reference tolerance.
        torch.testing.assert_close(
            output, default_parameters_output, atol=1.0, rtol=0.1
        )
    else:
        torch.testing.assert_close(
            output, default_parameters_output, atol=0.0, rtol=0.0
        )
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


def _routing_in_pair_order(views, ids, max_tiles):
    # The routing scatter assigns each (token, expert-slot) pair a row inside
    # its expert's tiles; the row a pair receives differs between routing
    # kernels (and from call to call), so the routing tables and the per-row
    # FC1 scratch are compared in pair order through token_to_permuted (the
    # map the finalize uses), never by row index.
    permuted = views["token_to_permuted"].long()
    total_tiles = int(views["total_tiles"].item())
    assert permuted.numel() == ids.numel()
    assert int(permuted.min()) >= 0 and int(permuted.max()) < total_tiles * 128
    # Every pair's row lies in a tile of the pair's expert, below the tile's
    # valid-row limit, and no two pairs share a row.
    tile = permuted // 128
    assert torch.equal(
        views["tile_expert"][:total_tiles].long()[tile], ids.view(-1).long()
    )
    assert bool((permuted < views["tile_mn_limit"][:total_tiles].long()[tile]).all())
    assert permuted.unique().numel() == permuted.numel()
    return {
        "expert_counts": views["expert_counts"].clone(),
        "expert_tile_offsets": views["expert_tile_offsets"].clone(),
        "expert_scatter_offsets": views["expert_scatter_offsets"].clone(),
        "total_tiles": total_tiles,
        "tile_expert": views["tile_expert"][:total_tiles].clone(),
        "tile_mn_limit": views["tile_mn_limit"][:total_tiles].clone(),
        "rows": permuted.sort().values,
        "route_map": views["route_map"][permuted].clone(),
        "intermediate_packed": views["intermediate_packed"][permuted].clone(),
        "intermediate_scales": _fc1_scales_by_row(views, max_tiles)[permuted].clone(),
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
    [(16384, "uniform"), (16384, "skewed")],
    ids=["m16384_uniform", "m16384_skewed"],
)
def test_cake_situ_per_cta_privatized_routing_matches_the_production_routing(
    num_tokens,
    routing,
    cake_situ_device,
    cake_situ_weights,
    cake_situ_workspace,
):
    # The sm_100a 16384-token route runs the routing histogram and scatter
    # kernels that privatize the expert histogram in per-CTA shared memory and
    # reserve a block of rows per CTA (1024 routed pairs per CTA). Their
    # routing tables, the FC1 scratch of every routed pair and the complete
    # call are compared, in pair order, with the production routing kernels
    # (256 pairs per CTA) run on the same workspace, inputs and stage
    # bindings: expert counts, tile tables and the per-pair expert, row set,
    # route map and FC1 output must be bitwise identical, and each call's
    # output must match the reference within the published tolerance. The two
    # outputs are not compared with each other: they are not bitwise
    # identical, because the accumulation order of the INT8 per-tile FC2
    # partial depends on tile membership, which differs between the two
    # routings; their largest absolute difference is recorded instead.
    device, weights, workspace = (
        cake_situ_device,
        cake_situ_weights,
        cake_situ_workspace,
    )
    arch = _arch(device)
    if _expected_selector(arch, num_tokens) != "large_q8i_p1024":
        pytest.skip(
            f"{arch} routes {num_tokens} tokens to {_expected_selector(arch, num_tokens)}; "
            "the per-CTA privatized routing programs are not delivered for it"
        )
    x, ids, route_weights, output, options = _submit_prepared(
        num_tokens, routing, device, weights, workspace
    )
    prepared = _prepared(workspace, num_tokens)
    assert prepared["selector"] == "large_q8i_p1024"
    production = {
        stage: cake_situ_program(arch, stage)
        for stage in ("route_histogram", "route_scatter")
    }
    for stage, program in production.items():
        assert prepared["modules"][stage] == cake_situ_program(
            arch, f"{stage}:n128_p1024"
        )
        assert prepared["modules"][stage] != program
        assert (
            MODULES[prepared["modules"][stage]]["arg_plan"]
            == MODULES[program]["arg_plan"]
        )
    bindings = _cake_situ_stage_bindings(options, prepared)
    total_pairs = prepared["total_pairs"]
    assert total_pairs == num_tokens * TOP_K
    for stage in production:
        assert bindings[stage]["grid"] == ((total_pairs + 1023) // 1024, 1, 1)
        assert bindings[stage]["total_pairs"] == total_pairs
    expected = _trtllm_reference(x, ids, route_weights, weights)
    torch.testing.assert_close(output, expected, atol=1.0, rtol=0.1)

    views = _cake_situ_workspace_views(workspace, num_tokens)
    max_tiles = prepared["max_tiles"]
    privatized = _routing_in_pair_order(views, ids, max_tiles)
    privatized_output = output.clone()
    assert 0 < privatized["total_tiles"] <= max_tiles
    for name in ("intermediate_packed", "intermediate_scales"):
        views[name].zero_()
    views["token_to_permuted"].fill_(-1)
    views["expert_output"].fill_(float("nan"))
    output.fill_(float("nan"))

    # The same call with the production routing kernels: every stage of the
    # route is launched alone with the values the host bound for it, the
    # histogram and scatter replaced by the production programs on their
    # 256-pairs-per-CTA grid.
    production_grid = ((total_pairs + 255) // 256, 1, 1)
    for stage in prepared["stages"]:
        if stage in production:
            _launch(
                production[stage], arch, {**bindings[stage], "grid": production_grid}
            )
        else:
            _launch(prepared["modules"][stage], arch, bindings[stage])
    torch.cuda.synchronize(device)
    reference = _routing_in_pair_order(views, ids, max_tiles)
    for name, value in privatized.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(reference[name], value), name
        else:
            assert reference[name] == value, name
    torch.testing.assert_close(output, expected, atol=1.0, rtol=0.1)
    # Recorded, not asserted: the two routings' outputs differ bitwise.
    recorded = (output.float() - privatized_output.float()).abs().max().item()
    print(f"largest absolute difference between the two routings' outputs: {recorded}")
