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

import pytest
import torch
import tvm_ffi

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
from flashinfer.fused_moe.cake_kimi_k3_situ import (
    _cake_situ_flat_args,
    _cake_situ_stage_bindings,
    _cake_situ_workspace_views,
    _geometry,
    _prepared,
    _route_flags,
    _workspace_layout,
)
from flashinfer.jit.cake_kimi_k3_situ import (
    PROGRAM_FLAGS,
    PROGRAMS,
    ROUTES,
    cake_situ_selector,
    cake_situ_sequence,
    get_cake_situ_module,
)
from flashinfer.tllm_enums import ActivationType, RoutingMethodType
from flashinfer.utils import device_support_pdl


HIDDEN, INTERMEDIATE, EXPERTS, TOP_K = 3584, 384, 896, 16
NVFP4_QUANT = QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4)
LARGE_TOKENS = (8192, 16384)


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


def test_cake_situ_route_program_keys():
    # GPU-free: every route binds a program whose static facts name that selector
    # and architecture, the static selection resolves every exported token count
    # to its program, and the 8192- and 16384-token rows of both architectures
    # (and no other token count) route to the nine-stage sequence that runs the
    # scale-factor writer right before FC1.
    c7_stages = [
        "route_reset",
        "quant",
        "route_histogram",
        "route_prefix",
        "route_scatter",
        "sfb_shuffle",
        "fc1",
        "fc2",
        "finalize",
    ]
    assert set(PROGRAM_FLAGS) == set(ROUTES.values())
    for (arch, selector), key in ROUTES.items():
        flags = PROGRAM_FLAGS[key]
        assert PROGRAMS[key]["arch"] == arch == flags["arch"]
        assert selector in flags["selectors"]
        bound = {s for (a, s), k in ROUTES.items() if k == key}
        assert flags["selectors"] == sorted(bound, key=str)
        assert flags["num_tokens"] == sorted(set(flags["num_tokens"]))
        # Retained bare tile-size bindings carry no exported token count.
        assert flags["formal"] == bool(flags["num_tokens"]) == isinstance(selector, str)
        for num_tokens in flags["num_tokens"]:
            static = _route_flags(arch, num_tokens)
            assert static["program_key"] == key
            assert static["stages"] == flags["stages"]
            assert static["selector"] in flags["selectors"]
            for name in (
                "tile_n",
                "single_token",
                "feature_finalize",
                "fc2_device_workfeed",
            ):
                assert static[name] == flags[name]
            assert static["large_c7"] == (static["selector"] == "large_c7")
        assert (
            (flags["stages"] == c7_stages)
            == (selector == "large_c7")
            == (flags["selectors"] == ["large_c7"])
        )
    for arch in ("sm_100a", "sm_103a"):
        for num_tokens in LARGE_TOKENS:
            key = _route_flags(arch, num_tokens)["program_key"]
            assert key == ROUTES[(arch, "large_c7")]
            assert PROGRAM_FLAGS[key]["stages"] == c7_stages
            assert PROGRAM_FLAGS[key]["num_tokens"] == list(LARGE_TOKENS)
        retained = ROUTES[(arch, 128)]
        assert PROGRAM_FLAGS[retained]["formal"] is False
        assert PROGRAM_FLAGS[retained]["num_tokens"] == []
        for num_tokens in (1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096):
            assert _route_flags(arch, num_tokens)["large_c7"] is False
        # The generated selector and sequence functions must agree with each
        # other and with the static selection the runtime binds, for every
        # supported token count.
        for num_tokens in range(1, 16385):
            static = _route_flags(arch, num_tokens)
            args = (
                arch,
                static["tile_n"],
                static["single_token"],
                static["feature_finalize"],
                static["m64_claim8"],
                static["mid_work5fd"],
                static["n32_claim8"],
                static["m256_c12"],
                static["large_c7"],
            )
            assert cake_situ_selector(*args) == static["selector"]
            assert cake_situ_sequence(*args) == static["program_key"]


def test_cake_situ_workspace_size_covers_every_pre_shuffled_token_count():
    # A workspace sized for max_num_tokens >= t must admit the t-token shape that
    # carries the pre-shuffled scale-factor images (its layout is larger than the
    # ordinary layout of the token counts just above it). Size query only.
    for t in LARGE_TOKENS:
        at_t = _workspace_size(max_num_tokens=t)
        for max_tokens in (t, t + 1, t + 128, t + 255, 16384):
            if max_tokens <= 16384:
                assert _workspace_size(max_num_tokens=max_tokens) >= at_t
    for max_tokens in range(65, 16385, 64):
        assert _workspace_size(max_num_tokens=max_tokens) >= _workspace_size(
            max_num_tokens=max_tokens - 1
        )


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
    assert "sfb_shuffled" not in _workspace_layout(4096)[0]
    assert "sfb_shuffled" not in _workspace_layout(2048)[0]


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
    [64, 256, 512, 2048, 8192, 16384],
    ids=["n8", "n16", "n32", "n128", "m8192", "m16384"],
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

    # Retain every input address while changing values and expert load balance.
    # Replay must consume these values, not stale routing or hidden states.
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


def _arch(device):
    return {(10, 0): "sm_100a", (10, 3): "sm_103a"}[
        torch.cuda.get_device_capability(device)
    ]


def _uniform_routing(num_tokens, device):
    slots = torch.arange(TOP_K, dtype=torch.int32, device=device)
    tokens = torch.arange(num_tokens, dtype=torch.int32, device=device)
    return ((tokens[:, None] * TOP_K + slots[None, :]) % EXPERTS).contiguous()


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


def _submit_large_route(num_tokens, routing, device, prepared, workspace):
    """Prepare and submit one complete call; return the inputs and the
    option dictionary the stage bindings consume for direct-launch checks."""
    generator = torch.Generator(device=device).manual_seed(7000 + num_tokens)
    x = torch.randn(
        num_tokens,
        HIDDEN,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    ids = routing(num_tokens, device)
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
    quant_scales = [
        torch.ones(1, device=device, dtype=torch.float32),
        prepared["gemm1_weights_scale"],
        prepared["output1_scale_gate_scalar"],
        prepared["output1_scale_scalar"],
        prepared["gemm2_weights_scale"],
        prepared["output2_scale_scalar"],
    ]
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
            prepared["gemm1_weights"],
            prepared["gemm2_weights"],
            torch.bfloat16,
            quant_scales,
            activation_type=ActivationType.Situ,
            tp_size=8,
            backend="cake",
            output=output,
            workspace_buffer=workspace,
        )
        is output
    )
    torch.cuda.synchronize(device)
    options = dict(
        input=x,
        workspace_buffer=workspace,
        token_selected_experts=ids,
        token_final_scales=route_weights,
        fc1_expert_weights=prepared["gemm1_weights"],
        fc2_expert_weights=prepared["gemm2_weights"],
        output=output,
        quant_scales=quant_scales,
        situ_beta=None,
        situ_linear_beta=None,
    )
    return x, ids, route_weights, output, options


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


def _launch_program(program_key, args):
    module = get_cake_situ_module(program_key)
    entry = getattr(module, PROGRAMS[program_key]["ffi_entry"])
    with tvm_ffi.use_torch_stream():
        entry(*args)


def _module_args(program_key, stage):
    # A single-kernel module takes the stage's bindings with unqualified names.
    return [
        stage["grid"][("grid_x", "grid_y", "grid_z").index(name)]
        if kind == "grid"
        else stage[name]
        for kind, name in PROGRAMS[program_key]["arg_plan"]
    ]


@pytest.mark.parametrize("num_tokens", LARGE_TOKENS, ids=["m8192", "m16384"])
@pytest.mark.parametrize(
    "routing", [_uniform_routing, _skewed_routing], ids=["uniform", "skewed"]
)
def test_cake_situ_pre_shuffled_route_matches_previous_fc1_program(
    num_tokens,
    routing,
    cake_situ_device,
    cake_situ_weights,
    cake_situ_workspace,
):
    # The 8192- and 16384-token routes run the scale-factor writer and the FC1
    # program that loads the pre-shuffled images. Their FC1 outputs (per routed
    # pair), the FC2 outputs and the complete call must be bitwise identical to
    # the previous FC1 program, which gathers and shuffles the scale factors
    # itself. That program is retained as the tile-N128 sequence, so it runs
    # here on the same workspace, inputs and stage bindings.
    device, weights, workspace = (
        cake_situ_device,
        cake_situ_weights,
        cake_situ_workspace,
    )
    x, ids, route_weights, output, options = _submit_large_route(
        num_tokens, routing, device, weights, workspace
    )
    expected = _trtllm_reference(x, ids, route_weights, weights)
    assert expected.abs().max() > 2.0
    torch.testing.assert_close(output, expected, atol=1.0, rtol=0.1)

    prepared = _prepared(workspace, num_tokens)
    arch = _arch(device)
    assert prepared["large_c7"]
    assert prepared["program_key"] == ROUTES[(arch, "large_c7")]
    previous_key = ROUTES[(arch, 128)]
    assert previous_key != prepared["program_key"]
    previous_names = {name for _, name in PROGRAMS[previous_key]["arg_plan"]}
    assert "fc1.SFB" in previous_names and "fc1.SFBS" not in previous_names
    assert not any(name.startswith("sfb_shuffle.") for name in previous_names)

    views = _cake_situ_workspace_views(workspace, num_tokens)
    total_tiles = int(views["total_tiles"].item())
    assert 0 < total_tiles < prepared["max_tiles"]
    pre_shuffled = _per_pair_scratch(views, prepared["max_tiles"])
    pre_shuffled_output = output.clone()
    for name in ("intermediate_packed", "intermediate_scales"):
        views[name].zero_()
    views["expert_output"].fill_(float("nan"))
    output.fill_(float("nan"))

    stages = _cake_situ_stage_bindings(options, prepared)
    _launch_program(previous_key, _cake_situ_flat_args(stages, previous_key))
    torch.cuda.synchronize(device)
    assert int(views["total_tiles"].item()) == total_tiles
    for name, tensor in _per_pair_scratch(views, prepared["max_tiles"]).items():
        assert torch.equal(tensor, pre_shuffled[name]), name
    assert torch.equal(output, pre_shuffled_output)


_WRITER_ARGUMENTS = [
    "SFB",
    "route_map",
    "tile_mn_limit",
    "total_tiles",
    "SFBS",
    "K",
    "K_tiles",
    "grid_n",
    "grid_x",
    "grid_y",
    "grid_z",
]


def _writer_program_key(arch):
    keys = [
        key
        for key, record in PROGRAMS.items()
        if record["arch"] == arch
        and [name for _, name in record["arg_plan"]] == _WRITER_ARGUMENTS
    ]
    assert len(keys) == 1, keys
    return keys[0]


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


@pytest.mark.parametrize("num_tokens", LARGE_TOKENS, ids=["m8192", "m16384"])
@pytest.mark.parametrize(
    "routing", [_uniform_routing, _skewed_routing], ids=["uniform", "skewed"]
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
    _, _, _, _, options = _submit_large_route(
        num_tokens, routing, device, weights, workspace
    )
    prepared = _prepared(workspace, num_tokens)
    assert prepared["large_c7"]
    views = _cake_situ_workspace_views(workspace, num_tokens)
    stages = _cake_situ_stage_bindings(options, prepared)
    stage = stages["sfb_shuffle"]
    max_tiles = prepared["max_tiles"]
    assert stage["grid"] == (max_tiles, 1, 1) and stage["grid_n"] == max_tiles
    total_tiles = int(views["total_tiles"].item())
    assert 0 < total_tiles < max_tiles
    rows_in_tile = views["tile_mn_limit"][:total_tiles].long() - 128 * torch.arange(
        total_tiles, device=device
    )
    assert rows_in_tile.min() >= 1 and rows_in_tile.max() <= 128
    assert (rows_in_tile < 128).any()
    if routing is _skewed_routing:
        assert (rows_in_tile <= 2).sum() >= 400

    sentinel = 0xFF
    images = views["sfb_shuffled"].view(max_tiles, -1)
    images.fill_(sentinel)
    key = _writer_program_key(_arch(device))
    _launch_program(key, _module_args(key, stage))
    torch.cuda.synchronize(device)
    expected = _reference_sfb_shuffle(
        views["x_scales"], views["route_map"], views["tile_mn_limit"], total_tiles
    )
    assert torch.equal(images[:total_tiles], expected)
    assert (images[total_tiles:] == sentinel).all()
    # FC1 consumes the same bytes through its 4-D TMA view: one 512-byte
    # (2 x 256) block per 64 elements of K for every tile.
    fc1_images = stages["fc1"]["SFBS"]
    assert fc1_images.data_ptr() == images.data_ptr()
    assert tuple(fc1_images.shape) == (max_tiles, HIDDEN // 64, 2, 256)
    assert fc1_images.numel() == images.numel()
