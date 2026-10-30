# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from flashinfer.prims_ts.utils import is_prims_ts_available
from flashinfer.utils import get_compute_capability


def _finalize_module():
    from flashinfer.fused_moe.core import get_trtllm_moe_sm100_module

    return get_trtllm_moe_sm100_module().moe_op


def _require_finalize_runtime():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if get_compute_capability(torch.device("cuda"))[0] != 10:
        pytest.skip("TRT-LLM SM100 finalize test requires an SM100-family GPU")
    if not is_prims_ts_available():
        pytest.skip("Prims-TS dependencies are unavailable")


def _make_finalize_case(
    num_tokens=4, top_k=2, hidden_size=256, hidden_size_padded=None
):
    hidden_size_padded = hidden_size_padded or hidden_size
    device = torch.device("cuda")
    total_padded = num_tokens * top_k
    gemm2_output = torch.randn(
        (total_padded, hidden_size_padded), device=device, dtype=torch.bfloat16
    )
    output = torch.empty((num_tokens, hidden_size), device=device, dtype=torch.bfloat16)
    expert_weights = torch.ones(
        (num_tokens, top_k), device=device, dtype=torch.bfloat16
    )
    expanded_to_permuted = torch.arange(total_padded, device=device, dtype=torch.int32)
    total_num_padded_tokens = torch.tensor(
        [total_padded], device=device, dtype=torch.int32
    )
    common = (
        expert_weights,
        expanded_to_permuted,
        total_num_padded_tokens,
        num_tokens,
        1,
        top_k,
        hidden_size,
        False,
        False,
    )
    return gemm2_output, output, common


@pytest.mark.parametrize(
    ("num_tokens", "top_k"),
    [
        pytest.param(4, 2, id="small-grid-topk2"),
        pytest.param(4, 5, id="small-grid-topk5"),
        pytest.param(4, 8, id="small-grid-topk8"),
        pytest.param(4, 64, id="small-grid-topk64"),
        pytest.param(1200, 2, id="large-grid-topk2"),
    ],
)
@pytest.mark.parametrize("use_expert_weights", [False, True])
def test_optional_second_input_finalize(num_tokens, top_k, use_expert_weights):
    _require_finalize_runtime()

    torch.manual_seed(0)
    device = torch.device("cuda")
    hidden_size = 256
    total_padded = num_tokens * top_k
    # Small integer BF16 values make the explicit BF16 add exact, so this
    # isolates indexing/reduction behavior rather than rounding order.
    input0 = torch.randint(
        -4, 5, (total_padded, hidden_size), device=device, dtype=torch.int32
    ).to(torch.bfloat16)
    input1 = torch.randint(
        -4, 5, (total_padded, hidden_size), device=device, dtype=torch.int32
    ).to(torch.bfloat16)
    expanded_to_permuted = torch.arange(total_padded, device=device, dtype=torch.int32)
    expanded_to_permuted[::3] = -1
    total_num_padded_tokens = torch.tensor(
        [total_padded], device=device, dtype=torch.int32
    )
    expert_weights = (
        torch.randint(
            -2,
            3,
            (num_tokens, top_k),
            device=device,
            dtype=torch.int32,
        ).to(torch.bfloat16)
        if use_expert_weights
        else None
    )
    expected = torch.empty(
        (num_tokens, hidden_size), device=device, dtype=torch.bfloat16
    )
    actual = torch.empty_like(expected)
    module = _finalize_module()
    common = (
        expert_weights,
        expanded_to_permuted,
        total_num_padded_tokens,
        num_tokens,
        1,
        top_k,
        hidden_size,
        False,
        False,
    )

    module.trtllm_moe_run_finalize(input0 + input1, expected, *common, None, True)
    module.trtllm_moe_run_finalize(input0, actual, *common, input1, True)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    without_second_input = torch.empty_like(expected)
    with_zero_second_input = torch.empty_like(expected)
    module.trtllm_moe_run_finalize(input0, without_second_input, *common, None, True)
    module.trtllm_moe_run_finalize(
        input0, with_zero_second_input, *common, torch.zeros_like(input0), True
    )

    assert torch.equal(without_second_input, with_zero_second_input)


@pytest.mark.parametrize("use_second_input", [False, True])
@pytest.mark.parametrize(
    ("top_k", "hidden_size", "hidden_size_padded"),
    [
        pytest.param(2, 256, 384, id="scalar"),
        pytest.param(8, 256, 384, id="small-batch"),
        pytest.param(64, 1000, 1024, id="small-batch-partial-block"),
    ],
)
def test_finalize_uses_gemm2_padded_hidden_stride(
    use_second_input, top_k, hidden_size, hidden_size_padded
):
    _require_finalize_runtime()

    torch.manual_seed(0)
    num_tokens = 4
    gemm2_output, output, common = _make_finalize_case(
        num_tokens, top_k, hidden_size, hidden_size_padded
    )
    gemm2_output = torch.randint(
        -4,
        5,
        gemm2_output.shape,
        device=gemm2_output.device,
        dtype=torch.int32,
    ).to(torch.bfloat16)
    second = (
        torch.randint(
            -4,
            5,
            gemm2_output.shape,
            device=gemm2_output.device,
            dtype=torch.int32,
        ).to(torch.bfloat16)
        if use_second_input
        else None
    )

    _finalize_module().trtllm_moe_run_finalize(
        gemm2_output, output, *common, second, True
    )

    combined = gemm2_output.float()
    if second is not None:
        combined = combined + second.float()
    expected = (
        combined[:, :hidden_size]
        .reshape(num_tokens, top_k, hidden_size)
        .sum(1)
        .to(torch.bfloat16)
    )
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        pytest.param(
            lambda inputs, output, common: (inputs.float(), output, common),
            "gemm2_output must be bfloat16",
            id="input-dtype",
        ),
        pytest.param(
            lambda inputs, output, common: (inputs.reshape(-1), output, common),
            "gemm2_output must be 2D",
            id="input-ndim",
        ),
        pytest.param(
            lambda inputs, output, common: (
                torch.empty(
                    (inputs.size(0), inputs.size(1) * 2),
                    device=inputs.device,
                    dtype=inputs.dtype,
                )[:, ::2],
                output,
                common,
            ),
            "gemm2_output must be contiguous",
            id="input-contiguity",
        ),
        pytest.param(
            lambda inputs, output, common: (inputs[:-1], output, common),
            "gemm2_output dim0 must have capacity",
            id="input-shape",
        ),
        pytest.param(
            lambda inputs, output, common: (inputs.cpu(), output, common),
            "gemm2_output is on the wrong device type",
            id="input-device",
        ),
        pytest.param(
            lambda inputs, output, common: (inputs, output.float(), common),
            "output must be bfloat16",
            id="output-dtype",
        ),
        pytest.param(
            lambda inputs, output, common: (
                inputs,
                torch.empty(
                    (output.size(1), output.size(0)),
                    device=output.device,
                    dtype=output.dtype,
                ).T,
                common,
            ),
            "output must be contiguous",
            id="output-contiguity",
        ),
        pytest.param(
            lambda inputs, output, common: (
                inputs,
                output[:, :-1].contiguous(),
                common,
            ),
            "output dim1 must match hidden_size",
            id="output-shape",
        ),
        pytest.param(
            lambda inputs, output, common: (inputs, output.cpu(), common),
            "output must be a CUDA tensor",
            id="output-device",
        ),
        pytest.param(
            lambda inputs, output, common: (
                inputs,
                output,
                (common[0], common[1].to(torch.int64), *common[2:]),
            ),
            "expanded_idx_to_permuted_idx must be int32",
            id="index-dtype",
        ),
        pytest.param(
            lambda inputs, output, common: (
                inputs,
                output,
                (common[0], common[1], common[2].reshape(1, 1), *common[3:]),
            ),
            "total_num_padded_tokens must be 1D",
            id="padded-count-ndim",
        ),
        pytest.param(
            lambda inputs, output, common: (
                inputs,
                output,
                (
                    common[0],
                    common[1].cpu(),
                    common[2],
                    *common[3:],
                ),
            ),
            "expanded_idx_to_permuted_idx is on the wrong device type",
            id="index-device",
        ),
        pytest.param(
            lambda inputs, output, common: (
                inputs,
                output,
                (
                    common[0][:, :1].contiguous(),
                    *common[1:],
                ),
            ),
            "expert_weights must have num_tokens \\* top_k entries",
            id="expert-weights-shape",
        ),
    ],
)
def test_finalize_validates_primary_and_metadata_tensors(mutate, error):
    _require_finalize_runtime()
    gemm2_output, output, common = _make_finalize_case()
    gemm2_output, output, common = mutate(gemm2_output, output, common)

    with pytest.raises(Exception, match=error):
        _finalize_module().trtllm_moe_run_finalize(
            gemm2_output, output, *common, None, True
        )


@pytest.mark.parametrize(
    ("make_second", "error"),
    [
        pytest.param(
            lambda source: source.float(),
            "gemm2_output2 must be bfloat16",
            id="dtype",
        ),
        pytest.param(
            lambda source: torch.cat((source, source[:, :1]), dim=1),
            "both inputs must have the same padded hidden size",
            id="shape",
        ),
        pytest.param(
            lambda source: source.cpu(),
            "gemm2_output2 is on the wrong device type",
            id="device",
        ),
    ],
)
def test_finalize_validates_second_input(make_second, error):
    _require_finalize_runtime()
    gemm2_output, output, common = _make_finalize_case()

    with pytest.raises(Exception, match=error):
        _finalize_module().trtllm_moe_run_finalize(
            gemm2_output, output, *common, make_second(gemm2_output), True
        )


def test_finalize_accepts_flat_expert_weights():
    _require_finalize_runtime()
    gemm2_output, output, common = _make_finalize_case()
    expert_weights = torch.randint(
        -2, 3, common[0].shape, device=common[0].device, dtype=torch.int32
    ).to(torch.bfloat16)
    flat_output = torch.empty_like(output)

    _finalize_module().trtllm_moe_run_finalize(
        gemm2_output, output, expert_weights, *common[1:], None, True
    )
    _finalize_module().trtllm_moe_run_finalize(
        gemm2_output, flat_output, expert_weights.reshape(-1), *common[1:], None, True
    )

    assert torch.equal(flat_output, output)


def test_finalize_zero_tokens_is_noop():
    _require_finalize_runtime()
    gemm2_output, output, common = _make_finalize_case(num_tokens=0)

    _finalize_module().trtllm_moe_run_finalize(
        gemm2_output, output, *common, torch.zeros_like(gemm2_output), True
    )
    torch.cuda.synchronize()


def test_finalize_unaligned_large_grid_uses_scalar_fallback():
    _require_finalize_runtime()
    num_tokens = 1200
    top_k = 2
    hidden_size = 130
    hidden_size_padded = 134
    gemm2_output, output, common = _make_finalize_case(
        num_tokens=num_tokens,
        top_k=top_k,
        hidden_size=hidden_size,
        hidden_size_padded=hidden_size_padded,
    )
    second = torch.randn_like(gemm2_output)

    _finalize_module().trtllm_moe_run_finalize(
        gemm2_output, output, *common, second, True
    )
    expected = (
        (gemm2_output.float() + second.float())[:, :hidden_size]
        .reshape(num_tokens, top_k, hidden_size)
        .sum(1)
        .to(torch.bfloat16)
    )
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


def _routes_with_and_without_local_experts(num_tokens, top_k, device):
    """Expanded-to-permuted rows: every third token has no local expert, the
    next one local expert and the one after it every other slot local."""
    expanded = torch.arange(num_tokens * top_k, device=device, dtype=torch.int32)
    expanded = expanded.reshape(num_tokens, top_k)
    expanded[0::3] = -1
    expanded[1::3, 1:] = -1
    expanded[2::3, 1::2] = -1
    return expanded


@pytest.mark.parametrize(
    ("num_tokens", "top_k", "use_second_input"),
    [
        pytest.param(4, 2, False, id="scalar"),
        pytest.param(4, 8, False, id="small-batch"),
        pytest.param(1200, 2, False, id="warp-per-token"),
        pytest.param(1200, 2, True, id="vec-load"),
    ],
)
def test_finalize_leaves_rows_without_local_expert_unwritten(
    num_tokens, top_k, use_second_input
):
    _require_finalize_runtime()

    torch.manual_seed(0)
    device = torch.device("cuda")
    hidden_size = 256
    shape = (num_tokens * top_k, hidden_size)
    gemm2_output = torch.randint(-4, 5, shape, device=device, dtype=torch.int32).to(
        torch.bfloat16
    )
    second = (
        torch.randint(-4, 5, shape, device=device, dtype=torch.int32).to(torch.bfloat16)
        if use_second_input
        else None
    )
    expanded = _routes_with_and_without_local_experts(num_tokens, top_k, device)
    no_local = (expanded == -1).all(dim=1)
    expert_weights = torch.randint(
        -2, 3, (num_tokens, top_k), device=device, dtype=torch.int32
    ).to(torch.bfloat16)
    common = (
        expert_weights,
        expanded.reshape(-1),
        torch.tensor([shape[0]], device=device, dtype=torch.int32),
        num_tokens,
        1,
        top_k,
        hidden_size,
        True,
        False,
        second,
    )
    zeroed = torch.full((num_tokens, hidden_size), float("nan"), device=device).to(
        torch.bfloat16
    )
    kept = zeroed.clone()

    _finalize_module().trtllm_moe_run_finalize(gemm2_output, zeroed, *common, True)
    _finalize_module().trtllm_moe_run_finalize(gemm2_output, kept, *common, False)

    assert torch.equal(zeroed[no_local], torch.zeros_like(zeroed[no_local]))
    assert torch.isnan(kept[no_local]).all()
    assert torch.equal(kept[~no_local], zeroed[~no_local])


def test_multi_expert_finalize_leaves_rows_without_local_expert_unwritten():
    """After an FC2 epilogue stored the single-expert tokens (here the
    prefilled rows), the finalize reduces the multi-expert tokens and writes
    the zero rows only when asked to."""
    _require_finalize_runtime()

    torch.manual_seed(0)
    device = torch.device("cuda")
    num_tokens, top_k, hidden_size = 1200, 4, 256
    shape = (num_tokens * top_k, hidden_size)
    gemm2_output = torch.randint(-4, 5, shape, device=device, dtype=torch.int32).to(
        torch.bfloat16
    )
    expanded = _routes_with_and_without_local_experts(num_tokens, top_k, device)
    num_local = (expanded != -1).sum(dim=1)
    expert_weights = torch.randint(
        -2, 3, (num_tokens, top_k), device=device, dtype=torch.int32
    ).to(torch.bfloat16)
    common = (
        expert_weights,
        expanded.reshape(-1),
        torch.tensor([shape[0]], device=device, dtype=torch.int32),
        num_tokens,
        1,
        top_k,
        hidden_size,
        True,
    )
    zeroed = torch.full((num_tokens, hidden_size), float("nan"), device=device).to(
        torch.bfloat16
    )
    kept = zeroed.clone()

    module = _finalize_module()
    module.trtllm_moe_run_finalize_multi_expert_tokens(
        gemm2_output, zeroed, *common, True
    )
    module.trtllm_moe_run_finalize_multi_expert_tokens(
        gemm2_output, kept, *common, False
    )

    rows = expanded.clamp(min=0).long()
    weights = expert_weights.float() * (expanded != -1)
    expected = (weights.unsqueeze(-1) * gemm2_output.float()[rows]).sum(1)
    multi = num_local > 1
    torch.testing.assert_close(
        zeroed[multi], expected[multi].to(torch.bfloat16), rtol=0, atol=0
    )
    assert torch.isnan(zeroed[num_local == 1]).all()
    assert torch.equal(zeroed[num_local == 0], torch.zeros_like(zeroed[num_local == 0]))
    assert torch.isnan(kept[num_local <= 1]).all()
    assert torch.equal(kept[multi], zeroed[multi])
