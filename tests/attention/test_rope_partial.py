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
from tests.attention.test_helpers.rope_reference import precompute_freqs_cis

import flashinfer


@pytest.fixture
def rope_launch(request):
    flashinfer.rope._set_rope_launch_override(*request.param)
    yield
    flashinfer.rope._set_rope_launch_override()


@pytest.mark.parametrize(
    "rope_launch",
    [(None, None), (False, 1), (True, 2)],
    ids=["auto", "one-head-split", "two-head-split"],
    indirect=True,
)
@pytest.mark.parametrize(
    "api_name",
    [
        "apply_rope",
        "apply_rope_inplace",
        "apply_rope_pos_ids",
        "apply_rope_pos_ids_inplace",
        "apply_llama31_rope",
        "apply_llama31_rope_inplace",
        "apply_llama31_rope_pos_ids",
        "apply_llama31_rope_pos_ids_inplace",
    ],
)
@pytest.mark.parametrize(
    "head_dim,rotary_dim", [(64, 20), (128, 24), (512, 48), (128, 64)]
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("interleave", [False, True], ids=["neox", "interleave"])
def test_partial_rope_vector_boundaries(
    api_name, head_dim, rotary_dim, dtype, interleave, rope_launch
):
    torch.manual_seed(42)
    nnz, num_q_heads, num_k_heads = 21, 4, 2
    packed = torch.randn(
        nnz, num_q_heads + 2 * num_k_heads, head_dim, device="cuda", dtype=dtype
    )
    original = packed.clone()
    q = packed[:, :num_q_heads]
    k = packed[:, num_q_heads : num_q_heads + num_k_heads]
    indptr = torch.tensor([0, 0, 3, 8, nnz], device="cuda", dtype=torch.int32)
    offsets = torch.tensor([23, 31, 55, 107], device="cuda", dtype=torch.int32)
    positions = torch.cat(
        [
            torch.arange(start, start + length, device="cuda")
            for start, length in [(31, 3), (55, 5), (107, 13)]
        ]
    ).int()
    is_llama31 = "llama31" in api_name
    freqs = precompute_freqs_cis(
        rotary_dim, 128, theta=5e5 if is_llama31 else 1e4, use_scaled=is_llama31
    )[positions]
    cos, sin = freqs.real[:, None, :], freqs.imag[:, None, :]

    def reference(x):
        rotated = x[..., :rotary_dim].float()
        if interleave:
            first, second = rotated[..., ::2], rotated[..., 1::2]
            result = torch.stack(
                (first * cos - second * sin, second * cos + first * sin), dim=-1
            ).flatten(-2)
        else:
            first, second = rotated.chunk(2, dim=-1)
            result = torch.cat(
                (first * cos - second * sin, second * cos + first * sin), dim=-1
            )
        return torch.cat((result.to(dtype), x[..., rotary_dim:]), dim=-1)

    api = getattr(flashinfer, api_name)
    metadata = (positions,) if "pos_ids" in api_name else (indptr, offsets)

    def run():
        result = api(q, k, *metadata, rotary_dim=rotary_dim, interleave=interleave)
        return (q, k) if "inplace" in api_name else result

    def check(outputs, source):
        for output, expected_input in zip(
            outputs,
            (
                source[:, :num_q_heads],
                source[:, num_q_heads : num_q_heads + num_k_heads],
            ),
            strict=True,
        ):
            tolerance = 1e-2 if dtype == torch.bfloat16 else 1e-3
            torch.testing.assert_close(
                output, reference(expected_input), rtol=tolerance, atol=tolerance
            )
            torch.testing.assert_close(
                output[..., rotary_dim:],
                expected_input[..., rotary_dim:],
                rtol=0,
                atol=0,
            )
        torch.testing.assert_close(
            packed[:, num_q_heads + num_k_heads :],
            source[:, num_q_heads + num_k_heads :],
            rtol=0,
            atol=0,
        )

    check(run(), original)
    packed.copy_(original)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_outputs = run()
    for scale in (0.5, -1.0):
        source = original * scale
        packed.copy_(source)
        graph.replay()
        check(graph_outputs, source)
