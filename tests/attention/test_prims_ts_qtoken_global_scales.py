# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""QToken global scale kinds remain independent of prepared metadata storage."""

import itertools
import math

import pytest
import torch

pytest.importorskip("cutlass", minversion="4.7.0a0")

from flashinfer.attention.prims_ts.q_token_kv_block_sparse_metadata import (
    QTokenKvBlockSparsePagedTSWrapper,
    get_q_token_kv_block_sparse_workspace_size,
    q_token_kv_block_sparse_attention_with_paged_kv_cache,
)
from flashinfer.utils import get_compute_capability
from tests.attention.sage_test_utils import ieee_float32_references  # noqa: F401
from tests.test_helpers.pdl import late_store

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or get_compute_capability(torch.device("cuda")) not in ((10, 0), (10, 3), (10, 7)),
    reason="PrimTS QToken attention requires an SM100-class GPU",
)


def _scale(value, tensor):
    return torch.tensor([value], device="cuda") if tensor else value


@pytest.mark.parametrize("kinds", tuple(itertools.product((False, True), repeat=2)))
@pytest.mark.parametrize("group_size", (1, 4))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float8_e4m3fn))
def test_qtoken_scale_overrides_and_graph_replay(kinds, group_size, dtype):
    generator = torch.Generator(device="cuda").manual_seed(42)
    q = (
        torch.randn((1, 1, group_size, 8, 128), device="cuda", generator=generator)
        * 0.5
    ).to(dtype)
    k = (torch.randn((2, 1, 16, 128), device="cuda", generator=generator) * 0.5).to(
        dtype
    )
    v = torch.randn(k.shape, device="cuda", generator=generator).to(dtype)
    table = torch.tensor([[0, 1]], dtype=torch.int32, device="cuda")
    blocks = (
        torch.arange(4, dtype=torch.int32, device="cuda")
        .expand(group_size, 4)
        .contiguous()
    )
    requests = torch.zeros(group_size, dtype=torch.int32, device="cuda")
    positions = torch.arange(15, 15 + group_size, dtype=torch.int64, device="cuda")
    out = torch.empty_like(q, dtype=torch.bfloat16)
    workspace = torch.empty(
        get_q_token_kv_block_sparse_workspace_size(
            q,
            k,
            table,
            block_topk=4,
            group_size=group_size,
            max_seq_len_kv=32,
            out_dtype=out.dtype,
        ),
        dtype=torch.uint8,
        device="cuda",
    )
    values = (0.125, 0.75)
    scales = tuple(
        _scale(value, kind) for value, kind in zip(values, kinds, strict=False)
    )
    wrapper = QTokenKvBlockSparsePagedTSWrapper()
    wrapper.plan(
        1,
        8,
        1,
        128,
        4,
        16,
        4,
        32,
        device=q.device,
        workspace_buffer=workspace,
        group_size=group_size,
        q_dtype=dtype,
        out_dtype=out.dtype,
        sm_scale=scales[0],
        v_scale=scales[1],
    )

    host_out = torch.empty_like(out)
    host_wrapper = QTokenKvBlockSparsePagedTSWrapper()
    host_wrapper.plan(
        1,
        8,
        1,
        128,
        4,
        16,
        4,
        32,
        device=q.device,
        workspace_buffer=torch.empty_like(workspace),
        group_size=group_size,
        q_dtype=dtype,
        out_dtype=out.dtype,
    )

    def run(**kwargs):
        return wrapper.run(
            q, (k, v), table, blocks, requests, positions, out=out, **kwargs
        )

    def check(multiplier):
        host_wrapper.run(
            q,
            (k, v),
            table,
            blocks,
            requests,
            positions,
            out=host_out,
            sm_scale=multiplier[0],
            v_scale=multiplier[1],
        )
        torch.testing.assert_close(out, host_out, atol=0, rtol=0)
        for row in range(group_size):
            length = 16 + row
            selected = torch.cat(
                (
                    torch.arange(min(length // 4, 4) * 4, device="cuda"),
                    torch.arange(length // 4 * 4, length, device="cuda"),
                )
            )
            keys = k[:, 0].reshape(-1, 128)[selected].float()
            vals = v[:, 0].reshape(-1, 128)[selected].float()
            scores = q[0, 0, row].float() @ keys.T
            if dtype == torch.float8_e4m3fn:
                # The sparse union fits one KV tile. Its FP8 P operand is
                # rounded before PV MMA; the denominator remains FP32.
                assert group_size * (4 + 1) * 4 <= 128
                scale = torch.tensor(
                    multiplier[0] * math.log2(math.e),
                    device=q.device,
                    dtype=torch.float32,
                )
                maximum = scores.amax(-1, keepdim=True)
                bias = torch.addcmul(
                    torch.full_like(maximum, math.log2(448.0)), maximum, -scale
                )
                probabilities = torch.addcmul(bias, scores, scale).exp2()
                weights = probabilities.to(dtype).float() / probabilities.sum(
                    -1, keepdim=True
                )
                expected = weights @ vals * multiplier[1]
                torch.testing.assert_close(
                    out[0, 0, row].float(), expected, atol=0.01, rtol=0.03
                )
            else:
                expected = (scores * multiplier[0]).softmax(-1) @ vals * multiplier[1]
                torch.testing.assert_close(
                    out[0, 0, row].float(), expected, atol=0.02, rtol=0.05
                )

    run()
    check(values)
    prepared = wrapper._prepared_plan
    overrides = tuple(
        _scale(value * 2, kind) for value, kind in zip(values, kinds, strict=False)
    )
    run(sm_scale=overrides[0], v_scale=overrides[1])
    check(tuple(value * 2 for value in values))
    assert wrapper._prepared_plan is prepared
    run()
    check(values)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    changed = tuple(
        value * 1.5 if kind else value
        for value, kind in zip(values, kinds, strict=False)
    )
    for scale, value, kind in zip(scales, changed, kinds, strict=False):
        if kind:
            scale.fill_(value)
    graph.replay()
    check(changed)

    q_token_kv_block_sparse_attention_with_paged_kv_cache(
        q,
        (k, v),
        table,
        blocks,
        requests,
        positions,
        workspace_buffer=workspace,
        max_seq_len_kv=32,
        group_size=group_size,
        sm_scale=scales[0],
        v_scale=scales[1],
        out=out,
    )
    check(changed)
    for index, name in enumerate(("sm_scale", "v_scale")):
        with pytest.raises(TypeError, match="same scalar/tensor type"):
            run(**{name: _scale(values[index], not kinds[index])})


@pytest.mark.parametrize("capture", (False, True), ids=("eager", "graph"))
@pytest.mark.parametrize(
    "kinds,updated_scale",
    (
        ((True, False), 0),
        ((False, True), 1),
        ((True, True), 0),
        ((True, True), 1),
    ),
    ids=("qk-tensor", "v-tensor", "both-update-qk", "both-update-v"),
)
def test_qtoken_persistent_direct_q1_pdl_scales(
    kinds: tuple[bool, bool], updated_scale: int, capture: bool
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(42)
    q = (torch.randn((1, 1, 1, 8, 128), device="cuda", generator=generator) * 0.5).to(
        torch.bfloat16
    )
    k = (torch.randn((2, 1, 16, 128), device="cuda", generator=generator) * 0.5).to(
        torch.bfloat16
    )
    v = torch.randn(k.shape, device="cuda", generator=generator).to(torch.bfloat16)
    table = torch.tensor([[0, 1]], dtype=torch.int32, device="cuda")
    blocks = torch.arange(4, dtype=torch.int32, device="cuda").reshape(1, 4)
    requests = torch.zeros(1, dtype=torch.int32, device="cuda")
    positions = torch.tensor([15], dtype=torch.int64, device="cuda")
    out = torch.empty_like(q)
    workspace = torch.empty(
        get_q_token_kv_block_sparse_workspace_size(
            q,
            k,
            table,
            block_topk=4,
            group_size=1,
            max_seq_len_kv=32,
            out_dtype=out.dtype,
            split_kv=False,
        ),
        dtype=torch.uint8,
        device="cuda",
    )
    values = (0.125, 0.75)
    scales = tuple(
        _scale(value, kind) for value, kind in zip(values, kinds, strict=True)
    )
    wrapper = QTokenKvBlockSparsePagedTSWrapper()
    wrapper.plan(
        1,
        8,
        1,
        128,
        4,
        16,
        4,
        32,
        device=q.device,
        workspace_buffer=workspace,
        group_size=1,
        split_kv=False,
        q_dtype=q.dtype,
        out_dtype=out.dtype,
        sm_scale=scales[0],
        v_scale=scales[1],
    )

    def run() -> None:
        wrapper.run(q, (k, v), table, blocks, requests, positions, out=out)

    # G1 bypasses the metadata producer; disabling split-KV selects persistent
    # scheduling, so the attention body must acquire its scale dependencies.
    run()
    attention_plan = wrapper._prepared_plan._attention_plan
    assert attention_plan._direct_q1_inputs
    assert attention_plan._compiled_reducer is None

    scores = q[0, 0, 0].float() @ k[0, 0].float().T
    vals = v[0, 0].float()
    expected = (scores * values[0]).softmax(-1) @ vals * values[1]
    stale_values = list(values)
    stale_values[updated_scale] *= 4
    stale_expected = (scores * stale_values[0]).softmax(-1) @ vals * stale_values[1]
    assert not torch.allclose(stale_expected, expected, atol=0.02, rtol=0.05)
    torch.testing.assert_close(out[0, 0, 0].float(), expected, atol=0.02, rtol=0.05)

    scale = scales[updated_scale]
    # Warm the producer's module before graph capture or the timed dependency.
    late_store(scale, values[updated_scale])
    torch.cuda.synchronize()

    def produce_and_run() -> None:
        # No intervening kernel may serialize the producer-to-attention handoff.
        late_store(scale, values[updated_scale])
        run()

    graph = None
    if capture:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            produce_and_run()

    for _ in range(3):
        scale.fill_(stale_values[updated_scale])
        if graph is None:
            produce_and_run()
        else:
            graph.replay()
        torch.testing.assert_close(out[0, 0, 0].float(), expected, atol=0.02, rtol=0.05)
