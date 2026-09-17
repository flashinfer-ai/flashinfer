"""
Copyright (c) 2023 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""FA2 NVFP4 repack coverage for the small KV tiles used on SM89."""

import pytest
import torch

import flashinfer
from tests.test_helpers.test_helpers import ref_single_prefill


def _make_kv(shape, dtype):
    packed = torch.randint(0, 256, shape, device="cuda", dtype=torch.uint8)
    choices = torch.tensor([0, 1, 32, 40, 48, 56], device="cuda", dtype=torch.uint8)
    sf = choices[
        torch.randint(0, len(choices), (*shape[:-1], shape[-1] // 8), device="cuda")
    ]
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device="cuda",
    )
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).long()
    values = lut[codes].reshape(*shape[:-1], shape[-1] * 2)
    values *= sf.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=-1)
    return packed, sf, values.to(dtype)


def _rope(x, positions):
    dim = x.shape[-1]
    angles = positions[:, None].float() * (
        10000 ** (-torch.arange(0, dim, 2, device=x.device).float() / dim)
    )
    cos, sin = angles.cos()[:, None], angles.sin()[:, None]
    a, b = x.float().chunk(2, dim=-1)
    return torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1).to(x.dtype)


@pytest.mark.parametrize("head_dim_vo", [128, 256])
@pytest.mark.parametrize("q_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE", "ROPE_LLAMA"])
@pytest.mark.parametrize("mode", ["single", "ragged", "paged"])
@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_nvfp4_prefill_small_tile(
    head_dim_vo, q_dtype, causal, pos_encoding_mode, mode, layout
):
    if tuple(map(int, torch.version.cuda.split(".")[:2])) < (12, 8):
        pytest.skip("NVFP4 requires CUDA 12.8 or newer")
    torch.manual_seed(4775)
    # 13 * 4 packed Q rows selects CTA_TILE_Q=64, including asymmetric heads.
    # Both Q and KV have tails; the final page contains only one valid token.
    batch = 1 if mode == "single" else 2
    qo_len, kv_len, page_size, q_heads, kv_heads, head_dim_qk = 13, 65, 16, 4, 1, 256
    pages_per_seq = (kv_len + page_size - 1) // page_size
    padded_len = pages_per_seq * page_size
    q = (
        torch.randn(batch, qo_len, q_heads, head_dim_qk, device="cuda", dtype=q_dtype)
        * 0.2
    )
    k, k_sf, k_ref = _make_kv((batch, padded_len, kv_heads, head_dim_qk // 2), q_dtype)
    v, v_sf, v_ref = _make_kv((batch, padded_len, kv_heads, head_dim_vo // 2), q_dtype)
    expected = []
    for i in range(batch):
        qi, ki, vi = q[i], k_ref[i, :kv_len], v_ref[i, :kv_len]
        if pos_encoding_mode == "ROPE_LLAMA":
            qi = _rope(qi, torch.arange(kv_len - qo_len, kv_len, device="cuda"))
            ki = _rope(ki, torch.arange(kv_len, device="cuda"))
        expected.append(ref_single_prefill(qi, ki, vi, causal)[0])
    expected = torch.cat(expected).to(q_dtype)
    q = q.flatten(0, 1)

    def pack_layout(t):
        if mode == "paged":
            t = t.reshape(batch * pages_per_seq, page_size, kv_heads, t.shape[-1])
            return (t.transpose(1, 2) if layout == "HND" else t).contiguous()
        t = t[:, :kv_len].reshape(batch * kv_len, kv_heads, t.shape[-1])
        return (t.transpose(0, 1) if layout == "HND" else t).contiguous()

    k, v, k_sf, v_sf = map(pack_layout, (k, v, k_sf, v_sf))
    kwargs = dict(causal=causal, pos_encoding_mode=pos_encoding_mode)
    if mode == "single":

        def run():
            return flashinfer.single_prefill_with_kv_cache(
                q,
                k,
                v,
                kv_layout=layout,
                kv_cache_sf=(k_sf, v_sf),
                backend="fa2",
                **kwargs,
            )

    else:
        workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        q_indptr = torch.arange(batch + 1, dtype=torch.int32) * qo_len
        plan_kwargs = dict(
            q_data_type=q_dtype,
            kv_data_type=torch.uint8,
            head_dim_vo=head_dim_vo,
            **kwargs,
        )
        if mode == "paged":
            wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                workspace, layout, backend="fa2"
            )
            wrapper.plan(
                q_indptr,
                torch.arange(batch + 1, dtype=torch.int32) * pages_per_seq,
                torch.arange(batch * pages_per_seq, dtype=torch.int32),
                torch.full((batch,), 1, dtype=torch.int32),
                q_heads,
                kv_heads,
                head_dim_qk,
                page_size,
                **plan_kwargs,
            )

            def run():
                return wrapper.run(q, (k, v), kv_cache_sf=(k_sf, v_sf))

        else:
            wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
                workspace, layout, backend="fa2"
            )
            wrapper.plan(
                q_indptr,
                torch.arange(batch + 1, dtype=torch.int32) * kv_len,
                q_heads,
                kv_heads,
                head_dim_qk,
                **plan_kwargs,
            )

            def run():
                return wrapper.run(q, k, v, kv_cache_sf=(k_sf, v_sf))

    tol = 2e-3 if q_dtype == torch.float16 else 2e-2
    actual = run()
    torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)
    if layout == "NHD" and pos_encoding_mode == "NONE":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run()
        graph.replay()
        torch.testing.assert_close(captured, expected, rtol=tol, atol=tol)
