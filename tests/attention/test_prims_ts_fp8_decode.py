# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import math

import pytest
import torch

pytest.importorskip("cutlass", minversion="4.7.0")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="PrimTS requires Blackwell",
)


@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize(
    ("mode", "page_size", "head_ratio"),
    [
        ("disabled", 16, 1),
        ("gmem_reduction", 32, 7),
        ("gmem_reduction_with_separate_kernel", 64, 8),
        ("cluster_smem_reduction", 128, 16),
    ],
)
def test_prims_ts_fp8_decode_bf16_device_scales(
    monkeypatch: pytest.MonkeyPatch,
    head_dim: int,
    mode: str,
    page_size: int,
    head_ratio: int,
) -> None:
    """Qualify BF16 stores/reducers and live device scales across graph replay."""
    from dataclasses import replace

    import cutlass

    from flashinfer.attention.prims_ts import decode
    from flashinfer.attention.prims_ts.kernels.fmha_decode.fmha_decode_config import (
        make_decode_config,
    )

    torch.manual_seed(123)
    batch_size, kv_heads = 2, 2
    qo_heads = kv_heads * head_ratio
    max_kv_len = 2048
    pages_per_seq = max_kv_len // page_size
    num_pages = batch_size * pages_per_seq
    q = torch.randn(batch_size, qo_heads, head_dim, device="cuda").to(
        torch.float8_e4m3fn
    )
    k = torch.randn(num_pages, kv_heads, page_size, head_dim, device="cuda").to(
        torch.float8_e4m3fn
    )
    v = torch.randn_like(k, dtype=torch.float32).to(torch.float8_e4m3fn)
    # Permuted pages and padded table rows catch assumptions about pool layout.
    table = torch.full(
        (batch_size, pages_per_seq + 3), -1, dtype=torch.int32, device="cuda"
    )
    table[:, :pages_per_seq] = torch.randperm(num_pages, device="cuda").reshape(
        batch_size, -1
    )
    lengths_host = [max_kv_len - 13, max_kv_len // 2 + 7]
    lengths = torch.tensor(lengths_host, dtype=torch.int32, device="cuda")
    bmm1 = torch.tensor([0.25 / math.sqrt(head_dim)], device="cuda")
    bmm2 = torch.tensor([1.75], device="cuda")
    output = torch.empty_like(q, dtype=torch.bfloat16)
    splits = 1 if mode == "disabled" else 4
    cfg = make_decode_config(
        headdim=head_dim,
        args={
            "tile_size_q": 16,
            "groups_tokens_heads_q": True,
            "use_keeps_mma_ab": False,
        },
        seq_len_q=1,
        seq_len_kv=max_kv_len,
        batch_size=batch_size,
        num_heads_q=qo_heads,
        num_heads_kv=kv_heads,
        qkv_dtype=cutlass.Float8E4M3FN,
        o_dtype=cutlass.BFloat16,
        qkv_layout="pagedKv",
        num_tokens_per_page=page_size,
        split_kv_mode=mode,
        splits_kv=splits,
        max_splits_kv=splits,
        mask_type="causal",
        auto_tuner=False,
    )

    def plan(output_dtype: torch.dtype, use_device_scales: bool):
        spec = decode._decode_launch_spec_from_config(
            replace(
                cfg,
                out_dtype=cutlass.BFloat16
                if output_dtype == torch.bfloat16
                else cutlass.Float16,
            ),
            batch_size=batch_size,
            num_qo_heads=qo_heads,
            num_kv_heads=kv_heads,
            head_dim=head_dim,
            seq_len_q=1,
            max_active_clusters=torch.cuda.get_device_properties(
                0
            ).multi_processor_count,
        )
        monkeypatch.setattr(
            decode, "_resolve_decode_launch_spec", lambda *args, **kwargs: spec
        )
        wrapper = decode.BatchDecodePagedTSWrapper()
        wrapper.plan(
            q.device,
            batch_size,
            qo_heads,
            kv_heads,
            head_dim,
            page_size,
            max_kv_len,
            q_dtype=q.dtype,
            out_dtype=output_dtype,
            mask_type="causal",
            use_device_scales=use_device_scales,
        )
        return wrapper

    wrapper = plan(torch.bfloat16, True)
    fp16_wrapper = plan(torch.float16, False)
    bf16_host_wrapper = plan(torch.bfloat16, False)

    def run() -> torch.Tensor:
        if mode == "gmem_reduction":
            wrapper._plan_state.workspace.split_kv_counter.zero_()
        return wrapper.run(
            q,
            (k, v),
            lengths,
            table,
            out=output,
            softmax_scale=bmm1,
            output_scale=bmm2,
            validate=False,
        )

    def reference() -> torch.Tensor:
        results = []
        for batch_idx, length in enumerate(lengths_host):
            ids = table[batch_idx, :pages_per_seq].long()
            keys = (
                k.float()[ids]
                .permute(1, 0, 2, 3)
                .reshape(kv_heads, -1, head_dim)[:, :length]
            )
            values = (
                v.float()[ids]
                .permute(1, 0, 2, 3)
                .reshape(kv_heads, -1, head_dim)[:, :length]
            )
            keys = keys.repeat_interleave(head_ratio, dim=0)
            values = values.repeat_interleave(head_ratio, dim=0)
            scores = torch.einsum("hd,htd->ht", q[batch_idx].float(), keys) * bmm1
            results.append(
                torch.einsum("ht,htd->hd", scores.softmax(-1), values) * bmm2
            )
        return torch.stack(results)

    def check_output() -> None:
        # The FP8 kernel rounds the intermediate P operand to E4M3. Compare
        # math with that quantization tolerance, and the existing FP16 output
        # path tightly to isolate BF16 stores and device-scale handling.
        torch.testing.assert_close(output.float(), reference(), atol=0.01, rtol=0.05)
        if mode == "gmem_reduction":
            fp16_wrapper._plan_state.workspace.split_kv_counter.zero_()
        fp16_output = fp16_wrapper.run(
            q,
            (k, v),
            lengths,
            table,
            softmax_scale=bmm1.item(),
            output_scale=bmm2.item(),
            validate=False,
        )
        torch.testing.assert_close(
            output.float(), fp16_output.float(), atol=0.0005, rtol=0.008
        )

        if mode == "gmem_reduction":
            bf16_host_wrapper._plan_state.workspace.split_kv_counter.zero_()
        bf16_host_output = bf16_host_wrapper.run(
            q,
            (k, v),
            lengths,
            table,
            softmax_scale=bmm1.item(),
            output_scale=bmm2.item(),
            validate=False,
        )
        torch.testing.assert_close(
            output.float(), bf16_host_output.float(), atol=0.0005, rtol=0.008
        )

    if mode == "gmem_reduction":
        wrapper._plan_state.workspace.split_kv_counter.zero_()
    wrapper.run(
        q,
        (k, v),
        lengths,
        table,
        out=output,
        softmax_scale=bmm1,
        output_scale=bmm2,
        validate=True,
    )
    torch.cuda.synchronize()
    check_output()

    def no_host_read(*args, **kwargs):
        raise AssertionError("decode must not read scales back to the host")

    with monkeypatch.context() as guard:
        guard.setattr(torch.Tensor, "item", no_host_read)
        guard.setattr(torch.Tensor, "__float__", no_host_read)
        sync_debug_mode = torch.cuda.get_sync_debug_mode()
        try:
            torch.cuda.set_sync_debug_mode("error")
            run()
        finally:
            torch.cuda.set_sync_debug_mode(sync_debug_mode)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        # Both attention scores and final output must consume updated scales.
        bmm1.mul_(2)
        bmm2.mul_(0.5)
        graph.replay()
    torch.cuda.synchronize()
    check_output()
