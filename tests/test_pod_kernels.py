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

import math
import random
from typing import Tuple

import pytest
import torch
from jit_utils import jit_prefill_attention_func_args

import flashinfer


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    if flashinfer.jit.has_prebuilt_ops:
        yield
    else:
        try:
            flashinfer.jit.parallel_load_modules(
                jit_prefill_attention_func_args(
                    [torch.float16],  # q_dtypes
                    [torch.float16],  # kv_dtypes
                    [128, 256],  # head_dims
                    [0, 1],  # pos_encoding_modes
                    [False],  # use_sliding_windows
                    [False],  # use_logits_soft_caps
                    [False],  # use_fp16_qk_reductions
                )
            )
        except Exception as e:
            # abort the test session if warmup fails
            pytest.exit(str(e))
        finally:
            yield


def _gen_reqs(bsz: int, qo_len: int, seq_len: Tuple[int, int], stride: int):
    reqs = []
    for i in range(bsz):
        if (i + 1) % stride == 0:
            len_q = qo_len
        else:
            len_q = 1

        len_kv = int(random.randint(seq_len[0], seq_len[1]))
        reqs.append((len_q, len_kv))
    return reqs


def _gen_metadata(
    reqs, page_size, kv_layout, num_qo_heads, num_kv_heads, head_dim, device
):
    total_qo_len = sum([r[0] for r in reqs])
    total_kv_len = sum([r[1] for r in reqs])

    q = torch.randn(
        total_qo_len,
        num_qo_heads,
        head_dim,
        device=device,
        dtype=torch.half,
    )

    kv_indptr_cpu = [0]
    qo_indptr_cpu = [0]
    kv_last_page_cpu = []
    for req in reqs:
        kv_indptr_cpu.append(kv_indptr_cpu[-1] + math.ceil(req[1] / page_size))
        kv_last_page_cpu.append((req[1] - 1) % page_size + 1)
        qo_indptr_cpu.append(qo_indptr_cpu[-1] + req[0])

    kv_indices_cpu = list(range(kv_indptr_cpu[-1]))
    kv_indices_cpu.extend([0] * 256)

    kv_indptr_cpu = torch.tensor(kv_indptr_cpu, dtype=torch.int32, device="cpu")
    kv_indices_cpu = torch.tensor(kv_indices_cpu, dtype=torch.int32, device="cpu")
    kv_last_page_cpu = torch.tensor(kv_last_page_cpu, dtype=torch.int32, device="cpu")
    qo_indptr_cpu = torch.tensor(qo_indptr_cpu, dtype=torch.int32, device="cpu")

    if kv_layout == "HND":
        kv_data = torch.randn(
            len(kv_indices_cpu),
            2,
            num_kv_heads,
            page_size,
            head_dim,
            device=device,
            dtype=torch.float32,
        )
    else:
        kv_data = torch.randn(
            len(kv_indices_cpu),
            2,
            page_size,
            num_kv_heads,
            head_dim,
            device=device,
            dtype=torch.float32,
        )

    return q, kv_data, kv_indptr_cpu, kv_indices_cpu, kv_last_page_cpu, qo_indptr_cpu


@pytest.mark.parametrize("batch_size", [12, 17, 64])
@pytest.mark.parametrize("kv_len", [54, 511, 2048])
@pytest.mark.parametrize("qo_len", [17, 47, 127, 577])
@pytest.mark.parametrize("stride", [1, 2, 5, 1024])
@pytest.mark.parametrize("page_size", [1, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 28])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("kv_layout", ["NHD", "HND"])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE"])
@pytest.mark.parametrize("use_cuda_graph", [False])
@pytest.mark.parametrize("logits_soft_cap", [0.0])
@pytest.mark.parametrize("return_lse", [False])
@pytest.mark.parametrize("contiguous_kv", [True])
def test_pod_with_paged_kv_cache(
    batch_size,
    kv_len,
    qo_len,
    stride,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    causal,
    kv_layout,
    pos_encoding_mode,
    use_cuda_graph,
    logits_soft_cap,
    return_lse,
    contiguous_kv,
):
    if qo_len > kv_len and causal:
        pytest.skip("qo_len > kv_len and causal is not supported")

    reqs = _gen_reqs(batch_size, qo_len, (kv_len, kv_len + 128), stride)
    (
        q,
        kv_data_fp32,
        kv_indptr_cpu,
        kv_indices_cpu,
        kv_last_page_len_cpu,
        q_indptr_cpu,
    ) = _gen_metadata(
        reqs, page_size, kv_layout, num_qo_heads, num_kv_heads, head_dim, "cuda:0"
    )
    kv_data = kv_data_fp32.half()

    workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    if not use_cuda_graph:
        q_indptr_gpu = q_indptr_cpu.to(0)
        kv_indptr_gpu = kv_indptr_cpu.to(0)
        kv_indices_gpu = kv_indices_cpu.to(0)
        kv_last_page_len_gpu = kv_last_page_len_cpu.to(0)
        wrapper = flashinfer.PODWithPagedKVCacheWrapper(workspace_buffer, kv_layout)
        wrapper.plan(
            q_indptr_gpu,
            kv_indptr_gpu,
            kv_indices_gpu,
            kv_last_page_len_gpu,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
        )
        if return_lse:
            o, _ = wrapper.run(q, kv_data, return_lse=True)
        else:
            o = wrapper.run(q, kv_data)

        # test with pre-allocated output
        o_buffer = torch.empty_like(o)
        wrapper.run(q, kv_data, out=o_buffer)
        torch.testing.assert_close(o, o_buffer, rtol=1e-3, atol=1e-3)

    for i in range(batch_size):
        perm_dims = [0, 2, 1, 3] if kv_layout == "HND" else [0, 1, 2, 3]
        perm_dims_last = [1, 0, 2] if kv_layout == "HND" else [0, 1, 2]
        qi = q[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]
        ki = torch.cat(
            [
                kv_data_fp32[kv_indptr_cpu[i] : kv_indptr_cpu[i + 1] - 1, 0]
                .permute(*perm_dims)
                .reshape(-1, num_kv_heads, head_dim),
                (
                    kv_data_fp32[
                        kv_indptr_cpu[i + 1] - 1, 0, :, : kv_last_page_len_cpu[i]
                    ]
                    if kv_layout == "HND"
                    else kv_data_fp32[
                        kv_indptr_cpu[i + 1] - 1, 0, : kv_last_page_len_cpu[i], :
                    ]
                )
                .permute(*perm_dims_last)
                .reshape(-1, num_kv_heads, head_dim),
            ],
            dim=0,
        ).half()
        vi = torch.cat(
            [
                kv_data_fp32[kv_indptr_cpu[i] : kv_indptr_cpu[i + 1] - 1, 1]
                .permute(*perm_dims)
                .reshape(-1, num_kv_heads, head_dim),
                (
                    kv_data_fp32[
                        kv_indptr_cpu[i + 1] - 1, 1, :, : kv_last_page_len_cpu[i]
                    ]
                    if kv_layout == "HND"
                    else kv_data_fp32[
                        kv_indptr_cpu[i + 1] - 1, 1, : kv_last_page_len_cpu[i], :
                    ]
                )
                .permute(*perm_dims_last)
                .reshape(-1, num_kv_heads, head_dim),
            ],
            dim=0,
        ).half()
        o_ref_i = flashinfer.prefill.single_prefill_with_kv_cache(
            qi,
            ki,
            vi,
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
        )
        o_i = o[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]
        torch.testing.assert_close(o_i, o_ref_i, rtol=1e-3, atol=1e-3)
