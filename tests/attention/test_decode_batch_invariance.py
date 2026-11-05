"""
Copyright (c) 2026 by FlashInfer team.

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

"""Batch invariance of paged decode under the split-KV controls (issue #7981).

One request with an 8k KV cache is decoded alone and inside batches of 8, 16,
32 and 64 shorter requests. With ``disable_split_kv`` or ``fixed_split_size``
set, its output row and LSE must be bitwise identical across the batches on
every backend that accepts the control. The default schedule splits KV only
while the batch is small, so it is the known-variant negative control.
"""

import pytest
import torch

import flashinfer
from flashinfer.utils import is_sm100a_supported

NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE = 32, 8, 128, 16
TARGET_KV_LEN = 8192
BATCH_SIZES = [1, 8, 16, 32, 64]
OTHER_KV_LENS = torch.randint(
    256, 4096, (max(BATCH_SIZES) - 1,), generator=torch.Generator().manual_seed(1)
).tolist()
SPLIT_CONTROLS = [
    pytest.param({"disable_split_kv": True}, id="disable_split_kv"),
    pytest.param({"fixed_split_size": 64}, id="fixed_split_size"),
]


@pytest.fixture(scope="module")
def kv_cache():
    num_pages = (TARGET_KV_LEN + sum(OTHER_KV_LENS)) // PAGE_SIZE + len(OTHER_KV_LENS)
    return torch.randn(
        num_pages,
        2,
        PAGE_SIZE,
        NUM_KV_HEADS,
        HEAD_DIM,
        device="cuda",
        dtype=torch.bfloat16,
        generator=torch.Generator("cuda").manual_seed(0),
    )


def _page_table(batch_size, device):
    kv_lens = [TARGET_KV_LEN] + OTHER_KV_LENS[: batch_size - 1]
    pages = torch.tensor([(n + PAGE_SIZE - 1) // PAGE_SIZE for n in kv_lens])
    indptr = torch.cat([torch.zeros(1, dtype=torch.int64), pages.cumsum(0)]).int()
    indices = torch.arange(int(indptr[-1]), dtype=torch.int32)
    last_page_len = torch.tensor([(n - 1) % PAGE_SIZE + 1 for n in kv_lens]).int()
    return indptr.to(device), indices.to(device), last_page_len.to(device)


def _target_rows(kv_cache, backend, use_tensor_cores, kv_dtype, control):
    device = kv_cache.device
    cache = kv_cache if kv_dtype == torch.bfloat16 else kv_cache.to(kv_dtype)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    gen = torch.Generator(device).manual_seed(2)
    q_target = torch.randn(
        NUM_QO_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16, generator=gen
    )
    rows, resolved = {}, set()
    for batch_size in BATCH_SIZES:
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace, "NHD", use_tensor_cores=use_tensor_cores, backend=backend
        )
        wrapper.plan(
            *_page_table(batch_size, device),
            NUM_QO_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            PAGE_SIZE,
            q_data_type=torch.bfloat16,
            kv_data_type=kv_dtype,
            **control,
        )
        resolved.add(wrapper.resolved_backend)
        q = torch.randn(
            batch_size,
            NUM_QO_HEADS,
            HEAD_DIM,
            device=device,
            dtype=torch.bfloat16,
            generator=gen,
        )
        q[0] = q_target
        o, lse = wrapper.run(q, cache, return_lse=True)
        torch.cuda.synchronize()
        rows[batch_size] = (o[0].clone(), lse[0].clone())
    return rows, resolved


def _assert_batch_invariant(rows):
    o_alone, lse_alone = rows[1]
    for batch_size in BATCH_SIZES[1:]:
        o, lse = rows[batch_size]
        max_diff = (o.float() - o_alone.float()).abs().max().item()
        assert torch.equal(o, o_alone), (
            f"output differs at batch {batch_size}: {max_diff:.2e}"
        )
        assert torch.equal(lse, lse_alone), f"lse differs at batch {batch_size}"


@pytest.mark.parametrize("control", SPLIT_CONTROLS)
@pytest.mark.parametrize(
    "kv_dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16_kv", "e4m3_kv"]
)
@pytest.mark.parametrize("backend", ["fa2", "auto"])
def test_tensor_core_decode_is_batch_invariant(kv_cache, backend, kv_dtype, control):
    rows, resolved = _target_rows(kv_cache, backend, True, kv_dtype, control)
    assert resolved == {"fa2"}
    _assert_batch_invariant(rows)


def test_tensor_core_decode_default_split_is_batch_variant(kv_cache):
    rows, _ = _target_rows(kv_cache, "fa2", True, torch.bfloat16, {})
    o_alone, lse_alone = rows[1]
    o, lse = rows[BATCH_SIZES[-1]]
    assert not (torch.equal(o, o_alone) and torch.equal(lse, lse_alone))


@pytest.mark.skipif(
    not is_sm100a_supported(torch.device("cuda")), reason="cute-dsl decode needs SM100a"
)
@pytest.mark.parametrize(
    "control",
    [
        pytest.param({"disable_split_kv": True}, id="disable_split_kv"),
        pytest.param(
            {"fixed_split_size": 64},
            id="fixed_split_size",
            marks=pytest.mark.xfail(
                strict=True,
                reason="cute-dsl turns fixed_split_size into a split count reduced "
                "with float atomics, so the result is still batch-variant (#7981)",
            ),
        ),
    ],
)
def test_cute_dsl_decode_is_batch_invariant(kv_cache, control):
    pytest.importorskip("cutlass", reason="cute-dsl decode needs nvidia-cutlass-dsl")
    rows, resolved = _target_rows(kv_cache, "cute-dsl", True, torch.bfloat16, control)
    assert resolved == {"cute-dsl"}
    _assert_batch_invariant(rows)


@pytest.mark.parametrize(
    "control",
    [
        pytest.param(
            {"disable_split_kv": True},
            id="disable_split_kv",
            marks=pytest.mark.skip(
                reason="CUDA-core decode drops disable_split_kv until #6458 lands"
            ),
        ),
        pytest.param({"fixed_split_size": 64}, id="fixed_split_size"),
    ],
)
def test_cuda_core_decode_rejects_split_control(kv_cache, control):
    device = kv_cache.device
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, "NHD", use_tensor_cores=False
    )
    with pytest.raises(ValueError):
        wrapper.plan(
            *_page_table(1, device),
            NUM_QO_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            PAGE_SIZE,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
            **control,
        )
