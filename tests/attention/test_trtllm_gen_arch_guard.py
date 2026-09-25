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

# Explicit backend="trtllm-gen" requests on devices without trtllm-gen FMHA
# cubins must fail at the Python entry point, not inside the C++ runner.

import pytest
import torch

import flashinfer
import flashinfer.mla._core as mla_core
from flashinfer.utils import check_trtllm_gen_fmha_arch, get_compute_capability

GUARD_MATCH = r"backend='trtllm-gen' requires SM100, SM103 or SM107, got SM"


@pytest.mark.parametrize("capability", [(10, 0), (10, 3), (10, 7)])
def test_check_trtllm_gen_fmha_arch_accepts_trtllm_gen_devices(monkeypatch, capability):
    monkeypatch.setattr(
        "flashinfer.utils.get_compute_capability", lambda _device: capability
    )
    check_trtllm_gen_fmha_arch(torch.device("cuda"), sm12x_alternative="backend='xqa'")


@pytest.mark.parametrize(
    ("capability", "expect_sm12x_hint"),
    [((9, 0), False), ((11, 0), False), ((12, 0), True), ((12, 1), True)],
)
def test_check_trtllm_gen_fmha_arch_rejects_other_devices(
    monkeypatch, capability, expect_sm12x_hint
):
    monkeypatch.setattr(
        "flashinfer.utils.get_compute_capability", lambda _device: capability
    )
    with pytest.raises(ValueError, match=GUARD_MATCH) as excinfo:
        check_trtllm_gen_fmha_arch(
            torch.device("cuda"), sm12x_alternative="backend='xqa'"
        )
    message = str(excinfo.value)
    assert f"got SM{capability[0]}{capability[1]}" in message
    assert ("On SM120/SM121 use backend='xqa'." in message) == expect_sm12x_hint


def _require_device_without_trtllm_gen():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    capability = get_compute_capability(torch.device("cuda"))
    if capability[0] == 10:
        pytest.skip("trtllm-gen serves this device; the guard only fires elsewhere")
    return capability


def _mla_decode_args(device, sparse_mla_top_k):
    return dict(
        query=torch.zeros(1, 1, 128, 576, dtype=torch.bfloat16, device=device),
        kv_cache=torch.zeros(1, 1, 64, 576, dtype=torch.bfloat16, device=device),
        workspace_buffer=torch.zeros(1 << 20, dtype=torch.uint8, device=device),
        qk_nope_head_dim=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        block_tables=torch.zeros(
            (1, 1, sparse_mla_top_k) if sparse_mla_top_k else (1, 1),
            dtype=torch.int32,
            device=device,
        ),
        seq_lens=torch.ones(1, dtype=torch.int32, device=device),
        max_seq_len=64,
        sparse_mla_top_k=sparse_mla_top_k,
    )


@pytest.mark.parametrize("sparse_mla_top_k", [0, 128])
def test_trtllm_batch_decode_mla_rejects_explicit_trtllm_gen(sparse_mla_top_k):
    capability = _require_device_without_trtllm_gen()
    device = torch.device("cuda")
    with pytest.raises(ValueError, match=GUARD_MATCH) as excinfo:
        flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            **_mla_decode_args(device, sparse_mla_top_k), backend="trtllm-gen"
        )
    if capability[0] == 12:
        assert "backend='sparse'" in str(excinfo.value)
        assert "backend='xqa'" in str(excinfo.value)


@pytest.mark.parametrize(
    ("sparse_mla_top_k", "routed_function"),
    [
        (0, "xqa_batch_decode_with_kv_cache_mla"),
        (128, "_trtllm_batch_decode_sparse_mla_v32_sm120"),
    ],
)
def test_trtllm_batch_decode_mla_auto_routes_around_trtllm_gen_on_sm12x(
    monkeypatch, sparse_mla_top_k, routed_function
):
    capability = _require_device_without_trtllm_gen()
    if capability[0] != 12:
        pytest.skip("backend='auto' routes to xqa/sparse on SM120/SM121 only")
    device = torch.device("cuda")
    sentinel = torch.empty(0, device=device)
    calls = []

    def stub(*args, **kwargs):
        calls.append(routed_function)
        return sentinel

    monkeypatch.setattr(mla_core, routed_function, stub)
    out = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
        **_mla_decode_args(device, sparse_mla_top_k), backend="auto"
    )
    assert out is sentinel
    assert calls == [routed_function]


def test_trtllm_batch_decode_rejects_explicit_trtllm_gen():
    _require_device_without_trtllm_gen()
    device = torch.device("cuda")
    with pytest.raises(ValueError, match=GUARD_MATCH):
        flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            query=torch.zeros(1, 8, 128, dtype=torch.bfloat16, device=device),
            kv_cache=torch.zeros(1, 2, 1, 16, 128, dtype=torch.bfloat16, device=device),
            workspace_buffer=torch.zeros(1 << 20, dtype=torch.uint8, device=device),
            block_tables=torch.zeros(1, 1, dtype=torch.int32, device=device),
            seq_lens=torch.ones(1, dtype=torch.int32, device=device),
            max_seq_len=16,
            backend="trtllm-gen",
        )


def test_trtllm_batch_context_rejects_explicit_trtllm_gen():
    _require_device_without_trtllm_gen()
    device = torch.device("cuda")
    with pytest.raises(ValueError, match=GUARD_MATCH):
        flashinfer.prefill.trtllm_batch_context_with_kv_cache(
            query=torch.zeros(1, 8, 128, dtype=torch.bfloat16, device=device),
            kv_cache=torch.zeros(1, 2, 1, 16, 128, dtype=torch.bfloat16, device=device),
            workspace_buffer=torch.zeros(1 << 20, dtype=torch.uint8, device=device),
            block_tables=torch.zeros(1, 1, dtype=torch.int32, device=device),
            seq_lens=torch.ones(1, dtype=torch.int32, device=device),
            max_q_len=1,
            max_kv_len=16,
            bmm1_scale=1.0,
            bmm2_scale=1.0,
            batch_size=1,
            cum_seq_lens_q=torch.tensor([0, 1], dtype=torch.int32, device=device),
            cum_seq_lens_kv=torch.tensor([0, 1], dtype=torch.int32, device=device),
            backend="trtllm-gen",
        )


def test_trtllm_ragged_attention_deepseek_rejects_explicit_trtllm_gen():
    _require_device_without_trtllm_gen()
    device = torch.device("cuda")
    with pytest.raises(ValueError, match=GUARD_MATCH):
        flashinfer.prefill.trtllm_ragged_attention_deepseek(
            query=torch.zeros(1, 8, 192, dtype=torch.bfloat16, device=device),
            key=torch.zeros(1, 8, 192, dtype=torch.bfloat16, device=device),
            value=torch.zeros(1, 8, 128, dtype=torch.bfloat16, device=device),
            workspace_buffer=torch.zeros(1 << 20, dtype=torch.uint8, device=device),
            seq_lens=torch.ones(1, dtype=torch.int32, device=device),
            max_q_len=1,
            max_kv_len=1,
            bmm1_scale=1.0,
            bmm2_scale=1.0,
            o_sf_scale=1.0,
            batch_size=1,
            window_left=-1,
            cum_seq_lens_q=torch.tensor([0, 1], dtype=torch.int32, device=device),
            cum_seq_lens_kv=torch.tensor([0, 1], dtype=torch.int32, device=device),
            enable_pdl=None,
            is_causal=True,
            return_lse=False,
            backend="trtllm-gen",
        )
