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

Public-API tests of
``flashinfer.mla.prepare_nvfp4_batch_decode_with_kv_cache_mla``, the
``backend="cake"`` entry of the experimental NVFP4 MLA decode package; the
package's own suite is ``test_cake_nvfp4_mla_decode.py``.
"""

import pytest
import torch

from flashinfer.experimental.nvfp4_mla_decode import cake_jit
from flashinfer.experimental.nvfp4_mla_decode.cake_backend import (
    DSV4_Q_LEN,
    HEAD_DIM,
    PAGE_SIZE,
    SF_VEC,
    kv_pages,
    nvfp4_mla_decode_workspace_size,
    quantize_nvfp4,
)
from flashinfer.mla import prepare_nvfp4_batch_decode_with_kv_cache_mla

ATOL = RTOL = 0.1
REL_L2_MAX = 0.05
LSE_ATOL = LSE_RTOL = 0.05
_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _gpu_skip_reason():
    if not torch.cuda.is_available():
        return "CUDA required"
    arch = cake_jit.arch_for_capability(torch.cuda.get_device_capability())
    if arch is None:
        return "SM100 or SM103 required"
    if not cake_jit.program_available(arch):
        return "generated programs not registered for this device"
    return None


def dequantize_nvfp4(packed, scale):
    """Exact FP32 decode of packed E2M1 codes with UE4M3 block-16 scales."""
    lo = packed & 0x0F
    hi = packed >> 4
    codes = torch.stack((lo, hi), dim=-1).reshape(
        *packed.shape[:-1], packed.shape[-1] * 2
    )
    table = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=packed.device)
    magnitude = table[(codes & 0x7).long()]
    sign = torch.where((codes & 0x8) != 0, -1.0, 1.0)
    values = magnitude * sign
    blocks = values.reshape(*values.shape[:-1], values.shape[-1] // SF_VEC, SF_VEC)
    scale = scale.view(torch.float8_e4m3fn).float()
    return (blocks * scale.unsqueeze(-1)).reshape(*values.shape)


def make_inputs(kv_lens, num_heads, *, q_len, device, seed=0):
    """Deterministic NVFP4 paged decode inputs (no sinks) with a peaked softmax."""
    gen = torch.Generator(device=device).manual_seed(seed)
    batch = len(kv_lens)
    pages_per_seq = [kv_pages(kv) for kv in kv_lens]
    total_pages = sum(pages_per_seq)
    max_pages = max(pages_per_seq)
    k_full = torch.randn(
        (total_pages, PAGE_SIZE, HEAD_DIM), generator=gen, device=device
    )
    permutation = torch.randperm(total_pages, generator=gen, device=device).to(
        torch.int32
    )
    block_tables = torch.zeros((batch, max_pages), dtype=torch.int32, device=device)
    offset = 0
    for b, count in enumerate(pages_per_seq):
        block_tables[b, :count] = permutation[offset : offset + count]
        offset += count
    q = torch.randn((batch * q_len, num_heads, HEAD_DIM), generator=gen, device=device)
    for b in range(batch):
        for i in range(q_len):
            visible = kv_lens[b] - q_len + i + 1
            target = int(
                torch.randint(0, visible, (1,), generator=gen, device=device).item()
            )
            page = int(block_tables[b, target // PAGE_SIZE].item())
            q[b * q_len + i] += 0.2 * k_full[page, target % PAGE_SIZE]
    query, query_scale = quantize_nvfp4(q)
    kv_cache, kv_scale = quantize_nvfp4(k_full)
    return dict(
        query=query,
        query_scale=query_scale,
        kv_cache=kv_cache,
        kv_scale=kv_scale,
        block_tables=block_tables,
        seq_lens=torch.tensor(kv_lens, dtype=torch.int32, device=device),
        sm_scale=HEAD_DIM**-0.5,
    )


def reference(inputs, kv_lens, q_len):
    """FP32 causal attention over the dequantized operands: ``(O bf16, LSE f32)``."""
    q_all = dequantize_nvfp4(inputs["query"], inputs["query_scale"])
    total_q, num_heads = q_all.shape[:2]
    out = torch.empty(
        (total_q, num_heads, HEAD_DIM), dtype=torch.float32, device=q_all.device
    )
    lse = torch.empty((total_q, num_heads), dtype=torch.float32, device=q_all.device)
    for b, kv_len in enumerate(kv_lens):
        pages = inputs["block_tables"][b, : kv_pages(kv_len)].long()
        k = dequantize_nvfp4(inputs["kv_cache"][pages], inputs["kv_scale"][pages])
        k = k.reshape(-1, HEAD_DIM)[:kv_len]  # K and V share the row
        rows = slice(b * q_len, (b + 1) * q_len)
        logits = torch.einsum("rhd,nd->hrn", q_all[rows], k) * inputs["sm_scale"]
        positions = torch.arange(kv_len, device=k.device)
        row_limit = kv_len - q_len + torch.arange(q_len, device=k.device) + 1
        mask = positions[None, :] < row_limit[:, None]  # [q_len, kv_len]
        logits = logits.masked_fill(~mask[None], float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        out[rows] = torch.einsum("hrn,nd->hrd", probs, k).permute(1, 0, 2)
        lse[rows] = torch.logsumexp(logits, dim=-1).permute(1, 0)
    return out.to(torch.bfloat16), lse


def check_outputs(out, lse, ref_out, ref_lse):
    assert torch.isfinite(out.float()).all()
    torch.testing.assert_close(out, ref_out, atol=ATOL, rtol=RTOL)
    rel = (out.float() - ref_out.float()).norm(dim=-1) / ref_out.float().norm(
        dim=-1
    ).clamp_min(1e-6)
    assert float(rel.max()) <= REL_L2_MAX, float(rel.max())
    assert torch.isfinite(lse).all()
    torch.testing.assert_close(lse, ref_lse, atol=LSE_ATOL, rtol=LSE_RTOL)


def test_public_entry_point_is_experimental():
    assert prepare_nvfp4_batch_decode_with_kv_cache_mla.is_experimental is True


def test_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend"):
        prepare_nvfp4_batch_decode_with_kv_cache_mla(
            None, None, None, None, None, None, None, sm_scale=1.0, backend="unknown"
        )


def test_rejects_host_tensors():
    inputs = make_inputs([64, 100], 16, q_len=6, device="cpu")
    workspace = torch.empty(1 << 20, dtype=torch.uint8)
    with pytest.raises(ValueError, match="CUDA"):
        prepare_nvfp4_batch_decode_with_kv_cache_mla(
            inputs["query"],
            inputs["query_scale"],
            inputs["kv_cache"],
            inputs["kv_scale"],
            inputs["block_tables"],
            inputs["seq_lens"],
            workspace,
            sm_scale=inputs["sm_scale"],
        )


def test_rejects_unsupported_compute_capability():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if cake_jit.arch_for_capability(torch.cuda.get_device_capability()) is not None:
        pytest.skip("device is supported; nothing to reject")
    inputs = make_inputs([64], 128, q_len=1, device="cuda")
    workspace = torch.empty(1 << 20, dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="compute capability"):
        prepare_nvfp4_batch_decode_with_kv_cache_mla(
            inputs["query"],
            inputs["query_scale"],
            inputs["kv_cache"],
            inputs["kv_scale"],
            inputs["block_tables"],
            inputs["seq_lens"],
            workspace,
            sm_scale=inputs["sm_scale"],
        )


def test_public_api_returns_out_without_lse():
    reason = _gpu_skip_reason()
    if reason:
        pytest.skip(reason)
    kv_lens = [512, 300]
    inputs = make_inputs(kv_lens, 16, q_len=DSV4_Q_LEN, device="cuda")
    num_sms = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    workspace = torch.empty(
        nvfp4_mla_decode_workspace_size(kv_lens, 16, num_sms=num_sms, q_len=DSV4_Q_LEN),
        dtype=torch.uint8,
        device="cuda",
    )
    decode = prepare_nvfp4_batch_decode_with_kv_cache_mla(
        inputs["query"],
        inputs["query_scale"],
        inputs["kv_cache"],
        inputs["kv_scale"],
        inputs["block_tables"],
        inputs["seq_lens"],
        workspace,
        sm_scale=inputs["sm_scale"],
        seq_lens_cpu=inputs["seq_lens"].cpu(),
    )
    out = decode()
    assert isinstance(out, torch.Tensor) and out.shape == (
        len(kv_lens) * DSV4_Q_LEN,
        16,
        HEAD_DIM,
    )
    torch.cuda.synchronize()
    ref_out, ref_lse = reference(inputs, kv_lens, DSV4_Q_LEN)
    check_outputs(out, decode.lse, ref_out, ref_lse)
