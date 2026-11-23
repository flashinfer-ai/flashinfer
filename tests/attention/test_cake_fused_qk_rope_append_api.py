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

Tests of the flashinfer public surface of the Cake fused QK RMSNorm + NeoX
RoPE + paged KV append (BF16) kernel: the ``flashinfer.cake_fused_qk_rope_append``
facade.  Its entry point is a pass-through to
``flashinfer.cake_fused_qk_rope_append_backend``; this module checks that
identity (the caller's ``out_q`` is returned, the same re-exported objects,
the same bits as the backend) and leaves the numerical contract to
``test_cake_fused_qk_rope_append.py``, which imports only the backend package.
"""

from __future__ import annotations

import pytest
import torch

import flashinfer
import flashinfer.cake_fused_qk_rope_append as facade
import flashinfer.cake_fused_qk_rope_append_backend as backend
from flashinfer.cake_fused_qk_rope_append import (
    cake_fused_qk_rmsnorm_rope_append_paged_kv_cache,
    launch_plan,
)
from flashinfer.cake_fused_qk_rope_append_backend import cake_jit as loader
from flashinfer.jit.env import has_flashinfer_jit_cache
from flashinfer.utils import get_compute_capability

HEAD_DIM = 128
POISON = 1234.0
SEED = 8920


def _device() -> torch.device:
    return torch.device("cuda:0")


def _arch():
    if not torch.cuda.is_available():
        return None
    return loader.arch_for_capability(get_compute_capability(_device()))


def _skip_unless_supported() -> None:
    if _arch() is None:
        pytest.skip("cake_fused_qk_rope_append needs an SM90 / SM100 / SM103 device")


@pytest.fixture(autouse=not has_flashinfer_jit_cache(), scope="module")
def warmup_jit():
    if _arch() is not None:
        flashinfer.jit.build_jit_specs(loader.specs_for_arch(_arch()), verbose=False)
    yield


def test_facade_reexports_the_backend_surface() -> None:
    assert facade.__all__ == [
        "cake_fused_qk_rmsnorm_rope_append_paged_kv_cache",
        "launch_plan",
    ]
    assert launch_plan is backend.launch_plan
    assert facade.HEAD_DIM == backend.HEAD_DIM == HEAD_DIM
    assert (
        facade.SUPPORTED_COMPUTE_CAPABILITIES
        == backend.SUPPORTED_COMPUTE_CAPABILITIES
        == ((9, 0), (10, 0), (10, 3))
    )
    # 32 decode rows of one kv head on 64-row pages, four warps per row: one row CTA
    # per row, eight 4 KiB clearing units per request, four clearing warps per CTA.
    plan = launch_plan(32, 32, 64, 1, 4)
    assert set(plan) == {"num_row_ctas", "clear_units_per_request", "grid"}
    assert plan["num_row_ctas"] == 32
    assert plan["clear_units_per_request"] == 8
    clear_ctas = -(-32 * plan["clear_units_per_request"] // 4)
    assert plan["grid"] == (plan["num_row_ctas"] + clear_ctas, 1, 1)
    assert launch_plan(32, 32, 64, 8, 20) == backend.launch_plan(32, 32, 64, 8, 20)


def _case(device: torch.device):
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    q_lens, ctx_lens, page_size = (
        [1] * 6 + [16, 3],
        [2048 + 3 * i for i in range(8)],
        64,
    )
    hq, hkv = 8, 1
    seq_lens = [c + q for c, q in zip(ctx_lens, q_lens, strict=True)]
    pages_per_req = [(s + page_size - 1) // page_size for s in seq_lens]
    total_pages = sum(pages_per_req) + 2
    perm = torch.randperm(total_pages, generator=generator)
    page_indices = torch.zeros((len(q_lens), max(pages_per_req)), dtype=torch.int32)
    cursor = 0
    for b, n in enumerate(pages_per_req):
        page_indices[b, :n] = perm[cursor : cursor + n].to(torch.int32)
        cursor += n
    q_indptr = torch.zeros(len(q_lens) + 1, dtype=torch.int32)
    q_indptr[1:] = torch.cumsum(torch.tensor(q_lens, dtype=torch.int32), 0)
    T = int(q_indptr[-1])
    qkv = torch.randn(
        (T, (hq + 2 * hkv) * HEAD_DIM), generator=generator, dtype=torch.float32
    ).to(torch.bfloat16)
    half = HEAD_DIM // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half, dtype=torch.float64) / half))
    freqs = torch.outer(torch.arange(max(seq_lens) + 1, dtype=torch.float64), inv_freq)
    cos_sin = torch.cat([freqs.cos(), freqs.sin()], dim=1).to(torch.float32)
    weight = (torch.rand(HEAD_DIM, generator=generator) + 0.5).to(torch.bfloat16)
    cache_shape = (total_pages, page_size, hkv, HEAD_DIM)
    return dict(
        qkv=qkv.to(device),
        cos_sin=cos_sin.to(device),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
        q_indptr=q_indptr.to(device),
        page_indices=page_indices.to(device),
        q_norm_weight=weight.float().to(device),
        k_norm_weight=weight.float().to(device),
        cache_shape=cache_shape,
        T=T,
        hq=hq,
        hkv=hkv,
    )


def _buffers(case, device: torch.device):
    kc = torch.full(case["cache_shape"], POISON, dtype=torch.bfloat16, device=device)
    vc = torch.full_like(kc, POISON)
    oq = torch.full(
        (case["T"], case["hq"], HEAD_DIM), POISON, dtype=torch.bfloat16, device=device
    )
    return kc, vc, oq


def test_facade_returns_the_caller_buffer_and_matches_the_backend() -> None:
    """The facade forwards every argument: it returns the caller's ``out_q`` and
    writes the same bits as the backend package's entry point."""
    _skip_unless_supported()
    device = _device()
    case = _case(device)
    kw = dict(
        num_q_heads=case["hq"],
        num_kv_heads=case["hkv"],
        qk_norm_policy=2,
        q_norm_weight=case["q_norm_weight"],
        k_norm_weight=case["k_norm_weight"],
    )
    args = (
        case["qkv"],
        case["cos_sin"],
        case["seq_lens"],
        case["q_indptr"],
        case["page_indices"],
    )
    kc, vc, oq = _buffers(case, device)
    ret = cake_fused_qk_rmsnorm_rope_append_paged_kv_cache(
        *args, kc, vc, out_q=oq, **kw
    )
    assert ret is oq
    bkc, bvc, boq = _buffers(case, device)
    bret = backend.cake_fused_qk_rmsnorm_rope_append_paged_kv_cache(
        *args, bkc, bvc, out_q=boq, **kw
    )
    assert bret is boq
    torch.cuda.synchronize()
    assert torch.equal(oq.view(torch.int16), boq.view(torch.int16))
    assert torch.equal(kc.view(torch.int16), bkc.view(torch.int16))
    assert torch.equal(vc.view(torch.int16), bvc.view(torch.int16))
    # without a caller-owned ``out_q`` the facade allocates one of the contract shape
    akc, avc, _ = _buffers(case, device)
    allocated = cake_fused_qk_rmsnorm_rope_append_paged_kv_cache(*args, akc, avc, **kw)
    torch.cuda.synchronize()
    assert allocated.dtype == torch.bfloat16
    assert tuple(allocated.shape) == (case["T"], case["hq"], HEAD_DIM)
    assert torch.equal(allocated.view(torch.int16), oq.view(torch.int16))
