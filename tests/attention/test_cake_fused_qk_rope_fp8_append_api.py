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
RoPE + FP8 quantize + paged KV append kernel: the
``flashinfer.cake_fused_qk_rope_fp8_append`` facade.  Its entry point is a
pass-through to ``flashinfer.cake_fused_qk_rope_fp8_append_backend``; this
module checks that identity (the caller's ``out_q`` / ``q_scale`` /
``split_k_flag`` are returned, the same re-exported objects, the same bytes as
the backend) and leaves the numerical contract to
``test_cake_fused_qk_rope_fp8_append.py``, which imports only the backend
package.
"""

from __future__ import annotations

import pytest
import torch

import flashinfer
import flashinfer.cake_fused_qk_rope_fp8_append as facade
import flashinfer.cake_fused_qk_rope_fp8_append_backend as backend
from flashinfer.cake_fused_qk_rope_fp8_append import (
    cake_fused_qk_rmsnorm_rope_quantize_fp8_append_paged_kv_cache,
    launch_plan,
)
from flashinfer.cake_fused_qk_rope_fp8_append_backend import cake_jit as loader
from flashinfer.jit.env import has_flashinfer_jit_cache
from flashinfer.utils import get_compute_capability

HEAD_DIM = 128
POISON_BYTE = 0xFF
SEED = 8930


def _device() -> torch.device:
    return torch.device("cuda:0")


def _arch():
    if not torch.cuda.is_available():
        return None
    return loader.arch_for_capability(get_compute_capability(_device()))


def _skip_unless_supported() -> None:
    if _arch() is None:
        pytest.skip(
            "cake_fused_qk_rope_fp8_append needs an SM90 / SM100 / SM103 device"
        )


@pytest.fixture(autouse=not has_flashinfer_jit_cache(), scope="module")
def warmup_jit():
    if _arch() is not None:
        flashinfer.jit.build_jit_specs(loader.specs_for_arch(_arch()), verbose=False)
    yield


def test_facade_reexports_the_backend_surface() -> None:
    assert facade.__all__ == [
        "cake_fused_qk_rmsnorm_rope_quantize_fp8_append_paged_kv_cache",
        "launch_plan",
    ]
    assert launch_plan is backend.launch_plan
    assert facade.HEAD_DIM == backend.HEAD_DIM == HEAD_DIM
    assert facade.FP8_MAX == backend.FP8_MAX == 448.0
    assert facade.SUPPORTED_HEADS == backend.SUPPORTED_HEADS == ((8, 1), (64, 8))
    assert (
        facade.SUPPORTED_COMPUTE_CAPABILITIES
        == backend.SUPPORTED_COMPUTE_CAPABILITIES
        == ((9, 0), (10, 0), (10, 3))
    )
    assert launch_plan(40, 32, 8) == {"grid": (72, 8, 1)}


def _poisoned_fp8(shape, device):
    t = torch.empty(shape, dtype=torch.float8_e4m3fn, device=device)
    t.view(torch.uint8).fill_(POISON_BYTE)
    return t


def _case(device: torch.device):
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    q_lens, ctx_lens, page_size = [16, 3, 0, 9], [2048, 60, 5, 0], 64
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
    return dict(
        qkv=qkv.to(device),
        cos_sin=cos_sin.to(device),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
        q_indptr=q_indptr.to(device),
        page_indices=page_indices.to(device),
        q_norm_weight=weight.float().to(device),
        k_norm_weight=weight.float().to(device),
        k_scale=torch.tensor([0.02], dtype=torch.float32, device=device),
        v_scale=torch.tensor([0.03], dtype=torch.float32, device=device),
        cache_shape=(total_pages, page_size, hkv, HEAD_DIM),
        T=T,
        B=len(q_lens),
        hq=hq,
        hkv=hkv,
        max_seqlen=16,
    )


def _buffers(case, device: torch.device):
    kc = _poisoned_fp8(case["cache_shape"], device)
    vc = _poisoned_fp8(case["cache_shape"], device)
    oq = _poisoned_fp8((case["T"], case["hq"], HEAD_DIM), device)
    qs = torch.full((case["B"], case["hq"], 128), float("nan"), device=device)
    flags = torch.full((case["B"], case["hkv"]), 7, dtype=torch.int32, device=device)
    return kc, vc, oq, qs, flags


def _bytes(kc, vc, oq, qs, flags):
    return [
        oq.view(torch.uint8),
        qs.view(torch.int32),
        flags,
        kc.view(torch.uint8),
        vc.view(torch.uint8),
    ]


def test_facade_returns_the_caller_buffers_and_matches_the_backend() -> None:
    """The facade forwards every argument: it returns the caller's ``out_q``,
    ``q_scale`` and ``split_k_flag`` and writes the same bytes as the backend
    package's entry point."""
    _skip_unless_supported()
    device = _device()
    case = _case(device)

    def call(fn, kc, vc, oq, qs, flags):
        return fn(
            case["qkv"],
            case["cos_sin"],
            case["seq_lens"],
            case["q_indptr"],
            case["page_indices"],
            (kc, vc),
            True,
            case["k_scale"],
            case["v_scale"],
            1,
            max_seqlen=case["max_seqlen"],
            q_norm_weight=case["q_norm_weight"],
            k_norm_weight=case["k_norm_weight"],
            qk_norm_policy=2,
            out_q=oq,
            q_scale=qs,
            split_k_flag=flags,
        )

    mine = _buffers(case, device)
    ret = call(cake_fused_qk_rmsnorm_rope_quantize_fp8_append_paged_kv_cache, *mine)
    assert ret[0] is mine[2] and ret[1] is mine[3] and ret[2] is mine[4]
    theirs = _buffers(case, device)
    bret = call(
        backend.cake_fused_qk_rmsnorm_rope_quantize_fp8_append_paged_kv_cache, *theirs
    )
    assert bret[0] is theirs[2] and bret[1] is theirs[3] and bret[2] is theirs[4]
    torch.cuda.synchronize()
    for got, want in zip(_bytes(*mine), _bytes(*theirs), strict=True):
        assert torch.equal(got, want)
    assert bool((mine[4] == 0).all())
    # without caller-owned outputs the facade allocates them in the contract shapes
    kc, vc, _oq, _qs, _flags = _buffers(case, device)
    out_q, q_scale, split_k_flag = (
        cake_fused_qk_rmsnorm_rope_quantize_fp8_append_paged_kv_cache(
            case["qkv"],
            case["cos_sin"],
            case["seq_lens"],
            case["q_indptr"],
            case["page_indices"],
            (kc, vc),
            True,
            case["k_scale"],
            case["v_scale"],
            1,
            max_seqlen=case["max_seqlen"],
            q_norm_weight=case["q_norm_weight"],
            k_norm_weight=case["k_norm_weight"],
            qk_norm_policy=2,
        )
    )
    torch.cuda.synchronize()
    assert out_q.dtype == torch.float8_e4m3fn
    assert tuple(out_q.shape) == (case["T"], case["hq"], HEAD_DIM)
    assert tuple(q_scale.shape) == (case["B"], case["hq"], 128)
    assert tuple(split_k_flag.shape) == (case["B"], case["hkv"])
    assert torch.equal(out_q.view(torch.uint8), mine[2].view(torch.uint8))
    assert torch.equal(split_k_flag, mine[4])
