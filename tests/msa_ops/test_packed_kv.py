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

Packed-KV tests: MSA sparse attention over K/V views split from a paged cache
that packs K and V in one 2*head_dim content dim per token (vLLM's layout).
"""

import pytest
import torch

from flashinfer.utils import is_sm12x_supported

B, HQ, HD, TOPK, PAGES_PER_REQ = 4, 8, 128, 16, 20
NUM_PAGES = B * PAGES_PER_REQ
SEQ_LEN = PAGES_PER_REQ * 128


def _skip_if_unsupported():
    if not torch.cuda.is_available() or not is_sm12x_supported(torch.device("cuda")):
        pytest.skip("MSA ops require SM120 or SM121 and CUDA >= 12.8")


def _split_cache(dtype, layout, hkv):
    """Packed cache split into K/V views, plus contiguous reference copies."""
    if layout == "HND":
        packed = torch.randn(
            NUM_PAGES, hkv, 128, 2 * HD, device="cuda", dtype=torch.bfloat16
        )
    else:  # NHD: heads inner of tokens
        packed = torch.randn(
            NUM_PAGES, 128, hkv, 2 * HD, device="cuda", dtype=torch.bfloat16
        ).permute(0, 2, 1, 3)
    packed = packed.to(dtype)
    k_view, v_view = packed.split(HD, dim=-1)
    return k_view, v_view, k_view.contiguous(), v_view.contiguous()


def _page_table_and_seqused():
    ptab = torch.arange(NUM_PAGES, device="cuda", dtype=torch.int32).view(
        B, PAGES_PER_REQ
    )
    seqused = torch.full((B,), SEQ_LEN, device="cuda", dtype=torch.int32)
    return ptab, seqused


def _indices(total_q, hkv):
    idx = torch.sort(
        torch.randperm(PAGES_PER_REQ, device="cuda")[:TOPK].to(torch.int32)
    )[0]
    return idx.view(1, 1, TOPK).expand(hkv, total_q, TOPK).contiguous()


@pytest.mark.parametrize("hkv", [1, 2])
@pytest.mark.parametrize("layout", ["HND", "NHD"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_packed_kv_decode_matches_contiguous(layout, dtype, hkv):
    _skip_if_unsupported()
    from flashinfer.msa_ops import msa_sparse_decode_attention

    torch.manual_seed(0)
    k_view, v_view, k_cont, v_cont = _split_cache(dtype, layout, hkv)
    ptab, seqused = _page_table_and_seqused()
    q = torch.randn(B, HQ, HD, device="cuda", dtype=torch.bfloat16)
    idx = _indices(B, hkv)

    ref = msa_sparse_decode_attention(
        q, k_cont, v_cont, idx, page_table=ptab, seqused_k=seqused, seqlen_q=1
    )
    out = msa_sparse_decode_attention(
        q, k_view, v_view, idx, page_table=ptab, seqused_k=seqused, seqlen_q=1
    )
    torch.testing.assert_close(out, ref, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("hkv", [1, 2])
@pytest.mark.parametrize("layout", ["HND", "NHD"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_packed_kv_prefill_matches_contiguous(layout, dtype, hkv):
    _skip_if_unsupported()
    from flashinfer.msa_ops import msa_sparse_attention

    torch.manual_seed(0)
    k_view, v_view, k_cont, v_cont = _split_cache(dtype, layout, hkv)
    ptab, seqused = _page_table_and_seqused()
    total_q = 200
    q = torch.randn(total_q, HQ, HD, device="cuda", dtype=torch.bfloat16)
    cu_q = torch.tensor([0, 50, 120, 160, total_q], device="cuda", dtype=torch.int32)
    idx = _indices(total_q, hkv)

    ref = msa_sparse_attention(
        q, k_cont, v_cont, idx, cu_q, causal=True, page_table=ptab, seqused_k=seqused
    )
    out = msa_sparse_attention(
        q, k_view, v_view, idx, cu_q, causal=True, page_table=ptab, seqused_k=seqused
    )
    torch.testing.assert_close(out, ref, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("capability", "expected"),
    [((10, 0), False), ((10, 3), False), ((12, 0), True), ((12, 1), True)],
)
def test_packed_kv_capability_is_architecture_aware(monkeypatch, capability, expected):
    import flashinfer.msa_ops as msa_ops

    assert msa_ops.SUPPORTS_PACKED_KV is True
    monkeypatch.setattr(msa_ops, "get_compute_capability", lambda _device: capability)
    assert msa_ops.supports_packed_kv("cuda:0") is expected
    assert msa_ops.supports_packed_kv("cpu") is False


def test_unrelated_strided_views_rejected():
    """Strided K/V that are not split from one packed cache must raise, since
    the kernels would otherwise read them at the wrong addresses."""
    _skip_if_unsupported()
    from flashinfer.msa_ops import msa_sparse_decode_attention

    torch.manual_seed(0)
    ptab, seqused = _page_table_and_seqused()
    q = torch.randn(B, HQ, HD, device="cuda", dtype=torch.bfloat16)
    idx = _indices(B, 2)
    k_bad = torch.randn(NUM_PAGES, 2, 128, 2 * HD, device="cuda", dtype=torch.bfloat16)[
        :, :, :, :HD
    ]
    v_bad = torch.randn(NUM_PAGES, 2, 128, 2 * HD, device="cuda", dtype=torch.bfloat16)[
        :, :, :, HD:
    ]
    with pytest.raises(ValueError):
        msa_sparse_decode_attention(
            q, k_bad, v_bad, idx, page_table=ptab, seqused_k=seqused, seqlen_q=1
        )


@pytest.mark.parametrize(
    "scale", [0, -0.125, float("inf"), float("-inf"), float("nan"), torch.tensor(0.1)]
)
def test_packed_fp8_host_scale_rejected_before_cuda_work(monkeypatch, scale):
    import flashinfer.msa_ops.sparse_decode as decode
    import flashinfer.jit.blackwell_msa as jit_msa

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "invalid scale must not be converted from a Tensor or launch work"
        )

    monkeypatch.setattr(decode, "is_blackwell_msa_device", lambda device: True)
    monkeypatch.setattr(torch.Tensor, "__float__", forbidden)
    monkeypatch.setattr(jit_msa, "load_msa_decode_metadata_module", forbidden)
    packed = torch.empty(1, 1, 128, 256, dtype=torch.float8_e4m3fn)
    error = TypeError if isinstance(scale, torch.Tensor) else ValueError
    with pytest.raises(error, match="softmax_scale"):
        decode.msa_sparse_decode_attention(
            torch.empty(4, 16, 128, dtype=torch.bfloat16),
            packed[..., :128],
            packed[..., 128:],
            torch.empty(1, 4, 16, dtype=torch.int32),
            softmax_scale=scale,
        )


def test_packed_fp8_dispatch_reuses_existing_msa_interface(monkeypatch):
    import flashinfer.msa_ops.sparse_decode as decode
    from flashinfer.msa_ops import _blackwell_sm100 as backend

    packed = torch.empty(1, 1, 128, 256, dtype=torch.float8_e4m3fn)
    q = torch.empty(4, 16, 128, dtype=torch.bfloat16)
    indices = torch.empty(1, 4, 16, dtype=torch.int32)
    workspace = object()

    def route(actual_q, k, v, actual_indices, **kwargs):
        assert actual_q is q and actual_indices is indices
        assert k.stride(-2) == v.stride(-2) == 256
        assert kwargs["seqlen_q"] == 4
        assert kwargs["workspace"] is workspace
        return q

    monkeypatch.setattr(decode, "is_blackwell_msa_device", lambda device: True)
    monkeypatch.setattr(backend, "_run_packed_fp8_decode", route)
    assert (
        decode.msa_sparse_decode_attention(
            q,
            packed[..., :128],
            packed[..., 128:],
            indices,
            seqlen_q=4,
            workspace=workspace,
        )
        is q
    )


def test_packed_kv_detection_memoized():
    """The memoized detection must match a fresh computation and hold no
    tensors (a cached view would pin the KV cache's storage)."""
    _skip_if_unsupported()
    from flashinfer.msa_ops import _common

    k, v, _, _ = _split_cache(torch.bfloat16, "HND", 2)
    _common._PACKED_GEO_CACHE.clear()
    miss, miss_rank = _common._packed_kv_view(k, v, HD)
    assert len(_common._PACKED_GEO_CACHE) == 1
    hit, hit_rank = _common._packed_kv_view(k, v, HD)
    assert hit_rank == miss_rank
    assert hit.shape == miss.shape
    assert hit.stride() == miss.stride()
    assert hit.data_ptr() == miss.data_ptr()
    for cached in _common._PACKED_GEO_CACHE.values():
        assert not any(torch.is_tensor(x) for x in cached)
