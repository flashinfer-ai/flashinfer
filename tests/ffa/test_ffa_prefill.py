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

Tests for flashinfer.ffa_kernels - MagiAttention FFA single-GPU integration.

Run: pytest tests/ffa/test_ffa_prefill.py -v
Requires: magi_attention installed + CUDA GPU
"""

import math

import pytest
import torch


def _has_magi():
    try:
        import magi_attention  # noqa: F401

        return True
    except ImportError:
        return False


requires_magi = pytest.mark.skipif(
    not _has_magi(), reason="magi_attention not installed"
)
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def reference_attention(q, k, v, causal=False, sm_scale=None):
    """Naive PyTorch reference for correctness checking."""
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])

    num_qo_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    group_size = num_qo_heads // num_kv_heads

    Sq, Sk = q.shape[0], k.shape[0]
    q_f = q.float()
    k_f = k.float()
    v_f = v.float()

    if group_size > 1:
        k_f = k_f.repeat_interleave(group_size, dim=1)
        v_f = v_f.repeat_interleave(group_size, dim=1)

    attn = torch.einsum("qhd,khd->hqk", q_f, k_f) * sm_scale

    if causal:
        row_idx = torch.arange(Sq, device=q.device).unsqueeze(1) + (Sk - Sq)
        col_idx = torch.arange(Sk, device=q.device).unsqueeze(0)
        causal_mask = col_idx <= row_idx
        attn = attn.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))

    attn = torch.softmax(attn, dim=-1)
    out = torch.einsum("hqk,khd->qhd", attn, v_f)
    return out.to(q.dtype)


@requires_cuda
@requires_magi
class TestFFACausalPrefill:
    """Test causal_prefill against naive reference."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("qo_len,kv_len", [(128, 128), (64, 256), (512, 512)])
    @pytest.mark.parametrize("num_qo_heads,num_kv_heads", [(32, 32), (32, 8)])
    def test_causal_single(self, dtype, qo_len, kv_len, num_qo_heads, num_kv_heads):
        import flashinfer.ffa_kernels as ffa

        head_dim = 128
        torch.manual_seed(42)
        q = torch.randn(qo_len, num_qo_heads, head_dim, dtype=dtype, device="cuda")
        k = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device="cuda")
        v = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device="cuda")

        out_ffa = ffa.causal_prefill(q, k, v)
        out_ref = reference_attention(q, k, v, causal=True)

        torch.testing.assert_close(
            out_ffa.float(), out_ref.float(), rtol=1e-2, atol=1e-2
        )


@requires_cuda
@requires_magi
class TestFFAFlexPrefill:
    """Test flex_prefill with explicit ranges."""

    def test_full_attention(self):
        import flashinfer.ffa_kernels as ffa

        dtype = torch.float16
        qo_len, kv_len = 128, 128
        num_heads, head_dim = 8, 64
        torch.manual_seed(0)
        q = torch.randn(qo_len, num_heads, head_dim, dtype=dtype, device="cuda")
        k = torch.randn(kv_len, num_heads, head_dim, dtype=dtype, device="cuda")
        v = torch.randn(kv_len, num_heads, head_dim, dtype=dtype, device="cuda")

        qr, kr, atm = ffa.full_ranges(qo_len, kv_len, q.device)
        out = ffa.flex_prefill(q, k, v, qr, kr, atm)
        out_ref = reference_attention(q, k, v, causal=False)

        torch.testing.assert_close(out.float(), out_ref.float(), rtol=1e-2, atol=1e-2)

    def test_varlen_causal(self):
        import flashinfer.ffa_kernels as ffa

        dtype = torch.bfloat16
        num_heads, head_dim = 16, 128

        doc_lens = [64, 128, 32]
        total = sum(doc_lens)
        cu_seqlens = torch.tensor(
            [0] + list(torch.tensor(doc_lens).cumsum(0)), dtype=torch.int32
        )

        torch.manual_seed(7)
        q = torch.randn(total, num_heads, head_dim, dtype=dtype, device="cuda")
        k = torch.randn(total, num_heads, head_dim, dtype=dtype, device="cuda")
        v = torch.randn(total, num_heads, head_dim, dtype=dtype, device="cuda")

        out_ffa = ffa.varlen_causal_prefill(q, k, v, cu_seqlens, cu_seqlens)

        refs = []
        for i in range(len(doc_lens)):
            s = int(cu_seqlens[i])
            e = int(cu_seqlens[i + 1])
            ref_i = reference_attention(q[s:e], k[s:e], v[s:e], causal=True)
            refs.append(ref_i)
        out_ref = torch.cat(refs, dim=0)

        torch.testing.assert_close(
            out_ffa.float(), out_ref.float(), rtol=1e-2, atol=1e-2
        )

    def test_return_lse_shape(self):
        import flashinfer.ffa_kernels as ffa

        dtype = torch.float16
        qo_len, kv_len = 64, 64
        num_heads, head_dim = 8, 64
        q = torch.randn(qo_len, num_heads, head_dim, dtype=dtype, device="cuda")
        k = torch.randn(kv_len, num_heads, head_dim, dtype=dtype, device="cuda")
        v = torch.randn(kv_len, num_heads, head_dim, dtype=dtype, device="cuda")

        out, lse = ffa.causal_prefill(q, k, v, return_lse=True)
        assert out.shape == (qo_len, num_heads, head_dim)
        assert lse.shape == (qo_len, num_heads)
        assert lse.dtype == torch.float32

    def test_mixed_mask_types(self):
        """Two slices: first is causal, second is full."""
        import flashinfer.ffa_kernels as ffa

        dtype = torch.float16
        num_heads, head_dim = 8, 64
        torch.manual_seed(99)

        q = torch.randn(256, num_heads, head_dim, dtype=dtype, device="cuda")
        k = torch.randn(256, num_heads, head_dim, dtype=dtype, device="cuda")
        v = torch.randn(256, num_heads, head_dim, dtype=dtype, device="cuda")

        q_ranges = torch.tensor(
            [[0, 128], [128, 256]], dtype=torch.int32, device="cuda"
        )
        k_ranges = torch.tensor(
            [[0, 128], [128, 256]], dtype=torch.int32, device="cuda"
        )
        attn_type_map = torch.tensor(
            [ffa.FFAMaskType.CAUSAL, ffa.FFAMaskType.FULL],
            dtype=torch.int32,
            device="cuda",
        )

        out = ffa.flex_prefill(q, k, v, q_ranges, k_ranges, attn_type_map)
        assert out.shape == (256, num_heads, head_dim)


class TestFFAOptionalDependency:
    """Tests that do not require CUDA or magi_attention to be installed."""

    def test_missing_magi(self, monkeypatch):
        import flashinfer.ffa_kernels.flex_flash_attn as ffa_impl

        monkeypatch.setattr(ffa_impl, "_magi_available", False)
        with pytest.raises(ImportError, match="magi_attention"):
            ffa_impl.flex_prefill(
                torch.zeros(1, 1, 1),
                torch.zeros(1, 1, 1),
                torch.zeros(1, 1, 1),
                torch.zeros(1, 2, dtype=torch.int32),
                torch.zeros(1, 2, dtype=torch.int32),
            )

    def test_qkv_check_rejects_cpu_tensors(self):
        from flashinfer.ffa_kernels.flex_flash_attn import _check_qkv_shape_for_ffa

        q = torch.zeros(1, 1, 1)
        with pytest.raises(ValueError, match="CUDA"):
            _check_qkv_shape_for_ffa(q, q, q)

    def test_single_range_cache_keys_mask_type(self):
        import flashinfer.ffa_kernels.flex_flash_attn as ffa_impl

        ffa_impl._single_ranges_cache.clear()
        full = ffa_impl._cached_single_ranges(
            4, 8, torch.device("cpu"), ffa_impl.FFAMaskType.FULL
        )
        full_again = ffa_impl._cached_single_ranges(
            4, 8, torch.device("cpu"), ffa_impl.FFAMaskType.FULL
        )
        causal = ffa_impl._cached_single_ranges(
            4, 8, torch.device("cpu"), ffa_impl.FFAMaskType.CAUSAL
        )

        assert full[0] is full_again[0]
        assert int(full[2].item()) == ffa_impl.FFAMaskType.FULL
        assert int(causal[2].item()) == ffa_impl.FFAMaskType.CAUSAL
        assert full[2] is not causal[2]
        ffa_impl._single_ranges_cache.clear()

    def test_varlen_range_cache_keys_tensor_identity(self):
        import flashinfer.ffa_kernels.flex_flash_attn as ffa_impl

        ffa_impl._varlen_ranges_cache.clear()
        cu = torch.tensor([0, 4, 8], dtype=torch.int32)
        cu_replacement = cu.clone()

        q_ranges, _, _ = ffa_impl._cached_varlen_causal_ranges(
            cu, cu, torch.device("cpu")
        )
        q_ranges_replacement, _, _ = ffa_impl._cached_varlen_causal_ranges(
            cu_replacement, cu_replacement, torch.device("cpu")
        )

        assert q_ranges is not q_ranges_replacement
        torch.testing.assert_close(q_ranges, q_ranges_replacement)
        ffa_impl._varlen_ranges_cache.clear()

    def test_varlen_range_cache_invalidates_on_indptr_update(self):
        import flashinfer.ffa_kernels.flex_flash_attn as ffa_impl

        ffa_impl._varlen_ranges_cache.clear()
        cu = torch.tensor([0, 4, 8], dtype=torch.int32)
        q_ranges, _, _ = ffa_impl._cached_varlen_causal_ranges(
            cu, cu, torch.device("cpu")
        )

        cu[1] = 2
        updated_q_ranges, _, _ = ffa_impl._cached_varlen_causal_ranges(
            cu, cu, torch.device("cpu")
        )

        assert q_ranges is not updated_q_ranges
        assert updated_q_ranges.tolist() == [[0, 2], [2, 8]]
        ffa_impl._varlen_ranges_cache.clear()


@requires_cuda
@requires_magi
class TestFFAErrorHandling:
    """Test that bad inputs raise clear errors."""

    def test_shape_mismatch(self):
        import flashinfer.ffa_kernels as ffa

        q = torch.randn(8, 4, 64, device="cuda")
        k = torch.randn(8, 4, 64, device="cuda")
        v = torch.randn(8, 4, 64, device="cuda")
        qr = torch.tensor([[0, 8]], dtype=torch.int32, device="cuda")
        kr_bad = torch.tensor([[0, 8], [0, 4]], dtype=torch.int32, device="cuda")
        with pytest.raises(ValueError, match="same num_ranges"):
            ffa.flex_prefill(q, k, v, qr, kr_bad)

    def test_empty_indptr_rejected(self):
        import flashinfer.ffa_kernels as ffa

        cu = torch.tensor([0], dtype=torch.int32, device="cuda")
        with pytest.raises(ValueError, match="at least one segment"):
            ffa.varlen_causal_ranges(cu, cu, torch.device("cuda"))


@requires_cuda
@requires_magi
class TestBatchPrefillFFAWrapper:
    """plan/run-style wrapper mirroring FlashInfer conventions."""

    def test_causal_via_indptr(self):
        import flashinfer.ffa_kernels as ffa

        dtype = torch.bfloat16
        num_qo_heads, num_kv_heads, head_dim = 32, 8, 128
        doc_lens = [128, 64, 256]
        total = sum(doc_lens)
        cu = torch.tensor(
            [0] + list(torch.tensor(doc_lens).cumsum(0)),
            dtype=torch.int32,
            device="cuda",
        )

        torch.manual_seed(3)
        q = torch.randn(total, num_qo_heads, head_dim, dtype=dtype, device="cuda")
        k = torch.randn(total, num_kv_heads, head_dim, dtype=dtype, device="cuda")
        v = torch.randn(total, num_kv_heads, head_dim, dtype=dtype, device="cuda")

        workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        wrapper = ffa.BatchPrefillFFAWrapper(workspace, kv_layout="NHD")
        wrapper.plan(
            qo_indptr=cu,
            kv_indptr=cu,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            causal=True,
        )

        out = wrapper.run(q, k, v)
        out_ref = ffa.varlen_causal_prefill(q, k, v, cu, cu)

        torch.testing.assert_close(out.float(), out_ref.float(), rtol=1e-2, atol=1e-2)

    def test_native_ranges_mixed_mask(self):
        import flashinfer.ffa_kernels as ffa

        dtype = torch.float16
        num_heads, head_dim = 8, 64
        torch.manual_seed(11)

        q = torch.randn(256, num_heads, head_dim, dtype=dtype, device="cuda")
        k = torch.randn(256, num_heads, head_dim, dtype=dtype, device="cuda")
        v = torch.randn(256, num_heads, head_dim, dtype=dtype, device="cuda")

        q_ranges = torch.tensor(
            [[0, 128], [128, 256]], dtype=torch.int32, device="cuda"
        )
        k_ranges = torch.tensor(
            [[0, 128], [128, 256]], dtype=torch.int32, device="cuda"
        )
        attn_type_map = torch.tensor(
            [ffa.FFAMaskType.CAUSAL, ffa.FFAMaskType.FULL],
            dtype=torch.int32,
            device="cuda",
        )

        workspace = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        wrapper = ffa.BatchPrefillFFAWrapper(workspace)
        wrapper.plan(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            num_qo_heads=num_heads,
            num_kv_heads=num_heads,
            head_dim=head_dim,
        )

        out = wrapper.run(q, k, v)
        out_ref = ffa.flex_prefill(q, k, v, q_ranges, k_ranges, attn_type_map)
        torch.testing.assert_close(out.float(), out_ref.float(), rtol=1e-2, atol=1e-2)

    def test_run_validates_changed_signature_after_cache(self):
        import flashinfer.ffa_kernels as ffa

        dtype = torch.float16
        num_heads, head_dim = 4, 64
        q = torch.randn(32, num_heads, head_dim, dtype=dtype, device="cuda")
        k = torch.randn(32, num_heads, head_dim, dtype=dtype, device="cuda")
        v = torch.randn(32, num_heads, head_dim, dtype=dtype, device="cuda")

        workspace = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        wrapper = ffa.BatchPrefillFFAWrapper(workspace)
        wrapper.plan(
            q_ranges=torch.tensor([[0, 32]], dtype=torch.int32, device="cuda"),
            k_ranges=torch.tensor([[0, 32]], dtype=torch.int32, device="cuda"),
            attn_type_map=torch.tensor(
                [ffa.FFAMaskType.FULL], dtype=torch.int32, device="cuda"
            ),
            num_qo_heads=num_heads,
            num_kv_heads=num_heads,
            head_dim=head_dim,
        )

        wrapper.run(q, k, v)
        q_wrong_heads = torch.randn(32, 8, head_dim, dtype=dtype, device="cuda")
        with pytest.raises(ValueError, match="planned num_qo_heads"):
            wrapper.run(q_wrong_heads, k, v)

    def test_run_fast_matches_run(self):
        import flashinfer.ffa_kernels as ffa

        dtype = torch.float16
        num_heads, head_dim = 4, 64
        q = torch.randn(32, num_heads, head_dim, dtype=dtype, device="cuda")
        k = torch.randn(32, num_heads, head_dim, dtype=dtype, device="cuda")
        v = torch.randn(32, num_heads, head_dim, dtype=dtype, device="cuda")
        q_ranges = torch.tensor([[0, 32]], dtype=torch.int32, device="cuda")
        k_ranges = torch.tensor([[0, 32]], dtype=torch.int32, device="cuda")
        attn_type_map = torch.tensor(
            [ffa.FFAMaskType.CAUSAL], dtype=torch.int32, device="cuda"
        )

        workspace = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        wrapper = ffa.BatchPrefillFFAWrapper(workspace)
        with pytest.raises(RuntimeError, match="run_fast"):
            wrapper.run_fast(q, k, v)
        wrapper.plan(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            num_qo_heads=num_heads,
            num_kv_heads=num_heads,
            head_dim=head_dim,
        )

        out = wrapper.run_fast(q, k, v)
        expected = ffa.flex_prefill(q, k, v, q_ranges, k_ranges, attn_type_map)
        torch.testing.assert_close(out.float(), expected.float(), rtol=1e-2, atol=1e-2)

    def test_run_writes_out_and_lse_buffers(self):
        import flashinfer.ffa_kernels as ffa

        dtype = torch.bfloat16
        num_heads, head_dim = 8, 64
        q = torch.randn(64, num_heads, head_dim, dtype=dtype, device="cuda")
        k = torch.randn(64, num_heads, head_dim, dtype=dtype, device="cuda")
        v = torch.randn(64, num_heads, head_dim, dtype=dtype, device="cuda")
        q_ranges = torch.tensor([[0, 64]], dtype=torch.int32, device="cuda")
        k_ranges = torch.tensor([[0, 64]], dtype=torch.int32, device="cuda")
        attn_type_map = torch.tensor(
            [ffa.FFAMaskType.CAUSAL], dtype=torch.int32, device="cuda"
        )

        workspace = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        wrapper = ffa.BatchPrefillFFAWrapper(workspace)
        wrapper.plan(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            num_qo_heads=num_heads,
            num_kv_heads=num_heads,
            head_dim=head_dim,
        )

        out = torch.empty_like(q)
        lse = torch.empty((q.shape[0], q.shape[1]), dtype=torch.float32, device="cuda")
        actual_out, actual_lse = wrapper.run(q, k, v, out=out, lse=lse, return_lse=True)
        expected_out, expected_lse = ffa.flex_prefill(
            q, k, v, q_ranges, k_ranges, attn_type_map, return_lse=True
        )

        assert actual_out is out
        assert actual_lse is lse
        torch.testing.assert_close(
            out.float(), expected_out.float(), rtol=1e-2, atol=1e-2
        )
        torch.testing.assert_close(lse, expected_lse, rtol=1e-2, atol=1e-2)

    def test_hnd_layout_not_transposed_twice(self):
        import flashinfer.ffa_kernels as ffa

        dtype = torch.float16
        num_heads, head_dim = 4, 64
        q = torch.randn(32, num_heads, head_dim, dtype=dtype, device="cuda")
        k_nhd = torch.randn(32, num_heads, head_dim, dtype=dtype, device="cuda")
        v_nhd = torch.randn(32, num_heads, head_dim, dtype=dtype, device="cuda")
        k_hnd = k_nhd.transpose(0, 1).contiguous()
        v_hnd = v_nhd.transpose(0, 1).contiguous()
        q_ranges = torch.tensor([[0, 32]], dtype=torch.int32, device="cuda")
        k_ranges = torch.tensor([[0, 32]], dtype=torch.int32, device="cuda")
        attn_type_map = torch.tensor(
            [ffa.FFAMaskType.FULL], dtype=torch.int32, device="cuda"
        )

        workspace = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        wrapper = ffa.BatchPrefillFFAWrapper(workspace, kv_layout="HND")
        wrapper.plan(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            num_qo_heads=num_heads,
            num_kv_heads=num_heads,
            head_dim=head_dim,
        )

        out = wrapper.run(q, k_hnd, v_hnd)
        expected = ffa.flex_prefill(q, k_nhd, v_nhd, q_ranges, k_ranges, attn_type_map)
        torch.testing.assert_close(out.float(), expected.float(), rtol=1e-2, atol=1e-2)

    def test_plan_replan_across_layers(self):
        """Simulate multi-layer loop: plan once, run many times."""
        import flashinfer.ffa_kernels as ffa

        dtype = torch.bfloat16
        num_heads, head_dim = 8, 64
        cu = torch.tensor([0, 64, 128], dtype=torch.int32, device="cuda")

        workspace = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        wrapper = ffa.BatchPrefillFFAWrapper(workspace)
        wrapper.plan(
            qo_indptr=cu,
            kv_indptr=cu,
            num_qo_heads=num_heads,
            num_kv_heads=num_heads,
            head_dim=head_dim,
            causal=True,
        )

        torch.manual_seed(0)
        for _ in range(3):
            q = torch.randn(128, num_heads, head_dim, dtype=dtype, device="cuda")
            k = torch.randn(128, num_heads, head_dim, dtype=dtype, device="cuda")
            v = torch.randn(128, num_heads, head_dim, dtype=dtype, device="cuda")
            out = wrapper.run(q, k, v)
            assert out.shape == (128, num_heads, head_dim)

    def test_run_before_plan_raises(self):
        import flashinfer.ffa_kernels as ffa

        workspace = torch.empty(1024, dtype=torch.uint8, device="cuda")
        wrapper = ffa.BatchPrefillFFAWrapper(workspace)
        q = torch.randn(4, 2, 64, dtype=torch.float16, device="cuda")
        with pytest.raises(RuntimeError, match="plan"):
            wrapper.run(q, q, q)

    def test_both_styles_rejected(self):
        import flashinfer.ffa_kernels as ffa

        workspace = torch.empty(1024, dtype=torch.uint8, device="cuda")
        wrapper = ffa.BatchPrefillFFAWrapper(workspace)
        cu = torch.tensor([0, 4], dtype=torch.int32, device="cuda")
        qr = torch.tensor([[0, 4]], dtype=torch.int32, device="cuda")
        with pytest.raises(ValueError, match="either"):
            wrapper.plan(
                qo_indptr=cu,
                kv_indptr=cu,
                q_ranges=qr,
                k_ranges=qr,
                num_qo_heads=2,
                num_kv_heads=2,
                head_dim=64,
            )


@requires_cuda
@requires_magi
class TestBackendFFAIntegration:
    """Verify ``backend='ffa'`` wiring in top-level FlashInfer APIs."""

    def test_single_prefill_backend_ffa_causal(self):
        import flashinfer

        dtype = torch.bfloat16
        qo_len, kv_len = 128, 128
        num_qo_heads, num_kv_heads, head_dim = 32, 8, 128
        torch.manual_seed(5)
        q = torch.randn(qo_len, num_qo_heads, head_dim, dtype=dtype, device="cuda")
        k = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device="cuda")
        v = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device="cuda")

        out_ffa = flashinfer.single_prefill_with_kv_cache(
            q, k, v, causal=True, backend="ffa"
        )
        out_ref = reference_attention(q, k, v, causal=True)
        torch.testing.assert_close(
            out_ffa.float(), out_ref.float(), rtol=1e-2, atol=1e-2
        )

    def test_single_prefill_backend_ffa_rejects_custom_mask(self):
        import flashinfer

        q = torch.randn(8, 4, 64, dtype=torch.float16, device="cuda")
        k = torch.randn(8, 4, 64, dtype=torch.float16, device="cuda")
        v = torch.randn(8, 4, 64, dtype=torch.float16, device="cuda")
        mask = torch.ones(8, 8, dtype=torch.bool, device="cuda")
        with pytest.raises(ValueError, match="custom_mask"):
            flashinfer.single_prefill_with_kv_cache(
                q, k, v, custom_mask=mask, backend="ffa"
            )

    def test_single_prefill_backend_ffa_rejects_fp8_scale(self):
        import flashinfer

        q = torch.randn(8, 4, 64, dtype=torch.float16, device="cuda")
        k = torch.randn(8, 4, 64, dtype=torch.float16, device="cuda")
        v = torch.randn(8, 4, 64, dtype=torch.float16, device="cuda")
        scale = torch.ones(4, dtype=torch.float32, device="cuda")

        with pytest.raises(ValueError, match="scale_q"):
            flashinfer.single_prefill_with_kv_cache(
                q, k, v, scale_q=scale, backend="ffa"
            )

    def test_single_prefill_backend_ffa_rejects_float32(self):
        import flashinfer

        q = torch.randn(8, 4, 64, dtype=torch.float32, device="cuda")
        k = torch.randn(8, 4, 64, dtype=torch.float32, device="cuda")
        v = torch.randn(8, 4, 64, dtype=torch.float32, device="cuda")

        with pytest.raises(ValueError, match="float16 and bfloat16"):
            flashinfer.single_prefill_with_kv_cache(q, k, v, backend="ffa")

    def test_single_prefill_backend_ffa_rejects_value_head_dim_mismatch(self):
        import flashinfer

        q = torch.randn(8, 4, 64, dtype=torch.float16, device="cuda")
        k = torch.randn(8, 4, 64, dtype=torch.float16, device="cuda")
        v = torch.randn(8, 4, 32, dtype=torch.float16, device="cuda")

        with pytest.raises(ValueError, match="same head_dim"):
            flashinfer.single_prefill_with_kv_cache(q, k, v, backend="ffa")

    def test_single_prefill_backend_ffa_rejects_invalid_kv_layout(self):
        import flashinfer

        q = torch.randn(8, 4, 64, dtype=torch.float16, device="cuda")
        k = torch.randn(8, 4, 64, dtype=torch.float16, device="cuda")
        v = torch.randn(8, 4, 64, dtype=torch.float16, device="cuda")

        with pytest.raises((KeyError, ValueError), match="kv_layout"):
            flashinfer.single_prefill_with_kv_cache(
                q, k, v, backend="ffa", kv_layout="bad"
            )

    def test_batch_ragged_wrapper_backend_ffa(self):
        import flashinfer

        dtype = torch.bfloat16
        num_qo_heads, num_kv_heads, head_dim = 32, 8, 128
        doc_lens = [128, 64]
        total = sum(doc_lens)
        cu = torch.tensor(
            [0] + list(torch.tensor(doc_lens).cumsum(0)),
            dtype=torch.int32,
            device="cuda",
        )

        torch.manual_seed(13)
        q = torch.randn(total, num_qo_heads, head_dim, dtype=dtype, device="cuda")
        k = torch.randn(total, num_kv_heads, head_dim, dtype=dtype, device="cuda")
        v = torch.randn(total, num_kv_heads, head_dim, dtype=dtype, device="cuda")

        workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            workspace, backend="ffa"
        )
        wrapper.plan(
            cu,
            cu,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            causal=True,
            q_data_type=dtype,
        )
        out = wrapper.run(q, k, v)

        # Reference: per-document causal attention.
        refs = []
        for i in range(len(doc_lens)):
            s, e = int(cu[i]), int(cu[i + 1])
            refs.append(reference_attention(q[s:e], k[s:e], v[s:e], causal=True))
        out_ref = torch.cat(refs, dim=0)
        torch.testing.assert_close(out.float(), out_ref.float(), rtol=1e-2, atol=1e-2)

    def test_batch_ragged_wrapper_backend_ffa_hnd_layout(self):
        import flashinfer
        import flashinfer.ffa_kernels as ffa

        dtype = torch.float16
        num_qo_heads, num_kv_heads, head_dim = 8, 4, 64
        doc_lens = [32, 16]
        total = sum(doc_lens)
        cu = torch.tensor(
            [0] + list(torch.tensor(doc_lens).cumsum(0)),
            dtype=torch.int32,
            device="cuda",
        )

        q = torch.randn(total, num_qo_heads, head_dim, dtype=dtype, device="cuda")
        k_nhd = torch.randn(total, num_kv_heads, head_dim, dtype=dtype, device="cuda")
        v_nhd = torch.randn(total, num_kv_heads, head_dim, dtype=dtype, device="cuda")
        k_hnd = k_nhd.transpose(0, 1).contiguous()
        v_hnd = v_nhd.transpose(0, 1).contiguous()

        workspace = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            workspace,
            kv_layout="HND",
            backend="ffa",
        )
        wrapper.plan(
            cu,
            cu,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            causal=True,
            q_data_type=dtype,
            kv_data_type=dtype,
        )

        out = wrapper.run(q, k_hnd, v_hnd)
        expected = ffa.varlen_causal_prefill(q, k_nhd, v_nhd, cu, cu)
        torch.testing.assert_close(out.float(), expected.float(), rtol=1e-2, atol=1e-2)

    def test_batch_ragged_wrapper_backend_ffa_fast_path_validates_inputs(self):
        import flashinfer

        dtype = torch.float16
        num_qo_heads, num_kv_heads, head_dim = 8, 4, 64
        cu = torch.tensor([0, 16, 32], dtype=torch.int32, device="cuda")
        q = torch.randn(32, num_qo_heads, head_dim, dtype=dtype, device="cuda")
        k = torch.randn(32, num_kv_heads, head_dim, dtype=dtype, device="cuda")
        v = torch.randn(32, num_kv_heads, head_dim, dtype=dtype, device="cuda")

        workspace = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            workspace, backend="ffa"
        )
        wrapper.plan(
            cu,
            cu,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            causal=True,
            q_data_type=dtype,
            kv_data_type=dtype,
        )

        wrapper.run(q, k, v)
        with pytest.raises(ValueError, match="planned num_qo_heads"):
            wrapper.run(q[:, :4].contiguous(), k, v)
        with pytest.raises(ValueError, match="exceeds tensor length"):
            wrapper.run(q, k[:16], v[:16])
        q_noncontiguous = torch.randn(
            64, num_qo_heads, head_dim, dtype=dtype, device="cuda"
        )[::2]
        with pytest.raises(ValueError, match="contiguous"):
            wrapper.run(q_noncontiguous, k, v)
        with pytest.raises(ValueError, match="q_scale"):
            wrapper.run(q, k, v, q_scale=2.0)

    def test_batch_ragged_wrapper_backend_ffa_run_before_plan_raises(self):
        import flashinfer

        workspace = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            workspace, backend="ffa"
        )
        q = torch.randn(8, 4, 64, dtype=torch.float16, device="cuda")

        with pytest.raises(RuntimeError, match="plan"):
            wrapper.run(q, q, q)

    def test_batch_ragged_wrapper_backend_ffa_rejects_o_dtype(self):
        import flashinfer

        workspace = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            workspace, backend="ffa"
        )
        cu = torch.tensor([0, 16], dtype=torch.int32, device="cuda")

        with pytest.raises(ValueError, match="o_data_type"):
            wrapper.plan(
                cu,
                cu,
                num_qo_heads=8,
                num_kv_heads=8,
                head_dim_qk=64,
                causal=True,
                q_data_type=torch.float16,
                o_data_type=torch.float32,
            )

    def test_batch_ragged_wrapper_backend_ffa_rejects_float32(self):
        import flashinfer

        workspace = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            workspace, backend="ffa"
        )
        cu = torch.tensor([0, 16], dtype=torch.int32, device="cuda")

        with pytest.raises(ValueError, match="float16 and bfloat16"):
            wrapper.plan(
                cu,
                cu,
                num_qo_heads=8,
                num_kv_heads=8,
                head_dim_qk=64,
                causal=True,
                q_data_type=torch.float32,
                kv_data_type=torch.float32,
            )

    def test_batch_ragged_wrapper_backend_ffa_rejects_rope(self):
        import flashinfer

        workspace = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            workspace, backend="ffa"
        )
        cu = torch.tensor([0, 16], dtype=torch.int32, device="cuda")
        with pytest.raises(ValueError, match="pos_encoding_mode"):
            wrapper.plan(
                cu,
                cu,
                num_qo_heads=8,
                num_kv_heads=8,
                head_dim_qk=64,
                causal=True,
                pos_encoding_mode="ROPE_LLAMA",
                q_data_type=torch.float16,
            )

    def test_batch_ragged_wrapper_backend_ffa_rejects_cuda_graph_at_init(self):
        import flashinfer

        workspace = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        with pytest.raises(ValueError, match="CUDA Graph"):
            flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
                workspace, backend="ffa", use_cuda_graph=True
            )
