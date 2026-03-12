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

import pytest
import torch
import torch.nn.functional as F

from flashinfer.utils import (
    is_sm100a_supported,
    is_sm110a_supported,
)

try:
    from flashinfer.gdn_kernels.blackwell_prefill.gdn import (
        chunk_gated_delta_rule,
    )

    GDN_BLACKWELL_PREFILL_CUTEDSL = True
except ImportError:
    GDN_BLACKWELL_PREFILL_CUTEDSL = False


testtype = torch.float16

oatol = 1e-2 if testtype is torch.bfloat16 else 1e-3
ortol = 1e-2 if testtype is torch.bfloat16 else 1e-3
satol = 5e-3 if testtype is torch.bfloat16 else 1e-3
srtol = 1e-3 if testtype is torch.bfloat16 else 1e-4


def recurrent_gated_delta_rule_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
):
    """Reference implementation of gated delta rule (recurrent version)."""
    q, k, v, beta, g = map(
        lambda x: x.transpose(1, 2).contiguous().to(torch.float32), [q, k, v, beta, g]
    )
    B, H, T, K, V = *k.shape, v.shape[-1]
    o = torch.zeros(B, H, T, V).to(v)
    h = torch.zeros(B, H, K, V).to(v)
    if initial_state is not None:
        h = initial_state
    if scale is None:
        scale = 1 / (q.shape[-1] ** 0.5)
    q = q * scale
    for i in range(T):
        b_q = q[:, :, i]
        b_k = k[:, :, i]
        b_v = v[:, :, i].clone()
        h = h.clone() * g[:, :, i].exp()[..., None, None]
        b_beta = beta[:, :, i]
        b_v = b_v - (h.clone() * b_k[..., None]).sum(-2)
        b_v = b_v * b_beta[..., None]
        h = h.clone() + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)
        o[:, :, i] = torch.einsum("bhd,bhdm->bhm", b_q, h)
    if not output_final_state:
        h = None
    o = o.transpose(1, 2).contiguous()
    return o, h


def _skip_if_gdn_blackwell_prefill_not_available():
    """Skip test if GDN Blackwell prefill CuTeDSL kernels not available or not Blackwell (SM100+) architecture."""
    if not GDN_BLACKWELL_PREFILL_CUTEDSL:
        pytest.skip("GDN Blackwell prefill CuTeDSL kernels not available")

    if not is_sm100a_supported(torch.device("cuda")) and not is_sm110a_supported(
        torch.device("cuda")
    ):
        pytest.skip("Only SM100A and SM110A are supported on this device")


@pytest.fixture(autouse=True)
def cuda_sync_and_cleanup():
    """Ensure CUDA operations are synchronized before and after each test."""
    torch.cuda.synchronize()
    yield
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


@pytest.fixture
def set_seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)


class TestGDNFixedLength:
    """Tests for fixed-length sequences."""

    @pytest.mark.parametrize(
        "H,D,T,B,dtype",
        [
            (16, 128, 256, 4, testtype),
            (1, 128, 128, 2, testtype),
            (16, 128, 256, 1, testtype),
            (3, 128, 131, 177, testtype),
        ],
    )
    def test_fixlen_output(self, set_seed, H, D, T, B, dtype):
        """Test fixed-length GDN output correctness."""
        _skip_if_gdn_blackwell_prefill_not_available()

        device = "cuda"

        q = torch.randn((B, T, H, D), dtype=dtype)
        k = F.normalize(torch.randn(B, T, H, D, dtype=torch.float32), p=2, dim=-1).to(
            dtype
        )
        v = torch.randn((B, T, H, D), dtype=dtype)
        g = F.logsigmoid(torch.rand(1, T * B, H, dtype=torch.float32))
        beta = torch.rand(1, T * B, H, dtype=torch.float32).sigmoid()
        h0 = torch.randn((B, H, D, D), dtype=torch.float)

        q, k, v, beta, g, h0 = map(lambda x: x.to(device), (q, k, v, beta, g, h0))

        state_output = torch.zeros_like(h0, dtype=torch.float)

        o, state_output = chunk_gated_delta_rule(
            q.clone(),
            k.clone(),
            v.clone(),
            g.clone(),
            beta.clone(),
            None,
            h0.clone(),
            True,
            None,
            False,
            None,
            state_output,
        )
        torch.cuda.synchronize()

        ref = []
        ref_ht = []
        h0_v_major = h0.transpose(-1, -2).contiguous()
        for i in range(B):
            ref_i, ref_ht_i = recurrent_gated_delta_rule_ref(
                q=q[i].unsqueeze(0).clone(),
                k=k[i].unsqueeze(0).clone(),
                v=v[i].unsqueeze(0).clone(),
                beta=beta[:, (i * T) : ((i + 1) * T)].clone(),
                g=g[:, (i * T) : ((i + 1) * T)].clone(),
                initial_state=h0_v_major[i].clone(),
                output_final_state=True,
            )
            if i == 0:
                ref = ref_i
                ref_ht = ref_ht_i
            else:
                ref = torch.cat((ref, ref_i), dim=0).contiguous()
                ref_ht = torch.cat((ref_ht, ref_ht_i), dim=0).contiguous()

        torch.cuda.synchronize()

        torch.testing.assert_close(ref, o.to(torch.float), atol=oatol, rtol=ortol)

    @pytest.mark.parametrize(
        "H,D,T,B,dtype",
        [
            (16, 128, 256, 4, testtype),
            (1, 128, 128, 2, testtype),
            (16, 128, 256, 1, testtype),
            (3, 128, 131, 177, testtype),
        ],
    )
    def test_fixlen_state(self, set_seed, H, D, T, B, dtype):
        """Test fixed-length GDN state output correctness."""

        _skip_if_gdn_blackwell_prefill_not_available()

        device = "cuda"

        q = torch.randn((B, T, H, D), dtype=dtype)
        k = F.normalize(torch.randn(B, T, H, D, dtype=torch.float32), p=2, dim=-1).to(
            dtype
        )
        v = torch.randn((B, T, H, D), dtype=dtype)
        g = F.logsigmoid(torch.rand(1, T * B, H, dtype=torch.float32))
        beta = torch.rand(1, T * B, H, dtype=torch.float32).sigmoid()
        h0 = torch.randn((B, H, D, D), dtype=torch.float)

        q, k, v, beta, g, h0 = map(lambda x: x.to(device), (q, k, v, beta, g, h0))

        state_output = torch.zeros_like(h0, dtype=torch.float)

        o, state_output = chunk_gated_delta_rule(
            q.clone(),
            k.clone(),
            v.clone(),
            g.clone(),
            beta.clone(),
            None,
            h0.clone(),
            True,
            None,
            False,
            None,
            state_output,
        )
        torch.cuda.synchronize()

        ref_ht = []
        h0_v_major = h0.transpose(-1, -2).contiguous()
        for i in range(B):
            _, ref_ht_i = recurrent_gated_delta_rule_ref(
                q=q[i].unsqueeze(0).clone(),
                k=k[i].unsqueeze(0).clone(),
                v=v[i].unsqueeze(0).clone(),
                beta=beta[:, (i * T) : ((i + 1) * T)].clone(),
                g=g[:, (i * T) : ((i + 1) * T)].clone(),
                initial_state=h0_v_major[i].clone(),
                output_final_state=True,
            )
            if i == 0:
                ref_ht = ref_ht_i
            else:
                ref_ht = torch.cat((ref_ht, ref_ht_i), dim=0).contiguous()

        ref_ht = torch.transpose(ref_ht, -1, -2).contiguous()
        torch.cuda.synchronize()

        torch.testing.assert_close(
            ref_ht, state_output.to(torch.float), atol=satol, rtol=srtol
        )


class TestGDNVariableLength:
    """Tests for variable-length sequences."""

    @pytest.mark.parametrize(
        "H,D,cu_seqlens,dtype",
        [
            (16, 128, [0, 256, 500, 1000, 1013], testtype),
            (16, 128, [0, 13, 77], testtype),
        ],
    )
    def test_varlen_output(self, set_seed, H, D, cu_seqlens, dtype):
        """Test variable-length GDN output correctness."""

        _skip_if_gdn_blackwell_prefill_not_available()

        device = "cuda"

        cu_seqlens_tensor = torch.tensor(cu_seqlens, device=device)
        T = cu_seqlens[-1]
        N = len(cu_seqlens) - 1

        q = torch.randn((1, T, H, D), dtype=dtype)
        k = F.normalize(torch.randn(1, T, H, D, dtype=torch.float32), p=2, dim=-1).to(
            dtype
        )
        v = torch.randn((1, T, H, D), dtype=dtype)
        g = F.logsigmoid(torch.rand(1, T, H, dtype=torch.float32))
        beta = torch.rand(1, T, H, dtype=torch.float32).sigmoid()
        h0 = torch.randn((N, H, D, D), dtype=torch.float)

        q, k, v, beta, g, h0 = map(lambda x: x.to(device), (q, k, v, beta, g, h0))

        state_output = torch.zeros_like(h0, dtype=torch.float)

        o, state_output = chunk_gated_delta_rule(
            q.clone(),
            k.clone(),
            v.clone(),
            g.clone(),
            beta.clone(),
            None,
            h0.clone(),
            True,
            cu_seqlens_tensor.clone(),
            False,
            None,
            state_output,
        )
        torch.cuda.synchronize()

        ref = []
        h0_v_major = h0.transpose(-1, -2)
        for i in range(N):
            ref_i, _ = recurrent_gated_delta_rule_ref(
                q=q[:, cu_seqlens[i] : cu_seqlens[i + 1]],
                k=k[:, cu_seqlens[i] : cu_seqlens[i + 1]],
                v=v[:, cu_seqlens[i] : cu_seqlens[i + 1]],
                beta=beta[:, cu_seqlens[i] : cu_seqlens[i + 1]],
                g=g[:, cu_seqlens[i] : cu_seqlens[i + 1]],
                initial_state=h0_v_major[i],
                output_final_state=True,
            )
            ref.append(ref_i)
        ref = torch.cat(ref, 1)
        torch.cuda.synchronize()

        torch.testing.assert_close(ref, o.to(torch.float), atol=oatol, rtol=ortol)

    @pytest.mark.parametrize(
        "H,D,cu_seqlens,dtype",
        [
            (16, 128, [0, 256, 500, 1000, 1013], testtype),
            (16, 128, [0, 13, 77], testtype),
        ],
    )
    def test_varlen_state(self, set_seed, H, D, cu_seqlens, dtype):
        """Test variable-length GDN state output correctness."""

        _skip_if_gdn_blackwell_prefill_not_available()

        device = "cuda"

        cu_seqlens_tensor = torch.tensor(cu_seqlens, device=device)
        T = cu_seqlens[-1]
        N = len(cu_seqlens) - 1

        q = torch.randn((1, T, H, D), dtype=dtype)
        k = F.normalize(torch.randn(1, T, H, D, dtype=torch.float32), p=2, dim=-1).to(
            dtype
        )
        v = torch.randn((1, T, H, D), dtype=dtype)
        g = F.logsigmoid(torch.rand(1, T, H, dtype=torch.float32))
        beta = torch.rand(1, T, H, dtype=torch.float32).sigmoid()
        h0 = torch.randn((N, H, D, D), dtype=torch.float)

        q, k, v, beta, g, h0 = map(lambda x: x.to(device), (q, k, v, beta, g, h0))

        state_output = torch.zeros_like(h0, dtype=torch.float)

        o, state_output = chunk_gated_delta_rule(
            q.clone(),
            k.clone(),
            v.clone(),
            g.clone(),
            beta.clone(),
            None,
            h0.clone(),
            True,
            cu_seqlens_tensor.clone(),
            False,
            None,
            state_output,
        )
        torch.cuda.synchronize()

        ref_ht = []
        h0_v_major = h0.transpose(-1, -2)
        for i in range(N):
            _, ref_ht_i = recurrent_gated_delta_rule_ref(
                q=q[:, cu_seqlens[i] : cu_seqlens[i + 1]],
                k=k[:, cu_seqlens[i] : cu_seqlens[i + 1]],
                v=v[:, cu_seqlens[i] : cu_seqlens[i + 1]],
                beta=beta[:, cu_seqlens[i] : cu_seqlens[i + 1]],
                g=g[:, cu_seqlens[i] : cu_seqlens[i + 1]],
                initial_state=h0_v_major[i],
                output_final_state=True,
            )
            ref_ht.append(ref_ht_i)
        ref_ht = torch.cat(ref_ht, 0)
        ref_ht = torch.transpose(ref_ht, -1, -2)
        torch.cuda.synchronize()

        torch.testing.assert_close(
            ref_ht, state_output.to(torch.float), atol=satol, rtol=srtol
        )


class TestGDNFixedLengthGVA:
    """Tests for fixed-length sequences with Grouped Value Attention (H_QK != H_V)."""

    @pytest.mark.parametrize(
        "H_QK,H_V,D,T,B,dtype",
        [
            (16, 32, 128, 555, 1, testtype),
            (8, 32, 128, 256, 2, testtype),
            (16, 64, 128, 128, 1, testtype),
        ],
    )
    def test_fixlen_gva_output(self, set_seed, H_QK, H_V, D, T, B, dtype):
        """Test fixed-length GVA output correctness."""

        _skip_if_gdn_blackwell_prefill_not_available()

        device = "cuda"

        q = torch.randn((B, T, H_QK, D), dtype=dtype)
        k = F.normalize(
            torch.randn(B, T, H_QK, D, dtype=torch.float32), p=2, dim=-1
        ).to(dtype)
        v = torch.randn((B, T, H_V, D), dtype=dtype)
        g = F.logsigmoid(torch.rand(1, T * B, H_V, dtype=torch.float32))
        beta = torch.rand(1, T * B, H_V, dtype=torch.float32).sigmoid()
        h0 = torch.randn((B, H_V, D, D), dtype=torch.float)

        q, k, v, beta, g, h0 = map(lambda x: x.to(device), (q, k, v, beta, g, h0))

        state_output = torch.zeros_like(h0, dtype=torch.float)

        o, state_output = chunk_gated_delta_rule(
            q.clone(),
            k.clone(),
            v.clone(),
            g.clone(),
            beta.clone(),
            None,
            h0.clone(),
            True,
            None,
            False,
            None,
            state_output,
        )
        torch.cuda.synchronize()

        repeat_factor = H_V // H_QK
        q = q.repeat_interleave(repeat_factor, dim=2)
        k = k.repeat_interleave(repeat_factor, dim=2)

        ref = []
        h0_v_major = h0.transpose(-1, -2).contiguous()
        for i in range(B):
            ref_i, _ = recurrent_gated_delta_rule_ref(
                q=q[i].unsqueeze(0).clone(),
                k=k[i].unsqueeze(0).clone(),
                v=v[i].unsqueeze(0).clone(),
                beta=beta[:, (i * T) : ((i + 1) * T)].clone(),
                g=g[:, (i * T) : ((i + 1) * T)].clone(),
                initial_state=h0_v_major[i].clone(),
                output_final_state=True,
            )
            if i == 0:
                ref = ref_i
            else:
                ref = torch.cat((ref, ref_i), dim=0).contiguous()

        torch.cuda.synchronize()
        torch.testing.assert_close(ref, o.to(torch.float), atol=oatol, rtol=ortol)

    @pytest.mark.parametrize(
        "H_QK,H_V,D,T,B,dtype",
        [
            (16, 32, 128, 555, 1, testtype),
            (8, 32, 128, 128, 2, testtype),
            (16, 64, 128, 130, 1, testtype),
        ],
    )
    def test_fixlen_gva_state(self, set_seed, H_QK, H_V, D, T, B, dtype):
        """Test fixed-length GVA state output correctness."""

        _skip_if_gdn_blackwell_prefill_not_available()

        device = "cuda"

        q = torch.randn((B, T, H_QK, D), dtype=dtype)
        k = F.normalize(
            torch.randn(B, T, H_QK, D, dtype=torch.float32), p=2, dim=-1
        ).to(dtype)
        v = torch.randn((B, T, H_V, D), dtype=dtype)
        g = F.logsigmoid(torch.rand(1, T * B, H_V, dtype=torch.float32))
        beta = torch.rand(1, T * B, H_V, dtype=torch.float32).sigmoid()
        h0 = torch.randn((B, H_V, D, D), dtype=torch.float)

        q, k, v, beta, g, h0 = map(lambda x: x.to(device), (q, k, v, beta, g, h0))

        state_output = torch.zeros_like(h0, dtype=torch.float)

        o, state_output = chunk_gated_delta_rule(
            q.clone(),
            k.clone(),
            v.clone(),
            g.clone(),
            beta.clone(),
            None,
            h0.clone(),
            True,
            None,
            False,
            None,
            state_output,
        )
        torch.cuda.synchronize()

        repeat_factor = H_V // H_QK
        q = q.repeat_interleave(repeat_factor, dim=2)
        k = k.repeat_interleave(repeat_factor, dim=2)

        ref_ht = []
        h0_v_major = h0.transpose(-1, -2).contiguous()
        for i in range(B):
            _, ref_ht_i = recurrent_gated_delta_rule_ref(
                q=q[i].unsqueeze(0).clone(),
                k=k[i].unsqueeze(0).clone(),
                v=v[i].unsqueeze(0).clone(),
                beta=beta[:, (i * T) : ((i + 1) * T)].clone(),
                g=g[:, (i * T) : ((i + 1) * T)].clone(),
                initial_state=h0_v_major[i].clone(),
                output_final_state=True,
            )
            if i == 0:
                ref_ht = ref_ht_i
            else:
                ref_ht = torch.cat((ref_ht, ref_ht_i), dim=0).contiguous()

        ref_ht = torch.transpose(ref_ht, -1, -2).contiguous()
        torch.cuda.synchronize()

        torch.testing.assert_close(
            ref_ht, state_output.to(torch.float), atol=satol, rtol=srtol
        )


class TestGDNVariableLengthGVA:
    """Tests for variable-length sequences with Grouped Value Attention (H_QK != H_V)."""

    @pytest.mark.parametrize(
        "H_QK,H_V,D,cu_seqlens,dtype",
        [
            (16, 32, 128, [0, 13, 287], testtype),
            (8, 32, 128, [0, 256, 500, 1000], testtype),
            (16, 64, 128, [0, 128], testtype),
        ],
    )
    def test_varlen_gva_output(self, set_seed, H_QK, H_V, D, cu_seqlens, dtype):
        """Test variable-length GVA output correctness."""

        _skip_if_gdn_blackwell_prefill_not_available()

        device = "cuda"

        cu_seqlens_tensor = torch.tensor(cu_seqlens, device=device)
        T = cu_seqlens[-1]
        N = len(cu_seqlens) - 1

        q = torch.randn((1, T, H_QK, D), dtype=dtype)
        k = F.normalize(
            torch.randn(1, T, H_QK, D, dtype=torch.float32), p=2, dim=-1
        ).to(dtype)
        v = torch.randn((1, T, H_V, D), dtype=dtype)
        g = F.logsigmoid(torch.rand(1, T, H_V, dtype=torch.float32))
        beta = torch.rand(1, T, H_V, dtype=torch.float32).sigmoid()
        h0 = torch.randn((N, H_V, D, D), dtype=torch.float)

        q, k, v, beta, g, h0 = map(lambda x: x.to(device), (q, k, v, beta, g, h0))

        state_output = torch.zeros_like(h0, dtype=torch.float)

        o, state_output = chunk_gated_delta_rule(
            q.clone(),
            k.clone(),
            v.clone(),
            g.clone(),
            beta.clone(),
            None,
            h0.clone(),
            True,
            cu_seqlens_tensor.clone(),
            False,
            None,
            state_output,
        )
        torch.cuda.synchronize()

        repeat_factor = H_V // H_QK
        q = q.repeat_interleave(repeat_factor, dim=2)
        k = k.repeat_interleave(repeat_factor, dim=2)

        ref = []
        h0_v_major = h0.transpose(-1, -2)
        for i in range(N):
            ref_i, _ = recurrent_gated_delta_rule_ref(
                q=q[:, cu_seqlens[i] : cu_seqlens[i + 1]],
                k=k[:, cu_seqlens[i] : cu_seqlens[i + 1]],
                v=v[:, cu_seqlens[i] : cu_seqlens[i + 1]],
                beta=beta[:, cu_seqlens[i] : cu_seqlens[i + 1]],
                g=g[:, cu_seqlens[i] : cu_seqlens[i + 1]],
                initial_state=h0_v_major[i],
                output_final_state=True,
            )
            ref.append(ref_i)
        ref = torch.cat(ref, 1)
        torch.cuda.synchronize()

        torch.testing.assert_close(ref, o.to(torch.float), atol=oatol, rtol=ortol)

    @pytest.mark.parametrize(
        "H_QK,H_V,D,cu_seqlens,dtype",
        [
            (16, 32, 128, [0, 13, 287], testtype),
            (8, 32, 128, [0, 256, 500, 1000], testtype),
        ],
    )
    def test_varlen_gva_state(self, set_seed, H_QK, H_V, D, cu_seqlens, dtype):
        """Test variable-length GVA state output correctness."""

        _skip_if_gdn_blackwell_prefill_not_available()

        device = "cuda"

        cu_seqlens_tensor = torch.tensor(cu_seqlens, device=device)
        T = cu_seqlens[-1]
        N = len(cu_seqlens) - 1

        q = torch.randn((1, T, H_QK, D), dtype=dtype)
        k = F.normalize(
            torch.randn(1, T, H_QK, D, dtype=torch.float32), p=2, dim=-1
        ).to(dtype)
        v = torch.randn((1, T, H_V, D), dtype=dtype)
        g = F.logsigmoid(torch.rand(1, T, H_V, dtype=torch.float32))
        beta = torch.rand(1, T, H_V, dtype=torch.float32).sigmoid()
        h0 = torch.randn((N, H_V, D, D), dtype=torch.float)

        q, k, v, beta, g, h0 = map(lambda x: x.to(device), (q, k, v, beta, g, h0))

        state_output = torch.zeros_like(h0, dtype=torch.float)

        o, state_output = chunk_gated_delta_rule(
            q.clone(),
            k.clone(),
            v.clone(),
            g.clone(),
            beta.clone(),
            None,
            h0.clone(),
            True,
            cu_seqlens_tensor.clone(),
            False,
            None,
            state_output,
        )
        torch.cuda.synchronize()

        repeat_factor = H_V // H_QK
        q = q.repeat_interleave(repeat_factor, dim=2)
        k = k.repeat_interleave(repeat_factor, dim=2)

        ref_ht = []
        h0_v_major = h0.transpose(-1, -2)
        for i in range(N):
            _, ref_ht_i = recurrent_gated_delta_rule_ref(
                q=q[:, cu_seqlens[i] : cu_seqlens[i + 1]],
                k=k[:, cu_seqlens[i] : cu_seqlens[i + 1]],
                v=v[:, cu_seqlens[i] : cu_seqlens[i + 1]],
                beta=beta[:, cu_seqlens[i] : cu_seqlens[i + 1]],
                g=g[:, cu_seqlens[i] : cu_seqlens[i + 1]],
                initial_state=h0_v_major[i],
                output_final_state=True,
            )
            ref_ht.append(ref_ht_i)
        ref_ht = torch.cat(ref_ht, 0)
        ref_ht = torch.transpose(ref_ht, -1, -2)
        torch.cuda.synchronize()

        torch.testing.assert_close(
            ref_ht, state_output.to(torch.float), atol=satol, rtol=srtol
        )


class TestGDNMultipleRuns:
    """Tests for running the kernel multiple times (regression test)."""

    def test_consecutive_runs_deterministic(self, set_seed):
        """Test that multiple consecutive runs produce identical results."""
        _skip_if_gdn_blackwell_prefill_not_available()
        H, D, T, B = 16, 128, 256, 4
        dtype = testtype
        device = "cuda"

        q = torch.randn((B, T, H, D), dtype=dtype)
        k = F.normalize(torch.randn(B, T, H, D, dtype=torch.float32), p=2, dim=-1).to(
            dtype
        )
        v = torch.randn((B, T, H, D), dtype=dtype)
        g = F.logsigmoid(torch.rand(1, T * B, H, dtype=torch.float32))
        beta = torch.rand(1, T * B, H, dtype=torch.float32).sigmoid()
        h0 = torch.randn((B, H, D, D), dtype=torch.float)

        q, k, v, beta, g, h0 = map(lambda x: x.to(device), (q, k, v, beta, g, h0))

        results = []
        for _run_idx in range(3):
            state_output = torch.zeros_like(h0, dtype=torch.float)

            o, state_output = chunk_gated_delta_rule(
                q.clone(),
                k.clone(),
                v.clone(),
                g.clone(),
                beta.clone(),
                None,
                h0.clone(),
                True,
                None,
                False,
                None,
                state_output,
            )
            torch.cuda.synchronize()
            results.append((o.clone(), state_output.clone()))

        for i in range(1, len(results)):
            torch.testing.assert_close(
                results[0][0], results[i][0], atol=1e-6, rtol=1e-6
            )
            torch.testing.assert_close(
                results[0][1], results[i][1], atol=1e-6, rtol=1e-6
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
