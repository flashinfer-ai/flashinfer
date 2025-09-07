from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

import flashinfer


def wmape(target: torch.Tensor, preds: torch.Tensor):
    sum_abs_error = (preds - target).abs().sum().detach().item()
    sum_scale = target.abs().sum().detach().item()
    return sum_abs_error / sum_scale


from rope_reference import *


class DeepseekV2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        DeepseekV2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


class DeepseekV2AttentionVanilla(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden_size = 5120
        self.num_heads = 128

        self.q_lora_rank = 1536
        self.qk_rope_head_dim = 64
        self.kv_lora_rank = 512
        self.v_head_dim = 128
        self.qk_nope_head_dim = 128
        self.q_head_dim = 192  # 192 = config.qk_nope_head_dim + config.qk_rope_head_dim

        self.rope_theta = 10000

        # W^DQ ~ [5120, 1536]
        self.q_a_proj = nn.Linear(
            self.hidden_size,
            self.q_lora_rank,
            bias=False,
        )
        torch.nn.init.normal_(self.q_a_proj.weight)

        self.q_a_layernorm = DeepseekV2RMSNorm(self.q_lora_rank)

        # W^UQ & W^QR = [1536, 128*(128+64)]
        self.q_b_proj = nn.Linear(
            self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
        )
        torch.nn.init.normal_(self.q_b_proj.weight)

        # We don't need these modules, since we already have cached k_pe and compressed_kv tensor.
        # self.kv_a_proj_with_mqa = nn.Linear( # [,5120]-->[, 512+64] W^DKV & W^KR = [5120, 512+64]
        #     self.hidden_size,
        #     self.kv_lora_rank + self.qk_rope_head_dim,
        #     bias=False,
        # )
        # self.kv_a_layernorm = DeepseekV2RMSNorm(self.kv_lora_rank)

        # W^UK & W^UV ~ [512, 128*(128+128)]
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads
            * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
        )
        torch.nn.init.normal_(self.kv_b_proj.weight)

        # W^O ~ [128*128, 5120]
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
        )
        torch.nn.init.normal_(self.o_proj.weight)

        self.softmax_scale = self.q_head_dim ** (-0.5)

    def run_decode(
        self,
        hidden_states: torch.Tensor,
        compressed_kv_normed_cache: torch.Tensor,
        k_pe_cache: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        if q_len != 1:
            raise ValueError(
                f"Only support decode, but got hidden_states[{hidden_states.size()}]"
            )

        ckv_bsz, kv_len, ckv_dim = compressed_kv_normed_cache.size()
        if ckv_bsz != bsz or ckv_dim != self.kv_lora_rank:
            raise ValueError(
                f"Unexpected shape: compressed_kv_normed_cache[{compressed_kv_normed_cache.size()}]"
            )

        kpe_bsz, kpe_len, kpe_dim = k_pe_cache.size()
        if kpe_bsz != bsz or kpe_dim != self.qk_rope_head_dim or kv_len != kpe_len:
            raise ValueError(f"Unexpected shape: k_pe_cache[{k_pe_cache.size()}]")

        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        # q_nope ~ [bsz, q_len, 128]   q_pe ~ [bsz, q_len, 64]
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        k_pe = k_pe_cache.view(bsz, kv_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv = (
            self.kv_b_proj(compressed_kv_normed_cache)
            .view(bsz, kv_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            .transpose(1, 2)
        )
        # k_nope ~ [bsz, num_heads, kv_len, 128]  value_states ~ [bsz, num_heads, kv_len, 128]
        k_nope, value_states = torch.split(
            kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        if k_nope.size() != (bsz, self.num_heads, kv_len, self.qk_nope_head_dim):
            raise ValueError(f"k_nope[{k_nope.size()}]")
        if value_states.size() != (bsz, self.num_heads, kv_len, self.v_head_dim):
            raise ValueError(f"value_states[{value_states.size()}]")

        freqs_cis = precompute_freqs_cis(
            self.qk_rope_head_dim, kv_len, self.rope_theta, use_scaled=False
        ).to(q_pe.device)
        q_pe, k_pe = apply_rotary_emb(
            q_pe.transpose(1, 2).repeat(1, kv_len, 1, 1),
            k_pe.transpose(1, 2),
            freqs_cis,
        )
        q_pe = q_pe[:, -1:, :, :].transpose(1, 2)
        k_pe = k_pe.transpose(1, 2)

        # Concat q_nope and q_pe to produce a new Q tensor with head_dim = 192
        query_states = q.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

        # Concat k_nope and k_pe to produce a new K tensor with head_dim = 192
        key_states = k_pe.new_empty(bsz, self.num_heads, kv_len, self.q_head_dim)
        key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim :] = k_pe

        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale
        )

        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).reshape(
            bsz, q_len, self.num_heads * self.v_head_dim
        )

        output = self.o_proj(attn_output)

        return output


class DeepseekV2AttentionMatAbsorbDecode(nn.Module):
    def __init__(self, mla_vanilla: DeepseekV2AttentionVanilla):
        super().__init__()

        self.hidden_size = mla_vanilla.hidden_size  # 5120
        self.num_heads = mla_vanilla.num_heads  # 128

        self.q_lora_rank = mla_vanilla.q_lora_rank  # 1536
        self.qk_rope_head_dim = mla_vanilla.qk_rope_head_dim  # 64
        self.kv_lora_rank = mla_vanilla.kv_lora_rank  # 512
        self.v_head_dim = mla_vanilla.v_head_dim  # 128
        self.qk_nope_head_dim = mla_vanilla.qk_nope_head_dim  # 128
        self.q_head_dim = (
            mla_vanilla.q_head_dim
        )  # qk_nope_head_dim + qk_rope_head_dim # 128+64=192

        self.softmax_scale = mla_vanilla.softmax_scale

        self.rope_theta = mla_vanilla.rope_theta

        # W^DQ ~ [5120, 1536]
        self.W_DQ = mla_vanilla.q_a_proj.weight.transpose(0, 1)

        self.q_a_layernorm = DeepseekV2RMSNorm(self.q_lora_rank)

        # W_UQ ~ [1536, 128, 128]
        W_UQ, W_QR = torch.split(
            mla_vanilla.q_b_proj.weight.t().view(
                self.q_lora_rank, self.num_heads, self.q_head_dim
            ),
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            -1,
        )
        # W_UQ ~ [1536, 128*64]
        self.W_QR = W_QR.reshape(
            self.q_lora_rank, self.num_heads * self.qk_rope_head_dim
        )

        # W_UK ~ [512, 128, 128]   W_UV ~ [512, 128, 128]
        W_UK, W_UV = torch.split(
            mla_vanilla.kv_b_proj.weight.t().view(
                self.kv_lora_rank,
                self.num_heads,
                self.qk_nope_head_dim + self.v_head_dim,
            ),
            [self.qk_nope_head_dim, self.v_head_dim],
            -1,
        )

        # Now we merge W_UQ and W_UK (absorb W_UK into W_UQ)
        # q~q_lora_rank  n~num_heads  d~qk_nope_head_dim  l~kv_lora_rank
        self.W_UQ_UK = torch.einsum("q n d, l n d -> q n l", W_UQ, W_UK).flatten(
            start_dim=1
        )  # [1536, 65536]

        W_O = mla_vanilla.o_proj.weight.view(
            self.hidden_size, self.num_heads, self.v_head_dim
        )

        # Merge W_UV and W_O (absorb W_UV into W_O)
        # l~kv_lora_rank  n~num_heads  d~v_head_dim  h~hidden_size
        self.W_UV_O = torch.einsum("l n d, h n d -> n l h", W_UV, W_O).flatten(
            start_dim=0, end_dim=1
        )  # [65536, 5120]

    def run_proof_of_concept(
        self,
        hidden_states: torch.Tensor,
        compressed_kv_normed_cache: torch.Tensor,
        k_pe_cache: torch.Tensor,
        use_flashinfer_kernel: bool,
        convert_float16: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        c_Q = torch.matmul(hidden_states, self.W_DQ)
        # c_Q ~ [bsz, q_lora_rank:1536]
        c_Q = self.q_a_layernorm(c_Q)

        q_pe = torch.matmul(
            c_Q,
            self.W_QR,  # c_Q ~ [bsz, q_lora_rank~1536]
        )  # W_QR ~ [1536, 128*64]
        # q_pe ~ [bsz, 128, 64]
        q_pe = q_pe.reshape(bsz, self.num_heads, self.qk_rope_head_dim)

        q_nope = torch.matmul(c_Q, self.W_UQ_UK)  # W_UQ_UK~[1536, 128*512]
        # q_nope ~ [bsz, 128, 512]
        q_nope = q_nope.reshape(bsz, self.num_heads, self.kv_lora_rank)

        q_kv_dtype = torch.float16
        if convert_float16:
            q_nope = q_nope.to(q_kv_dtype)
            q_pe = q_pe.to(q_kv_dtype)
            compressed_kv_normed_cache = compressed_kv_normed_cache.to(q_kv_dtype)
            k_pe_cache = k_pe_cache.to(q_kv_dtype)

        if not use_flashinfer_kernel:
            freqs_cis = precompute_freqs_cis(
                self.qk_rope_head_dim, kv_len, self.rope_theta, use_scaled=False
            ).to(k_pe_cache.device)
            q_pe, k_pe_cache = apply_rotary_emb(
                q_pe.unsqueeze(1).repeat(1, kv_len, 1, 1),
                k_pe_cache.unsqueeze(2),
                freqs_cis,
            )
            q_pe = q_pe[:, -1:, :, :].squeeze(1)
            k_pe_cache = k_pe_cache.squeeze(2)

            # attn_weights_pe ~ [bsz, 128, kv_len]
            attn_weights_pe = torch.matmul(
                q_pe,  # [bsz, num_heads, qk_rope_head_dim]
                k_pe_cache.transpose(
                    1, 2
                ),  # [bsz, kv_len, 64] view(bsz, kv_len, self.qk_rope_head_dim)
            )
            # attn_weights_nope ~ [bsz, 128, kv_len]
            attn_weights_nope = torch.matmul(
                q_nope,  # [bsz, 128, 512]
                compressed_kv_normed_cache.transpose(1, 2),  # view(bsz, kv_len, 512)
            )

            attn_weights = (attn_weights_pe + attn_weights_nope) * self.softmax_scale

            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(q_nope.dtype)

            # attn_output ~ {attn_output.shape}") # [bsz, 128, 512]
            attn_output = torch.matmul(
                attn_weights,  # [bsz, 128, kv_len]
                compressed_kv_normed_cache,  # [bsz, kv_len, 512]
            )

        else:
            print("Now use MLA decode kernel!\n")
            if kv_len % page_size != 0:
                raise ValueError(
                    "For simplicity, kv_len should be multiple of page_size."
                )
            freqs_cis = precompute_freqs_cis(
                self.qk_rope_head_dim, kv_len, self.rope_theta, use_scaled=False
            ).to(k_pe_cache.device)
            q_pe, k_pe_cache = apply_rotary_emb(
                q_pe.unsqueeze(1).repeat(1, kv_len, 1, 1),
                k_pe_cache.unsqueeze(2),
                freqs_cis,
            )
            q_pe = q_pe[:, -1:, :, :].squeeze(1).contiguous()
            k_pe_cache = k_pe_cache.squeeze(2)
            num_pages_per_seq = kv_len // page_size
            total_num_pages = num_pages_per_seq * bsz

            kv_indptr = torch.arange(0, bsz + 1).to(dev_id).int() * num_pages_per_seq
            kv_indices = torch.arange(0, total_num_pages).to(dev_id).int()
            kv_last_page_len = torch.full((bsz,), page_size, dtype=torch.int32).to(
                dev_id
            )

            paged_ckv_cache = compressed_kv_normed_cache.reshape(
                total_num_pages, page_size, self.kv_lora_rank
            )
            paged_kpe_cache = k_pe_cache.reshape(
                total_num_pages, page_size, self.qk_rope_head_dim
            )

            workspace_buffer = torch.empty(64 * 1024 * 1024, dtype=torch.int8).to(
                dev_id
            )
            wrapper = flashinfer.BatchDecodeMlaWithPagedKVCacheWrapper(
                workspace_buffer,
                use_cuda_graph=True,
                use_tensor_cores=True,
                paged_kv_indptr_buffer=kv_indptr,
                paged_kv_indices_buffer=kv_indices,
                paged_kv_last_page_len_buffer=kv_last_page_len,
            )
            wrapper.plan(
                kv_indptr,
                kv_indices,
                kv_last_page_len,
                num_qo_heads=self.num_heads,
                head_dim_compressed_kv=self.kv_lora_rank,
                page_size=page_size,
                sm_scale=self.softmax_scale,
                rope_theta=self.rope_theta,
                data_type=q_kv_dtype,
                q_data_type=q_kv_dtype,
            )

            attn_output = wrapper.run(q_nope, q_pe, paged_ckv_cache, paged_kpe_cache)

            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    o, lse = wrapper.run(
                        q_nope, q_pe, paged_ckv_cache, paged_kpe_cache, return_lse=True
                    )
            torch.cuda.current_stream().wait_stream(s)

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                attn_output = wrapper.run(
                    q_nope, q_pe, paged_ckv_cache, paged_kpe_cache
                )
            g.replay()

        # output ~ [bsz, 5120]
        output = torch.matmul(
            attn_output.to(self.W_UV_O.dtype).reshape(
                bsz, self.num_heads * self.kv_lora_rank
            ),
            self.W_UV_O,
        )  # W_UV_O ~ [65536, 5120]

        return output


if __name__ == "__main__":
    dev_id = 0

    torch.manual_seed(666)
    torch.set_grad_enabled(False)

    mla_vanilla = DeepseekV2AttentionVanilla().cuda(device=dev_id)

    bsz = 6
    kv_len = 640
    page_size = 16

    hidden_states = torch.randn([bsz, 1, mla_vanilla.hidden_size]).to(dev_id)
    compressed_kv_normed_cache = torch.randn(
        [bsz, kv_len, mla_vanilla.kv_lora_rank]
    ).to(dev_id)
    k_pe_cache = torch.randn([bsz, kv_len, mla_vanilla.qk_rope_head_dim]).to(dev_id)

    output_vanilla = mla_vanilla.run_decode(
        hidden_states, compressed_kv_normed_cache, k_pe_cache
    )

    mla_mat_absorb = DeepseekV2AttentionMatAbsorbDecode(mla_vanilla).cuda(device=dev_id)
    output_mat_absorbed_use_torch_f32 = mla_mat_absorb.run_proof_of_concept(
        hidden_states.squeeze(1),
        compressed_kv_normed_cache,
        k_pe_cache,
        use_flashinfer_kernel=False,
        convert_float16=False,
    )
    output_mat_absorbed_use_torch_f16 = mla_mat_absorb.run_proof_of_concept(
        hidden_states.squeeze(1),
        compressed_kv_normed_cache,
        k_pe_cache,
        use_flashinfer_kernel=False,
        convert_float16=True,
    )
    output_mat_absorbed_use_flashinfer = mla_mat_absorb.run_proof_of_concept(
        hidden_states.squeeze(1),
        compressed_kv_normed_cache,
        k_pe_cache,
        use_flashinfer_kernel=True,
        convert_float16=True,
    )

    cos_use_torch_f32 = F.cosine_similarity(
        output_vanilla.reshape(-1), output_mat_absorbed_use_torch_f32.reshape(-1), dim=0
    )
    print(f"cos_use_torch_f32 = {cos_use_torch_f32}")
    assert cos_use_torch_f32 > 0.99

    wmape_use_torch_f32 = wmape(
        output_vanilla.reshape(-1), output_mat_absorbed_use_torch_f32.reshape(-1)
    )
    print(f"wmape_use_torch_f32 = {wmape_use_torch_f32}")
    assert wmape_use_torch_f32 < 0.02

    mse_use_torch_f32 = F.mse_loss(
        output_vanilla.reshape(-1), output_mat_absorbed_use_torch_f32.reshape(-1)
    )
    print(f"mse_use_torch_f32={mse_use_torch_f32}\n")

    cos_use_torch_f16 = F.cosine_similarity(
        output_vanilla.reshape(-1), output_mat_absorbed_use_torch_f16.reshape(-1), dim=0
    )
    print(f"cos_use_torch_f16 = {cos_use_torch_f16}")
    assert cos_use_torch_f16 > 0.99

    wmape_use_torch_f16 = wmape(
        output_vanilla.reshape(-1), output_mat_absorbed_use_torch_f16.reshape(-1)
    )
    print(f"wmape_use_torch_f16 = {wmape_use_torch_f16}")
    assert wmape_use_torch_f16 < 0.03

    mse_use_torch_f16 = F.mse_loss(
        output_vanilla.reshape(-1), output_mat_absorbed_use_torch_f16.reshape(-1)
    )
    print(f"mse_use_torch_f16 = {mse_use_torch_f16}\n")

    cos_use_flashinfer = F.cosine_similarity(
        output_vanilla.reshape(-1),
        output_mat_absorbed_use_flashinfer.reshape(-1),
        dim=0,
    )
    print(f"cos_use_flashinfer = {cos_use_flashinfer}")
    assert cos_use_flashinfer > 0.99

    wmape_use_flashinfer = wmape(
        output_vanilla.reshape(-1), output_mat_absorbed_use_flashinfer.reshape(-1)
    )
    print(f"wmape_use_flashinfer = {wmape_use_flashinfer}")
    assert wmape_use_flashinfer < 0.02

    mse_use_flashinfer = F.mse_loss(
        output_vanilla.reshape(-1), output_mat_absorbed_use_flashinfer.reshape(-1)
    )
    print(f"mse_use_flashinfer = {mse_use_flashinfer}")
