"""Small GLM-4.7-Flash MLA decode example: 20 heads, compressed dim 512+64."""

import torch
from flashinfer.mla import BatchMLAPagedAttentionWrapper

device, dtype = "cuda", torch.bfloat16
batch, heads, seq_len, page_size = 2, 20, 1024, 16
pages_per_request = seq_len // page_size
workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
wrapper = BatchMLAPagedAttentionWrapper(workspace, backend="fa2")

# Q already includes absorption of the K-up projection; RoPE is already applied.
q_nope = torch.randn(batch, heads, 512, dtype=dtype, device=device)
q_pe = torch.randn(batch, heads, 64, dtype=dtype, device=device)
# Each token stores one shared latent vector and one shared RoPE key.
ckv = torch.randn(batch * pages_per_request, page_size, 512, dtype=dtype, device=device)
kpe = torch.randn(batch * pages_per_request, page_size, 64, dtype=dtype, device=device)

wrapper.plan(
    qo_indptr=torch.arange(batch + 1, dtype=torch.int32),
    kv_indptr=torch.arange(batch + 1, dtype=torch.int32) * pages_per_request,
    kv_indices=torch.arange(batch * pages_per_request, dtype=torch.int32),
    kv_len_arr=torch.full((batch,), seq_len, dtype=torch.int32),
    num_heads=heads,
    head_dim_ckv=512,
    head_dim_kpe=64,
    page_size=page_size,
    causal=False,  # One decode query sees all cached tokens, including itself.
    sm_scale=(192 + 64) ** -0.5,
    q_data_type=dtype,
    kv_data_type=dtype,
)
o, lse = wrapper.run(q_nope, q_pe, ckv, kpe, return_lse=True, return_lse_base_on_e=True)
torch.cuda.synchronize()
print("latent output:", o.shape, o.dtype)  # [2, 20, 512]
print("natural-log LSE:", lse.shape, lse.dtype)  # [2, 20], FP32
# The model then applies each head's W_UV (512 -> 256) and output projection.
