"""Run QSA sparse paged attention over an NVFP4 KV cache (SM80 through SM90)."""

import torch

import flashinfer
from flashinfer.sparse import qsa_sparse_paged_attention

rows, num_qo_heads, num_kv_heads, head_dim = 4, 24, 2, 256
num_blocks, page_size, selection_width = 64, 16, 512
device = "cuda"

# Quantize K/V into an NHD NVFP4 cache: packed E2M1 data plus one E4M3 scale per 16 dims.
k, v = (
    torch.randn(
        num_blocks * page_size, num_kv_heads, head_dim, device=device
    ).bfloat16()
    for _ in range(2)
)
cache_shape = (num_blocks, page_size, num_kv_heads)
data = tuple(
    torch.empty(*cache_shape, head_dim // 2, dtype=torch.uint8, device=device)
    for _ in range(2)
)
sf = tuple(
    torch.empty(*cache_shape, head_dim // 16, dtype=torch.float8_e4m3fn, device=device)
    for _ in range(2)
)
k_scale, v_scale = (x.abs().max().item() / (6 * 448) for x in (k, v))
slots = torch.arange(num_blocks * page_size, device=device)
flashinfer.nvfp4_quantize_append_paged_kv_cache_with_slot_mapping(
    k, v, slots, data, sf, k_scale, v_scale
)

# One request owning every block; each row attends to selection_width logical tokens,
# with the valid-entry count in the trailing column.
block_table = torch.arange(num_blocks, dtype=torch.int32, device=device)[None]
token_to_req = torch.zeros(rows, dtype=torch.int32, device=device)
indices = torch.randint(
    num_blocks * page_size, (rows, selection_width + 1), device=device
)
indices[:, -1] = selection_width
q, gate = (
    torch.randn(rows, num_qo_heads, head_dim, device=device).bfloat16()
    for _ in range(2)
)

out = qsa_sparse_paged_attention(
    q,
    data[0],
    data[1],
    indices.int(),
    block_table,
    token_to_req,
    False,
    k_scale=k_scale,
    v_scale=v_scale,
    output_gate=gate,
    k_sf=sf[0],
    v_sf=sf[1],
)
print(out.shape, out.dtype)
