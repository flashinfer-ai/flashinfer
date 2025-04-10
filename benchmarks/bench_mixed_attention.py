import numpy as np
import torch
from triton.testing import do_bench

import flashinfer

page_block_size = 1
batch_size = 130
num_kv_heads = 4
num_qo_heads = 28
head_dim = 128
np.random.seed(42)

seq_lens = torch.tensor([600] * 122 + [10000] * 8)

seq_lens = seq_lens.int()
torch.random.manual_seed(42)
seq_lens_blocks = torch.ceil(seq_lens / page_block_size).int()
q_lens = torch.tensor([1] * 122 + [16] * 8).int()
q_indptr = torch.cat([torch.tensor([0]), torch.cumsum(q_lens, 0)], dim=0).int()
kv_indptr = torch.cat(
    [torch.tensor([0]), torch.cumsum(seq_lens_blocks, 0)], dim=0
).int()
last_page_len = seq_lens - (seq_lens_blocks - 1) * page_block_size
num_blocks = kv_indptr[-1].item()

q = torch.rand(q_indptr[-1].item(), num_qo_heads, head_dim).half().to(0)
kv_data = (
    torch.randn(num_blocks, 2, page_block_size, num_kv_heads, head_dim).to(0).half()
)
wrapper = flashinfer.BatchAttention(
    kv_layout="NHD",
)
wrapper.plan(
    q_indptr.to(0),
    kv_indptr.to(0),
    torch.arange(num_blocks).int().to(0),
    seq_lens.to(0),
    batch_size,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    head_dim,
    page_block_size,
    causal=True,
    q_data_type=torch.float16,
    kv_data_type=torch.float16,
)
print(wrapper._plan_info)

ms = do_bench(lambda: wrapper.run(q, kv_data))
print((q.numel() * q.element_size() + (kv_data.numel()) * kv_data.element_size()))
print(
    100
    * (q.numel() * q.element_size() + (kv_data.numel()) * kv_data.element_size())
    / ((ms * 3352 * 1024**3) / 1000)
)
