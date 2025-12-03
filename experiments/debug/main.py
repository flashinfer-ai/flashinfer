import torch
import torch.nn.functional as F
from flashinfer.decode import trtllm_batch_decode_with_kv_cache
from flashinfer.prefill import trtllm_batch_context_with_kv_cache


def ref_batch_decode_with_paged_kv_cache(
    query, kv_cache, block_tables, seq_lens, cum_seq_lens_q, cum_seq_lens_kv, 
    max_q_len, max_kv_len, dtype=torch.bfloat16
):
    """
    Reference implementation of batch decode with paged KV cache.
    
    Args:
        query: [num_q_tokens, num_q_heads, head_size] - query tokens
        kv_cache: [num_pages, 2, num_kv_heads, page_size, head_size] - paged KV cache
        block_tables: [batch_size, max_blocks_per_seq] - block tables for each sequence
        seq_lens: [batch_size] - sequence lengths (q_len + kv_len)
        cum_seq_lens_q: [batch_size + 1] - cumulative query lengths
        cum_seq_lens_kv: [batch_size + 1] - cumulative KV lengths
        max_q_len: max query length
        max_kv_len: max KV length
    
    Returns:
        output: [num_q_tokens, num_q_heads, head_size] - attention output
    """
    batch_size = block_tables.shape[0]
    num_q_heads = query.shape[1]
    head_size = query.shape[2]
    num_kv_heads = kv_cache.shape[2]
    page_size = kv_cache.shape[3]
    num_pages = kv_cache.shape[0]
    
    # Number of heads per KV head (for GQA)
    num_heads_per_kv = num_q_heads // num_kv_heads
    
    output = torch.zeros_like(query)
    
    # Process each sequence in the batch
    for b in range(batch_size):
        q_start = cum_seq_lens_q[b]
        q_end = cum_seq_lens_q[b + 1]
        kv_start = cum_seq_lens_kv[b]
        kv_end = cum_seq_lens_kv[b + 1]
        
        q_len = q_end - q_start
        kv_len = kv_end - kv_start
        
        # Get query for this sequence: [q_len, num_q_heads, head_size]
        q = query[q_start:q_end]  # [q_len, num_q_heads, head_size]
        
        # Reconstruct K and V from paged cache
        block_table = block_tables[b]  # [max_blocks_per_seq]
        
        # Flatten paged KV cache into [kv_len, num_kv_heads, head_size]
        k_list = []
        v_list = []
        
        tokens_in_cache = 0
        for block_idx in block_table:
            if block_idx < 0:  # Invalid block
                break
            
            page = kv_cache[block_idx]  # [2, num_kv_heads, page_size, head_size]
            k_page = page[0]  # [num_kv_heads, page_size, head_size]
            v_page = page[1]  # [num_kv_heads, page_size, head_size]
            
            # Determine how many tokens are in this page
            remaining = kv_len - tokens_in_cache
            num_tokens_in_page = min(page_size, remaining)
            
            k_list.append(k_page[:, :num_tokens_in_page, :])  # [num_kv_heads, num_tokens, head_size]
            v_list.append(v_page[:, :num_tokens_in_page, :])  # [num_kv_heads, num_tokens, head_size]
            
            tokens_in_cache += num_tokens_in_page
            if tokens_in_cache >= kv_len:
                break
        
        # Concatenate pages: [num_kv_heads, kv_len, head_size]
        k = torch.cat(k_list, dim=1)
        v = torch.cat(v_list, dim=1)
        
        # Expand for GQA if needed: [num_q_heads, kv_len, head_size]
        if num_heads_per_kv > 1:
            k = k.repeat_interleave(num_heads_per_kv, dim=0)
            v = v.repeat_interleave(num_heads_per_kv, dim=0)
        
        # Compute attention: Q @ K^T -> [q_len, num_q_heads, kv_len]
        # q: [q_len, num_q_heads, head_size]
        # k: [num_q_heads, kv_len, head_size]
        # We need to reshape for batched matrix multiply
        
        q_reshaped = q.transpose(0, 1)  # [num_q_heads, q_len, head_size]
        
        # Compute scores: [num_q_heads, q_len, kv_len]
        scores = torch.matmul(q_reshaped, k.transpose(1, 2)) / (head_size ** 0.5)
        
        # Apply softmax: [num_q_heads, q_len, kv_len]
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values: [num_q_heads, q_len, head_size]
        attn_output = torch.matmul(attn_weights, v)
        
        # Transpose back: [q_len, num_q_heads, head_size]
        attn_output = attn_output.transpose(0, 1)
        
        # Store in output
        output[q_start:q_end] = attn_output
    
    return output


def main():
    dtype = torch.bfloat16
    batch_size = 2
    num_pages = 2
    page_size = 64
    num_q_heads = 1
    num_kv_heads = 1
    head_size = 128
    workspace_size = 256 * 1024 * 1024

    q_lens = torch.tensor([1, 2], dtype=torch.int32).cuda()
    kv_lens = torch.tensor([33, 44], dtype=torch.int32).cuda()
    cum_seq_lens_q = torch.cat([torch.tensor([0], dtype=torch.int32).cuda(), torch.cumsum(q_lens, dim=0)]).cuda()
    cum_seq_lens_kv = torch.cat([torch.tensor([0], dtype=torch.int32).cuda(), torch.cumsum(kv_lens, dim=0)]).cuda()
    seq_lens = q_lens + kv_lens
    max_q_len = q_lens.max().item()
    num_q_tokens = q_lens.sum().item()
    query = torch.ones(num_q_tokens, num_q_heads, head_size, dtype=dtype).cuda()
    kv_cache = torch.randn(num_pages, 2, num_kv_heads, page_size, head_size, dtype=dtype).cuda()
    workspace = torch.empty(workspace_size, dtype=torch.uint8).cuda()
    block_tables = torch.tensor([[0], [1]], dtype=torch.int32).cuda()
    max_seq_len = seq_lens.max().item()
    kv_layout = "HND"

    decode_output = trtllm_batch_decode_with_kv_cache(
        query=query,
        kv_cache=kv_cache,
        workspace_buffer=workspace,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_kv_len=max_seq_len,
        kv_layout=kv_layout,
        backend='trtllm-gen',
        max_q_len=max_q_len,
        cum_seq_lens_q=cum_seq_lens_q,
        cum_seq_lens_kv=cum_seq_lens_kv
    )
    # print(query)
    # print(kv_cache)
    print("Decode output:")
    print(decode_output)

    prefill_output = trtllm_batch_context_with_kv_cache(
        batch_size=batch_size,
        query=query,
        kv_cache=kv_cache,
        workspace_buffer=workspace,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_q_len=max_q_len,
        max_kv_len=max_seq_len,
        kv_layout=kv_layout,
        cum_seq_lens_q=cum_seq_lens_q,
        cum_seq_lens_kv=cum_seq_lens_kv,
        bmm1_scale=1.0,
        bmm2_scale=1.0,
    )
    print("Prefill output:")
    print(prefill_output)

    expected_output = ref_batch_decode_with_paged_kv_cache(
        query=query,
        kv_cache=kv_cache,
        block_tables=block_tables,
        seq_lens=seq_lens,
        cum_seq_lens_q=cum_seq_lens_q,
        cum_seq_lens_kv=cum_seq_lens_kv,
        max_q_len=max_q_len,
        max_kv_len=max_seq_len,
        dtype=dtype
    )
    print("Ref output:")
    print(expected_output)

    print("Difference(prefill vs. ref):")
    print(prefill_output - expected_output)




if __name__ == "__main__":
    main()
