import torch
from flashinfer.prefill import trtllm_batch_context_with_kv_cache


def main():
    dtype = torch.bfloat16
    page_size = 64
    num_q_heads = 1
    num_kv_heads = 1
    head_size = 128
    workspace_size = 256 * 1024 * 1024
    batch_size = 2

    # Following the test exactly:
    # q_lens: number of query tokens
    # in_kv_lens: number of existing KV tokens  
    # seq_lens = q_lens + in_kv_lens (TOTAL tokens for which K/V must be in cache)
    q_lens = torch.tensor([1, 1], dtype=torch.int32)
    in_kv_lens = torch.tensor([5, 5], dtype=torch.int32) 
    seq_lens = q_lens + in_kv_lens  # Total = [6, 6]
    
    # Create cumulative indices
    cum_seq_lens_q = torch.cat([torch.tensor([0], dtype=torch.int32).cuda(), torch.cumsum(q_lens, dim=0).cuda()]).contiguous()
    
    # Calculate pages needed - based on TOTAL seq_lens
    page_per_seq = (seq_lens + page_size - 1) // page_size
    cum_seq_lens_kv = torch.cat([torch.tensor([0], dtype=torch.int32).cuda(), torch.cumsum(page_per_seq, dim=0).cuda()]).contiguous()
    
    print(f"q_lens: {q_lens}")
    print(f"in_kv_lens: {in_kv_lens}")
    print(f"seq_lens: {seq_lens}")
    print(f"page_per_seq: {page_per_seq}")
    print(f"cum_seq_lens_q: {cum_seq_lens_q}")
    print(f"cum_seq_lens_kv: {cum_seq_lens_kv}")
    print()
    
    num_q_tokens = q_lens.sum().item()
    num_pages = page_per_seq.sum().item()
    
    # Create tensors
    query = torch.ones(num_q_tokens, num_q_heads, head_size, dtype=dtype).cuda()
    kv_cache = torch.zeros(num_pages, 2, num_kv_heads, page_size, head_size, dtype=dtype).cuda()
    workspace = torch.zeros(workspace_size, dtype=torch.uint8).cuda()
    block_tables = torch.tensor([[0], [1]], dtype=torch.int32).cuda()
    
    # Fill KV cache for ALL positions (0 to seq_lens-1 for each batch)
    # Batch 0 uses page 0, batch 1 uses page 1
    # Each batch has seq_lens[i] = 6 tokens
    k_cache = kv_cache[:, 0, :, :, :]
    v_cache = kv_cache[:, 1, :, :, :]
    
    for batch_idx in range(batch_size):
        page_idx = block_tables[batch_idx, 0].item()
        for pos in range(seq_lens[batch_idx].item()):
            k_cache[page_idx, :, pos, :] = 1.0
            v_cache[page_idx, :, pos, :] = float(pos)
    
    print(f"Query shape: {query.shape}")
    print(f"KV cache shape: {kv_cache.shape}")
    print(f"Block tables: {block_tables}")
    print()
    
    # Debug print
    for batch_idx in range(batch_size):
        page_idx = block_tables[batch_idx, 0].item()
        print(f"Batch {batch_idx}, Page {page_idx} - K/V for first 6 positions:")
        for pos in range(6):
            print(f"  pos {pos}: K={k_cache[page_idx, 0, pos, 0]:.1f}, V={v_cache[page_idx, 0, pos, 0]:.1f}")
    print()
    
    # Call the function following test exactly
    output = trtllm_batch_context_with_kv_cache(
        query=query.contiguous(),
        kv_cache=kv_cache,
        workspace_buffer=workspace,
        block_tables=block_tables,
        seq_lens=seq_lens.cuda(),
        max_q_len=q_lens.max().item(),
        max_kv_len=seq_lens.max().item(),
        bmm1_scale=1.0,  # bmm1_scale
        bmm2_scale=1.0,  # bmm2_scale  
        batch_size=batch_size,
        cum_seq_lens_q=cum_seq_lens_q,
        cum_seq_lens_kv=cum_seq_lens_kv,
        window_left=-1,  # window_left
        kv_layout="HND",
    )
    
    print(f"Output shape: {output.shape}")
    for b in range(batch_size):
        print(f"Batch {b}: {output[b, 0, 0]:.2f}")
    print()
    print("Expected for causal: Each query attends to all previous + itself")
    print("Batch 0, token 0: attends to [0,1,2,3,4,5] -> avg = 2.5")
    print("Batch 1, token 0: attends to [0,1,2,3,4,5] -> avg = 2.5")


if __name__ == "__main__":
    main()
