import torch
from flashinfer.prefill import trtllm_batch_context_with_kv_cache
from flashinfer.decode import trtllm_batch_decode_with_kv_cache


def main():
    dtype = torch.bfloat16
    page_size = 64
    num_q_heads = 1
    num_kv_heads = 1
    head_size = 128
    workspace_size = 256 * 1024 * 1024

    # Following the test exactly:
    # q_lens: number of query tokens
    # in_kv_lens: number of existing KV tokens  
    # seq_lens = q_lens + in_kv_lens (TOTAL tokens for which K/V must be in cache)
    q_lens = torch.tensor([1, 1, 5], dtype=torch.int32)
    in_kv_lens = torch.tensor([5, 5, 5], dtype=torch.int32)
    seq_lens = q_lens + in_kv_lens  # Total = [6, 6]
    batch_size = q_lens.size(0)
    
    # Create cumulative indices
    cum_seq_lens_q = torch.cat([
        torch.tensor([0], dtype=torch.int32).cuda(),
        torch.cumsum(q_lens.cuda(), dim=0, dtype=torch.int32)
    ]).contiguous()
    
    # Calculate pages needed - based on TOTAL seq_lens
    page_per_seq = (seq_lens + page_size - 1) // page_size
    cum_seq_lens_kv = torch.cat([
        torch.tensor([0], dtype=torch.int32).cuda(), 
        torch.cumsum(page_per_seq.cuda(), dim=0, dtype=torch.int32)
    ]).contiguous()
    
    num_q_tokens = q_lens.sum().item()
    num_pages = page_per_seq.sum().item()
    
    # Create tensors
    query = torch.ones(num_q_tokens, num_q_heads, head_size, dtype=dtype).cuda()
    kv_cache = torch.zeros(num_pages, 2, num_kv_heads, page_size, head_size, dtype=dtype).cuda()
    workspace = torch.zeros(workspace_size, dtype=torch.uint8).cuda()
    block_tables = torch.tensor([[i] for i in range(batch_size)], dtype=torch.int32).cuda()
    
    k_cache = kv_cache[:, 0, :, :, :]
    v_cache = kv_cache[:, 1, :, :, :]
    
    for batch_idx in range(batch_size):
        page_idx = block_tables[batch_idx, 0].item()
        for pos in range(seq_lens[batch_idx].item()):
            k_cache[page_idx, :, pos, :] = 1.0
            v_cache[page_idx, :, pos, :] = float(pos)
            print(f"Batch {batch_idx} Pos {pos} set in page {page_idx}")
    
    # prefill
    prefill_output = trtllm_batch_context_with_kv_cache(
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

    # decode
    decode_output = trtllm_batch_decode_with_kv_cache(
        query=query.contiguous(),
        kv_cache=kv_cache,
        workspace_buffer=workspace,
        block_tables=block_tables,
        seq_lens=seq_lens.cuda(),
        max_q_len=q_lens.max().item(),
        max_kv_len=seq_lens.max().item(),
        bmm1_scale=1.0,  # bmm1_scale
        bmm2_scale=1.0,  # bmm2_scale  
        cum_seq_lens_q=cum_seq_lens_q,
        cum_seq_lens_kv=cum_seq_lens_kv,
        window_left=-1,  # window_left
        kv_layout="HND",
    )

    print("Prefill Output:")
    for b in range(batch_size):
        print(f"Batch {b}: {prefill_output[b, 0, 0]:.2f}")
    print()

    print("Decode Output:")
    for b in range(batch_size):
        print(f"Batch {b}: {decode_output[b, 0, 0]:.2f}")   
    print()


if __name__ == "__main__":
    main()
