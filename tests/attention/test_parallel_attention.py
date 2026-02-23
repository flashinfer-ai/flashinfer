import math
import torch
import torch.distributed as dist
import torch.nn.functional as F

from flashinfer.parallel_attention.parallel_attention import ParallelAttention
from flashinfer.parallel_attention.parallel_config import AttnParallelConfig
from flashinfer.parallel_attention.parallel_config import UnevenCPConfig
from flashinfer.parallel_attention.parallel_config import VarlenCPConfig
from flashinfer.parallel_attention.utils import split_varlen_input


def sample_tensors(num_heads, seq_len, head_dim, world_size):
    """Create sample tensors for attention testing."""

    shape = (num_heads, seq_len, head_dim)
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")

    # Prepare inputs
    q = torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=False)
    k = torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=False)
    v = torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=False)

    dist.broadcast(q, src=0)
    dist.broadcast(k, src=0)
    dist.broadcast(v, src=0)

    local_q = q.chunk(world_size, dim=1)[rank]
    local_k = k.chunk(world_size, dim=1)[rank]
    local_v = v.chunk(world_size, dim=1)[rank]
    return q, k, v, local_q, local_k, local_v


def sample_ring_varlen_tensors(num_heads, head_dim, world_size, seq_len_list):
   
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")

    total_seq_len = sum(seq_len_list)
    
    shape = (num_heads, total_seq_len, head_dim)

    q = torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=False)
    k = torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=False)
    v = torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=False)
    
    dist.broadcast(q, src=0)
    dist.broadcast(k, src=0)
    dist.broadcast(v, src=0)

    local_q = split_varlen_input(q, seq_len_list, world_size, rank)
    local_k = split_varlen_input(k, seq_len_list, world_size, rank)
    local_v = split_varlen_input(v, seq_len_list, world_size, rank)
    
    return q, k, v, local_q, local_k, local_v



def test_attn_parallel(
    num_heads, seq_len, head_dim, world_size, ulysses_size, ring_size, attn_type, tensor_layout="HND"
):
    """Test basic parallel attention functionality."""

    query, key, value, local_query, local_key, local_value = sample_tensors(
        num_heads, seq_len, head_dim, world_size
    )

    if tensor_layout == "NHD":
        local_query = local_query.permute(1, 0, 2).contiguous()
        local_key = local_key.permute(1, 0, 2).contiguous()
        local_value = local_value.permute(1, 0, 2).contiguous()

    attn_parallel_config = AttnParallelConfig()
    attn_parallel_config.set_config(ulysses_size=ulysses_size, ring_size=ring_size)
    attn = ParallelAttention(attn_type=attn_type, attn_parallel_config=attn_parallel_config, uneven_cp_config=UnevenCPConfig(), varlen_cp_config=VarlenCPConfig())
    
    
    local_output = attn.run(local_query, 
                        local_key, 
                        local_value, 
                        tensor_layout)
    if tensor_layout == "NHD":
        local_output = local_output.permute(1, 0, 2)

    ref_output = F.scaled_dot_product_attention(query.unsqueeze(0), key.unsqueeze(0), value.unsqueeze(0), is_causal=False)
    local_ref_output = ref_output.chunk(world_size, dim=2)[dist.get_rank()]

    cos_sim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    cos_similarity = cos_sim(local_output.reshape(-1).to(torch.float32), local_ref_output.reshape(-1).to(torch.float32))
    print("cos_similarity total: ", cos_similarity)
    if cos_similarity < 0.99:
        raise RuntimeError("Accuracy test failed")



def test_uneven_attn_parallel(
    num_heads, seq_len_padded, head_dim, world_size, ulysses_size, ring_size, attn_type
):
    """Test uneven parallel attention functionality."""

    query, key, value, local_query, local_key, local_value = sample_tensors(
        num_heads, seq_len_padded, head_dim, world_size
    )

    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")
    uneven_number = world_size - 1

    seq_len_cur_rank = torch.tensor([local_query.shape[1]], dtype=torch.int32, device=device)
    if dist.get_rank() == world_size - 1:
        seq_len_cur_rank = seq_len_cur_rank - uneven_number
    attn_parallel_config = AttnParallelConfig()
    attn_parallel_config.set_config(ulysses_size=ulysses_size, ring_size=ring_size)
    uneven_cp_config = UnevenCPConfig()
    uneven_cp_config.set_uneven_cp_config(seq_len_padded - uneven_number, seq_len_padded, seq_len_cur_rank, attn_parallel_config)
    varlen_cp_config = VarlenCPConfig()
    attn = ParallelAttention(attn_type=attn_type, attn_parallel_config=attn_parallel_config, uneven_cp_config=uneven_cp_config, varlen_cp_config=varlen_cp_config)


    local_output = attn.run(local_query, local_key, local_value, tensor_layout="HND")

    query = query[:, :-uneven_number, :]
    key = key[:, :-uneven_number, :]
    value = value[:, :-uneven_number, :]

    ref_output = F.scaled_dot_product_attention(query.unsqueeze(0), key.unsqueeze(0), value.unsqueeze(0), is_causal=False)
    local_ref_output = ref_output.chunk(world_size, dim=2)[dist.get_rank()]

    if dist.get_rank() == world_size - 1:
        local_output = local_output[:, :-uneven_number, :]

    cos_sim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    cos_similarity = cos_sim(local_output.reshape(-1).to(torch.float32), local_ref_output.reshape(-1).to(torch.float32))
    print("cos_similarity total: ", cos_similarity)
    if cos_similarity < 0.99:
        raise RuntimeError("Accuracy test failed")



def test_ulysses_varlen_attn_parallel(
    num_heads, seq_len_list, head_dim, world_size, ulysses_size, ring_size, attn_type
):
    """Test uneven parallel attention functionality."""


    total_seq_len = sum(seq_len_list)
    seq_len_padded = math.ceil(total_seq_len / world_size) * world_size
    uneven_number = seq_len_padded - total_seq_len

    query, key, value, local_query, local_key, local_value = sample_tensors(
        num_heads, seq_len_padded, head_dim, world_size
    )

    attn_parallel_config = AttnParallelConfig()
    attn_parallel_config.set_config(ulysses_size=ulysses_size, ring_size=ring_size)
    varlen_cp_config = VarlenCPConfig()
    varlen_cp_config.set_ulysses_varlen_config(seq_len_list, seq_len_list, attn_parallel_config)
    uneven_cp_config = UnevenCPConfig()
    attn = ParallelAttention(attn_type=attn_type, attn_parallel_config=attn_parallel_config, uneven_cp_config=uneven_cp_config, varlen_cp_config=varlen_cp_config)
    
    local_output = attn.run(local_query, local_key, local_value, tensor_layout="HND")

    cu_seqlens_q = varlen_cp_config.cu_seqlens_q_cur_ulysses_group.cpu()
    cu_seqlens_kv = varlen_cp_config.cu_seqlens_kv_cur_ulysses_group.cpu()
    local_ref_output_list = []
    for i in range(len(seq_len_list)):
        q_tmp = query[:, cu_seqlens_q[i]:cu_seqlens_q[i+1], :]
        k_tmp = key[:, cu_seqlens_kv[i]:cu_seqlens_kv[i+1], :]
        v_tmp = value[:, cu_seqlens_kv[i]:cu_seqlens_kv[i+1], :]
        tmp_output = F.scaled_dot_product_attention(q_tmp.unsqueeze(0), k_tmp.unsqueeze(0), v_tmp.unsqueeze(0), is_causal=False)
        local_ref_output_list.append(tmp_output)

    ref_output = torch.cat(local_ref_output_list, dim=2) 

    local_ref_output = ref_output.chunk(world_size, dim=2)[dist.get_rank()]

    if dist.get_rank() == world_size - 1 and seq_len_padded > total_seq_len:
        local_output = local_output[:, :-uneven_number, :]

    cos_sim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    cos_similarity = cos_sim(local_output.reshape(-1).to(torch.float32), local_ref_output.reshape(-1).to(torch.float32))
    print("cos_similarity total: ", cos_similarity)
    if cos_similarity < 0.99:
        raise RuntimeError("Accuracy test failed")



def test_ring_varlen_attn_parallel(
    num_heads, seq_len_list, head_dim, world_size, ulysses_size, ring_size, attn_type
):
    """Test uneven parallel attention functionality."""

    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")
    

    full_cu_seqlens = [0]
    for seq_len in seq_len_list:
        full_cu_seqlens.append(full_cu_seqlens[-1] + seq_len)


    query, key, value, local_query, local_key, local_value = sample_ring_varlen_tensors(num_heads, head_dim, world_size, seq_len_list)


    attn_parallel_config = AttnParallelConfig()
    attn_parallel_config.set_config(ulysses_size=1, ring_size=ring_size)
    
    varlen_cp_config = VarlenCPConfig()
    varlen_cp_config.set_ring_varlen_config(seq_len_list, seq_len_list, attn_parallel_config)
    uneven_cp_config = UnevenCPConfig()

    attn = ParallelAttention(attn_type=attn_type, attn_parallel_config=attn_parallel_config, uneven_cp_config=uneven_cp_config, varlen_cp_config=varlen_cp_config)
    local_output = attn.run(local_query, local_key, local_value, tensor_layout="HND")

    local_ref_output_list = []
    for i in range(len(seq_len_list)):
        q_tmp = query[:, full_cu_seqlens[i]:full_cu_seqlens[i+1], :]
        k_tmp = key[:, full_cu_seqlens[i]:full_cu_seqlens[i+1], :]
        v_tmp = value[:, full_cu_seqlens[i]:full_cu_seqlens[i+1], :]
        tmp_output = F.scaled_dot_product_attention(q_tmp.unsqueeze(0), k_tmp.unsqueeze(0), v_tmp.unsqueeze(0), is_causal=False)
        local_ref_output_list.append(tmp_output)

    ref_output = torch.cat(local_ref_output_list, dim=2).squeeze(0)
    local_ref_output = split_varlen_input(ref_output, seq_len_list, world_size, rank)
    
    cos_sim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    cos_similarity = cos_sim(local_output.reshape(-1).to(torch.float32), local_ref_output.reshape(-1).to(torch.float32))
    print("cos_similarity total: ", cos_similarity)
    if cos_similarity < 0.99:
        raise RuntimeError("Accuracy test failed")




if __name__ == "__main__":
    dist.init_process_group("nccl")
    world_size = dist.get_world_size()

    capability = torch.cuda.get_device_capability(0)
    sm = f"{capability[0]}{capability[1]}"
    

    test_attn_parallel(
        num_heads=24,
        seq_len=6 * 8 * 1024,
        head_dim=128,
        world_size=world_size,
        ulysses_size=2,
        ring_size=2,
        attn_type="flash-attn3",
    )

    test_uneven_attn_parallel(
            num_heads=24,
            seq_len_padded=6 * 8 * 1024,
            head_dim=128,
            world_size=world_size,
            ulysses_size=2,
            ring_size=2,
            attn_type="flash-attn3",
        )

    test_ulysses_varlen_attn_parallel(
            num_heads=24,
            seq_len_list=[1 * 8 * 1024 - 1, 3 * 8 * 1024],
            head_dim=128,
            world_size=world_size,
            ulysses_size=world_size,
            ring_size=1,
            attn_type="flash-attn3",
        )

    test_ring_varlen_attn_parallel(
            num_heads=24,
            seq_len_list=torch.tensor([1021, 1024, 1027, 750, 826], dtype=torch.int32),
            head_dim=128,
            world_size=world_size,
            ulysses_size=1,
            ring_size=world_size,
            attn_type="flash-attn3",
        )

    dist.destroy_process_group()






