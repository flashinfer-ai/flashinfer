import torch
import math

def convert_qkv_layout(q, k, v, src_layout, dst_layout):
        if src_layout == "HND" and dst_layout == "NHD":
            # [H, S, D] -> [S, H, D]
            q = q.permute(1, 0, 2).contiguous()
            k = k.permute(1, 0, 2).contiguous()
            v = v.permute(1, 0, 2).contiguous()
        elif src_layout == "NHD" and dst_layout == "HND":
            # [S, H, D] -> [H, S, D]
            q = q.permute(1, 0, 2).contiguous()
            k = k.permute(1, 0, 2).contiguous()
            v = v.permute(1, 0, 2).contiguous()
        else:
            raise NotImplementedError(f"Unsupported tensor layout conversion: {src_layout} -> {dst_layout}")
        return q, k, v

def convert_output_layout(out, src_layout, dst_layout):
    if src_layout == "HND" and dst_layout == "NHD":
        # [S, H, D] -> [H, S, D]
        out = out.permute(1, 0, 2).contiguous()
    elif src_layout == "NHD" and dst_layout == "HND":
        # [H, S, D] -> [S, H, D]
        out = out.permute(1, 0, 2).contiguous()
    else:
        raise NotImplementedError(f"Unsupported tensor layout conversion: {src_layout} -> {dst_layout}")
    return out


def split_varlen_input(tensor, seq_len_list, world_size, rank, tensor_layout="HND"): 

    """Split a concatenated variable-length tensor into equal chunks across ranks.

    Given a tensor of shape [total_seq_len, head_num, head_dim] where total_seq_len
    is the sum of multiple sub-sequences, split each sub-sequence into `world_size`
    parts and return the `rank`-th chunk concatenated together.
    For each sub-sequence, the first (world_size - 1) ranks each get
    ceil(seq_len / world_size) elements, and the last rank gets whatever
    remains.

    Args:
        tensor: Tensor of shape [total_seq_len, head_num, head_dim].
        seq_len_list: List of individual sequence lengths that sum to
                  total_seq_len, e.g. [1021, 1024, 1027].
        world_size: Number of ranks to split into.
        rank: Which chunk to return (0-indexed).

    Returns:
        A tensor of shape [chunk_seq_len, head_num, head_dim] where chunk_seq_len
        is the sum of the rank-th chunk of every sub-sequence.
    """
    if tensor_layout == "NHD":
        chunk_dim = 0
    elif tensor_layout == "HND":
        chunk_dim = 1
    else:
        raise ValueError(f"Invalid tensor layout: {tensor_layout}")

    seq_len_padded = torch.ceil(seq_len_list / world_size).to(torch.int32) * world_size
    total_seq_len_padded = sum(seq_len_padded)
    seq_len_padded_cur_rank = torch.ceil(total_seq_len_padded / world_size).to(torch.int32)

    chunks = []
    offset = 0
    for seq_len in seq_len_list:
        seq_len = int(seq_len)
        # First (world_size - 1) ranks get ceil(seq_len / world_size),
        # last rank gets whatever is left.
        base = math.ceil(seq_len / world_size)
        if rank < world_size - 1:
            chunk_len = base
            start = offset + base * rank
        else:
            # Last rank gets the remainder
            start = offset + base * (world_size - 1)
            chunk_len = seq_len - base * (world_size - 1)

        end = start + chunk_len
        chunks.append(tensor.narrow(chunk_dim, start, chunk_len))
        offset += seq_len

    res = torch.cat(chunks, dim=chunk_dim)

    if res.shape[chunk_dim] < seq_len_padded_cur_rank:
        pad_len = seq_len_padded_cur_rank - res.shape[chunk_dim]
        pad_shape = list(res.shape)
        pad_shape[chunk_dim] = pad_len
        res = torch.cat([res, torch.zeros(pad_shape, device=res.device, dtype=res.dtype)], dim=chunk_dim)
    
    return res