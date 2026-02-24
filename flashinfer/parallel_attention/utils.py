import torch


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
        raise NotImplementedError(
            f"Unsupported tensor layout conversion: {src_layout} -> {dst_layout}"
        )
    return q, k, v


def convert_output_layout(out, src_layout, dst_layout):
    if src_layout == "HND" and dst_layout == "NHD":
        # [S, H, D] -> [H, S, D]
        out = out.permute(1, 0, 2).contiguous()
    elif src_layout == "NHD" and dst_layout == "HND":
        # [H, S, D] -> [S, H, D]
        out = out.permute(1, 0, 2).contiguous()
    else:
        raise NotImplementedError(
            f"Unsupported tensor layout conversion: {src_layout} -> {dst_layout}"
        )
    return out


def split_varlen_input(tensor, seq_len_list, world_size, rank, tensor_layout="HND"):
    """Split a packed variable-length tensor across ranks for context parallelism.

    Given a tensor whose sequence dimension is the concatenation of multiple
    sub-sequences, split each sub-sequence into ``world_size`` chunks and return
    the ``rank``-th chunk concatenated together. The first ``world_size - 1``
    ranks each get ``ceil(seq_len / world_size)`` tokens per sub-sequence;
    the last rank gets the remainder. The result is zero-padded so that all
    ranks have the same total sequence length.

    Args:
        tensor: Input tensor of shape ``[H, total_seq_len, D]`` (HND) or
            ``[total_seq_len, H, D]`` (NHD).
        seq_len_list: Individual sequence lengths that sum to ``total_seq_len``,
            e.g. ``[1021, 1024, 1027]``. Can be a list, tuple, or torch.Tensor.
        world_size: Number of ranks to split across.
        rank: Which rank's chunk to return (0-indexed).
        tensor_layout: ``"HND"`` or ``"NHD"``.

    Returns:
        torch.Tensor: The rank's chunk, zero-padded to uniform length across ranks.
    """
    if not isinstance(seq_len_list, torch.Tensor):
        seq_len_list = torch.tensor(seq_len_list, dtype=torch.int32)

    if tensor_layout == "NHD":
        chunk_dim = 0
    elif tensor_layout == "HND":
        chunk_dim = 1
    else:
        raise ValueError(f"Invalid tensor layout: {tensor_layout}")

    seq_len_padded = (seq_len_list + world_size - 1) // world_size * world_size
    total_seq_len_padded = sum(seq_len_padded)
    seq_len_padded_cur_rank = (
        (total_seq_len_padded + world_size - 1) // world_size
    ).to(torch.int32)

    chunks = []
    offset = 0
    for seq_len in seq_len_list:
        seq_len = int(seq_len)
        # First (world_size - 1) ranks get ceil(seq_len / world_size),
        # last rank gets whatever is left.
        base = (seq_len + world_size - 1) // world_size
        if rank < world_size - 1:
            chunk_len = base
            start = offset + base * rank
        else:
            # Last rank gets the remainder
            start = offset + base * (world_size - 1)
            chunk_len = seq_len - base * (world_size - 1)

        chunks.append(tensor.narrow(chunk_dim, start, chunk_len))
        offset += seq_len

    res = torch.cat(chunks, dim=chunk_dim)

    if res.shape[chunk_dim] < seq_len_padded_cur_rank:
        pad_len = seq_len_padded_cur_rank - res.shape[chunk_dim]
        pad_shape = list(res.shape)
        pad_shape[chunk_dim] = pad_len
        res = torch.cat(
            [res, torch.zeros(pad_shape, device=res.device, dtype=res.dtype)],
            dim=chunk_dim,
        )

    return res
