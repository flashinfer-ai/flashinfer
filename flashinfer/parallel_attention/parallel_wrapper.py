import logging

import torch
import torch.distributed as dist
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def all_to_all(tensor, scatter_idx, gather_idx, tensor_layout, group=None):
    """Perform all-to-all communication on a tensor.

    Args:
        tensor (torch.Tensor): Input tensor for all-to-all communication
        scatter_idx (int): Dimension to scatter, will split along this dimension
            and then scatter to all processes
        gather_idx (int): Dimension to gather, will gather from all processes
            and then concatenate along this dimension
        group (ProcessGroup, optional): Process group to use for communication

    Returns:
        torch.Tensor
    """
    if not dist.is_initialized():
        return tensor

    world_size = dist.get_world_size(group)
    if world_size == 1:
        return tensor

    if scatter_idx == gather_idx:
        raise ValueError("scatter_idx and gather_idx must be different")

    def chunk_tensor(tensor, scatter_idx):
        t_shape = list(tensor.shape)
        if t_shape[scatter_idx] % world_size != 0:
            raise ValueError(
                f"Dimension {scatter_idx} of tensor {tensor.shape} "
                f"must be divisible by world size {world_size}"
            )
        chunk_size = t_shape[scatter_idx] // world_size
        new_shape = []
        for i in range(len(t_shape)):
            if i != scatter_idx:
                new_shape.append(t_shape[i])
            else:
                new_shape.extend([world_size, chunk_size])
        tensor = tensor.reshape(*new_shape)
        # move scatter_idx to front
        tensor = tensor.permute(
            scatter_idx,
            *[i for i in range(len(new_shape)) if i != scatter_idx],
        ).contiguous()
        return tensor

    # chunk tensor for all_to_all
    tensor = chunk_tensor(tensor, scatter_idx)

    # Perform all2all
    output = torch.empty_like(tensor)
    dist.all_to_all_single(output, tensor, group=group)

    # output: e.g., [world_size, chunked_H, chunked_S, D]
    # if scatter_idx == 0, gather_idx == 1 -> [chunked_H, S, D]
    def reorder_tensor(tensor, gather_idx):
        t_shape = list(tensor.shape)
        world_size = t_shape[0]
        # insert front to gather_idx + 1
        permute_idx = []
        for i in range(1, len(t_shape)):
            if i != gather_idx + 1:
                permute_idx.append(i)
            else:
                permute_idx.extend([0, i])
        tensor = tensor.permute(*permute_idx).contiguous()

        # reshape tensor
        new_shape = []
        for i in range(1, len(t_shape)):
            if i != gather_idx + 1:
                new_shape.append(t_shape[i])
            else:
                new_shape.append(world_size * t_shape[i])

        tensor = tensor.reshape(*new_shape)

        return tensor

    output = reorder_tensor(output, gather_idx)

    return output


def ulysses_a2a_in(
    query,
    key,
    value,
    attn_mask,
    tensor_layout,
    ulysses_size=1,
    ulysses_rank=0,
    ulysses_group=None,
    fuse_qkv=False,
):
    if ulysses_size == 1:
        return query, key, value, attn_mask

    if attn_mask is not None:
        raise NotImplementedError("Attn mask not supported for ulysses_a2a_in")

    if tensor_layout == "HND":
        scatter_idx = 0
        gather_idx = 1
    elif tensor_layout == "NHD":
        scatter_idx = 1
        gather_idx = 0
    else:
        raise ValueError(f"Invalid tensor layout: {tensor_layout}")

    # [H, S/N, D] -> [H/N, S, D]
    if fuse_qkv:
        # Fused communication: concatenate q/k/v into [3, H, S/N, D],
        # single all-to-all, then split.
        # This reduces 3 NCCL calls to 1, improving efficiency.
        query = torch.unsqueeze(query, 0)
        key = torch.unsqueeze(key, 0)
        value = torch.unsqueeze(value, 0)
        qkv = torch.cat([query, key, value], dim=0)
        qkv = all_to_all(
            qkv,
            scatter_idx=scatter_idx + 1,
            gather_idx=gather_idx + 1,
            group=ulysses_group,
            tensor_layout=tensor_layout,
        )
        query, key, value = torch.chunk(qkv, 3, dim=0)
        query = query.squeeze(0)
        key = key.squeeze(0)
        value = value.squeeze(0)
    else:
        # Independent communication: 3 separate all-to-all operations (default, safe)
        query = all_to_all(
            query,
            scatter_idx=scatter_idx,
            gather_idx=gather_idx,
            group=ulysses_group,
            tensor_layout=tensor_layout,
        )
        key = all_to_all(
            key,
            scatter_idx=scatter_idx,
            gather_idx=gather_idx,
            group=ulysses_group,
            tensor_layout=tensor_layout,
        )
        value = all_to_all(
            value,
            scatter_idx=scatter_idx,
            gather_idx=gather_idx,
            group=ulysses_group,
            tensor_layout=tensor_layout,
        )

    return query, key, value, attn_mask


def ulysses_a2a_out(output, tensor_layout, ulysses_size=1, ulysses_group=None):
    if ulysses_size == 1:
        return output

    assert tensor_layout in ["NHD", "HND"], (
        f"tensor_layout must be NHD or HND, but got {tensor_layout}"
    )
    if tensor_layout == "HND":
        scatter_idx = 1
        gather_idx = 0
    elif tensor_layout == "NHD":
        scatter_idx = 0
        gather_idx = 1
    else:
        raise ValueError(f"Invalid tensor layout: {tensor_layout}")
    # [H/N, S, D] -> [H, S/N, D]
    output = all_to_all(
        output,
        scatter_idx=scatter_idx,
        gather_idx=gather_idx,
        tensor_layout=tensor_layout,
        group=ulysses_group,
    )
    return output


def ring_fwd_out_correction(
    out: torch.Tensor,
    out_per_step: torch.Tensor,
    softmax_lse: torch.Tensor,
    softmax_lse_per_step: torch.Tensor,
):
    """Merge partial outputs of each step in ring attention"""
    new_out = out - F.sigmoid(
        softmax_lse_per_step.unsqueeze(-1) - softmax_lse.unsqueeze(-1)
    ) * (out - out_per_step)
    out.copy_(new_out)


def ring_fwd_softmax_lse_correction(
    softmax_lse: torch.Tensor,
    softmax_lse_per_step: torch.Tensor,
):
    """Merge softmax stats of each step in ring attention"""
    new_lse = softmax_lse - F.logsigmoid(softmax_lse - softmax_lse_per_step)
    softmax_lse.copy_(new_lse)


def ring_attn_p2p_communicate(
    rank, send_tensor, send_dst, recv_tensor, recv_src, ring_group
):
    """Point-to-point communications of KV and dKV in ring attention"""
    send_recv_ops = []
    if rank % 2 == 0:
        send_op = torch.distributed.P2POp(
            torch.distributed.isend,
            send_tensor,
            group_peer=send_dst,
            group=ring_group,
        )
        recv_op = torch.distributed.P2POp(
            torch.distributed.irecv,
            recv_tensor,
            group_peer=recv_src,
            group=ring_group,
        )
        send_recv_ops.append(send_op)
        send_recv_ops.append(recv_op)
    else:
        recv_op = torch.distributed.P2POp(
            torch.distributed.irecv,
            recv_tensor,
            group_peer=recv_src,
            group=ring_group,
        )
        send_op = torch.distributed.P2POp(
            torch.distributed.isend,
            send_tensor,
            group_peer=send_dst,
            group=ring_group,
        )
        send_recv_ops.append(recv_op)
        send_recv_ops.append(send_op)
    send_recv_reqs = torch.distributed.batch_isend_irecv(send_recv_ops)

    return send_recv_reqs


def ulysses_wrapper(func):
    def wrapper(self, query, key, value, tensor_layout, attn_mask=None, **kwargs):
        ulysses_group = self.ulysses_group
        ring_group = self.ring_group
        uneven_cp_config = self.uneven_cp_config
        varlen_cp_config = self.varlen_cp_config

        ulysses_size = (
            dist.get_world_size(ulysses_group) if ulysses_group is not None else 1
        )
        ring_size = dist.get_world_size(ring_group) if ring_group is not None else 1

        if kwargs.get("return_lse", False):
            raise ValueError("return_lse=True is not supported in parallel attention")

        if ulysses_size == 1:
            return func(self, query, key, value, tensor_layout, attn_mask, **kwargs)

        ulysses_rank = dist.get_rank(ulysses_group)

        assert tensor_layout in ["NHD", "HND"], (
            f"tensor_layout must be NHD or HND, but got {tensor_layout}"
        )
        if tensor_layout == "HND":
            seq_dim = 1
            head_dim = 0
        elif tensor_layout == "NHD":
            seq_dim = 0
            head_dim = 1
        else:
            raise ValueError(f"Invalid tensor layout: {tensor_layout}")

        if query.shape[head_dim] % ulysses_size != 0:
            raise ValueError(
                f"Head dim {head_dim} of query {query.shape} "
                f"must be divisible by ulysses size {ulysses_size}"
            )
        if key.shape[head_dim] % ulysses_size != 0:
            raise ValueError(
                f"Head dim {head_dim} of key {key.shape} "
                f"must be divisible by ulysses size {ulysses_size}"
            )
        if value.shape[head_dim] % ulysses_size != 0:
            raise ValueError(
                f"Head dim {head_dim} of value {value.shape} "
                f"must be divisible by ulysses size {ulysses_size}"
            )

        # Apply ulysses_a2a_in before the function call
        query, key, value, attn_mask = ulysses_a2a_in(
            query,
            key,
            value,
            attn_mask,
            tensor_layout,
            ulysses_size=ulysses_size,
            ulysses_rank=ulysses_rank,
            ulysses_group=ulysses_group,
            fuse_qkv=self.fuse_qkv,
        )

        # truncate and pad if cp is uneven
        truncate_and_pad = uneven_cp_config is not None

        if ring_size == 1 and truncate_and_pad:
            # there is no ring, so we can use uneven_cp_config.seq_len
            # to do truncate and pad
            seq_len = uneven_cp_config.seq_len

            # Truncate key and value tensors using torch.narrow
            key = torch.narrow(key, seq_dim, 0, seq_len).contiguous()
            value = torch.narrow(value, seq_dim, 0, seq_len).contiguous()

        if ring_size == 1 and varlen_cp_config is not None:
            cu_seqlens_q = varlen_cp_config.cu_seqlens_q_cur_ulysses_group
            cu_seqlens_kv = varlen_cp_config.cu_seqlens_kv_cur_ulysses_group
            kwargs["cur_rank_cu_seqlens_q"] = cu_seqlens_q
            kwargs["cur_rank_cu_seqlens_k"] = cu_seqlens_kv
            kwargs["cur_rank_max_seqlen_q"] = (
                varlen_cp_config.max_seq_len_q_cur_ulysses_group
            )
            kwargs["cur_rank_max_seqlen_k"] = (
                varlen_cp_config.max_seq_len_kv_cur_ulysses_group
            )

            if key.shape[seq_dim] != cu_seqlens_kv[-1]:
                # Truncate kv_inputs using torch.narrow
                key = torch.narrow(key, seq_dim, 0, cu_seqlens_kv[-1])
                value = torch.narrow(value, seq_dim, 0, cu_seqlens_kv[-1])

        # Call the original function
        result = func(self, query, key, value, tensor_layout, attn_mask, **kwargs)

        # if ring size is 1, return_lse is false, result only has output.
        if ring_size == 1 and truncate_and_pad and result.shape[seq_dim] > seq_len:
            # Zero out padding using torch.narrow
            padding_part = torch.narrow(
                result, seq_dim, seq_len, result.shape[seq_dim] - seq_len
            )
            padding_part.zero_()

        if (
            ring_size == 1
            and varlen_cp_config is not None
            and result.shape[seq_dim]
            > varlen_cp_config.cu_seqlens_q_cur_ulysses_group[-1]
        ):
            # Zero out padding using torch.narrow
            cu_end = varlen_cp_config.cu_seqlens_q_cur_ulysses_group[-1]
            padding_part = torch.narrow(
                result, seq_dim, cu_end, result.shape[seq_dim] - cu_end
            )
            padding_part.zero_()

        result = ulysses_a2a_out(
            result,
            tensor_layout,
            ulysses_size=ulysses_size,
            ulysses_group=ulysses_group,
        )

        return result

    return wrapper


def get_kv_rank(ring_size, ring_rank, cur_iter):
    # get the the source rank of kv tensor in current iter
    return (ring_size + ring_rank - cur_iter) % ring_size


def ring_wrapper(func):
    def wrapper(self, query, key, value, tensor_layout, attn_mask=None, **kwargs):
        ring_group = self.ring_group
        uneven_cp_config = self.uneven_cp_config
        varlen_cp_config = self.varlen_cp_config

        ring_size = dist.get_world_size(ring_group) if ring_group is not None else 1

        if ring_size == 1:
            return func(self, query, key, value, tensor_layout, attn_mask, **kwargs)

        rank = dist.get_rank(ring_group)
        send_dst = (rank + 1) % ring_size
        recv_src = (rank - 1) % ring_size

        # Determine sequence dimension based on tensor layout
        if tensor_layout == "HND":
            seq_dim = 1  # query, key, value shape are: [H, S, D], so seq_dim is 1
        elif tensor_layout == "NHD":
            seq_dim = 0  # query, key, value shape are: [S, H, D], so seq_dim is 0
        else:
            raise ValueError(f"Invalid tensor layout: {tensor_layout}")

        p2p_comm_buffers = [None, None]
        p2p_comm_buffers[0] = torch.cat((key.unsqueeze(0), value.unsqueeze(0)), dim=0)
        send_recv_reqs = [[], []]

        out = None
        softmax_lse = None
        for i in range(ring_size):
            kv_rank = get_kv_rank(ring_size, rank, i)
            # wait until KV is received
            for req in send_recv_reqs[(i + 1) % 2]:
                req.wait()

            if i < (ring_size - 1):
                p2p_comm_buffers[(i + 1) % 2] = torch.empty_like(
                    p2p_comm_buffers[i % 2]
                )
                send_recv_reqs[i % 2] = ring_attn_p2p_communicate(
                    rank,
                    p2p_comm_buffers[i % 2],
                    send_dst,
                    p2p_comm_buffers[(i + 1) % 2],
                    recv_src,
                    ring_group,
                )
            kv_inputs = p2p_comm_buffers[i % 2]

            # do truncate and pad if cp is uneven,
            if uneven_cp_config is not None:
                # seq_dim+1 because kv_inputs is concated to
                # [2, H, S, D] or [2, S, H, D]
                if (
                    kv_inputs.shape[seq_dim + 1]
                    != uneven_cp_config.seq_len_cur_ring_group[kv_rank]
                ):
                    # Truncate kv_inputs using torch.narrow
                    kv_inputs = torch.narrow(
                        kv_inputs,
                        seq_dim + 1,
                        0,
                        uneven_cp_config.seq_len_cur_ring_group[kv_rank],
                    )

            if varlen_cp_config is not None:
                cu_seqlens_q = varlen_cp_config.cu_seqlens_q_cur_ring_group[rank]
                cu_seqlens_kv = varlen_cp_config.cu_seqlens_kv_cur_ring_group[kv_rank]
                kwargs["cur_rank_cu_seqlens_q"] = cu_seqlens_q
                kwargs["cur_rank_cu_seqlens_k"] = cu_seqlens_kv

                if kv_inputs.shape[seq_dim + 1] != cu_seqlens_kv[-1]:
                    # Truncate kv_inputs using torch.narrow
                    kv_inputs = torch.narrow(
                        kv_inputs, seq_dim + 1, 0, cu_seqlens_kv[-1]
                    )

                kwargs["cur_rank_max_seqlen_q"] = (
                    varlen_cp_config.max_seq_len_q_cur_ring_group
                )
                kwargs["cur_rank_max_seqlen_k"] = (
                    varlen_cp_config.max_seq_len_kv_cur_ring_group
                )

            kwargs["return_lse"] = True
            # we need this line because a bug in flash-attn4
            # https://github.com/Dao-AILab/flash-attention/pull/1793
            with torch.cuda.device(query.device.index):
                block_out = func(
                    self,
                    query,
                    kv_inputs[0],
                    kv_inputs[1],
                    tensor_layout,
                    attn_mask,
                    **kwargs,
                )

            out_per_step = block_out[0]
            softmax_lse_per_step = block_out[1]

            if i == 0:
                softmax_lse = torch.clone(softmax_lse_per_step).to(torch.float)
                out = torch.clone(out_per_step)
            else:
                ring_fwd_out_correction(
                    out, out_per_step, softmax_lse, softmax_lse_per_step
                )
                ring_fwd_softmax_lse_correction(softmax_lse, softmax_lse_per_step)

        # Determine output sequence dimension based on tensor layout
        # (for output tensor)
        if tensor_layout == "HND":
            out_seq_dim = 1  # out is [H, S, D], so seq_dim is 1
        elif tensor_layout == "NHD":
            out_seq_dim = 0  # out is [S, H, D], so seq_dim is 0
        else:
            raise ValueError(f"Invalid tensor layout: {tensor_layout}")

        start_pos = out.shape[out_seq_dim]

        if (
            uneven_cp_config is not None
            and out.shape[out_seq_dim] > uneven_cp_config.seq_len_cur_ring_group[rank]
        ):
            start_pos = uneven_cp_config.seq_len_cur_ring_group[rank]

        if (
            varlen_cp_config is not None
            and out.shape[out_seq_dim]
            > varlen_cp_config.cu_seqlens_q_cur_ring_group[rank][-1]
        ):
            start_pos = varlen_cp_config.cu_seqlens_q_cur_ring_group[rank][-1]

        if start_pos < out.shape[out_seq_dim]:
            padding_length = out.shape[out_seq_dim] - start_pos
            padding_part = torch.narrow(out, out_seq_dim, start_pos, padding_length)
            padding_part.zero_()

        return out

    return wrapper
