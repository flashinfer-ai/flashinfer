import logging
import torch
from torch.distributed.device_mesh import init_device_mesh

logger = logging.getLogger(__name__)


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


def ulysses_varlen_config(seq_lens_q, seq_lens_kv):
    """Compute cumulative sequence lengths for Ulysses-only variable-length parallelism.

    In Ulysses-only mode (``ring_size == 1``), the packed sequences are treated
    as a whole and split across heads via all-to-all. This function builds the
    ``cu_seqlens`` arrays that the attention kernel needs to locate sequence
    boundaries within the packed input.

    Args:
        seq_lens_q: Per-sequence query lengths, e.g. ``[1021, 2048, 512]``.
            Can be a list, tuple, or torch.Tensor.
        seq_lens_kv: Per-sequence key/value lengths, same format as
            ``seq_lens_q``.

    Returns:
        Tuple of four elements:

        - **cu_seqlens_q** (*torch.Tensor*): Cumulative query sequence lengths
          of shape ``[num_seqs + 1]``, starting with 0.
        - **cu_seqlens_kv** (*torch.Tensor*): Cumulative key/value sequence
          lengths, same shape.
        - **max_seqlen_q** (*int*): Maximum query sequence length.
        - **max_seqlen_kv** (*int*): Maximum key/value sequence length.
    """
    rank = torch.distributed.get_rank()
    device = torch.device(f"cuda:{rank}")

    cu_seqlens_q = (
        torch.cat(
            [
                torch.zeros(1, dtype=torch.int32),
                torch.cumsum(torch.tensor(seq_lens_q, dtype=torch.int32), dim=0),
            ]
        )
        .to(device)
        .to(torch.int32)
    )

    cu_seqlens_kv = (
        torch.cat(
            [
                torch.zeros(1, dtype=torch.int32),
                torch.cumsum(torch.tensor(seq_lens_kv, dtype=torch.int32), dim=0),
            ]
        )
        .to(device)
        .to(torch.int32)
    )

    max_seqlen_q = max(seq_lens_q)
    max_seqlen_k = max(seq_lens_kv)

    return cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_k


def ring_varlen_config(seq_lens_q, seq_lens_kv, ring_group):
    """Compute per-rank cumulative sequence lengths for Ring-only variable-length parallelism.

    In Ring-only mode (``ulysses_size == 1``), each individual sequence is
    split along the sequence dimension across ring ranks. Each rank holds a
    chunk of every sequence, so ``cu_seqlens`` are stored as a 2-D tensor of
    shape ``[ring_size, num_seqs + 1]`` — one row per rank — because each
    rank's chunk has different per-sequence lengths (the last rank gets the
    remainder after padding).

    Sequences are padded to be divisible by ``ring_size`` so that the first
    ``ring_size - 1`` ranks each get ``ceil(seq_len / ring_size)`` tokens per
    sequence, and the last rank gets the remainder.

    Args:
        seq_lens_q: Per-sequence query lengths, e.g. ``[1021, 1024, 1027]``.
            Can be a list, tuple, or torch.Tensor.
        seq_lens_kv: Per-sequence key/value lengths, same format as
            ``seq_lens_q``.
        ring_group: The ring attention process group, used to determine
            ``ring_size`` via ``torch.distributed.get_world_size(ring_group)``.

    Returns:
        Tuple of four elements:

        - **cu_seqlens_q_all_ranks** (*torch.Tensor*): Cumulative query
          sequence lengths for all ranks, shape ``[ring_size, num_seqs + 1]``.
        - **cu_seqlens_kv_all_ranks** (*torch.Tensor*): Cumulative key/value
          sequence lengths, same shape.
        - **max_seq_len_q** (*torch.Tensor*): Maximum per-rank padded query
          sequence length.
        - **max_seq_len_kv** (*torch.Tensor*): Maximum per-rank padded
          key/value sequence length.

    Example::

        # seq_lens_q = [1021, 1024, 1027], ring_size = 4
        #
        # Padding: 1021 -> 1024, 1024 -> 1024, 1027 -> 1028
        # Per-rank: 256,          256,          257
        # Last rank gets remainder for each sequence:
        #   1021: 256 - (1024-1021) = 253
        #   1024: 256 - (1024-1024) = 256
        #   1027: 257 - (1028-1027) = 256
        #
        # cu_seqlens_q_all_ranks (shape [4, 4]):
        #   rank 0: [0, 256, 512, 769]
        #   rank 1: [0, 256, 512, 769]
        #   rank 2: [0, 256, 512, 769]
        #   rank 3: [0, 253, 509, 765]  (last rank, shorter chunks)
    """
    if not isinstance(seq_lens_q, torch.Tensor):
        seq_lens_q = torch.tensor(seq_lens_q, dtype=torch.int32)
    if not isinstance(seq_lens_kv, torch.Tensor):
        seq_lens_kv = torch.tensor(seq_lens_kv, dtype=torch.int32)

    world_size = (
        torch.distributed.get_world_size(ring_group) if ring_group is not None else 1
    )

    rank = torch.distributed.get_rank()
    device = torch.device(f"cuda:{rank}")

    padded_seq_lens_q = (seq_lens_q + world_size - 1) // world_size * world_size
    padded_seq_lens_kv = (seq_lens_kv + world_size - 1) // world_size * world_size

    padded_seq_len_q_cur_rank = padded_seq_lens_q // world_size
    padded_seq_len_kv_cur_rank = padded_seq_lens_kv // world_size

    max_seq_len_q = padded_seq_len_q_cur_rank.max()
    max_seq_len_kv = padded_seq_len_kv_cur_rank.max()

    cu_seqlens_q_all_ranks = []
    cu_seqlens_kv_all_ranks = []

    for i in range(world_size):
        if i == world_size - 1:
            seq_len_q_cur_rank = padded_seq_len_q_cur_rank - (
                padded_seq_lens_q - seq_lens_q
            )
            seq_len_kv_cur_rank = padded_seq_len_kv_cur_rank - (
                padded_seq_lens_kv - seq_lens_kv
            )
        else:
            seq_len_q_cur_rank = padded_seq_len_q_cur_rank
            seq_len_kv_cur_rank = padded_seq_len_kv_cur_rank

        cu_seqlens_q = (
            torch.cat(
                [
                    torch.zeros(1),
                    torch.cumsum(seq_len_q_cur_rank, dim=0),
                ]
            )
            .to(device)
            .to(torch.int32)
        )
        cu_seqlens_q_all_ranks.append(cu_seqlens_q)

        cu_seqlens_kv = (
            torch.cat(
                [
                    torch.zeros(1),
                    torch.cumsum(seq_len_kv_cur_rank, dim=0),
                ]
            )
            .to(device)
            .to(torch.int32)
        )
        cu_seqlens_kv_all_ranks.append(cu_seqlens_kv)

    return (
        torch.stack(cu_seqlens_q_all_ranks),
        torch.stack(cu_seqlens_kv_all_ranks),
        max_seq_len_q.item(),
        max_seq_len_kv.item(),
    )


def uneven_cp_config(
    seq_len,
    seq_len_padded,
    seq_len_cur_rank,
    ulysses_group=None,
    ring_group=None,
):
    """Gather per-rank sequence lengths and compute the current ring group sequence length.

    Args:
        seq_len: Actual (unpadded) total sequence length.
        seq_len_padded: Padded total sequence length (divisible by world_size).
        seq_len_cur_rank: Number of real (non-padding) tokens on this rank. for example, if the total sequence
        length is 1023 and the world size is 8, and the rank is 0, then seq_len_cur_rank is 128. If the rank is 7, then seq_len_cur_rank is 127.
        ulysses_group: Ulysses process group.
        ring_group: Ring process group.
    """

    rank = torch.distributed.get_rank()
    device = torch.device(f"cuda:{rank}")

    seq_len_cur_rank = torch.tensor(
        [seq_len_cur_rank], dtype=torch.int32, device=device
    )
    gather_list = [
        torch.empty_like(seq_len_cur_rank)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(gather_list, seq_len_cur_rank)
    seq_len_all_ranks = torch.cat(gather_list, dim=0).cpu()

    ring_size = (
        torch.distributed.get_world_size(ring_group) if ring_group is not None else 1
    )
    ulysses_size = (
        torch.distributed.get_world_size(ulysses_group)
        if ulysses_group is not None
        else 1
    )

    if ring_size == 1:
        return

    if ulysses_size == 1:
        ring_ranks = torch.distributed.get_process_group_ranks(ring_group)
        seq_len_cur_ring_group = seq_len_all_ranks[torch.tensor(ring_ranks)]
        return seq_len_cur_ring_group

    ulysses_ranks = torch.distributed.get_process_group_ranks(ulysses_group)
    seq_len_cur_ulysses_group = seq_len_all_ranks[torch.tensor(ulysses_ranks)]
    ring_seq_cur_ring_rank = torch.sum(seq_len_cur_ulysses_group, dtype=torch.int32).to(
        device
    )
    gather_list = [
        torch.empty(1, dtype=torch.int32, device=device) for _ in range(ring_size)
    ]

    torch.distributed.all_gather(
        gather_list,
        ring_seq_cur_ring_rank,
        group=ring_group,
    )
    seq_len_cur_ring_group = torch.cat(gather_list, dim=0)

    return seq_len_cur_ring_group


def get_parallel_groups(
    ulysses_size: int,
    ring_size: int,
    device_type: str = "cuda",
):
    """Create a device mesh and return the Ring and Ulysses process groups.

    Builds a ``DeviceMesh`` with up to three dimensions — ``redundant``
    (when ``world_size > ulysses_size * ring_size``), ``ring``, and
    ``ulysses`` — and extracts the corresponding process groups for use
    in :class:`ParallelAttention`.

    Args:
        ulysses_size: Ulysses parallel degree (number of ranks that
            participate in all-to-all head splitting).
        ring_size: Ring attention parallel degree (number of ranks that
            exchange KV chunks in a ring).
        device_type: Device type for the mesh, defaults to ``"cuda"``.

    Returns:
        Tuple of two elements:

        - **ring_group** (*Optional[ProcessGroup]*): The ring attention
          process group, or ``None`` if ``ring_size == 1``.
        - **ulysses_group** (*Optional[ProcessGroup]*): The Ulysses
          process group, or ``None`` if ``ulysses_size == 1``.

    Raises:
        ValueError: If ``world_size`` is not divisible by
            ``ulysses_size * ring_size``.

    Note:
        The device mesh dimensions are created in the order:
        ``redundant`` (if needed) → ``ring`` → ``ulysses``.
    """

    total_parallel_size = ulysses_size * ring_size
    world_size = torch.distributed.get_world_size()
    if world_size % total_parallel_size != 0:
        raise ValueError(
            f"World size ({world_size}) is not divisible by "
            f"total parallel size ({total_parallel_size})"
        )

    logger.debug(
        f"Setting up device mesh with total parallel size: {total_parallel_size} "
        f"ulysses_size: {ulysses_size}, ring_size: {ring_size}"
    )

    if total_parallel_size == 1:
        logger.debug("No parallelism needed, skipping device mesh setup")
        return None, None

    mesh_dims = []
    mesh_sizes = []

    if world_size != total_parallel_size:
        mesh_dims.append("redundant")
        mesh_sizes.append(world_size // total_parallel_size)
        logger.debug(
            f"Added redundant dimension: "
            f"{world_size // total_parallel_size}, "
            f"world_size={world_size}, "
            f"total_parallel_size={total_parallel_size}"
        )

    if ring_size > 1:
        mesh_dims.append("ring")
        mesh_sizes.append(ring_size)
        logger.debug(f"Added Ring dimension: {ring_size}")
    if ulysses_size > 1:
        mesh_dims.append("ulysses")
        mesh_sizes.append(ulysses_size)
        logger.debug(f"Added Ulysses dimension: {ulysses_size}")

    if not mesh_dims:
        logger.debug("No mesh dimensions needed")
        return None, None
    else:
        logger.info(f"Creating device mesh: dims={mesh_dims}, sizes={mesh_sizes}")
        device_mesh = init_device_mesh(
            device_type,
            tuple(mesh_sizes),
            mesh_dim_names=tuple(mesh_dims),
        )
        logger.info("Device mesh created successfully")

    ring_group = None
    ulysses_group = None
    if ring_size > 1:
        ring_group = device_mesh.get_group("ring")
    if ulysses_size > 1:
        ulysses_group = device_mesh.get_group("ulysses")

    return ring_group, ulysses_group
