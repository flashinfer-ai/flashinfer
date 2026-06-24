import torch
import cutlass
import cutlass.cute as cute


CP_CHUNK_LEN_GRANULARITY = 512
CP_HBM_PARALLELISM_THRESHOLD_NUMERATOR = 1
CP_HBM_PARALLELISM_THRESHOLD_DENOMINATOR = 2
CP_GDDR_PARALLELISM_THRESHOLD_NUMERATOR = 1
CP_GDDR_PARALLELISM_THRESHOLD_DENOMINATOR = 3


def chunk_bound_host(num_items: int, total: int, chunk_size: int) -> int:
    if chunk_size <= 0:
        raise RuntimeError(f"chunk_size must be positive, got {chunk_size}")
    m = min(num_items, total)
    return m + (total - m) // chunk_size


def workspace_num_chunks_host(
    cu_seqlens: torch.Tensor, chunk_size: int, total_seqlen: int
) -> int:
    if cu_seqlens.ndim != 1:
        raise RuntimeError(f"cu_seqlens must be 1D, got {tuple(cu_seqlens.shape)}")
    num_seqs = cu_seqlens.numel() - 1
    return chunk_bound_host(num_seqs, total_seqlen, chunk_size)


def max_num_chunks_host(max_seqlen: int, chunk_size: int) -> int:
    return (max_seqlen + chunk_size - 1) // chunk_size


def is_gddr_device_host(device_name: str) -> bool:
    """Best-effort device-class check for host-side CP dispatch.

    Unknown datacenter names default to the HBM threshold. Consumer/workstation
    names default to the GDDR threshold.
    """
    lowered = device_name.lower()
    gddr_markers = ("geforce", "rtx", "workstation")
    return any(marker in lowered for marker in gddr_markers)


def cp_parallelism_threshold_host(device_name: str) -> tuple[int, int]:
    if is_gddr_device_host(device_name):
        return (
            CP_GDDR_PARALLELISM_THRESHOLD_NUMERATOR,
            CP_GDDR_PARALLELISM_THRESHOLD_DENOMINATOR,
        )
    return (
        CP_HBM_PARALLELISM_THRESHOLD_NUMERATOR,
        CP_HBM_PARALLELISM_THRESHOLD_DENOMINATOR,
    )


def should_use_cp_host(num_parallel_work: int, num_sms: int, device_name: str) -> bool:
    """Return whether a public wrapper should dispatch to the CP path.

    `num_parallel_work` is the non-CP kernel parallelism, typically batch times
    output/state heads. CP is selected only when that parallelism is strictly
    below the card-specific threshold.
    """
    threshold_num, threshold_den = cp_parallelism_threshold_host(device_name)
    return num_parallel_work * threshold_den < num_sms * threshold_num


def choose_cp_chunk_len_host(
    max_seqlen: int,
    num_heads: int,
    num_sms: int,
    chunk_len_granularity: int = CP_CHUNK_LEN_GRANULARITY,
) -> int:
    """Choose a CP chunk length for the CP workspace kernels.

    The TTFT path is usually one long sequence. For that case, MN precompute
    launches `ceil_div(max_seqlen, chunk_len) * num_heads` CTAs. Pick the
    smallest granularity-aligned chunk length whose CTA count is at most one wave.
    """
    if max_seqlen <= 0:
        raise RuntimeError(f"max_seqlen must be positive, got {max_seqlen}")
    if num_heads <= 0:
        raise RuntimeError(f"num_heads must be positive, got {num_heads}")
    if num_sms <= 0:
        raise RuntimeError(f"num_sms must be positive, got {num_sms}")
    if chunk_len_granularity <= 0:
        raise RuntimeError(
            f"chunk_len_granularity must be positive, got {chunk_len_granularity}"
        )
    if chunk_len_granularity % 64 != 0:
        raise RuntimeError(
            f"chunk_len_granularity must be a multiple of 64, got {chunk_len_granularity}"
        )

    target_chunks = max(1, num_sms // num_heads)
    min_chunk_len = (max_seqlen + target_chunks - 1) // target_chunks
    return (
        (min_chunk_len + chunk_len_granularity - 1) // chunk_len_granularity
    ) * chunk_len_granularity


@cute.jit
def chunk_bound(
    seq_idx: cutlass.Int32, total: cutlass.Int32, chunk_size: cutlass.Int32
) -> cutlass.Int32:
    m = seq_idx
    if total < m:
        m = total
    return m + (total - m) // chunk_size


@cute.jit
def chunks_for_len(seq_len: cutlass.Int32, chunk_size: cutlass.Int32) -> cutlass.Int32:
    return (seq_len + chunk_size - cutlass.Int32(1)) // chunk_size


@cute.jit
def logical_chunk_to_work_desc(
    cu_seqlens: cute.Tensor,
    logical_chunk_idx: cutlass.Int32,
    chunk_size: cutlass.Int32,
    num_seqs: cutlass.Int32,
):
    seq_idx = cutlass.Int32(0)
    chunk_idx_in_seq = logical_chunk_idx
    running = cutlass.Int32(0)
    for candidate_seq in cutlass.range(num_seqs, unroll=1):
        seq_start = cutlass.Int32(cu_seqlens[candidate_seq])
        seq_end = cutlass.Int32(cu_seqlens[candidate_seq + cutlass.Int32(1)])
        seq_chunks = chunks_for_len(seq_end - seq_start, chunk_size)
        next_running = running + seq_chunks
        if logical_chunk_idx >= running and logical_chunk_idx < next_running:
            seq_idx = candidate_seq
            chunk_idx_in_seq = logical_chunk_idx - running
        running = next_running
    return seq_idx, chunk_idx_in_seq


@cute.jit
def varlen_chunk_idx(
    seq_idx: cutlass.Int32,
    tok_idx_start: cutlass.Int32,
    chunk_idx_in_seq: cutlass.Int32,
    chunk_size: cutlass.Int32,
) -> cutlass.Int32:
    return chunk_bound(seq_idx, tok_idx_start, chunk_size) + chunk_idx_in_seq


@cute.jit
def varlen_chunk_valid_len(
    seq_len: cutlass.Int32,
    chunk_idx_in_seq: cutlass.Int32,
    chunk_size: cutlass.Int32,
) -> cutlass.Int32:
    remaining = seq_len - chunk_idx_in_seq * chunk_size
    if remaining > chunk_size:
        remaining = chunk_size
    return remaining
